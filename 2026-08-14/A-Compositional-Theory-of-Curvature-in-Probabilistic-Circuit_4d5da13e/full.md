# A Compositional Theory of Curvature in Probabilistic Circuits

Hrithik Suresh\*<sup>1</sup>, Sahil Sidheekh\*<sup>2,3†</sup>, Shelar Parth Vijay <sup>1</sup>, Yasir Z <sup>1</sup>,

Sriraam Natarajan<sup>3</sup>, Narayanan Chatapuram Krishnan<sup>1</sup>

<sup>1</sup> Mehta Family School of Data Science and Artificial Intelligence, Department of Data Science, Indian Institute of Technology Palakkad, Kerala, India

<sup>2</sup> AT&T CDO

<sup>3</sup> Erik Jonsson School of Engineering & Computer Science, The University of Texas at Dallas, Richardson, TX, USA

## Abstract

Probabilistic Circuits (PCs) are generative models that support exact inference and, unlike deep neural networks, admit an exact and tractable measure of loss-surface curvature: the trace of the Hessian of the log-likelihood. Recent work regularizes this trace globally to bias learning toward flatter, better generalizing optima. We show that treating sharpness as a global regularizer can be misspecified for PCs, whose curvature is inherently compositional. We prove that each sum node’s contribution to the Hessian trace factorizes exactly into its circuit flow, which measures how heavily the node is used, and a local sharpness term determined by its output distribution. This decomposition provides insights into why global sharpness regularization is depth biased and can lead to underfitting. Building on it, we introduce an adaptive sharpness aware regularizer that penalizes nodes based on intrinsic local curvature and preserves closed form EM updates. We also show that empirically, this targeted regularization recovers the generalization that global regularization sacrifices while retaining the robustness and benefits of sharpness aware learning.

## Introduction

Probabilistic circuits (PCs) are generative models whose structural constraints make a broad class of probabilistic queries exactly tractable and computable in time linear in the circuit size (Choi, Vergari, and den Broeck 2020). This distinguishes PCs from mainstream deep generative models such as GANs (Goodfellow et al. 2014), VAEs (Kingma and Welling 2014), and normalizing flows (Papamakarios et al. 2021), and makes them particularly useful in settings where exact and reliable inference is required, including constrained generation (Zhang et al. 2023), image inpainting (Liu, Niepert, and den Broeck 2024), robust representation learning (Braun et al. 2025), multimodal fusion (Sidheekh et al. 2025), neurosymbolic AI (Ahmed et al. 2022; Karanam et al. 2025) and algorithmic recourse (Nemeˇ cek, Pevnˇ y, and Marecek 2025;\` Sabu, S, and Krishnan 2026), among others. As PCs become deeper and more expressive, however, they are increasingly prone to overfitting in low-data regimes and may converge to sharp optima that generalize poorly (Suresh et al. 2026).

Sharpness-aware learning addresses this problem by biasing optimization toward flatter regions of the loss landscape (Foret et al. 2021). In deep neural networks, the relevant Hessian-based curvature is generally intractable and must be estimated through stochastic approximations (Kaur, Cohen, and Lipton 2023; Sankar et al. 2021). Recent work (Suresh et al. 2026) has shown that PCs are a rare exception in this context: their structure makes the Hessian trace of the negative log-likelihood exactly computable in a single forward–backward pass, enabling a tractable global trace regularizer that improves generalization when data are scarce. While this result established that sharpness can be computed and penalized efficiently in PCs, it treats the trace as a single global measure offlatness.

In this work, we ask what this global curvature actually represents inside a PC, and whether it is always an appropriate learning signal. Our motivation comes from three empirical observations. First, in the high-data regime, global trace regularization reaches flatter optima but lowers both training and test log-likelihood (as illustrated in Figure 1), indicating under-fitting rather than improved generalization. Second, the node-wise contributions to the trace are highly concentrated, with a small fraction of sum nodes accounting for most of the total curvature. Third, restricting regularization to the largest-contributing nodes degrades performance further. These observations suggest that the issue is not only the overall strength of regularization, but how it is allocated across the circuit. Our central thesis is that these quantities separate exactly, and that separating them helps resolve the puzzle. Specifically, we show that a sum node’s contribution to the Hessian trace factorizes exactly into two semantically distinct terms:

$$
T _ { n } ( \mathbf { x } ) = \underbrace { F _ { n } ( \mathbf { x } ) ^ { 2 } } _ { \mathrm { c o n t e x t u a l ~ u s a g e } } \cdot \underbrace { t _ { n } ( \mathbf { x } ) } _ { \mathrm { l o c a l ~ c u r v a t u r e } } ,
$$

where $F _ { n }$ is the circuit flow through node n, measuring how strongly the node is used in explaining the input, while $t _ { n } ( \mathbf { x } )$ is a local curvature measure that depends only on the node n. Thus, sharpness in a PC is local in origin and contextual in its global effect. A node may contribute strongly to the Hessian trace because it is locally sharp, because it lies on a high-flow path, or because both effects coincide. Consequently, ranking nodes by their global contribution need not identify the nodes with the largest intrinsic local curvature.

We first characterize and study both factors theoretically. For the local term, we show that the Hessian of a sum node is rank one and that its unique nonzero eigenvalue is exactly $t _ { n } .$ . Consequently, $t _ { n }$ fully captures the node’s ambient second-order geometry. For the contextual term, we build on the standard circuit-flow recursion to analyze how structural usage shapes global curvature, and show that it causes a depth bias in the global trace toward shallow, high-flow nodes. We formalize this and derive an exact condition under which rankings by global contribution $T _ { n }$ and local curvature $t _ { n }$ disagree, explaining why selecting nodes by their global trace contribution can preferentially target highly used circuit components rather than intrinsically sharp mixtures. Finally, we show how this decomposition can be algorithmically useful. Since $T _ { n }$ conflates local curvature with structural usage, it is not a reliable quantity for deciding where regularization should act. We propose an adaptive sharpness regularizer use the empirical local curvature $\widehat { t } _ { n }$ to gate the existing trace penalty, assigning stronger regularization to intrinsically sharper mixture nodes while preserving the tractable updates and linear-time complexity of the global method. The resulting learner retains the low-data gains of sharpnessaware training while reducing the under-fitting induced by uniform regularization at higher data scales.

![](images/f23197fc9af0c611ab6969161b367767120fbea49893f4da3fc399cb55132cb2.jpg)  
Figure 1: Global Flatness Can Underfit: Local Curvature Determines Where to Regularize. Training trajectories of a PC on a 2D data distribution, from a shared initialization (star), shown over the (a) train NLL, (b) test NLL, and (c) sharpness surface. The unregularized model reaches a sharp optimum that generalizes poorly. The global trace regularizer’s uniform penalty helps achieve flattest region, but at the expense of under-fitting (poor train and test NLL), whereas our adaptive penalty targets nodes of high local curvature $t _ { n }$ and generalizes best, while also reducing sharpness. Both approaches use the same regularization strength $\mu .$

## Background and Preliminaries

## Probabilistic Circuits

A probabilistic circuit (PC) over random variables ${ \textbf { X } } =$ $\{ \hat { X _ { 1 } } , \dotsc , X _ { d } \}$ is a rooted directed acyclic graph in which each node n computes a distribution $p _ { n }$ , defined recursively:

$$
p _ { n } ( \mathbf x ) = \left\{ \prod _ { c \in \mathrm { c h } ( n ) } p _ { c } ( \mathbf x ) , \quad \begin{array} { l l } { n \mathrm { ~ a n i n p u t ~ n o d e , } } \\ { { \prod _ { c \in \mathrm { c h } ( n ) } p _ { c } ( \mathbf x ) } , \quad } & { n \mathrm { ~ a ~ p r o d u c t ~ n o d e , } } \\ { \sum _ { c \in \mathrm { c h } ( n ) } \theta _ { n c } p _ { c } ( \mathbf x ) , } & { n \mathrm { ~ a ~ s u m ~ n o d e , } } \end{array} \right.\tag{1}
$$

where $f _ { n }$ is a univariate input distribution, ch(n) are the children of n, and the sum-node weights satisfy $\dot { \theta } _ { n c } \geq 0$ and $\begin{array} { r } { \sum _ { c \in \mathrm { c h } ( n ) } \theta _ { n c } = 1 } \end{array}$ , so that the likelihood of a sample is simply the value computed at the root, $p ( \mathbf { x } ) = p _ { r } ( \mathbf { x } )$ . Product nodes represent factorizations over disjoint scopes (decomposability), while sum nodes represent convex mixtures over children with identical scope (smoothness). Together, these two structural conditions are what make a broad class of probabilistic queries tractable, computable exactly and in time linear in circuit size (Choi, Vergari, and den Broeck 2020). This formalism is general enough to subsume arithmetic circuits (Darwiche 2003), sum-product networks (Poon and Domingos 2011), PSDDs (Kisa et al. 2014), and cutset networks (Rahman, Kothalkar, and Gogate 2014) as special cases. Throughout, we write $s$ for the set of sum nodes and collect their weights into $\pmb { \theta } = \{ \theta _ { n c } \}$ . Unless stated otherwise, we treat these sum-edge weights as untied free parameters, and measure curvature w.r.t this representation.

Circuit Flows and EM Learning . While the upward evaluation of a PC computes the likelihood at the root, analyzing learning and curvature requires attributing this computation to individual nodes and edges. Circuit flows (Liu and den Broeck 2021) provide this downward attribution by measuring how much probability mass the root node routes through a given node when explaining a particular sample, and it is this quantity that will allow us later to localize curvature to specific parts of the circuit. Flow propagates from the root downward, and at each node its value is determined by the type of the node’s parents. Setting $F _ { r } ( \mathbf { x } ) = 1$ at the root, the flow at node n is

$$
F _ { n } ( \mathbf { x } ) = \sum _ { \tiny { m \in \mathrm { p a } ( n ) } \atop { m \mathrm { i s p r o d u c t } } } F _ { m } ( \mathbf { x } ) + \sum _ { \tiny { m \in \mathrm { p a } ( n ) } \atop { m \mathrm { i s s u m } } } F _ { m } ( \mathbf { x } ) \theta _ { m n } \frac { p _ { n } ( \mathbf { x } ) } { p _ { m } ( \mathbf { x } ) } ,
$$

meaning a product parent passes its flow to each child undivided, whereas a sum parent attenuates the flow it passes to child n by the routing responsibility $\theta _ { m n } p _ { n } ( \mathbf { x } ) / p _ { m } ( \mathbf { x } )$ Equivalently, $F _ { n } ( \mathbf { x } ) = \bar { \partial } \log \bar { p } _ { r } ( \mathbf { x } ) / \partial \log p _ { n } ( \mathbf { \bar { x } } ) \colon$ the flow is the sensitivity of the root log-likelihood to the node’s logoutput. Flow at an edge $( n , c )$ leaving a sum node n is

$$
F _ { n c } ( \mathbf { x } ) = { \theta } _ { n c } { \frac { p _ { c } ( \mathbf { x } ) } { p _ { n } ( \mathbf { x } ) } } F _ { n } ( \mathbf { x } ) ,\tag{2}
$$

and from this the gradient of the log-likelihood with respect to the corresponding sum weight follows directly, $\bar { \partial } \log p _ { r } ( \mathbf { x } ) / \partial \theta _ { n c } = \bar { F } _ { n c } ( \bar { \mathbf { x } } ) / \theta _ { n c }$ . We also write $r _ { n c } ( \mathbf { x } ) =$ $\theta _ { n c } p _ { c } ( \mathbf { x } ) / p _ { n } ( \mathbf { x } )$ for the posterior routing responsibility of child $c ,$ which satisfies $\begin{array} { r } { \sum _ { c \in \mathrm { c h } ( n ) } r _ { n c } ( \mathbf { x } ) \bar { = } 1 } \end{array}$ and so behaves as a proper distribution over n’s children. Node and edge flows are obtained together from a single forward–backward pass over the circuit, and this same pass suffices to compute gradients and closed-form expectation-maximization (EM) updates in linear time (Liu and den Broeck 2021). In the EM view, the edge flow $F _ { n c } ( \mathbf { x } )$ is the expected number of times sample x traverses edge $( n , c )$ , and the M-step simply aggregates these expectations across a dataset.

## Sharpness and Regularization in PCs

Recent works have pushed PCs toward deeper, more expressive, and more scalable architectures while preserving tractable inference (Peharz et al. 2019, 2020; Sidheekh, Kersting, and Natarajan 2023; Sidheekh and Natarajan 2024; Liu,

Ahmed, and den Broeck 2024; Sidheekh and Natarajan 2026). This increased capacity, however, also makes them more susceptible to overfitting, especially in limited-data regimes. Existing approaches have attempted to exploit the circuit’s tractable structure to address this issue via dropout (Ventola et al. 2023), parameter smoothing, data softening, and entropy regularization (Liu and den Broeck 2021). A complementary view studies overfitting through the geometry of the loss landscape. Flat minima have long been associated with better generalization (Hochreiter and Schmidhuber 1997; Keskar et al. 2017), motivating methods such as sharpness-aware minimization (Foret et al. 2021). This relationship is however not universal: sharpness can depend on the chosen parameterization, and its connection to generalization varies across models and tasks (Dinh et al. 2017; Andriushchenko et al. 2023). More recently, (Suresh et al. 2026) brought this perspective to PCs, showing that overfitting can coincide with convergence to sharp optima and that the Hessian trace of the negative log-likelihood can be computed exactly for general PCs. They showed that as $p _ { r }$ is multilinear in each sum-edge weight, the diagonal of the Hessian of the log-likelihood coincides with the squared first derivative along that weight, so the trace of the Hessian reduces to a sum of squared edge gradients. Writing the per-example negative log-likelihood (NLL) as $\ell ( \pmb \theta ; \mathbf x ) \bar { = } - \bar { \log } p _ { r } ( \mathbf x )$ , the trace

$$
\mathrm { T r } \big ( \nabla _ { \pmb \theta } ^ { 2 } \ell ( \pmb \theta ; \mathbf x ) \big ) = \sum _ { n \in \pmb S } \sum _ { c \in \mathrm { c h } ( n ) } \bigg ( \frac { F _ { n c } ( \mathbf x ) } { \theta _ { n c } } \bigg ) ^ { 2 } \ \geq \ 0 .\tag{3}
$$

quantifies the sharpness of the loss landscape at $\theta ,$ , i.e. curvature information, and is computable exactly and in time linear in circuit size. Overfitting in PCs is typically correlated with convergence to sharp optima, and incorporating this trace as a constraint in the EM M-step yields a closed-form, trace-regularized update for each sum weight,

$$
\theta _ { n c } = \frac { N _ { n c } + \sqrt { N _ { n c } ^ { 2 } + 4 \lambda \mu N _ { n c } } } { 2 \lambda } ,\tag{4}
$$

where $\begin{array} { r } { N _ { n c } = \sum _ { i } F _ { n c } ( \mathbf { x } _ { i } ) } \end{array}$ is the batch-aggregated expected count and $\lambda , \mu ~ \geq ~ 0$ are the Lagrange multipliers enforcing, respectively, the simplex constraint and the regularization strength. This enables convergence to flatter optima that can generalize better. However, in this framework a single strength $\mu$ is applied uniformly to every sum node, we refer to this method as the global trace regularizer, and we take it as our baseline and point of departure. To reason about where in the circuit this trace originates, rather than only its aggregate value, we attribute it to individual sum nodes. The sample-wise global trace contribution of a sum node n collects the terms of Eq. (3) belonging to its outgoing edges,

$$
T _ { n } ( \mathbf { x } ) = \sum _ { c \in \mathrm { c h } ( n ) } \bigg ( \frac { F _ { n c } ( \mathbf { x } ) } { \theta _ { n c } } \bigg ) ^ { 2 } , \widehat { T } _ { n } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } T _ { n } ( \mathbf { x } _ { i } )\tag{5}
$$

so that $\textstyle \sum _ { n \in S } T _ { n } ( \mathbf { x } )$ recovers the trace in Eq. (3) exactly, and $\widehat { T } _ { n }$ is its empirical average at node n. The collection $\{ \widehat { T } _ { n } \}$ thus gives a node-level view of where the model’s sharpness actually resides, we analyze this in the next section.

Table 1: Mean percentage change in test NLL (∆NLL), reduction in degree of overfitting (∆DoF), and decrease in the loss-surface sharpness (∆Sharp) across 20 DEBD binary datasets, achieved by global-trace regularized HCLT, compared to non-regularized baseline, averaged over 5 runs.
<table><tr><td>Data %</td><td>∆Test-NLL</td><td>∆DoF</td><td>∆Sharp</td></tr><tr><td>1%</td><td>7.12</td><td>8.81</td><td>23.04</td></tr><tr><td>5%</td><td>2.19</td><td>5.79</td><td>15.32</td></tr><tr><td>10%</td><td>0.27</td><td>2.73</td><td>10.42</td></tr><tr><td>50%</td><td>-0.48</td><td>0.21</td><td>5.55</td></tr><tr><td>100%</td><td>-0.31</td><td>0.06</td><td>4.04</td></tr></table>

## When Global Sharpness Fails

The global trace regularizer was motivated by the observation that reducing sharpness improves generalization by enabling convergence to flatter optima (Suresh et al. 2026). However, as we show in this section, treating sharpness as a single global measure obscures how curvature is distributed across the circuit and which components give rise to it. We identify three empirical observations that motivate a more structured view of sharpness and its regularization.

) Train LL (solid) and Test LL (dashed) per Epoch ( =0.4)  
![](images/67ef0ff940465116f3ccd1baaddf433dff69cf8192c3cbb99fe5da4a19176b1f.jpg)  
Figure 2: Training and test log-likelihood trajectories for the vanilla and global trace-regularized models on accidents, baudio, and ad using the full training data. Global trace regularization reduces the train-test gap but lowers both training and test likelihood, indicating underfitting rather than improved generalization.

Observation 1: A flatter optimum need not be a better one. Table 1 shows that the global trace regularizer helps reach flatter optima but can nevertheless attain lower test log likelihood than the unregularized model when sufficient data are available. This degradation is not explained by a widening train-test gap, as the corresponding train log-likelihood also decreases, as shown in Figure 2. It therefore drives learning toward solutions that are flatter yet fit both the training and test distributions less well, indicating underfitting. Reducing overall curvature is thus not sufficient, understanding how that curvature is important to preserve model capacity.

![](images/9288cee343253987fc186c57b799ababb7e1d3ff8a7a0b51a77f21149fc2331b.jpg)  
Figure 3: Only a small fraction of sum nodes accounts for most of the global Hessian trace. Cumulative share of the total trace as sum nodes are ranked by their empirical contribution $( { \widehat { T } } _ { n } )$ . Across datasets, the distribution is highly concentrated: a small set of nodes dominates the global sharpness signal, while most nodes contribute negligibly.

Observation 2: Global sharpness is concentrated in a small set of nodes. Examining how the global Hessian trace is distributed across the circuit reveals that the nodewise contributions $( \widehat { T } _ { n } )$ defined in Eq (5) are highly nonuniform. As shown in Figure 3, a small minority $( < 1 0 \% )$ of sum nodes accounts for nearly the entire trace (99.99%), while the contributions of other nodes are negligible. Global sharpness is therefore not spread evenly across the model, but concentrated in a small set of circuit components. This also reveals a limitation of using a single global regularization strength. The regularizer penalizes all sum nodes equally despite their markedly different contributions to the total curvature, and cannot distinguish the few nodes that dominate the sharpness signal from the many nodes that are already effectively flat.

Observation 3: Targeting the largest contributors makes underfitting worse. The concentration observed above suggests a natural solution: restrict regularization to the nodes with the largest contributions to the global trace. If underfitting were caused primarily by applying the penalty to many negligible contributors that are already sufficiently flat, this strategy could preserve capacity while retaining curvature control. However, as shown in Figure 4, the test log-likelihood degrades further as the penalty is restricted to progressively fewer top- $( \widehat { T } _ { n } )$ nodes. Thus, although $( \widehat { T } _ { n } )$ measures a node’s contribution to global sharpness, it does not identify where regularization should act. Taken together, these observations suggest that neither greater flattening nor restricting regularization to fewer high-contribution nodes is sufficient to resolve the observed degradation. They instead raise a more fundamental question: what does a node’s global trace contribution actually measure, and why can it be a poor signal for allocating regularization?

![](images/af6f480155ccff68c068df5fcfc708fda2832ded83116cd9728ab32a3efb7793.jpg)  
Figure 4: Targeting the largest global trace contributors worsens underfitting. Change in test log-likelihood relative to the unregularized model when trace regularization is applied to all sum nodes or restricted to progressively smaller fractions ranked by ${ \widehat { T } } _ { n } .$ . Selecting fewer top-ranked nodes generally worsens performance, showing that global contribution is a poor criterion for allocating regularization.

## A Compositional Theory of Sharpness

To address the above puzzle, we first show theoretically that a node’s global trace contribution factorizes exactly into a structural factor and a local curvature factor. We characterize them geometrically, and derive the conditions under which the two induce different node rankings. Together, these results formalize the thesis that curvature in a PC is generated locally and expressed globally through circuit flow.

## Exact Global-Local Decomposition

Theorem 1. Consider any sum node n with strictly positive outgoing weights in a smooth, decomposable PC. For any input x, its global trace contribution $\mathbf { \bar { \rho } } _ { T _ { n } ( \mathbf { x } ) }$ factorizes as $\begin{array} { r } { \hat { T _ { n } } ( \mathbf x ) = F _ { n } \mathbf { \bar { ( } x ) } ^ { 2 } t _ { n } ( \mathbf x ) , t _ { n } ( \mathbf x ) = \sum _ { c \in \mathrm { c h } ( n ) } ( \dot { p } _ { c } ( \mathbf x ) / p _ { n } ( \mathbf x ) ) ^ { 2 } } \end{array}$ Consequently, $\begin{array} { r } { \operatorname { T r } \left( \nabla _ { \pmb { \theta } } ^ { 2 } \ell ( \pmb { \theta } ; \mathbf { x } ) \right) = \sum _ { n \in S } F _ { n } ( \mathbf { x } ) ^ { 2 } t _ { n } ( \mathbf { x } ) } \end{array}$

The decomposition is exact and follows directly from the edge-flow factorization. Moreover, both factors are available from the same upward-downward computation used to evaluate circuit flows, so computing the decomposition adds no asymptotic cost. The two factors, however, have distinct interpretations. The term $F _ { n } ( \mathbf { x } ) ^ { 2 }$ measures the node’s contextual usage: how strongly the circuit output depends on node n for input x. The term $t _ { n } ( \mathbf { x } )$ , as we will show, is the node’s local trace, determined entirely by the outputs of its local mixture. Hence, a large global contribution $\bar { T } _ { n }$ may arise from large local curvature, large contextual usage, or both. A node that is frequently used can dominate the global trace even when its local mixture is not among the sharpest, while a locally sharp node can contribute little globally if little flow reaches it. Under uniform penalization, this can suppress high-usage components regardless of their local geometry, leading regularization to act where curvature is globally amplified rather than where it is intrinsically largest.

Figure 5 makes the distinction concrete. The nodes that dominate the global trace in panel (c) align more closely with high contextual usage in panel (a) than with the largest local traces in panel (b). This mismatch motivates a closer examination of the two factors: we first characterize the local geometry summarized by $t _ { n } .$ , and then study how $F _ { n }$ transports that geometry through the circuit.

![](images/a2ee96eab39b5da2dd6bfb4c0ce85e8de915c45e36b12a1807f92713ca94f30d.jpg)  
(a) Contextual Usage (F<sub>n</sub>)

![](images/a271a9e11e89eabf874ddd747d008c9bf877b1de13a2a50c93511dbe624e7d93.jpg)  
(b) Intrinsic local curvature (t<sub>n</sub>)

![](images/d8f5130db777ce3ea2ea91901c017e36a1e28ef50de157276d8f90710b048728.jpg)  
(c) Global trace contribution (T<sub>n</sub>)

![](images/f8f0bc357a60390ea4c5f7ff9e4859ab2bab99871889db2647b81b38dd166cd7.jpg)  
(d) Flow-curvature interaction  
Figure 5: Compositional structure of sharpness in a trained probabilistic circuit. For each input, a sum node’s global trace contribution factorizes as $\begin{array} { r } { T _ { n } = F _ { n } ^ { 2 } t _ { n } . } \end{array}$ , separating local curvature from its amplification through circuit flow. Panels (a)–(c) show the same circuit nodes colored by empirical contextual usage, local curvature, and global contribution, while panel (d) visualizes their interaction, showing that nodes with the large global curvature contribution need not have the large local curvature.

## Rank-One Local Geometry

The local factor $t _ { n }$ is more than a convenient scalar summary, and as we show below, it captures the entire local second-order geometry of a sum node. Consider a sum node n in isolation, with local negative log-output $\ell _ { n } ( \pmb \theta _ { n } ; \mathbf x ) =$ $\begin{array} { r } { - \log \left( \sum _ { c } \theta _ { n c } p _ { c } ( \mathbf { x } ) \right) } \end{array}$ , and write $\rho _ { n c } ( \mathbf { x } ) = p _ { c } ( \mathbf { x } ) / p _ { n } ( \mathbf { x } )$ for the output ratios, collected into the vector $\rho _ { n }$

Proposition 1 (Rank-one local Hessian). The gradient and Hessian of $\ell _ { n }$ in the edge weights $\pmb \theta _ { n }$ are $\nabla _ { \pmb { \theta } _ { n } } \ell _ { n } = - \pmb { \rho } _ { n }$ and $H _ { n } ( { \bf x } ) ~ = ~ \nabla _ { \pmb { \theta } _ { n } } ^ { 2 } \ell _ { n } ~ = ~ \rho _ { n } \mathbf { \bar { \rho } } _ { n } ^ { \top }$ . Hence, $H _ { n }$ is positive semidefinite and, whenever $\rho _ { n } \neq 0$ , has rank one, with unique nonzero eigenvalue $\lambda _ { \operatorname* { m a x } } ( H _ { n } ) = \| \pmb { \rho _ { n } } \| _ { 2 } ^ { 2 } = t _ { n }$ . Consequently $t _ { n } = \mathrm { T r } ( H _ { n } ) = \| H _ { n } \| _ { 2 } = \| H _ { n } \| _ { F }$

Thus, the local second-order geometry of a sum node collapses to a single scalar, the quantity $t _ { n }$ , which is simultaneously the local hessian trace, the maximum ambient curvature, and the total Hessian magnitude.

## Context Propagation and the Depth Bias

The contextual factor $F _ { n } ^ { 2 }$ is what makes $T _ { n }$ and $t _ { n }$ differ, and it induces a systematic depth bias in the global trace: because flow is partitioned among children at every sum node on the way down from the root, shallow nodes accumulate more of it than deep ones, so the global contribution $T _ { n } =$ $F _ { n } ^ { 2 } t _ { n }$ favors structurally shallow nodes regardless of their intrinsic curvature. To understand this bias better, recall a standard consequence of the circuit-flow recursion: product nodes transmit flow unchanged, whereas sum nodes scale it by posterior routing responsibilities. Unrolling the flow recursion thus shows that flow is in fact attenuated by the specific edges it traverses, rather than depth as such.

Lemma 1. For a tree-structured PC, the node flow is the product of routing responsibilities along the unique root-to-n path $\pi ( r , n )$ , taken over its sum edges $( E _ { s u m } )$ only: $F _ { n } ( \mathbf { x } ) \ =$ $\begin{array} { r } { \prod _ { e \in \pi ( r , n ) \cap E _ { \mathrm { s u m } } } r _ { e } ( \mathbf { x } ) } \end{array}$ , with $r _ { ( m , n ^ { \prime } ) } ~ = ~ \theta _ { m n ^ { \prime } } p _ { n ^ { \prime } } / p _ { m }$ . For a DAG, $\begin{array} { r } { F _ { n } ( \mathbf { x } ) = \sum _ { \pi \in \Pi ( r , n ) } \prod _ { e \in \pi \cap E _ { \mathrm { s u m } } } r _ { e } ( \mathbf { x } ) } \end{array}$ , summing over all root-to-n paths $ { \Pi } ( r , n )$

Two consequences follow. First, product depth does not attenuate flow: only sum edges do, so the notion of “depth” relevant to sharpness is the number of upstream sum edges rather than the raw topological depth. Second, in a DAG a shared node can receive flow through several contexts, and converging paths may compensate for path-wise attenuation; flow therefore need not decay monotonically with depth. The following conditional bound however holds along tree paths.

Corollary 1. Consider a tree-structured PC. If every sumedge responsibility on the root-to-n path satisfies $r _ { e } ( { \bf x } ) \le$ $\rho < 1$ , then $F _ { n } ( \mathbf { x } ) \leq \rho ^ { d _ { \Sigma } ( n ) }$ , and $T _ { n } ( \mathbf { x } ) \leq \rho ^ { 2 d _ { \Sigma } ( n ) } t _ { n } ( \mathbf { x } )$ where $d _ { \Sigma } ( n )$ is the number of upstream sum edges.

The corollary formalizes a mechanism in which the contextual factor can suppress a locally sharp node n reached only through many attenuating routing decisions: its curvature is present, but its global trace contribution is discounted geometrically in the number of upstream sum edges $d _ { \Sigma } ( n )$ . Thus, the global trace can concentrate on shallow, heavily-used nodes, which we also confirm as illustrated in Figure 6.

## Global-Local Ranking Reversals

The decomposition also implies that ranking nodes by their global sharpness contribution is not the same as ranking them by their intrinsic local curvature. A simple analysis gives us the following exact condition for disagreement.

Proposition 2. For any two sum nodes $i , j , T _ { i } > T _ { j } \iff$ $t _ { i } / t _ { j } > ( F _ { j } / F _ { i } ) ^ { 2 }$ . In particular, a locally less curved node i with $t _ { i } < t _ { j }$ outranks a locally sharper node j whenever $F _ { i } / F _ { j } > \sqrt { t _ { j } / t _ { i } }$

The two orderings answer different questions: ranking by $T _ { n }$ identifies which parameter block strongly affects the root likelihood, whereas ranking by $t _ { n }$ identifies which local mixture computation is most sharply curved. Because $T _ { n }$ is amplified by the structural factor ${ \dot { F } } _ { n } ^ { 2 }$ , which is largest at shallow, heavily-used nodes, the top-T<sub>n</sub> nodes need not be the locally sharp ones, accounting for the failure of $\mathrm { t o p } { - } T _ { n }$ selection we saw earlier. The ranking reversals are not a theoretical edge case: in the trained circuit of Figure 5, the root node combines maximal usage with the lowest local trace among the labeled nodes yet attains the largest global contribution, while a deep node with local trace larger by nearly three orders of magnitude contributes almost nothing globally.

Trace contribution by depth zone: accidents / baudio / ad  
![](images/2d14b2d618c366232aee4fadbd19adf590740991530a08adc0de05c1c9d01585.jpg)  
Figure 6: Global trace contributions are biased toward early circuit partitions. Normalized global trace contribution across depth partitions on three representative benchmarks. The observed concentration near earlier partitions is consistent with attenuation through upstream sum-node routing, although raw depth alone does not determine flow.

## Adaptive Sharpness-Aware Learning

The decomposition suggests that global contribution and local curvature should play different roles during learning. The global term $\widehat { T } _ { n } = \widehat { \mathbb { E } } [ F _ { n } ^ { 2 } t _ { n } ]$ measures a node’s contribution to root-level curvature, but also reflects its contextual usage. We therefore propose to use the local trace to determine where regularization should act, while retaining the global trace penalty as the quantity being controlled. This gives a simple modification of global sharpness-aware learning: each sum node receives a gate derived from its local curvature, and the existing trace penalty is scaled by that gate.

Local-Curvature Gating Let $\begin{array} { r } { \widehat { t } _ { n } \ = \ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } t _ { n } ( \mathbf { x } _ { i } ) } \end{array}$ denote the empirical local trace of node n. We assign each sum node a gate $\omega _ { n } \in [ 0 , 1 ]$ and define

$$
R _ { \omega } ( \theta ) = \sum _ { n \in S } \omega _ { n } \widehat { T } _ { n } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { n \in S } \omega _ { n } \sum _ { c \in \mathrm { c h } ( n ) } \left( \frac { F _ { n c } ( \mathbf { x } _ { i } ) } { \theta _ { n c } } \right) ^ { 2 }
$$

The global trace regularizer is recovered when $\omega _ { n } = 1$ for every node. We choose $\omega _ { n } = g ( \widehat { t } _ { n } )$ with g monotone increasing, so that nodes with larger local curvature receive stronger regularization independently of their flow. To reduce sensitivity to the scale of $\widehat { t _ { n } } ,$ we use the bounded gate

$$
\omega _ { n } = \frac { \widehat { t } _ { n } } { \operatorname* { m a x } _ { m \in \mathcal { S } } \widehat { t } _ { m } } ,\tag{6}
$$

![](images/382f10adaae5ef156d05fa5e5ff11a25e14704c0f954505533ecf0329ae1a880.jpg)  
Figure 7: Local trace is lesser concentrated. Cumulative share of $\textstyle \sum _ { n } { \widehat { t } } _ { n }$ as nodes are ranked by $\widehat { t } _ { n }$ . On the evaluated circuits, the local trace is distributed more broadly than the global contribution, indicating that part of the concentration in $\widehat { T } _ { n }$ is introduced by contextual usage.

The node with the largest empirical local trace receives the full penalty, while the remaining nodes are scaled by their curvature relative to that maximum. Gates are recomputed from the current model and held fixed during each parameter update. This separates two roles that are coupled by the global regularizer: $\widehat { t } _ { n }$ allocates regularization across nodes, while $\widehat { T } _ { n }$ remains the curvature contribution being penalized.

Gated EM Update Let $\begin{array} { r } { N _ { n c } = \sum _ { i } F _ { n c } ( \mathbf { x } _ { i } ) } \end{array}$ and $S _ { n c } =$ $\textstyle \sum _ { i } F _ { n c } ( \mathbf { x } _ { i } ) ^ { 2 }$ denote the expected edge count and edge-flow second moment. With fixed gates, directly optimizing the empirical trace penalty gives the following objective:

$$
\operatorname* { m a x } _ { \theta _ { n } } \sum _ { c } N _ { n c } \log \theta _ { n c } - \mu \omega _ { n } \sum _ { c } \frac { S _ { n c } } { \theta _ { n c } ^ { 2 } } \quad \mathrm { s . t . } \quad \sum _ { c } \theta _ { n c } = 1 .
$$

Its stationarity condition, $N _ { n c } / \theta _ { n c } - \lambda + 2 \mu \omega _ { n } S _ { n c } / \theta _ { n c } ^ { 3 } = 0 ,$ is cubic in $\theta _ { n c } .$ . To preserve the tractable EM updates of global trace-regularized learning, we apply the same surrogate as used by (Suresh et al. 2026) with the node-specific strength $\mu _ { n } = \mu \omega _ { n } ,$ , which yields the following update equation.

Proposition 3. For fixed gates $\{ \omega _ { n } \} _ { n \in S }$ , under the surrogate used by global trace-regularized EM, the update for each outgoing weight of sum node n satisfies $\lambda _ { n } \theta _ { n c } ^ { \dot { 2 } } - N _ { n c } \theta _ { n c } -$ $\mu \omega _ { n } N _ { n c } = 0$ , and its positive solution is given by

$$
\theta _ { n c } = \frac { N _ { n c } + \sqrt { N _ { n c } ^ { 2 } + 4 \lambda _ { n } \mu \omega _ { n } N _ { n c } } } { 2 \lambda _ { n } } ,\tag{7}
$$

The update thus preserves the form and asymptotic complexity of the global method, changing only the effective node-wise strength, $\mu \mapsto \mu \omega _ { n }$ . It recovers global trace regularization when $\omega _ { n } = 1$ and approaches the unregularized EM update as $\omega _ { n }  0$ . Though we adopt this simple gate as proof of concept, Prop. 3 holds for any monotone gate, and richer gate functions are natural directions for future work.

## Experiments

We organize the experiments around three questions that connect the theory to the proposed method:

![](images/4b1acf330fc82414cd1d68619f82fbae4efeca1ef252e863289a55146befb3d3.jpg)  
Figure 8: Local-curvature selection preserves fit more effectively than global-contribution selection. Change in test log-likelihood relative to the unregularized model as regularization is applied to all sum nodes or restricted to progressively smaller fractions ranked by global contribution or local curvature. Selecting by global contribution degrades performance whereas selecting by local curvature preserves the baseline on baudio and improves it on ad.

(Q1) How strongly does contextual flow shape the distribution of global trace contributions?

(Q2) Does selecting nodes by local curvature preserve model fit more effectively than selecting by global contribution? (Q3) Does global trace regularization underfit, and can adaptive local-curvature gating recover the lost capacity?

Experimental setup. We evaluate on the 20 binary densityestimation benchmarks (DEBD) using Hidden Chow-Liu Trees (HCLTs) (Liu and den Broeck 2021). All models are implemented in PyJuice (Liu, Ahmed, and den Broeck 2024) with a latent size 100 and trained using EM. We compare the unregularized model, global trace regularization, and the proposed local-curvature-gated regularizer. Results are averaged over 5 random seeds, and the regularization strength µ is selected independently for each method using validation log-likelihood. Further experimental details are provided in the appendix.

Q1: Anatomy of the global trace. We first examine how the factors in $\dot { T _ { n } } = \dot { F _ { n } ^ { 2 } } t _ { n }$ shape global sharpness. Figure 6 shows that global trace contributions concentrate in earlier circuit partitions, consistent with attenuation theory through upstream sum-node routing. Figure 7 further shows that local curvature is substantially less concentrated than global contribution: the top 10% of nodes account for more than 99.99% of $\widehat { T } _ { \mathcal { L } }$ , but it takes over 60% of nodes to contribute the same for bt<sub>n</sub>. Thus, concentration in the global trace is primarily caused by the contextual usage than local curvature.

Q2: Controlled node selection. We next isolate the effect of the ranking criterion. Holding the regularization strength and selected fraction fixed, we apply the trace penalty to nodes chosen by global contribution $\widehat { T } _ { n }$ or local curvature $\widehat { t } _ { n }$ . Figure 8 compares the two criteria on baudio and ad. Restricting the penalty to progressively smaller sets of high-$\widehat { T } _ { n }$ nodes reduces test log-likelihood on both datasets. In contrast, selecting by $\widehat { t } _ { n }$ preserves the unregularized fit on baudio and improves it slightly on ad. These results show that global contribution and local curvature provide materially different signals for allocating regularization.

Table 2: Test log-likelihood on DEBD benchmark datasets (higher is better, mean ± std over 5 seeds). Best µ/method selected via validation LL.
<table><tr><td>Dataset</td><td>Vanilla</td><td>Global Trace</td><td>Gated Regularization</td></tr><tr><td>accidents</td><td>-26.64±0.02</td><td>-29.79±0.13</td><td>-26.55±0.01</td></tr><tr><td>ad</td><td>-18.40±0.09</td><td>-20.36±0.05</td><td>-18.10±0.04</td></tr><tr><td>baudio</td><td>-39.62±0.02</td><td>-42.45±0.02</td><td>-39.62±0.01</td></tr><tr><td>bbc</td><td>-269.58±0.41</td><td>-256.40±0.10</td><td>-260.07±0.26</td></tr><tr><td>bnetflix</td><td>-56.22±0.02</td><td>-59.67±0.02</td><td>-56.21±0.02</td></tr><tr><td>book</td><td>-34.34±0.04</td><td>-36.35±0.03</td><td>-34.16±0.01</td></tr><tr><td>c20ng</td><td>-151.99±0.13</td><td>-157.64±0.05</td><td>-152.22±0.13</td></tr><tr><td>cr52</td><td>-99.09±1.15</td><td>-102.23±1.45</td><td>-97.99±0.70</td></tr><tr><td>cwebkb</td><td>-154.71±0.33</td><td>-157.30±0.02</td><td>-152.90±0.15</td></tr><tr><td>dna</td><td>-87.78±0.24</td><td>-81.33±0.00</td><td>-81.28±0.08</td></tr><tr><td>jester</td><td>-52.82±0.03</td><td>-55.74±0.05</td><td>-52.82±0.03</td></tr><tr><td>kdd</td><td>-2.22±0.01</td><td>-2.36±0.00</td><td>-2.22±0.00</td></tr><tr><td>kosarek</td><td>-10.59±0.01</td><td>-11.08±0.04</td><td>-10.59±0.02</td></tr><tr><td>msnbc</td><td>-6.12±0.01</td><td>-6.47±0.01</td><td>-6.08±0.00</td></tr><tr><td>msweb</td><td>-9.73±0.01</td><td>-10.40±0.02</td><td>-9.73±0.01</td></tr><tr><td>nltcs</td><td>-6.00±0.00</td><td>-6.48±0.00</td><td>-6.00±0.00</td></tr><tr><td>plants</td><td>-12.68±0.03</td><td>-15.99±0.02</td><td>-12.67±0.01</td></tr><tr><td>pumsb_star</td><td>-22.78±0.06</td><td>-28.29±0.06</td><td>-22.75±0.04</td></tr><tr><td>tmovie</td><td>-42.16±0.03</td><td>-48.84±0.04</td><td>-42.21±0.06</td></tr><tr><td>tretail</td><td>-10.84±0.01</td><td>-10.98±0.01</td><td>-10.85±0.01</td></tr></table>

Q3: Underfitting and recovery. Table 2 compares the three methods across all 20 DEBD datasets. Global trace regularization improves over the unregularized model on only 2 datasets and degrades performance on the remaining 18, indicating underfitting. In contrast, the proposed gated method outperforms global trace regularization and matches or improves upon the unregularized model on majority of the datasets. These results show that allocating the trace penalty according to local curvature substantially mitigates the loss of fit induced by uniform global regularization.

## Conclusion

Overall, in this paper we studied sharpness aware learning in probabilistic circuits through a compositional lens. Our analysis showed that a node’s global curvature contribution separates exactly into contextual usage and intrinsic local curvature, providing insights into why global trace regularization can flatten the wrong parts of the model and induce underfitting. Guided by this decomposition, we introduced a gated regularizer that preserves tractable learning, while allocating regularization effectively to prevent underfitting. Future work involves developing richer gating functions that jointly account for local geometry and contextual influence. More broadly, the decomposition also opens up directions for curvature-aware model compression, targeted robustness interventions, and adaptive circuit design, where contextual usage and local sensitivity can jointly guide which components to preserve, regularize, prune, or expand.

## Acknowledgments

SN and SS gratefully acknowledge the generous support by the AFOSR award FA9550-23-1-0239, the ARO award W911NF2010224 and the DARPA Assured Neuro Symbolic Learning and Reasoning (ANSR) award HR001122S0039. CK and HS gratefully acknowledge Dr. Anji Liu for the discussions related to the work and CK, HS, SPV and YZ thank IIT Palakkad for the access to Madhava Cluster.

## References

Ahmed, K.; Teso, S.; Chang, K.; den Broeck, G. V.; and Vergari, A. 2022. Semantic Probabilistic Layers for Neuro-Symbolic Learning. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022.

Andriushchenko, M.; Croce, F.; Muller, M.; Hein, M.; and¨ Flammarion, N. 2023. A Modern Look at the Relationship between Sharpness and Generalization. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202, 840–902. PMLR.

Bekker, J.; Davis, J.; Choi, A.; Darwiche, A.; and Van den Broeck, G. 2015. Tractable learning for complex probability queries. Advances in Neural Information Processing Systems, 28.

Braun, S.; Sidheekh, S.; Vergari, A.; Mundt, M.; Natarajan, S.; and Kersting, K. 2025. Tractable Representation Learning with Probabilistic Circuits. Transactions on Machine Learning Research.

Choi, Y.; Vergari, A.; and den Broeck, G. V. 2020. Probabilistic Circuits: A Unifying Framework for Tractable Probabilistic Models.

Darwiche, A. 2003. A Differential Approach to Inference in Bayesian Networks. Journal ofthe ACM, (3): 280–305.

Dinh, L.; Pascanu, R.; Bengio, S.; and Bengio, Y. 2017. Sharp minima can generalize for deep nets. In International Conference on Machine Learning, 1019–1028. PMLR.

Foret, P.; Kleiner, A.; Mobahi, H.; and Neyshabur, B. 2021. Sharpness-aware Minimization for Efficiently Improving Generalization. In 9th International Conference on Learning Representations, 2021.

Goodfellow, I. J.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A. C.; and Bengio, Y. 2014. Generative Adversarial Nets. In Advances in Neural Information Processing Systems 27: Annual Conference.

Hochreiter, S.; and Schmidhuber, J. 1997. Flat minima. Neural computation, (1): 1–42.

Karanam, A.; Mathur, S.; Sidheekh, S.; and Natarajan, S. 2025. A Unified Framework for Human-Allied Learning of Probabilistic Circuits. In AAAI Conference on Artificial Intelligence, 2025, volume 39, 17779–17787.

Kaur, S.; Cohen, J.; and Lipton, Z. C. 2023. On the Maximum Hessian Eigenvalue and Generalization. In Proceedings of Machine Learning Research.

Keskar, N. S.; Mudigere, D.; Nocedal, J.; Smelyanskiy, M.; and Tang, P. T. P. 2017. On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima. In 5th International Conference on Learning Representations, 2017.

Kingma, D. P.; and Welling, M. 2014. Auto-Encoding Variational Bayes. In 2nd International Conference on Learning Representations, 2014.

Kisa, D.; Broeck, G. V. D.; Choi, A.; and Darwiche, A. 2014. Probabilistic sentential decision diagrams. In International Conference on Knowledge Representation and Reasoning.

Li, H.; Xu, Z.; Taylor, G.; Studer, C.; and Goldstein, T. 2018. Visualizing the Loss Landscape of Neural Nets. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, 6391–6401.

Liu, A.; Ahmed, K.; and den Broeck, G. V. 2024. Scaling Tractable Probabilistic Circuits: A Systems Perspective. In Forty-first International Conference on Machine Learning, 2024.

Liu, A.; and den Broeck, G. V. 2021. Tractable Regularization of Probabilistic Circuits. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems, 2021, 3558–3570.

Liu, A.; Niepert, M.; and den Broeck, G. V. 2024. Image Inpainting via Tractable Steering of Diffusion Models. In The Twelfth International Conference on Learning Representations, 2024.

Nemeˇ cek, J.; Pevnˇ y, T.; and Marecek, J. 2025. Generating\` likely counterfactuals using sum-product networks. In International Conference on Learning Representations, volume 2025, 69804–69835.

Papamakarios, G.; Nalisnick, E. T.; Rezende, D. J.; Mohamed, S.; and Lakshminarayanan, B. 2021. Normalizing Flows for Probabilistic Modeling and Inference. Journal ofMachine Learning Research, 57:1–57:64.

Peharz, R.; Lang, S.; Vergari, A.; Stelzner, K.; Molina, A.; Trapp, M.; den Broeck, G. V.; Kersting, K.; and Ghahramani, Z. 2020. Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits. In 37th International Conference on Machine Learning, 2020, Proceedings of Machine Learning Research, 7563–7574.

Peharz, R.; Vergari, A.; Stelzner, K.; Molina, A.; Trapp, M.; Shao, X.; Kersting, K.; and Ghahramani, Z. 2019. Random Sum-Product Networks: A Simple and Effective Approach to Probabilistic Deep Learning. In Thirty-Fifth Conference on Uncertainty in Artificial Intelligence, 2019, Proceedings of Machine Learning Research, 334–344.

Poon, H.; and Domingos, P. M. 2011. Sum-Product Networks: A New Deep Architecture. In Twenty-Seventh Conference on Uncertainty in Artificial Intelligence, 2011, 337–346.

Rahman, T.; Kothalkar, P.; and Gogate, V. 2014. Cutset networks: A simple, tractable, and scalable approach for improving the accuracy of chow-liu trees. In Machine Learning and Knowledge Discovery in Databases: European Conference, 2014, 630–645. Springer.

Sabu, A.; S, V.; and Krishnan, N. C. 2026. PAR: Plausibility aware Amortized Recourse Generation. arXiv:2601.17309.

Sankar, A. R.; Khasbage, Y.; Vigneswaran, R.; and Balasubramanian, V. N. 2021. A Deeper Look at the Hessian Eigenspectrum of Deep Neural Networks and its Applications to Regularization. In Thirty-Fifth AAAI Conference on Artificial Intelligence, 2021, Thirty-Third Conference on Innovative Applications ofArtificial Intelligence, The Eleventh Symposium on Educational Advances in Artificial Intelligence, 9481–9488.

Sidheekh, S.; Kersting, K.; and Natarajan, S. 2023. Probabilistic Flow Circuits: Towards Unified Deep Models for Tractable Probabilistic Inference. In Uncertainty in Artificial Intelligence, 2023, Proceedings of Machine Learning Research, 1964–1973.

Sidheekh, S.; and Natarajan, S. 2024. Building Expressive and Tractable Probabilistic Generative Models: A Review. In Thirty-Third International Joint Conference on Artificial Intelligence, 2024, 8234–8243.

Sidheekh, S.; and Natarajan, S. 2026. Geometry-Aware Probabilistic Circuits via Voronoi Tessellations. In Forty-third International Conference on Machine Learning.

Sidheekh, S.; Tenali, P.; Mathur, S.; Blasch, E.; Kersting, K.; and Natarajan, S. 2025. Credibility-Aware Multimodal Fusion Using Probabilistic Circuits. In The 28th International Conference on Artificial Intelligence and Statistics.

Suresh, H.; Sidheekh, S.; P, V. S. M.; Natarajan, S.; and Krishnan, N. C. 2026. Tractable Sharpness-Aware Learning of Probabilistic Circuits. Proceedings ofthe AAAI Conference on Artificial Intelligence, 40(30): 25736–25744.

Van Haaren, J.; and Davis, J. 2012. Markov network structure learning: A randomized feature generation approach. In AAAI Conference on Artificial Intelligence, volume 26, 1148–1154.

Ventola, F.; Braun, S.; Yu, Z.; Mundt, M.; and Kersting, K. 2023. Probabilistic circuits that know what they don’t know. In Uncertainty in Artificial Intelligence, 2023, Proceedings of Machine Learning Research, 2157–2167.

Zhang, H.; Dang, M.; Peng, N.; and den Broeck, G. V. 2023. Tractable Control for Autoregressive Language Generation. In International Conference on Machine Learning,2023, Proceedings of Machine Learning Research, 40932–40945.

# Supplementary Material: A Compositional Theory of Curvature in Probabilistic Circuits

## Preliminaries and Notation

We first revisit the definitions and notation used throughout the proofs, so that each subsequent argument can be read in isolation.

Probabilistic circuits. We consider smooth and decomposable probabilistic circuits with alternating layers of sum and product nodes over a set of tractable input distributions. The root node r computes the model density $p _ { r } ( \mathbf { x } )$ . Each sum node $n \in S$ computes a convex combination of its children, $\begin{array} { r } { p _ { n } ( \mathbf { x } ) = \sum _ { c \in \mathrm { c h } ( n ) } \bar { \theta _ { n c } } p _ { c } ( \mathbf { x } ) } \end{array}$ , with nonnegative weights normalized on the simplex, $\textstyle \sum _ { c \in \mathrm { c h } ( n ) } \theta _ { n c } = 1$ . We write $\ell ( \pmb \theta ; \mathbf x ) = - \log p _ { r } ( \mathbf x )$ for the per-sample negative log-likelihood (NLL) and study the trace of its Hessian with respect to the sum weights θ.

Output ratios and routing responsibilities. For a sum node n and child c we define the output ratio $\rho _ { \underline { { n } } c } ( \mathbf { x } ) = p _ { c } ( \mathbf { x } ) / p _ { n } ( \mathbf { x } )$ and the routing responsibility $r _ { n c } ( \mathbf { x } ) = \theta _ { n c } \rho _ { n c } ( \mathbf { x } )$ . The responsibilities form a distribution over children, $\begin{array} { r } { \sum _ { c \in \mathrm { c h } ( n ) } r _ { n c } ( \mathbf { x } ) = 1 } \end{array}$ and we collect the output ratios into the vector $\rho _ { n } ( \mathbf { x } ) = ( \rho _ { n c } ( \mathbf { x } ) ) _ { c \in \mathrm { c h } ( n ) }$

Circuit flow. The circuit flow measures the top-down usage of each node, and is defined by the node-type recursion below. Definition 1 (Circuit flow). The flow $F _ { n } ( \mathbf { x } )$ of a node n is

$$
F _ { n } ( \mathbf { x } ) = \left\{ \begin{array} { l l } { 1 , } & { n = r \mathrm { ( r o o t ) } , } \\ { \displaystyle \sum _ { m \in \mathbf { p a } ( n ) } F _ { m } ( \mathbf { x } ) , } & { n \mathrm { a s u m n o d e } , } \\ { \displaystyle \sum _ { m \in \mathbf { p a } ( n ) } \theta _ { m n } \frac { p _ { n } ( \mathbf { x } ) } { p _ { m } ( \mathbf { x } ) } F _ { m } ( \mathbf { x } ) , } & { n \mathrm { a p r o d u c t \ o r i n p u t n o d e } , } \end{array} \right.\tag{8}
$$

where $\operatorname { p a } ( n )$ denotes the parents of n. By the alternating-layer property, the responsibility factor $\theta _ { m n } p _ { n } / p _ { m } = r _ { m n }$ attaches on exactly the sum edges of the circuit (edges out of a sum node), while product and input nodes inherit their parents’ flow undivided.

Edge flow. The flow carried by the edge from sum node n to child c is $F _ { n c } ( \mathbf { x } ) = \theta _ { n c } \rho _ { n c } ( \mathbf { x } ) F _ { n } ( \mathbf { x } ) = r _ { n c } ( \mathbf { x } ) F _ { n } ( \mathbf { x } )$ . It satisfies $F _ { n c } / \theta _ { n c } = \rho _ { n c } F _ { n }$

Local and global trace. The local trace of sum node n is $\begin{array} { r } { t _ { n } ( \mathbf { x } ) = \sum _ { c \in \mathrm { c h } ( n ) } \rho _ { n c } ( \mathbf { x } ) ^ { 2 } } \end{array}$ , and its global trace contribution is $\begin{array} { r } { T _ { n } ( \mathbf { x } ) = \sum _ { c \in \mathrm { c h } ( n ) } ( F _ { n c } ( \mathbf { x } ) / \theta _ { n c } ) ^ { 2 } } \end{array}$ . Theorem 1 shows these are related by the exact identity $T _ { n } = F _ { n } ^ { 2 } t _ { n }$ . Summing over sum nodes recovers the full NLL Hessian trace, $\begin{array} { r } { \mathrm { T r } ( \nabla _ { \theta } ^ { 2 } \ell ) = \sum _ { n \in S } T _ { n } \geq 0 } \end{array}$ . Empirical batch estimators are denoted $\widehat { t } _ { n }$ and $\widehat { T } _ { n }$

## Proofs for Theoretical Results

In this section, we provide complete, self-contained proofs for all of the key lemmas, propositions, and theorems stated in the main paper

Theorem 2. Consider any sum node n with strictly positive outgoing weights in a smooth, decomposable PC. For any input x, its global trace contribution $T _ { n } ( \mathbf { x } )$ factorizes as $\begin{array} { r } { \dot { T _ { n } } ( \mathbf { x } ) = F _ { n } ( \mathbf { \check { x } } ) ^ { 2 } \tilde { t _ { n } } ( \mathbf { x } ) \dot { , } t _ { n } ( \mathbf { x } ) = \sum _ { c \in \mathrm { c h } ( n ) } ( p _ { c } ( \mathbf { \check { x } } ) / p _ { n } ( \mathbf { x } ) ) ^ { 2 } } \end{array}$ Consequently, Tr $\begin{array} { r } { \left( \nabla _ { \pmb { \theta } } ^ { 2 } \ell ( \pmb { \theta } ; \mathbf { x } ) \right) = \sum _ { n \in \cal S } F _ { n } ( \mathbf { x } ) ^ { 2 } t _ { n } ( \mathbf { x } ) } \end{array}$

Proof. The global trace contribution of node n is the sum over its outgoing edges of the squared diagonal Hessian entries, which equal $( F _ { n c } ( \mathbf { \check { x } } ) / \theta _ { n c } ) ^ { 2 }$ . Substituting the edge-flow identity $F _ { n c } / \theta _ { n c } = \rho _ { n c } F _ { n }$ from Section ,

$$
T _ { n } ( \mathbf { x } ) = \sum _ { c \in \mathrm { c h } ( n ) } \left( \frac { F _ { n c } ( \mathbf { x } ) } { \theta _ { n c } } \right) ^ { 2 } = \sum _ { c \in \mathrm { c h } ( n ) } \left( \frac { p _ { c } ( \mathbf { x } ) } { p _ { n } ( \mathbf { x } ) } F _ { n } ( \mathbf { x } ) \right) ^ { 2 } .\tag{9}
$$

The flow $F _ { n } ( \mathbf { x } )$ does not depend on the child index $c ,$ so it factors out of the sum,

$$
T _ { n } ( \mathbf { x } ) = F _ { n } ( \mathbf { x } ) ^ { 2 } \sum _ { c \in \mathrm { c h } ( n ) } \left( \frac { p _ { c } ( \mathbf { x } ) } { p _ { n } ( \mathbf { x } ) } \right) ^ { 2 } .\tag{10}
$$

Recognizing the remaining sum as the local trace $\begin{array} { r } { { t } _ { n } ( \mathbf { x } ) = \sum _ { c \in \mathrm { c h } ( n ) } \rho _ { n c } ( \mathbf { x } ) ^ { 2 } } \end{array}$ yields the factorization,

$$
T _ { n } ( \mathbf { x } ) = F _ { n } ( \mathbf { x } ) ^ { 2 } t _ { n } ( \mathbf { x } ) .\tag{11}
$$

Summing over all sum nodes gives the full Hessian trace $\begin{array} { r } { \mathrm { T r } ( \nabla _ { \pmb { \theta } } ^ { 2 } \ell ) = \sum _ { n \in \cal S } F _ { n } ^ { 2 } t _ { n } \geq 0 } \end{array}$ , since each term is a product of squares. □

Proposition 4 (Rank-one local Hessian). The gradient and Hessian of $\ell _ { n }$ in the edge weights $\pmb \theta _ { n }$ are $\nabla _ { \pmb { \theta } _ { n } } \ell _ { n } = - \pmb { \rho } _ { n }$ and $H _ { n } ( \mathbf { \bar { x } } ) = \nabla _ { \pmb { \theta } _ { n } } ^ { 2 } \ell _ { n } = \pmb { \rho } _ { n } \pmb { \rho } _ { n } ^ { \top }$ . Hence, $H _ { n }$ is positive semidefinite and, whenever ${ \pmb \rho } _ { n } \neq 0$ , has rank one, with unique nonzero eigenvalue $\lambda _ { \operatorname* { m a x } } ( H _ { n } ) = \| \pmb { \rho _ { n } } \| _ { 2 } ^ { 2 } = t _ { n }$ . Consequently $t _ { n } = { \mathrm { T r } } ( H _ { n } ) = \| H _ { n } \| _ { 2 } = \| H _ { n } \| _ { F }$

$$
\begin{array} { r l } & { P r o o f . \ \ell _ { n } ( \pmb { \theta } _ { n } ; \mathbf { x } ) = - \log \big ( \sum _ { c } \theta _ { n c } p _ { c } ( \mathbf { x } ) \big ) , \rho _ { n c } ( \mathbf { x } ) = p _ { c } ( \mathbf { x } ) / p _ { n } ( \mathbf { x } ) , p _ { n } ( \mathbf { x } ) = \sum _ { c } \theta _ { n c } p _ { c } ( \mathbf { x } ) } \end{array}
$$

First order derivative

$$
\frac { \partial \ell _ { n } ( \pmb { \theta } _ { n } ; \mathbf { x } ) } { \partial \theta _ { n c } } = \frac { - \partial \log \big ( \sum _ { c } \theta _ { n c } p _ { c } ( \mathbf { x } ) \big ) } { \partial \theta _ { n c } } = \frac { - 1 } { p _ { n } ( \mathbf { x } ) } \frac { \partial \sum _ { c } \theta _ { n c } p _ { c } ( \mathbf { x } ) } { \partial \theta _ { n c } }\tag{12}
$$

$$
\frac { \partial \ell _ { n } ( \pmb { \theta } _ { n } ; \mathbf { x } ) } { \partial \theta _ { n c } } = - \left( \frac { p _ { c } ( \mathbf { x } ) } { p _ { n } ( \mathbf { x } ) } \right)\tag{13}
$$

$$
\implies \nabla _ { \pmb { \theta } _ { n } } \ell _ { n } = - \pmb { \rho } _ { n }\tag{14}
$$

Second order derivative

$$
\frac { \partial ^ { 2 } \ell _ { n } ( \pmb { \theta } _ { n } ; \mathbf { x } ) } { \partial \theta _ { n c ^ { \prime } } \theta _ { n c } } = - \frac { \partial \left( p _ { c } ( \mathbf { x } ) / p _ { n } ( \mathbf { x } ) \right) } { \partial \theta _ { n c ^ { \prime } } } = - \left[ \frac { p _ { n } ( \mathbf { x } ) p _ { c } ( \mathbf { x } ) ^ { \prime } - p _ { n } ( \mathbf { x } ) ^ { \prime } p _ { c } ( \mathbf { x } ) } { p _ { n } ( \mathbf { x } ) ^ { 2 } } \right]\tag{15}
$$

As $p _ { c } ( \mathbf { x } )$ is independent of $\theta _ { n c ^ { \prime } }$ we get

$$
\frac { \partial \left( p _ { c } ( \mathbf { x } ) / p _ { n } ( \mathbf { x } ) \right) } { \partial \theta _ { n c ^ { \prime } } } = - \left[ \frac { - p _ { c ^ { \prime } } ( \mathbf { x } ) p _ { c } ( \mathbf { x } ) } { p _ { n } ( \mathbf { x } ) ^ { 2 } } \right] = \frac { p _ { c ^ { \prime } } ( \mathbf { x } ) } { p _ { n } ( \mathbf { x } ) } \frac { p _ { c } ( \mathbf { x } ) } { p _ { n } ( \mathbf { x } ) } = \rho _ { n c ^ { \prime } } \rho _ { n c }\tag{16}
$$

Thus $H _ { n } = \rho _ { n } \pmb { \rho } _ { n } ^ { \top }$ . A nonzero outer product has rank one and unique nonzero eigenvalue $\| \pmb { \rho } _ { n } \| _ { 2 } ^ { 2 }$ . For a rank-one positive semidefinite matrix, the trace, spectral norm, and Frobenius norm all equal this eigenvalue. Trace of local Hessian

$$
\operatorname { T r } ( H _ { n } ( \mathbf { x } ) ) = \operatorname { T r } ( \pmb { \rho _ { n } \rho _ { n } ^ { \top } } ) = \sum _ { c } \left( \frac { p _ { c } ( \mathbf { x } ) } { p _ { n } ( \mathbf { x } ) } \right) ^ { 2 } = t _ { n } ( \mathbf { x } )\tag{17}
$$

Proposition 5. For any two sum nodes $i , j , T _ { i } > T _ { j } \iff t _ { i } / t _ { j } > ( F _ { j } / F _ { i } ) ^ { 2 }$ . In particular, a locally less curved node i with $t _ { i } < t _ { j }$ outranks a locally sharper node j wheneve $F _ { i } / F _ { j } > \sqrt { t _ { j } / t _ { i } }$

Proof.

$$
T _ { i } = F _ { i } ^ { 2 } t _ { i } , T _ { j } = F _ { j } ^ { 2 } t _ { j }
$$

$$
T _ { i } > T _ { j } \iff F _ { i } ^ { 2 } t _ { i } > F _ { j } ^ { 2 } t _ { j }
$$

$$
T _ { i } > T _ { j } \iff { \frac { t _ { i } } { t _ { j } } } > \left( { \frac { F _ { j } } { F _ { i } } } \right) ^ { 2 }
$$

Lemma 2. For a tree-structured PC, the node flow is the product of routing responsibilities along the unique root-to-n path $\pi ( r , n )$ , taken over its sum edges $( E _ { s u m } )$ only: $\begin{array} { r } { F _ { n } ( \mathbf { x } ) = \prod _ { e \in \pi ( r , n ) \cap E _ { \mathrm { s u m } } } r _ { e } ( \mathbf { x } ) } \end{array}$ , with $r _ { ( m , n ^ { \prime } ) } = \theta _ { m n ^ { \prime } } p _ { n ^ { \prime } } / p _ { m }$ . For a DAG, $\begin{array} { r } { F _ { n } ( \mathbf { x } ) = \sum _ { \pi \in \Pi ( r , n ) } \prod _ { e \in \pi \cap E _ { \mathrm { s u m } } } r _ { e } ( \mathbf { x } ) } \end{array}$ , summing over all root-to-n paths $ { \Pi } ( r , n )$

Proof. We proceed by induction on the depth d of node n in the tree-structured PC.

Base case $( d = 0 )$ . Node n is the root r. By definition, $F _ { r } ( \mathbf { x } ) = 1$ . The product over an empty index set (no edges on the root-to-root path) is also 1, so the claim holds trivially.

Inductive step. Assume the claim holds for every node at depth strictly less than d. Let n be a node at depth $d ,$ and let $m = \operatorname { p a } ( n )$ denote its unique parent ( PC is tree-structured). We consider two cases according to the type of the edge $( m , n )$

Case 1: $( m , n )$ is a product edge (i.e. m is a product node).

By the flow recursion, product nodes propagate their flow without weighting:

$$
F _ { n } ( \mathbf { x } ) \ = \ F _ { m } ( \mathbf { x } ) .
$$

Since m is at depth $d - 1 .$ , the inductive hypothesis gives

$$
F _ { m } ( \mathbf { x } ) \ = \prod _ { \substack { e \in \pi ( r , m ) \cap E _ { \mathrm { s u m } } } } r _ { e } ( \mathbf { x } ) .
$$

Because $( m , n ) \notin E _ { \mathrm { s u m } }$ , the sum-edge sets along the two paths coincide: $\pi ( r , n ) \cap E _ { \mathrm { s u m } } = \pi ( r , m ) \cap E _ { \mathrm { s u m } }$ . Therefore

$$
F _ { n } ( \mathbf { x } ) \ = \prod _ { e \in \pi ( r , n ) \cap E _ { \mathrm { s u m } } } r _ { e } ( \mathbf { x } ) .\tag{18}
$$

Case $\pmb { 2 } \colon ( m , n )$ is a sum edge (i.e. m is a sum node).

By the flow recursion and the definition of routing responsibility $r _ { ( m , n ) } ( \mathbf { x } ) = \theta _ { m n } p _ { n } ( \mathbf { x } ) / p _ { m } ( \mathbf { x } ) \mathrm { : }$

$$
F _ { n } ( \mathbf { x } ) = \theta _ { m n } { \frac { p _ { n } ( \mathbf { x } ) } { p _ { m } ( \mathbf { x } ) } } F _ { m } ( \mathbf { x } ) = r _ { ( m , n ) } ( \mathbf { x } ) F _ { m } ( \mathbf { x } ) .
$$

Since m is at depth d − 1, the inductive hypothesis gives

$$
F _ { m } ( \mathbf { x } ) \ = \prod _ { \substack { e \in \pi ( r , m ) \cap E _ { \mathrm { s u m } } } } r _ { e } ( \mathbf { x } ) .
$$

Because $( m , n ) \in E _ { \mathrm { s u m } }$ and $\pi ( r , n ) = \pi ( r , m ) \cup \{ ( m , n ) \}$ , the new sum-edge factor yields

$$
\begin{array} { l } { F _ { n } ( \mathbf x ) \ = \ r _ { ( m , n ) } ( \mathbf x ) \cdot \displaystyle \prod _ { e \in \pi ( r , m ) \cap E _ { \mathrm { s u m } } } r _ { e } ( \mathbf x ) } \\ { \displaystyle = \ \prod _ { e \in \pi ( r , n ) \cap E _ { \mathrm { s u m } } } r _ { e } ( \mathbf x ) . } \end{array}
$$

Both cases establish the inductive step, so by strong induction the formula

$$
F _ { n } ( \mathbf { x } ) \ = \prod _ { e \in \pi ( r , n ) \cap E _ { \mathrm { s u m } } } r _ { e } ( \mathbf { x } )
$$

holds for every node n in the tree-structured PC.

For general DAG structure

Base case $( d = 0 )$ . Node n is the root r. By definition $F _ { r } ( \mathbf { x } ) = 1$ . The only root-to-root path is the empty path $\pi _ { \mathcal { O } } \mathrm { : }$ ; its edge product equals 1 (empty product), and the sum over this single path equals 1. Hence the claim holds.

Inductive step. Assume the claim holds for every node at depth strictly less than $d \geq 1$ . Let n be a node at depth d with parent se ${ \mathrm { t } } \operatorname { p a } ( n ) = \left\{ m _ { 1 } , \ldots , m _ { k } \right\}$ , each parent at depth d − 1. By the inductive hypothesis, for every $m _ { i } \in \mathrm { p a } ( n )$

$$
F _ { m _ { i } } ( \mathbf { x } ) \ = \sum _ { \pi \in \Pi ( r , m _ { i } ) \ : e \in \pi \cap E _ { \mathrm { s u m } } } \prod _ { r _ { e } ( \mathbf { x } ) . }\tag{19}
$$

We consider two cases according to the type of $n .$

Case 1: n is a sum node.

By the alternating-layer property every parent $m _ { i }$ is a product node, so every edge $( m _ { i } , n ) \notin E _ { \mathrm { s u m } }$ . The flow recursion for a sum node gives

$$
F _ { n } ( \mathbf { x } ) \ = \ \sum _ { m _ { i } \in \mathrm { p a } ( n ) } F _ { m _ { i } } ( \mathbf { x } ) .
$$

Substituting (19):

$$
F _ { n } ( \mathbf { x } ) \ = \ \sum _ { m _ { i } \in \mathrm { { p a } } ( n ) } \sum _ { \pi \in \Pi ( r , m _ { i } ) } \prod _ { e \in \pi \cap E _ { \mathrm { { s u m } } } } r _ { e } ( \mathbf { x } ) .
$$

Every root-to-n path $\pi \in \Pi ( r , n )$ passes through exactly one parent $m _ { i } \in \mathrm { p a } ( n )$ and then traverses the product edge $( m _ { i } , n )$ Because $( m _ { i } , n ) \notin E _ { \mathrm { s u m } }$ , adding this edge does not change the set of sum edges on the path:

$$
\pi \cap E _ { \mathrm { s u m } } = \pi ^ { \prime } \cap E _ { \mathrm { s u m } } , \qquad { \mathrm { w h e r e ~ } } \pi ^ { \prime } = \pi ( r , m _ { i } ) { \mathrm { ~ i s ~ t h e ~ p r e f i x ~ o f ~ } } \pi { \mathrm { ~ u p ~ t o ~ } } m _ { i } .
$$

Hence the double sum exactly enumerates all root-to-n paths:

$$
F _ { n } ( \mathbf { x } ) \ = \ \sum _ { \pi \in \Pi ( r , n ) } \prod _ { e \in \pi \cap E _ { \mathrm { s u m } } } r _ { e } ( \mathbf { x } ) .
$$

Case 2: n is a product or input node.

By the alternating-layer property every parent $m _ { i }$ is a sum node, so every edge $( m _ { i } , n ) \in E _ { \mathrm { s u m } }$ carries routing responsibility $r _ { ( m _ { i } , n ) } ( \mathbf { x } ) = \theta _ { m _ { i } n } p _ { n } ( \mathbf { x } ) / p _ { m _ { i } } ( \mathbf { x } )$ . The flow recursion reads

$$
F _ { n } ( \mathbf { x } ) \ = \ \sum _ { m _ { i } \in \mathrm { { p a } } ( n ) } r _ { ( m _ { i } , n ) } ( \mathbf { x } ) F _ { m _ { i } } ( \mathbf { x } ) .
$$

Substituting (19):

$$
F _ { n } ( \mathbf { x } ) \ = \ \sum _ { m _ { i } \in \mathrm { { p a } } ( n ) } r _ { ( m _ { i } , n ) } ( \mathbf { x } ) \sum _ { \pi ^ { \prime } \in \Pi ( r , m _ { i } ) } \prod _ { e \in \pi ^ { \prime } \cap E _ { \mathrm { { s u m } } } } r _ { e } ( \mathbf { x } ) .
$$

Every root-to-n path $\pi \in \Pi ( r , n )$ decomposes uniquely as a prefix $\pi ^ { \prime } \in \Pi ( r , m _ { i } )$ for exactly one $m _ { i } \in \mathrm { p a } ( n )$ followed by the sum edge $( m _ { i } , n )$ . Since $( m _ { i } , n ) \in E _ { \mathrm { s u m } }$ , appending this edge extends the edge product by exactly one factor:

$$
\prod _ { e \in \pi \cap E _ { \mathrm { { s u m } } } } r _ { e } ( \mathbf x ) \ = \ r _ { ( m _ { i } , n ) } ( \mathbf x ) \cdot \prod _ { e \in \pi ^ { \prime } \cap E _ { \mathrm { { s u m } } } } r _ { e } ( \mathbf x ) .
$$

Re-indexing the double sum over all root-to-n paths therefore gives

$$
\begin{array} { l } { F _ { n } ( \mathbf x ) \ = \ \displaystyle \sum _ { m _ { i } \in \mathrm { p a } ( n ) } \ \sum _ { \pi ^ { \prime } \in \Pi ( r , m _ { i } ) } r _ { ( m _ { i } , n ) } ( \mathbf x ) \prod _ { e \in \pi ^ { \prime } \cap E _ { \mathrm { s u m } } } r _ { e } ( \mathbf x ) } \\ { \displaystyle = \ \sum _ { \pi \in \Pi ( r , n ) } \prod _ { e \in \pi \cap E _ { \mathrm { s u m } } } r _ { e } ( \mathbf x ) . } \end{array}
$$

Conclusion. Both cases establish the inductive step. By strong induction,

$$
F _ { n } ( \mathbf { x } ) \ = \ \sum _ { \pi \in \Pi ( r , n ) \ : e \in \pi \cap E _ { \mathrm { s u m } } } r _ { e } ( \mathbf { x } )
$$

Corollary 2. Consider a tree-structured PC. If every sum-edge responsibility on the root-to-n path satisfies $r _ { e } ( \mathbf { x } ) \leq \rho < 1$ , then $F _ { n } ( \mathbf { x } ) \leq \rho ^ { d _ { \Sigma } ( n ) }$ , and $T _ { n } ( \mathbf { x } ) \leq \rho ^ { 2 d _ { \Sigma } ( n ) } t _ { n } ( \mathbf { x } )$ , where $d _ { \Sigma } ( n )$ is the number of upstream sum edges.

Proof. Part 1: Bound on $F _ { n } ( \mathbf { x } )$

By the Lemma 1, the node flow at n in a tree-structured PC is the product of routing responsibilities over all sum edges on the unique root-to-n path:

$$
F _ { n } ( \mathbf { x } ) \ = \prod _ { e \in \pi ( r , n ) \cap E _ { \mathrm { s u m } } } r _ { e } ( \mathbf { x } ) .
$$

This product contains exactly $d _ { \Sigma } ( n )$ factors. By hypothesis, each factor satisfies $r _ { e } ( \mathbf { x } ) \leq \rho < 1$ , so:

$$
F _ { n } ( \mathbf { x } ) \ = \prod _ { \substack { e \in \pi ( r , n ) \cap E _ { \mathrm { { s u m } } } } } r _ { e } ( \mathbf { x } ) \ \leq \ \rho ^ { d _ { \Sigma } ( n ) } .
$$

Part 2: Bound on $T _ { n } ( \mathbf { x } )$

By the decomposition in Theorem 1:

$$
T _ { n } ( { \bf x } ) = F _ { n } ( { \bf x } ) ^ { 2 } \cdot t _ { n } ( { \bf x } ) .
$$

Since $t _ { n } ( \mathbf { x } ) \geq 0$ (it is a sum of squares), we may apply the flow bound directly:

$$
T _ { n } ( { \bf x } ) = F _ { n } ( { \bf x } ) ^ { 2 } \cdot t _ { n } ( { \bf x } ) \le \Big ( \rho ^ { d _ { \Sigma } ( n ) } \Big ) ^ { 2 } \cdot t _ { n } ( { \bf x } ) = \rho ^ { 2 d _ { \Sigma } ( n ) } t _ { n } ( { \bf x } ) .
$$

## Adaptive Gated Regularization

Proposition 6. For fixed gates $\{ \omega _ { n } \} _ { n \in { \cal S } } ,$ , under the surrogate used by global trace-regularized EM, the update for each outgoing weight of sum node n satisfies $\bar { \lambda } _ { n } \theta _ { n c } ^ { 2 } - N _ { n c } \theta _ { n c } - \mu \omega _ { n } \bar { N } _ { n c } = 0$ , and its positive solution is given by

$$
\theta _ { n c } = \frac { N _ { n c } + \sqrt { N _ { n c } ^ { 2 } + 4 \lambda _ { n } \mu \omega _ { n } N _ { n c } } } { 2 \lambda _ { n } } ,\tag{20}
$$

Proof. Adding the adaptive regularizer to the EM objective of the PC we obtain

$$
\theta _ { n \cdot } ^ { * } = \arg \operatorname* { m a x } _ { \theta _ { n c } } \ \sum _ { c \in \mathrm { c h } ( n ) } N _ { n c } ( x ) \ \log \ \theta _ { n c }
$$

$$
\mathrm { s u b j e c t \ t o } \sum _ { c \in \mathrm { c h } ( n ) } \theta _ { n c } = 1 ,
$$

$$
\omega _ { n } ( x ) \sum _ { c \in c h ( n ) } \left( \frac { N _ { n c } ( x ) } { \theta _ { n c } } \right) \leq m
$$

The Lagrangian formulation of the constrained maximization objective is

$$
\begin{array} { c } { { \displaystyle { \mathcal { L } } ( \theta _ { n \cdot } , \lambda , \mu ) = \sum _ { c \in \mathrm { c h } ( n ) } N _ { n c } ( x ) \log \theta _ { n c } - \lambda \left( \sum _ { c \in \mathrm { c h } ( n ) } \theta _ { n c } - 1 \right) } } \\ { { - \displaystyle { \mu \left( \omega _ { n } ( x ) \sum _ { c \in \mathrm { c h } ( n ) } \left( \frac { N _ { n c } ( x ) } { \theta _ { n c } } \right) ^ { 2 } - m \right) } } } \end{array}
$$

Differentiating wrt $\theta _ { n c }$ we $\mathrm { g e t . }$ ,

$$
\frac { \partial \mathcal { L } ( \theta _ { n } . , \lambda , \mu ) } { \partial \theta _ { n c } } = \frac { N _ { n c } } { \theta _ { n c } } - \lambda + \mu \left( \omega _ { n } \frac { N _ { n c } } { \theta _ { n c } ^ { 2 } } \right)
$$

equating this to 0 yields the quadratic equation

$$
\lambda \theta _ { n c } ^ { 2 } - N _ { n c } \theta _ { n c } - \mu \omega _ { n } N _ { n c } = 0
$$

whose solution obtained by the quadratic formula is

$$
\theta _ { n c } = \frac { N _ { n c } + \sqrt { N _ { n c } ( x ) ^ { 2 } + 4 \lambda \mu \omega _ { n } N _ { n c } } } { 2 \lambda }
$$

## Experimental Setup and Implementation Details

In this section, we provide the details regarding the datasets, model architecture, hyperparameter used for the experimental results reported in the main paper.

Real World Data. We consider the standard suite of 20 binary density estimation benchmark (Van Haaren and Davis 2012; Bekker et al. 2015). These include small to large domains such as nltcs (16 variables), up to ad (1556 variables). Table 3 summarizes the number of variables and data points in the train, validation, and test splits for each of the 20 datasets.

## Model Architectures

• PyJuice (Liu, Ahmed, and den Broeck 2024), for our experiments on the binary density estimation datasets. We use the Hidden Chow-Liu Tree structure (Liu and den Broeck 2021; Liu, Ahmed, and den Broeck 2024) which is a generative probabilistic model that extends the classical Chow-Liu tree by introducing latent (hidden) variables to model complex dependencies among observed variables more effectively. The tree topology is learned from data using maximum-likelihood (Chow-Liu algorithm). The observed variables are then pushed to the leaves, introducing latent variables to occupy the internal nodes, forming a latent tree structure. As a result, the learned structure can vary across datasets, adapting to the underlying statistical relationships. The latent size (num latent) refers to the number of states each hidden variable can take, and it serves as a key hyperparameter that controls the model’s capacity. We set num latent= 100 for all our binary density estimation datasets.

Table 3: Overview of the 20 binary density estimation datasets, showing the number of variables and the number of instances in the training, validation, and test splits.
<table><tr><td>Dataset Name</td><td>#vars</td><td>#train</td><td>#valid</td><td>#test</td></tr><tr><td>nltcs</td><td>16</td><td>16181</td><td>2157</td><td>3236</td></tr><tr><td>msnbc</td><td>17</td><td>291326</td><td>38843</td><td>58265</td></tr><tr><td>kdd</td><td>65</td><td>180092</td><td>19907</td><td>34955</td></tr><tr><td>plants</td><td>69</td><td>17412</td><td>2321</td><td>3482</td></tr><tr><td>baudio</td><td>100</td><td>15000</td><td>2000</td><td>3000</td></tr><tr><td>jester</td><td>100</td><td>9000</td><td>1000</td><td>4116</td></tr><tr><td>bnetflix</td><td>100</td><td>15000</td><td>2000</td><td>3000</td></tr><tr><td>accidents</td><td>111</td><td>12758</td><td>1700</td><td>2551</td></tr><tr><td>tretail</td><td>135</td><td>22041</td><td>2938</td><td>4408</td></tr><tr><td>pumsb_star</td><td>163</td><td>12262</td><td>1635</td><td>2452</td></tr><tr><td>dna</td><td>180</td><td>1600</td><td>400</td><td>1186</td></tr><tr><td>kosarek</td><td>190</td><td>33375</td><td>4450</td><td>6675</td></tr><tr><td>msweb</td><td>294</td><td>29441</td><td>3270</td><td>5000</td></tr><tr><td>tmovie</td><td>500</td><td>4524</td><td>1002</td><td>591</td></tr><tr><td>book</td><td>500</td><td>8700</td><td>1159</td><td>1739</td></tr><tr><td>cwebkb</td><td>839</td><td>2803</td><td>558</td><td>838</td></tr><tr><td>cr52</td><td>889</td><td>6532</td><td>1028</td><td>1540</td></tr><tr><td>c20ng</td><td>910</td><td>11293</td><td>3764</td><td>3764</td></tr><tr><td>bbc</td><td>1058</td><td>1670</td><td>225</td><td>330</td></tr><tr><td>ad</td><td>1556</td><td>2461</td><td>327</td><td>491</td></tr></table>

## Additional Results

This section provides additional evidence for the empirical observations and method comparisons reported in the main paper. We first visualize the two components of the curvature decomposition and present a controlled synthetic example. We then report full-dataset results for underfitting, trace concentration, node-selection criteria, and performance across data regimes.

## Visualizing the Curvature Decomposition

Figure 9 visualizes the empirical contextual usage, local curvature, and global trace contribution of the sum nodes in a trained circuit. In this example, several nodes with large local curvature receive little flow and therefore contribute weakly to the global trace. Conversely, nodes near the root receive substantial flow and can dominate the global contribution despite moderate local curvature. The labeled nodes illustrate the ranking reversal characterized in Proposition 2. Node A has the smallest local curvature among the three but the largest global contribution because it receives the most flow. Node B is locally sharper but contributes less globally because its contextual usage is small. The example illustrates why global contribution and local curvature provide different signals for allocating regularization.

## A 2D Synthetic Data Distribution

We construct a two-dimensional example to illustrate how uniform and locally gated trace regularization can favor different regions of parameter space. The training set contains 50 samples from a mixture of three isotropic Gaussians with standard deviation σ = 0.25, centered at the vertices of an equilateral triangle. We additionally include eight noise samples distributed around a ring of radius 2.3, with radial jitter 0.12. The test set contains 800 samples drawn only from the Gaussian mixture. We train a PC with Gaussian input distributions and 16 sum and input units under three regimes: unregularized maximum likelihood, global trace regularization, and adaptive local-curvature gating. All methods use the same initialization, and both regularized methods use $\mu = 0 . 2 0$ . Following Li et al. (2018), we project the optimization trajectories onto the plane spanned by the leading two principal components of the parameter iterates. We then evaluate the train NLL, test NLL, and exact Hessian trace over this plane. Figure 10 shows that the three methods converge to different regions. The unregularized model attains the lowest train NLL but higher test NLL and curvature. Global regularization reaches a flatter region but sacrifices both train and test fit. Adaptive gating retains lower curvature while reaching a region with better test fit than uniform global regularization.

## Underfitting at full data regime

Figure 11 reports the training and test log-likelihood trajectories of the unregularized and global trace-regularized models across the DEBD benchmarks using the full training sets. On most datasets, global regularization lowers both training and test likelihood. The resulting degradation is therefore consistent with reduced model fit rather than a larger train–test gap.

## Sparsity of Global and Local Trace

Figures 12 and 13 compare how global trace contribution and local curvature are distributed across sum nodes. The global contribution is highly concentrated across the evaluated datasets, with a small fraction of nodes accounting for most of the total trace. Local curvature is generally distributed across a broader set of nodes, indicating that contextual flow contributes to the concentration observed in the global trace.

## Thresholding based node selection results

We next compare two criteria for selecting where trace regularization acts. In Figure 14, nodes are ranked by global contribution. Restricting the penalty to progressively smaller sets of high-contribution nodes generally fails to recover the unregularized likelihood and often degrades it further. Figure 15 instead ranks nodes by local curvature. Relative to global-contribution selection, local-curvature selection more consistently preserves the unregularized fit across the evaluated datasets.

## Low Data Regime Experiments

Table 4: Test log-likelihood comparison across data fractions for DEBD. For gated methods, the variant is selected per dataset/fraction by highest mean validation LL.
<table><tr><td rowspan="2">Dataset</td><td colspan="3">25% Data</td><td colspan="3">50% Data</td><td colspan="3">100% Data</td></tr><tr><td>Vanilla Global Trace Best Gated Vanilla Global Trace Best Gated Vanilla Global Trace Best Gated</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>accidents</td><td>-28.59</td><td>-29.83</td><td>-28.64</td><td>-27.13</td><td>-29.84</td><td>-27.12</td><td>-26.63</td><td>-29.69</td><td>-26.64</td></tr><tr><td>ad</td><td>-28.55</td><td>-29.64</td><td>-28.86</td><td>-20.51</td><td>-22.07</td><td>-20.47</td><td>-18.30</td><td>-20.37</td><td>-18.22</td></tr><tr><td>baudio</td><td>-41.13</td><td>-42.46</td><td>-41.16</td><td>-39.99</td><td>-42.40</td><td>-40.01</td><td>-39.63</td><td>-42.45</td><td>-39.63</td></tr><tr><td>bbc</td><td>-379.30</td><td>-281.33</td><td>-377.54</td><td>-312.26</td><td>-262.64</td><td>-309.58</td><td>-269.69</td><td>-256.43</td><td>-269.39</td></tr><tr><td>bnetflix</td><td>-59.06</td><td>-59.87</td><td>-59.24</td><td>-56.85</td><td>-59.83</td><td>-56.95</td><td>-56.22</td><td>-59.67</td><td>-56.21</td></tr><tr><td>book</td><td>-37.91</td><td>-37.48</td><td>-37.45</td><td>-35.44</td><td>-36.75</td><td>-35.29</td><td>-34.37</td><td>-36.31</td><td>-34.23</td></tr><tr><td>c20ng</td><td>-162.01</td><td>-160.21</td><td>-161.97</td><td>-154.92</td><td>-158.67</td><td>-154.94</td><td>-151.97</td><td>-157.60</td><td>-152.04</td></tr><tr><td>cr52</td><td>-107.98</td><td>-103.75</td><td>-106.04</td><td>-102.79</td><td>-102.07</td><td>-101.00</td><td>-99.35</td><td>-102.15</td><td>-100.26</td></tr><tr><td>cwebkb</td><td>-204.80</td><td>-164.74</td><td>-202.88</td><td>-171.74</td><td>-159.32</td><td>-170.11</td><td>-154.78</td><td>-157.34</td><td>-154.89</td></tr><tr><td>dna</td><td>-93.73</td><td>-82.61</td><td>-93.44</td><td>-90.35</td><td>-81.76</td><td>-90.07</td><td>-87.70</td><td>-81.35</td><td>-87.11</td></tr><tr><td>jester</td><td>-57.38</td><td>-55.65</td><td>-57.43</td><td>-54.05</td><td>-55.90</td><td>-54.26</td><td>-52.82</td><td>-55.81</td><td>-52.88</td></tr><tr><td>kdd</td><td>-2.22</td><td>-2.37</td><td>-2.21</td><td>-2.22</td><td>-2.36</td><td>-2.21</td><td>-2.23</td><td>-2.37</td><td>-2.22</td></tr><tr><td>kosarek</td><td>-10.83</td><td>-11.11</td><td>-10.79</td><td>-10.64</td><td>-11.06</td><td>-10.67</td><td>-10.59</td><td>-11.09</td><td>-10.59</td></tr><tr><td>msnbc</td><td>-6.10</td><td>-6.46</td><td>-6.08</td><td>-6.11</td><td>-6.47</td><td>-6.08</td><td>-6.12</td><td>-6.46</td><td>-6.08</td></tr><tr><td>msweb</td><td>-10.00</td><td>-10.45</td><td>-9.97</td><td>-9.82</td><td>-10.39</td><td>-9.81</td><td>-9.72</td><td>-10.40</td><td>-9.74</td></tr><tr><td>nltcs</td><td>-6.09</td><td>-6.50</td><td>-6.10</td><td>-6.02</td><td>-6.47</td><td>-6.02</td><td>-6.00</td><td>-6.48</td><td>-6.00</td></tr><tr><td>plants</td><td>-12.97</td><td>-15.90</td><td>-12.96</td><td>-12.77</td><td>-15.97</td><td>-12.77</td><td>-12.67</td><td>-15.98</td><td>-12.66</td></tr><tr><td>pumsb_star -23.82</td><td></td><td>-28.75</td><td>-23.65</td><td>-22.79</td><td>-28.35</td><td>-22.77</td><td>-22.79</td><td>-28.35</td><td>-22.78</td></tr><tr><td>tmovie</td><td>-51.23</td><td>-50.51</td><td>-50.89</td><td>-44.55</td><td>-49.57</td><td>-44.58</td><td>-42.17</td><td>-48.84</td><td>-42.11</td></tr><tr><td>tretail</td><td>-11.31</td><td>-11.03</td><td>-11.27</td><td>-10.96</td><td>-11.00</td><td>-10.98</td><td>-10.84</td><td>-10.98</td><td>-10.85</td></tr></table>

Table 4 compares the unregularized model, global trace regularization, and the best gated variant across the evaluated data fractions. For each dataset and fraction, the gated variant is selected using validation log-likelihood. The results show that global regularization can improve performance for some datasets, particularly when data are limited, but frequently degrades likelihood as the available data increase. The gated variants more consistently preserve the fit while retaining gains on datasets where sharpness regularization remains beneficial.

Algorithm 2: Gated Marginals EM for Probabilistic Circuits   
Require: Probabilistic circuit PC with sum-node parameters $P = \{ \theta _ { n c } \} _ { ( n , c ) \in E } ;$ dataset $D = \{ x ^ { ( i ) } \} _ { i = 1 } ^ { N } ;$ regularization weight   
$\mu \geq 0 ;$ simplex constraint weight $\lambda > 0 ;$ smoothing factor $\alpha \in ( 0 , 1 ]$ ; monotone gate function $g : \mathbb { R } _ { \geq 0 }  [ 0 , 1 ] ;$ ; number of   
epochs E   
Ensure: Updated sum-node parameters P minimising the gated marginals regularised log-likelihood   
1: Initialise: for each sum-node n, set $\theta _ { n } .$ uniformly on its simplex   
2: Set $\omega _ { n } \gets g ( 0 )$ for all sum-nodes n // gate weights; recomputed after first epoch   
3: for epoch $= \dot { 1 } , \dots , E$ do // repeat until convergence or max epochs   
E-step: Compute expected edge flows and node marginals   
4: Run forward–backward passes on PC over D   
5: Obtain edge flows $\{ F _ { n c } \tilde { ( } x ) \} _ { ( n , c ) \in E , x \in D }$ $/ / O ( | P | | D | )$ time   
6: Obtain node marginals $\{ p _ { n } ( x ) , p _ { c } ( x ) \} _ { n \in V , x \in D }$ // available from the forward pass at no extra cost   
Local Trace Computation (marginal ratio):   
7: for all sum-nodes n do   
8: Aggregate flows: $\bar { F } _ { n c } \gets \sum F _ { n c } ( x )$   
$\overline { { x } } \in \overline { { D } }$   
9: Compute dataset-averaged marginal ratio for each child c:   
$\bar { r } _ { n c }  \frac { 1 } { | D | } \sum _ { x \in D } \frac { p _ { c } ( x ) } { p _ { n } ( x ) }$   
10: Compute local trace via marginals:   
$\hat { t } _ { n } \gets \sum _ { c \in \mathrm { c h } ( n ) } \bar { r } _ { n c } ^ { 2 } = \sum _ { c \in \mathrm { c h } ( n ) } \left( \frac { 1 } { | D | } \sum _ { x \in D } \frac { p _ { c } ( x ) } { p _ { n } ( x ) } \right) ^ { 2 }$   
// depth-agnostic: no $F _ { n }$ factor; no $\theta _ { n c }$ dependence   
11: end for   
Gate Weight Update:   
12: for all sum-nodes n do   
13: $\omega _ { n } \gets g \big ( \hat { t } _ { n } \big )$ $\begin{array} { r } { \operatorname { \mathrm { / / e . g . ~ } } g ( \hat { t } _ { n } ) = \hat { t } _ { n } / \operatorname* { m a x } _ { n ^ { \prime } } \hat { t } _ { n ^ { \prime } } } \end{array}$ (Proposed);   
14: end for   
M-step: Gated sharpness-aware parameter update   
15: for all sum-nodes n do // updates are independent per nod   
16: for all child edges $( n \to c )$ do   
17: $\tilde { \theta } _ { n c }  \frac { \bar { F } _ { n c } + \sqrt { \bar { F } _ { n c } ^ { 2 } + 4 \lambda ( \mu \omega _ { n } ) \bar { F } _ { n c } } } { \mathrm { ~ o ~ x ~ } }$ // same closed form as global trace; effective strength $\mu _ { n } = \mu \omega _ { n }$   
2 λ   
18: end for   
19: Normalise: $\tilde { \theta } _ { n c } \gets \tilde { \theta } _ { n c } / \sum \tilde { \theta } _ { n c ^ { \prime } } \quad \forall c \in \mathrm { c h } ( n )$ // project onto probability simplex   
c<sup>′</sup>∈ch(n)   
20: end for   
21: $\theta  ( 1 - \alpha ) \theta + \alpha \tilde { \theta }$ // running-average smoothing for noisy mini-batch flows   
22: end for   
23: return $P \equiv \theta$

![](images/ce0c2a0e53db9c24c5d19d7e71b56f06586d3651f602aec486538616f4199c57.jpg)  
(a) Contextual usage Fb<sup>2</sup><sub>n</sub>

![](images/9663e0f8abfbb2fd25c67ecdb623b08190d0fb8729b95558b1780b4c6deeae61.jpg)

![](images/47b911338e16399b361a93bc786d4b7b2d85fa257e5cef30a418e8a41591de28.jpg)

![](images/1eddbb9d977a97d7d96f05697fe344f4f34f119d2dcc52b4bbe26b75a036b6d3.jpg)  
(c) Global contribution $\widehat { T } _ { n }$

$$
\widehat { T } _ { n } = \widehat { F } _ { n } ^ { 2 } \widehat { t } _ { n }
$$

Figure 9: The curvature of a trained probabilistic circuit factorizes into usage and local sharpness. Each sum node contributes $T _ { n } = F _ { n } ^ { 2 } t _ { n }$ to the NLL Hessian trace. Panels (a)-(c) show the same circuit, with identical node coordinates, colored by (a) squared flow ${ \widehat F } _ { n } ^ { 2 } .$ , (b) local trace $\widehat { t } _ { n } .$ , and (c) global contribution ${ \widehat { T } } _ { n } ;$ ; panel (d) plots log $\widehat { F } _ { n } ^ { 2 }$ against log $\widehat { t } _ { n }$ colored by log $\widehat { T } _ { n }$ , with dashed iso-contribution lines of constant $\widehat { T } _ { n }$ . The two factors are anticorrelated across the circuit: the locally sharpest node (dark in b) sit at low flow (light in a) and contribute little globally, while the high-usage root region drives the global trace despite modest local curvature. Nodes A, B, and C mark the three regimes: high-usage/low-local (A), low-usage/high-local (B), and high on both (C).

![](images/bbc4aa362620b736e7fdf2f5a54fc9c24cafae353d5ba073d9522c75abc00bbb.jpg)  
Figure 10: Vanilla, global, and adaptive regularization settle in distinct regions of parameter space. Vanilla reaches the lowest train NLL but a sharp, high-curvature region that generalizes poorly. Global regularization flattens curvature indiscriminately and underfits, while adaptive gating flattens only the non-generalizing sharpness, reaching a region with competitive test NLL at markedly lower sharpness.

![](images/d84136559b456ad029b255bccbb5a75794b5f71e2dbb2e7db772483637d69abc.jpg)  
Figure 11: (Underfitting at full data regime)Training and test log-likelihood trajectories for the vanilla and global trace-regularized models DEBD using the full training data. Global trace regularization reduces the train-test gap but lowers both training and test likelihood, indicating underfitting rather than improved generalization.

![](images/eca450e2744684e8a0a32e6bbe6b2fb71d8e7311490a0d19ab451c17e74bef11.jpg)  
Figure 12: (Sparsity of Global trace) Cumulative share of the total trace as sum nodes are ranked by theirempirical contribution $( \widehat { T } _ { n } )$ . Across datasets, the distribution is highly concentrated: a small set of nodes dominates the global sharpness signal, while most nodes contribute negligibly.

|t<sub>n</sub>| Pareto chart all datasets (100% data)  
![](images/129d029244d6e318bd5a98fa54ee7fb9f899a79cf31e64315eaed587e26d8ff6.jpg)  
Figure 13: (Sparsity of local trace) Cumulative share of $\textstyle \sum _ { n } { \widehat { t } } _ { n }$ as nodes are ranked by $\widehat { t } _ { n } .$ . On the evaluated circuits, the local trace concentration is less severe than the global contribution, indicating that part of the concentration in $\widehat { T } _ { n }$ is introduced by contextual usage.

![](images/214e19ec9380354857c86a6deec236f48f42ba1c361e9c916d3090b37705e11c.jpg)  
Figure 14: (Thresholding based node selection results) Change in test log-likelihood relative to the unregularized model when trace regularization is applied to all sum nodes or restricted to progressively smaller fractions ranked by $\widehat { T } _ { n }$ . Selecting fewer top-ranked nodes generally worsens performance, showing that global contribution is a poor criterion for allocating regularization.

![](images/7f277237f7b011ce5d1fecdb88eae92f66c716f071cd22af30782cf098b46330.jpg)

![](images/260ed580f63876e19c32d00b75bc7bc5590e4e5c1b8736932f7e4b1e99096dd5.jpg)

![](images/82c4c9881673a899fcf7268b317856abe9e76162173b75e1bb7c213d65032c91.jpg)

![](images/ee4170da41db4d226f121bbdf2f47cf09815eb00e47bf859f344b7fcf3d1ddfd.jpg)

![](images/a095e92341e1e6db962938a27c844ff909b8c24c0bc09aaeb4a20c86b3c5eb09.jpg)  
Figure 15: (Thresholding based node selection results) Change in test log-likelihood relative to the unregularized model when trace regularization is applied to all sum nodes or restricted to progressively smaller fractions ranked by $\widetilde { t _ { n } } .$ . Selecting fewer top ranked nodes to regularize generally improves the performance, showing that $\widehat { t } _ { n }$ is a better criterion for allocating regularization.
# LipCache: A Local Inference Proxy with Certified Caching for Edge Image Classification Service

Zhengzhe Xiang, Yinlin Chen, Fuli Ying, Binbin Zhou, Hailiang Zhao Member, IEEE, Schahram Dustdar Fellow, IEEE

Abstract—As edge-side vision services continue to expand toward low-latency, high-throughput scenarios, reducing the inference cost of vision models without sacrificing reliability has become a central concern. Existing semantic caching methods largely rely on empirical similarity thresholds; while such thresholds improve hit rates, they tend to introduce silent misclassifications near decision boundaries. To address this, we propose LipCache, a certified semantic caching framework for image classification. Without modifying the existing deployed main model, MainNet, the framework introduces a lightweight network, GuardNet, that maps inputs into a low-dimensional feature space subject to a Lipschitz constraint. It then computes a per-sample certified reuse radius from the local classification margin and the spectral norm of the classification head. At runtime, a cached result is reused only when the query feature falls inside the certified reuse ball; otherwise, the query falls back to MainNet. Thus, cache hits are transformed from empirica threshold tests into geometric certification decisions with explicit theoretical boundaries. Across standard image classification tasks like CIFAR, Tiny-ImageNet, and SVHN, LipCache achieves a measured speedup of up to 1.65× with limited end-to-end accuracy degradation, while all accepted cache hits satisfy the GuardNet-side certified-consistency condition. Furthermore, an enhanced GuardNet training recipe substantially improves cache hit rates in the Tiny-ImageNet multi-class extension while maintaining a certified-consistency rate of 100%. These results demonstrate that per-sample certified reuse can reduce main-model fallback while preserving theoretical consistency, providing a feasible approach to reliable cache-assisted inference at the edge.

Index Terms—Edge intelligence, image classification, semantic caching

## 1 INTRODUCTION

D <sup>EEP-neural-network-based</sup> <sup>vision</sup> <sup>services</sup> <sup>have</sup> <sup>been</sup>widely deployed across diverse domains such as widely deployed across diverse domains such as autonomous-driving perception, intelligent manufacturing inspection, real-time surveillance, and mobile augmented reality. The dual demand of these services for inference accuracy and response speed has made high-accuracy neural models a de facto standard component. However, the parameter count and computational complexity of such models keep growing, so that a single inference already consumes considerable compute and memory resources.

This tension is further amplified in industrial edge scenarios. Unlike occasional photo recognition on a phone, industrial edge devices—such as production-line cameras and roadside perception units—typically operate continuously at high sampling rates, injecting a large volume of concurrent queries into the inference service within very short intervals. When these queries converge at the cloud, they readily trigger spikes in queries-per-second, network congestion, and server overload, severely threatening the overall service availability and tail latency. In other words, the root cause of the problem lies not only in the high cost of a single inference, but more critically in the fact that a large number of semantically redundant queries are repeatedly executed without discrimination.

The rise of the edge computing paradigm offers a structural opportunity to mitigate this risk. By pushing part of the computation or storage capability to edge nodes close to the data source, the system can potentially absorb a fraction of the query load locally, thereby reducing its dependence on the remote cloud link. New challenges arise, however: the latency and memory budgets of edge hardware are extremely tight and far from sufficient to host a full-sized model comparable to its cloud counterpart [1], [2], [3], [4]. Directly deploying a high-accuracy classifier on an edge device almost inevitably faces a sharp conflict between accuracy and resources.

To address this tension, existing work proceeds along roughly two lines. The first line aims to reduce the execution cost of the model at the edge, including compact architecture design, model compression, and early-exit strategies [5], [6], [7]. These methods substantially improve edge deployability, yet their unit of optimization remains “how to run the model more cheaply”; they do not touch upon a more fundamental question: for the many semantically repetitive queries, is it really necessary to run the model every time?

The cache-assisted inference starts precisely from this gap. It exploits the pervasive redundancy of visual patterns in edge workloads and reuses previously computed results to skip part of the model invocations directly [8], [9], [10], [11], [12]. However, current reuse strategies suffer from a structural weakness in reliability: exact matching is extremely sensitive to compression, cropping, and illumination changes, and can hardly enable effective reuse in real deployments; semantic caching, although it enlarges the reuse region, mostly relies on heuristic similarity thresholds and provides no formal guarantee on the correctness of the reused labels. Consequently, cache hits near decision boundaries may introduce silent misclassifications and uncontrollable accuracy loss. A caching strategy that combines an explainable safety boundary with practical reuse efficiency is still missing.

This paper proposes LipCache, a certified semantic caching framework that comprehensively exploits the storage and computation capability of edge devices. Its core design principle is modularity: rather than modifying the deployed high-accuracy main model (MainNet), it introduces a lightweight guard network, GuardNet, as a new branch to construct the semantic cache. Through operatornorm-controlled convolutions and spectral normalization, GuardNet maps inputs to a smooth low-dimensional feature space [13], [14], and, starting from the local classification margin and the Lipschitz bound of the classification head, computes a per-sample certified reuse radius offline for each cached sample. Online, only when the query feature falls inside the certified ball of some cached sample, namely when their distance is less than the certified reuse radius, does the system reuse the stored label; otherwise it falls back to MainNet to preserve accuracy. Thereby, a cache hit is converted from an empirical threshold decision into a geometric certification decision with an explicit theoretical boundary.

We validate the core certification mechanism of LipCache on benchmarks with markedly different statistical structure: CIFAR (natural images), Tiny-ImageNet (higher-resolution fine-grained inter-class differences), and SVHN (domain-shifted digits and characters). This choice is designed to isolate the stability of the certified radius under different task geometries, rather than to exhaust all visual task scales; larger-scale settings (e.g., full ImageNet) and non-i.i.d. streaming scenarios are left as open directions discussed in Section 7. On these three tasks, the average hit rates under the tight certified radius reach 0.342, 0.472, and 0.502, respectively, with end-to-end accuracies of 0.921, 0.867, and 0.960, and corresponding measured speedups of 1.32×, 1.31×, and 1.65×; moreover, the certified-consistency rate of all results is 100%. Comparisons with empirical thresholds and the global nearestneighbor baseline show that the certified radius is the only reuse boundary that can stably preserve certified consistency across the three tasks. Furthermore, in the Tiny-ImageNet multi-class extension, an enhanced GuardNet training recipe improves the hit rates of the 20/30/50-class tasks from 0.056/0.005/0.0004 to 0.423/0.254/0.124, while both the end-to-end accuracy and the certified consistency are preserved. The main contributions are as follows:

1) We propose a dual-model cache-assisted inference framework that comprehensively exploits the storage and computation capability of edge devices and, without modifying MainNet, makes certified reuse decisions through a lightweight GuardNet.

2) Starting from the local classification margin and the Lipschitz bound of the GuardNet classification head, we derive a per-sample certified reuse radius in the feature space, providing a computable and interpretable rule for safe semantic reuse.

3) We systematically evaluate LipCache across multiple benchmark tasks, multi-class extensions, and edgedeployment boundaries, verifying the effectiveness of the certified radius as the only cross-task stable certified boundary and revealing the potential of the enhanced training recipe in enlarging the certified reuse region.

## 2 RELATED WORK

Edge image classification faces a structural tension: highaccuracy neural models keep growing in their computational and memory demands, whereas edge devices must satisfy tight latency, energy, storage, and connectivity constraints. A practical edge system therefore requires more than just a smaller classifier. It must reduce computation, exploit redundancy among correlated inputs, and decide when a low-cost shortcut is safe rather than merely fast. The literature addresses these needs along three main lines and one important bridge. Efficient edge inference reduces the cost of running a model through compact architectures, adaptive execution, hardware–software co-design, or edge– cloud partitioning. Cache-assisted inference and semantic reuse avoid part of the execution by reusing results from temporally or semantically correlated inputs. Certified nearest-neighbor retrieval forms a bridge, studying when a nearest-neighbor decision in an embedding space remains invariant under perturbation. Lipschitz control and certification provide the tools that turn margins and continuity bounds into local stability guarantees. LipCache connects these lines: it uses a lightweight GuardNet as a cacheoriented front end to the high-accuracy MainNet, and accepts reuse only inside a per-sample certified reuse radius in the GuardNet feature space.

## 2.1 Inference for Edge Vision Systems

The dominant strategy for edge inference is to reduce the cost per invocation, or to place computation where it can be executed more efficiently. Survey work points out the underlying pressure: the scale and computational complexity of DNNs grow faster than the performance and efficiency gains that embedded platforms can provide [1]. Recent surveys organize the response around efficient architectures, model compression, hardware acceleration, algorithm–hardware co-design, and deployment techniques for on-device or edge–cloud execution [2], [3], [4], [15]. Edge–cloud collaboration further partitions a DNN so that shallow feature extraction runs on the device while deeper layers run on the cloud or an edge server, additionally attending to privacy, communication overhead, and resource allocation [16], [17].

Representative architecture-level methods reduce the cost of the backbone itself. MobileNetV2 uses inverted residual blocks, thin linear bottlenecks, and depthwise separable convolutions to improve mobile classification, detection, and segmentation across model sizes [5]. MobileNetV3 combines hardware-aware neural architecture search with NetAdapt and manual architectural refinements, yielding both large and small mobile models tuned for phone CPUs [18]. ShuffleNet targets extremely low computational budgets by combining pointwise grouped convolutions with channel shuffling, lowering cost while preserving mobile accuracy [6]. Representative execution-level methods adaptively reduce the average cost: BranchyNet adds side-branch classifiers so that high-confidence samples can exit early, while harder samples continue through deeper layers [7]; MSDNet treats test-time computation as an anytime or budgeted classification problem and reuses multi-scale dense features across multiple early exits [19]. At the collaborative-system level, MAE reduces intermediate-transmission and scheduling costs by activating sparse channel-level experts during DNN inference on edge devices [17], and more recent edgecloud co-inference combines spike-driven compression with dynamic early-exit to cut both transmission volume and end-to-end latency [20].

These works substantially improve deployability, yet their unit of optimization remains executing the current query—through the model, an early-exit path, or a device– cloud partition. They do not explicitly exploit a per-sample certified reuse region around cached samples to decide whether the current query can entirely skip the main classifier. LipCache is therefore complementary: it keeps MainNet unchanged as a reliable fallback predictor and adds an independent GuardNet-guided cache decision layer to avoid selected MainNet invocations.

## 2.2 Cache-Assisted Neural Network Inference

Cache-assisted inference follows a different strategy: rather than reducing the cost per invocation, it tries to avoid invocations by reusing outputs or intermediate states of correlated inputs. Exact-match memoization is too brittle for vision tasks, since repeated content may differ due to camera motion, illumination, scale, cropping, or semantic changes. Practical systems therefore extend reuse beyond equality by exploiting temporal locality, video-frame coherence, low-cost filters, intermediate feature maps, or semantic memory.

Shadow Puppets proposes a semantic caching service that breaks the binary choice between pure cloud inference and full edge execution, leveraging edge-side caching to approximate cloud-level accuracy at lower latency and cost [21]. DeepCache targets continuous mobile vision: it uses heuristics inspired by video compression to detect reusable regions in the video input, propagates these reusable regions through the layers of the CNN, works with unmodified models, and reports reductions in inference time and energy [10]. NoScope specializes video analytics to a specific video and object, searching over a cascade of cheap specialized models and difference detectors to mimic a reference network at a chosen accuracy level [9]. DeltaCNN likewise exploits video coherence but at the implementation level: it propagates sparse inter-frame differences through typical CNN layers and accelerates per-frame inference without retraining [12]. Semantic Memory turns reuse toward semantic features, encoding high-dimensional feature maps into low-dimensional semantic vectors, and uses hierarchical memory and adaptive cache management to accelerate mobile CNN inference with acceptable accuracy loss [11]. More recent work continues to expand cache-assisted inference along several axes: synergistic lazy loading and layer-wise caching reduce the cold-start and loading cost of serverless edge models [22]; task-aware expert caching decomposes multitask models into modular experts for finegrained reuse [23]; and distributed inference caching has been formalized as a submodular maximization problem under storage constraints [24]. The closest line on certified retrieval is RetrievalGuard, which studies provably robust 1-nearest-neighbor image retrieval. Given a base retrieval model and a query, RetrievalGuard constructs a smoothed retrieval model, analyzes the 1-NN search process in the high-dimensional embedding space, and derives a computable $\ell _ { 2 }$ radius within which the Recall@1 result remains unchanged [25].

These systems demonstrate that reuse can be a powerful source of acceleration, yet their acceptance criteria are mainly empirical or workload-specific. Video-based methods rely on temporal continuity or sparse inter-frame differences; specialized cascades rely on a fixed camera/query distribution; semantic memory relies on learned or measured feature similarity and exit policies; certified retrieval certifies the stability of the nearest-neighbor result itself, rather than the safe reuse of a cached payload in an edge inference pipeline. LipCache retains the idea of reuse but changes the acceptance criterion: the cache is built in a dedicated GuardNet feature space, and a reuse is accepted only when the query falls inside the per-sample certified reuse radius of some cached sample, a radius derived from the GuardNet-side margin and Lipschitz bounds.

## 2.3 Lipschitz Control and Certification

Lipschitz control and certification address the safety problem of efficient inference. Their common strategy is to bound how fast a neural mapping can change, and then use this bound, together with margins or probabilistic arguments, to derive a region within which the prediction remains stable. This body of literature therefore provides the mathematical tools for turning similarity into local consistency statements.

At the layer and architecture level, spectral normalization constrains the operator norm of the weights through a lightweight normalization procedure, originally introduced to stabilize GAN discriminators [14]. For convolutional layers, Sedghi et al. characterize the singular values of standard multi-channel convolutions, enabling efficient computation and projection onto operator-norm balls [13]. Araujo et al. provide a unified algebraic view of 1-Lipschitz neural networks, showing that several layers based on orthogonality, spectra, and convex potentials can be understood through a common semidefinite-programming condition [26]. These works explain how to build or constrain networks whose continuity can be controlled.

At the certification level, CROWN bounds general activation functions with linear or quadratic surrogates to produce certified lower bounds on adversarial distortion [27]; randomized smoothing, in turn, turns a classifier that remains accurate under Gaussian noise into an $L _ { 2 ^ { - } }$ certified smoothed classifier, extending certification to fullresolution ImageNet [28]. More recent work tightens or extends Lipschitz-based certification. Huang et al. exploit activation states to remove inactive rows and columns from the induced-norm computation, thereby computing efficient local Lipschitz upper bounds [29]. Fazlyab et al. use directional Lipschitz bounds and improved regularization to maximize a differentiable lower bound on the distance to the decision boundary [30]. Hu et al. show that achieving certifiable robustness under a Lipschitz constraint requires careful capacity and data design, including larger Lipschitz-controlled residual dense layers and data augmentation [31]. Wang et al. improve scalability by reformulating semidefinite-programming Lipschitz estimation as an eigenvalue optimization problem [32]; and ECLipsE decomposes the large verification problem into layer-scale subproblems or closed-form relaxations to obtain faster compositional estimates [33].

From the perspective of cache-assisted edge inference, the common limitation of these methods is that they certify model robustness or stability, rather than cache reuse. Their certificates typically answer whether a classifier’s prediction remains invariant under perturbation; they do not decide whether a cached payload should replace an expensive MainNet invocation. LipCache repurposes the same mathematical ingredients for a system-level reuse decision: the GuardNet margin and the GuardNet-side Lipschitz control define a per-sample certified reuse radius in the GuardNet feature space. The resulting guarantee is local to GuardNet, whereas consistency with MainNet is treated as an empirical system property.

In summary, existing work leaves three gaps with respect to reliable edge caching. Edge-inference methods reduce the cost of running a model but do not address whether the model can be skipped. Semantic caching and video reuse exploit redundancy but typically rely on empirical thresholds or workload locality. Certified-retrieval and robustness methods provide stability guarantees, but they do not formulate the cache-hit acceptance criterion as a GuardNet-side reuse certificate coupled with MainNet fallback. LipCache fills exactly this intersection by combining a dedicated GuardNet feature space, a per-sample certified reuse radius for each cached sample, and an empirical evaluation of the consistency between the accepted cache hits and MainNet.

## 3 PROBLEM DESCRIPTION

## 3.1 System Setting and Cache Motivation

This section provides a formal model of cache-assisted inference in an edge image classification system, establishing a unified notational foundation for the framework design in Section 4 and the certification analysis in Section 5.

Consider an image classification system whose input space is $\mathcal X \subseteq \mathbb { R } ^ { n }$ and whose label set is $\mathbf { \dot { \mathcal { V } } } = \{ 1 , \dots , C \}$ . The system may invoke a high-accuracy classifier $M : \mathcal { X }  \mathcal { Y } _ { \ }$ but a single inference of M is costly—in edge scenarios this may amount to a combination of GPU time, cloud communication latency, and other overhead. For any input $x \in \mathcal { X } ,$ the system ultimately needs to output a prediction $\hat { y } \in \mathcal { V }$

A key observation is that edge vision workloads are not composed of mutually independent random samples. The input–label pairs $( x , y )$ are drawn from a joint distribution D that often exhibits pronounced structural redundancy: influenced by factors such as a fixed camera viewpoint, periodic sampling, and slow scene changes, consecutively arriving queries are highly clustered in pixel space or semantic space, and adjacent inputs frequently share the same label. Under these conditions, executing M in full for every query means that a large amount of computation is wasted on samples whose answers could have been inferred from existing results. This observation points to a natural optimization path: cache a subset of historical predictions, and when a new query is sufficiently similar to some cached sample, directly reuse its label, thereby skipping the invocation of M.

However, the above path faces a fundamental tension. If the reuse condition is too loose, queries near a decision boundary may be incorrectly matched to an inappropriate cached label, leading to silent accuracy loss; if the reuse condition is too strict, cache hits will be so rare that the savings in M invocations become negligible. The core challenge is therefore not “whether to introduce a cache,” but how to define a reuse rule that simultaneously achieves a sufficient hit rate to yield meaningful system gains and constrains the resulting prediction errors to a controllable range.

## 3.2 Formalization of the Optimization Objective

To precisely characterize the above trade-off, we abstract cache-assisted inference as an optimization problem over deployment policies.

A deployment policy π produces two outputs for each input $x \in { \mathcal { X } } ;$ a predicted label $\hat { y } ( x ) \in \mathcal { V }$ , and a binary routing decision $\rho ( x ) ~ \in$ {cache, invoke}, the latter indicating whether the prediction comes from cache reuse or from a direct inference of M.

Let $( x , y ) \sim \mathcal { D }$ denote a test sample drawn from the data distribution. The system accuracy of policy π is defined as

$$
\operatorname { A c c } _ { \mathrm { s y s } } ( \pi ) \triangleq \operatorname* { P r } _ { ( x , y ) \sim \mathcal { D } } [ \hat { y } ( x ) = y ] ,\tag{1}
$$

and the standalone accuracy of M is $\begin{array} { l l } { \operatorname { A c c } _ { M } } & { \triangleq } \end{array}$ $\mathrm { P r } _ { ( x , y ) \sim \mathcal { D } } [ M ( x ) = y ]$ . In practice, the above distributional expectations are approximated by empirical frequencies on a reference validation or test set.

The invocation rate of policy π on M is defined as

$$
\mathrm { I n v o k e R a t e } _ { M } ( \pi ) \triangleq \operatorname* { P r } _ { ( x , y ) \sim \mathcal D } [ \rho ( x ) = \mathrm { i n v o k e } ] .\tag{2}
$$

This quantity directly determines the degree of cost savings on M inference—the lower the invocation rate, the higher the fraction of M invocations that are skipped.

Cache-assisted inference can thus be formalized as the following constrained optimization problem:

$$
\begin{array} { r l } { \underset { \pi } { \operatorname* { m i n } } } & { \mathrm { I n v o k e R a t e } \boldsymbol { M } ( \pi ) } \\ { \mathrm { s . t . } } & { \mathrm { A c c } _ { \mathrm { s y s } } ( \pi ) \geq \mathrm { A c c } _ { M } - \Delta _ { \mathrm { a c c } } , } \end{array}\tag{3}
$$

where $\Delta _ { \mathrm { a c c } } \geq 0$ is a user-specified accuracy tolerance. A smaller $\Delta _ { \mathrm { a c c } }$ enforces a more conservative reuse strategy,

tending to depress the hit rate to reduce accuracy risk; a larger $\Delta _ { \mathrm { a c c } }$ allows more aggressive cache reuse, trading potential accuracy for a lower M invocation frequency.

## 4 THE LIPCACHE

Section 3 formalizes cache-assisted inference as a deployment-level optimization problem that minimizes the main-model invocation rate under an accuracy-tolerance constraint. This section gives a concrete instantiation of that problem—the LipCache framework. Its workflow proceeds in three stages: first, a Lipschitz-constrained lightweight guard network, GuardNet, is trained offline (Section 4.2); next, a feature cache is built on the trained GuardNet, and each cached sample is assigned a per-sample certified reuse radius (Section $4 . 3 ) ;$ finally, in the online phase, GuardNet guides cache-hit detection and the fallback decision (Section 4.4). We first establish the unified notation.

## 4.1 Setup and Notation

LipCache adopts a dual-model design, comprising a highaccuracy main model M and a lightweight guard model G. The main model provides the reference prediction:

$$
\hat { y } _ { M } ( x ) = \arg \operatorname* { m a x } _ { c \in \mathcal { V } } M _ { c } ( x ) .\tag{4}
$$

The guard model maps each input to a low-dimensional feature space, $G : \mathcal { X } \overset { \cdot } {  } \mathbb { R } ^ { d }$ . Let $z = G ( x ) ;$ ; an affine classifier operates in this space:

$$
s ( z ) = W z + b ,\tag{5}
$$

where $W \in \mathbb { R } ^ { C \times d } , b \in \mathbb { R } ^ { C }$ , yielding the GuardNet prediction

$$
{ \hat { y } } _ { G } ( x ) = \arg \operatorname* { m a x } _ { c \in \mathcal { V } } s _ { c } ( G ( x ) ) .\tag{6}
$$

We denote the full logit map by $F _ { G } ( x ) = s ( G ( x ) )$ and the affine head by $f ( z ) = W z \dot { + } b .$ When two logit vectors are subtracted, the bias term cancels, so it does not affect the Lipschitz constant.

For each cached sample, the certified reuse radius is derived from the local classification margin and the Lipschitz bound of f (Section 5). At runtime, a query that falls inside the certified reuse region of some cached sample reuses the stored payload (the label returned upon a cache hit); otherwise the system falls back to M. Table 1 summarizes the notation used.

Each cache entry is a triple $( z _ { i } , \tilde { y } _ { i } , r _ { i } )$ , containing a GuardNet feature, a payload label, and a certified reuse radius. Figure 1 shows the offline training and featureconstraint pipeline, and Figure 2 shows the online inference.

## 4.2 Offline GuardNet Training

With the notation established, this subsection answers the first core question: how to train a GuardNet so that its learned feature geometry both supports discriminative classification and provides a sufficient certified reuse region for subsequent certified caching. To this end, the training objective is designed as a layered framework:

$$
L _ { \mathrm { t o t a l } } = ( 1 - \alpha ) L _ { \mathrm { C E } } + \alpha L _ { \mathrm { d i s t i l l } } + \lambda _ { m } L _ { \mathrm { m a r g i n } } ,\tag{7}
$$

Main symbols used in the formulation.  
TABLE 1
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\overleftarrow { \mathcal { X } , \mathcal { Y } }$ </td><td>Input space and label set</td></tr><tr><td> $M$ </td><td>High-accuracy main model (MainNet)</td></tr><tr><td> $G$ </td><td>Feature map of the guard model (GuardNet)</td></tr><tr><td> $W$ </td><td>Weight matrix of the GuardNet affine classifier</td></tr><tr><td> $^ d$ </td><td>Dimensionality of the GuardNet feature space</td></tr><tr><td> $L _ { G }$ </td><td>Lipschitz constant of the GuardNet feature map G</td></tr><tr><td> $\sigma _ { \operatorname* { m a x } } ( W )$ </td><td>Spectral norm of W</td></tr><tr><td> $z _ { i }$   $y _ { i } ^ { G }$ </td><td>GuardNet feature of cached sample  $x _ { i }$ </td></tr><tr><td></td><td>GuardNet prediction of xi</td></tr><tr><td> $\tilde { y } _ { i }$ </td><td>Cached payload associated with  $x _ { i }$ </td></tr><tr><td> $r _ { i }$ </td><td>Certified reuse radius of cached feature  $z _ { i }$ </td></tr><tr><td> $z _ { \mathrm { n e w } }$ </td><td>GuardNet feature of a new input</td></tr><tr><td>mi</td><td>GuardNet classification margin of  $x _ { i }$ </td></tr><tr><td> $c$ </td><td>Feature cache maintained by the system</td></tr></table>

where $\alpha ~ \in ~ [ 0 , 1 ]$ balances direct supervision against an optional distillation signal from $M ,$ and $\lambda _ { m }$ controls the margin-regularization strength. When $\alpha = 0 ,$ , the objective reduces to the pure cross-entropy training used in the 10- class main experiments of this paper; when $\alpha \ > \ 0 ,$ , it can be used for additional distillation ablations or teacheralignment experiments. The above objective subsumes three complementary training signals, which we elaborate on one by one below, starting with the structural constraint that supports them.

Lipschitz control. The theoretical certificate in Section 5 requires the feature map to satisfy $\operatorname { L i p } ( G ) ~ \leq ~ 1$ . To this end, GuardNet imposes an operator-norm constraint on all parameterized layers. For a convolutional layer whose kernel is $K \in \mathbb { R } ^ { c _ { \mathrm { o u t } } \times c _ { \mathrm { i n } } \times k _ { 1 } \times k _ { 2 } }$ , the Lipschitz constant of the convolution operator is given by its frequency-domain representation:

$$
\operatorname { L i p } ( K ) = \operatorname* { m a x } _ { \omega } \sigma _ { \operatorname* { m a x } } \bigl ( \hat { K } ( \omega ) \bigr ) ,\tag{8}
$$

where $\hat { K } ( \omega )$ is the frequency-domain matrix obtained by zero-padding the kernel and applying a two-dimensional real FFT. Dividing the kernel by $\operatorname* { m a x } ( \operatorname { L i p } ( K ) , 1 )$ before each forward pass enforces the 1-Lipschitz constraint [13]. Linear layers use standard spectral normalization [14]; ReLU, nonoverlapping average pooling, and the final tanh projection are all non-expansive [26]. Since the Lipschitz constant of each individual layer does not exceed 1, by submultiplicativity of operator norms under composition, the overall guard network satisfies $\operatorname { L i p } ( G ) \leq 1$

Distillation loss (optional). To optionally leverage the knowledge of $M ,$ one may minimize the KL divergence between the softened outputs of M and $F _ { G }$ at temperature $T \colon$

$$
\begin{array} { r } { L _ { \mathrm { d i s t i l l } } = T ^ { 2 } \mathrm { K L } \big ( \mathrm { S o f t m a x } \big ( \frac { M ( x ) } { T } \big ) \big \| \mathrm { S o f t m a x } \big ( \frac { F _ { G } ( x ) } { T } \big ) \big ) . } \end{array}\tag{9}
$$

The standard cross-entropy $L _ { \mathrm { C E } }$ with the ground-truth label y preserves discriminative performance; the main experiments in this paper use $\alpha = 0 , \mathrm { i . e . }$ , this term is disabled.

Margin regularization. The two terms above guarantee the discriminative accuracy of GuardNet, whereas margin regularization directly serves cache efficiency: the certified reuse radius $r _ { i }$ is proportional to the GuardNet classification margin (see Section 5), so explicitly encouraging large margins is a key lever for improving the hit rate. Let $c _ { 1 }$ and $c _ { 2 }$ denote the indices of the top-1 and top-2 logits:

![](images/38d09a9518f453074efba6d94493c873ca90849e3349f32c6c56b2c9eb6ad813.jpg)  
Fig. 1. Offline training pipeline of GuardNet. Lipschitz control on the convolutions and spectral normalization on the linear layers are used to constrain the GuardNet feature map, thereby supporting the subsequent GuardNet-side certification; the main model may provide reference predictions when needed, but it is not a required training signal in the three 10-class main experiments of this paper.

$$
m ( x ) = [ F _ { G } ( x ) ] _ { c _ { 1 } } - [ F _ { G } ( x ) ] _ { c _ { 2 } } .\tag{10}
$$

To avoid blindly enlarging the margin on already misclassified samples, the regularizer is applied only to the correctly classified subset of a batch B:

$$
S _ { \mathrm { c o r r } } = \left\{ ( x , y ) \in \mathcal { B } \mid \arg \operatorname* { m a x } _ { c \in \mathcal { V } } [ F _ { G } ( x ) ] _ { c } = y \right\} ,\tag{11}
$$

$$
L _ { \mathrm { m a r g i n } } = - \frac { 1 } { \mathrm { m a x } ( 1 , | S _ { \mathrm { c o r r } } | ) } \sum _ { ( x , y ) \in S _ { \mathrm { c o r r } } } m ( x ) .\tag{12}
$$

This selectively enlarges the margins around samples that are already correctly predicted, and these margins directly determine the certified cache radius.

Enhanced training recipe (optional). For multi-class extensions at higher class counts, cross-entropy alone or a single margin term is often insufficient to sustain a usable certified radius. We therefore additionally consider an enhanced training recipe: beyond the core objective above, it further adds a center constraint that promotes intraclass compactness and an additional inter-class separation term that promotes inter-class separation, while optionally retaining the distillation signal. These additional terms only change the feature geometry learned by GuardNet; they do not change the online hit rule of the cache, nor do they change the certification criterion in Section 5.

In the main-text experiments, the three 10-class main results use cross-entropy supervision only $( \alpha = 0 , \lambda _ { m } = 0 ) ;$ the CIFAR-10 training-objective ablation examines the independent effects of the margin and center constraints, respectively; and the Tiny-ImageNet multi-class extension uses the full enhanced recipe comprising the margin, interclass, center, and distill terms.

## 4.3 Feature Cache Construction

Once GuardNet is trained, the framework enters the offline cache-construction phase. Its task is to select a set of highquality cache entries from the reference set $\mathcal { D } _ { \mathrm { r e f } } .$ , so that it can cover the queries in the test distribution as much as possible under a given capacity budget.

For each candidate sample $x _ { i } ,$ we first compute $z _ { i } =$ $G ( x _ { i } )$ , the GuardNet label $\begin{array} { r } { \dot { y } _ { i } ^ { G } = \arg \operatorname* { m a x } _ { c } [ F _ { G } ( x _ { i } ) ] _ { c } , } \end{array}$ and the payload $\tilde { y } _ { i }$ selected by the construction strategy. This paper considers three payload semantics. The first is the proxy label, which directly stores $\tilde { y } _ { i } = y _ { i } ^ { G } ;$ in this case the cached payload is naturally aligned with the certificate, and it is the default setting adopted in the main-text experiments. The second is the MainNet prediction scheme, which stores $\tilde { y } _ { i } = \hat { y } _ { M } ( x _ { i } )$ ; the third is the ground-truth scheme, which stores $\tilde { y } _ { i } = y _ { i }$ . The latter two are mainly used to analyze the effect of varying the payload source on system behavior.

The GuardNet margin $m _ { i } = [ F _ { G } ( x _ { i } ) ] _ { c _ { 1 } } - [ F _ { G } ( x _ { i } ) ] _ { c _ { 2 } }$ is used to compute the certified radius $r _ { i }$ . Depending on the radius mode $\rho ,$ this paper supports two choices: the conservative radius $r _ { i } = \hat { m } _ { i } \hat { / } ( \sigma _ { \operatorname* { m a x } } \hat { ( W ) } \sqrt { 2 } )$ (Lemma 5.1) or the tighter per-pair radius $r _ { i } ^ { * }$ (Remark 5.1). Candidates with non-positive radius are discarded outright.

The certificate is attached to $y _ { i } ^ { G }$ . When the cache-center label consistency $y _ { i } ^ { G } = \tilde { y } _ { i }$ holds, the returned payload is directly aligned with the certificate; otherwise the reliability of the payload is assessed empirically. To maximize the coverage under a fixed budget, the selection strategy balances radius size against feature-space diversity. Algorithm 1 summarizes this construction process.

Algorithm 1 Offline construction of the feature store.   
Require: Reference set ${ \mathcal { D } } _ { \mathrm { r e f } } ,$ models M, G, W, cache budget   
$B _ { C }$   
Require: Payload construction rule, payload source (proxy,   
MainNet M prediction, or label $y ) ,$ , cache selection   
strategy $s$   
Require: Radius mode $\rho ,$ optional cache-center label  
consistency flag   
Ensure: Feature cache $\mathcal { C }$   
1: Initialize candidate pool $\mathcal { P }  \emptyset$   
2: Compute $\sigma _ { \operatorname* { m a x } } ( W )$   
3: if $\sigma _ { \mathrm { m a x } } ( W ) \leq 0$ then   
4: Report that the GuardNet classification head is de  
generate and stop construction   
5: end if   
6: for each candidate sample $x _ { i } \in \mathcal { D } _ { \mathrm { r e f } }$ do   
7: Compute $z _ { i } = G ( x _ { i } )$ and GuardNet logits $F _ { G } ( x _ { i } )$   
8: Compute GuardNet label $y _ { i } ^ { G } .$ , payload $\tilde { y } _ { i } ,$ and margin   
m<sub>i</sub>   
9: Compute certified radius $r _ { i }$ using radius mode ρ   
10: if cache-center label consistency is required and $y _ { i } ^ { G } \neq$   
$\tilde { y } _ { i }$ then   
11: Skip $x _ { i }$   
12: else if $r _ { i } \le 0$ then   
13: Skip $x _ { i }$   
14: else   
15: Add candidate entry $( z _ { i } , \tilde { y } _ { i } , r _ { i } )$ to $\mathcal { P }$   
16: end if   
17: end for   
18: Use S to select at most $B _ { C }$ entries from P   
19: The selection process balances certified radius and fea  
ture diversity   
20: Store the selected entries in $\mathcal { C }$   
21: return C

## 4.4 Online Inference and Cache-Hit Detection

After offline training and cache construction are complete, LipCache enters the online inference state. At runtime (Figure 2), each input $x _ { \mathrm { n e w } }$ is encoded as $z _ { \mathrm { n e w } } = G ( x _ { \mathrm { n e w } } )$ and the hit set is

$$
\mathcal { H } ( z _ { \mathrm { n e w } } ) = \{ i \ | \ \| z _ { \mathrm { n e w } } - z _ { i } \| _ { 2 } < r _ { i } \} .\tag{13}
$$

If H $\neq \emptyset ,$ , it takes the payload from the nearest hit entry:

$$
i ^ { * } ( x _ { \mathrm { n e w } } ) = \arg \operatorname* { m i n } _ { i \in \mathcal { H } ( z _ { \mathrm { n e w } } ) } \| z _ { \mathrm { n e w } } - z _ { i } \| _ { 2 } .\tag{14}
$$

The system predicts

$$
\begin{array} { r } { \hat { y } _ { \mathrm { s y s } } ( x _ { \mathrm { n e w } } ) = \left\{ \begin{array} { l l } { \tilde { y } _ { i ^ { * } } ( x _ { \mathrm { n e w } } ) , } & { \mathcal { H } ( z _ { \mathrm { n e w } } ) \neq \emptyset , } \\ { \hat { y } _ { M } ( x _ { \mathrm { n e w } } ) , } & { \mathcal { H } ( z _ { \mathrm { n e w } } ) = \emptyset . } \end{array} \right. } \end{array}\tag{15}
$$

On a miss, the system falls back to $M ;$ if write-back is enabled, a missed sample may be added to the cache under the same construction strategy.

The computational overhead of the above pipeline directly determines the practical gain of the framework. Let $C _ { G }$ and $C _ { M }$ denote the per-query cost of GuardNet and MainNet, respectively. When maintaining N cache entries in a d-dimensional feature space, the expected cost per query is

Algorithm 2 Online inference with cache reuse.   
Require: Query batch $\{ x _ { k } \} _ { k = 1 } ^ { B } ,$ main model M, trained   
GuardNet encoder G and classification head $W ,$ feature   
cache $\mathcal { C } = \{ ( z _ { i } , \tilde { y } _ { i } , r _ { i } ) \} _ { i = 1 } ^ { N } ,$ optional write-back flag   
Ensure: Predictions $\{ \hat { y } _ { \mathrm { s y s } } ( x _ { k } ) \} _ { k = 1 } ^ { B }$   
1: Compute GuardNet features $\mathsf { \bar { z } } _ { k } \gets G ( { \boldsymbol { x } } _ { k } )$ for all k   
2: Compute pairwise distances $D _ { k , i } = \| z _ { k } - z _ { i } \| _ { 2 }$   
3: for $k = 1$ to B do   
4: Construct hit set $\mathcal { H } ( z _ { k } ) = \{ i \ | \ D _ { k , i } < r _ { i } \}$   
5: if $\mathcal { H } ( z _ { k } ) \neq \emptyset$ then   
6: $\begin{array} { r } { i _ { k } ^ { * } = \arg \operatorname* { m i n } _ { i \in \mathcal { H } ( z _ { k } ) } D _ { k , i } } \end{array}$   
7: $\hat { y } _ { \mathrm { s y s } } ( x _ { k } ) \gets \tilde { y } _ { i _ { k } ^ { * } }$   
8: else   
9: $\hat { y } _ { \mathrm { s y s } } ( x _ { k } ) \gets \mathrm { a r g }$ max $M ( x _ { k } )$   
10: if write-back is enabled and the required payload   
source is available then   
11: Compute radius and payload for $x _ { k }$ using   
GuardNet logits and $W ;$ update C   
12: end if   
13: end if   
14: end for   
15: return $\{ \hat { y } _ { \mathrm { s y s } } ( x _ { k } ) \} _ { k = 1 } ^ { B }$

$$
\mathbb { E } [ \mathrm { C o s t } ] = O ( C _ { G } + N d + \operatorname { I n v o k e R a t e } _ { M } C _ { M } ) .\tag{16}
$$

When $C _ { G } + N d$ is much smaller than $C _ { M } ,$ the system can reduce the overall overhead in expectation even if the hit rate is not high. For large caches, hierarchical indexing or a vector database can further reduce the search complexity to sublinear, without changing the decision rule itself. Algorithm 2 summarizes the online pipeline.

Combining the three stages above, LipCache instantiates the abstract deployment policy of Section 3 as a parametric policy family $\pi = ( \bar { \theta _ { G } } , \mathcal { C } , \rho , \mathcal { R } , \omega ) _ { \cdot }$ , where $\theta _ { G }$ is the GuardNet parameter set, C is the feature cache, $\rho$ is the radius mode, R is the hit-resolution rule, and ω controls the payload-source and cache-center label-consistency options. Different parameter combinations correspond to different operating points—these are precisely the core objects of the experimental evaluation in Section 6.

## 5 THEORETICAL ANALYSIS

Section 4 gives LipCache a complete engineering pipeline, yet its core decision—the cache-hit test—is still an empirical definition: $\| z _ { \mathrm { n e w } } - z _ { i } \| _ { 2 } < r _ { i }$ . This section injects theoretical content into that decision: starting from the Lipschitz property of the classifier in the GuardNet feature space and the local classification margin, we derive a per-sample certified reuse radius $r _ { i }$ and prove that any query falling inside this radius necessarily preserves the same GuardNet prediction label as the cache center. Thereby, the accuracy constraint $\Delta _ { \mathrm { a c c } }$ in the optimization of Section 3 obtains a computable lower bound at the GuardNet level—as long as cache hits occur strictly within the certified radius, label consistency on the GuardNet side is guaranteed mathematically.

![](images/0e566c3999d0a156e1d7c8c866712b2f43b226b9f1d0ce3f51f1b02fc125528f.jpg)  
Fig. 2. Online inference pipeline of LipCache. For a new input, GuardNet extracts a compact feature and uses it to search the cache for certified reuse. A cache hit returns the stored payload; a miss triggers a MainNet fallback.

## 5.1 Lipschitz Setup

The structure of the certified reuse radius can be intuitively understood as “classification margin divided by the Lipschitz bound”: the larger the margin, the larger the admissible perturbation; the smoother the mapping, the smaller the logit change induced by the same input variation. We therefore first establish the Lipschitz bound of the GuardNet composite map $F _ { G } = f \circ { \mathsf { \bar { G } } }$

All Lipschitz constants are measured under the $\ell _ { 2 }$ norm. The feature encoder G has a finite Lipschitz constant

$$
L _ { G } \triangleq \operatorname { L i p } ( G ) ,\tag{17}
$$

and the per-layer spectral-normalization construction of Section 4.2 enforces $\bar { L } _ { G } \ \leq \ 1$ . For the affine classification head $f ( z ) ~ = ~ W z + b ,$ noting that the bias term cancels automatically in a logit difference, we have

$$
\begin{array} { r } { \| f ( z _ { 1 } ) - f ( z _ { 2 } ) \| _ { 2 } = \| W ( z _ { 1 } - z _ { 2 } ) \| _ { 2 } \leq \sigma _ { \operatorname* { m a x } } ( W ) \| z _ { 1 } - z _ { 2 } \| _ { 2 } , } \end{array}\tag{18}
$$

so $f$ is $\sigma _ { \operatorname* { m a x } } ( W )$ -Lipschitz. By submultiplicativity,

$$
\operatorname { L i p } ( F _ { G } ) = \operatorname { L i p } ( f \circ G ) \leq L _ { G } \sigma _ { \operatorname* { m a x } } ( W ) .\tag{19}
$$

Under the constraint $L _ { G } \leq 1 .$ , this simplifies to $\mathrm { L i p } ( F _ { G } ) \leq$ $\sigma _ { \operatorname* { m a x } } ( W )$ . This means that along any direction in the feature space, the rate of change of $F _ { G } { ' } \mathbf { s }$ output is controlled by $\sigma _ { \operatorname* { m a x } } ( W )$ . All certificates below are stated in the GuardNet feature space and depend only on $\sigma _ { \operatorname* { m a x } } ( W )$ —this is also the reason why the cache-construction phase in Section 4 needs to compute only this single spectral norm.

## 5.2 Certified Local Label Consistency

With the global Lipschitz bound in hand, we now turn to the local analysis: for a fixed cached sample, within

what neighborhood does its GuardNet prediction remain unchanged.

For a cached sample $x _ { i } ,$ let its feature be $z _ { i } ~ = ~ G ( x _ { i } )$ and its GuardNet prediction label be $y _ { i } ^ { G } = \arg$ ma $\tau _ { c } f _ { c } ( z _ { i } )$ Define the local classification margin of this sample as

$$
m _ { i } = [ f ( z _ { i } ) ] _ { y _ { i } ^ { G } } - \operatorname* { m a x } _ { j \neq y _ { i } ^ { G } } [ f ( z _ { i } ) ] _ { j } .\tag{20}
$$

Geometrically, $m _ { i }$ measures the logit-space margin between $z _ { i }$ and the nearest decision boundary—the larger the margin, the farther the sample lies from a decision boundary, and the larger the perturbation required to flip its label. This margin-based certification idea is widely adopted in the certified-robustness literature [29], [30], [31], [32], [33]. Accordingly, the certified reuse radius is defined as

$$
r _ { i } = \frac { m _ { i } } { \sigma _ { \mathrm { m a x } } ( W ) \sqrt { 2 } } ,\tag{21}
$$

where $\sigma _ { \operatorname* { m a x } } ( W ) > 0 ;$ the degenerate case $\sigma _ { \mathrm { m a x } } ( W ) = 0$ means that the classification head degenerates to a constant map, in which case the reuse radius of any positive-margin sample is unbounded, but this does not occur in practice. The $\sqrt { 2 }$ factor in the denominator comes from the worst-case analysis that simultaneously bounds all competing classes, and it will be tightened in Remark 5.1.

The following lemma is the core theoretical result of this framework; it establishes the guarantee that “the label is invariant inside the certified ball.”

Lemma 5.1 (Local label consistency). Let $z _ { i }$ be a cached feature with GuardNet label $y _ { i } ^ { \check { G } }$ and positive margin $m _ { i } > 0 _ { i }$ , and assume $\sigma _ { \mathrm { m a x } } ( W ) > 0$ . For any new sample $x _ { \mathrm { n e w } }$ whose GuardNet feature is $z _ { \mathrm { n e w } } = G ( x _ { \mathrm { n e w } } )$ , if

$$
\| z _ { \mathrm { n e w } } - z _ { i } \| _ { 2 } < r _ { i } = \frac { m _ { i } } { \sigma _ { \mathrm { m a x } } ( W ) \sqrt 2 } ,\tag{22}
$$

then the GuardNet prediction at $z _ { \mathrm { n e w } }$ is still $y _ { i } ^ { G }$

The core idea of the proof is to use the Lipschitz bound to translate the displacement in feature space into an upper bound on the perturbation in logit space, and then to show that this perturbation is insufficient to cover the original margin $m _ { i }$

Proof: Let $\delta = f ( z _ { \mathrm { n e w } } ) - f ( z _ { i } )$ . By the Lipschitz property of $f ,$

$$
\| \delta \| _ { 2 } \leq \sigma _ { \mathrm { m a x } } ( W ) \| z _ { \mathrm { n e w } } - z _ { i } \| _ { 2 } .\tag{23}
$$

To preserve the label $y _ { i } ^ { G } .$ , it suffices to show that for every competing class $j \neq y _ { i } ^ { \tilde { G } }$ we have $[ f ( z _ { \mathrm { n e w } } ) ] _ { y _ { i } ^ { G } } > [ f ( z _ { \mathrm { n e w } } ) ] _ { j } ^ { \cdot } .$ Substituting $f ( z _ { \mathrm { n e w } } ) = f ( z _ { i } ) + \delta$ into the logit difference:

$$
\begin{array} { r l } & { [ f ( z _ { \mathrm { n e w } } ) ] _ { y _ { i } ^ { G } } - [ f ( z _ { \mathrm { n e w } } ) ] _ { j } = [ f ( z _ { i } ) ] _ { y _ { i } ^ { G } } + \delta _ { y _ { i } ^ { G } } - [ f ( z _ { i } ) ] _ { j } - \delta _ { j } } \\ & { \qquad = \bigl ( [ f ( z _ { i } ) ] _ { y _ { i } ^ { G } } - [ f ( z _ { i } ) ] _ { j } \bigr ) - \bigl ( \delta _ { j } - \delta _ { y _ { i } ^ { G } } \bigr ) . } \end{array}\tag{24}
$$

By the definition of the margin, $[ f ( z _ { i } ) ] _ { y _ { i } ^ { G } } - [ f ( z _ { i } ) ] _ { j } \geq m _ { i }$ so a sufficient condition for preserving the label is

$$
\delta _ { j } - \delta _ { y _ { i } ^ { G } } < m _ { i } .\tag{25}
$$

To bound the left-hand side, introduce the auxiliary vector $e _ { j , y _ { i } ^ { G } } \mathrm { : }$ : it takes the value 1 at position $j , - 1$ at position $y _ { i } ^ { G } .$ and 0 elsewhere. Then $\| e _ { j , y _ { i } ^ { G } } \| _ { 2 } = \sqrt { 2 }$ , and

$$
\delta _ { j } - \delta _ { y _ { i } ^ { G } } = e _ { j , y _ { i } ^ { G } } ^ { \top } \delta \leq \| e _ { j , y _ { i } ^ { G } } \| _ { 2 } \| \delta \| _ { 2 } = \sqrt { 2 } \| \delta \| _ { 2 } ,\tag{26}
$$

where the inequality follows from Cauchy–Schwarz. Combining with the Lipschitz bound,

$$
\begin{array} { r } { \delta _ { j } - \delta _ { y _ { i } ^ { G } } \leq \sqrt { 2 } \sigma _ { \operatorname* { m a x } } ( W ) \lVert z _ { \mathrm { n e w } } - z _ { i } \rVert _ { 2 } . } \end{array}\tag{27}
$$

If $\| z _ { \mathrm { n e w } } \_ - \ z _ { i } \| _ { 2 } \ < \ m _ { i } / ( \sigma _ { \mathrm { m a x } } ( W ) \sqrt { 2 } )$ , then $\delta _ { j } \ - \ \delta _ { y _ { i } ^ { G } } \ <$ $m _ { i } ,$ i.e., the sufficient condition holds. Substituting gives $[ f ( z _ { \mathrm { n e w } } ) ] _ { y _ { i } ^ { G } } - [ f ( z _ { \mathrm { n e w } } ) ] _ { j } > 0$ for all $j \neq y _ { i } ^ { G }$ , and hence $y _ { i } ^ { G }$ remains the argmax label at z<sub>new</sub>. ■

This lemma directly provides the safety criterion for online hits in Section 4: as long as the query feature falls inside the certified ball, the GuardNet classification of that query cannot differ from the cache center—in other words, the cached payload $\tilde { y } _ { i }$ (when set to the proxy label $y _ { i } ^ { G } )$ is theoretically reliable.

Corollary 5.1 (Input-level sufficient condition). Given $L _ { G } \leq$ 1, any input satisfying $\| x _ { \mathrm { n e w } } - x _ { i } \| _ { 2 } < r _ { i }$ is guaranteed to satisfy $\| z _ { \mathrm { n e w } } - z _ { i } \| _ { 2 } < r _ { i } ,$ and hence its GuardNet prediction is $y _ { i } ^ { G }$ . More generally, if $L _ { G } > 0$ , the sufficient condition is $\| x _ { \mathrm { n e w } } - x _ { i } \| _ { 2 } < r _ { i } / L _ { G }$

Proof: $\| z _ { \mathrm { n e w } } - z _ { i } \| _ { 2 } = \| G ( x _ { \mathrm { n e w } } ) - G ( x _ { i } ) \| _ { 2 } \leq L _ { G } \| x _ { \mathrm { n e w } } -$ $x _ { i } \| _ { 2 } < r _ { i }$ . The conclusion follows directly from Lemma 5.1. ■

The above corollary shows that, when GuardNet is 1- Lipschitz, the r<sub>i</sub>-ball in input space directly constitutes a sufficient region for certified reuse. The online cache decision, however, still uses the feature-space check directly—it is tighter and requires no back-propagation through $L _ { G }$

## 5.3 Tighter Radius and Certificate Scope

Remark 5.1 (Tighter per-pair radius). The $\sqrt { 2 }$ factor in the radius $r _ { i }$ of Lemma 5.1 comes from taking the worst case over all $C - 1$ competing classes simultaneously. By analyzing each class separately, a tighter radius can be obtained. Let $W _ { c } \in \mathbb { R } ^ { d }$ denote the c-th row of $W .$ For each competing class $j \neq y _ { i } ^ { G }$ , decompose the logit difference as

$$
\begin{array} { r l } & { [ f ( z ) ] _ { y _ { i } ^ { G } } - [ f ( z ) ] _ { j } = \big ( [ f ( z _ { i } ) ] _ { y _ { i } ^ { G } } - [ f ( z _ { i } ) ] _ { j } \big ) } \\ & { \qquad + ( W _ { y _ { i } ^ { G } } - W _ { j } ) ^ { \top } ( z - z _ { i } ) . } \end{array}\tag{28}
$$

By Cauchy–Schwarz, the cross term is bounded by $- \| W _ { y _ { i } ^ { G } } - \mathbf { \bar { W } } _ { j } \| _ { 2 } \| z - z _ { i } \| _ { 2 } ,$ so when $\| z - z _ { i } \| _ { 2 } < r _ { i , j }$ we have $[ ^ { \prime } f ( z ) ] _ { y _ { i } ^ { G } } > [ f ( z ) ] _ { j }$ , where

$$
r _ { i , j } = \frac { [ f ( z _ { i } ) ] _ { y _ { i } ^ { G } } - [ f ( z _ { i } ) ] _ { j } } { \| W _ { y _ { i } ^ { G } } - W _ { j } \| _ { 2 } } .\tag{29}
$$

If $\| W _ { y _ { i } ^ { G } } \ - \ W _ { j } \| _ { 2 } \ = \ 0 ,$ the per-pair logit difference is constant and corresponds to no finite constraint. The tight per-sample radius takes the minimum: $r _ { i } ^ { * } =$ $\begin{array} { r } { \operatorname* { m i n } _ { j \not = y _ { i } ^ { G } } \hat { r _ { i , j } } . \mathrm { S i n c e } \ \| W _ { y _ { i } ^ { G } } - W _ { j } \| _ { 2 } \leq \sqrt { 2 } \sigma _ { \operatorname* { m a x } } ( W ) } \end{array}$ and $[ f ( z _ { i } ) ] _ { y _ { i } ^ { G } } ^ { \sim } - [ f ( z _ { i } ) ] _ { j } \geq m _ { i }$ , we always have $r _ { i , j } \geq r _ { i }$ , and hence $r _ { i } ^ { * } \geq r _ { i }$ . For consistency of exposition, the rest of the paper mainly uses the conservative radius $r _ { i } ;$ the experiments report results for both radii.

Remark 5.2 (Certificate scope and capability boundary). The object certified by Lemma 5.1 is the local consistency of the GuardNet classifier in the feature space: it guarantees that the GuardNet prediction label $y _ { i } ^ { G }$ is invariant inside the certified ball. When the cached payload $\tilde { y } _ { i }$ happens to equal $y _ { i } ^ { G }$ (the proxy-label mode), the label returned online is directly aligned with the certificate—this is the default setting adopted in the main experiments of this paper.

However, the lemma does not provide formal guarantees for the following cases: (1) when the payload source is the MainNet prediction ${ \hat { y } } _ { M } ( \boldsymbol { x } _ { i } )$ or the ground-truth label $y _ { i }$ and it is inconsistent with $y _ { i } ^ { G } .$ , the cached payload $\tilde { y } _ { i }$ returned on a hit may differ from the certified $\dot { y } _ { i } ^ { G } ; ( 2 )$ whether the MainNet prediction over the entire certified ball is consistent with the payload. In other words, the theoretical certificate of LipCache is on the GuardNet side, not on the MainNet side. The consistency between GuardNet and MainNet is indirectly promoted by the training objective and evaluated empirically in the experiments. To obtain a formal MainNet-side guarantee, one may further stack Lipschitz certification or randomized smoothing of MainNet itself.

Combining the derivations of this section, the online cache-hit decision of LipCache obtains a complete theoretical skeleton: starting from the Lipschitz control of GuardNet (Section 4.2), Lemma 5.1 turns the local classification margin into a computable per-sample certified radius, Remark 5.1 provides a larger and less conservative persample certified radius, Corollary 5.1 extends the condition to input space, and finally Remark 5.2 clarifies the applicable scope of the certificate. Under this theoretical framework, the experimental evaluation in Section 6 reports not only the hit rate and accuracy, but also verifies, via the “certifiedconsistency rate,” whether every accepted cache reuse indeed occurs inside the certified boundary.

## 6 EXPERIMENTS

Section 5 establishes the theoretical guarantee for cache reuse in LipCache: any query that falls inside the certified radius $r _ { i }$ necessarily has a GuardNet prediction label consistent with the cache center. This section subjects that theoretical guarantee to empirical examination, organizing the experimental evaluation around three questions. First, on real image classification tasks, can certified caching form non-trivial reuse and stably hold under limited accuracy loss. Second, is the certified radius necessary relative to empirical thresholds—that is, does there exist a simpler alternative boundary that equally satisfies certified consistency. Third, with the certification condition held fixed, how do cache strategy, feature dimensionality, training objective, and class count jointly determine the operating point among hit rate, accuracy, and latency gains. It should be noted that the evaluation goal of this section is the viability of the certification mechanism itself under i.i.d. test distributions, i.e., examining whether the certified radius can form non-trivial reuse while preserving theoretical consistency. Temporal redundancy, concept drift, and consecutive-frame reuse rates in streaming deployment constitute evaluation problems of a different nature, and their benchmark design lies outside the scope of this paper (see Section 7.1).

The main text consistently reports three core metrics: the cache hit rate H (the frequency with which the system reuses the cache), the end-to-end accuracy (the agreement between the overall system output and the ground truth), and the certified-consistency rate (the fraction of all accepted cache hits that satisfy the certification condition; 100% means that no theoretical violation occurred). In addition, the measured system-level speedup is defined as

$$
S _ { \mathrm { t i m e } } = \frac { T _ { \mathrm { M a i n } } } { T _ { \mathrm { g u a r d } } + T _ { \mathrm { s e a r c h } } + T _ { \mathrm { f a l l b a c k } } } ,
$$

where the numerator is the measured time of executing MainNet alone, and the denominator is the measured total time of GuardNet encoding, cache search, and, on a miss, the MainNet fallback. Although the speedup is sometimes written as $1 / ( 1 - H )$ , this expression only reflects the reduction in MainNet invocations and ignores the encoding and search overhead; we therefore adopt the measured definition $S _ { \mathrm { t i m e } }$ above as the reported metric.

## 6.1 Experimental Setup

To keep the experiments comparable, the main-text experiments share a common base protocol. MainNet is responsible only for fallback inference on misses and serves as the standalone-accuracy and timing baseline; GuardNet is responsible for extracting features that satisfy the 1-Lipschitz constraint and accordingly defining the theoretical radius for cache reuse. Unless stated otherwise, the three 10-class tasks all use the tight certified radius, a cache-construction method that prioritizes the certified radius combined with de-duplication, returns the proxy label on a hit, caches 200 samples per class, and repeats the experiments over random seeds {42, 123, 2024}. The standalone accuracies of the three MainNet models are as follows: 94.78% on CIFAR-10 [34], 87.0% on Tiny-ImageNet 10 classes (denoted Tiny-10) [35], and 97.19% on SVHN [36]. In the current final experiments, the certified-consistency rate of all certified-radius results is 100%, and no Lipschitz-constraint violation or theoretical violation at the proxy-decision level is observed.

The training-objective ablation changes only how GuardNet is trained, with everything else held fixed; the multi-class extension experiment changes only the class count and the corresponding GuardNet training recipe; the perturbation and deployment experiments examine the behavior of the system under distributional perturbation and latency constraints on top of the default operating point.

Scope of the evaluation: The experiments are designed to validate the viability and cross-task stability of the persample certified reuse mechanism, rather than to propose a new SOTA image classification benchmark. The three tasks respectively represent three typical 10-class distributions— natural images, fine-grained classification, and domain shift—which suffice to examine the sensitivity of the certified radius to task geometry. Extension to full ImageNet (1000 classes), CIFAR-100, and streaming video benchmarks lies outside the scope of this paper and is left to future work.

## 6.2 Feasibility of Certified Caching

Figure 3 reports the hit rate, end-to-end accuracy, certifiedconsistency rate, and measured speedup on the three 10- class tasks. Figure 4 portrays the three tasks side by side from the perspective of relative operating points.

Under the conservative certified radius, the average hit rates of CIFAR-10, Tiny-10, and SVHN are $0 . 3 4 1 \pm \check { 0 } . 0 1 0 ,$ $0 . 3 0 3 \pm 0 . 0 1 3 ,$ and $0 . 4 7 0 \pm 0 . 0 0 7 ,$ respectively; switching to the tight certified radius, the hit rates rise to $0 . 3 4 2 \pm 0 . \dot { 0 } 1 1$ $0 . 4 7 2 \pm 0 . 0 1 2 ,$ and $0 . 5 0 2 \pm 0 . 0 1 3$ . The corresponding endto-end accuracies remain at 0.921, 0.867, and 0.960, and all results pass the certified-consistency check.

A non-trivial certified reuse region is observed on all three tasks. Under the tight certified radius, the hit rates of CIFAR-10, Tiny-10, and SVHN are all significantly above zero, indicating that LipCache does not trigger theoretically safe reuse only in the vicinity of a tiny minority of samples, but rather can form stable usable hits over the full test set. Meanwhile, the end-to-end accuracy is only about 2.70, 0.27, and 1.19 percentage points lower than the respective standalone MainNet accuracy, showing that certified caching does not trade a large loss in prediction quality for reuse opportunities.

The difference between the conservative and the tight certified radius reveals the basic principle that “the certification boundary can be tightened but cannot be loosened arbitrarily.” On CIFAR-10, the two radii are already close, and the tight radius brings only a 0.15 percentage-point gain in hit rate and about 0.04× in speedup, with accuracy nearly unchanged. On Tiny-10, the tight certified radius raises the hit rate from 0.303 to 0.472 and the measured speedup from $1 . 1 1 \times \mathrm { t o } 1 . 3 1 \times .$ , with accuracy dropping by only about 0.20 percentage points—the conservative radius is overly cautious on higher-resolution images. SVHN lies between the two: the hit rate rises from 0.470 to 0.502, the speedup from $1 . 5 8 \times \mathrm { t o } 1 . 6 5 \times ,$ and the accuracy drops by about 0.21 percentage points. Overall, the tight certified radius trades a moderately enlarged certified reuse region for a higher hit rate, but without breaking certified consistency.

![](images/4413158dc29ff472a72c9170687c96eca330d3bfee044bcfc57f2bc6abfc766d.jpg)

![](images/f2381cd22851c673ba00f57dcbf4a8ba5b1ff6581b784f9e929e2f5231210104.jpg)

![](images/36c8195f5919b4ff63b786e6b127e0d43aae3e777151619ecdea979a3f911bfb.jpg)

![](images/3a09d27d32da7e328fb01ec8b3e650eb737352f4abe3e500366956027ce55d60.jpg)  
Fig. 3. Main results on the three 10-class tasks. From left to right: cache hit rates under the conservative versus tight certified radius; end-toend accuracy under the tight radius, with the standalone MainNet accuracy marked by a dashed line; certified-consistency rate; and measured speedup. All certified-radius results pass the consistency check.

The differences across the three datasets reflect the systematic influence of task geometry on the efficiency of certified reuse. SVHN simultaneously exhibits the highest hit rate and the highest measured speedup—on tasks such as digit images, where intra-class variation is constrained, the local geometry learned by GuardNet is more stable, and the certified reuse region more easily covers test samples. The hit rate of Tiny-10 is notably higher than that of CIFAR-10, but its speed gain does not increase proportionally, because the higher input resolution raises the cost of GuardNet encoding and nearest-neighbor search, weakening the translation of hit-rate growth into real wall-clock gains. CIFAR-10 has the lowest hit rate, yet it still maintains a measured speedup above 1.3×—under a lower encoding cost, medium-scale certified hits are already sufficient to yield stable system gains.

Figure 4 displays, in a column-normalized side-by-side manner, the hit rate under the tight certified radius, the hitrate gain relative to the conservative radius, the accuracy retention relative to MainNet, the overall prediction agreement with MainNet, and the measured speedup. SVHN dominates simultaneously on hit rate, accuracy retention, and speedup; Tiny-10 stands out most on hit-rate gain; and CIFAR-10 maintains a high level of agreement with MainNet but has a limited hit-rate gain. This structured comparison further shows that the different benefit dimensions of LipCache are emphasized differently across tasks.

In terms of random-seed stability, the fluctuations of the three main results are all small. Under the tight certified radius, the hit-rate standard deviations of CIFAR-10, Tiny-10, and SVHN are 0.011, 0.012, and 0.013, respectively, and the accuracy standard deviations all remain at about the 0.002 level or below, indicating that the main conclusions do not depend on a particular training seed or cache-sample composition.

Overall, Figures 3 and 4 jointly establish a consistent result pattern for LipCache on three 10-class tasks with markedly different statistical structure: non-zero certified hits, limited accuracy loss, measurable latency gains, and zero theoretical violations.

## 6.3 Comparison with Related Efficiency Systems

Related work shares the goal of reducing inference cost, but the savings occur at different points in the execution path. BranchyNet [7] and MSDNet [19] retrain multi-exit networks so that easy inputs leave early at the execution node; they reduce computation after a request reaches that node and do not reuse results associated with prior inputs. NoScope [9] specializes a cascade to fixed-camera video, DeepCache [10] reuses regions across consecutive video frames, and Semantic Memory [11] applies empirical semantic caching through cross-layer similarity. Their locality assumptions, acceptance conditions, and deployment paths differ from ours. LipCache keeps MainNet unchanged, uses a lightweight GuardNet to retrieve cached proxy labels at the edge, and contacts the cloud only on a miss. It therefore first reduces uploads and cloud invocations, and can then be composed with a cloud early-exit network to reduce the cloud computation of misses.

## 6.3.1 Edge–Cloud Resource Accounting

Table 2 reports both edge-resident model footprint and a common edge–cloud deployment accounting model. Parameter counts and raw FP32 weights describe static weights in the current CIFAR-10 implementations, excluding activations, runtime state, cache entries, and framework overhead; they are therefore necessary but not sufficient evidence of edge-deployment feasibility. If an early-exit network is placed at the edge, its complete backbone and all exits must remain resident: the three-exit BranchyNet has 1.759M parameters/6.71 MiB and the seven-exit MS-DNet has 2.987M/11.39 MiB, whereas the base GuardNet $( d \ = \ 6 4 , c \ = \ 3 2 )$ used by LipCache has 0.282M/1.08 MiB. Normalize the compute budget of a complete cloud prediction to 100%, let the certified CIFAR-10 operating point have hit rate $H = 0 . 3 4 8$ , and use the native earlyexit compute speedups relative to each system’s complete backend, $S _ { B } ~ \overset { \cdot } { = } ~ 2 . 3 \overset { \cdot } { 1 }$ for BranchyNet and $S _ { M } ~ = ~ 2 . 1 4$ for MSDNet. For LipCache, the uploaded-request fraction is $1 - H ;$ when a cloud early-exit backend handles only misses, normalized cloud computation is $( 1 - H ) / S _ { B }$ or $( 1 - H ) / S _ { M }$ . Cloud-compute saving is $1 - C$ and cloudcapacity gain is $1 / C ,$ , where C is the normalized cloud computation.

The accounting exposes the complementarity. Cloud BranchyNet and MSDNet reduce cloud computation to

Column-normalized operating-profile heatmap  
![](images/334363f44547b0ca9e1865d0e41bbfe3509b32a1fa7a84893988adbff767208c.jpg)  
Fig. 4. Operating-point profile heatmap of the three 10-class tasks. The columns correspond to the hit rate under the tight certified radius, the hit-rate gain relative to the conservative radius, the accuracy retention relative to MainNet, the overall prediction agreement with MainNet, and the measured speedup. Colors are normalized within each column; the annotated numbers retain their original magnitudes.

TABLE 2  
Edge-resident model footprint and edge–cloud resource accounting at the CIFAR-10 operating point. Footprint denotes the static FP32 weight volume of the edge-resident model, distinct from the runtime compute cost analyzed in later subsections. Uplink traffic and cloud computation are normalized to a complete cloud prediction (100%); cloud computation applies early-exit speedups as conditional-compute multipliers, yielding a deployment accounting model rather than a cross-architecture comparison of FLOPs, latency, accuracy, or energy.
<table><tr><td>ID (edge → cloud)</td><td>Configuration</td><td>Edge footprint (params/size in FP32 MiB)</td><td>Uplink traffic (vs. all-cloud)</td><td>Cloud compute</td><td>Compute saved</td><td>Capacity gain</td></tr><tr><td colspan="7">Cloud-only baselines</td></tr><tr><td>(1)</td><td>— → MainNet</td><td></td><td>100.0%</td><td>100.0%</td><td>0.0%</td><td>1.00×</td></tr><tr><td>(2)</td><td>— → BranchyNet</td><td>1.759M† / 6.71†</td><td>100.0%</td><td>43.3%</td><td>56.7%</td><td>2.31×</td></tr><tr><td>(3)</td><td>— → MSDNet</td><td>2.987M† / 11.39†</td><td>100.0%</td><td>46.7%</td><td>53.3%</td><td>2.14×</td></tr><tr><td colspan="7">Edge LipCache + cloud backend</td></tr><tr><td>(4)</td><td>GuardNet → MainNet</td><td>0.282M / 1.08</td><td>65.2%</td><td>65.2%</td><td>34.8%</td><td>1.53×</td></tr><tr><td>(5)</td><td>GuardNet → BranchyNet</td><td>0.282M / 1.08</td><td>65.2%</td><td>28.2%</td><td>71.8%</td><td>3.54×</td></tr><tr><td>(6)</td><td>GuardNet → MSDNet</td><td>0.282M / 1.08</td><td>65.2%</td><td>30.5%</td><td>69.5%</td><td>3.28×</td></tr></table>

<sup>†</sup>: footprint if the cloud-resident network were deployed at the edge (shown for size comparison only).

43.3% and 46.7%, respectively, but leave uploads at 100%. LipCache first hits at the edge for 34.8% of requests, reducing both uploads and cloud invocations to 65.2%. In composition, cloud computation further falls to 28.2% or 30.5%, corresponding to 3.54× or 3.28× cloud-capacity gain. The H = 0.348 value is the controlled CIFAR-10 operating point of this section and differs from the H = 0.3295 realdevice path in Section 6.8; the two are not mixed. Early-exit networks are not restricted to cloud deployment; the table describes their resource effect when used as the cloud backend. Certification still covers only local GuardNet proxylabel consistency when a cached label is returned, not the early-exit network, MainNet, or ground-truth correctness. Combined end-to-end latency, accuracy, edge energy, and live network traffic require measurement on a unified path.

## 6.4 Certified Radius and Empirical Thresholds

Figure 5 compares three kinds of reuse boundaries: the tight certified radius, an empirical radius based on interclass distance statistics, and a single global nearest-neighbor threshold. The three datasets exhibit a consistent pattern: the certified radius does not pursue the highest hit rate, but it is the only boundary definition that can stably keep the certified-consistency rate at 100% across all three tasks.

Taking CIFAR-10 as an example, the hit rate, accuracy, and consistency corresponding to the tight certified radius are 0.344, 0.922, and 100%, respectively; the inter-classdistance empirical radius slightly raises the hit rate to 0.352, but the certified-consistency rate drops to 37.9%; the global nearest-neighbor threshold pushes the hit rate up to 0.494, yet it drives the end-to-end accuracy down to 0.777. On Tiny-10, the empirical thresholds are likewise more aggressive, but their corresponding certified-consistency rates are only 33.3% and 23.9%; on SVHN, the empirical thresholds do not even achieve a higher hit rate, yet they still depress the certified-consistency rate to 45.7% and 25.4%. The empirical thresholds either trade accuracy and consistency for a higher hit rate, or damage the certification condition without notably improving the hit rate; they do not form a better accuracy–hit-rate frontier across the three tasks.

This result defines the role of the certified radius in the main text: it does not pursue the most aggressive empirical hit rate, but provides a reproducible, interpretable, and cross-task consistent reuse boundary.

![](images/e53eafb20b81cfa11f23113cc92a7b3cc32b41ad819a6cc3de8da98204fafff9.jpg)  
Fig. 5. Comparison of the certified radius and empirical thresholds. Empirical thresholds sometimes yield a higher hit rate, but only the tight certified radius keeps the certified-consistency rate at 100% across all three 10-class tasks while maintaining high end-to-end accuracy.

## 6.5 Default Operating Point Configuration

Figures 6 and 7 answer why the main-text default configuration adopts the operating point of “tight certified radius + radius-prioritized de-duplication + proxy label + 64-dimensional features + 200 samples per class.”

On the entry-selection strategy, radius-prioritized deduplication gives the highest hit rate on CIFAR-10 and also outperforms random sampling and K-center on Tiny-10; on SVHN, the hit rate of K-means is slightly higher than that of radius-prioritized de-duplication, but the gap is small. Considering the consistency and interpretability across the three tasks, radius-prioritized de-duplication constitutes a more robust unified default strategy, which also indicates that the certified-caching scenario requires not only geometric coverage but also prioritized retention of high-quality center samples with large certified radii.

Regarding return-value semantics, the proxy label is directly aligned with the theoretical guarantee and provides the clearest semantics. On CIFAR-10, the end-to-end accuracies of the proxy-label, MainNet-label, and groundtruth schemes are 0.9215, 0.9134, and 0.9134, respectively; on Tiny-10, they are 0.8680, 0.8680, and 0.8700, respectively; and on SVHN, they are 0.9617, 0.9241, and 0.9237, respectively. These results show that the proxy label achieves higher end-to-end accuracy on CIFAR-10 and SVHN, whereas the differences among the three semantics are small on Tiny-10. In all cases, the proxy label remains fully aligned with the certification condition. The main text adopts the proxy label as the default precisely because it has the clearest theoretical semantics under a unified protocol.

Cache capacity remains an effective lever. On all three datasets, increasing the cache from 100 to 400 samples per class raises the hit rate: CIFAR-10 from 0.288 to 0.392, Tiny-10 from 0.446 to 0.480, and SVHN from 0.434 to 0.531. Capacity growth, however, does not automatically bring better accuracy or lower latency; the final operating point should therefore be selected jointly with the radiustightening strategy discussed below.

The optimal feature dimensionality is datasetdependent. On CIFAR-10, the hit rates for $\dot { d } = 3 2 / 6 4 / 1 2 8$ are 0.348/0.344/0.336, respectively, and 32 dimensions are already near-optimal; on Tiny-10, the corresponding hit rates are 0.456/0.470/0.452, with little difference between 32 and 64 dimensions; on SVHN, the hit rate continues to rise from 0.498/0.484 to 0.527, so higher dimensions still bring gains. The main text retains 64 dimensions as the unified default in order to share the same protocol across the three tasks; the optimal dimensionality of different datasets is not fully consistent, and this is precisely the degree of freedom that needs to be re-tuned per task in subsequent deployment.

## 6.6 Hit Mechanism and Radius Tightening

After determining the default operating point, two phenomena still need to be explained: why incorrect hits occur mainly near the boundary, and why tightening the radius can improve accuracy without breaking the certification condition. Figure 8 provides the answer.

On all three tasks, the normalized distance ratio $\parallel z _ { q } -$ $z _ { i } \| _ { 2 } / r _ { i }$ of correct hits is consistently and markedly lower than that of incorrect hits. The mean distance ratios of correct hits on CIFAR-10, Tiny-10, and SVHN are 0.802, 0.752, and 0.767, respectively, whereas those of incorrect hits rise to 0.891, 0.910, and 0.908, systematically closer to the certification boundary. Meanwhile, the cache-center classification margin of correct hits is significantly larger: 2.43, 3.77, and 2.38 on the three datasets, versus only 1.63, 1.97, and 1.49 for incorrect hits. This indicates that highrisk hits do not appear at random, but are concentrated in the region “closer to the boundary and with a smaller classification margin.”

The lower panels of Figure 8 show the radius-tightening phenomenon directly. When the radius is progressively tightened on top of the conservative certified radius, all three datasets exhibit a decrease in hit rate and an increase in accuracy, while the certified-consistency rate remains at 100% throughout. For example, on CIFAR-10 the hit rate drops from 0.344 to 0.200 and the accuracy rises from 0.922 to 0.940; on SVHN the hit rate drops from 0.460 to 0.312 and the accuracy rises from 0.964 to 0.970. Radius tightening does not change whether the certification holds; rather, by removing high-risk hits closer to the boundary, it pushes the system toward a more conservative but higher-accuracy operating point.

![](images/3098de94267aa08bd25f1969e0bb70170fadb5861eb90532752f4f922150bba5.jpg)

![](images/01e36b2f7a26359fd848acd7794f577662b9ac912cf948fb1fd67545a91c47b7.jpg)

![](images/e078fa8b8e634b3200775919ef49e81b9d6c3b37073a2d449253cd6c099efb6e.jpg)  
Fig. 6. Default-operating-point ablation on the three 10-class tasks. Left: effect of entry-selection strategy on the hit rate; middle: effect of returnvalue semantics on end-to-end accuracy; right: effect of per-class cache capacity on the hit rate.

![](images/98d7072f1023dd8ee769b0d2ab2aeb09bb6b6c28cf3882819e2b9a846f304f7c.jpg)

![](images/9f4215eabde27a52ad8506749c5e975d846c0f92b1eb8d77ccc9438ef0e263a3.jpg)  
Fig. 7. Effect of feature dimensionality on the hit rate and the measured speedup. CIFAR-10 and Tiny-10 are already near their respective optima at smaller dimensions, whereas SVHN continues to improve at higher dimensions, indicating that the unified default dimensionality is not fully identica to each dataset’s optimum.

![](images/68471ac542db9059d0ed5abc382edbcefe4e72beb931ad4816cb068cc3772261.jpg)

![](images/3e64702ade2d19072a80f4c83c9c85bab7fecb737a690eadbca2d9a9d8419c63.jpg)

![](images/88231ea4e0809edbf02b907cea50930748268bdd88a98ff2d6c4f84c1d1099e6.jpg)

![](images/021acbcd04d09e9c5d1408b2d2eb1257c85694d1ec8f999b845822def5e17431.jpg)  
Fig. 8. Hit mechanism and radius tightening. Top: incorrect hits are closer to the certified boundary $\| z _ { q } - z _ { i } \| _ { 2 } = r _ { i }$ and have a smaller cache-hit margin. Bottom: progressively tightening the certified radius lowers the hit rate while raising accuracy and preserving theoretical self-consistency on all three 10-class tasks.

## 6.7 Training Objective and Multi-Class Extension

The above results show that the main degrees of freedom of the caching system lie not in whether to certify, but in how to change the hit-rate–accuracy operating point through training and radius selection. We therefore examine, respectively, the training-objective ablation on CIFAR-10 and the multiclass extension experiment on Tiny-ImageNet.

Figure 9 gives the training-objective ablation results on CIFAR-10. Cross-entropy supervision alone provides the most robust default operating point among the current 10- class main experiments, with a hit rate and accuracy of 0.358 and 0.920, respectively. The margin-enlargement regularizer mildly raises the hit rate: under a stronger marginregularization setting, the hit rate rises to 0.401 and the accuracy drops to 0.909. The center constraint that compresses the intra-class distribution is more aggressive: the hit rate can be further raised to 0.544, 0.596, and even 0.644, with the corresponding accuracy dropping to 0.894, 0.888, and 0.877. It is worth emphasizing that all training-objective variants pass the certified-consistency check—the training objective changes the operating point, not the theoretical validity itself.

Figure 10 answers a more critical question: after a stronger GuardNet training recipe, can certified caching scale from 10 classes to larger class counts? On the crossentropy-only multi-class baseline, the hit rates under the tight certified radius at 20/30/50 classes are only 0.056, 0.0047, and 0.0004, respectively; after adopting the enhanced training recipe, these three values rise to 0.423 ± $0 . 0 0 9 , 0 . 2 5 4 \pm \bar { 0 } . 0 0 4$ , and $0 . 1 2 4 \pm 0 . 0 0 1$ , respectively, while the end-to-end accuracy remains at 0.838, 0.847, and 0.836, and the certified-consistency rate stays at 100%. This indicates that the main bottleneck in the multi-class scenario lies not in the caching mechanism itself, but in whether GuardNet can learn a sufficiently compact and separable certified feature geometry.

Nevertheless, the growth in class count remains the dominant factor determining the upper bound of reusability. Even with the enhanced training recipe, the hit rate keeps declining as the class count increases. Increasing the feature dimensionality cannot fundamentally reverse this trend: under the 20/30/50-class settings, $\dot { d } \ = \ 3 2$ achieves the highest hit rates of 0.431, 0.266, and 0.158, respectively, all outperforming $d \ = \ 6 4$ under the same settings. The message conveyed by the multi-class extension is not “one can abandon certification when the class count grows,” but rather “improving GuardNet training can markedly enlarge the usable class scale while preserving the certification condition, yet the class count still determines the upper bound of certified reuse.”

## 6.8 Query Perturbation and Deployment Boundaries

Figure 11 gives the empirical stress-test results under query perturbation. This group of experiments should be understood as an empirical generalization assessment rather than a direct extrapolation of the theoretical certificate. All three datasets exhibit the same pattern: additive noise is the most stable, JPEG compression brings a mild decline, and cropping and scaling are the most damaging to the hit rate. For example, on CIFAR-10 the baseline hit rate under the tight certified radius is 0.343; it drops to 0.310 under strong JPEG perturbation, and drops to 0.093 and 0.152 under strong cropping and strong scaling, respectively. Tiny-10 and SVHN preserve the same trend.

![](images/8f021615685f15f388541583de435a89c6658d4c4584f97dad6783250afe2b9f.jpg)  
Fig. 9. Training-objective ablation on CIFAR-10. Cross-entropy supervision gives the most robust default operating point; margin regularization can mildly raise the hit rate; the center constraint can further raise the hit rate, but at a more noticeable accuracy cost.

More importantly, the performance degradation under perturbation comes mainly from “not hitting” rather than “hitting but being wrong.” In all perturbation experiments, the certified-consistency rate remains at 100%, indicating that certified consistency itself is not broken; the performance change is mainly because the perturbed query points more easily escape the original certified ball, rather than because local consistency inside the certified ball has failed.

It should be emphasized that the above perturbation tests examine the robustness of the certified radius to singlepoint local perturbations and remain within the i.i.d. evaluation framework. Temporal redundancy (e.g., inter-frame coherence in video) and concept drift in streaming scenarios involve sequential correlations and require dedicated streaming benchmarks together with an analysis of how the hit rate evolves over time; these are left as future work. The results in this section can serve as an upper-bound reference for streaming deployment: if the i.i.d. hit rate is $H _ { 0 . }$ , then under strong temporal correlation the probability that consecutive frames fall inside the same certified ball is typically no lower than $H _ { 0 }$

We further deploy the baked CIFAR-10 GuardNet on an Atlas 200I DK A2 equipped with a DaVinci 310B4 NPU. The edge board executes GuardNet and cache lookup, while the remote MainNet latency is measured with ResNet50 [37] on an RTX 4070 Ti. The device evaluation uses 2,000 queries and a 2,000-entry cache, obtaining a hit rate of 0.3295 and an end-to-end accuracy of 0.9170.

Figure 12 reports the real-device latency components and the RTT-based split-inference estimate. The NPU takes 1.34 ms/query for GuardNet encoding and the cKDTree lookup takes 0.22 ms/query, giving an edge-local path of 1.56 ms/query. Cloud batch-1 MainNet inference takes 8.38 ms/query. Combining these measured components with LAN, WiFi, and cellular RTTs of $2 / 1 0 / 4 0$ ms gives expected speedups of 1.22×, 1.32×, and 1.42×, respectively.

Figure 13 exposes two limits that are hidden by the batch-1 result. First, the cloud GPU reduces its per-image time from 8.38 ms at batch 1 to 0.207 ms at batch 128, whereas the edge NPU encoding time follows a shallow U-shape (1.34/0.81/0.87/1.18 ms). Second, a discrete-event simulation driven by these measured per-operation times shows a saturation boundary: at 1,000 queries/s, the edge reaches approximately 99% utilization and LipCache p99 latency rises to 177.4 ms, compared with 27.8 ms for allcloud inference. Thus, edge caching is advantageous when RTT is material and the edge retains spare capacity, but it can move the bottleneck from the cloud to the edge under extreme load.

![](images/e4ccaf59f2b109052c747156487c3a5f5dfc79d60e890cc4004c9379c05d3691.jpg)

![](images/fe77940fae15bb3d4a04d0170e7152ed675224052f925dc11a7f1bb33d267fbf.jpg)

![](images/524091ed76fc1d68ff907a53b54eee59c5d7c9cc1dbdaa89e8e10dd509c792a9.jpg)

![](images/3d1496b9a3087d7bc6fec106879b3ddc317ef8689bb606ba1b9b45e755446da6.jpg)

Fig. 10. Tiny-ImageNet multi-class extension experiment. Top left: comparison of the tight-certified-radius hit rates of the cross-entropy-only baseline and the enhanced training recipe at different class counts; top right: conservative versus tight certified-radius hit rates of the enhanced training recipe at 20/30/50 classes and the corresponding end-to-end accuracy; bottom left: relationship between feature dimensionality and hit rate at different class scales; bottom right: certification consistency of different radius definitions in the multi-class scenario.  
![](images/9f6b064a7fff254d1a688becb777413030cb83882cd99961f548dcbfd8ed470b.jpg)

![](images/c69d2db0755a3afa4653b84f1a0bb61470d0f376ee61870aa95e96c8ec60fbbe.jpg)

![](images/2db7b723e59f40f9bb14c8bb73892883ccf27aec0c3395a935ed2d3bb1567a15.jpg)

![](images/065c8ea6acf06854af0edc9baf1f66bdc5c6d3265fda978b8b2f36e07e02fb86.jpg)  
Fig. 11. Empirical stress test under query perturbation. Noise perturbation barely affects the certified hit rate of the three 10-class tasks; JPEG compression brings a mild decline; cropping and scaling more easily push query points out of the certified ball, so the hit-rate decline is more pronounced.

The measurement scope is deliberately separated in the figures. NPU encoding, cache lookup, and cloud inference are hardware measurements; the network speedups combine those measurements with modeled RTT rather than a live HTTP round trip; and the load experiment is a discrete-event simulation parameterized by measured operation times rather than a real traffic trace.

## 6.9 Retrieval Complexity Analysis

Figure 14 isolates the variable retrieval term in Eq. (16). We fix $C _ { G }$ and InvokeRate $\mathfrak { a } _ { M } C _ { M }$ at the final CIFAR-10 seed-42 operating point $( d = 6 4 , N = 2 , 0 0 0$ , hit rate 0.3437, hence InvokeRate $_ { M } = 0 . 6 5 6 3 )$ . The left panel fixes N = 2,000 and varies d; the right fixes d = 64 and varies N. The unchanged additive terms are normalized to one work unit, so both panels directly show the linear contribution of N d. This is an analytical complexity analysis of the online decision rule, not a hardware-latency measurement.

## 6.10 Section Conclusions

The experiments in this section support the following conclusions. First, LipCache can stably achieve certified cache reuse on the three 10-class tasks of CIFAR-10, Tiny-10, and SVHN, and the main results consistently pass the certified-consistency check. Second, although the certified radius is more conservative than empirical thresholds, it is the only reuse boundary that can stably preserve certified consistency across tasks. Third, cache strategy, return-value semantics, feature dimensionality, and radius tightening mainly determine the operating point among hit rate, accuracy, and latency, rather than whether certification holds. Fourth, the training-objective ablation on CIFAR-10 and the multi-class extension on Tiny-ImageNet jointly show that improving GuardNet training can markedly change the operating point and even enlarge the usable class scale, yet the growth in class count remains the core factor limiting the upper bound of certified reuse. Finally, the perturbation, deployment, and complexity analyses show that the applicability domain of LipCache is jointly determined by the local feature geometry, cache scale, the edge encoding cost, and the network RTT, providing clear boundary conditions for subsequent system-level extensions.

## 7 CONCLUSION AND FUTURE WORK

This paper proposed LipCache, a certified semantic caching framework for resource-constrained edge image classification. LipCache retains the high-accuracy MainNet as a fallback predictor and introduces a lightweight, Lipschitz-controlled GuardNet that makes cache-hit decisions in a compact feature space. By deriving a per-sample certified reuse radius from the local GuardNet margin and the Lipschitz bound of the classification head, the framework turns cache lookup into an interpretable local-consistency check, rather than an exact-match or global heuristic-threshold decision.

(b) RTT-based split-inference estimate  
![](images/eb3f01c5cf427647a77656b568b4d5881a04fb640d9c604752e90f356e110203.jpg)

![](images/94c1d7a52ba1c1244183fb6871fecb45d9339a604d6052da69043c426dffdd11.jpg)  
Fig. 12. Real-device measurements for the CIFAR-10 operating point. Left: measured latency components on the DaVinci 310B4 edge NPU and RTX 4070 Ti cloud GPU. Right: expected speedup obtained by combining the measured components with modeled LAN, WiFi, and cellular RTTs. The right panel is an analytical estimate, not a measured HTTP round trip.

The experimental results show that, on three 10- class tasks with markedly different statistical structure, LipCache consistently exhibits the following pattern of results: “non-zero certified hits, limited accuracy loss, measurable latency gains, and zero theoretical violations.” The comparison between the certified radius and empirical thresholds further shows that the per-sample safety criterion does not pursue the most aggressive hit rate, but provides the only reuse boundary that can stably preserve certified consistency across tasks. Entry-selection strategy, return-value semantics, feature dimensionality, and radius tightening mainly affect the operating point among hit rate, accuracy, and latency, rather than whether certification holds. The multi-class extension experiment further reveals that the main limitation of the current method lies not in the cache criterion itself, but in whether GuardNet can learn a sufficiently compact and separable certified feature geometry. The edge-deployment analysis further indicates that the system gain of LipCache is jointly determined by the hit rate, the guard-side encoding cost, and the network RTT, and that it is best suited to deployment scenarios where “cloud invocations are expensive while the edge side can afford a lightweight proxy.”

![](images/49db604e6664b4e7dc66fd5671080c1bcd39c9e3b852738d2d9768036c3905d6.jpg)

![](images/6a35e430bc2fd2e64a0ad8fa9843fa3ba4d5ebec71e414a529e716412665891f.jpg)  
Fig. 13. Operating boundary of the edge deployment. Left: measured per-query latency under different batch sizes, showing the much stronger batching benefit of the cloud GPU. Right: p99 latency from a discrete-event simulation parameterized by measured operation times; the sharp rise near 1,000 queries/s marks edge saturation.

Eq. (16): fixed $C _ { G }$ and InvokeRat $\mathsf { \Pi } _ { M } ^ { \mathsf { \Gamma } } C _ { M }$ (InvokeRat $\mathsf { e } _ { M } = 0 . 6 5 6 3 )$  
![](images/30865a36236ed10b6d0b079197824732a7fb0cdf656a9f3dd7098a8b47cda439.jpg)

![](images/24f3754932e1e3d70563b0fc7ebc38cd84c1ae9e88d3f2737c5dc96cc83322b3.jpg)  
Fig. 14. Complexity analysis corresponding to Eq. (16). $C _ { G }$ and InvokeRate C are fixed at the final CIFAR-10 seed-42 operating point; the unchanged additive terms are normalized to one work unit. Left: with $N \ = \ \overset { \cdot } { 2 , } 0 0 0$ fixed, the expected cost grows linearly with d. Right: with d = 64 fixed, it grows linearly with N.

In summary, LipCache offers a feasible path toward reliable cache-assisted inference at the edge, supported by theoretical analysis and experiments across multiple tasks. The remainder of this section discusses a number of open questions and extension directions along this path.

## 7.1 Future Work

Before outlining future work, we first clarify the scope and boundaries of the current study, so that subsequent research can advance in a targeted manner.

Task scale and type. The experiments in this paper validate the viability of the certification mechanism on three 10-class image classification tasks, designed to isolate the sensitivity of the certified radius to task geometry. Extension to larger-scale benchmarks such as full ImageNet (1000 classes) and CIFAR-100 is a natural next step.

Evaluation distribution. All experiments are conducted on i.i.d. test sets, targeting a controlled validation of the certification mechanism itself. Temporal redundancy, concept drift, and consecutive-frame reuse rates in streaming deployment involve sequential correlations and require dedicated benchmarks together with a temporal analysis of the hit rate. The perturbation experiments in Section 6.8 can serve as an upper-bound reference for streaming deployment, but a full temporal validation is left as future work.

Energy boundary. The goal of this paper is cache-reuse efficiency and inference acceleration, rather than end-toend energy reduction. Reducing the number of MainNet invocations mechanically reduces energy consumption, but a complete energy assessment would require hardware-inthe-loop measurement, which lies outside the methodological scope of this paper.

Given these boundaries, although LipCache already achieves a good balance among accuracy, latency, and throughput, the current design still centers on cache-reuse efficiency and inference acceleration, rather than end-to-end energy reduction. For resource-constrained edge deployment, reducing the number of main-model invocations is only part of the problem. A natural next step is therefore to introduce a spiking neural network (SNN) variant of GuardNet on the guard side, so that the system can further reduce energy consumption while preserving discriminative performance and certified cache-reuse validity. This subsection first reports an already-implemented exploratory SNN prototype and its preliminary observations, and then, building on it, discusses broader open directions.

## 7.2 Exploration: An SNN-Based GuardNet

The above discussion is not purely speculative. An SNNbased GuardNet variant has been implemented to evaluate its structural compatibility with the LipCache pipeline. The current design retains the same teacher–student training philosophy as the ANN GuardNet, but replaces static activations with multi-step spiking dynamics, so that the guard model may benefit from event-driven computation in future low-power deployments.

Figure 15 summarizes the architecture of the current SNN-based GuardNet. Each stage follows a “convolution– batch normalization–LIF neuron–average pooling” pattern; higher-layer feature maps are further projected through linear feature layers and additional LIF units, then averaged over time and mapped back into a bounded feature space. The implementation explicitly exposes the number of time steps T, the membrane constant τ , the input scaling, and an optional Poisson-style input encoding, making the model suitable for further analysis of temporal-encoding behavior.

![](images/5915e0e63b059439454797ce3c4ae5311c3052e65e1b865dc0c943f8ba9850b9.jpg)  
Fig. 15. Architecture of the current SNN-based GuardNet. The model replaces the static activations of the original ANN GuardNet with multi-step spiking dynamics, while still producing compact features for GuardNet-side classification and future cache lookup.

Preliminary observations on CIFAR-10 show that the SNN-based GuardNet is trainable and inherits part of the semantic structure learned by the ANN teacher, but they also clearly indicate that the current version is not yet sufficient to replace the ANN guard in the main LipCache pipeline. Specifically, the SNN-based GuardNet reaches a classification accuracy of 74.26% and an agreement rate of 74.78% with MainNet, whereas under the same feature dimensionality the ANN GuardNet reaches 76.84% accuracy and 77.13% agreement. In the current software implementation, the SNN-based GuardNet is also slower— the measured per-sample latency is 6.97 ms versus 0.50 ms for the ANN GuardNet, although the two checkpoints are comparable in size (338.8 KB vs. 355.9 KB). Figure 16 summarizes this preliminary comparison.

These results suggest a balanced reading. On the positive side, they indicate that the GuardNet learning framework of LipCache is compatible with more than one model family: the guard does not necessarily have to remain an ANN. At the same time, the current SNN study should be understood as exploratory evidence rather than a deploymentlevel conclusion—the SNN-based GuardNet has not yet reached the predictive quality required for reliable cache hits, and its measured latency was obtained on a conventional PyTorch/SpikingJelly software stack rather than on event-driven hardware. The current evidence therefore supports feasibility, not superiority.

![](images/09bc8d3c3735a6a4dadf7a28550f95082822210765ec2dc5fc19d01fedbb1c32.jpg)  
Fig. 16. Preliminary comparison on CIFAR-10 between the ANN-based GuardNet and the currently distilled SNN-based GuardNet: (a) classifica tion accuracy, (b) consistency with MainNet, (c) per-sample latency, (d) checkpoint size.

Based on these preliminary observations, we outline below a number of open questions in future work.

First, energy-efficiency evaluation requires a unified and physically meaningful accounting rule. Directly comparing a heuristic spike-count estimate of an SNN with a FLOPstyle approximation of an ANN is insufficient. Both models should be analyzed under an operation-level formulation— ANN inference is measured in multiply–accumulate operations, and event-driven SNN inference is measured in accumulate operations [38], [39].

Second, the SNN-based GuardNet itself needs significant improvement before it can be integrated as a reliable guard. Promising directions include stronger distillation objectives, better temporal-encoding schemes, explicit marginaware training tailored to spiking features, and meaningful architectural constraints capable of recovering the Lipschitz control used by the ANN GuardNet.

Third, the resulting energy and performance claims should be validated on neuromorphic hardware, including Loihi, TrueNorth, or SpiNNaker-class systems [40], [41], [42]. Such studies should go beyond coarse spike statistics and explicitly account for synaptic events, routing overhead, memory access, and board-level power measurement.

Finally, an improved SNN-based GuardNet should be integrated into the complete LipCache inference pipeline, so that its impact on hit rate, fallback rate, end-to-end latency, and overall throughput can be evaluated at the fullsystem level rather than only at the standalone level.

Taken together, these directions point to a broader research path: LipCache evolving from a cache-assisted acceleration framework into an energy-aware edge inference system. If GuardNet modeling, cache design, and hardware deployment can be jointly optimized, LipCache may ultimately become a unified solution that balances accuracy, latency, throughput, and energy.

## 8 ACKNOWLEDGMENT

Z. Xiang conceived the LipCache idea and developed the theoretical proofs. Y. Chen and F. Ying conducted the experiments, produced the figures, H. Zhao contributed to the framework design. Z. Xiang and Y. Chen proposed the SNNbased GuardNet, and Y. Chen carried out the preliminary experiments. B. Zhou and S. Dustdar played an important role in problem modeling and feasibility validation. The manuscript was polished with the assistance of generative-AI tools (GLM 5.2<sup>1</sup> and DeepSeek V4<sup>2</sup>).

## REFERENCES

[1] X. Xu, Y. Ding, S. X. Hu, M. Niemier, J. Cong, Y. Hu, and Y. Shi, “Scaling for edge inference of deep neural networks,” Nature Electronics, vol. 1, no. 4, pp. 216–222, 2018.

[2] M. M. H. Shuvo, S. K. Islam, J. Cheng, and B. I. Morshed, “Efficient acceleration of deep learning inference on resource-constrained edge devices: A review,” Proceedings of the IEEE, vol. 111, no. 1, pp. 42–91, 2023.

[3] H.-I. Liu, M. Galindo, H. Xie, L.-K. Wong, H.-H. Shuai, Y.-H. Li, and W.-H. Cheng, “Lightweight deep learning for resourceconstrained environments: A survey,” ACM Computing Surveys, vol. 56, no. 10, pp. 1–42, 2024.

[4] X. Wang, Z. Tang, J. Guo, T. Meng, C. Wang, T. Wang, and W. Jia, “Empowering edge intelligence: A comprehensive survey on ondevice ai models,” ACM Computing Surveys, vol. 57, no. 9, pp. 1–39, 2025.

[5] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “Mobilenetv2: Inverted residuals and linear bottlenecks,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 4510–4520.

[6] X. Zhang, X. Zhou, M. Lin, and J. Sun, “Shufflenet: An extremely efficient convolutional neural network for mobile devices,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 6848–6856.

[7] S. Teerapittayanon, B. McDanel, and H. T. Kung, “Branchynet: Fast inference via early exiting from deep neural networks,” in 2016 23rd International Conference on Pattern Recognition (ICPR), 2016, pp. 2464–2469.

[8] S. S. Ogden, G. Gilman, R. J. Walls, and T. Guo, “Many models at the edge: Scaling deep inference via model-level caching,” in IEEE International Conference on Autonomic Computing and Self-Organizing Systems, ACSOS 2021, Washington, DC, USA, September 27 - Oct. 1, 2021, E. El-Araby, V. Kalogeraki, D. Pianini, F. Lassabe, B. Porter, S. Ghahremani, I. Nunes, M. Bakhouya, and S. Tomforde, Eds. IEEE, 2021, pp. 51–60.

[9] D. Kang, J. Emmons, F. Abuzaid, P. Bailis, and M. Zaharia, “No-Scope: Optimizing deep CNN-based queries over video streams at scale,” in Proceedings of the VLDB Endowment, vol. 10, no. 11, 2017, pp. 1586–1597.

[10] M. Xu, M. Zhu, Y. Liu, F. X. Lin, and X. Liu, “Deepcache: Principled cache for mobile deep vision,” in Proceedings of the 24th Annual International Conference on Mobile Computing and Networking, 2018, pp. 129–144.

1. https://www.bigmodel.cn   
2. https://chat.deepseek.cn

[11] Y. Li, C. Zhang, S. Han, L. L. Zhang, B. Yin, Y. Liu, and M. Xu, “Boosting mobile CNN inference through semantic memory,” in Proceedings of the 29th ACM International Conference on Multimedia, 2021, pp. 2362–2371.

[12] M. Parger, C. Tang, C. D. Twigg, C. Keskin, R. Wang, and M. Steinberger, “Deltacnn: End-to-end cnn inference of sparse frame differences in videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 12 497–12 506.

[13] H. Sedghi, V. Gupta, and P. M. Long, “The singular values of convolutional layers,” in International Conference on Learning Representations, 2019.

[14] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, “Spectral normalization for generative adversarial networks,” in International Conference on Learning Representations, 2018.

[15] K. Sun, X. Wang, X. Miao, and Q. Zhao, “A review of AI edge devices and lightweight CNN and LLM deployment,” Neurocomputing, vol. 614, p. 128791, 2025.

[16] X. Zhang, R. Razavi-Far, H. Isah, A. David, G. Higgins, and M. Zhang, “A survey on deep learning in edge–cloud collaboration: Model partitioning, privacy preservation, and prospects,” Knowledge-Based Systems, vol. 310, p. 112965, 2025.

[17] J. Fang, Y. An, Y. Liu, Z. Teng, X. Zhai, H. Tang, and H. Chen, “MAE: Collaborative inference acceleration with efficient DNN partitioning and resource allocation in resource-constrained edge computing,” Computer Networks, vol. 278, p. 112073, 2026.

[18] A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, and H. Adam, “Searching for mobilenetv3,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 1314–1324.

[19] G. Huang, D. Chen, T. Li, F. Wu, L. van der Maaten, and K. Q. Weinberger, “Multi-scale dense networks for resource efficient image classification,” in International Conference on Learning Representations, 2018.

[20] M. Hassan, S. Davy, M. Zawish, O. B. Zuber, and N. Ashraf, “Neucodex: Edge-cloud co-inference with spike-driven compression and dynamic early-exit,” in International Conference on Machine Learning and Applications, ICMLA 2025, Boca Raton, FL, USA, December 3-5, 2025. IEEE, 2025, pp. 784–789.

[21] S. Venugopal, M. Gazzetti, Y. Gkoufas, and K. Katrinis, “Shadow puppets: Cloud-level accurate AI inference at the speed and economy of edge,” in USENIX Workshop on Hot Topics in Edge Computing (HotEdge 18). Boston, MA: USENIX Association, 2018.

[22] Z. Huang, F. Dong, X. Guo, and D. Yin, “Fasei: Fast serverless edge inference with synergistic lazy loading and layer-wise caching,” in 44th IEEE International Conference on Computer Communications, INFOCOM 2025, London, United Kingdom, May 19-22, 2025. IEEE, 2025.

[23] A. K. Sinthia, N. I. Mahbub, M. N. Sultan, and E. Huh, “Cachemoe: Task-aware expert model caching for multitask inference in distributed edge iot networks,” IEEE Internet Things J., vol. 12, no. 24, pp. 55 725–55 741, 2025.

[24] Q. Chen, X. Chen, and K. Huang, “Slimcaching: Edge caching of mixture-of-experts for distributed inference,” IEEE Trans. Mob. Comput., vol. 25, no. 7, pp. 10 924–10 938, 2026.

[25] Y. Wu, H. Zhang, and H. Huang, “RetrievalGuard: Provably robust 1-nearest neighbor image retrieval,” in Proceedings of the 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 162. PMLR, 2022, pp. 24 266– 24 279.

[26] A. Araujo, A. J. Havens, B. Delattre, A. Allauzen, and B. Hu, “A unified algebraic perspective on lipschitz neural networks,” in The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

[27] H. Zhang, T. Weng, P. Chen, C. Hsieh, and L. Daniel, “Efficient neural network robustness certification with general activation functions,” in Advances in Neural Information Processing Systems 31, 2018.

[28] J. Cohen, E. Rosenfeld, and J. Z. Kolter, “Certified adversarial robustness via randomized smoothing,” in Proceedings of the 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97, 2019, pp. 1310–1320.

[29] Y. Huang, H. Zhang, Y. Shi, J. Z. Kolter, and A. Anandkumar, “Training certifiably robust neural networks with efficient local lipschitz bounds,” in Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, M. Ranzato,

A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, Eds., 2021, pp. 22 745–22 757.

[30] M. Fazlyab, T. Entesari, A. Roy, and R. Chellappa, “Certified robustness via dynamic margin maximization and improved lipschitz regularization,” in Advances in Neural Information Processing Systems, 2023.

[31] K. Hu, K. Leino, Z. Wang, and M. Fredrikson, “A recipe for improved certifiable robustness,” in The Twelfth International Conference on Learning Representations, 2024.

[32] Z. Wang, B. Hu, A. J. Havens, A. Araujo, Y. Zheng, Y. Chen, and S. Jha, “On the scalability and memory efficiency of semidefinite programs for lipschitz constant estimation of neural networks,” in The Twelfth International Conference on Learning Representations, 2024.

[33] Y. Xu and S. Sivaranjani, “Eclipse: Efficient compositional lipschitz constant estimation for deep neural networks,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[34] A. Krizhevsky, “Learning multiple layers of features from tiny images,” University of Toronto, Tech. Rep., 2009.

[35] Y. Le and X. Yang, “Tiny imagenet visual recognition challenge,” CS231N Course Project, Stanford University, 2015.

[36] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng, “Reading digits in natural images with unsupervised feature learning,” in NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011.

[37] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[38] C. Lee, S. S. Sarwar, P. Panda, G. Srinivasan, and K. Roy, “Enabling spike-based backpropagation for training deep neural network architectures,” Frontiers in Neuroscience, vol. 14, p. 119, 2020.

[39] M. Horowitz, “1.1 computing’s energy problem (and what we can do about it),” in 2014 IEEE International Solid-State Circuits Conference Digest of Technical Papers (ISSCC), 2014, pp. 10–14.

[40] M. Davies, N. Srinivasa, T.-H. Lin, G. Chinya, Y. Cao, S. H. Choday, G. Dimou, P. Joshi, N. Imam, S. Jain, Y. Liao, C.-K. Lin, A. Lines, R. Liu, D. Mathaikutty, S. McCoy, A. Paul, J. Tse, G. Venkataramanan, Y.-H. Weng, A. Wild, Y. Yang, and H. Wang, “Loihi: A neuromorphic manycore processor with on-chip learning,” IEEE Micro, vol. 38, no. 1, pp. 82–99, 2018.

[41] P. A. Merolla, J. V. Arthur, R. Alvarez-Icaza, A. S. Cassidy, J. Sawada, F. Akopyan, B. L. Jackson, N. Imam, C. Guo, Y. Nakamura, B. Brezzo, I. Vo, S. K. Esser, R. Appuswamy, B. Taba, A. Amir, M. D. Flickner, W. P. Risk, R. Manohar, and D. S. Modha, “A million spiking-neuron integrated circuit with a scalable communication network and interface,” Science, vol. 345, no. 6197, pp. 668–673, 2014.

[42] T. Matsuo, “Neuromorphic processors for SNN-based edge AI: A review,” Micromachines, vol. 15, no. 1, p. 21, 2024.

![](images/0a4f673bd08bac182adf54b91beac88d622b0c656f3a89c5d8b6753894a5d66c.jpg)

Zhengzhe Xiang received the B.S. and Ph.D. degrees in computer science and technology from Zhejiang University, Hangzhou, China. He was a Visiting Scholar with Shanghai Jiao Tong University, Shanghai, China, in 2022. He is currently an Associate Professor with Hangzhou City University, Hangzhou, China. His research interests focus on service computing and edge computing. He serves as a reviewer for several international journals, such as IEEE Transactions on Mobile Computing, IEEE Transactions on Services Computing, IET Communications, and Digital Communications and Networks. He also serves as a program committee member for many international conferences.

![](images/b681fff5db39dda549c26d134c054123f8b14e75289074b989234914aca30090.jpg)

Yinlin Chen is currently pursuing the M.S. degree with the School of Computer and Computational Sciences, Hangzhou City University, Hangzhou, China. His research interests include edge computing, large language model applications, and distributed intelligent systems, with a focus on application optimization, task offloading, and intelligent resource scheduling for large language models in resource-constrained edge computing environments.

![](images/34c5d1bbee53fb1db1456ee4007fe7583a7854666ab4b9a88623612d39013600.jpg)

Fuli Ying is currently pursuing the M.S. degree with the School of Computer and Computational Sciences, Hangzhou City University, Hangzhou, China. Her research interests span mobile edge computing, deep-learning architectures, and image processing, with a focus on developing efficient algorithms for distributed computing environments and intelligent visual analytics systems.

![](images/b1a4772994d86c47a488b5df131ad6c0f738d3c81b398d4f3b6527b05e0a956f.jpg)

![](images/1ae9b560ed10be4bd1ab1b596c7cf7c1cd6869c1ae61c443c080bfd00e735e85.jpg)

Binbin Zhou received the Ph.D. degree in computer science from Zhejiang University, Hangzhou, China, in 2021. She is currently an Associate Professor with the School of Computer Science and Computing, Hangzhou City University, Hangzhou, China. Her main research areas include spatiotemporal deep learning, multimodal fusion, artificial intelligence, and brain-inspired computing.

![](images/14ad3889bfffb87ccc05d0896ba23ba9b4b4de6ad34485247ed4bd3444cf9adc.jpg)

Hailiang Zhao (Member, IEEE) received the PhD degree in Computer Science and Technology in 2024. He is currently an assistant professor with the School of Software Technology, Zhejiang University, China. His research interests include distributed computing and services computing. He has published several papers in flagship conferences and journals such as IEEE ICWS, IEEE Transactions on Parallel and Distributed Systems, IEEE Transactions on Mobile Computing, IEEE Transactions on Services

Computing, Proc. IEEE, etc. He was a recipient of the Best Student Paper Award of IEEE ICWS 2019.

Schahram Dustdar is a Full Professor of Computer Science and heads the Distributed Systems Group at TU Wien, Vienna, Austria. He is an ICREA Research Professor at UPF Barcelona, Spain. His research interests include distributed systems, edge intelligence, and complex and autonomous software systems. He is the Editor-in-Chief of Computing; an Associate Editor of ACM Transactions on the Web, ACM Transactions on Internet Technology, IEEE Transactions on Cloud Computing, and IEEE

Transactions on Services Computing. He also serves on the editorial boards of IEEE Internet Computing and IEEE Computer. He received the ACM Distinguished Scientist Award, the Distinguished Speaker Award, and the IBM Faculty Award. He is an elected member of Academia Europaea and served as the Chair of its Informatics Section from 2015 to 2022. He is an IEEE Fellow and an AAIA Fellow, and currently serves as the President of AAIA.
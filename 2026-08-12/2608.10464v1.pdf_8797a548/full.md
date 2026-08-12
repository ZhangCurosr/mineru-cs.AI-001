# Quantum Incremental Learning with Mixed State Prototypes

Yu Wu, Qianli Zhou, Xinyang Deng, Wen Jiang, Kang Hao Cheong, Witold Pedrycz, Life Fellow, IEEE

Abstract—Incremental learning models are required to learn new classes sequentially without catastrophic forgetting, while operating under parameter and memory constraints. In the Noisy Intermediate-Scale Quantum (NISQ) era, although quantum neural networks offer advantages in feature mapping, hardware limitations restrict circuit width. Furthermore, traditional quantum classifiers are constrained by the number of orthogonal basis states, limiting their capacity to accommodate a continually growing number of categories. Thus, we introduce a novel quantum incremental learning framework based on trainable mixed-state prototypes. Its original design incorporates new classes by adding class prototypes rather than increasing the circuit width of the shared quantum backbone. The use of mixed-state prototypes is another key contribution, since they have representation capabilities to represent information than a single pure-state prototype. And the decomposable mixed-state calculation provides lower production costs and a convenient Hilbert–Schmidt (HS) distance metric for classification. Simulation results show that our model achieves high-dimensional feature concentration using a minimal number of qubits, while demonstrating lower computational complexity and robust representation in incremental learning tasks compared with classical baselines.

Index Terms—Quantum neural network, incremental learning, quantum metric learning, quantum mixed states, continual learning.

## I. INTRODUCTION

D <sup>EEP</sup> <sup>learning</sup> <sup>has</sup> <sup>achieved</sup> <sup>remarkable</sup> <sup>success</sup> <sup>under</sup> <sup>the</sup>assumption that all training data is available at once. assumption that all training data is available at once. However, real-world data typically arrives in a continuous, ever-changing stream. Iteratively retraining a model from scratch to integrate new classes is computationally exorbitant and inefficient for resource-constrained applications [1]. Consequently, incremental learning has emerged to continuously update models with new knowledge while minimizing storage and computing overhead [2], [3].

In incremental learning, the fundamental challenge is catastrophic forgetting, where neural networks erase previously learned representations upon adapting to new tasks [4], [5]. Early works introduced memory replay to explicitly rehearse a small buffer of historical exemplars [6]–[8], alongside parameter regularization and knowledge distillation to constrain critical weight updates [9]–[11]. As storing raw data is often restricted by memory limits or privacy concerns, research subsequently explored exemplar-free statistical prototype matching [12], [13], extracting and freezing abstract representations to avoid iterative retraining [14]–[16]. More recently, dynamic networks have been proposed to mitigate forgetting by expanding architectural branches [17], [18]. Despite these diverse efforts, classical incremental learning still confronts the bottleneck of computational overhead and residual representational drift, which limits their real-world scalability [1].

Harnessing quantum mechanics, quantum computing introduces a fundamentally distinct paradigm for information processing, catalyzing the rapid exploration of quantum machine learning [19], [20]. In the NISQ era, hybrid architectures based on variational quantum circuits (VQC) have shown immense promise [21]. By projecting classical data into an exponentially expansive Hilbert space, VQCs facilitate the efficient manipulation of highly non-linear features [22]– [24]. Thus, quantum models achieve competitive accuracy with significantly reduced parameters compared to classical networks [25]. Moreover, quantum entanglement captures non-local correlations, providing a physical foundation for enhancing model expressiveness [26], [27]. Recent studies have extended quantum neural architectures to tasks including human reliability analysis [28], multimodal feature fusion [29], [30] and protein-folding prediction [31]. Furthermore, Shi et al. [32] creatively proposed a quantum self-attention mechanism based on quantum logic similarity, which can efficiently and accurately calculate attention scores and outputs while reducing intermediate measurements, further exploring the development potential of quantum machine intelligence in the NISQ era.

Despite these theoretical merits, classical incremental learning strategies cannot be directly quantized, as the structural differences between quantum and classical networks [33]. In classical deep learning, accommodating new classes is often as straightforward as expanding the final linear layer. However, a large class of commonly used quantum classifiers constructs class scores from a fixed set of basis-state probabilities or expectation values [34], [35]. As the label space grows, these classifiers may require an expanded measurement or readout rule, additional observables, or more class-dependent parameters. In basis-state encoding schemes, accommodating additional classes may also require additional qubits. Crucially, unlike modular classical expansion, appending a qubit fundamentally alters the global quantum state and the underlying Hilbert space. This structural shift disrupts previously learned entanglement correlations, exacerbating catastrophic forgetting [36]. Furthermore, blindly increasing the circuit width or depth inevitably triggers the barren plateau phenomenon, rendering the quantum network untrainable [37].

![](images/11d5e728c338d7ccafee0e9a081c31354a12f527838b7b20119ad81533febc76.jpg)  
Fig. 1. Illustration of incremental learning strategies. (a) Exemplar replay; (b) Regularization; (c) Prototype classification.

To circumvent this structural expansion dilemma, quantum metric learning emerges as a promising paradigm. By classifying instances based on state distances rather than relying on explicit parameterized classification heads, it naturally bypasses the need for circuit widening. Its effectiveness heavily depends on exploring advanced quantum information representations [38], [39]. In quantum feature mapping, pure states encode individual inputs into single unit state vectors in the Hilbert space [40]. Consequently, they lack the statistical capacity to represent a diverse distribution compared to mixed states. By formulating information as a convex combination of orthogonal states, mixed states encapsulate data randomness and complex intra-class variance [41]. Compared to classical information representations, this quantum method exponentially broadens the representation dimensionality. Furthermore, it provides physically grounded distance metrics, such as the HS distance, trace distance, and Bures distance [42]. Ultimately, these comprehensive advantages help to solve the problems in incremental learning tasks. [43].

Motivated by the representational power of mixed states and the physical constraints of quantum circuits, this paper proposes a novel quantum incremental learning framework based on mixed-state prototypes. To transcend the capacity bottleneck of traditional quantum classifiers, our design shifts from explicit softmax regression classification to an implicit, distance-based prototype matching mechanism. Moreover, recognizing the difficulty of dynamically altering quantum network, we design a hybrid architecture: we append a lightweight, expandable classical module (i.e., an MLP) to manage the incremental class adjustments without disrupting the underlying quantum states. The main contributions of this work are summarized as follows:(1) A quantum prototype framework with an expandable module that avoids the capacity constraint associated with orthogonal basis states. (2) Mixedstate prototypes via dynamic weighting of pure states avoid redundant qubits and provide PCA-like noise-filtering properties. (3) Experiments on static and incremental classification tasks demonstrate high-dimensional semantic concentration, exhibiting parameter advantages and noise robustness.

The rest of this paper is organized as follows. Section II introduces the preliminaries of incremental learning and mixedstate compiling. Section III details the construction of the mixed-state quantum prototype classifier. Section IV presents the overall incremental learning framework at length. Section V provides the experimental evaluation, and Section VI concludes the paper and discusses some directions.

## II. PRELIMINARIES

This section provides a concise overview of the key concepts in incremental learning and quantum mixed state compilation that form the foundation of our proposed framework.

## A. Incremental learning strategies

Incremental learning requires a model to sequentially learn new classes from a continuous data stream without catastrophically forgetting previously acquired knowledge. To strike a balance between the plasticity required to learn new tasks and the stability needed to retain old ones, current classical incremental learning frameworks generally rely on three fundamental strategies, as Fig. 1 shows.

1) Exemplar replay strategy: Replay mechanisms mitigate forgetting by preserving a bounded memory of old data. A prominent example is iCaRL [6], which utilizes a herding algorithm to select and store a small set of highly representative samples, for each observed class. During the incremental training phase, these exemplars are replayed alongside new class instances. This allows the model to retrospectively review past knowledge, effectively preventing the decision boundaries from catastrophic forgetting.

2) Regularization techniques: Instead of directly reusing raw data, regularization methods impose mathematical constraints on the network’s optimization trajectory. Parameterbased regularization approaches like EWC [9] evaluate the importance of individual network weights for past tasks and penalize drastic deviations in these critical parameters. Furthermore, knowledge distillation techniques such as LwF [10] mitigate the representational drift of the network. They force the updated model to mimic the output logits or intermediate feature representations of the frozen old model, thereby implicitly preserving the topological structure of past knowledge.

3) Prototype-based classification: Due to the data imbalance between old and new classes, traditional classifiers suffer from task-recency bias in incremental learning. To address this, prototype-based inference relies on distance metrics within the embedding space [43]. Methods like the nearest-mean-ofexemplars classifier in iCaRL or centroid-based classifiers in FeTrIL and FeCAM [14], [15] represent each class by a global prototype. An input instance is classified by assigning it to the nearest class prototype. This decoupling of feature extraction from explicit classification effectively alleviates the inherent prediction bias and serves as a theoretical cornerstone for our proposed quantum prototype design.

## B. Quantum mixed state compiling

Quantum state compilation aims to learn a parameterized quantum circuit that can prepare a reliable approximation of target state. While early works primarily focused on pure states, compiling mixed states provides a more generalized framework for handling classical probabilities, noise, and complex data distributions in the NISQ era [44].

1) Density matrix and mixed state representation: In quantum mechanics, a closed quantum system with complete information is described by a pure state, denoted by a state vector |ψ⟩ in a Hilbert space. However, in practical open quantum systems, particularly within the NISQ era, a principal system inevitably interacts with its environment and undergoes decoherence. Tracing out the environment produces a reduced mixed state. A mixed state is described with a density matrix $\rho .$ It is a positive-semidefinite Hermitian operator with a unit trace $\operatorname { T r } ( \rho ) ~ = ~ 1$ . According to the spectral decomposition theorem, any mixed state $\rho$ can be expressed as a probabilistic ensemble of orthogonal pure states:

$$
\rho = \sum _ { i } \lambda _ { i } | v _ { i } \rangle \langle v _ { i } | ,\tag{1}
$$

where $\lambda _ { i } \geq 0$ are the eigenvalues representing the classical probability of the system being in the corresponding pure eigenstate $\left| v _ { i } \right.$ , satisfying $\textstyle \sum _ { i } \lambda _ { i } = 1$

![](images/4e8c53c3fdd61e32f049ec8c79a9fa0f2565370c530341dba682838a3f3abd9e.jpg)  
Fig. 2. HS distance measurement via CCPS. VQC and SWAP test is shared.

2) Convex Combination of Pure States ansatz: Statepurification ansatzes prepare a larger pure state and trace out ancillary registers, which increases the required circuit width. The convex combination of pure states (CCPS) ansatz avoids this overhead by representing a rank-R mixed state as a probabilistic mixture of transformed basis states [41]:

$$
\sigma _ { \mathrm { C C P S } } ( \alpha , R ) : = \sum _ { i = 0 } ^ { R - 1 } p _ { \phi } ( i ) U _ { \theta } | i \rangle \langle i | U _ { \theta } ^ { \dagger } ,\tag{2}
$$

where $U _ { \theta }$ is a parameterized unitary circuit, |i⟩ denotes a selected computational basis state, and $p _ { \phi } ( i )$ is a trainable probability satisfying $\begin{array} { r } { \sum _ { i } p _ { \phi } ( i ) = 1 } \end{array}$ . The trainable variables α include both the circuit parameters θ and the probability parameters ϕ.

Under HS-distance optimization, a rank-R CCPS state provides a variational low-rank approximation of the target mixed state, yielding a PCA-like procedure that retains its leading components [41]. It controls the prototype rank without introducing ancillary qubits and can suppress low-contribution components. These properties make CCPS suitable for constructing the class prototypes in our framework.

3) Distance measurement via HS distance and SWAP test: The HS distance is used to measure the distance between two quantum states, and also to describe the quantum distance in quantum geometry [42]. Using the Frobenius norm, the squared HS distance is $\begin{array} { r } { \begin{array} { l } { \mathcal { L } } \end{array} = \begin{array} { r } { \Vert \rho \mathrm { ~ - ~ } \sigma \Vert _ { F } ^ { 2 } } \end{array} } \end{array}$ , which can be decomposed into state purities and their quantum overlap:

$$
\begin{array} { r } { \mathcal { L } = \mathrm { T r } [ \rho ^ { 2 } ] + \mathrm { T r } [ \sigma ^ { 2 } ] - 2 \mathrm { T r } [ \rho \sigma ] . } \end{array}\tag{3}
$$

CCPS’s method that can decouple the calculation into classical probability evaluations and a series of quantum SWAP tests between mixed and pure states.

Firstly, due to the orthogonality of the computational basis states $| i \rangle$ , the purity of the CCPS state simplifies entirely to the collision probability of its classical weights:

$$
\mathrm { T r } [ \sigma _ { \mathrm { C C P S } } ^ { 2 } ] = \sum _ { i = 0 } ^ { R - 1 } p _ { \phi } ( i ) ^ { 2 } .\tag{4}
$$

Secondly, to expand the cross term $\operatorname { T r } [ \rho \sigma _ { \mathrm { C C P S } } ]$ , let $| \psi _ { i } ( \theta ) \rangle =$ $U _ { \theta } | i \rangle$ denote the i-th parameterized pure state generated by the ansatz. The cross term can be rewritten as:

$$
{ \mathrm { T r } } [ \rho \sigma _ { \mathrm { C C P S } } ] = \sum _ { i = 0 } ^ { R - 1 } p _ { \phi } ( i ) { \mathrm { T r } } \left[ \rho | \psi _ { i } ( \theta ) \rangle \langle \psi _ { i } ( \theta ) | \right] .\tag{5}
$$

The term $\mathcal { F } _ { i } ( \theta ) \equiv \mathrm { T r } \left[ \rho | \psi _ { i } ( \theta ) \rangle \langle \psi _ { i } ( \theta ) | \right]$ represents the quantum state overlap, between the target mixed state $\rho$ and the pure state $| \psi _ { i } ( \theta ) \rangle$ ⟩. And each $\mathcal { F } _ { i } ( \boldsymbol { \theta } )$ can be evaluated on quantum hardware using a SWAP test. The circuit is shown in Fig. 3.

![](images/5e586a4a39a2f179be6de918f3454338ca33f84051cf29f9bc18a13524272cc5.jpg)  
Fig. 3. SWAP test circuit for 2-qubit states $\rho$ and $\phi .$

Specifically, the SWAP test requires one |0⟩ initialized ancillary qubit, utilizing a controlled-swap gate sandwiched between two Hadamard gates. By measuring the ancilla, the probability of observing the state |0⟩, denoted as $P ( 0 )$ , is represented as:

![](images/a3d80c8c91456ecd27fd7cd8c5a109d9d62d408cff575fdae40f014afcfe1242.jpg)  
Fig. 4. Overview of the proposed framework. Stage 1: Backbone training. Classical inputs are compressed and amplitude-encoded into a QCNN. Th backbone is optimized via an auxiliary head. Stage 2: Prototype Fitting. Basis states are transformed to construct mixed-state prototypes, fitting to the class quantum representations. Stage 3: Memory Update. Samples yielding the minimum HS distance to the prototypes are stored for future incremental tasks.

$$
P ( 0 ) = \frac { 1 } { 2 } + \frac { 1 } { 2 } \mathrm { T r } \left[ \rho | \psi _ { i } ( \theta ) \rangle \langle \psi _ { i } ( \theta ) | \right] .\tag{6}
$$

Thus, the required quantum state overlap can be directly expressed from the measurement statistic as $\mathcal { F } _ { i } ( \theta ) = 2 P ( 0 ) - 1$

By substituting Eq. (4) and Eq. (5) back into Eq. (3), the overall HS distance is expressed as a combination of these single-parameter squares and state SWAP test results:

$$
\mathcal { L } _ { \mathrm { C C P S } } ( \boldsymbol { \theta } , \boldsymbol { \phi } ) = \mathrm { T r } [ \rho ^ { 2 } ] + \sum _ { i = 0 } ^ { R - 1 } p _ { \phi } ( i ) ^ { 2 } - 2 \sum _ { i = 0 } ^ { R - 1 } p _ { \phi } ( i ) \mathcal { F } _ { i } ( \boldsymbol { \theta } ) .\tag{7}
$$

During the CCPS’s optimization, the target mixed state $\rho$ is fixed. Therefore, the loss function is reduced to:

$$
\mathcal { L } _ { \mathrm { o p t } } ( \pmb { \theta } , \phi ) = \sum _ { i = 0 } ^ { R - 1 } p _ { \phi } ( i ) ^ { 2 } - 2 \sum _ { i = 0 } ^ { R - 1 } p _ { \phi } ( i ) \mathcal { F } _ { i } ( \theta ) .\tag{8}
$$

This decomposition is the main reason for using the HS distance in our framework. In dense-matrix simulation, evaluating the HS distance between two d×d states requires $O ( d ^ { 2 } )$ operations. The trace distance, fidelity, Bures distance, and quantum Jensen–Shannon divergence generally require $O ( d ^ { 3 } )$ operations because they involve spectral decompositions or matrix square roots. Therefore, the HS distance is more suitable for repeated prototype fitting and comparison.

## III. MIXED-STATE BASED QUANTUM PROTOTYPE CLASSIFIER

## A. Overview of the quantum prototype classifier

The core architecture of our proposed framework is a mixed-state quantum prototype classifier, as illustrated in Fig. 4. A large class of commonly used quantum classifiers constructs class scores from a fixed set of measured basisstate probabilities or expectation values. Their measurement or readout structures may need to be expanded as the label space grows. Our design instead performs distance-based prototype matching within a shared density-matrix space.

Formally, given an input instance $\textbf { x } \in { \mathcal { X } }$ , the quantum feature extraction backbone acts as a parameterized mapping function $F _ { q } ( \cdot ; \Theta _ { b b } )$ . It encodes and processes x to output an intermediate quantum mixed state, represented by a density matrix $\rho ( \mathbf { x } ) \in \mathbb { C } ^ { d \times d }$ , where $d = 2 ^ { 4 } = 1 6$ for our 4-qubit subsystem. In parallel we construct an independent mixedstate quantum prototype circuit for each category $c \in \mathcal { V }$ . This circuit generates a class prototype $\rho _ { c }$ residing in the identical $d \times d$ Hilbert space. The classification prediction $\hat { y }$ is then determined by identifying the class prototype that yields the minimum HS distance to the query instance:

$$
\hat { y } = \arg \operatorname* { m i n } _ { c \in \mathcal { V } } \mathcal { D } _ { c } ( \mathbf { x } ) = \arg \operatorname* { m i n } _ { c \in \mathcal { V } } \big \| \rho ( \mathbf { x } ) - \rho _ { c } \big \| _ { F } ^ { 2 } .\tag{9}
$$

Then the HS distance is decomposed into a convex combination of quantum SWAP test circuits, as elaborated in Section II-B3.

This design decouples representation learning from prototype classification. The backbone learns characteristic information, while the prototype circuit learns the characteristic distribution of specific categories within the same density matrix space. In the following subsections, we detail the construction and optimization of these core components.

## B. Quantum feature extraction backbone

The quantum feature extraction backbone serves as a hybrid classical-quantum bridge, mapping high-dimensional classical inputs into compact quantum density matrices. This backbone mainly consists of a classical pre-processing module and a subsequent quantum evolution network.

Initially, a classical neural network is employed to extract high-level semantic features from the raw input data. To bridge the dimensional gap between classical feature spaces and the quantum Hilbert space, the extracted features undergo a dimension reduction and normalization process. The resulting compact vector is then embedded into the initial pure state of the quantum system using amplitude encoding, translating classical data into probability amplitudes.

The embedded pure quantum state is processed by the quantum backbone, which typically employs a sequence of parameterized unitary operations interleaved with dimensionality reduction mechanisms, such as quantum pooling. The primary objective of these reduction operations is to achieve semantic concentration, extract robust features, and filter out uncorrelated noise. However, operations that discard redundant degrees of freedom, such as partial tracing, inevitably induce information loss regarding the global pure state. That means the retained subsystem is taken into a quantum mixed state.

Therefore, the emergence of a mixed-state representation is a physical consequence of performing feature abstraction and denoising within a quantum architecture. The resulting density matrix $\rho ( x )$ intrinsically encapsulates the structural uncertainty and hierarchical semantic information of the input, serving as a highly robust, noise-resilient foundational input for the subsequent prototype-based classification.

Algorithm 1 Classification via Weighted Swap Tests   
input x // query instance   
require $F _ { q } ( \cdot )$ // frozen quantum feature backbone   
require ${ \mathcal { V } } _ { v i s } ^ { ( t ) }$ // currently visible classes   
require $\left\{ U _ { c } , { \pmb w } _ { c } \right\} _ { c \in \mathcal { V } _ { v i s } ^ { ( t ) } }$ // trained prototype parameters   
$\rho ( x ) \gets F _ { q } ( x )$ // extract query density matrix   
for $c \in \mathcal { V } _ { v i s } ^ { ( t ) }$ do   
// compute weighted quantum overlap via SWAP tests:   
$\begin{array} { r } { \mathcal { O } _ { c } ( x ) \dot { \langle - \sum _ { i = 0 } ^ { k - 1 } w _ { c , i } \mathrm { T r } \left[ \rho ( x ) U _ { c } | i \rangle \langle i | \dot { U } _ { c } ^ { \dagger } \right] } } \end{array}$   
// calculate prediction logit with prototype purity offset:   
logi $\begin{array} { r } { \mathfrak { t } _ { c } ( x ) \gets \dot { \mathcal { O } } _ { c } ( x ) - \frac { 1 } { 2 } \sum _ { i = 0 } ^ { k - 1 } w _ { c , i } ^ { 2 } } \end{array}$   
end for   
$\hat { y } \gets \arg \operatorname* { m a x } _ { c \in \mathcal { V } _ { \eta ; \mathrm { ~ s ~ } } ^ { ( t ) } }$ logit (x) // nearest-prototype decision   
output predicted class label yˆ and logits $L = \{ \log \mathrm { i t } _ { c } ( x ) \}$

## C. Mixed quantum prototype preparation circuit

For each class, the model maintains an independent parameterized prototype circuit. The purpose of this circuit is to generate a mixed quantum state in the identical density-matrix space as the backbone’s quantum feature output.

Following the CCPS architecture, the prototype for class c is constructed as a probabilistic mixture of transformed basis states. By selecting a subset of k computational basis states |i⟩, the prototype density matrix $\rho _ { c }$ is formulated as:

$$
\rho _ { c } = \sum _ { i = 0 } ^ { k - 1 } w _ { c , i } U _ { c } | i \rangle \langle i | U _ { c } ^ { \dagger } ,\tag{10}
$$

where $U _ { c }$ is the VQC for class $c ,$ and $w _ { c , i }$ represents the corresponding weights. This convex mixture allows a single class prototype to encapsulate complex intra-class variations, thereby circumventing the representational bottleneck of forcing an entire diverse class into a rigid, singular pure state.

To ensure high expressivity, the circuit $U _ { c }$ is constructed using a hierarchical multi-layer structure. Within each layer, generic two-qubit unitary blocks are applied in an alternating entanglement topology. This configuration enables the prototype circuit to span a highly expressive transformation space, maintaining both representation capacity and parameter efficiency.

A challenge in optimizing the CCPS ansatz lies in the constraints of the mixture weights $w _ { c , i }$ , which must satisfy the constraints of a valid probability simplex $( \mathrm { i } . \mathrm { e } . , \ w _ { c , i } \ \geq \ 0 $ and $\textstyle \sum _ { i } w _ { c , i } = 1 )$ . The traditional approach enforces this by optimizing a subset of weights, deriving the remainder via subtraction, and applying penalty functions and truncations to prevent negative values. However, such constraints often lead to severe gradient instability and discontinuous optimization landscapes, which are detrimental to training. To address this, we introduce softmax parameterization for the mixture weights. We define a set of unconstrained trainable logits $s _ { c , i } ~ \in ~ \mathbb { R }$ for each class, and the valid probabilities can be represented as the softmax function:

$$
w _ { c , i } = \frac { \exp ( s _ { c , i } ) } { \sum _ { j = 0 } ^ { k - 1 } \exp ( s _ { c , j } ) } .\tag{11}
$$

Both theory and experiments have shown that softmax parameterization has a smoother and more stable optimization process.

## D. Prototype-based classification algorithm

Our prototype-based classifier operates on a decoupled twostage training paradigm. In the first stage, the quantum feature extraction backbone is optimized via an auxiliary head and subsequently frozen. In the second stage, the mixed-state prototypes are fitted without altering the backbone. During the evaluation phase, predictions are executed through a direct quantum overlap measurement.

The classifier determines the label by finding the nearest mixed-state prototype $\rho _ { c }$ among all currently visible classes $c \in \mathcal { V } _ { v i s } ^ { ( t ) }$ . While the objective is to minimize the HS distance $\mathcal { D } _ { c } ( x ) = \lVert \rho ( x ) - \rho _ { c } \rVert _ { F } ^ { 2 }$ . As discussed in Section II-B3, the measurement is reduced to a weighted sum of pure-state overlaps, since the query purity $\mathrm { T r } [ \bar { \rho } ( x ) ^ { 2 } ]$ is identical across all classes, and the prototype purity $\begin{array} { r } { \dot { \mathrm { T r } } [ \rho _ { c } ^ { 2 } ] = \sum _ { i } w _ { c , i } ^ { 2 } } \end{array}$ can be pre-computed classically:

$$
\mathcal { O } _ { c } ( x ) \equiv \mathrm { T r } [ \rho ( x ) \rho _ { c } ] = \sum _ { i = 0 } ^ { k - 1 } w _ { c , i } \mathrm { T r } [ \rho ( x ) | \psi _ { c , i }   \psi _ { c , i } ] ] ,\tag{12}
$$

where $| \psi _ { c , i } \rangle = U _ { c } | i \rangle$ is the i-th pure state component of the class prototype c.

The overlap term $\mathcal { F } _ { c , i } ( \boldsymbol { x } ) ~ = ~ \mathrm { T r } [ \rho ( \boldsymbol { x } ) | \psi _ { c , i } \rangle \langle \psi _ { c , i } | ]$ can be efficiently and directly evaluated on quantum hardware using SWAP test. Therefore, the overall matching score for class c is simply the weighted sum of these Swap Test results, using the trained classical probabilities $w _ { c , i }$

To integrate this quantum measurement with classical evaluation metrics, we formulate the classification logit as:

$$
\mathrm { l o g i t } _ { c } ( x ) = \mathcal { O } _ { c } ( x ) - \frac { 1 } { 2 } \sum _ { i = 0 } ^ { k - 1 } w _ { c , i } ^ { 2 } ,\tag{13}
$$

which is mathematically equivalent $\mathrm { t o } - \frac { 1 } { 2 } \mathcal { D } _ { c } ( x )$ (ignoring the constant query purity). The final prediction selects the class with the maximum logit. This hardware-friendly, weighted SWAP test evaluation procedure is detailed in Algorithm 1.

## IV. QUANTUM INCREMENTAL LEARNING FRAMEWORK

Building upon the mixed-state quantum prototype classifier introduced in Section III, this section details our complete incremental learning framework. The framework is designed to continuously accommodate new categories over sequential tasks while mitigating catastrophic forgetting through a density-aware replay mechanism.

## A. Problem formalization

In incremental learning, the model learns from a continuous stream of data delineated by a sequence of tasks $\mathcal { T } = \{ T _ { 0 } , T _ { 1 } , \dots , T _ { T } \}$ . At the initial task $T _ { 0 } ,$ , the model is trained on a base dataset $\mathcal { D } _ { 0 }$ comprising an initial set of classes $C _ { 0 }$ . For each subsequent incremental task $T _ { t } ~ ( t > 0 )$ , the model receives a new dataset $\mathcal { D } _ { t }$ containing instances from a disjoint set of novel classes $C _ { t }$

At any given task t, the currently visible class set is the union of all classes encountered so far: $\textstyle \mathcal { V } _ { v i s } ^ { ( t ) } ~ = ~ \bigcup _ { i = 0 } ^ { t } C _ { i }$ Due to strict memory constraints, the model cannot access previous datasets $\{ \mathcal { D } _ { 0 } , \ldots , \mathcal { D } _ { t - 1 } \}$ directly. Instead, it is only permitted to maintain a limited memory buffer M containing a small number of exemplars from the old classes. The purpose is to incrementally update the model using only $\mathcal { D } _ { t } \cup \mathcal { M }$ maximizing the classification accuracy over all visible classes ${ \mathcal { V } } _ { v i s } ^ { ( t ) }$ without catastrophically forgetting the old ones.

## B. Details of quantum Incremental Learning Framework

To instantiate the theoretical model introduced in Section III, we constructed the hybrid classical-quantum framework for incremental learning scenarios. This overall architecture is composed of four primary components: a classical preprocessing module, a quantum feature extraction backbone, an auxiliary MLP head for representation learning, and a set of mixed-state prototype networks.

![](images/80e2f1a70439c725fcff24f031ce1507b9f4af3a19a94d3b9057b6e5cb215052.jpg)  
Fig. 5. Overall flowchart of the quantum incremental learning framework. The process is decoupled into incremental training phase (Steps 1–3) and global inference phase (Step 4). During training, the quantum backbone is updated using new data and buffered exemplars, followed by mixed-state prototype fitting and memory updating. The auxiliary MLP head is discarded after backbone training.

1) Classical pre-processing module: The classical preprocessing module acts as the initial semantic extractor and the dimensional bridge between raw input data and the quantum computational space. Given an input instance $\mathbf { x } \in \mathcal { X }$ from the current incremental task, it is first augmented and normalized to $\mathbf { x } ^ { \prime } .$ Then the module maps the high-dimensional raw image into a dense semantic feature vector for quantum embedding.

Firstly, we employ a ResNet adapted to different input sizes, denoted as $f _ { r e s } ( \cdot ; \theta _ { r e s } )$ . After that, the network outputs a robust d<sub>res</sub>-dimensional feature vector:

$$
\mathbf { h } = f _ { r e s } ( \mathbf { x } ^ { \prime } ; \theta _ { r e s } ) \in \mathbb { R } ^ { d _ { r e s } } ,\tag{14}
$$

where $d _ { r e s } = 5 1 2$ in our implementation.

In order to map the dimension of h to that of the quantum embedding, we append a linear reducer. It compresses the 512-dim representation into $d _ { q } { \mathrm { - } } \mathrm { d i m }$ , where $d _ { q } = 2 ^ { n } = 2 5 6$ as we chose the quantum circuit with width $n \ = \ 8 .$ . The transformation is formulated as:

$$
\mathbf { z } = \mathrm { D r o p o u t } \left( \sigma ( W \mathbf { h } + \mathbf { b } ) \right) \in \mathbb { R } ^ { d _ { q } } ,\tag{15}
$$

where $W \in \mathbb { R } ^ { 2 5 6 \times 5 1 2 }$ is the weight matrix, and b $\in \mathbb { R } ^ { 2 5 6 }$ is the trainable bias, while $\sigma ( \cdot )$ denotes the ReLU function.

For quantum embedding, a valid pure quantum state must reside in a Hilbert space with a unit norm. Therefore, the $L _ { 2 ^ { - } }$

normalized vector a can be represented as:

$$
\mathbf { a } = \frac { \mathbf { z } } { \| \mathbf { z } \| _ { 2 } } , \quad \mathrm { s . t . } \sum _ { j = 0 } ^ { 2 ^ { n } - 1 } a _ { j } ^ { 2 } = 1 .\tag{16}
$$

Through amplitude embedding, this normalized classical vector is mapped onto the probability amplitudes of an n-qubit initial pure state:

$$
| \psi ( \mathbf { x } ) \rangle = \sum _ { j = 0 } ^ { 2 ^ { n } - 1 } a _ { j } | j \rangle ,\tag{17}
$$

where $| j \rangle$ represents the computational basis states spanning the $2 ^ { n }$ -dim Hilbert space.

2) Quantum feature extraction backbone: The initial pure state $| \psi ( \mathbf { x } ) \rangle$ prepared by the classical pre-processing module serves as the direct input to the quantum feature extraction backbone. Defining the initial density matrix as $\rho _ { i n } ~ =$ $| \psi ( \mathbf { x } ) \rangle \langle \psi ( \mathbf { x } ) |$ |, the backbone adopts a quantum convolutional neural network (QCNN) architecture to distill high-level quantum semantic features [45], [46].

The QCNN adopts the pyramidal structure of classical CNNs to the quantum paradigm. It alternately stacks quantum convolutional and pooling layers. In the quantum convolutional layer, the network emulates the principles of local receptive fields and weight sharing. It applies a parameterized global unitary operator $U _ { c } ( \theta _ { c } )$ , which is composed of multiple twoqubit unitary blocks acting on adjacent qubit pairs. The state evolution through the convolutional layer is governed by:

$$
\rho _ { c o n v } = U _ { c } ( \theta _ { c } ) \rho _ { i n } U _ { c } ^ { \dagger } ( \theta _ { c } ) .\tag{18}
$$

This operation entangles local qubits to capture complex spatial and feature correlations within the high-dimensional Hilbert space, mapping linearly inseparable classical data into an exponentially expansive quantum feature space. By leveraging quantum entanglement and weight sharing, the QCNN captures non-local correlations and constructs highly non-linear feature distribution. With a few dozen trainable parameters, it can avoid the parameter explosion typical of classical networks.

Following the convolution, a quantum pooling layer is applied to compress the feature representation and improve robustness. The pooling operation first applies parameterized controlled rotations $U _ { p } ( \theta _ { p } )$ between adjacent retained and discarded qubits, gathering information into the retained wires. Subsequently, a subset of qubits, S, is traced out. The retained subsystems are in a mixed state. Thus, the intermediate feature representation output by the QCNN is formulated as:

$$
\rho _ { p o o l } = \mathrm { T r } _ { S } \left[ U _ { p } ( \theta _ { p } ) \rho _ { c o n v } U _ { p } ^ { \dagger } ( \theta _ { p } ) \right] .\tag{19}
$$

In our 8-qubit implementation, the pooling layer traces out exactly half of the qubits, reducing the system to 4 wires. Thereafter, a group of quantum convolutional layers, $U _ { c 2 } ( \theta _ { c 2 } )$ continues to be stacked on this reduced 4-qubit subsystem. The final intermediate feature representation is thus formulated as:

$$
\rho ( \mathbf { x } ) = U _ { c 2 } ( \theta _ { c 2 } ) \rho _ { p o o l } U _ { c 2 } ^ { \dagger } ( \theta _ { c 2 } ) .\tag{20}
$$

Consequently, $\rho ( \mathbf { x } )$ emerges as a $1 6 \times 1 6$ complex-valued density matrix. The partial trace operation in the pooling layer naturally abstracts structural uncertainty and discards redundant noise. The quantum state after pooling is a density operator, which can characterize information uncertainty with more degrees of freedom than a pure state.

3) Auxiliary MLP head for representation learning: To ensure that the intermediate density matrix $\rho ( \mathbf { x } )$ is highly discriminative over the currently visible classes, we append an auxiliary classical MLP head during the initial representation learning stage. This design is crucial when the number of visible classes $C _ { v i s }$ exceeds the dimension of the measured basis states $( \mathbf { i . e . , \ 2 ^ { 4 } \ = \ 1 6 } )$ . This scenario bottlenecks traditional explicit quantum classifiers, as the explicit method directly measures the basis-state probabilities as the logit outputs.

Specifically, we perform projective measurements on the 4 retained qubits in the computational basis. The resulting 16- dim probability distribution vector $\mathbf { p } ( \mathbf { x } ) \in \mathbb { R } ^ { 1 6 }$ is extracted by calculating the diagonal elements of the output density matrix:

$$
p _ { k } ( \mathbf { x } ) = \operatorname { T r } \left[ \rho ( \mathbf { x } ) | k \rangle \langle k | \right] , \quad k \in \{ 0 , 1 , \ldots , 1 5 \} .\tag{21}
$$

This 16-dimensional probability vector is then mapped to the class logits $\hat { \mathbf { y } } ( \mathbf { x } ) ~ \in ~ \mathbb { R } ^ { C _ { v i s } }$ via a two-layer MLP. The mapping is formulated as:

$$
\hat { \mathbf { y } } ( \mathbf { x } ) = W _ { o u t } \sigma ( W _ { h i d } \mathbf { p } ( \mathbf { x } ) + \mathbf { b } _ { h i d } ) + \mathbf { b } _ { o u t } ,\tag{22}
$$

where $W _ { h i d } ~ \in ~ \mathbb { R } ^ { d _ { h i d } \times 1 6 }$ and $W _ { o u t } ~ \in ~ \mathbb { R } ^ { C _ { v i s } \times d _ { h i d } }$ are the trainable weight matrices, $ { \mathbf { b } } _ { h i d }$ and $\mathbf { b } _ { o u t }$ are the biases. In our implementation, $d _ { h i d }$ is set to be 32.

To optimize the classical-quantum backbone, we compute the standard cross-entropy loss between the predicted logits $\hat { \mathbf { y } } ( \mathbf x )$ and the ground-truth one-hot label $\mathbf { y } _ { t r u e } .$

$$
\mathcal { L } _ { r e p } = - \sum _ { c = 1 } ^ { C _ { v i s } } y _ { t r u e , c } \log \left( \frac { \exp ( \hat { y } _ { c } ( \mathbf { x } ) ) } { \sum _ { j } \exp ( \hat { y } _ { j } ( \mathbf { x } ) ) } \right) .\tag{23}
$$

This auxiliary MLP head is utilized exclusively to provide supervised gradients for shaping a discriminative quantum density matrix space. Once the optimal feature representation is learned, this classical head is completely discarded during the subsequent mixed-state prototype fitting and decision.

4) Mixed-state prototype and prototype classification: To align with the backbone’s output, the mixed-state prototype preparation circuit is instantiated on a 4-qubit system, intrinsically operating within the identical $1 6 ~ \times ~ 1 6$ densitymatrix space. Concretizing the CCPS ansatz, the prototype circuit adopted the hardware efficient tiling. Specifically, we constructed three layers of VQC. And each layer consists of four two-qubit $S U ( 4 )$ blocks arranged in an alternating ring topology [47]. Since each $S U ( 4 )$ block consumes exactly 15 parameters, a single class prototype circuit requires a footprint of $4 \times 3 \times 1 5 = 1 8 0$ trainable parameters.

$$
\begin{array}{c} \underbrace { \overbrace { U 3 ( \theta _ { 1 } , \phi _ { 2 } , \lambda _ { 3 } ) } ^ { \substack { ( U 3 ( \theta _ { 1 } , \phi _ { 2 } , \lambda _ { 3 } ) ) } } \bullet \overbrace { ( R _ { y } ( \theta _ { 7 } ) ) } ^ { \substack { ( R _ { y } ( \theta _ { 7 } ) ) } } \overbrace { - \{ \underbrace { \big ( R _ { y } ( \theta _ { 9 } ) \big ) } _ { \substack { ( \mathrm { U 3 } ( \theta _ { 1 3 } , \phi _ { 1 4 } , \lambda _ { 1 5 } ) ) } } } ^ { \substack { ( U 3 ( \theta _ { 1 0 } , \phi _ { 1 1 } , \lambda _ { 1 2 } ) ) } } } \end{array}
$$

Fig. 6. The internal gate decomposition of a single two-qubit $S U ( 4 )$ block. It consumes exactly 15 trainable parameters, providing highly expressive transformations for the prototype circuit.

Given the unitary operator $U _ { c } ( \theta _ { c } )$ , the circuit transforms the first $K \left( K \leq 1 6 \right)$ computational basis states |i⟩ into a set of parameterized pure state components:

$$
| \psi _ { c , i } ( \theta _ { c } ) \rangle = U _ { c } ( \theta _ { c } ) | i \rangle .\tag{24}
$$

To form a valid mixed state, these pure states are probabilistically mixed using classical weights $w _ { c , i }$ . To ensure the adherence to the probability simplex constraint $( \sum w _ { c , i } ~ =$ $1 , w _ { c , i } \ge 0 )$ , we map a set of unconstrained trainable weights $s _ { c , i } \in \mathbb { R }$ via the softmax function Eq. (11).

The class-specific mixed-state prototype $\rho _ { c }$ is thus formally assembled as the weighted sum of these orthogonal pure states:

$$
\rho _ { c } = \sum _ { i = 0 } ^ { K - 1 } w _ { c , i } | \psi _ { c , i } ( \theta _ { c } ) \rangle \langle \psi _ { c , i } ( \theta _ { c } ) | .\tag{25}
$$

During the prototype fitting phase, the objective is to align $\rho _ { c }$ with the distribution of the training data. For a batch of $N$ training samples $\{ { \bf x } _ { n } \} _ { n = 1 } ^ { N }$ belonging to class c, the quantum backbone extracts their corresponding intermediate density matrices $\{ \rho _ { n } \} _ { n = 1 } ^ { N }$ . The overlap between each sample $\rho _ { n }$ and the prototype’s pure state component $| \psi _ { c , i } \rangle$ can be efficiently estimated on quantum hardware using the SWAP test:

$$
\mathcal { F } _ { c , i } ^ { ( n ) } = \mathrm { T r } \left[ \rho _ { n } | \psi _ { c , i } ( \theta _ { c } ) \rangle \langle \psi _ { c , i } ( \theta _ { c } ) | \right] .\tag{26}
$$

By minimizing the mean squared Hilbert-Schmidt distance $\begin{array} { r } { \frac { 1 } { N } \sum _ { n } \| \rho _ { n } - \rho _ { c } \| _ { F } ^ { 2 } } \end{array}$ and discarding the constant target purity $\mathrm { \bar { T r } } [ \rho _ { n } ^ { 2 } ]$ , the empirical loss function for optimizing the class prototype elegantly simplifies to:

$$
\mathcal { L } _ { p r o t o } ^ { c } ( \theta _ { c } , s _ { c } ) = \sum _ { i = 0 } ^ { K - 1 } w _ { c , i } ^ { 2 } - \frac { 2 } { N } \sum _ { n = 1 } ^ { N } \sum _ { i = 0 } ^ { K - 1 } w _ { c , i } \mathcal { F } _ { c , i } ^ { ( n ) } .\tag{27}
$$

By directly optimizing this loss, the 180-parameter circuit and the K weights are jointly updated to make the prototype fit the centroid of the samples. This concrete instantiation ensures robust semantic concentration with low parameter complexity, making it scalable for continuous incremental tasks. Let $\begin{array} { r } { \bar { \rho } _ { c } = \frac { 1 } { N } \bar { \sum _ { n = 1 } ^ { N } } \rho _ { n } } \end{array}$ denote the mean density matrix of class c. The mean squared HS distance differs from $\| \bar { \rho } _ { c } - \rho _ { c } \| _ { F } ^ { 2 }$ only by a constant independent of $\rho _ { c }$ . Therefore, optimizing Eq. (27) can be interpreted as applying the QMSC-based variational quantum PCA procedure to the class-level mean density matrix.

## C. Quantum incremental learning algorithm

The quantum incremental learning algorithm is a multistage process. Assuming a continuous data stream divided into sequential tasks $t \in \{ 0 , 1 , \ldots , T \}$ , let $\mathcal { V } _ { t }$ denote the set of all visible classes up to task t. At task $t ,$ the model receives a new dataset $\mathcal { D } _ { t }$ and has access to a limited replay memory $\mathcal { E } _ { t - 1 }$ containing exemplars from old classes. The incremental learning procedure follows a four-step algorithmic pipeline, which is summarized in Algorithm 2.

1) Step 1: Backbone representation learning: The training batch is constructed as $\mathcal { D } _ { t r a i n } ^ { ( \bar { t } ) } = \mathcal { D } _ { t } \cup \mathcal { E } _ { t - 1 }$ . The classical preprocessing module and the quantum feature extraction backbone are jointly updated. During this phase, the intermediate density matrix $\rho ( \mathbf { x } )$ is mapped to class logits via the auxiliary MLP head. The entire backbone is optimized by minimizing loss $\mathcal { L } _ { r e p } \ ( \mathrm { E q . } \ 2 3 )$ over all visible classes in $\mathcal { V } _ { t }$

2) Step 2: Mixed-state prototype fitting: Once the shared representation is learned, the auxiliary MLP head is detached. For each class $c \in \mathcal { V } _ { t }$ , the parameters $\left( \theta _ { c } , s _ { c } \right)$ of the classspecific prototype circuit are then independently optimized by minimizing loss $\mathcal { L } _ { p r o t o } ^ { c } \left( \mathrm { E q . ~ } 2 7 \right)$ , guided by the hardwareefficient SWAP test.

3) Step 3: Replay memory management: After prototype fitting, the exemplar set $\mathcal { E } _ { t }$ is updated. For each class, evaluate the squared HS distance between the sample density matrix $\rho ( \mathbf { x } )$ and the fitted prototype $\rho _ { c }$ via SWAP test. Samples yielding the minimum distance are stored in $\mathcal { E } _ { t }$ . The memory budget is distributed equally across classes.

4) Step 4: Prototype-based global inference: Any query instance $\bf { x } _ { \mathrm { t e s t } }$ is passed through the frozen backbone to extract its density matrix $\rho ( \mathbf { x } _ { \mathrm { t e s t } } )$ in the form of quantum states. The prediction is determined by identifying the nearest mixed-state prototype across all visible classes $\mathcal { V } _ { t }$

$$
\hat { y } = \arg \operatorname* { m i n } _ { c \in \mathcal { V } _ { t } } \mathcal { D } _ { c } \big ( \mathbf { x } _ { t e s t } \big ) .\tag{28}
$$

In practical implementation, the negative distance $- \mathcal { D } _ { c } ( \mathbf { x } _ { t e s t } )$ directly serves as the output logit.

Algorithm 2 Quantum Incremental Learning Algorithm   
input $\mathcal { D } _ { 1 } , \ldots , \mathcal { D } _ { T }$ // sequential training tasks   
input M // total memory budget   
require $\Theta _ { b b }$ // quantum backbone parameters   
require $\Theta _ { h e a d }$ // auxiliary MLP head parameters   
require $\mathcal { P } = \emptyset$ // mixed-state prototypes   
require $\mathcal { E } = \emptyset$ // exemplar memory buffer   
for $t = 1 , \dots , T \ \mathbf { d o }$   
Y ← VisibleClasses $( \mathcal { D } _ { 1 } , \ldots , \mathcal { D } _ { t } )$   
$\mathcal { D } _ { t r a i n }  \mathcal { D } _ { t } \cup \mathcal { E }$ // construct training batch   
// Step 1: Backbone representation learning   
$\Theta _ { b b } , \Theta _ { h e a d } $ UPDATEBACKBONE $( \mathcal { D } _ { t r a i n } , \Theta _ { b b } , \Theta _ { h e a d } )$   
// Step 2: Mixed-state prototype fitting   
P ← FITMIXEDPROTOTYPES $( \mathcal { D } _ { t r a i n } , \mathcal { V } _ { t } , \Theta _ { b b } )$   
// Step 3: Replay memory management   
$m  \lfloor M / \lvert \mathcal { V } _ { t } \rvert \rfloor \qquad / /$ number of exemplars per class   
$\mathcal { E }  \emptyset$ // reset memory buffer   
for $c \in \mathcal { V } _ { t }$ do   
E ← E ∪ CONSTRUCTEXEMPLARS(   
$\mathcal { D } _ { t r a i n } , c , m , \mathcal { P } , \Theta _ { b b } )$   
end for   
// Step 4: Prototype-based global inference   
for $\mathbf { \bar { x } } _ { t e s t } \in \mathcal { D } _ { t e s t } ^ { ( t ) }$ do   
$\hat { y } \gets \mathrm { C L A S S I F Y } \big ( \mathbf { x } _ { t e s t } , \mathcal { P } , \Theta _ { b b } \big )$   
end for   
end for

## V. EXPERIMENT

## A. Experimental Setup

1) Datasets: To evaluate the performance of the proposed framework, we conduct experiments on CIFAR-100 and Tiny-ImageNet.

CIFAR-100: It consists of 60,000 RGB color images with a spatial resolution of 32 × 32 pixels. The dataset contains 100 classes, representing a diverse set of everyday objects. For each class, it is divided into 500 images for training and 100 for testing. To evaluate the sensitivity to class selection, we construct three fixed and mutually disjoint 32-class subsets, denoted as Split A, Split B, and Split C, which cover 96 classes in total.

TinyImageNet: This dataset serves as a scaled-down modification of the full ImageNet dataset, frequently utilized to bridge the gap between low-resolution datasets and large-scale images. It comprises 200 classes. Each class contains 500 training images, 50 validation images, and 50 testing images, all possessing a spatial resolution of 64 × 64 pixels.

For each CIFAR-100 split and TinyImageNet, we selected 32 categories and divided them into 16 initial classes and 4 groups of 4 categories for incremental tasks. Experiments resulted in five tasks from the 16 to 32 classes. These splits were then fixed in the experiment. The detailed dataset split protocol is provided in the source code.

2) Experimental configuration: This study utilizes the Py-Torch and PennyLane frameworks for hybrid model construction and quantum circuit simulation. All experiments are executed on an x86 platform equipped with a single NVIDIA GeForce RTX 3090 (24GB) GPU. In the statistical significance experiments, we randomly select four seeds and report the results as mean ± standard deviation. For all other experiments, the random seed is fixed to 1993. All experiments used a validation split of 10%. We used ResNet-18 preprocessing adapted to the image resolution, followed by a 256-dimensional MLP projection and an 8-qubit quantum backbone. To avoid dataset-specific hyperparameter selection, we fix the prototype rank to K = 7 for every dataset and class split. This value is specified uniformly before the comparative evaluation and is not selected according to test-set performance on individual datasets. The initial task was trained for 240 epochs with a batch size of 128 and a learning rate of 0.01. Incremental tasks used prototype-nearest exemplar replay with a fixed memory budget of 640 samples, allocated in a perclass manner (approximately 20 samples per class after all 32 classes are observed). The prototype branch was trained for 60 epochs with a learning rate of 0.04 and a density batch size of 16, while the incremental finetuning stage was trained for 36 epochs with a learning rate of 0.01. We used SGD with momentum 0.9 and weight decay 1e-4 for backbone/classifier training, and Adam for prototype optimization. A warm-up cosine scheduler was used for the initial task, and a cosine scheduler was used for incremental finetuning. The source code is available at: https://anonymous.4open.science/r/QPIL-3E43.

3) Quantum circuit configuration: Due to the constraints of current quantum resources, the quantum classification backbone is implemented with an 8-qubit circuit, while the mixedstate prototype circuit for each class is instantiated on a 4-qubit system. We use the HS distance between density matrices, rather than constructing an entire 13-qubit circuit, when the SWAP test will also introduce an auxiliary qubit. By disassembling the modules, a significant amount of storage was saved. The 8-qubit quantum classifier exposes an intermediate 4-qubit density matrix on the retained wires after the partial QCNN transformation, and each CCPS prototype is also represented as a 16 × 16 density matrix in the same 4-qubit Hilbert space. 4) Evaluation Metrics: We evaluate our method and the baselines using two primary metrics: average accuracy (Avg Acc) and last accuracy (Last Acc).

Let T be the total number of incremental tasks. We denote A<sub>t</sub> as the top-1 classification accuracy evaluated on the test set of all seen classes from task 1 to task t, after the model has finished training on task t.

Last Acc: This metric measures the final performance of the model after the entire sequence of incremental learning is completed. It is defined as the accuracy on all learned classes after the last task T:

$$
\mathcal { A } _ { l a s t } = A _ { T }\tag{29}
$$

Avg Acc: This metric reflects the historical performance and the robustness of the model against catastrophic forgetting throughout the entire continual learning process. It is defined as the average of the top-1 incremental accuracies over all T tasks:

$$
\mathcal { A } _ { a v g } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { A } _ { t }\tag{30}
$$

## B. Effectiveness of the Quantum Prototype Classifier

We evaluate the effectiveness of the mixed-state quantum prototype classifier proposed in Section III. The purpose of this experiment is to compare the proposed prototypebased decision mechanism with simple classical classifiers and representative quantum classifiers.

We compare our method with four simple classical classifiers and four quantum classifiers. Linear Softmax and Cosine Classifier use parametric classification heads. Nearest Centroid and RBF-SVM are fitted on normalized features extracted by independently trained backbones. The quantum baselines retain the ResNet-18 preprocessing and use 8-qubit amplitude encoding. HEA-VQC applies RY-RZ layers and ring CNOT gates, followed by direct five-qubit probability measurement for 32-class prediction [48]. QCNN uses SU(4) convolution and quantum pooling, followed by the same direct probability measurement [45]. TTN-QNN uses a hierarchical 8-qubit tree circuit and a 16-to-32 classical readout layer [49]. Fidelity Classifier represents each class using a trainable pure-state prototype and performs classification based on state fidelity [50]. Our method instead represents each class using a mixedstate prototype and performs classification based on the HS distance. All methods are evaluated under the same nonincremental setting on the same 32 selected classes. The results are reported as mean ± standard deviation over four runs.

As shown in Table I, our method achieves mean accuracies of 0.8323, 0.7766, 0.8222, and 0.6997 on the four datasets.

TABLE I  
COMPARISON WITH CLASSICAL AND QUANTUM CLASSIFIERS.
<table><tr><td>Type</td><td>Classifier</td><td>Split A</td><td>Split B</td><td>Split C</td><td>TinyImageNet</td><td>Avg</td></tr><tr><td>Classical</td><td>Linear Softmax</td><td> $0 . 8 2 8 1 \pm 0 . 0 0 6 2$ </td><td> $0 . 7 8 9 5 \pm 0 . 0 0 4 6$ </td><td> $\mathbf { 0 . 8 2 4 5 \ : \pm { \ : 0 . 0 0 5 9 } }$ </td><td> $0 . 6 8 4 4 \pm 0 . 0 1 0 0$ </td><td> $0 . 7 8 1 6 \pm 0 . 0 0 6 2$ </td></tr><tr><td>Classical</td><td>Cosine Classifier</td><td> $0 . 8 2 5 9 \pm 0 . 0 0 9 1$ </td><td> $\mathbf { 0 . 7 9 2 9 \ : \pm { \ : 0 . 0 0 6 2 } }$ </td><td> $0 . 8 2 2 3 \pm 0 . 0 0 4 0$ </td><td> $0 . 6 7 8 3 \pm 0 . 0 0 4 3$ </td><td> $0 . 7 7 9 8 \pm 0 . 0 0 3 7$ </td></tr><tr><td>Classical</td><td>Nearest Centroid</td><td> $0 . 8 2 8 0 \pm 0 . 0 0 4 2$ </td><td> $0 . 7 8 8 4 \pm 0 . 0 0 4 5$ </td><td> $0 . 8 2 4 0 \pm 0 . 0 0 4 9$ </td><td> $0 . 6 8 4 5 \pm 0 . 0 0 9 4$ </td><td> $0 . 7 8 1 3 \pm 0 . 0 0 5 2$ </td></tr><tr><td>Classical</td><td>RBF-SVM</td><td> $0 . 8 2 5 7 \pm 0 . 0 0 4 5$ </td><td> $0 . 7 9 0 9 \pm 0 . 0 0 7 3$ </td><td> $0 . 8 2 3 3 \pm 0 . 0 0 5 0$ </td><td> $0 . 6 8 8 1 \pm 0 . 0 1 0 9$ </td><td> $0 . 7 8 2 0 \pm 0 . 0 0 4 8$ </td></tr><tr><td>Quantum</td><td>HEA-VQC [48]</td><td> $0 . 6 6 0 1 \pm 0 . 0 0 3 8$ </td><td> $0 . 6 0 6 6 \pm 0 . 0 1 0 5$ </td><td> $0 . 6 4 8 4 \pm 0 . 0 0 6 6$ </td><td> $0 . 5 5 1 7 \pm 0 . 0 1 2 9$ </td><td> $0 . 6 1 6 7 \pm 0 . 0 0 6 4$ </td></tr><tr><td>Quantum</td><td>QCNN [45]</td><td> $0 . 6 7 4 1 \pm 0 . 0 2 9 3$ </td><td> $0 . 6 2 5 9 \pm 0 . 0 1 6 1$ </td><td> $0 . 6 7 8 1 \pm 0 . 0 4 7 9$ </td><td> $0 . 5 7 0 9 \pm 0 . 0 4 5 8$ </td><td> $0 . 6 3 7 2 \pm 0 . 0 3 4 2$ </td></tr><tr><td>Quantum</td><td>TTN-QNN [49]</td><td> $0 . 1 0 8 9 \pm 0 . 0 3 1 7$ </td><td> $0 . 1 3 4 1 \pm 0 . 0 2 1 8$ </td><td> $0 . 1 3 4 6 \pm 0 . 0 2 4 4$ </td><td> $0 . 1 1 6 6 \pm 0 . 0 3 2 2$ </td><td> $0 . 1 2 3 6 \pm 0 . 0 1 1 9$ </td></tr><tr><td>Quantum</td><td>Fidelity Classifier [50]</td><td> $0 . 7 3 1 1 \pm 0 . 0 4 3 6$ </td><td> $0 . 7 5 8 4 \pm 0 . 0 2 4 7$ </td><td> $0 . 7 9 6 5 \pm 0 . 0 2 9 5$ </td><td> $0 . 6 5 8 6 \pm 0 . 0 4 6 9$ </td><td> $0 . 7 3 6 2 \pm 0 . 0 1 9 4$ </td></tr><tr><td>Quantum</td><td>Mixed-state Prototype Classifier (Ours)</td><td> $\mathbf { 0 . 8 3 2 3 \ : \pm { \ : 0 . 0 0 8 2 } }$ </td><td> $0 . 7 7 6 6 \pm 0 . 0 1 6 5$ </td><td> $0 . 8 2 2 2 \pm 0 . 0 1 1 7$ </td><td> $\mathbf { 0 . 6 9 9 7 \ : \pm { \ : 0 . 0 1 0 6 } }$ </td><td> $\mathbf { 0 . 7 8 2 7 \ : \pm { \ : 0 . 0 0 3 3 } }$ </td></tr></table>

Avg denotes the average accuracy over the four datasets.

TABLE II  
CLASS-INCREMENTAL COMPARISON WITH REPLAY-BASED METHODS.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Memory</td><td rowspan="2">Parameter magnitude</td><td colspan="2">Split A</td><td colspan="2">Split B</td><td colspan="2">Split C</td><td colspan="2">TinyImageNet</td></tr><tr><td>Last</td><td> $\mathbf { A v } \mathbf { g }$ </td><td>Last</td><td> $\mathbf { A v } \mathbf { g }$ </td><td>Last</td><td> $\mathbf { A v } \mathbf { g }$ </td><td>Last</td><td>Avg</td></tr><tr><td>FOSTER [7]</td><td>640</td><td>O(nd)</td><td>0.6847</td><td>0.7796</td><td>0.5850</td><td>0.7006</td><td>0.6722</td><td>0.7711</td><td>0.3588</td><td>0.4756</td></tr><tr><td>iCaRL [6]</td><td>2000</td><td>O(nd)</td><td>0.7170</td><td>0.7754</td><td>0.6690</td><td>0.7468</td><td>0.7280</td><td>0.7960</td><td>0.5780</td><td>0.6488</td></tr><tr><td>PODNet [8]</td><td>2000</td><td>O(nmd)</td><td>0.6940</td><td>0.7746</td><td>0.6480</td><td>0.7362</td><td>0.6840</td><td>0.7760</td><td>0.5440</td><td>0.6194</td></tr><tr><td> $\mathbf { O u r s } \ : ( K = 7 )$ </td><td>640</td><td> $\mathcal { O } ( n \log _ { 2 } ^ { ' } ( \operatorname* { m a x } ( n , d ) ) )$ </td><td>0.5728</td><td>0.7178</td><td>0.4947</td><td>0.6465</td><td>0.5481</td><td>0.7097</td><td>0.3838</td><td>0.5371</td></tr><tr><td>BiC [51]</td><td>640</td><td> $\mathcal { O } ( n d )$ </td><td>0.5308</td><td>0.6719</td><td>0.1997</td><td>0.4802</td><td>0.2578</td><td>0.5848</td><td>0.0511</td><td>0.2524</td></tr><tr><td>SSRE [52]</td><td>2000</td><td>O(nd)</td><td>0.4947</td><td>0.5970</td><td>0.4522</td><td>0.5776</td><td>0.5025</td><td>0.6276</td><td>0.4319</td><td>0.5259</td></tr><tr><td>DER [53]</td><td>640</td><td>O(nd)</td><td>0.3941</td><td>0.5785</td><td>0.5374</td><td>0.6466</td><td>0.5603</td><td>0.6996</td><td>0.0313</td><td>0.1551</td></tr><tr><td>LUCIR [54]</td><td>2000</td><td>O(nd)</td><td>0.1690</td><td>0.2206</td><td>0.2090</td><td>0.2486</td><td>0.1800</td><td>0.2394</td><td>0.1800</td><td>0.2178</td></tr></table>

Memory denotes the maximum number of retained real training samples. n: number of classes; d: feature dimension; m: number of proxies per class.

It obtains the highest mean accuracy among the compared quantum classifiers on all four datasets. It also achieves the highest accuracy among all compared classifiers on Split A and TinyImageNet. On Split B and Split C, it is 1.63 and 0.23 percentage points lower than the strongest classical classifier, respectively. Its average accuracy over the four datasets is 0.7827, which is the highest among the compared methods. Compared with the direct-measurement QCNN, our method improves the mean accuracy by 15.83, 15.07, 14.41, and 12.88 percentage points, respectively. These results demonstrate that directly relying on the measured output of the quantum backbone is insufficient for robust multi-class classification because the measured probability vector is constrained by the retained quantum basis states. It assigns the same fixed Hilbert space size to each category and therefore cannot effectively handle uneven feature distributions. In contrast, our approach is to adaptively divide the feature space into different ranges for different classes.

comparison, we organize the representative baselines into four categories. Replay-based methods include FOSTER, iCaRL, and PODNet [6]–[8]; BiC, SSRE, and DER [51]–[53]; and LUCIR [54]. Prototype-based methods include FeTrIL, PASS, and NCM [14], [16], [55], as well as Eucl-NCM and TOPIC+ [15], [56]. Statistics-based methods include FeCAM, SDC, and DeeSIL [15], [57], [58]. Regularization-based methods include ABD, MUC-LwF, and LwF-MC [10], [11], [59], as well as EWC [9]. Finetune is included as a conventional sequential baseline. Our method combines mixed-state prototype classification with exemplar replay and is therefore included in both tables as a common reference. All baseline results were reproduced using their official open-source implementations. The same class subsets and class order were used for all methods. Each baseline used either the same ResNet backbone as ours or the backbone provided by its official implementation. All methods are evaluated by the final accuracy after the last task, denoted as Last Acc, and the average accuracy over all incremental stages, denoted as Avg Acc.

## C. Effectiveness on Incremental Learning

This section evaluates the effectiveness of the proposed method under the incremental learning setting. The purpose is not to claim state-of-the-art performance over all classical methods, but to verify whether the proposed quantum method has an effective incremental learning capability. Since the model operates with a limited-width quantum circuit and a fixed memory budget, the key question is whether it can alleviate catastrophic forgetting.

We follow the fixed incremental protocol introduced in Section V-A. The model starts from 16 initial classes and then learns four incremental tasks, each containing 4 new classes, resulting in five evaluation stages from 16 to 32 classes. For

As shown in Tables II and III, with the same fixed rank $K =$ 7 on all datasets, the proposed method achieves 0.5728 Last Acc and 0.7178 Avg Acc on CIFAR-100 Split A. Although it does not outperform the strongest classical baselines, its average accuracy is higher than that of BiC, Eucl-NCM, NCM, MUC-LwF, SSRE, DeeSIL, TOPIC+, SDC, and LUCIR. The corresponding Avg Acc values are 0.6465 and 0.7097 on Split B and Split C, respectively, showing that the method remains sensitive to class selection even under a uniform rank. On TinyImageNet, the same K = 7 configuration obtains 0.3838 Last Acc and 0.5371 Avg Acc. The stage-wise curves in Fig. 7 further show that the proposed method exhibits a gradual accuracy decline comparable to several prototype and statistics baselines, rather than an abrupt collapse. These results demonstrate meaningful class-incremental learning capability without selecting a separate rank for each benchmark.

TABLE III  
CLASS-INCREMENTAL COMPARISON WITH PROTOTYPE, STATISTICS, AND REGULARIZATION METHODS.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Type</td><td rowspan="2">Parameter magnitude</td><td colspan="2">Split A</td><td colspan="2">Split B</td><td colspan="2">Split C</td><td colspan="2">TinyImageNet</td></tr><tr><td>Last</td><td>Avg</td><td>Last</td><td>Avg</td><td>Last</td><td>Avg</td><td>Last</td><td>Avg</td></tr><tr><td>FeCAM [15]</td><td>Statistics</td><td>O(nd2)</td><td>0.6869</td><td>0.7826</td><td>0.6181</td><td>0.7330</td><td>0.6728</td><td>0.7785</td><td>0.5919</td><td>0.6653</td></tr><tr><td>FeTrIL [14]</td><td>Prototype</td><td>O(nd)</td><td>0.6453</td><td>0.7417</td><td>0.5462</td><td>0.6613</td><td>0.6319</td><td>0.7474</td><td>0.5544</td><td>0.6291</td></tr><tr><td>PASS [16]</td><td>Prototype</td><td>O(4nd)</td><td>0.6347</td><td>0.7236</td><td>0.5822</td><td>0.7073</td><td>0.5984</td><td>0.7142</td><td>0.4538</td><td>0.5579</td></tr><tr><td>Ours (K = 7)</td><td>Prototype</td><td>O(n log2(max(n, d)))</td><td>0.5728</td><td>0.7178</td><td>0.4947</td><td>0.6465</td><td>0.5481</td><td>0.7097</td><td>0.3838</td><td>0.5371</td></tr><tr><td>ABD [11]</td><td>Regularization</td><td>O(nd)</td><td>0.5445</td><td>0.6487</td><td>0.5218</td><td>0.6431</td><td>0.5544</td><td>0.6976</td><td>0.4815</td><td>0.6076</td></tr><tr><td>NCM [55]</td><td>Prototype</td><td>O(nd)</td><td>0.4863</td><td>0.6598</td><td>0.4438</td><td>0.6115</td><td>0.4700</td><td>0.6519</td><td>0.4431</td><td>0.5962</td></tr><tr><td>MUC-LwF [59]</td><td>Regularization</td><td>O(nd)</td><td>0.4925</td><td>0.6299</td><td>0.4178</td><td>0.5889</td><td>0.4863</td><td>0.6643</td><td>0.3069</td><td>0.4981</td></tr><tr><td>Eucl-NCM [15]</td><td>Prototype</td><td>O(nd)</td><td>0.4612</td><td>0.6291</td><td>0.5475</td><td>0.6918</td><td>0.5600</td><td>0.7127</td><td>0.5075</td><td>0.6306</td></tr><tr><td>Finetune</td><td>Baseline</td><td>O(nd)</td><td>0.3459</td><td>0.5212</td><td>0.3378</td><td>0.5202</td><td>0.3869</td><td>0.5838</td><td>0.2856</td><td>0.4764</td></tr><tr><td>DeeSIL [58]</td><td>Statistics</td><td>O(nd)</td><td>0.2553</td><td>0.4904</td><td>0.2078</td><td>0.4555</td><td>0.2534</td><td>0.5191</td><td>0.1988</td><td>0.3597</td></tr><tr><td>LwF-MC [10]</td><td>Regularization</td><td>O(nd)</td><td>0.4290</td><td>0.5319</td><td>0.3914</td><td>0.5280</td><td>0.3513</td><td>0.5451</td><td>0.1412</td><td>0.3366</td></tr><tr><td>TOPIC+ [56]]</td><td>Prototype</td><td>O(nd)</td><td>0.1597</td><td>0.3741</td><td>0.1578</td><td>0.3763</td><td>0.1159</td><td>0.3856</td><td>0.2056</td><td>0.4151</td></tr><tr><td>SDC [57]</td><td>Statistics</td><td>O(nd)</td><td>0.1403</td><td>0.3566</td><td>0.1225</td><td>0.3286</td><td>0.0997</td><td>0.3546</td><td>0.1213</td><td>0.2972</td></tr><tr><td>EWC [9]</td><td>Regularization</td><td>O(nd)</td><td>0.0259</td><td>0.1987</td><td>0.1805</td><td>0.3931</td><td>0.1525</td><td>0.3972</td><td>0.0273</td><td>0.1592</td></tr></table>

n: number of classes; d: feature dimension; m: number of proxies per class.

![](images/d864115c68398d4305a2789b9b7eec04b2f74592929ade5eee9b66a123fd4312.jpg)  
Fig. 7. Accuracy at each incremental stage for the prototype, statistics, regularization, and finetuning methods in Table III. Solid, dashed, dash-dotted, and dotted lines denote prototype, statistics, regularization, and baseline methods, respectively.

More importantly, this result is achieved under a compact quantum setting. The quantum backbone only uses 8 qubits, and the prototypes use 4 qubits. The trainable quantum backbone contains only a small number of quantum parameters, while each class prototype uses 180 circuit parameters and K mixture weights. Despite this limited quantum width and fixed memory budget, the model shows a gradual decline in accuracy over the five incremental stages. These results suggest that mixed-state quantum prototypes and prototypeguided replay can mitigate forgetting to a certain extent and provide a feasible direction for quantum incremental learning under NISQ-era resource constraints.

## D. Ablation Studies

We conduct ablation studies on two core designs: the mixed-state prototype representation and the exemplar replay mechanism. The former is used to improve the representation capacity of class prototypes by modeling each class as a density matrix rather than a single pure state. The latter is used to preserve old-class information during incremental updates by replaying a small number of representative exemplars.

For the mixed-state prototype ablation, we replace the proposed CCPS mixed-state prototype with a pure-state prototype, denoted as w/o Mixed State. This variant keeps the same incremental learning protocol but represents each class using a single pure state. For the backbone ablation, we freeze the shared backbone and only fit new-class prototypes, denoted as Fixed Backbone. For the replay ablation, we remove the exemplar replay mechanism, denoted as w/o Replay. This variant updates the model using only new-task data.

As shown in Table IV and Fig. 8, on CIFAR-100 Split A, replacing the fixed-rank $( K = 7 )$ mixed-state prototype with a pure-state prototype leads to a clear performance drop. The Last Acc decreases from 0.5728 to 0.5450, and the Avg Acc drops from 0.7178 to 0.6668. This demonstrates that a single pure state is not sufficient to represent complex intraclass variations in the incremental setting. In contrast, the proposed mixed-state prototype uses a convex combination of multiple pure-state components, which provides a more flexible representation for each class. The full model also retains the highest Avg Acc among the compared variants on Split B, Split C, and TinyImageNet.

TABLE IV  
ABLATION STUDY OF MIXED-STATE PROTOTYPES AND EXEMPLAR REPLAY.
<table><tr><td>Method</td><td colspan="2">Split A</td><td colspan="2">Split B</td><td colspan="2">Split C</td><td colspan="2">TinyImageNet</td></tr><tr><td></td><td>Last</td><td>Avg</td><td>Last</td><td>Avg</td><td>Last</td><td>Avg</td><td>Last</td><td>Avg</td></tr><tr><td>w/o Mixed State</td><td>0.5450</td><td>0.6668</td><td>0.4156</td><td>0.5634</td><td>0.4772</td><td>0.6288</td><td>0.3656</td><td>0.4994</td></tr><tr><td>Fixed Backbone</td><td>0.4788</td><td>0.6586</td><td>0.3997</td><td>0.5624</td><td>0.4559</td><td>0.6355</td><td>0.3769</td><td>0.5242</td></tr><tr><td>w/o Replay</td><td>0.5122</td><td>0.6735</td><td>0.4425</td><td>0.5931</td><td>0.4772</td><td>0.6509</td><td>0.3850</td><td>0.5202</td></tr><tr><td>Ours (full,  $K = 7 )$ </td><td>0.5728</td><td>0.7178</td><td>0.4947</td><td>0.6465</td><td>0.5481</td><td>0.7097</td><td>0.3838</td><td>0.5371</td></tr></table>

The Fixed Backbone variant obtains only 0.4788 Last Acc and 0.6586 Avg Acc on Split A. The w/o Replay variant improves the result to 0.5122 Last Acc and 0.6735 Avg Acc, but it still performs worse than the fixed-K = 7 full model. This shows that directly updating the model with only newtask data is insufficient for maintaining a stable prototype space. By replaying exemplars nearest to the learned quantum prototypes, the full method provides old-class constraints during incremental training and helps different prototype positions remain appropriately matched.

![](images/70edf492817a76bb96a1be0d057c8e560e501672c60c344eede6febb15675fe2.jpg)  
Fig. 8. Accuracy at each incremental stage for different ablation variants on CIFAR-100 Split A, Split B, Split C, and TinyImageNet.

## E. Effect of Prototype Rank

This section further investigates the PCA-like property of the proposed mixed-state prototype. As discussed in the previous sections, the CCPS prototype represents each class as a convex combination of multiple transformed basis states. The number of retained basis states K controls the effective rank of the prototype density matrix. Therefore, K plays a role similar to the number of principal components in PCA: a small K may discard useful class information, while an excessively large K may introduce redundant or noisy components. This experiment evaluates whether an appropriate rank selection can improve the robustness of the mixed-state prototype.

We conduct a rank sweep by varying K in the CCPS prototype while keeping the other incremental learning settings unchanged. Specifically, we evaluate K from 1 to 16 under the same 16-to-32 class-incremental protocol. When $K = 1 6 .$ , the prototype uses all basis states in the 4-qubit Hilbert space, corresponding to a full-rank representation. Smaller values of K force the prototype to retain only a subset of basis components, which can be regarded as a low-rank approximation of the class density distribution.

As shown in Table V, the performance does not monotonically improve as K increases. Although K = 16 provides the largest representational capacity, it does not consistently obtain the best Avg Acc. This suggests that using all basis components may preserve redundant or noisy variations in the class density matrix, which weakens the robustness of the prototype. In contrast, moderate-rank settings generally achieve better performance. The retrospective sweep reaches its best Avg Acc at $K \ = \ 1 2 .$ $K \ = \ 7 ,$ , and $K \ = \ 1 5$ on Split A, Split B, and Split C, respectively, while TinyImageNet reaches its best Avg Acc at K = 7. Importantly, these sweep maxima are not used to choose a separate setting for each dataset: all main results in the preceding sections use the same prespecified value, $K = 7 .$ The sweep is reported only as a sensitivity analysis. This protocol avoids selecting the rank according to test performance and provides a consistent basis for cross-dataset comparison.

Fig. 9 further illustrates the influence of K from the perspective of stability. When K is too small, the prototype only keeps a few dominant components, which may remove noise but also risks discarding useful intra-class variations. When K is too large, especially at the full-rank setting $\begin{array} { r l r } { K } & { { } = } & { 1 6 . } \end{array}$ , the prototype absorbs more low-contribution components, increasing the fluctuation across incremental tasks and reducing the denoising effect. The middle range of K provides a better trade-off: it preserves the main class structure while suppressing redundant components.

These results support the PCA-like interpretation of the mixed-state prototype. By controlling the rank K, the CCPS prototype can filter out low-contribution components that are regarded as noise. Thus, it can focus on the dominant structure of the class distribution. Therefore, the mixed-state prototype is not merely a higher-capacity representation than a pure state, but also provides a controllable low-rank approximation mechanism that improves robustness under incremental learning.

TABLE V  
SENSITIVITY OF CLASS-INCREMENTAL LEARNING TO PROTOTYPE RANK K.
<table><tr><td rowspan="2">K</td><td colspan="2">Split A</td><td colspan="2">Split B</td><td colspan="2">Split C</td><td colspan="2">TinyImageNet</td></tr><tr><td>Last</td><td> $\mathbf { A v \mathbf { g } }$ </td><td>Last</td><td> $\mathbf { A v \mathbf { g } }$ </td><td>Last</td><td> $\mathbf { A v \mathbf { g } }$ </td><td>Last</td><td> $\mathbf { A v } \mathbf { g }$ </td></tr><tr><td>1</td><td>0.5184</td><td>0.6913</td><td>0.4103</td><td>0.5567</td><td>0.4834</td><td>0.6369</td><td>0.3656</td><td>0.4994</td></tr><tr><td>2</td><td>0.5372</td><td>0.6758</td><td>0.4706</td><td>0.5848</td><td>0.4400</td><td>0.6197</td><td>0.3600</td><td>0.5043</td></tr><tr><td>3</td><td>0.5459</td><td>0.6995</td><td>0.4603</td><td>0.6104</td><td>0.5072</td><td>0.6590</td><td>0.3844</td><td>0.5122</td></tr><tr><td>4</td><td>0.5516</td><td>0.7066</td><td>0.4900</td><td>0.6311</td><td>0.5122</td><td>0.6880</td><td>0.3931</td><td>0.5212</td></tr><tr><td>5</td><td>0.5972</td><td>0.7193</td><td>0.4872</td><td>0.6254</td><td>0.5063</td><td>0.6797</td><td>0.3744</td><td>0.5124</td></tr><tr><td>6</td><td>0.6009</td><td>0.7273</td><td>0.4881</td><td>0.6397</td><td>0.5291</td><td>0.6989</td><td>0.3788</td><td>0.5148</td></tr><tr><td>7</td><td>0.5728</td><td>0.7178</td><td>0.4947</td><td>0.6465</td><td>0.5481</td><td>0.7097</td><td>0.3838</td><td>0.5371</td></tr><tr><td>8</td><td>0.6000</td><td>0.7260</td><td>0.4831</td><td>0.6325</td><td>0.5547</td><td>0.7142</td><td>0.3588</td><td>0.5049</td></tr><tr><td>9</td><td>0.5756</td><td>0.7215</td><td>0.4988</td><td>0.6436</td><td>0.5513</td><td>0.7122</td><td>0.3819</td><td>0.5160</td></tr><tr><td>10</td><td>0.6016</td><td>0.7232</td><td>0.4919</td><td>0.6255</td><td>0.5253</td><td>0.6798</td><td>0.3606</td><td>0.5023</td></tr><tr><td>11</td><td>0.6050</td><td>0.7271</td><td>0.4913</td><td>0.6285</td><td>0.5603</td><td>0.7118</td><td>0.3744</td><td>0.5282</td></tr><tr><td>12</td><td>0.6144</td><td>0.7286</td><td>0.4813</td><td>0.6205</td><td>0.5469</td><td>0.6950</td><td>0.3750</td><td>0.5191</td></tr><tr><td>13</td><td>0.5647</td><td>0.7132</td><td>0.4781</td><td>0.6337</td><td>0.5391</td><td>0.6781</td><td>0.3594</td><td>0.5082</td></tr><tr><td>14</td><td>0.5919</td><td>0.7148</td><td>0.4938</td><td>0.6331</td><td>0.5563</td><td>0.6953</td><td>0.3388</td><td>0.5102</td></tr><tr><td>15</td><td>0.5853</td><td>0.7126</td><td>0.4966</td><td>0.6370</td><td>0.5741</td><td>0.7159</td><td>0.3744</td><td>0.5154</td></tr><tr><td>16</td><td>0.5731</td><td>0.7069</td><td>0.4897</td><td>0.6321</td><td>0.5656</td><td>0.7080</td><td>0.3650</td><td>0.5173</td></tr></table>

![](images/915c346820ca830d941623dd38f930fa016b61b4bdba1e87e12bdd1947c15a5f.jpg)  
Fig. 9. Effect of Prototype Rank K on Avg Acc for CIFAR-100 Split A, Split B, Split C, and TinyImageNet.

## VI. CONCLUSION

In this paper, we introduced a mixed-state quantum prototype framework for class-incremental learning. Within constrained circuit width, our method matches quantum features with class prototypes to classify. We utilized CCPS ansatz to construct the mixed-state prototypes. It shows higher representational ability than classic single pure state. Furthermore, we designed a prototype-guided replay strategy to achieve incremental learning on quantum hardware. Simulation results indicate that the mixed-state classifier outperforms direct measurement-based quantum classification. And the prototype rank analysis further supported its PCA-like noise-filtering properties.

For future work, we will explore more hardware-efficient prototype circuits and investigate noise-aware training methodologies on real quantum devices. Additionally, extending this framework to larger-scale incremental scenarios is a promising direction. Accommodating more diverse datasets will further advance quantum models in continual class learning.

## REFERENCES

[1] D.-W. Zhou, Q.-W. Wang, Z.-H. Qi, H.-J. Ye, D.-C. Zhan, and Z. Liu, “Class-incremental learning: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 9851–9873, 2024.

[2] J. Zhang, L. Liu, O. Silven, M. Pietik´ ainen, and D. Hu, “Few-shot¨ class-incremental learning for classification and object detection: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 4, pp. 2924–2945, 2025.

[3] M. Masana, X. Liu, B. Twardowski, M. Menta, A. D. Bagdanov, and J. Van De Weijer, “Class-incremental learning: survey and performance evaluation on image classification,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 5, pp. 5513–5533, 2022.

[4] Q. Gao, C. Zhao, B. Ghanem, and J. Zhang, “R-dfcil: Relation-guided representation learning for data-free class incremental learning,” in European Conference on Computer Vision. Springer, 2022, pp. 423– 439.

[5] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks: The sequential learning problem,” in Psychology of learning and motivation. Elsevier, 1989, vol. 24, pp. 109–165.

[6] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, “icarl: Incremental classifier and representation learning,” in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2017, pp. 2001–2010.

[7] F.-Y. Wang, D.-W. Zhou, H.-J. Ye, and D.-C. Zhan, “Foster: Feature boosting and compression for class-incremental learning,” in European conference on computer vision. Springer, 2022, pp. 398–414.

[8] A. Douillard, M. Cord, C. Ollion, T. Robert, and E. Valle, “Podnet: Pooled outputs distillation for small-tasks incremental learning,” in European conference on computer vision. Springer, 2020, pp. 86–102.

[9] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the national academy of sciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[10] Z. Li and D. Hoiem, “Learning without forgetting,” IEEE transactions on pattern analysis and machine intelligence, vol. 40, no. 12, pp. 2935– 2947, 2017.

[11] J. Smith, Y.-C. Hsu, J. Balloch, Y. Shen, H. Jin, and Z. Kira, “Always be dreaming: A new approach for data-free class-incremental learning,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 9374–9384.

[12] L. Huang, J. Fan, S. Wang, G. Liu, and A. W.-C. Liew, “Incomplete data classification via distribution alignment with evidence combination,” Machine Intelligence Research, vol. 23, no. 3, pp. 593–615, 2026.

[13] L. Huang and G. Liu, “Dual joint covariance alignment method for incomplete data classification,” IEEE Transactions on Neural Networks and Learning Systems, 2026.

[14] G. Petit, A. Popescu, H. Schindler, D. Picard, and B. Delezoide, “Fetril: Feature translation for exemplar-free class-incremental learning,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2023, pp. 3911–3920.

[15] D. Goswami, Y. Liu, B. Twardowski, and J. Van De Weijer, “Fecam: Exploiting the heterogeneity of class distributions in exemplar-free continual learning,” Advances in Neural Information Processing Systems, vol. 36, pp. 6582–6595, 2023.

[16] F. Zhu, X.-Y. Zhang, C. Wang, F. Yin, and C.-L. Liu, “Prototype augmentation and self-supervision for incremental learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 5871–5880.

[17] G. M. Van de Ven, T. Tuytelaars, and A. S. Tolias, “Three types of incremental learning,” Nature Machine Intelligence, vol. 4, no. 12, pp. 1185–1197, 2022.

[18] J. Zhao and K. H. Cheong, “Multidomain evolutionary optimization on adversarial link perturbation in imbalanced-size complex systems,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2026.

[19] L. Xu, X.-y. Zhang, M. Li, and S.-q. Shen, “Quantum classifiers with a trainable kernel,” Physical Review Applied, vol. 21, no. 5, p. 054056, 2024.

[20] T. Zhan, Y. He, and Y. Deng, “Ternary coding of maximum deng entropy,” Fuzzy Sets and Systems, p. 109913, 2026.

[21] L. Li, J. Li, Y. Song, S. Qin, Q. Wen, and F. Gao, “An efficient quantum proactive incremental learning algorithm,” Science China Physics, Mechanics & Astronomy, vol. 68, no. 1, p. 210313, 2025.

[22] J. Shi, B. Yuan, Y. Lu, S. Zhang, and X. Li, “Quantum option profit and loss encoding for risk assessment,” IEEE Transactions on Emerging Topics in Computational Intelligence, 2025.

[23] M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, J. R. McClean, K. Mitarai, X. Yuan, L. Cincio et al., “Variational quantum algorithms,” Nature Reviews Physics, vol. 3, no. 9, pp. 625– 644, 2021.

[24] F. Xiao, Y. Zhou, and W. Pedrycz, “An adaptive quantum circuit of dempster’s rule of combination for uncertain pattern classification,” Advances in Neural Information Processing Systems, vol. 38, pp. 17 848– 17 881, 2026.

[25] O. Shindi, Q. Yu, P. Girdhar, and D. Dong, “Model-free quantum gate design and calibration using deep reinforcement learning,” IEEE Transactions on Artificial Intelligence, vol. 5, no. 1, pp. 346–357, 2023.

[26] X. Deng, S. Xue, and W. Jiang, “A novel quantum model of mass function for uncertain information fusion,” Information Fusion, vol. 89, pp. 619–631, 2023.

[27] F. Xiao and W. Pedrycz, “Negation of the quantum mass function for multisource quantum information fusion with its application to pattern classification,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 2, pp. 2054–2070, 2022.

[28] X. Su, X. Huang, X. Pan, and D. Meng, “A dependence assessment method based on quantum model of mass function in human reliability analysis,” Expert Systems with Applications, vol. 299, p. 129992, 2026.

[29] Y. Wu, Q. Zhou, J. Geng, X. Deng, and W. Jiang, “Feature entanglementbased quantum multimodal fusion neural network,” arXiv preprint arXiv:2601.07856, 2026.

[30] X. Yang, H. Xing, F. Zhu, Y. Chen, D. Camacho, X. Dong, and W. Pedrycz, “Multimodal remote sensing of thunderstorm charge motion: A radar echo and electric field fusion approach,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–13, 2025.

[31] J. Shi, P. Du, W. Zeng, W. Wang, S. Peng, and X. Li, “Qsyncfold: quantum neural network for multidimensional sync-discovery in protein folding,” Briefings in Bioinformatics, vol. 27, no. 3, p. bbag234, 2026.

[32] J. Shi, R.-X. Zhao, W. Wang, S. Zhang, and X. Li, “Qsan: A nearterm achievable quantum self-attention network,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 8, pp. 13 995– 14 008, 2024.

[33] W. Jiang, Z. Lu, and D.-L. Deng, “Quantum continual learning overcoming catastrophic forgetting,” Chinese Physics Letters, vol. 39, no. 5, p. 050303, 2022.

[34] Y. Du, Y. Yang, D. Tao, and M.-H. Hsieh, “Problem-dependent power of quantum neural networks on multiclass classification,” Physical Review Letters, vol. 131, no. 14, p. 140601, 2023.

[35] N. A. Nghiem, S. Y.-C. Chen, and T.-C. Wei, “Unified framework for quantum classification,” Physical Review Research, vol. 3, no. 3, p. 033056, 2021.

[36] C. Zhang, Z. Lu, L. Zhao, S. Xu, W. Li, K. Wang, J. Chen, Y. Wu, F. Jin, X. Zhu et al., “Experimental demonstration of quantum continual learning with superconducting qubits,” npj Quantum Information, 2026.

[37] P. Yuan and S. Zhang, “Optimal (controlled) quantum state preparation and improved unitary synthesis by quantum circuits with any number of ancillary qubits,” Quantum, vol. 7, p. 956, 2023.

[38] S. Lloyd, M. Schuld, A. Ijaz, J. Izaac, and N. Killoran, “Quantum embeddings for machine learning,” arXiv preprint arXiv:2001.03622, 2020.

[39] T. Huang, Q. Zhang, W. Pedrycz, and S. Yang, “Individual linguistic granular computing: A granulation–degranulation-based approach,” IEEE Transactions on Cybernetics, 2026.

[40] D. Dong, X. Xing, H. Ma, C. Chen, Z. Liu, and H. Rabitz, “Learningbased quantum robust control: algorithm, applications, and experiments,” IEEE transactions on cybernetics, vol. 50, no. 8, pp. 3581–3593, 2019.

[41] N. Ezzell, E. M. Ball, A. U. Siddiqui, M. M. Wilde, A. T. Sornborger, P. J. Coles, and Z. Holmes, “Quantum mixed state compiling,” Quantum Science and Technology, vol. 8, no. 3, p. 035001, 2023.

[42] V. Travn´ ´ıcek, K. Bartkiewicz, A.ˇ Cernoch, and K. Lemr, “Experimental<sup>ˇ</sup> measurement of the hilbert-schmidt distance between two-qubit states as a means for reducing the complexity of machine learning,” Physical review letters, vol. 123, no. 26, p. 260501, 2019.

[43] N. Asadi, M. Davari, S. Mudur, R. Aljundi, and E. Belilovsky, “Prototype-sample relation distillation: towards replay-free continual learning,” in International conference on machine learning. PMLR, 2023, pp. 1093–1106.

[44] M. Cerezo, A. Poremba, L. Cincio, and P. J. Coles, “Variational quantum fidelity estimation,” Quantum, vol. 4, p. 248, 2020.

[45] I. Cong, S. Choi, and M. D. Lukin, “Quantum convolutional neural networks,” Nature Physics, vol. 15, no. 12, pp. 1273–1278, 2019.

[46] Y. Du, T. Huang, S. You, M.-H. Hsieh, and D. Tao, “Quantum circuit architecture search for variational quantum algorithms,” npj Quantum Information, vol. 8, no. 1, p. 62, 2022.

[47] I. MacCormack, C. Delaney, A. Galda, N. Aggarwal, and P. Narang, “Branching quantum convolutional neural networks,” Physical Review Research, vol. 4, no. 1, p. 013117, 2022.

[48] M. Schuld, A. Bocharov, K. M. Svore, and N. Wiebe, “Circuit-centric quantum classifiers,” Physical Review A, vol. 101, no. 3, p. 032308, 2020.

[49] E. Grant, M. Benedetti, S. Cao, A. Hallam, J. Lockhart, V. Stojevic, A. G. Green, and S. Severini, “Hierarchical quantum classifiers,” npj Quantum Information, vol. 4, no. 1, p. 65, 2018.

[50] S. A. Stein, B. Baheri, D. Chen, Y. Mao, Q. Guan, A. Li, S. Xu, and C. Ding, “Quclassi: A hybrid deep neural network architecture based on quantum state fidelity,” Proceedings of Machine Learning and Systems, vol. 4, pp. 251–264, 2022.

[51] Y. Wu, Y. Chen, L. Wang, Y. Ye, Z. Liu, Y. Guo, and Y. Fu, “Large scale incremental learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 374–382.

[52] K. Zhu, W. Zhai, Y. Cao, J. Luo, and Z.-J. Zha, “Self-sustaining representation expansion for non-exemplar class-incremental learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 9296–9305.

[53] P. Buzzega, M. Boschini, A. Porrello, D. Abati, and S. Calderara, “Dark experience for general continual learning: a strong, simple baseline,” Advances in neural information processing systems, vol. 33, pp. 15 920– 15 930, 2020.

[54] S. Hou, X. Pan, C. C. Loy, Z. Wang, and D. Lin, “Learning a unified classifier incrementally via rebalancing,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 831–839.

[55] T. Mensink, J. Verbeek, F. Perronnin, and G. Csurka, “Metric learning for large scale image classification: Generalizing to new classes at nearzero cost,” in European conference on computer vision. Springer, 2012, pp. 488–501.

[56] X. Tao, X. Hong, X. Chang, S. Dong, X. Wei, and Y. Gong, “Few-shot class-incremental learning,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 12 180– 12 189.

[57] L. Yu, B. Twardowski, X. Liu, L. Herranz, K. Wang, Y. Cheng, S. Jui, and J. Van de Weijer, “Semantic drift compensation for class-incremental learning,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 6980–6989.

[58] E. Belouadah and A. Popescu, “Deesil: Deep-shallow incremental learning.” in Proceedings of the European conference on computer vision (ECCV) workshops, 2018, pp. 0–0.

[59] Y. Liu, S. Parisot, G. Slabaugh, X. Jia, A. Leonardis, and T. Tuytelaars, “More classifiers, less forgetting: A generic multi-classifier paradigm for incremental learning,” in European Conference on Computer Vision. Springer, 2020, pp. 699–716.
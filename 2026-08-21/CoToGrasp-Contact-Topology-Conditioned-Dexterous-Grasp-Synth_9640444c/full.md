# CoToGrasp: Contact-Topology-Conditioned Dexterous Grasp Synthesis via Canonical Workspace Learning

Julien Mérand<sup>1</sup> , Boris Meden<sup>1</sup> , Liming Chen<sup>2</sup> , and Mathieu Grossard<sup>1</sup>

<sup>1</sup> Université Paris-Saclay, CEA, List, F-91120 Palaiseau, France 2 Ecole Centrale Lyon, CNRS, LIRIS, UMR5205, Institut Universitaire de France (IUF), F-69130 Ecully, France

Abstract. Current dexterous grasp planners primarily optimize for physical stability, focusing on whether an object can be grasped rather than how it should be grasped to support downstream functional tasks. However, conditioning grasp synthesis on specific human grasp taxonomies typically requires prohibitively expensive, object-annotated datasets. To address these limitations, we propose CoToGrasp, a novel generative framework that synthesizes diverse, stable grasps strictly conditioned on specific contact topologies. To bypass the data collection bottleneck, Co-ToGrasp is trained entirely in an object-agnostic manner. We introduce a feature-based canonical workspace that projects local object features into a unified gripper-centric domain, efectively decoupling the semantic functional intent from the arbitrary object geometry. By learning the intrinsic contact manifold of the gripper within this workspace, our model achieves zero-shot generalization to unseen objects at inference. Extensive evaluations on the large-scale DexGraspNet dataset demonstrate that CoToGrasp achieves state-of-the-art performance, outperforming existing taxonomy-guided planners. Finally, we demonstrate the physical viability and kinematic feasibility of our synthesized contact topologies on a physical robot platform. Code is available on our project website https://cea-list.github.io/cotograspweb/.

## 1 Introduction

As robotic systems evolve toward embodied agents capable of complex interaction, grasping can no longer be treated solely as a geometric stability optimization problem. The recent rise of humanoid robots, increasingly guided by high-level reasoning frameworks such as Large Language Models (LLMs) [18], demands grasps that directly answer functional, task-level intents. Fine-grained robot manipulation necessitates dexterous, multi-fingered hands with high Degrees of Freedom (DoF) because many downstream tasks – such as in-hand object reorientation, precision tool insertion, finger gaiting, and handle-based grasping – demand controllable, distributed multi-point contact topologies that fundamentally exceed the symmetric pinch and enveloping capabilities of simple parallel-jaw grippers.

![](images/041a5d9e7c8267b74b0ff90c798df24889180689ee55bb46c4cd24cfd6df451a.jpg)  
Fig. 1: Contact-Topology-Conditioned Grasp Synthesis. Given a desired semantic contact-topology condition (top left) – categorized into Precision, Object-Specific (highly constrained topologies tailored for specific tool use) or Power functional groups – and a novel, unseen object (bottom left), our framework synthesizes functionally diverse and physically stable grasps (right). Rather than learning contact topologies directly on the object geometry, we project local object features into a feature-based canonical workspace. This unified spatial representation efectively decouples the functional intent from the specific object identity. Within this workspace, we learn a latent manifold (center) that models the intrinsic contact capabilities of the gripper, enabling zero-shot generalization to diverse target geometries.

However, the transition to high-DoF systems exposes a critical limitation in existing grasp planners: while they can produce physically stable grasps, they lack the structural properties required to generate task-aligned contact topologies. Modern data-driven grasp planners predominantly focus on whether an object can be grasped rather than how it should be manipulated. By optimizing purely for geometric stability and force closure, these methods introduce a severe generative bias toward energetically stable but functionally uniform grasps. Although dexterous hands ofer rich articulated contact capabilities, current planners do not fully exploit their potential in terms of grasp pattern diversity, instead overwhelmingly defaulting to enveloping power grasps. Consequently, synthesizing a specific precision grasp required for a downstream task becomes highly ineficient, as it necessitates the generation and rejection of an impractical volume of functionally unsuitable candidates.

To achieve task-aligned grasp synthesis, it is necessary to introduce a structural prior over valid contact configurations. In this work, rather than relying on arbitrary stable grasps, we condition grasp synthesis on structured contact topologies derived from human grasp taxonomies. Specifically, we adapt the Gonzalez taxonomy [12], which categorizes grasps based strictly on the hand’s active contact surfaces rather than the object’s shape. This hand-centric formulation allows for seamless adaptation to anthropomorphic grippers independent of their specific kinematics. Furthermore, to avoid the severe generalization bottlenecks associated with explicitly mapping contact topologies to specific object geometries in training datasets, we propose building on the object-agnostic training paradigm introduced in [32]. By learning the intrinsic contact capabilities of the robotic hand independently of specific objects, we completely decouple grasp semantics from object topology.

We introduce CoToGrasp (Contact-Topology-Conditioned Dexterous Grasp Synthesis via Canonical Workspace Learning), a gripper-oriented framework for generating diverse, contact-topology-conditioned grasps. Unlike standard methods that learn contact maps directly on object point cloud, our approach utilizes a Canonical Feature-based Workspace anchored to the gripper frame. This workspace acts as a domain-agnostic bridge: we first extract local geometric features via a DGCNN [34] encoder from either the gripper (training) or the object (inference), then aggregate these features into fixed points within the canonical workspace using k-Nearest Neighbors (kNN). To capture the complex spatial structure of functional grasps, we treat each populated workspace point as a discrete token and process this set of workspace-anchored embeddings using a Transformer [40] encoder. This attention mechanism allows the network to learn structural constraints by modeling the specific spatial distribution and co-occurrence patterns of contact points inherent to each topology. These refined features are distilled into a compact latent descriptor via a Set Transformer [20], which conditions a Conditional Variational Auto-Encoder (CVAE) [37] to reconstruct probabilistic contact masks. Finally, we synthesize the physical grasp through a test-time energy-based optimization that aligns the gripper’s active surface with the predicted contact zones, enforcing kinematic constraints and collision avoidance.

We evaluate our method extensively in both simulation and real-world settings. We first demonstrate the severe functional bias of current taxonomyunaware planners by measuring the entropy of their retrieved contact topology distributions. We then compare CoToGrasp against state-of-the-art taxonomyguided methods, demonstrating superior topology compliance and stability.

In summary, our main contribution is the introduction of CoToGrasp, a novel object-agnostic grasp planner that synthesizes grasps conditioned on structured contact topologies derived from human taxonomies. Furthermore, to properly assess these capabilities, we establish a rigorous evaluation methodology to quantify the functional diversity and semantic bias inherent in dexterous grasp planners.

## 2 Related Work

Learning-Based Grasp Planners. The rise of humanoid robots demands robust, task-oriented multi-fingered grasp synthesis [13, 21]. However, contemporary data-driven grasp planners – whether directly predicting explicit joint configurations [15,16,25,27,42,44,46,50,52] or learning intermediate grasp representations [1, 17, 24, 43] – prioritize physical stability over functional intent.

Heavily reliant on synthetic databases [38, 41, 49] generated via analytical forceclosure resolution [26, 33], these models frequently sufer from mode collapse. They default to enveloping power grasps, underutilizing hand dexterity. Furthermore, explicitly mapping grippers to specific objects biases these models, hindering shape generalization. To circumvent this, frameworks like GOAG [32] utilize object-agnostic paradigms, demonstrating that learning contact topologies directly within the gripper’s canonical space yields superior generalization to novel geometries.

Human Taxonomies and Task-Oriented Semantics. To address how an object should be grasped, researchers draw from biomechanics. Cutkosky [7] established early categorizations based on object properties and task requirements, while Bullock et al. [3] considered the motion dynamics. Feix et al. [11] later proposed the comprehensive GRASP taxonomy. Crucially for haptics, Gonzalez et al. [12] synthesized these frameworks by categorizing 21 distinct contact topologies based strictly on the hand’s active contact surfaces.

Concurrently, task-oriented planners have leveraged these taxonomies to inject high-level semantics into robotic actions. Early studies utilized CNNs to decode manipulation semantics [8, 29, 36], while recent approaches employ Vision-Language Models [22,23] to guide grasp selection. However, bridging the gap between high-level language semantics and low-level physical interactions remains challenging. Methods like FunGrasp [14] retarget human-object interactions to grippers, but remain heavily dependent on task-specific interaction priors.

Taxonomy-Conditioned Grasp Synthesis. To explicitly control functional intent, recent methods condition their generative pipelines on taxonomy types. These representations vary from holistic couplings of joints, contacts, and object geometry [19, 28], to manual grouping [45], to purely kinematic "eigen grasps" [10, 31]. Recent frameworks like Dexonomy [5] and OmniDexVLG [51] represent types via explicit joint and contact combinations. Despite these advancements, current planners share a critical limitation: the deeply ingrained coupling of taxonomy types with specific object geometries during dataset construction. Methods like Dexonomy [5] require massive, analytically computed datasets where grasp types are strictly annotated against specific object meshes. If a novel object’s local geometry does not perfectly accommodate the rigid precomputed joint template, the optimization fails or collapses into a functionally incorrect type. CoToGrasp subverts this limitation by utilizing a purely handcentered taxonomy [12] based on semantic contact masks rather than rigid joint configurations. By training our generative model in a strictly object-agnostic manner, we entirely bypass the need for costly, object-annotated datasets.

## 3 Method

Figure 2 provides an overview of the CoToGrasp framework. Formally, we define a grasp as a tuple (R, t, Q), where $( R , \mathbf { t } ) \in S { \bar { O } } ( 3 ) \times \mathbb { R } ^ { 3 }$ represents the 6D pose of the gripper relative to the object frame O, and Q denotes the internal joint configuration. Our primary objective is to synthesize a diverse set of physically stable grasps that strictly respect a requested semantic contact topology.

![](images/ca1eccbbd1b0ca604d2bd50708ebfd7209f620884f19c6e791620f6146e9a0eb.jpg)  
Fig. 2: CoToGrasp Method Overview. The proposed framework operates in two distinct phases. Top (Object-Agnostic Training): The model learns an intrinsic, gripper-centric contact manifold within a canonical feature-based workspace, independent of object geometry. Bottom (Grasp Synthesis): At inference, a target object is transformed into the canonical frame. The network’s contact-topology-conditioned prediction is strictly filtered through a validation pipeline before energy-based optimization aligns the gripper to yield the final stable grasp (Q ).

To succesfully decouple high-level functional semantics from arbitrary object geometries, we adopt an object-agnostic learning paradigm. Unlike standard methods, our architecture is trained exclusively on gripper point clouds and inferred on target object point clouds. During the training phase, the model operates entirely within the canonical gripper frame, rendering it independent of the global pose (R, t). The network takes as input only the gripper’s local surface geometry and a semantic contact topology, learning to reconstruct the corresponding physical contact template mask.

## 3.1 Gripper-Oriented Contact Topology

Let $\mathcal { T } = \{ 1 , . . . , N _ { H } \}$ be the fixed index set of the points on the gripper’s active grasping surface – the specific subset of the gripper geometry designed to establish physical contact – representing a configuration-independent domain. We define the gripper handprint, $\mathcal { H } ( R , \mathbf { t } , Q )$ , as the spatial embedding of these points given a 6D pose (R, t) and a joint configuration Q:

$$
\mathcal { H } ( R , { \bf t } , Q ) = \{ { \bf h } _ { i } ( R , { \bf t } , Q ) \in \mathbb { R } ^ { 6 } \ | \ i \in \mathbb { Z } \}\tag{1}
$$

where ${ \bf h } _ { i } ( R , { \bf t } , Q )$ denotes the global Cartesian coordinates $( x , y , z )$ of the i-th point concatenated with its corresponding surface normal $( n _ { x } , n _ { y } , n _ { z } )$ . The hand-

![](images/2835e06bb1bfdcbc1aeb2cd29beb7b576f2bdcd3a3346db6f7ac61a0ad5345bf.jpg)  
Fig. 3: Semantic Grasp Taxonomy and Contact Mapping. (Left) Correspondences between the classical Feix [11] (F) taxonomy (top), the haptic Gonzalez [12] (M) taxonomy (middle row) and our derived point cloud contact templates $A _ { m }$ (bottom row). We categorized the 21 templates into three distinct functional groups: Precision, Power and Object-Specific (highly constrained topologies tailored for specific tool use). (Right) The 22 anatomical contact zones defined by Gonzalez (top) and the direct surjective mapping (ζ) onto our discrete gripper handprint H (bottom).

print H is strictly defined as the discretization of the gripper’s active grasping surfaces (detailed in Supp. Mat. A).

We adapt the taxonomy T from [12], consisting of $M = 2 1$ pairs of topology names and contact templates. Each template $\mathcal { A } _ { m } : \mathcal { T } \longrightarrow \{ 0 , \dots , N \}$ acts as a semantic mask, assigning a Zone ID to points required for contact topology m:

$$
\mathcal { T } = \{ ( \mathrm { N a m e } _ { m } ,  { A _ { m } } ) \ | \ m \in [ 1 , M ] \} , \qquad { A _ { m } } ( i ) = \left\{ \begin{array} { l l } { \zeta ( i ) } & { \mathrm { i f ~ } i \in  { S _ { m } } } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{2}
$$

where $\zeta : \mathcal { T } \to \{ 1 , \dots , N \}$ is a surjective mapping associating every hand point, from H, to a physical zone and $S _ { m }$ is the set of active points for contact topology m. This formulation, illustrated in Figure 3, enables the systematic adaptation of the human-centric taxonomy [12] to anthropomorphic grippers. Crucially, this taxonomy serves purely as a labeling interface. CoToGrasp easily adapts to any alternative contact-based representation.

## 3.2 Geometric Transfer via Canonical Workspace Learning

The Duality of Contact. Contact between a gripper and an object is fundamentally a symmetric geometric relation: a point on the gripper surface is in contact with the object if and only if a corresponding object point is in contact with the gripper. [17, 24] formulate contact detection as an object-centric problem, parameterizing the search for valid grasp locations over the object surface O to define the contact set $\mathcal { C } _ { \mathrm { o b j } } \subset \mathcal { O }$ . Conceptually, they ask, "where on the object can I touch?". However, this duality allows us to invert the paradigm. Instead, we ask, "which regions of my hand are activated by this object?" By flipping the formulation, we define the gripper contact set $\mathcal { C } _ { \mathrm { g r i p } }$ directly on the hand’s surface. We reformulate the contact condition as:

$$
\mathcal { C } _ { \mathrm { g r i p } } ( \mathcal { H } ( R , \mathbf { t } , Q ) ) = \{ h _ { i } \in \mathcal { H } ( R , \mathbf { t } , Q ) \ | \ \operatorname* { m i n } _ { o \in \mathcal { O } } \mathcal { D } _ { \mathrm { a l i g n e d } } ( h _ { i } , o ) < \epsilon \}\tag{3}
$$

with the distance $\mathcal { D } _ { a l i g n e d }$ defined between any two points x and $\mathbf { y } ,$ , introduced in [24], as:

$$
\mathcal { D } _ { a l i g n e d } ( \mathbf { x } , \mathbf { y } ) = e ^ { \gamma ( 1 - \langle \mathbf { x } - \mathbf { y } , n _ { \mathbf { x } } \rangle ) } \sqrt { \| \mathbf { x } - \mathbf { y } \| _ { 2 } }\tag{4}
$$

where $n _ { \mathbf { x } }$ the surface normal at x and $\gamma$ is a scaling factor modulating the influence of the kinematic alignment compared to the pure Euclidean distance.

This duality matters profoundly because it fundamentally changes the learning domain. The object surface O represents an unbounded domain with infinite variability and high geometric entropy. In contrast, the gripper surface H possesses a fixed structure, a bounded domain, and a finite kinematic manifold. By shifting the formulation from an unbounded object space to the fixed, mechanism-intrinsic gripper manifold, the solution space $\mathcal { C } _ { \mathrm { g r i p } }$ is strictly bounded by H, simplifying the generative task. Consequently, the network avoids memorizing arbitrary target geometries, instead learning grasp templates purely as fundamental, object-agnostic capabilities of the gripper.

Shift to a Gripper-Centric Frame. Conceptually, this duality allows us to invert the traditional learning paradigm by anchoring our perspective entirely on a canonical gripper frame, where the palm rests permanently at the origin $( R = I , \mathbf { t } = \mathbf { 0 } )$ . Here, the handprint’s spatial embedding, $\mathcal { H } ( Q )$ , depends strictly on the internal joint configuration $Q ,$ completely isolated from the global grasp pose. The target object $\bar { \boldsymbol { \mathcal { O } } }$ is brought into this shared space via the inverse transformation $( R , { \bf t } ) ^ { - 1 }$ , yielding $\tilde { \mathcal { O } }$ . Crucially, because physical contact involves opposing surfaces $\left( \operatorname { E q } . \ 4 \right)$ , we invert the object’s surface normals, transforming its local geometry into a negative mold of the expected gripper surface.

Feature-Based Workspace Representation. $\mathrm { A }$ central challenge arises from the fact that training and inference operate on geometrically distinct domains. Directly learning over these heterogeneous domains would entangle contact reasoning with object-specific topology. To resolve this, we introduce a canonical feature-based workspace $\mathcal { W } = \{ w _ { j } \in \mathbb { R } ^ { 3 } \} _ { j = 1 } ^ { N _ { W } }$ , defined as a fixed set of spatial basis points anchored to the gripper frame.

To capture fine-grained surface details, we extract a dense local description from the input geometry $\mathcal { P } = \{ \mathbf { p } _ { i } \in \mathbb { R } ^ { 6 } \} _ { i = 1 } ^ { N _ { P } }$ . We employ a modified DGCNN [34] to extract point-wise local features $\mathcal { F } _ { \mathcal { P } } = \{ { \bf f } _ { i } \} _ { i = 1 } ^ { N _ { P } }$

To disentangle these features from the input topology, we project them onto the fixed basis W using a weighted k-Nearest Neighbors (kNN) aggregation. To ensure surface orientation strictly dictates contact viability, we gate the aggregation using the aligned distance $\left( \mathcal { D } _ { \mathrm { a l i g n e d } } \right)$ . Let ${ \mathcal { N } } _ { j }$ denote the indices of the k nearest points in $\mathcal { P }$ to the workspace basis point $w _ { j }$ . The aggregated feature $\mathcal { F } _ { \mathcal { W } _ { j } }$ is computed as:

$$
\mathcal { F } _ { \mathcal { W } _ { j } } = \frac { 1 } { k } \sum _ { i \in \mathcal { N } _ { j } } \mathbf { f } _ { i } \cdot e ^ { - \mathcal { D } _ { \mathrm { a l i g n e d } } \left( \mathbf { p } _ { i } , w _ { j } \right) }\tag{5}
$$

By embedding local geometric features directly into the canonical workspace basis W, we construct a consistent set of spatial tokens, efectively decoupling the semantic contact reasoning from the arbitrary input geometry (further projection details and justifications are provided in Supp. Mat. B).

Projected Ground Truth Templates. To construct the supervisory signal used during training, we project the templates $A _ { m }$ onto the canonical workspace W. We strictly gate this projection using the distance threshold $\epsilon = 0 . 8 .$ , consistent with our contact formulation in Eq. 3. Each workspace point $w _ { j }$ is assigned a label according to its nearest kinematically aligned hand point:

$$
\begin{array} { r } { A _ { m } ( j ) = \left\{ \begin{array} { l l } { { \mathcal A } _ { m } ( i ^ { * } ) } & { \mathrm { i f ~ } { \mathcal D } _ { \mathrm { a l i g n e d } } ( { \mathbf h } _ { i ^ { * } } ( Q ) , w _ { j } ) < \epsilon } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \end{array}\tag{6}
$$

where i<sup>∗</sup> = argmin $\mathcal { D } _ { \mathrm { a l i g n e d } } ( \mathbf { h } _ { i } ( Q ) , w _ { j } )$ $\stackrel { \triangledown } { i } \in \boldsymbol { \mathcal { I } }$

Workspace points lying beyond this threshold are assigned the label 0, indicating no physical contact. Consequently, $\smash { \varLambda _ { m } \in \{ 0 , \ldots , N \} ^ { N _ { W } } }$ serves as the explicit ground truth contact map.

## 3.3 Learning Contact Topology Templates using Self-Attention

To model non-local dependencies between potential contact regions, we process the workspace embeddings using a Transformer [40] encoder. To prevent the semantic signal of the requested contact topology from diluting across deep attention layers, we explicitly concatenate the learnable topology embedding $\mathcal { F } _ { \mathcal { T } }$ to every workspace point feature $\mathcal { F } _ { \mathcal { W } }$ , denoted ${ \mathcal { F } } _ { \mathcal { W } \oplus \mathcal { T } }$ . We inject spatial awareness via sinusoidal positional encodings (PE), allowing the multi-head attention mechanism to recover spatial relationships purely from the fixed basis indices:

$$
\varPhi = \mathrm { T r a n s f o r m e r } ( \mathcal { F } _ { \mathcal { W } \oplus \mathcal { T } } + \mathrm { P E } )\tag{7}
$$

To bridge the modality gap, the projected templates $\varLambda _ { m }$ are passed through a Multi-Layer Perceptron (MLP) to create a continuous embedding ${ \mathcal { F } } _ { A }$ . These are fused with the transformer features Φ and compressed into a global latent descriptor $\mathcal { Z }$ using a Set Transformer [20]. We leverage a Pooling by MultiHead Attention (PMA) block and a Set Attention Block (SAB) to capture complex structural dependencies:

$$
\mathcal { Z } = \mathrm { S A B } ( \mathrm { P M A } _ { s } ( \phi \oplus \mathcal { F } _ { A } ) )\tag{8}
$$

Finally, a CVAE [37] models the localized, intra-topology stochasticity of the contact patches. Because macroscopic grasp multi-modality is explicitly resolved by the topology conditioning, a standard unimodal Gaussian prior $\mathcal { N } ( 0 , I )$ proves highly eficient. A convolutional encoder maps $\mathcal { Z }$ to the posterior distribution $q ( z | \mathcal { Z } )$ , from which a latent variable $z \in \mathbb { R } ^ { \psi }$ is sampled. The decoder, conditioned on Φ via Adaptive Normalization [47] layers (AdaLN), modulates the features to output the predicted zone probabilities $\hat { A } _ { m }$

The network is trained end-to-end to minimize the standard variational lower bound (ELBO), consisting of a Multi-Class Cross-Entropy reconstruction loss over the workspace points and a Kullback-Leibler regularization term weighted by β. (See Supp. Mat. B for architectural justifications and loss formulations.)

## 3.4 Grasp Synthesis Through Contact Points Generation

In the inference phase, our goal is to synthesize a valid joint configuration Q that realizes a specific contact topology m for an object O at a candidate pose $( R , \mathbf { t } )$ We first transform the object point cloud into the canonical gripper frame via the inverse rigid transformation $( R , { \bf t } ) ^ { - 1 }$ . This transformed geometry $\tilde { \mathcal { O } }$ , along with the requested contact topology m, is fed into the network – bypassing the latent encoder by sampling z from the prior $\mathcal { N } ( 0 , I ) - \mathrm { t o }$ predict a contact template $\hat { A } _ { m }$ over the workspace. To ensure the physical viability of the grasp before costly optimization, we enforce a strict cascading validation pipeline:

Label-Consistency Check. A contact topology is fundamentally constrained by the object’s local geometry. To prevent generating ill-posed grasps, we apply a template-consistency filter immediately after inference. We perform a strict binary check on the symmetric diference between the ground-truth $\left( { \varLambda } _ { m } \right)$ and predicted $( \hat { A } _ { m } )$ templates. We allow a maximum deviation of one missing or hallucinated contact zone, safely rejecting fundamentally distinct failures where the object cannot support the requisite contacts.

Contact Points Force-Closure Validation. For candidates passing the labelconsistency check, we assess the potential grasp stability using the rapid, approximated force-closure estimation proposed in [32]. Specifically, we extract the 3D workspace points $w _ { j } \in \mathcal W$ assigned a non-zero label by $\hat { A } _ { m }$ . We compute the barycenters of these active regions to construct the theoretical grasp wrench space. Crucially, if the force-closure condition is not met, it implies the proposed contact distribution is physically unviable. In this case, we discard the prediction and exploit the generative capacity of our network by resampling the latent variable z to produce a fresh contact template $\hat { A } _ { m }$ . We limit this resampling process to a maximum of 20 iterations to prevent exhaustive searches in poorly conditioned poses. (Detailed formulations are provided in Supp. Mat. D).

Joint Configuration Optimization. Finally, we retrieve the optimal joint configuration $Q ^ { * }$ by minimizing an energy function that aligns the gripper’s active surfaces with the predicted 3D spatial workspace points, while respecting kinematic constraints:

$$
Q ^ { * } = \underset { Q } { \mathrm { a r g m i n } } \left( \lambda _ { d } E _ { d i s t } + \lambda _ { p } E _ { p e n } + \lambda _ { s } E _ { s p e n } + \lambda _ { j } E _ { j } + \lambda _ { r } E _ { r e p } \right)\tag{9}
$$

We adopt the standard terms $E _ { d i s t }$ (contact alignment), $E _ { p e n }$ (object penetration), $E _ { s p e n }$ (self-collision), and $E _ { j }$ (joint limits) used in [17,24,32]. Additionally, to best match the desired contact topology without prior shape knowledge, we introduce a repulsive term $E _ { r e p } .$ This term creates a repulsive field around the object for all unused fingers and palm regions (i.e., those not assigned a valid label in $\hat { \varLambda _ { m } } )$ , enforcing a minimum safety margin of 5 mm to guide the optimizer toward strictly topology-compliant, collision-free configurations. (Detailed mathematical formulations and weighting factors are provided Supp. Mat. D).

## 4 Experiments

## 4.1 Experimental Setup

Training Data Formulation. To construct the object-agnostic training dataset, we first define the semantic mapping between the physical surface regions of the target gripper and the required active contact zones $( N = 2 2 )$ for the $M = 2 1$ contact topology templates. We uniformly sample 10, 000 kinematically valid joint configurations Q and compute their corresponding spatial handprints H(Q) via forward kinematics. By pairing every sampled handprint with all 21 contact templates, we construct a comprehensive training dataset of 210,000 unique data points. Crucially, because this formulation operates entirely within the canonical gripper frame, dataset generation requires zero object meshes or computationally expensive physical simulations. (Further implementation and training details are provided in Supp. Mat. A).

Grasp Pose Constraints and Generation. Because CoToGrasp operates in a canonical, gripper-centric workspace (Sec. 3.2), it inherently decouples the global 6D grasp pose $( R , \mathbf { t } ) \in S O ( 3 ) \times \mathbb { R } ^ { 3 }$ from local contact generation. By treating this pose as an external prior rather than inferring it end-to-end, our architecture allows task planners or human operators to dictate functionally suitable approaches. This modularity facilitates advanced applications like kinematic keyframing for in-hand manipulation, enabling the synthesis of continuous, topology-consistent grasps along defined object trajectories.

To autonomously evaluate CoToGrasp without external planners, we sample diverse candidate poses using a topology-conditioned heuristic depending on whether the requested topology involves the palm or relies strictly on distal precision. (Detailed sampling formulations are provided in Supp. Mat. E).

Automatic Topology Recognition from Grasp. To verify that CoToGrasp accurately synthesizes the specified contact topologies, we introduce an automated classification step for the generated grasps. First, we extract the subset of gripper points in physical contact with the object (using Eq. 3) and map them back to their semantic zone identifiers (visible in Fig. 3). We then compare these observed active zones against the ground-truth required zones of each target template $A _ { m }$ . We evaluate this match by computing a similarity score $s _ { m }$ based on the Tversky index [39]. Crucially, we apply an asymmetric penalization that punishes extra, unintended contact regions more strictly than missing ones, efectively penalizing clumsy or degenerated grasps. Finally, a generated grasp is evaluated against all M templates and assigned the class m yielding the highest similarity score. To ensure robustness and mitigate false positives, any grasp failing to reach a minimum similarity threshold $\left( s _ { m } < 0 . 5 , \forall m \in \left[ 1 , M \right] \right)$ is strictly classified as "unknown". (Detailed mathematical formulations of this metric are provided in Supp. Mat. F).

<table><tr><td rowspan="2">Method</td><td rowspan="2">Object-Agnostic Training</td><td rowspan="2">SR ↑</td><td rowspan="2"> $\mathbf { H } _ { T C } \uparrow$ </td><td rowspan="2">Speed (sec.t/(gràsps)</td><td colspan="2">Diversity (avg.)↑</td></tr><tr><td>R (rad) Q (rad)</td><td></td></tr><tr><td>DFC [26]</td><td>V</td><td>72.15</td><td>0.7389</td><td>&gt;1800</td><td>0.0607 1.424</td><td>0.3579</td></tr><tr><td>GenDexGrasp [24]</td><td>x</td><td>71.15</td><td>0.5956</td><td>14.65</td><td>0.0519 1.416</td><td>0.2567</td></tr><tr><td>DRO-Grasp [43]</td><td>x</td><td>63.30</td><td>0.6504</td><td>1.72</td><td>0.0546 1.515</td><td>0.2892</td></tr><tr><td>GOAG [32]</td><td>√</td><td>77.90</td><td>0.6527</td><td>0.20</td><td>0.0479 1.401</td><td>0.3170</td></tr><tr><td>CoToGrasp</td><td>V</td><td>36.94</td><td>0.83</td><td>0.11</td><td>0.0674 1.4927</td><td>0.3458</td></tr></table>

Table 1: Comparison with taxonomy-unaware baselines. CoToGrasp achieves the highest semantic entropy (H<sub>TC</sub>) and generation speed, overcoming the functional mode collapse typical of unconditioned planners.

## 4.2 Evaluation Metrics

To evaluate CoToGrasp, we utilize a combination of standard physical stability metrics and novel semantic compliance metrics. For physical stability (Success Rate, SR), Generation Speed, and Spatial Diversity, we strictly follow the evaluation protocols established in [24, 32, 43] (Detailed formulations for these metrics are provided in Supp. Mat. G).

To quantify the functional accuracy and distributional fairness of the generated grasps, we introduce the following novel metrics:

Topology Compliance (TC). This metric evaluates how accurately the generated grasp adheres to the functional constraints of the requested contact topology. Across the M evaluated topologies, the average TC is defined as:

$$
\mathbf { T C } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \frac { \mathrm { E f f e c t i v e } _ { m } } { \mathrm { A t t e m p t } _ { m } }\tag{10}
$$

where $\mathrm { A t t e m p t } _ { m }$ is the total number of simulated grasps conditioned on target topology $m ,$ and ${ \mathrm { E f f e c t i v e } } _ { m }$ is the subset of those grasps that successfully satisfy the valid semantic contact template for topology m.

Entropy. To evaluate the topological diversity and assess potential mode collapse, we compute the normalized Shannon entropy across all contact topologies:

$$
\mathbf { H } = \frac { - \sum _ { m = 1 } ^ { M } \tilde { f } _ { m } \ln ( \tilde { f } _ { m } ) } { \ln ( M ) }\tag{11}
$$

A value of H → 1 indicates a uniform, unbiased generative distribution, whereas H → 0 reflects severe mode collapse toward a few dominant topologies. We report two distinct variants of this metric:

Stability Entropy $( \mathbf { H } _ { S R } ) { : }$ Evaluates if the model generates physically stable grasps equally well across all topologies. Here, $\tilde { f } _ { m }$ is the normalized SR of topology m, defined as $\begin{array} { r } { \tilde { f } _ { m } = \mathbf { S } \mathbf { R } _ { m } / \tilde { \sum _ { i = 1 } ^ { M } } \mathbf { S } \mathbf { R } _ { j } } \end{array}$

Semantic Entropy $( \mathbf { H } _ { T C } ) { \vdots }$ Evaluates if the model preserves true functional intent without defaulting to simpler power grasps. Here, $\tilde { f } _ { m }$ is the normalized TC of the topology m, defined as $\begin{array} { r } { \tilde { f } _ { m } = \mathbf { T C } _ { m } / \sum _ { j = 1 } ^ { M } \mathbf { T C } _ { j } } \end{array}$

![](images/7d2ddfdbb10c3fd4def13d4ac5a54e292624bf218e985bd3378eee8280a43c37.jpg)  
Fig. 4: Functional contact topology distribution across taxonomy-unaware planners. The histogram illustrates the distribution of grasps generated by unconditioned baselines compared to CoToGrasp on the Multidex objects set. The unknown category represents physically stable grasps with unnatural contact patterns that fail to match any contact topology. Notably, unconditioned baselines exhibit a severe generative bias (mode collapse) toward enveloping power grasps (red box).

## 4.3 Overall Performances

We evaluate CoToGrasp against state-of-the-art baselines using the Shadow Hand. First, we evaluate against unconditioned generative planners to highlight the necessity of topology conditioning. Second, we evaluate against a taxonomyaware baseline to demonstrate CoToGrasp’s superior functional control. For all experiments, we cap generation at 20 attempts per topology. This resampling budget mitigates heuristic approach poses (R, t) that are geometrically incompatible with the requested contact topology.

Taxonomy-Unaware Grasp Planners Analysis and Comparison. Evaluating on a MultiDex subset, Table 1 shows unconditioned baselines [24,26,32,43] achieve higher success rates (SR) by naturally defaulting to geometrically stable, enveloping power grasps (Fig. 4). Consequently, they struggle with precision grasps, yielding low semantic entropy $\left( \mathbf { H } _ { T C } \right)$ . Conversely, CoToGrasp achieves a balanced topology distribution and the highest $\mathbf { H } _ { T C }$ . Its overall SR (36.94%, with a Top-5 average of 56.18%) occurs because it is forced to synthesize diverse topologies (e.g., precision pinches) regardless of an object’s natural geometric afordances, inherently prioritizing target contacts over simulated physics stability. While methods like DFC show high spatial variance (t, R, Q), this primarily reflects joint range exploitation rather than functional diversity. Furthermore, CoToGrasp demonstrates the fastest generation speed at 0.11s per grasp.

Taxonomy-Aware Grasp Synthesis. To evaluate precise functional control, we compare CoToGrasp against Dexonomy [5] on the DexGraspNet [41] test set. After mapping Dexonomy’s Feix [11] taxonomy to our contact-based representations for direct comparison (Fig. 3), Table 2 shows CoToGrasp achieves higher SR and TC (17.18% vs. 14.28%) across all functional categories. While absolute TC scores appear modest, they reflect the extreme stringency of our asymmetric metric, which heavily penalizes the minor, physically necessary finger adjustments naturally occurring during dynamic simulation. Outperforming the baseline under these strict conditions demonstrates CoToGrasp’s superior zero-shot functional control. Dexonomy [5] struggles with TC because its conditioning is rigidly coupled to explicit joint configurations end-to-end; when object geometry forces deviations, the resulting grasp frequently violates the requested topology. Conversely, CoToGrasp robustly projects valid semantic contact zones via object-agnostic training. The ablation of our validation pipeline (No Check) demonstrates a sharp performance drop, emphasizing that verifying topological viability against the object’s local geometry via the Label-Consistency check is critical before optimization. (A detailed analysis of TC bottlenecks before and after physics simulation is provided in Supp. Mat. I).

<table><tr><td rowspan="2">Method</td><td colspan="5">SR (%)</td><td rowspan="2"> $\mathbf { H } _ { S R }$ </td><td rowspan="2">TC (%)|</td><td rowspan="2"> $\mathbf { H } _ { T C }$ </td></tr><tr><td>Power Precision Obj. Spe.|Avg. Topo.|Avg. Obj.</td><td></td><td></td><td></td><td></td></tr><tr><td>Dexonomy [5] CoToGrasp</td><td>27.16 29.75</td><td>12.36 22.71</td><td>19.62 25.50</td><td>21.13 26.72</td><td>23.80 27.56</td><td>0.91 |0.96</td><td>14.28 17.18</td><td>0.77 0.84</td></tr><tr><td>CoToGrasp (w/o Label-Consistency)|</td><td>25.11</td><td>14.77</td><td>20.85</td><td>21.14</td><td>22.97</td><td>0.94</td><td>14.45</td><td>0.81</td></tr><tr><td>CoToGrasp (w/o Force-Closure)</td><td>26.90</td><td>14.87</td><td>21.08</td><td>22.08</td><td>23.65</td><td>0.95</td><td>16.26</td><td>0.81</td></tr><tr><td>CoToGrasp (No Check)</td><td>25.09</td><td>14.73</td><td>20.58</td><td>21.06</td><td>23.00</td><td>0.94</td><td>14.72</td><td>0.81</td></tr></table>

Table 2: Taxonomy-aware grasp synthesis. CoToGrasp outperforms the baseline in both physical stability (SR) and semantic accuracy (TC), particularly on highly constrained precision grasps.

Per-Topology Analysis. Table 3 highlights CoToGrasp’s distinct advantage in synthesizing highly constrained precision grasps (e.g. M2: 30.3% vs. 10.5%). An apparent pseudo-anomaly occurs with topologies M11 (Object-Specific) and M21 (Power), where Dexonomy [5] reports artificially inflated SR (60.5% and 37.2%). Qualitative analysis reveals this is driven by severe mode collapse: when faced with challenging geome-

<table><tr><td rowspan="2">Method</td><td colspan="3">SR (%)</td></tr><tr><td>M2 M6</td><td>M11 M13 M18 M21</td></tr><tr><td>Dexonomy [5] CoToGrasp</td><td>10.5 15.2 60.5† 30.3 21.7 29.829.6 31.3 33.5</td><td>20.3 29.6 37.2†</td></tr></table>

Table 3: Per-Topology results: 3 Power grasps (M13, M18, M21), 2 Precision (M2, M6) and 1 Object-Specific (M11). <sup>†</sup>Indicates artificially inflated scores due to mode collapse, where Dexonomy defaults to unverified enveloping grasps.

tries, Dexonomy [5] frequently abandons the conditional query and defaults to an unverified M21 enveloping grasp. Because it lacks a strict posterior topological check, it erroneously records these power grasps as successes for diferent input topologies (like M11). Conversely, CoToGrasp explicitly detects and rejects hallucinated contacts, maintaining a competitive true-positive rate for M21 (33.5%) without sacrificing the integrity of the true TC metric on precision tasks.

## 4.4 Real-World Validation

To demonstrate the physical viability and kinematic feasibility of the grasps synthesized by CoToGrasp, we conducted real-world validation experiments. These were performed in a tabletop environment using a 6-DoF Universal Robots UR10 manipulator equipped with an Allegro Right Hand. Because the Allegro Hand is a four-fingered anthropomorphic gripper (lacking a little finger), we explicitly suppressed the generation of contact topology M6, as it strictly requires five fingers to form a valid contact template. We successfully planned and executed grasps on diverse objects from the YCB [4] dataset. Crucially, we empirically validated the functional diversity of our method by demonstrating that the generated grasp for various contact topologies can be successfully formed and held statically on the physical system. A qualitative overview of the successful realworld grasps, categorized by their respective contact topologies, is presented in Figure 5. Execution sequences are provided in the supplementary video material. The inherent hardware challenges and control limitations of executing precise semantic grasps using a multifingered gripper like the Allegro Hand are discussed extensively in Supp. Mat. J.

![](images/cfea554355c4029a8c285a9e09bbdf34f7b6962ef6334954770618c840931251.jpg)  
Fig. 5: Real-World kinematic validation. CoToGrasp synthesizes diverse, topology-compliant grasps that are physically executable on a physical Allegro Hand using YCB [4] objects. The target contact topologies (indicated above each frame) demonstrate the physical viability of the generated grasps across both precision and power categories.

## 5 Conclusion

This paper introduces CoToGrasp, a novel generative framework for contacttopology-conditioned dexterous grasp synthesis. By fundamentally decoupling functional intent from object identity, our approach efectively mitigates the severe mode collapse observed in contemporary grasp planners. This allows CoToGrasp to synthesize highly constrained, topology-compliant precision and power grasps on unseen geometries without relying on costly object-annotated datasets. Extensive evaluations demonstrate that our validation pipeline robustly filters topologically invalid configurations, yielding state-of-the-art semantic diversity and physical stability. Finally, successful real-world deployments on the Allegro Hand confirm that the structural properties of our generated contact topologies are physically executable on a real robot platform.

## Acknowledgments

This publication was made possible by the use of the FactoryIA supercomputer, financially supported by the Ile-De-France Regional Council. Experiments presented in this paper were carried out thanks to a platform funded by DIM AI4IDF and PRAIRIE-PSAI. The authors would like to thank Timothée Carecchio for his valuable assistance in achieving the experimental results presented in this paper. This project has received funding from the European Union’s Horizon Europe research and innovation program under grant agreement $\mathrm { n } ^ { \mathrm { 0 } }$ 101135708 (JARVIS Project). Liming Chen in this research was in part supported by the French Research Agency ANR, l’Agence Nationale de Recherche, through the projects Aristotle (ANR-21-FAI1-0009-01), Astérix (ANR-23-EDIA-0002), DEMETER (ANR-25-HTCE-0002) and PROTEUS (ANR-25-TSIA-0011- 01), and the French national investment prioritary program through the PSPC FAIR WASTE project.

## References

1. Attarian, M., Asif, M.A., Liu, J., Hari, R., Garg, A., Gilitschenski, I., Tompson, J.: Geometry matching for multi-embodiment grasping. In: Conference on Robot Learning (2023)

2. Brahmbhatt, S., Ham, C., Kemp, C.C., Hays, J.: Contactdb: Analyzing and predicting grasp contact via thermal imaging. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2019)

3. Bullock, I.M., Ma, R.R., Dollar, A.M.: A hand-centric classification of human and robot dexterous manipulation. IEEE transactions on Haptics (2012)

4. Calli, B., Singh, A., Walsman, A., Srinivasa, S., Abbeel, P., Dollar, A.M.: The ycb object and model set: Towards common benchmarks for manipulation research. In: 2015 international conference on advanced robotics (ICAR) (2015)

5. Chen, J., Ke, Y., Peng, L., Wang, H.: Dexonomy: Synthesizing all dexterous grasp types in a grasp taxonomy. Robotics: Science and Systems (2025)

6. Chitta, S., Sucan, I., Cousins, S.: Moveit![ros topics]. IEEE robotics & automation magazine (2012)

7. Cutkosky, M.R., et al.: On grasp choice, grasp models, and the design of hands for manufacturing tasks. IEEE Transactions on robotics and automation (1989)

8. Deng, Z., Fang, B., He, B., Zhang, J.: An adaptive planning framework for dexterous robotic grasping with grasp type detection. Robotics and Autonomous Systems (2021)

9. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021)

10. Fang, H.S., Yan, H., Tang, Z., Fang, H., Wang, C., Lu, C.: Anydexgrasp: General dexterous grasping for diferent hands with human-level learning eficiency. arXiv preprint arXiv:2502.16420 (2025)

11. Feix, T., Romero, J., Schmiedmayer, H.B., Dollar, A.M., Kragic, D.: The grasp taxonomy of human grasp types. IEEE Transactions on human-machine systems (2015)

12. Gonzalez, F., Gosselin, F., Bachta, W.: Analysis of hand contact areas and interaction capabilities during manipulation and exploration. IEEE transactions on haptics (2014)

13. Gu, Z., Li, J., Shen, W., Yu, W., Xie, Z., McCrory, S., Cheng, X., Shamsah, A., Grifin, R., Liu, C.K., et al.: Humanoid locomotion and manipulation: Current progress and challenges in control, planning, and learning. IEEE/ASME Transactions on Mechatronics (2026)

14. Huang, L., Zhang, H., Wu, Z., Christen, S., Song, J.: Fungrasp: Functional grasping for diverse dexterous hands. IEEE Robotics and Automation Letters (2025)

15. Huang, S., Wang, Z., Li, P., Jia, B., Liu, T., Zhu, Y., Liang, W., Zhu, S.C.: Difusion-based generation, optimization, and planning in 3d scenes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2023)

16. Jiang, H., Liu, S., Wang, J., Wang, X.: Hand-object contact consistency reasoning for human grasps generation. In: Proceedings of the IEEE/CVF international conference on computer vision (2021)

17. Khargonkar, N., Casas, L.F., Prabhakaran, B., Xiang, Y.: Robotfingerprint: Unified gripper coordinate space for multi-gripper grasp synthesis and transfer. In: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (2025)

18. Kim, Y., Kim, D., Choi, J., Park, J., Oh, N., Park, D.: A survey on integration of large language models with intelligent robots. Intelligent Service Robotics (2024)

19. Kleer, N., Keil, O., Feick, M., Gomaa, A., Schwartz, T., Feld, M.: Incorporation of the intended task into a vision-based grasp type predictor for multi-fingered robotic grasping. In: 2024 33rd IEEE International Conference on Robot and Human Interactive Communication (ROMAN) (2024)

20. Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., Teh, Y.W.: Set transformer: A framework for attention-based permutation-invariant neural networks. In: International conference on machine learning, PMLR (2019)

21. Li, G., Wang, R., Xu, P., Ye, Q., Chen, J.: The developments and challenges towards dexterous and embodied robotic manipulation: A survey. IEEE Robotics & Automation Magazine (2025)

22. Li, H., Mao, W., Deng, W., Meng, C., Fan, H., Wang, T., Osamu, Y., Tan, P., Wang, H., Deng, X.: Multi-graspllm: A multimodal llm for multi-hand semantic guided grasp generation. arXiv preprint arXiv:2412.08468 (2024)

23. Li, K., Wang, J., Yang, L., Lu, C., Dai, B.: Semgrasp: Semantic grasp generation via language aligned discretization. In: European Conference on Computer Vision (2024)

24. Li, P., Liu, T., Li, Y., Geng, Y., Zhu, Y., Yang, Y., Huang, S.: Gendexgrasp: Generalizable dexterous grasping. In: 2023 IEEE International Conference on Robotics and Automation (ICRA) (2023)

25. Liu, M., Pan, Z., Xu, K., Ganguly, K., Manocha, D.: Deep diferentiable grasp planner for high-dof grippers. arXiv preprint arXiv:2002.01530 (2020)

26. Liu, T., Liu, Z., Jiao, Z., Zhu, Y., Zhu, S.C.: Synthesizing diverse and physically stable grasps with arbitrary hand structures using diferentiable force closure estimator. IEEE Robotics and Automation Letters (2021)

27. Lu, J., Kang, H., Li, H., Liu, B., Yang, Y., Huang, Q., Hua, G.: Ugg: Unified generative grasping. In: European Conference on Computer Vision (2024)

28. Lu, Q., Hermans, T.: Modeling grasp type improves learning-based grasp planning. IEEE Robotics and Automation Letters (2019)

29. Lu, Q., Van der Merwe, M., Sundaralingam, B., Hermans, T.: Multifingered grasp planning via inference in deep neural networks: Outperforming sampling by learning diferentiable models. IEEE Robotics & Automation Magazine (2020)

30. Makoviychuk, V., Wawrzyniak, L., Guo, Y., Lu, M., Storey, K., Macklin, M., Hoeller, D., Rudin, N., Allshire, A., Handa, A., et al.: Isaac gym: High performance gpu based physics simulation for robot learning. In: NeurIPS Datasets and Benchmarks (2021)

31. Mao, C., Yuan, H., Huang, Z., Xu, C., Ma, K., Lu, Z.: Demofungrasp: Universal dexterous functional grasping via demonstration-editing reinforcement learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 986–995 (2026)

32. Mérand, J., Meden, B., Grossard, M., Chen, L.: Goag: Generative and objectagnostic grasp planner for dexterous robotic manipulation. In: IEEE Int. Conf. Intelligent Robots and Systems (2026)

33. Miller, A.T., Allen, P.K.: Graspit! a versatile simulator for robotic grasping. IEEE Robotics & Automation Magazine (2004)

34. Phan, A.V., Le Nguyen, M., Nguyen, Y.L.H., Bui, L.T.: Dgcnn: A convolutional neural network over large-scale labeled graphs. Neural Networks (2018)

35. Qi, C.R., Su, H., Mo, K., Guibas, L.J.: Pointnet: Deep learning on point sets for 3d classification and segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition (2017)

36. Rao, A.B., Krishnan, K., He, H.: Learning robotic grasping strategy based on natural-language object descriptions. In: 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (2018)

37. Sohn, K., Lee, H., Yan, X.: Learning structured output representation using deep conditional generative models. Advances in neural information processing systems (2015)

38. Turpin, D., Zhong, T., Zhang, S., Zhu, G., Heiden, E., Macklin, M., Tsogkas, S., Dickinson, S.J., Garg, A.: Fast-grasp’d: Dexterous multi-finger grasp generation through diferentiable simulation. In: 2023 IEEE International Conference on Robotics and Automation (ICRA) (2023)

39. Tversky, A.: Features of similarity. Psychological review, American Psychological Association (1977)

40. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems (2017)

41. Wang, R., Zhang, J., Chen, J., Xu, Y., Li, P., Liu, T., Wang, H.: Dexgraspnet: A large-scale robotic dexterous grasp dataset for general objects based on simulation. In: 2023 IEEE International Conference on Robotics and Automation (ICRA) (2023)

42. Wei, Y.L., Jiang, J.J., Xing, C., Tan, X.T., Wu, X.M., Li, H., Cutkosky, M., Zheng, W.S.: Grasp as you say: Language-guided dexterous grasp generation. Advances in Neural Information Processing Systems (2024)

43. Wei, Z., Xu, Z., Guo, J., Hou, Y., Gao, C., Cai, Z., Luo, J., Shao, L.: D(R, O) grasp: A unified representation of robot and object interaction for cross-embodiment dexterous grasping. In: 2025 IEEE International Conference on Robotics and Automation (ICRA) (2025)

44. Weng, Z., Lu, H., Kragic, D., Lundell, J.: Dexdifuser: Generating dexterous grasps with difusion models. IEEE Robotics and Automation Letters (2024)

45. Wu, R., Zhu, T., Lin, X., Sun, Y.: Cross-category functional grasp transfer. IEEE Robotics and Automation Letters (2024)

46. Xu, G.H., Wei, Y.L., Zheng, D., Wu, X.M., Zheng, W.S.: Dexterous grasp transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)

47. Xu, J., Sun, X., Zhang, Z., Zhao, G., Lin, J.: Understanding and improving layer normalization. Advances in neural information processing systems (2019)

48. Xu, Q., Xu, Z., Philip, J., Bi, S., Shu, Z., Sunkavalli, K., Neumann, U.: Point-nerf: Point-based neural radiance fields. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2022)

49. Xu, Y., Wan, W., Zhang, J., Liu, H., Shan, Z., Shen, H., Wang, R., Geng, H., Weng, Y., Chen, J., et al.: Unidexgrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2023)

50. Xu, Z., Gao, C., Liu, Z., Yang, G., Tie, C., Zheng, H., Zhou, H., Peng, W., Wang, D., Hu, T., et al.: Manifoundation model for general-purpose robotic manipulation of contact synthesis with arbitrary objects and robots. In: 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (2024)

51. Zhang, L., Zheng, D., Bai, K., Bing, Z., Marton, Z.C., Chen, Z., Knoll, A.C., Zhang, J.: Omnidexvlg: Learning dexterous grasp generation from vision language model-guided grasp semantics, taxonomy and functional afordance. arXiv preprint arXiv:2512.03874 (2025)

52. Zhong, Y., Jiang, Q., Yu, J., Ma, Y.: Dexgrasp anything: Towards universal robotic dexterous grasping with physics awareness. In: Proceedings of the Computer Vision and Pattern Recognition Conference (2025)

## Supplementary Material

This supplementary material provides additional technical details to support the main manuscript. Section A details the taxonomy transfer and handprint discretization. Section B outlines the network architecture and hyperparameters. Section C justifies the train-test domain shift quantitatively and qualitatively. Section D provides the mathematical formulations for our validation pipeline and energy-based optimization. Finally, Sections E through K ofer extended experimental protocols, metric definitions, and comprehensive qualitative results.

## A Applying contact topologies to grippers

Handprint Discretization. To generate the spatial handprints $\mathcal { H } ( Q )$ for the training dataset, we compute the forward kinematics for each sampled joint configuration Q. To represent the gripper’s active grasping surfaces, we discretize its geometry into a fixed point cloud of $N _ { H } = 2$ , 048 points. We filter the dense surface points based on their spatial dot products relative to the approach direction, ensuring a uniform, configuration-independent resolution across the functional contact areas. Specifically, let $\mathbf { G } \in \mathbb { R } ^ { \bar { M } \times 3 }$ represent the matrix of surface normals for a dense set of M candidate points on the gripper’s geometry. We define a reference direction vector, $\mathbf { v } _ { \mathrm { r e f } } \in \mathbb { R } ^ { 3 \times 1 }$ , representing the palm’s facing direction $( \mathrm { e . g . } \ \mathbf { v } _ { \mathrm { r e f } } = [ 0 , - 1 , 0 ] ^ { T }$ for the Shadow Hand). To isolate the active grasping surfaces, we evaluate the alignment of all candidate points simultaneously by computing the dot products between their normals and the reference direction:

$$
\mathbf { s } = \mathbf { G } \mathbf { v } _ { \mathrm { r e f } }\tag{12}
$$

The resulting scalar values in the vector $\mathbf { s } \in \mathbb { R } ^ { M \times 1 }$ are then used to threshold and sample the most relevant contact surfaces, culminating in the final $N _ { H }$ points that form $\mathcal { H } ( Q )$ . Figure 6 shows the resulting handprints for the Shadow and Allegro hands, alongside the manually defined zone divisions detailed in Section 3.1.

Taxonomy Transfer to Anthropomorphic Grippers. To semantically ground the generated handprints $\mathcal { H } ( Q )$ , we map established human grasp taxonomies onto the robotic kinematics. We manually partition the active grasping surfaces of both the Shadow Hand and the Allegro Hand into a discrete set of anatomical zones. Specifically, the mechanical links and palm areas are segmented into the $M = 2 1$ contact topology templates. During the handprint generation process, each sampled point h $\in { \mathcal { H } } ( Q )$ is assigned to its corresponding zone $\boldsymbol { \mathcal { A } } _ { k }$ . Notably, because the Allegro Hand lacks a fifth finger, the contact topology templates for $\boldsymbol { A } _ { 5 }$ and $\mathcal { A } _ { 6 }$ are functionally identical. To prevent redundancy, we explicitly omit $\mathcal { A } _ { 6 }$ from the Allegro Hand’s template set. Ultimately, this taxonomy transfer (illustrated Figure 7) ensures that despite the morphological diferences between the grippers, their spatial handprints share a common semantic representation, enabling structured, cross-embodiment comparisons of grasp strategies.

![](images/8237585cd1907295fba047a3a88c4371f0e62f75aeebdbef1da91f4890ae1ad4.jpg)  
Fig. 6: Handprint areas and workspace constitution. Left: Discretized handprints of the Shadow and Allegro hands, illustrating the manually defined anatomical zone divisions. Right: A three-quarter view of the Shadow Hand’s workspace.

![](images/7890954eb6af46869dc015a8101a5e2d687608ce36fa2d0756725caf5aca6735.jpg)  
Fig. 7: Taxonomy transfer mapping for anthropomorphic grippers. The handprints of the Shadow Hand (left) and the Allegro Hand (right) are segmented into manually defined, corresponding anatomical zones (A<sub>1</sub>–A<sub>21</sub>).

Workspace Computation. We define the gripper’s workspace as an aggregate point cloud representing its reachable surface area. This representation is generated by superimposing 10,000 gripper handprints sampled from our synthetic dataset, efectively capturing the complete interaction space of the fingers and palm independently of any external environment. To construct our fixed-size set of spatial basis points $\mathcal { W } = \{ w _ { j } \in \mathbb { R } ^ { 3 } \} _ { j = 1 } ^ { N _ { W } }$ , this dense accumulation is downsampled using Farthest Point Sampling (FPS). This yields a spatially uniform distribution of $N _ { W }$ that comprehensively encapsulate the gripper’s kinematic workspace. Figure 6 shows an illustration of the Shadow Hand’s workspace.

## B Architectural Choices and Implementation Details

To ensure the CoToGrasp framework accurately models the complex physics and semantics of dexterous grasping, several deliberate architectural choices were implemented. This section details the theoretical justifications for these design choices.

Local Geometry Extraction vs. Global Shape Descriptors. Standard 3D generative pipelines often employ global shape descriptors (e.g. standard PointNet [35]) that compress the entire object into a single latent vector. While efective for classification, global compression destroys high-frequency spatial details – such as local curvature and surface normals – which are absolutely critical for identifying physically viable contact patches. For this, we implement a DGCNN [34] encoder using the architecture presented in [43]. Unlike the standard Edge Convolution operation that dynamically recomputes K-Nearest Neighbors at every network layer, this architecture utilizes a fixed graph structure throughout, resulting in a "Static Graph CNN". Specifically, we set the neighborhood size to $K = 1 6$ to capture highly localized geometric information. By utilizing this "Static Graph CNN" architecture, we preserve point-wise local features $\mathcal { F } _ { \mathcal { P } }$ , ensuring the network reasons over local geometric fidelity while remaining invariant to the global spatial arrangement of the point cloud. Its learned weights successfully generalize across the domain shift – from the structured kinematic handprint seen during training to the highly variable target objects processed during inference.

Features Aggregation during the Workspace Projection. During the feature projection phase onto the canonical workspace W, we utilize a modified k-Nearest Neighbors aggregation. Inspired by scattered data interpolation in neural rendering [48], this local pooling strategy preserves high-frequency details more efectively than dense voxel grids or global pooling. In standard scattered data interpolation, it is common to normalize the aggregated features by the sum of the exponential weights. We intentionally omit this normalization step, instead computing an unnormalized average divided strictly by k (Eq. 5). Because the workspace covers the entire kinematic reach of the gripper, many basis points lie in "empty space," far from the input geometry. By leaving the distance weights unnormalized, they naturally decay to zero for distant neighbors. This consequently suppresses feature activation in empty regions, efectively preventing the network from hallucinating object geometry where none exists.

Persistent Hard Conditioning in Self-Attention. Standard vision transformer architectures [9] (ViT) typically condition the generation process by prepending the condition variable as an isolated global [CLS] token. However, grasp synthesis relies heavily on precise local contact arrangements across thousands of spatial tokens. A single global token is prone to signal dilution across deep attention layers [58]. To circumvent this, we explicitly concatenate the semantic topology embedding $\mathcal { F } _ { T }$ to every single workspace point feature. This point-wise fusion enforces a persistent, "hard" conditioning across the entire spatial domain, ensuring that every local geometric feature is evaluated strictly under the lens of the target functional intent at every layer of the network.

Since the projected workspace features $\mathcal { F } _ { \mathcal { W } }$ are processed by the transformer as an unordered sequence of tokens – lacking the explicit 3D coordinates $w _ { i }$ of the original basis points – we inject spatial awareness via sinusoidal positional encodings (PE). This allows the multi-head attention mechanism to recover spatial relationships purely from the fixed basis set indices. Crucially, the network is not provided with explicit contact priors at inference time. Instead, the semantic knowledge of valid contact configurations is acquired implicitly through backpropagation. The supervision signal drives the multi-head self-attention mechanism to discover and amplify geometric correlations between spatially distant points that correspond to stable configurations.

Global Latent Pooling via Set Transformer. Because our architecture explicitly omits a global learnable [CLS] token to preserve dense local conditioning, we cannot simply extract a designated output token (such as the $\mathbf { z } _ { L } ^ { 0 }$ state in standard ViTs) to form our global representation. Consequently, an explicit aggregation strategy is required to synthesize the dense sequence of fused geometric and semantic features into a unified global latent descriptor Z. For this task, we opted against static pooling operators $( \underline { { \mathrm { e . g . } } }$ . max-pooling or average-pooling). Static operators process elements independently, discarding non-extremal data and failing to capture spatial correlations. Instead, we utilize a Set Transformer. The Pooling by MultiHead Attention (PMA) block adaptively weighs input instances based on task relevance, while the subsequent Set Attention Block (SAB) explicitly models pairwise and higher-order interactions. This allows the network to successfully encode the structural co-occurrences and spatial distributions of distant contact patches.

CVAE and Contact-Topology-Conditioned Multi-Modality. A well-documented limitation of standard CVAEs in grasp generation is their tendency to sufer from posterior collapse [15], [54], [55] or blurred predictions when faced with the highly multi-modal nature of general grasping (e.g. the exact same object can be pinched, enveloped, or grasped by the rim). CoToGrasp bypasses this limitation. Because the macroscopic multi-modality is explicitly resolved by the strict topology conditioning (which dictates the exact functional approach), the CVAE is relieved of the burden of modeling drastically diferent grasp modes. It is only required to model the localized, intra-topology spatial variations of the contact patches (e.g. slight finger shifts along an edge). For this highly constrained stochasticity, a standard unimodal Gaussian prior N(0, I) proves to be both mathematically suficient and computationally highly eficient. The variational lower bound (ELBO) consists of the reconstruction term, a Multi-Class Cross-Entropy loss $\left( L _ { \mathrm { r e c o n } } \right)$ over the workspace points $\mathcal { I } = \{ 1 , \dots , n _ { W } \}$ , and a Kullback-Leibler regularization term:

$$
L = L _ { \mathrm { { r e c o n } } } + \beta D _ { \mathrm { { K L } } } ( q ( z | \mathcal { Z } ) \parallel \mathcal { N } ( 0 , I ) ) , L _ { \mathrm { { r e c o n } } } = - \sum _ { j \in \mathcal { I } } \mathbf { y } _ { j } \cdot \log ( \hat { \mathbf { y } } _ { j } )\tag{13}
$$

where $\beta$ is a weighting parameter balancing latent capacity and reconstruction accuracy, $\mathbf { y } _ { j }$ is the one-hot encoded ground truth derived from $A _ { m } ( j )$ and $\hat { \mathbf { y } } _ { j }$ is the predicted probability vector.

Network Hyperparameters and Training Details. Table 4 summarizes the comprehensive architectural details, layer dimensions, and hyperparameters used to implement and train the CoToGrasp framework. All training procedures were executed across a cluster of 4 Nvidia A100 GPUs, requiring approximately 35 hours.

## C Train-Test Domain Shift and Feature Alignment

To demonstrate that our architectural choices enable robust cross-domain generalization – specifically, the transfer from the gripper surface (training domain) to the object surface (inference domain) – we analyze the latent feature alignment across both domains. This evaluation verifies whether our framework successfully bridges the inherent geometric and structural diferences between hands and objects.

Quantitative Alignment Analysis. To measure the transfer quality, we extracted intermediate feature representations from both domains and evaluated their alignment at active topological contact points using Average Cosine Similarity. We compared corresponding spatial points (Matched Pairs) against random, non-corresponding points (Random Pairs) to establish a baseline (Table 5). Initially, the raw point-wise features extracted from the DGCNN $( \mathcal { F } _ { \mathcal { P } } )$ exhibit a significant domain gap, achieving a poor similarity score of 0.35 for matched pairs and 0.23 for random pairs. This confirms that the raw domain shift remains substantial despite static graph modifications. However, the efectiveness of projecting into our feature-based canonical workspace via kNN aggregation $( \mathcal { F } _ { \mathcal { W } } )$ is immediately evident: the matched pair cosine similarity jumps to 0.61. This quantitatively proves that the canonical projection successfully disentangles features from the input topology, efectively bridging the domain gap. Finally, introducing the contact topology embedding $( \mathcal { F } _ { \mathcal { T } } )$ into the Transformer encoder (Φ) further sharpens this representation. The self-attention mechanism increases matched similarity to 0.66 while actively suppressing mismatched, domain-specific geometric noise (reducing random pair similarity from 0.33 to 0.27).

Qualitative Latent Space Visualization. This quantitative improvement is visually corroborated by the t-SNE projection of our canonical workspace features $( \mathcal { F } _ { \mathcal { W } } )$ , illustrated in Figure 8. While global domain distinctions naturally persist – reflecting the inherent macro-structural diferences between an anthropomorphic hand and arbitrary objects – the features distinctly interleave within shared manifold clusters at task-relevant regions. This selective alignment visually and mathematically justifies our reliance on the self-attention layers (Φ). The Transformer is necessary to isolate and attend to these domain-invariant contact primitives, resolving non-local spatial dependencies while ignoring irrelevant structural disparities. Ultimately, this ensures that the generative process remains robust regardless of the input surface.

<table><tr><td>Module / Parameter</td><td>Notation</td><td>Value / Size</td></tr><tr><td colspan="3">Training &amp; Optimization</td></tr><tr><td>Batch Size</td><td></td><td>32</td></tr><tr><td>Learning Rate</td><td></td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>Training Epochs</td><td></td><td>50</td></tr><tr><td>KLD Regularization Weight</td><td>β</td><td>0.1</td></tr><tr><td>Attention Factor</td><td>α</td><td>3.0</td></tr><tr><td colspan="3">Workspace &amp; Geometric Projection</td></tr><tr><td>Workspace Resolution</td><td>Nw</td><td>8,192</td></tr><tr><td>Projection kNN</td><td>k</td><td>5</td></tr><tr><td>Aligned Distance Scaling</td><td>γ</td><td>2.0</td></tr><tr><td colspan="3">Pointwise Feature Extraction</td></tr><tr><td>Local Graph kNN</td><td>K</td><td>16</td></tr><tr><td>Hidden Layer Sizes</td><td></td><td>[12, 64, 64, 128, 256, 512]</td></tr><tr><td>Output Feature Dimension</td><td> $\mathcal { F } _ { \mathcal { P } } , \mathcal { F } _ { \mathcal { W } }$ </td><td>1,024</td></tr><tr><td colspan="3">Conditioning Embeddings</td></tr><tr><td>Topology Embedding Dim.</td><td> $\mathcal { F } _ { \mathcal { T } }$ </td><td>64</td></tr><tr><td>Label Embedding (MLP)</td><td> ${ \mathcal { F } } _ { A }$ </td><td>[128, 64]</td></tr><tr><td colspan="3">Self-Attention Modules</td></tr><tr><td colspan="3">Transformer Encoder Feature Dim. Φ</td></tr><tr><td>Transformer Encoder Blocks</td><td></td><td>1088 4</td></tr><tr><td>Transformer Encoder Heads</td><td></td><td>8</td></tr><tr><td>Set Transformer (Pooling) Heads</td><td></td><td>8</td></tr><tr><td>Global Latent Descriptor Dim.</td><td>Z</td><td>1,024</td></tr><tr><td colspan="3">CVAE &amp; Latent Space</td></tr><tr><td colspan="3"></td></tr><tr><td>Latent Encoder Hidden Dim.</td><td></td><td>512 64</td></tr><tr><td>Latent Variable Dimension</td><td>ψ</td><td></td></tr><tr><td>AdaLN Decoder Output Classes</td><td> $N + 1$ </td><td>23</td></tr></table>

Table 4: Implementation Details and Network Hyperparameters for the CoToGrasp framework.

<table><tr><td colspan="3">Raw DGCNN Workspace + Attn.</td></tr><tr><td>Matched Pairs</td><td>0.35</td><td>0.61</td></tr><tr><td>Random Pairs</td><td>0.23</td><td>0.66 0.33 0.27</td></tr><tr><td colspan="3">Table 5: Feature Alignment (Cosine Similarity).</td></tr></table>

t-SNE Alignment - Canonical Workspace Projection Features  
![](images/e8160c1f23756cb22602e763d61828f14e11b73d5441674db02ff5f4d3ec7367.jpg)  
Fig. 8: t-SNE visualization of features after canonical workspace projection.

## D Validation Pipeline and Energy Optimization Details

This section details the post-inference validation and optimization pipeline, which is designed to filter out physically and topologically invalid predictions before finalizing the hand’s kinematics. To ensure both semantic correctness and physical feasibility, our framework processes the generated contact templates through a sequential validation strategy: a label-consistency check, a preliminary forceclosure evaluation, and an energy-based joint optimization. The average runtime per grasp evaluates to 110 ms, with the computational cost broken down as follows: pose initialization (2.8 ms), forward network inference (0.3 ms), labelconsistency verification (42.4 ms), force-closure computation (14.7 ms), and the final joint configuration optimization (50.7 ms).

Label-Consistency Check Formulation. Let $\mathcal { U } ( A ) = \{ A ( j ) ~ | ~ A ( j ) \neq 0 \}$ be an operator that extracts the set of unique, active semantic zone identifiers from a given contact template. To prevent the generation of ill-posed grasps, we define the validation function $V ( A _ { m } , \hat { A } _ { m } )$ as a strict binary check on the symmetric diference between the extracted zone sets of the ground truth and predicted templates::

$$
V ( A _ { m } , \hat { A } _ { m } ) = \mathbb { I } ( | \mathcal { U } ( A _ { m } ) \Delta \mathcal { U } ( \hat { A } _ { m } ) | \leq 1 )\tag{14}
$$

where I is the indicator function and $\varDelta$ is the symmetric diference operator defined as $\mathcal { U } ( \Lambda _ { m } ) \Delta \mathcal { U } ( \hat { \Lambda } _ { m } ) = ( \mathcal { U } ( \Lambda _ { m } ) \setminus \mathcal { U } ( \hat { \Lambda } _ { m } ) ) \cup ( \mathcal { U } ( \hat { \Lambda } _ { m } ) \setminus \mathcal { U } ( \Lambda _ { m } ) )$ . Setting the tolerance threshold to 1 allows for a maximum deviation of one missing or hallucinated contact zone $( \underline { { \mathrm { e . g . } } }$ . a missing palm contact in a power grasp) while successfully rejecting fundamentally distinct topological failures.

Force-Closure Formulation. Unlike methods trained on pre-validated grasp datasets, our generative approach requires an explicit assessment of grasp stability. To ensure the inferred contact points can theoretically yield a stable grasp before performing the computationally expensive joint configuration optimization, we evaluate the Force-Closure condition using the explicit spatial coordinates of the active workspace points.

Contact Modeling: We first extract the subset of workspace points assigned a valid contact label: $\hat { \mathcal { C } } ( \mathcal { W } ) = \{ w _ { j } \in \mathcal { W } \mid \hat { A } _ { m } ( j ) \neq 0 \}$ . We then group these active points by their predicted semantic zone indices. To enforce a unique contact location per required zone, we compute the spatial barycenter of each point cluster and project it onto the object’s surface ${ \tilde { \mathcal { O } } } _ { : }$ yielding a set of discrete contact locations $b _ { i }$ . We assume a Coulomb friction model with a friction coeficient of $\mu = 0 . 3$ . Because this estimation is performed entirely within the canonical gripper’s reference frame to assess geometric feasibility, we do not factor in gravity or the object’s mass at this stage.

– Force-Closure Computation: We compute the grasp wrench space, defined as the convex hull of all possible wrenches generated by unit contact forces at locations $b _ { i }$ . The total wrench is given by:

$$
d = \sum _ { i = 1 } ^ { k } d _ { i } = \sum _ { i = 1 } ^ { k } G _ { i } f _ { i }
$$

where $G _ { i }$ is the partial grasp matrix for contact $b _ { i } ,$ and $f _ { i }$ represents the primitive force vectors along the edges of the friction cone, normalized such that $\| f _ { i } \| = 1$ . We verify the force-closure condition by checking if the origin of the wrench space lies strictly within the interior of this convex hull.

Theoretical vs. Physical Quality: It is important to note that this analytical step strictly validates the theoretical stability potential of the semantic contact configuration using a simplified barycenter model. The final, true physical grasp quality is determined during the energy-based optimization, which forces the gripper’s complex kinematics to align with the full, dense 3D points of $\boldsymbol { \hat { c } } ( \boldsymbol { w } )$ rather than just these idealized discrete barycenters.

Joint Configuration Optimization Formulation. During the final optimization phase, the joint configuration $Q ^ { * }$ is retrieved by minimizing a composite energy function. The weighting factors (λ) balancing the individual energy terms are empirically set to ensure stable convergence. The individual energy terms and their respective weights are defined as follows:

– Contact Distance $( E _ { d i s t } , \lambda _ { d } = 1 . 0 )$ : To encourage precise alignment between the predicted spatial contact points $\hat { \mathcal { C } } ( W )$ and the gripper’s geometry, we minimize the squared Euclidean distance between each target contact point and the closest point on its assigned kinematic link:

$$
E _ { d i s t } = \sum _ { p \in \hat { \mathcal { C } } ( W ) } \operatorname* { m i n } _ { h \in \mathcal { H } _ { l ( p ) } ( Q ) } \| p - h \| _ { 2 } ^ { 2 }\tag{15}
$$

where $\mathcal { H } _ { l ( p ) } ( Q )$ represents the subset of handprint points belonging to the specific link $l ( p )$ assigned to the contact point $p .$

Object Penetration $( E _ { p e n } , ~ \lambda _ { p } ~ = ~ 0 . 5 )$ : To ensure physically plausible grasps, we explicitly penalize gripper points that penetrate the target object’s volume. Utilizing the object’s Signed Distance Field (SDF), denoted as $\varPhi _ { \mathcal { O } } ( \cdot )$ , this term is formulated as:

$$
E _ { p e n } = \sum _ { h \in \mathcal { H } ( Q ) } \operatorname* { m a x } ( 0 , - \varPhi _ { \mathcal { O } } ( h ) )\tag{16}
$$

Self-Penetration $( E _ { s p e n } , \ \lambda _ { s } = 0 . 0 1 ) \colon$ : We prevent self-collisions between diferent fingers or parts of the robotic hand by enforcing a minimum spatial safety threshold $\tau$ (set to 0.025 m) between disjoint kinematic links i and $j \colon$

$$
E _ { s p e n } = \sum _ { i , j } \operatorname* { m a x } ( 0 , \tau - \mathrm { d i s t } ( k _ { i } ( Q ) , k _ { j } ( Q ) ) ) ^ { 2 }\tag{17}
$$

where $k _ { i } ( Q )$ denotes the spatial centroid of the bounding box for link i in a given configuration $Q .$

Joint Limits $( E _ { j } , \ \lambda _ { j } \ = \ 1 . 0 )$ . We strictly constrain the optimized joint angles to remain within the physical hardware’s kinematic limits, defined by $[ Q _ { m i n } , Q _ { m a x } ] ;$

$$
E _ { j } = \| \operatorname* { m a x } ( 0 , Q - Q _ { m a x } ) \| _ { 2 } ^ { 2 } + \| \operatorname* { m a x } ( 0 , Q _ { m i n } - Q ) \| _ { 2 } ^ { 2 }\tag{18}
$$

Repulsive Energy Term $( E _ { r e p } , \lambda _ { r } = 0 . 0 1 )$ : During the energy-based joint optimization, we aim to align the gripper to the predicted contacts while strictly preventing unintended parts of the hand from resting on the object, which would violate the requested contact topology. To achieve this, we introduce the repulsive energy term $E _ { r e p } ,$ which penalizes object proximity for all gripper handprint points belonging to non-active regions:

$$
E _ { r e p } = \sum _ { \substack { h \in \mathcal { H } ( Q ) \backslash \mathcal { H } _ { l ( p ) } ( Q ) } } \operatorname* { m a x } ( 0 , \delta - \mathrm { d i s t } ( h , \tilde { \mathcal { O } } ) )\tag{19}
$$

We enforce a minimum safety margin of $\delta = 0 . 0 0 5 \mathrm { ~ m ~ }$

## E Topology-Conditioned Pose Sampling Heuristic

As discussed in the main text, CoToGrasp treats the global 6D grasp pose as an external prior. To autonomously evaluate the framework without relying on external task planners, we sample diverse candidate object poses $( R , { \bf t } ) ^ { - 1 }$ using a topology-conditioned heuristic tailored to the specific active zones of the requested grasp template. Figure 9 illustrates the following sampling strategy for three distinct contact topologies.

Palm-Constrained Topologies. For power grasps and contact topologies requiring palm contact, we adopt the spatial heuristic from [24, 26, 32]. We uniformly sample approach translations and orientations on the object’s convex hull, directed toward its volumetric center. The object is then transformed into the canonical gripper frame via the inverse transformation $( R , \mathbf { t } ) ^ { - 1 }$

Distal and Precision Topologies. For grasps explicitly excluding the palm (e.g. fingertips only), standard convex hull sampling often yields kinematically unreachable configurations. Instead, we sample the object pose $( R , { \bf t } ) ^ { - 1 }$ directly within a restricted kinematic sub-workspace of the target template’s active zones. To ensure the object is placed in a feasible region away from the palm, we systematically truncate the sub-workspace using three geometric constraints: (1) a radial distance threshold from a predefined workspace center to exclude outof-reach boundary points, (2) a directional half-space filter ensuring points lie strictly in front of the palm’s normal axis, and (3) a minimum height threshold along the z-axis to avoid the lower hand geometry. Furthermore, to guarantee topological validity and prevent trivial local minima during the energy-based joint optimization, we explicitly enforce a strict 1 cm clearance margin between the object geometry and the palm.

![](images/932f6a9fd9e01420a734063b29704d99ba577deae8f61b04d7345af9b8495f72.jpg)  
Fig. 9: Topology-Conditioned Pose Sampling. For specific grasp topologies (such as M4 and M12), the initial object pose is sampled within a restricted kinematic region. Middle: The template’s full active sub-workspace is shown in light blue, while the truncated sub-workspace – filtered for reachability and palm clearance – is highlighted in red. Right: Examples of the initialized object point cloud O<sup>˜</sup> (green) successfully placed within this feasible region after the sampled spatial transformation.

## F Automated Grasp Classification Metric

This section details the mathematical formulation of our contact topology recognition metric.

Contact Extraction and Zone Mapping. First, we extract the subset of gripper points in contact with the object, ${ \mathcal { C } } _ { \mathrm { g r i p } } ( { \mathcal { H } } ( Q ) )$ ), using the aligned distance formulation with an empirically chosen proximity threshold of $\epsilon = 0 . 8$ . To compare these physical contacts against the taxonomy $\tau _ { \ast }$ , we map the raw contact points back to their semantic zone identifiers. Let $I _ { \mathrm { o b s } } = \{ i \in \mathbb { Z } \mid \mathbf { h } _ { i } ( Q ) \in \mathcal { C } _ { \mathrm { g r i p } } ( \mathcal { H } ( Q ) ) \}$ be the indices of the gripper points currently touching the object. Using the surjective mapping ζ, we define the observed active zones $\left( A _ { \mathrm { o b s } } \right)$ as the set of unique physical regions engaged in the grasp, and the ground-truth required zones $\left( A _ { m } \right)$ extracted from the target template $A _ { m }$ (explicitly excluding the label 0):

$$
A _ { \mathrm { o b s } } = \{ \zeta ( i ) \mid i \in I _ { \mathrm { o b s } } \} \qquad \mathrm { a n d } \qquad A _ { m } = \{ A _ { m } ( i ) \mid i \in \mathbb { Z } \} \setminus \{ 0 \}\tag{20}
$$

Asymmetric Tversky Index. We evaluate the match between the generated grasp and a specific contact topology m by computing a similarity score $s _ { m }$ based on the Tversky index [39]:

$$
s _ { m } = \frac { \left| A _ { \mathrm { o b s } } \cap A _ { m } \right| } { | A _ { \mathrm { o b s } } \cap A _ { m } | + w _ { 1 } | A _ { \mathrm { o b s } } \setminus A _ { m } | + w _ { 2 } | A _ { m } \setminus A _ { \mathrm { o b s } } | }\tag{21}
$$

where $| \cdot |$ denotes set cardinality. We set the weighting parameters to $w _ { 1 } = 2 . 0$ and $w _ { 2 } = 0 . 5$ . This asymmetric penalization, determined empirically, is critical: it punishes extra contact regions $( | A _ { \mathrm { o b s } }  \backslash A _ { m } | )$ , which typically denote clumsy or unintended enveloping power grasps, much more strictly than missing contacts $( | A _ { m }    A _ { \mathrm { o b s } } | )$

Any grasp failing to reach a similarity threshold of $s _ { m } \ge 0 . 5$ is strictly classified as ’unknown’. Given our asymmetric Tversky weights, $w _ { 1 }$ and $w _ { 2 } , 0 . 5$ represents the mathematical tipping point where the number of correctly aligned contact zones strictly outweighs the heavily penalized hallucinated contacts (e.g., a failed precision pinch collapsing into a power grasp), ensuring robust rejection of degenerated topologies.

## G Standard Evaluation Metrics Protocol

As mentioned in the main text, CoToGrasp utilizes the standard evaluation protocols widely established in [24, 26, 32, 43] to measure baseline physical performance.

Success Rate (SR). We evaluate the physical stability of the synthesized grasps using the Isaac Gym simulator [30]. The target object (standardized to a mass of 100 g) and the gripper are initialized in the optimized configuration $Q ^ { * }$ . We sequentially apply external perturbations on the object along the $\pm x , y , z$ axes for one second each. A grasp is considered successful if the object’s translation deviates by less than $2 \mathrm { c m }$ from its initial pose. Furthermore, any grasp exhibiting severe interpenetration (contact forces exceeding 500 N) is strictly classified as a failure to penalize unrealistic physical states.

It is crucial to note that the theoretical force-closure validation performed during inference serves only as a rapid geometric estimation based on idealized point contacts. It ofers no absolute guarantees regarding final physical stability, as the Isaac Gym simulation rigorously tests the complex reconciliation of these points against the complete object geometry, strict kinematics, and the torque saturation limits of the gripper’s simulated actuators when compensating for external perturbation forces.

Generation Speed. We report the average computational time (in seconds) required to generate a single grasp, calculated over 100 generation attempts. This metric strictly isolates the network inference and the energy-based joint optimization time, explicitly excluding the heavy physics simulation overhead of Isaac Gym.

Spatial Diversity. We quantify generative spatial diversity by calculating the standard deviation of the gripper translation (t), orientation (R), and joint values (Q). Because the initial global poses (R, t) are derived from our topologyconditioned sampling heuristic, the diversity in t and R directly reflects the method’s spatial reachability and adaptability across various object geometries.

## H Detailed Experimental Protocols and Baselines

To ensure a rigorous and fair evaluation, this section provides the extended protocols and baseline configurations.

Taxonomy-Unaware Evaluation Protocol. To evaluate the inherent functional bias of unconditioned grasp planners, we compare CoToGrasp against four state-of-the-art baselines: DFC [26], GenDexGrasp [24], DRO-Grasp [43], and GOAG [32]. For this experiment, we utilize the test subset from the MultiDex dataset, comprising 10 objects from ContactDB [2] and YCB [4]. Each baseline was tasked with generating exactly 100 grasps per object. To ensure a fair comparison, the baselines are deployed as follows:

– DFC: as DFC does not require training, we utilize pre-generated grasps from the CMapDataset. These poses were originally synthesized via DFC and subsequently refined by [24] through a post-filtering process to ensure geometric validity.

GenDexGrasp: We use the oficial pre-trained checkpoint (trained on multihand data) with default hyperparameters. We first infer the contact maps for the test objects, which are then passed into the authors’ optimization framework to derive the final grasp poses.

– DRO-Grasp: We select the most performant variant, which includes configuration-invariant pre-training. We retain all default hyperparameters but explicitly deactivate their simple grasp controller to ensure a fair, purely kinematic comparison.

– GOAG: We train GOAG from scratch on the Shadow Hand kinematics using the default hyperparameters, and generate grasps following the standard inference pipeline.

All successfully generated, physically stable grasps were subsequently fed into our automated classification pipeline (Sec. 4.1, Supp. Mat. F) to evaluate their Topology Compliance (TC), Entropy and Spatial Diversity.

Because these baselines do not take a functional condition as input, this test strictly isolates their natural generative distribution, exposing the severe mode collapse toward power grasps driven by standard physics-based optimization. Furthermore, while CoToGrasp achieves a higher TC than the baselines, the absolute scores remain relatively low across all methods. This shared ceiling suggests that TC is heavily bottlenecked by the specific object set utilized; the limited geometric diversity of these test objects naturally restricts the subset of physically viable grasps, making certain functional topologies kinematically unattainable regardless of the generative method.

Taxonomy-Aware Evaluation Protocol. To evaluate the precise functional control of CoToGrasp, we conducted an extensive comparison against Dexonomy [5] on the full test set of the DexGraspNet [41] dataset, which comprises 1,126 geometrically diverse objects. We selected Dexonomy [5] as our sole baseline for this experiment, as it currently stands as the first and only state-ofthe-art framework capable of taxonomy-conditioned generative grasp synthesis. While other recent methods tackle dexterous grasping, they rely on assumptions orthogonal to our object-agnostic setting. For instance, functional grasp transfer and retargeting methods, such as FunGrasp [14] and others [53, 57], operate in a fundamentally diferent problem space; they inherently require explicit human grasp demonstrations or a dense spatial prior—such as a source human hand mesh—to initialize their optimization. Furthermore, unconditioned frameworks like AnyDexGrasp [10] lack semantic topology control, while methods like OmniDexVLG [51] rely on 2D VLM supervision and are currently publicly unavailable. In contrast, CoToGrasp and Dexonomy [5] function as true grasp planners, synthesizing functionally compliant grasps from scratch relying purely on raw object geometry and a discrete semantic label.

Taxonomy Mapping and Grouping: Dexonomy [5] natively outputs grasps categorized under the classic Feix taxonomy [11]. To ensure a direct one-to-one semantic comparison, we mapped their analytically computed dataset to our contact-based topologies (illustrated in Figure 3 of the main paper). Crucially, this evaluation space ensures an unbiased comparison. By focusing exclusively on contact regions, the Feix taxonomy naturally reduces to our adopted Gonzalez framework (e.g., F12/F13 → M6). While Dexonomy couples joint configurations with contact semantics, CoToGrasp relies strictly on contact topologies. Because our metrics (TC, H<sub>TC</sub>) measure semantic compliance purely through these spatial contact assignments, the two taxonomies are structurally equivalent for this assessment. Finally, we explicitly grouped the evaluated grasps into three functional categories to analyze performance across diferent dexterity levels: Power grasps (enveloping, high stability), Precision grasps (fingertip manipulation, low stability), and Object-Specific grasps (tool use).

Handling Geometric Incompatibility: A critical challenge in large-scale grasp evaluation is that not all contact topologies are geometrically possible on all objects (e.g. it is physically impossible to execute a tiny fingertip precision pinch on a massive, smooth sphere). If an object cannot support a specific contact topology, penalizing the network for failing to generate it skews the metric and misrepresents the model’s actual generative capability.

To ensure a strictly fair comparison, we implemented a geometric compatibility filter. Any object-topology pair that yielded a 0% success rate across all generation attempts was deemed fundamentally geometrically incompatible and explicitly excluded from the final average calculations.

Following this rigorous filtering process, CoToGrasp successfully covers an average of 80.14% of the objects per requested contact topology, which closely and fairly matches Dexonomy’s coverage of 81.05%. Consequently, the Success Rate and Topology Compliance metrics reported in Table 2 strictly reflect Co-ToGrasp’s superior generative control on valid geometries, rather than an artifact of object selection.

## I Topology Compliance Analysis and Simulation Bottlenecks

To further understand the discrepancy between requested and efective contact topologies observed in the main paper, we analyze CoToGrasp’s TC at diferent stages of the generation pipeline (Table 6).

<table><tr><td rowspan=1 colspan=1>LabelConsistency</td><td rowspan=1 colspan=1>Eval.Isaac</td><td rowspan=1 colspan=1> $\mathbf { H } _ { S R }$ </td><td rowspan=1 colspan=1>TC (%)</td><td rowspan=1 colspan=1> $\mathbf { H } _ { T C }$ </td></tr><tr><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>VX</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>17.1821.92</td><td rowspan=1 colspan=1>0.840.53</td></tr><tr><td rowspan=1 colspan=1>xX</td><td rowspan=1 colspan=1>√X</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>14.4519.34</td><td rowspan=1 colspan=1>0.810.50</td></tr></table>

Table 6: Topology compliance ablation.

Value of Topological Filtering. Comparing the top and bottom halves of the table demonstrates the value of topological filtering. By rejecting geometrically incompatible templates before the energy-based optimization, the pipeline prevents the optimizer from coercing the hand into unnatural configurations that would ultimately fail in simulation, thereby improving the overall semantic quality of the successful grasps.

The Physics-Based Metric Drop. The most striking shift occurs between the pre-physics (✗ Eval. Isaac) and post-physics evaluations (✓ Eval. Isaac). Prior to the Isaac Gym simulation, the pipeline achieves its highest absolute TC but exhibits a remarkably low semantic entropy $\left( \mathbf { H } _ { T C } \right)$ . This indicates that the purely geometric, energy-based optimizer frequently converges on configurations that technically satisfy the target contact masks but lack true physical stability. Once subjected to gravity and external perturbations, a significant portion of these superficial grasps naturally fails. While this physics-based filtering prunes weak configurations and drastically rebalances the distribution (restoring $\mathbf { H } _ { T C } )$ , it also causes a drop in the raw TC. This drop exposes a sensitivity to the rigid definition of our similarity score formula. During simulation, fingers naturally shift slightly along the object’s local curvature to achieve a stable force-closure. Because our metric enforces strict mathematical boundaries, these minor, functionally viable adjustments often lead to misclassifications. Specifically, despite the presence of a repulsive term in the optimization energy function, certain phalanges cannot be pushed away from the object surface due to inherent kinematic constraints, such as fixed finger lengths or joint limits. In these scenarios, the optimization weighting factors privilege the primary contact required by the topology over the repulsive term, leading to incidental contacts. This phenomenon frequently results in intended precision pinches (M2 or M3) being penalized and reclassified as M12 due to an adjacent phalanx resting too closely to the surface – as illustrated in Figure 10. Consequently, while the final generated grasps are physically stable, this rigid metric provides an incomplete picture of functional success, as it fails to capture whether the intended core contacts were actually achieved.

![](images/0b32cb7fc90a87ff76b02d16dc1dc141b8ddacc2487d499c8595b6cd1dac1453.jpg)  
Fig. 10: Illustration of metric-induced misclassification. While CoToGrasp strictly respects the target contact topology, the kinematic optimization may result in incidental contacts where an adjacent phalanx rests against the object surface. Despite the repulsive term in our optimization energy function, these phalanges often cannot be pushed away due to inherent kinematic constraints. For instance, the left grasp illustrates an intended M3 pinch reclassified as M12, while the right shows an M4 grasp reclassified as M15. Although the intended contacts are successfully achieved and the grasps remain physically stable, these incidental contacts trigger a strict reclassification by our automated pipeline.

Global Topological Distribution. Beyond the specific pipeline bottlenecks, Figure 11 provides a comprehensive visual breakdown of the attempted versus efective topologies across all 21 topologies. As illustrated, while both methods experience the aforementioned metric-induced shifts during simulation, Co-ToGrasp maintains a substantially more balanced and faithful distribution of functional grasps (TC = 17.18%) compared to the Dexonomy baseline (TC = 14.28%).

![](images/6396685e9e725f5412f14bcbb6b847d32c30f452923d6bd61b1c8ea4e2493d77.jpg)

![](images/904b22da3df31c95c48c0bd32ed1b3f336a3da3256f88a88460b52a9c2dc8597.jpg)  
Fig. 11: Histogram comparing the frequencies of efective topologies and attempted topologies (as defined Sec. 4.2) among all stable grasps generated by Co-ToGrasp (top) and Dexonomy [5] (bottom).

Object-Level Geometric Complexity. To quantify geometric dificulty and evaluate the inherent trade-of between strict semantic control and geometric flexibility, we analyzed performance grouped by object convexity $( c = V _ { o b j } / V _ { h u l l } )$ As detailed in Table $^ { 7 , }$ we define SR Retained as the comparison of Success Rates (SR) between non-convex $( c < 0 . 4 )$ and convex $( c > 0 . 9 )$ objects. When transitioning to these challenging geometries, CoToGrasp demonstrates remarkable robustness, retaining 58.72% of its physical SR. In contrast, Dexonomy’s rigid templates retain only 37.42% of their performance, and the unconditioned GOAG drops to 28.09%. This performance gap widens on objects with severe concavities $( c < 0 . 4 .$ , representing roughly 4% of the dataset). On these hardest geometries, CoToGrasp achieves an 18.30% SR, outperforming Dexonomy’s 12.05%. Strikingly, CoToGrasp’s semantic compliance (TC) actually increases to 19.17% on these objects, whereas Dexonomy’s TC collapses to 8.94%. These results prove that our contact-centric approach thrives on leveraging complex local features to anchor semantics. While strict semantic control typically limits geometric flexibility, CoToGrasp pushes this boundary significantly further than planners relying on rigid, joint-coupled templates, which systematically fail or abandon the semantic query entirely on complex shapes.

<table><tr><td>Object Complexity</td><td>Metric</td><td>CoToGrasp</td><td>Dexonomy [5]</td></tr><tr><td>Convex → Non-Convex</td><td>SR Retained</td><td>58.72%</td><td>37.42%</td></tr><tr><td>Severe Concavities</td><td>SR</td><td>18.30%</td><td>12.05%</td></tr><tr><td> $( c < 0 . 4 , \sim$  4% data)</td><td>TC</td><td>19.17%</td><td>8.94%</td></tr></table>

Table 7: Object-Level Analysis. Evaluating performance across geometric complexity $( c = V _ { o b j } / V _ { h u l l } )$ . CoToGrasp shows superior retention of both physical stability (SR) and semantic compliance (TC) on challenging non-convex objects.

## J Real-World Experiments

Execution Protocol: For a given synthesized grasp (R, t, Q), we implemented a strict execution pipeline. To ensure collision-free trajectories and safe hardware operation, all motions were initially planned and validated within a ROS2 digital twin using the MoveIt [6] motion planning framework. The execution sequence begins by computing an approach pose, defined by translating the target 6D pose (R, t) backward by 10 cm along the normal vector of the gripper’s palm. The robot first navigates to this approach pose with the fingers fully extended. Subsequently, the arm linearly interpolates to the target spatial pose $( R , \mathbf { t } )$ ， and the fingers are actuated to converge on a squeezed configuration $Q _ { s } ^ { * }$ based on the optimized joint configuration $Q ^ { * } . \ Q _ { s } ^ { * }$ is computed such as finger links involved in the grasps are closer to the object’s center of mass, similarly as $[ 5 ,$ 43]. To empirically verify the physical stability of the grasp against gravity, the manipulator lifts the object 15 cm directly above the tabletop, holds it statically for 3 seconds, and finally opens the hand to release the object.

Discussion and Limitations. The execution of synthesized precision grasps on physical hardware exposes the fundamental mismatch between deterministic kinematic planning and the stochastic physics of mechanical contact. While standard position-control frameworks (e.g. OMPL pipelines in MoveIt) are efective for coarse manipulation, they fail to maintain the integrity of complex contact topologies. We observed that relying on a simple linearly interpolated squeezed configuration $Q _ { s } ^ { * }$ is critically insuficient; because $Q _ { s } ^ { * }$ is derived from geometric distances to the object’s barycenter, digits move at uniform velocities regardless of external reaction forces. This inevitably results in asynchronous contact, where leading fingers strike the surface milliseconds early, generating unbalanced moments that displace the object before force-closure is achieved. Consequently, physical contact points drift from predicted optimal zones, misaligning force vectors with the intended functional semantics. Furthermore, current evaluation benchmarks – often restricted to tabletop environments – contradict the objective of omnidirectional grasp synthesis. Such environments artificially inflate the success of grasps that may only be viable in a bimanual setup or when the object is presented in a specific, pre-constrained 6D pose. To maintain fidelity to simulated topologies, future work must transition toward contact-aware interaction paradigms, such as hybrid force/position control or tactile-reactive policies [56].

## K Qualitative Results: CoToGrasp Visualization

Figure 12 and Figure 13 present qualitative results across a diverse set of YCB and DexGraspNet objects, highlighting the framework’s ability to strictly adhere to the intended contact topology. Rather than merely reaching for the object’s center of mass, the synthesized configurations demonstrate precise alignment between the fingers and specific semantic contact zones. As shown in Figure 13, when grasps are grouped by their topological identifiers (M1-M21), the planner consistently recovers the prescribed contact manifolds. This adherence ensures that the resulting configurations maintain the intended grasp type – whether a delicate fingertip pinch or a complex multi-digit wrap – efectively preserving the functional semantics of the interaction across both convex and non-convex surfaces.

## Supplementary References

53. Antotsiou, D., Garcia-Hernando, G., Kim, T.K.: Task-oriented hand motion retargeting for dexterous manipulation imitation. In: Proceedings of the European conference on computer vision (ECCV) workshops (2018)

54. Fu, H., Li, C., Liu, X., Gao, J., Celikyilmaz, A., Carin, L.: Cyclical annealing schedule: A simple approach to mitigating kl vanishing. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (2019)

![](images/4ec657949d66e3ac160938bb3d95ccfad631fc7f7a2a97741938bbfbd570ea07.jpg)

Fig. 12: Qualitative Synthesis Gallery on the Allegro Hand. Synthesized grasp configurations for a diverse subset of the YCB object dataset.

55. He, J., Spokoyny, D., Neubig, G., Berg-Kirkpatrick, T.: Lagging inference networks and posterior collapse in variational autoencoders. In: International Conference on Learning Representations (2019)

56. Jahanshahi, H., Zhu, Z.H.: Review of machine learning in robotic grasping control in space application. Acta Astronautica (2024)

57. Yang, L., Li, K., Zhan, X., Wu, F., Xu, A., Liu, L., Lu, C.: Oakink: A large-scale knowledge repository for understanding hand-object interaction. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2022)

58. Zhou, D., Kang, B., Jin, X., Yang, L., Lian, X., Jiang, Z., Hou, Q., Feng, J.: Deepvit: Towards deeper vision transformer. arXiv preprint arXiv:2103.11886 (2021)

![](images/ebcaae014d2833252c99be66b8458e7aa592f6570efced51f32844ba95cd76ab.jpg)  
Fig. 13: Topological Clustering of Shadow Hand Grasps. Grasps grouped by contact topology (M1-M21), demonstrating consistent semantic alignment across varied object classes.
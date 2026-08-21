# GOAG: Generative and Object-Agnostic Grasp Planner for Dexterous Robotic Manipulation

Julien Mérand<sup>1</sup>, Boris Meden<sup>1</sup>, Mathieu Grossard<sup>1</sup> and Liming Chen<sup>2</sup>

Abstract— Multifingered grasping is a crucial robotic skill, but current deep-learning grasp planners often struggle to generalize to new objects because they are trained on limited, object-specific datasets. We introduce a fundamentally different approach, grounded in the observation that the gripper and the object share identical surface geometry at their mutual contact points. We propose GOAG: Generative and Object-Agnostic Grasp Planner for Dexterous Robotic Manipulation, a novel deep generative model that learns a compact latent representation of a specific gripper’s contact surface distribution, enabling the efficient sampling of valid grasp configurations without relying on object-specific training data. We show that by introducing object features only at inference time, our model can effectively retrieve admissible contact areas that are compatible with the gripper’s capabilities. We validate our approach through extensive experiments on established grasp protocols in both simulated and real-world scenarios, demonstrating its effectiveness with different grippers from the literature. Our method delivers state-of-the-art results on the objects from the MultiDex dataset, achieving an average success rate of 86.93%. It offers significantly faster processing when generating numerous grasps, while matching the performance of leading approaches specifically trained on this dataset. Unlike these methods, our approach does not rely on objectspecific training data, highlighting the advantages of objectagnostic learning. It effectively addresses the generalization challenges faced by traditional data-driven grasp planners. Code and videos are available on our project website https: //cea-list.github.io/goagweb/.

## I. INTRODUCTION

Dexterous grasping remains a highly complex and largely unsolved challenge for multi-fingered robotic hands. While sophisticated hardware like Allegro [1], ShadowHand [2], and Barrett [3] exists, fully unlocking their potential requires grasp planners that can match their kinematic complexity in real-time. Traditional analytical methods like GraspIt! [4] struggle with these time constraints. Conversely, modern data-driven approaches offer faster inference but typically evaluate performance on restricted sets of objects, severely limiting their generalization to novel, real-world shapes.

![](images/b2953ce230158e862a7df0545cd6e03c1b40d27a0891c718ee2f74137d5b855f.jpg)  
Fig. 1: GOAG Paradigm. A successful grasp on an object induces dual contact zones on both object and gripper, at the intersection of the two geometries. Our method is built on this key observation: these contact zones (C(.)) are closely the same from either perspective. GOAG capitalizes on this by training exclusively on gripper geometry, allowing it to learn a robust and generalizable grasping strategy without ever being exposed to a grasp database with specific objects geometries.

Our research addresses this generalization bottleneck by shifting from the common grasp-centric or object-centric paradigms to a gripper-centric perspective. As illustrated in Figure 1, we aim to capture the combinatorial possibilities of contact areas inherent to a specific gripper design prior to seeing any object. To this end, we formulate the first data-driven grasp planner that is object-agnostic at training time, meaning no object geometry is utilized during training, named GOAG (Generative and Object-Agnostic Grasp Planner for Dexterous Robotic Manipulation).

Instead of learning from a set of stable grasps achieved on a finite set of object geometries, GOAG learns the intrinsic distribution of feasible contact areas from a synthetic dataset that maps grasp taxonomy labels [13] (Figure 2) to randomly sampled kinematic configurations.

Specifically, we train a Conditional Variational Auto Encoder (CVAE) [14] to learn the conditional distribution of these contact points. This strategy makes the model’s training inherently object-agnostic, as it relies exclusively on the gripper’s intrinsic geometry and kinematics, enabling zeroshot generalization to arbitrary object shapes by introducing object features solely at inference time. The model’s input is a Basis Point Set (BPS) [15] encoding of a point cloud. During inference, using the same BPS encoder, the model adeptly retrieves potential contact areas on the object shape that are compatible with the gripper’s kinematics. An additional PointNet++ [16] network then associates these identified contact zones with specific gripper links, and the gripper joints are subsequently optimized to solve this mapping problem, enabling precise grasp execution. In summary, our contributions are the following: 1) Object-Agnostic Learning Strategy. We introduce a novel training paradigm for dexterous grasping that relies exclusively on the gripper’s intrinsic geometry and kinematics. By decoupling the learning phase from object data, we eliminate the bias toward specific object datasets inherent in traditional methods. 2) Demonstrated Efficiency and Generalization. We provide extensive experimental validation in both simulated and realworld environments. By following established evaluation protocols [10] and extending testing to novel objects, we demonstrate our approach’s generalization compared to existing methods.

<table><tr><td></td><td>Grasp Representation</td><td>Gripper Pose</td><td>Gripper Joint Values</td><td>Force Closure</td><td>Non- Penetration</td><td>Training Set</td><td>Working Reference Frame</td><td>Optional Grasp Preference Interface</td></tr><tr><td>DFC [5]</td><td>Direct</td><td>Optimized</td><td>Optimized</td><td></td><td></td><td>No</td><td>Object</td><td>X</td></tr><tr><td>UniGrasp [6]</td><td>Intermediate</td><td>IK Solved</td><td>IK Solved</td><td></td><td>X</td><td>Objects + Grippers</td><td>Object</td><td>x</td></tr><tr><td>GeoMatch [7]</td><td>Intermediate</td><td>IK Solved</td><td>IK Solved</td><td></td><td>x</td><td>Objects + Grippers</td><td>Object</td><td>×</td></tr><tr><td>GenDexGrasp [8]</td><td>Intermediate</td><td>Optimized</td><td>Optimized</td><td></td><td></td><td> $\mathrm { O b j e c t s + G r i p p e r s }$ </td><td>Object</td><td>x</td></tr><tr><td>ManiFM [9]</td><td>Intermediate</td><td>Optimized</td><td>Optimized</td><td></td><td></td><td>Objects + Grippers</td><td>Object</td><td>Contact Region</td></tr><tr><td>DRO-Grasp [10]</td><td>Intermediate</td><td>Optimized</td><td>Optimized</td><td></td><td></td><td> $\mathrm { O b j e c t s + G r i p p e r s }$ </td><td>Object</td><td>Palm Orientation</td></tr><tr><td>DexDiffuser [11]</td><td>Direct</td><td>Learned</td><td>Learned</td><td>A posteriori</td><td>A posteriori</td><td>Objects + Grippers</td><td>Object</td><td>X</td></tr><tr><td>DexGrasp Anything [12]</td><td>Direct</td><td>Learned</td><td>Learned</td><td></td><td></td><td>Objects + Grippers</td><td>Object</td><td>x</td></tr><tr><td>GOAG (Ours)</td><td>Intermediate</td><td>Sampled</td><td>Optimized</td><td>4</td><td></td><td>Gripper Only</td><td>Gripper</td><td>Palm Full Pose</td></tr></table>

TABLE I: Comparison of Dexterous Grasp Planning Methods.

## II. RELATED WORK

Recent advancements in dexterous grasp planning have been largely propelled by data-driven approaches [17]. Table I summarizes their characteristics.

## A. Grasp Databases

The success of modern data-driven models heavily relies on large-scale grasp databases. While early datasets relied on time-intensive human demonstrations [18], motion capture [19], [20], teleoperation [21] or even thermal imaging [19], [22], recent works have pioneered the automated, synthetic generation of thousands of physically validated candidate grasps [8], [12], [23]–[25], based on analytical grasp planners [4], [5] and validating stability using physically realistic simulators like Isaac Gym [26].

These databases are costly to generate and because they map specific grippers to specific objects, models trained on them remain intrinsically biased by the training geometry, hindering true generalization to unseen shapes.

## B. Learning Explicit Grasps

A primary family of approaches directly learns the mapping from an object’s representation to a complete grasp configuration, including the gripper’s pose and joint values. This category spans from early regression models [27], [28] to sophisticated generative architectures by, modeling the conditional probability distribution [29], [30], using diffusion models [11], [31], [32], or foundation models [9], [12], [33]. Despite their expressiveness and ability to process complex point clouds, these direct-prediction methods share a common limitation: the generated poses do not always inherently satisfy physical constraints. They frequently require a computationally expensive post-hoc validation step or a learned discriminator to filter out penetrating or unstable grasps.

![](images/4ea1f456c715bc7cdabcfbcbf035fb5eb3cb72ba0db5d5987a70629598955738.jpg)  
Fig. 2: Grasp Taxonomy Adaptation and Contact Sampling. (Top) We adapt the human grasp taxonomy from [13] to the Allegro Hand geometry. For each grasp type (e.g., C6, F27), we define a corresponding admissible contact region (black points), distinguishing it from the non-contact surface (blue points). (Bottom) Data generation mechanism: We randomly sample specific contact points (red) strictly within the admissible black regions. This allows the model to learn structured, feasible contact distributions based solely on gripper kinematics, independent of any object.

## C. Learning Intermediate Representation

Instead of directly predicting a grasp pose, alternative methods learn an intermediate representation, most commonly a contact map defining desired grasp regions on the object’s surface [7], [8], [29], [34]. The final grasp pose is then derived using an inverse kinematics solver, which ensures non-penetration and inherits stability from the training data. Other methods extract gripper-specific features to to tackle the multi-embodiment challenge [6], [10], [35]. While these intermediate approaches improve physical validity, they still depend on pre-existing object-grasp databases for training. As highlighted in Table I, GOAG fundamentally breaks from this paradigm. By modeling the gripper’s contact capabilities entirely independently of object data, we achieve an object-agnostic training process that does not require pre-computed grasp examples, which are often costly to generate.

## III. METHOD

Our goal is to estimate numerous grasps when presented with a specific object shape. The specificity of our approach is that the training phase only considers the gripper geometry, while the inference phase only considers the targeted object geometry. The underlying notion is a shift in perspective regarding the contact region: instead of defining it on the object, we define it as an intrinsic property of the gripper.

![](images/e4161f16f760a9829eb6a98b902cace776f2ba18663c61bbce0945965b578142.jpg)  
Fig. 3: Overview of GOAG. Geometrical graspability is learned in an object-agnostic manner by focusing on the gripper’s capabilities. Training: We sample gripper configurations Q to generate $\mathcal { H } ( Q )$ and corresponding contact points $( { \mathcal { C } } ( { \mathcal { H } } ( { \mathbf { \breve { Q } } } ) ) )$ . To ensure transferability, we use a Basis Point Set (BPS) encoding tied to the gripper’s workspace. A Conditional Variational Autoencoder (CVAE) is trained to reconstruct these contact distributions, while a Links Mapper (PointNet++) learns to associate contact points with specific gripper links. Inference: A novel object O, positioned at the inverse gripper pose $[ R , \dot { T } ] ^ { - 1 }$ , is BPS-encoded. By sampling a latent variable $z \in \mathbb { R } ^ { \psi }$ the CVAE Decoder generatively predicts diverse, plausible contact points $\widehat { \mathcal { C } } ( \mathcal { O } )$ . The Links Mapper then labels which gripper link should reach each point. Generation: Finally, a Force Closure check ensures the predicted contacts yield a stable grasp, and a Grasp Optimization step outputs the final, refined gripper configuration $Q ^ { * }$

For the remainder of this section, we fix the pose of the gripper, defined by the rotation matrix $R \in \mathrm { S O } ( 3 )$ , and the translation vector $T \in \mathbb { R } ^ { 3 }$ , relative to the reference frame of the object O. We characterize a grasp by the pose of the gripper [R, T] coupled with its joint values, Q. We define the gripper handprint $\mathcal { H } = \{ h _ { i } \}$ as the set of points on the gripper’s surface that represent its intrinsic contact capabilities, located on the active grasping surfaces (i.e., the palm and the inward-facing surfaces of the links).

## A. Shifting to a Gripper-Oriented Paradigm

Given a gripper handprint $\mathcal { H } = \{ h _ { i } \}$ and an object point cloud $\mathcal { O } = \{ o _ { i } \}$ , the set of contact points C(.) is typically defined within the object’s coordinate frame and consists of points on the object’s surface that fulfill a certain criterion of distance with the gripper. Following [8], we express it as:

$$
\begin{array} { r l } & { \mathcal { C } ( \mathcal { O } ) = } \\ & { \{ o _ { i } \in \mathcal { O } , \exists h _ { j } \in \mathcal { H } ( R , T , Q ) \ | \ \mathcal { D } _ { a l i g n e d } ( o _ { i } , h _ { j } ) < \epsilon \} , } \end{array}\tag{1}
$$

where $\mathcal { H } ( R , T , Q )$ represents the gripper handprint positioned at the grasp pose $[ R , T ]$ with the joint configuration Q, and where the function $\mathcal { D } _ { a l i g n e d }$ is the aligned distance between any two points x and y, as introduced by [8]:

$$
\mathcal { D } _ { a l i g n e d } ( x , y ) = e ^ { \gamma ( 1 - \langle x - y , n _ { x } \rangle ) } \sqrt { \| x - y \| _ { 2 } } ,\tag{2}
$$

with $n _ { x }$ the surface normal at x and $\gamma \textbf { a }$ scaling factor. In our approach, we propose to recast this paradigm in an alternative gripper-oriented viewpoint: By applying the inverse transformation $[ R , T ] ^ { - 1 }$ to the object, we can analyze the grasp in the gripper’s canonical coordinate frame. This refraiming allows us to define the contact points as a subset of the gripper’s surface rather than the object’s:

$$
\begin{array} { r l } & { \mathcal { C } ( \mathcal { H } ( Q ) ) = } \\ & { \{ h _ { i } \in \mathcal { H } ( Q ) , \exists o _ { j } \in \mathcal { O } ( [ R , T ] ^ { - 1 } ) \ | \ \mathcal { D } _ { a l i g n e d } ( h _ { i } , o _ { j } ) < \epsilon \} , } \end{array}\tag{3}
$$

where $\mathcal { H } ( Q )$ is the gripper in its default pose (at origin with no rotation) and joint configuration Q, while ${ \mathcal { O } } ( [ R , T ] ^ { - 1 } )$ is the object transformed into the gripper’s frame.

While Equation 3 defines the physical condition for contact with a specific object, the set of all kinematically feasible contact patterns is bounded by the gripper’s geometry and constraints.

Object-Agnostic Contact Priors. Our key insight is that we can learn this intrinsic distribution of feasible contact zones solely from the gripper’s kinematics and grasp taxonomy. This shifts the problem from finding specific contacts for a specific object (which requires O) to learning the gripper’s intrinsic contact priors (which is object-independent). Consequently, we can pre-compute a manifold of valid contact surfaces directly on the gripper in a multitude of configurations, which are then queried against object shapes only at inference time.

## B. Training GOAG from C(H(Q))

We build on this observation to formulate our objectagnostic training procedure. The training process requires generating diverse gripper configurations, encoding them into a consistent spatial representation, and training two parallel networks to predict contact distributions and kinematic assignments. The following architecture has been explored further in [36].

Contact Data Generation. The training data is generated exclusively from the gripper’s kinematics and an adapted taxonomy, as illustrated in Figure 2. We first uniformly sample valid joint configurations $Q$ within the gripper’s kinematic limits. For each configuration, we randomly select a grasp type from the taxonomy presented in [13], which defines specific admissible contact regions on the gripper surface. Finally, we generate the target contact map $\mathcal { C } ( \mathcal { H } ( Q ) )$ by randomly sampling active contact points strictly within these admissible regions. While unconstrained random sampling could theoretically be used to populate the dataset, leveraging a taxonomy ensures that every generated sample is structurally valid. It acts as a crucial prior to guarantee that the sampled contact points exhibit a correct kinematic harmony, representing realistic, synergistic grasps rather than arbitrary and independent surface contacts. This procedural generation creates a diverse dataset of kinematically feasible contact distributions independent of any object geometry.

BPS Encoding and Distance Field. To allow our neural networks to process variable-length point clouds efficiently, we project our input data – denoted generally as $P ,$ which represents the gripper point cloud $\mathcal { H } ( Q )$ during training or the object point cloud O during inference – into a fixedsize representation using a specialized Basis Point Set [15] (BPS). Instead of a standard bounding box, we discretize the gripper’s kinematic workspace to form our BPS, denoted as $W \in \mathbb { R } ^ { M \times 3 }$ . This workspace represents the volume of points reachable by the gripper relative to a fixed palm frame, naturally concentrating the data where grasps occur. This projection implies an inherent train-test domain shift between the training $( { \mathcal { H } } ( Q ) )$ and testing (O) geometry, a crossdomain transfer challenge that has been explored further in [36]. For any input point cloud $P \in \mathbb { R } ^ { p \times 6 }$ , comprising 3D spatial coordinates concatenated with their corresponding 3D surface normals, we compute an aligned distance field $\mathcal { F } _ { P } \in \mathbb { R } ^ { M }$ . Following [8], the distance from each basis point $w _ { i } \in W$ to the point cloud P is defined as:

$$
{ \mathcal { D } } ( P , w _ { i } ) = \operatorname* { m i n } _ { p _ { j } \in P } { \mathcal { D } } _ { a l i g n e d } ( p _ { j } , w _ { i } ) .\tag{4}
$$

To provide a supervision signal, we define the projected contact field $\tilde { \mathcal { C } } ( \bar { P } ) = \{ c _ { i } \} _ { i = 1 } ^ { M } \bar { }$ . Each component $c _ { i } \in [ 0 , 1 ]$ quantifies the proximity and alignment of each w<sub>i</sub> to the active contact points $\mathcal { C } ( P )$ . Following [29], this is computed as:

$$
c _ { i } = 1 - 2 \times ( \mathrm { S i g m o i d } ( \mathcal { D } ( \mathcal { C } ( P ) , w _ { i } ) ) - 0 . 5 ) .\tag{5}
$$

This function provides a continuous "contact likelihood," approaching 1 when $w _ { i }$ is close and aligned to a true contact point, and 0 when it is distant or misaligned.

CVAE Architecture and Training. With our data structured into the BPS representation $\mathcal { F } _ { P }$ and contact labels $\tilde { \mathcal { C } } ( P )$ , we train a Conditional Variational Autoencoder [14] (CVAE) to learn the distribution of feasible contacts. The encoder takes the concatenated input $[ \mathcal { F } _ { P } , \tilde { \mathcal { C } } ( P ) ]$ of shape $M \times 2$ and predicts the parameters of the posterior distribution $q _ { \theta } ( z \mid$ $\mathcal { F } _ { P } , \tilde { \mathcal { C } } ( P ) ,$ ). A latent variable $z \in \mathbb { R } ^ { \psi }$ is then sampled from this distribution. The decoder, following [30], consists of two Fully Connected ResBlocks that take the concatenation of z and $\mathcal { F } _ { P }$ to output predicted contact values $\widehat { \mathcal { C } } ( P )$ . The network is optimized using the following loss function:

$$
L = L _ { \mathrm { { r e c o n } } } + \beta D _ { \mathrm { { K L } } } ( q _ { \theta } ( z \mid \mathcal { F } _ { P } , \tilde { \mathcal { C } } ( P ) ) \parallel \mathcal { N } ( 0 , I ) ) .\tag{6}
$$

where $\beta$ weights the Kullback-Leibler divergence. The reconstruction loss $L _ { \mathrm { r e c o n } }$ is an $L _ { 2 } { \mathrm { - N o r m } }$ scaled by an attention weight $e ^ { \alpha c _ { i } }$ to heavily prioritize the accuracy of highlikelihood contact regions:

$$
L _ { \mathrm { r e c o n } } = \sqrt { \frac { \sum _ { i = 1 } ^ { M } ( c _ { i } - \hat { c } _ { i } ) ^ { 2 } \times e ^ { \alpha c _ { i } } } { \sum _ { k = 1 } ^ { M } e ^ { \alpha c _ { i } } } } .\tag{7}
$$

Links Mapper. In parallel to the CVAE, a PointNet++ [16] is trained on the sampled point clouds to associate each point $p _ { i } \in { \mathcal { C } } ( P )$ with the corresponding gripper phalanx link $l _ { i } .$ This assigns kinematic meaning to the spatial points, which is crucial for retrieving the full gripper joint configuration during the downstream optimization phase.

## C. GOAG Inference on ${ \mathcal { O } } ( [ R , T ] ^ { - 1 } )$

Once trained, the models can be deployed to predict contact zones on unseen objects. Given a novel object point cloud O transformed into the gripper’s canonical workspace via $[ R , T ] ^ { - 1 }$ , we first compute its aligned distance field $\mathcal { F } _ { \mathcal { O } }$ against the basis set $W .$ , identical to the training phase. Bypassing the encoder, we sample a latent vector z ∼ $\mathcal { N } ( 0 , I )$ . The decoder takes the concatenation $[ z , \mathcal { F } _ { \mathcal { O } } ]$ and predicts the continuous contact field $\widehat { \mathcal { C } } ( \mathcal { O } )$ . Instead of projecting these values back onto the object’s surface, we operate directly within the workspace domain. We extract the intended contact locations by isolating the basis points that exhibit a high predicted contact likelihood $( \hat { c } _ { i } > \tau$ , where $\tau$ is a confidence threshold). This yields a discrete set of contact points $\mathcal { C } ( W )$

Finally, the pre-trained Links Mapper evaluates $\mathcal { C } ( W )$ to assign each point to a specific gripper phalanx. While this generates a geometrically plausible contact map, these predictions are sampled from a latent prior and do not guarantee physical stability. Consequently, the pipeline concludes with a rapid force-closure estimation followed by a full kinematic optimization to refine the final grasp execution, using $\mathcal { C } ( W )$ as the targets for the gripper’s links.

Contact Points Force-Closure Estimation. Unlike methods trained on pre-validated grasp datasets [8], [10], our generative approach requires an explicit assessment of grasp stability. To ensure the inferred contact points can theoretically yield a stable grasp before performing the computationally expensive joint configuration optimization, we evaluate the Force-Closure condition directly on the predicted contact points $\mathcal { C } ( W )$ .

a) Contact Modeling: We first group the predicted contact points by their associated gripper phalanx indices. To enforce a unique contact location per phalanx, we compute the barycenter of each cluster and project it onto the object’s surface, yielding a set of discrete contact locations $b _ { i } .$ . We assume a Coulomb friction model with a friction coefficient of $\mu = 0 . 3$ . As this estimation is performed in the gripper’s reference frame to assess geometric feasibility, we do not consider the gravity or object’s weight.

b) Force-Closure Computation: We compute the grasp wrench space, defined as the convex hull of all possible wrenches generated by unit contact forces at locations $b _ { i }$ The total wrench is given by:

$$
d = \sum _ { i = 1 } ^ { k } d _ { i } = \sum _ { i = 1 } ^ { k } G _ { i } f _ { i }\tag{8}
$$

where $G _ { i }$ is the partial grasp matrix for contact $b _ { i } ,$ and $f _ { i }$ represents the primitive force vectors along the edges of the friction cone, normalized such that $\| f _ { i } \| = 1$ . We verify the force-closure condition by checking if the origin of the wrench space lies strictly within the interior of this convex hull.

c) Resampling Strategy: If the force-closure condition is not met, it implies the predicted contact distribution is unstable. In this case, we discard the prediction and sample a new latent variable $z \in \mathbb { R } ^ { \psi }$ to generate a fresh $\hat { \mathcal { C } } ( \mathcal { O } )$ . To prevent exhaustive searches when an object pose $( [ R , T ] ^ { - 1 } )$ is poorly conditioned, specifically when it falls near the boundaries of the gripper’s workspace where valid contacts are scarce, we limit this resampling process to a maximum of 20 iterations. It is important to note that this step validates the stability potential of the contact configuration using the simplified barycenter model. The final physical grasp quality is determined by the gripper’s ability to reach these point during the subsequent optimization, which optimizes the gripper to fit the full dense contact regions rather than just these discrete barycenters.

Grasp Generation. Determining the gripper joint configuration $Q ^ { * }$ that enables contact with the inferred object contact points is formulated as an optimization problem under nonpenetration constraints. While some methods minimize the Euclidean distance between a predicted grasp and the gripper’s actual link transformations [10] or between specific points on the gripper and the object [34], this often lacks the necessary mechanisms to prevent hand-object and selfpenetrations, which are essential for physical feasibility.

Instead, building on approaches like [8], [34], we determine the optimal joint configuration $Q ^ { * }$ by aligning the gripper with the inferred contact points while strictly penalizing collisions. We achieve this by minimizing the following energy function:

$$
Q ^ { * } = \underset { Q } { \mathrm { a r g m i n } } ( \lambda _ { d } E _ { d i s t } + \lambda _ { p } E _ { p e n } + \lambda _ { s } E _ { s p e n } + \lambda _ { j } E _ { j } )\tag{9}
$$

where $\lambda$ terms are weighting coefficients set to $\lambda _ { d } = 1 . 0 $ $\lambda _ { p } = 5 0 . 0 , \lambda _ { s } = 0 . 1 , \lambda _ { j } = 1 . 0$ . The components are defined as follows:

a) Contact Distance $( E _ { d i s t } ) .$ . We minimize the distance from the predicted contact points $\mathcal { C } ( W )$ on the object to their assigned gripper links. Utilizing the link indices predicted by the Links Mapper (Section III-B), we define:

$$
E _ { d i s t } = \sum _ { p \in \mathcal { C } ( W ) } \operatorname* { m i n } _ { h \in \mathcal { H } _ { l ( p ) } ( Q ) } | | p - h | | _ { 2 } ^ { 2 } ,
$$

where $\mathcal { H } _ { l ( p ) } ( Q )$ represents the subset of handprint points belonging to the specific link $l ( p )$ assigned to contact point $p \in { \mathcal { C } } ( W )$ . This ensures the gripper fingers move precisely to the target regions.

b) Object Penetration $\phantom { } ( E _ { p e n } ) \colon$ To prevent the gripper from intersecting the object volume, we penalize gripper points that penetrate the object’s signed distance field, denoted as $\Phi _ { \mathcal { O } } ( \cdot )$

$$
E _ { p e n } = \sum _ { h \in \mathcal { H } ( Q ) } \operatorname* { m a x } ( 0 , - \Phi _ { \mathcal { O } } ( h ) )
$$

c) Self-Penetration $( E _ { s p e n } ) .$ : We enforce self-collision avoidance by penalizing distances between disjoint gripper links i and $j$ that fall below a safety threshold ϵ (set to 0.025m):

$$
E _ { s p e n } = \sum _ { i , j } \operatorname* { m a x } ( 0 , \epsilon - \mathrm { d i s t } ( k _ { i } ( Q ) , k _ { j } ( Q ) ) ) ^ { 2 } ,
$$

where $k _ { i } ( Q )$ denotes the centroid of the bounding box for link i in configuration $Q .$

d) Joint Limits $( E _ { j } ) \colon$ Finally, we constrain the joints to remain within their hardware kinematic limits $[ Q _ { m i n } , Q _ { m a x } ]$

$$
E _ { j } = \left\| \operatorname* { m a x } ( 0 , Q - Q _ { m a x } ) \right\| + \left\| \operatorname* { m a x } ( 0 , Q _ { m i n } - Q ) \right\|
$$

## IV. EXPERIMENTS

## A. Implementation Details

Training Data Generation. To ensure dense coverage of the gripper’s reachable workspace, we uniformly sampled 10, 000 kinematically valid gripper configurations Q and computed their corresponding surface point clouds $\mathcal { H } ( Q )$ These active point clouds are extracted from the full mesh by applying a gripper-specific threshold to the dot product of each vertex normal and the palm’s facing direction, effectively isolating the primary contact pads and the immediate lateral sides of the fingers. We utilized the grasp taxonomy adapted from [13], selecting the 6 most common and suitable grasp types for robotic grippers, as detailed in Figure 2. According to the literature [37], [38], these grasp types are utilized for 92.5% of the total working time by machinists and 96.2% of the time by housemaids. For each configuration and its assigned grasp type, we uniformly sampled 50 sets of admissible contact points strictly within each active regions defined by the taxonomy. This process yielded a dataset of 3, 000, 000 labeled point clouds.

Computational Efficiency. A significant advantage of our object-agnostic formulation is the speed of data generation. The entire dataset creation required approximately 1 GPU hour on a single Nvidia RTX 4090. In contrast, a previous work [8] reported a much longer generation time of 1,400 GPU hours using Nvidia A100.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Data Driven</td><td rowspan="2">Object-Agnostic Training</td><td colspan="4">Success Rate (%)↑</td><td colspan="3">Efficiency (sec. / grasps) ↓</td><td colspan="3">Diversity (avg.) ↑</td></tr><tr><td>Barrett</td><td>Allegro</td><td>ShadowHand</td><td>Avg.</td><td>Barrett</td><td>Allegro</td><td>ShadowHand</td><td>T (m)</td><td>R (rad)</td><td>Q (rad)</td></tr><tr><td>DFC [5]</td><td> $x$ </td><td>√</td><td>83.10</td><td>82.71</td><td>72.15</td><td>79.32</td><td>&gt;1800</td><td>&gt;1800</td><td>&gt;1800</td><td>0.0607</td><td>1.424</td><td>0.3579</td></tr><tr><td>GenDexGrasp [8] (full)</td><td>V</td><td>x</td><td>70.26</td><td>71.48</td><td>71.15</td><td>70.96</td><td>9.78</td><td>16.45</td><td>14.65</td><td>0.0519</td><td>1.416</td><td>0.2567</td></tr><tr><td>DRO-Grasp [10] (pretrain, w/o controller)</td><td> $\checkmark$ </td><td>x</td><td>78.30</td><td>75.80</td><td>63.30</td><td>72.47</td><td>0.88</td><td>0.42</td><td>1.72</td><td>0.0546</td><td>1.515</td><td>0.2892</td></tr><tr><td>GOAG (w/o FC)</td><td>√</td><td>√</td><td>86.30</td><td>91.20</td><td>74.70</td><td>84.07</td><td>0.09</td><td>0.13</td><td>0.15</td><td>0.0480</td><td>1.396</td><td>0.3162</td></tr><tr><td>GOAG</td><td>V</td><td>√</td><td>87.40</td><td>93.20</td><td>77.90</td><td>86.93</td><td>0.18</td><td>0.19</td><td>0.20</td><td>0.0479</td><td>1.401</td><td>0.3170</td></tr></table>

TABLE II: In-depth grasp performance analysis on the Multidex [8] test set. We report our grasp results, on three classic dexterous grippers in terms of success rate, efficiency and diversity.

Network Hyperparameters. We discretized the workspace W using $M \ = \ 8 , 1 9 2$ points, set the scaling factor $\gamma =$ 2.0 for the aligned distance (Eq. 2) and used $\tau = 0 . 8$ as confidence threshold. The CVAE was trained with a latent dimension size of $\psi = 1 2 8$ , using a weighting parameter $\beta = 0 . 0 1$ and an attention factor $\alpha = 3 . 0$ . We trained the CVAE for 100 epochs and the PointNet++ links mapper for 50 epochs. Training was parallelized across multiple Nvidia A100 GPUs with a batch size of 128.

## B. Grasp Pose Constraints and Generation

A key advantage of this method is the flexibility to impose the object’s pose [R, T]. In a practical grasping scenario, the transformation between the gripper and the object is typically constrained by the task and the physical environment. Consequently, generating grasp poses that are not physically feasible is inefficient. Furthermore, objects with translational or rotational symmetries (e.g., a cylinder rotating around its axis of revolution) can be grasped effectively with a single pose inference, as it provides a broad range of valid grasp configurations for that object type.

Grasp Pose Generation Strategy. To address the need for comprehensive grasp coverage, particularly for objects lacking significant symmetry, we propose a strategy, founded on [23], for generating grasp candidates. We sample gripper poses uniformly on the object’s convex hull, which is dilated by 110%. The gripper’s palm is oriented to point opposite to the hull’s normal vector. This approach ensures that the generated grasp poses are both diverse and well-distributed across the object’s surface.

## C. Evaluation Metrics

Success Rate: We quantify grasp success using the Isaac Gym [26] simulator. We initialize the object and gripper in the optimized configuration $Q ^ { * }$ and, similar to [8], apply sequential external forces on the object along the ±xyz directions for one second each. A grasp is considered successful if the object deviates less than 2 cm from its original pose. This simulation-based verification is essential because the prior Force Closure Estimation (Section III-C) validates only the theoretical stability of simplified barycenters. The subsequent Joint Optimization (Section III-C) is the first step to incorporate the full object geometry and gripper kinematics. Consequently, contact distributions deemed valid by the force-closure estimation may prove kinematically unreachable or geometrically obstructed by the object’s shape in practice, preventing a perfect match and necessitating physical validation of the final grasp.

![](images/7072306350a16836476befd08116c427ab4bde5387d5d72c82d27ab930ea6303.jpg)  
Fig. 4: GOAG grasp results on Multidex [8] objects. Grasps are shown for the Barrett (green), Allegro (pink), and Shadow Hand (purple) grippers.

Efficiency: We evaluate the efficiency of all models by calculating the average time required to generate a single grasp. This measurement is based on the total time to produce 100 grasps, including both the model inference and optimization phases. The resulting value is then normalized to a single grasp. Following [10], the time for the Isaac Gym simulation is excluded from this metric.

Diversity: We measure diversity as the standard deviation of the gripper position (T), orientation (R), and joint values (Q). Although the candidate gripper poses (R, T) are initially sampled, the variability observed in successful grasps is not arbitrary; it is intrinsically shaped by the object’s specific geometry and the gripper’s kinematic capabilities, which naturally filter the reachable contact manifold. We therefore report diversity across all components to demonstrate this geometric adaptation, while specifically highlighting the variance in Q to verify our model’s generative ability to synthesize distinct finger configurations for similar poses.

## D. Experimental Protocols

In-depth grasp performance analysis. We evaluated our method and several baselines on three distinct robotic grippers: the Barrett Hand [3], Allegro Hand [1], and Shadow Hand [2]. Our evaluation involved 100 inference runs per method, using a batch size of 10 to manage GPU memory. We extensively compare GOAG against the following baselines using the test set of the Multidex [8] dataset. It comprises 10 objects from the ContactDB [22] and YCB [39] object sets. We evaluated DFC [5] on grasps from the CMap-Dataset. The grasp poses were originally generated using DFC’s method and subsequently refined, in [8], through a post-filtering process to get a clean dataset. This led to a higher success rate than what was originally reported. We used a pre-trained model checkpoint of GenDexGrasp [8] with data from multiple robot hands and default hyperparameters. We first generated contact maps for the object set and then derived the grasp poses using the optimization framework provided by the authors. We selected the most effective version of DRO-Grasp [10], which includes configurationinvariant pre-training. We kept all hyperparameters at their default values and deactivated the simple grasp controller for a fair comparison.

<table><tr><td>Method</td><td>Per-Dataset Training</td><td>DexGraspNet ↑</td><td>UniDexGrasp ↑</td><td>MultiDex ↑</td><td>RealDex ↑</td><td>DexGRAB ↑</td><td>Avg. (%)</td></tr><tr><td>UniDexGrasp [25]</td><td>√</td><td>33.9</td><td>23.7</td><td>21.6</td><td>27.1</td><td>20.8</td><td>25.42</td></tr><tr><td>GraspTTA [29]</td><td></td><td>18.6</td><td>21.0</td><td>30.3</td><td>13.3</td><td>14.4</td><td>19.52</td></tr><tr><td>SceneDiffuser [31]</td><td></td><td>26.6</td><td>28.3</td><td>69.8</td><td>21.7</td><td>39.1</td><td>37.10</td></tr><tr><td>UGG [32]</td><td>V</td><td>46.9</td><td>46.0</td><td>55.3</td><td>32.7</td><td>42.7</td><td>44.72</td></tr><tr><td>DGA [12]</td><td>V</td><td>57.5</td><td>53.1</td><td>79.1</td><td>44.8</td><td>57.9</td><td>58.48</td></tr><tr><td>GOAG</td><td>X</td><td>43.07</td><td>49.51</td><td>77.90</td><td>37.37</td><td>62.13</td><td>53.97</td></tr></table>

TABLE III: Grasp generalization assessment across multiple grasp datasets. Following [12] we evaluate our method performances with the Shadow hand on multiple grasp test sets. We report the success rates as defined in IV-C. It is worth noting that state-of-the-art methods are retrained for each dataset. GOAG has only been trained once on Shadow hand kinematics.

Generalization across multiple grasp datasets. To extend the evaluation of our method on a wide range of objects, we followed [12] by using the test set of three simulated datasets [8], [23], [25], a real-world dataset [21] and a human hand dataset [20] retargeted to dexterous hand parameters by [12]. This makes a total of 3438 objects from five different benchmarks. The evaluation protocol remains consistent with the procedure described above, with all experiments conducted exclusively using the Shadow Hand.

## E. Overall Performances

In-depth grasp performance analysis. On the Multidex dataset (Table II), our method, GOAG, demonstrates superior performance. While being trained in an object-agnostic manner, GOAG achieved a higher average success rate for generating accurate grasps across all three grippers compared to the baselines. This high accuracy is paired with good efficiency. While DRO-Grasp [10] may appear faster for a single grasp, its processing time scales linearly with the number of grasps, as each is optimized independently. Our method, in contrast, benefits from a vectorized optimization process that minimizes a single energy function for all grasp candidates simultaneously. This approach, while having a higher initial overhead, results in a more consistent and significantly faster processing time when generating a large number of grasps. Furthermore, our network is more compact than DRO-Grasp’s in terms of parameter count. Compared to GenDexGrasp [8], which does not vectorize its optimization, our approach gains a substantial efficiency advantage. The strong performance of our model even without an explicit force closure estimation confirms a key design principle: the model successfully learns to map gripper capabilities and shapes onto objects. This demonstrates that the expressiveness of our training set enables the model to implicitly learn robust grasp mechanics. Because our model is grippercentric, the diversity of grasps for a given object depends on how comprehensively we sample object poses, rather than being a property of the network’s output itself. However, our model can still generate multiple grasp types for a single object pose by sampling from the CVAE’s latent space, providing a means for diverse grasp synthesis.

Generalization across multiple grasp datasets. We report the success rates achieved across all five benchmarks using the Shadow Hand [2] in Table III. Our method achieved the second-highest average success rate across all five datasets. This is particularly noteworthy because all competing methods were specifically trained on each respective dataset, whereas GOAG was trained only once on the Shadow Hand in an object-agnostic manner. These results were achieved using a simple pose sampling method (IV-B), which is well-suited for objects with a volume enclosed within the gripper’s workspace. For larger objects, however, this approach is less effective. Adopting a more advanced sampling strategy specifically adapted for such objects would likely yield stronger grasp performance.

![](images/63f65add046a5beb4d764dc23c8758a4a2397d44285c3cdbf7f7ec2eb24fc411.jpg)  
Fig. 5: Real-world setup and results with Allegro hand on YCB [39] objects. First row presents the real robot grasps. Second row presents corresponding virtual grasps. Objects have been rotated around the z-axis for a better understanding of the grasp poses.

## F. Real-Robot Experiments

We conducted experiments using an Allegro Left Hand mounted on a 7-DoF robot arm. We successfully grasped 11 objects from the YCB Dataset, demonstrating the method’s ability to transfer to real-world objects. Successful grasps are illustrated Figure 5 and the setup is shown Figure 5. Videos of the experiments are provided in supplementary material.

## V. CONCLUSION

This paper introduces GOAG, a novel learning paradigm for data-driven grasp planners. Our approach leverages the strengths of deep learning while eliminating the need for a large-scale, object-specific grasp database. By adopting a gripper-centric training phase that makes no assumptions about object shapes, our model learns a generalizable grasp strategy. Extensive evaluations across multiple benchmarks demonstrate that our method achieves performance competitive with state-of-the-art approaches, even without training on their specific datasets. This highlights the strong generalization capabilities of our formulation, which we also validate with a successful real-robot deployment.

## REFERENCES

[1] “Allegro Hand V4.” [Online]. Available: https://www.allegrohand. com/v4

[2] P. Tuffield and H. Elias, “The shadow robot mimics human actions,” Industrial Robot: An International Journal, MCB UP Ltd, 2003.

[3] W. Townsend, “The barretthand grasper–programmably flexible part handling and assembly,” Industrial Robot: an international journal, MCB UP Ltd, 2000.

[4] A. T. Miller and P. K. Allen, “Graspit! a versatile simulator for robotic grasping,” IEEE Robotics & Automation Magazine, 2004.

[5] T. Liu, Z. Liu, Z. Jiao, Y. Zhu, and S.-C. Zhu, “Synthesizing diverse and physically stable grasps with arbitrary hand structures using differentiable force closure estimator,” IEEE Robotics and Automation Letters, 2021.

[6] L. Shao, F. Ferreira, M. Jorda, V. Nambiar, J. Luo, E. Solowjow, J. A. Ojea, O. Khatib, and J. Bohg, “Unigrasp: Learning a unified model to grasp with multifingered robotic hands,” IEEE Robotics and Automation Letters, 2020.

[7] M. Attarian, M. A. Asif, J. Liu, R. Hari, A. Garg, I. Gilitschenski, and J. Tompson, “Geometry matching for multi-embodiment grasping,” in Conference on Robot Learning, 2023.

[8] P. Li, T. Liu, Y. Li, Y. Geng, Y. Zhu, Y. Yang, and S. Huang, “Gendexgrasp: Generalizable dexterous grasping,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023.

[9] Z. Xu, C. Gao, Z. Liu, G. Yang, C. Tie, H. Zheng, H. Zhou, W. Peng, D. Wang, T. Hu et al., “Manifoundation model for general-purpose robotic manipulation of contact synthesis with arbitrary objects and robots,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024.

[10] Z. Wei, Z. Xu, J. Guo, Y. Hou, C. Gao, Z. Cai, J. Luo, and L. Shao, “D(R, O) grasp: A unified representation of robot and object interaction for cross-embodiment dexterous grasping,” in 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025.

[11] Z. Weng, H. Lu, D. Kragic, and J. Lundell, “Dexdiffuser: Generating dexterous grasps with diffusion models,” IEEE Robotics and Automation Letters, 2024.

[12] Y. Zhong, Q. Jiang, J. Yu, and Y. Ma, “Dexgrasp anything: Towards universal robotic dexterous grasping with physics awareness,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025.

[13] J. M. Escorcia-Hernandez, M. Grossard, and F. Gosselin, “Taskoriented methodology combining human manual gestures and robotic grasp stability analyses: application to the specification of dexterous robotic grippers,” Journal of Mechanical Design, American Society of Mechanical Engineers, 2023.

[14] K. Sohn, H. Lee, and X. Yan, “Learning structured output representation using deep conditional generative models,” Advances in neural information processing systems, 2015.

[15] S. Prokudin, C. Lassner, and J. Romero, “Efficient learning on point clouds with basis point sets,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019.

[16] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “Pointnet++: Deep hierarchical feature learning on point sets in a metric space,” Advances in neural information processing systems, 2017.

[17] R. Newbury, M. Gu, L. Chumbley, A. Mousavian, C. Eppner, J. Leitner, J. Bohg, A. Morales, T. Asfour, D. Kragic et al., “Deep learning approaches to grasp synthesis: A review,” IEEE Transactions on Robotics, 2023.

[18] T. Eiband and D. Lee, “Identification of common force-based robot skills from the human and robot perspective,” in 2020 IEEE-RAS 20th International Conference on Humanoid Robots (Humanoids), 2021.

[19] S. Brahmbhatt, C. Tang, C. D. Twigg, C. C. Kemp, and J. Hays, “Contactpose: A dataset of grasps with object contact and hand pose,” in European Conference on Computer Vision, 2020.

[20] O. Taheri, N. Ghorbani, M. J. Black, and D. Tzionas, “Grab: A dataset of whole-body human grasping of objects,” in European conference on computer vision, 2020.

[21] Y. Liu, Y. Yang, Y. Wang, X. Wu, J. Wang, Y. Yao, S. Schwertfeger, S. Yang, W. Wang, J. Yu et al., “Realdex: towards human-like grasping for robotic dexterous hand,” in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, 2024.

[22] S. Brahmbhatt, C. Ham, C. C. Kemp, and J. Hays, “Contactdb: Analyzing and predicting grasp contact via thermal imaging,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019.

[23] R. Wang, J. Zhang, J. Chen, Y. Xu, P. Li, T. Liu, and H. Wang, “Dexgraspnet: A large-scale robotic dexterous grasp dataset for general objects based on simulation,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023.

[24] D. Turpin, T. Zhong, S. Zhang, G. Zhu, E. Heiden, M. Macklin, S. Tsogkas, S. J. Dickinson, and A. Garg, “Fast-grasp’d: Dexterous multi-finger grasp generation through differentiable simulation,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023.

[25] Y. Xu, W. Wan, J. Zhang, H. Liu, Z. Shan, H. Shen, R. Wang, H. Geng, Y. Weng, J. Chen et al., “Unidexgrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[26] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa et al., “Isaac gym: High performance gpu based physics simulation for robot learning,” in NeurIPS Datasets and Benchmarks, 2021.

[27] M. Liu, Z. Pan, K. Xu, K. Ganguly, and D. Manocha, “Deep differentiable grasp planner for high-dof grippers,” arXiv preprint arXiv:2002.01530, 2020.

[28] G.-H. Xu, Y.-L. Wei, D. Zheng, X.-M. Wu, and W.-S. Zheng, “Dexterous grasp transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[29] H. Jiang, S. Liu, J. Wang, and X. Wang, “Hand-object contact consistency reasoning for human grasps generation,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021.

[30] V. Mayer, Q. Feng, J. Deng, Y. Shi, Z. Chen, and A. Knoll, “Ffhnet: Generating multi-fingered robotic grasps for unknown objects in realtime,” in 2022 International Conference on Robotics and Automation (ICRA), 2022.

[31] S. Huang, Z. Wang, P. Li, B. Jia, T. Liu, Y. Zhu, W. Liang, and S.-C. Zhu, “Diffusion-based generation, optimization, and planning in 3d scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[32] J. Lu, H. Kang, H. Li, B. Liu, Y. Yang, Q. Huang, and G. Hua, “Ugg: Unified generative grasping,” in European Conference on Computer Vision, 2024.

[33] Y.-L. Wei, J.-J. Jiang, C. Xing, X.-T. Tan, X.-M. Wu, H. Li, M. Cutkosky, and W.-S. Zheng, “Grasp as you say: Languageguided dexterous grasp generation,” Advances in Neural Information Processing Systems, 2024.

[34] N. Khargonkar, L. F. Casas, B. Prabhakaran, and Y. Xiang, “Robotfingerprint: Unified gripper coordinate space for multi-gripper grasp synthesis and transfer,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025.

[35] Z. Xu, B. Qi, S. Agrawal, and S. Song, “Adagrasp: Learning an adaptive gripper-aware grasping policy,” in 2021 IEEE International Conference on Robotics and Automation (ICRA), 2021.

[36] J. Mérand, B. Meden, L. Chen, and M. Grossard, “Cotograsp: Contact-topology-conditioned dexterous grasp synthesis via canonical workspace learning,” in ECCV, 2026.

[37] J. Z. Zheng, S. De La Rosa, and A. M. Dollar, “An investigation of grasp type and frequency in daily household and machine shop tasks,” in 2011 IEEE international conference on robotics and automation, 2011.

[38] F. Gonzalez, F. Gosselin, and W. Bachta, “Analysis of hand contact areas and interaction capabilities during manipulation and exploration,” IEEE transactions on haptics, 2014.

[39] B. Calli, A. Singh, A. Walsman, S. Srinivasa, P. Abbeel, and A. M. Dollar, “The ycb object and model set: Towards common benchmarks for manipulation research,” in 2015 international conference on advanced robotics (ICAR), 2015.
# HiPHI: A Large-Scale Benchmark for High-Precision Human Motion and Object-Interaction

Jiahao Ji<sup>1,2,∗</sup>, Ji Ma<sup>1,5,∗</sup>, Runhan Zhang<sup>1,3,∗</sup>, Runyi Yu<sup>1,4</sup>, Wenjia Wang<sup>1,5</sup>, Weiheng Chi<sup>1</sup>, Qianqian Peng<sup>1</sup>, Weichao Yan<sup>1</sup>, Yongfei Gu<sup>1</sup>, Ye Tian<sup>1</sup>, Ting Wu<sup>1</sup>, Longwei Li<sup>1</sup>, Chun Yuan<sup>3</sup>, Ruoli Dai<sup>1,†</sup>, Lei Han<sup>1,†</sup>

<sup>1</sup>Noitom Robotics <sup>2</sup>National University of Singapore <sup>3</sup>SIGS, Tsinghua University <sup>4</sup>The Hong Kong University of Science and Technology <sup>5</sup>The University of Hong Kong

<sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding authors. [tristan@noitomrobotics.com], [lhan@noitomrobotics.com] Project page: https://noitom-robotics.github.io/hiphi

![](images/f83a57779c79dce356ad97dea918d70f98d45a76b9da05e0f8c445345e708f73.jpg)  
Figure 1: Overview of HiPHI. A large-scale, high-fidelity dataset for humanoid learning.

Abstract: Humanoid intelligence requires learning over an extremely diverse space of whole-body motions and physically grounded interactions. However, existing embodied datasets remain fundamentally limited: internet-scale video data lack precise physical states and interaction grounding, while laboratory motion datasets provide high fidelity but only narrow behavioral coverage. This mismatch creates a critical bottleneck for scalable humanoid policy learning. We present HiPHI, a 600+ hour scale high-fidelity whole-body human motion dataset designed to systematically maximize coverage of the human motion and interaction manifold. HiPHI is theoretically guided by FrameNet, a linguistic framework organizing human primitives. Created using an optical motion capture pipeline, HiPHI provides sub-millimeter spatial marker tracking accuracy for full-body human motion and mesh-level object trajectories. We further introduce a benchmark suite evaluating motion-space diversity, interaction grounding, object consistency, and physical AI applications. Our analyses demonstrate that HiPHI significantly expands motion coverage compared to existing motion datasets while maintaining high-fidelity interaction quality, and establishes a scalable data foundation for training, evaluating, and generalizing humanoid policies in real-world embodied tasks, where similar extensions are also applicable to motion prior models in computer graphics.

Keywords: Large-scale Motion Capture, Humanoid Robot Learning, Human-Object Interaction, Reinforcement Learning, Motion Tracking

## 1 Introduction

Humanoid learning requires reference motion that is both broad in coverage and physically faithful. Robot policies must acquire balance, posture transitions, limb coordination, contact, and loaddependent strategies. However, most existing data sources are not designed to capture high-precision human behavior for humanoid learning. Robot demonstrations and teleoperation data are limited by scale and cost, and are tied to specific embodiments [1, 2]. Human videos and egocentric data provide broad behavioral diversity at a large scale, but are limited to visual observations and therefore need to rely on reconstruction or proxy signals rather than directly measured physical states [3]. MoCap and HOI datasets provide accurate motion and sometimes object state, but are primarily designed for motion synthesis, language grounding, reconstruction, or interaction understanding [4, 5, 6, 7]. Humanoid policy learning, therefore, calls for data that are broad, precise, and grounded in objectconstrained whole-body control.

We formulate human data collection for humanoid learning as a problem of motion-space design. Rather than manually curating behavioral scripts, HiPHI uses FrameNet [8, 9] as a semantic scaffold, where frames represent event types and lexical units (LUs) identify the word senses that evoke them. We select frames and LUs most relevant to embodied intelligence, and use each LU under a specific frame as a seed for performer-facing capture scripts. We then expand each seed along multiple dimensions, including direction, speed, amplitude, body posture, body-part involvement, and object/contact conditions, producing a diverse set of repeatable motion instances for systematic capture. This turns handcrafted script authoring into a structured and scalable process for expanding motion-space coverage.

Using this pipeline, we present HiPHI, a 600-hour-scale, high-precision human motion and objectinteraction benchmark, all captured from 132 performers using a large optical motion capture (MoCap) system with sub-millimeter spatial tracking accuracy. Specifically, HiPHI combines 371.8 hours of diverse short-horizon whole-body motion and 245.7 hours of real-object interaction data, spanning 214 Frame–LU motion units across 22 frames. The interaction subset covers 40 distinct real-world objects from 12 categories. Each clip is indexed by a Frame–LU label and paired with a naturallanguage description. For human-object interaction sequences, synchronized object trajectories and meshes are captured together with human motion, making object state an integral part of the motion record. Capturing real objects is essential, as load, friction, inertia, resistance, and contact substantially shape control strategies in ways that pantomime motion cannot reflect.

We evaluate HiPHI in terms of motion-space coverage, data quality, interaction consistency, and humanoid policy learning performance. Motion coverage is measured using label-free kinematic embeddings; data quality is assessed through smoothness, ground contact, and interaction geometry; and downstream applications are tested through humanoid motion tracking and real-world deployment. Together, these analyses show that HiPHI is diverse, physically stable, interaction-grounded, and usable as executable reference data in humanoid learning, substantially surpassing existing datasets.

Our contributions are threefold: (i) a FrameNet-guided motion-space construction pipeline for systematic and scalable coverage of whole-body motion and interaction; (ii) a 617.5-hour, highprecision dataset including 245.7 hours of human–object interaction, synchronized object trajectories and meshes, Frame–LU indexing, and natural-language descriptions; and (iii) a comprehensive evaluation protocol for humanoid robot learning, including motion coverage, data quality, interaction consistency, motion tracking, and real-robot deployment.

## 2 Related Work

Robot Teleoperation and Egocentric Vision Data. Embodied intelligence increasingly demands large and diverse data sources. One major category is real-robot data, which avoids the embodiment gap, including teleoperated robot data [1, 10, 11], manipulation benchmarks [12, 13, 14, 15], and lowcost interfaces such as UMI [2]. These data sources avoid gaps in robot embodiment or end-effector configuration, but are often limited by collection cost, data scale, behavioral diversity, and dependence on specific robots, hardware, sensors, and viewpoints. Recently, egocentric visual data have become increasingly popular, as they provide life-scale observations of long-tail human behaviors in the real world [3, 16, 17, 18, 19]. However, such data are typically limited to visual modalities, making it difficult to recover precise human motion and physical interaction states.

![](images/c8edc8f2afadb6128fca99f8484319a422ac47eba0b39e86e2fe9a1a82a7ce85.jpg)  
Figure 2: From FrameNet to HiPHI data construction. Motion-relevant frames and lexical units define motion seeds, which are expanded through controlled factors such as speed, direction, amplitude, body-part involvement, and object/contact conditions.

Human Motion Capture Data. Motion capture data can accurately record human motion, with widely used datasets including AMASS [4], LAFAN1 [20], BABEL/KIT/HumanML3D [21, 22, 5], Motion-X/Motion-X++ [23, 24], and the recent BONES-SEED [25]. These datasets were originally designed mainly for motion synthesis, annotation, generation, or animation, but have recently become widely used in humanoid learning. However, most were not systematically designed for embodied intelligence: few simultaneously provide large scale, high-precision motion, systematic motion-space coverage, and synchronized object states. In contrast, HiPHI is a large-scale dataset specifically designed for embodied intelligence and humanoid learning. It uses the well-established FrameNet theory to structure data collection and directly evaluates motion-space coverage.

Human-Object Interaction. Similarly, many HOI datasets have been introduced for motion synthesis, animation, and interaction understanding [26, 6, 27, 28, 29, 30, 7, 31]. However, these datasets are typically small in scale, often only a few hours long, and their contact-level accuracy and interaction diversity remain limited. In contrast, our dataset contains 245.7 hours of HOI data and explicitly emphasizes the physical accuracy of contact-rich interactions. This design supports efficient humanoid learning based on motion imitation methods [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42].

## 3 FrameNet-Guided Motion-Space Construction

## 3.1 Motion Coverage

The most suitable data for humanoid robot learning is not necessarily the full distribution of motions that appear in everyday human life. For example, we do not expect robots to over-learn behaviors such as sleeping, eating, or watching television, which robots do not need to perform. Instead, the data most relevant to humanoid learning should cover control-centric motion patterns, including locomotion, posture transitions, limb coordination, dynamic balance, and responses to external constraints. The goal is therefore not to reproduce the full catalog of human activities, but to cover the motion space that matters for humanoid learning.

On the other hand, the human behaviors or tasks in existing motion datasets are often derived from manually designed scripts. These scripts usually describe semantically meaningful daily actions, which makes the resulting motion clips natural and interpretable. However, because the tasks are manually designed at the semantic level, they can lead to substantial overlap in motion space. For example, “raising a hand to wipe sweat from the forehead” and “raising a hand to block sunlight” are two different scripts at the task level, but may produce nearly identical motions. Counting them as two different tasks does not give us two genuinely different motions. The reverse is also true: one simple action such as walking can produce many different whole-body motions when its direction, speed, stride length, turning pattern, or posture changes. A growing list of manually designed scripts therefore provides no clear way to tell which parts of the motion space have already been covered and which are still missing. For humanoid learning, we instead aim to enable robots to span the full reachable human motion space, so that learned policies can better adapt to diverse downstream tasks. This requires a structured set of motion units that can be systematically enumerated and expanded, rather than an open-ended list of task descriptions.

Finally, existing datasets often contain many human-object interaction motions without capturing the corresponding object states, leading to, for example, “a person sitting in the air”. In this example, the body trajectory is recorded, but the chair that provides the support surface and contact relation is missing, leaving the physical interaction incomplete. Data for object-constrained humanoid learning should therefore capture the relevant object states together with the human motion. For instance, pushing, carrying, dragging, or leaning can substantially change posture, foot placement, and centerof-mass motion through the object’s geometry, friction, load, and trajectory. These factors are essential for learning physically grounded humanoid behaviors. HiPHI records these objects together with the human body, preserving the complete interaction as physically executable motion reference.

## 3.2 A FrameNet-Guided Collection Pipeline

To systematically address the problem of motion-space coverage, we use FrameNet as a semantic foundation for motion construction. FrameNet organizes word meanings according to the events they describe. A frame represents a type of event, while a lexical unit (LU) represents a word used in one particular meaning within that frame. For example, walk, jog, and run under the Self motion frame describe related forms of self-propelled movement. This distinction is important because the same word may describe completely different events: “run across the field” refers to body motion, whereas “run a company” does not. A Frame–LU pair identifies the first meaning precisely instead of treating every use of the term “run” as the same motion.

This follows the same organizing idea as ImageNet [43], where WordNet [44] provided a structured set of visual concepts to collect: FrameNet provides HiPHI with a structured set of motion meanings to collect. The goal is to avoid relying on manually designed task scripts and instead enumerate the motion space from a theoretically grounded and well-structured system. The action lexical units in FrameNet naturally serve this purpose, making them suitable seeds for generating a broad motion space. Figure 2 shows this complete pipeline from the FrameNet scaffold to the captured HiPHI motion space.

We select frames and lexical units that are most relevant to humanoid robots, including those related to body motion, posture change, directional movement, body-part motion, object actuation, and human-object interaction. Starting from these units, we further expand them along dimensions such as path, direction, speed, rhythm, amplitude, body posture, body-part involvement, support relation, and object/contact conditions. For example, a walking seed can be expanded with different routes, directions, speeds, stride lengths, turning patterns, and postures. A pushing seed can vary the object, load, contact point, pushing direction, and object trajectory. These variations change the actual human or object motion, rather than merely changing the story attached to it. These expanded scripts are then provided to motion-capture performers for execution. In this way, each Frame–LU pair maps to a family of capturable motions rather than a single clip.

Figure 3 makes the difference from manual script design explicit. In a conventional workflow, scripts are typically written case by case, guided mainly by the designers’ experience and intuition, without an explicit taxonomy or coverage criterion. Dataset growth therefore becomes an ad hoc, trial-anderror process: there is no principled way to determine which motions are still missing, whether a new script genuinely expands the motion space, or whether it merely wraps an already collected motion in a different story. HiPHI instead starts from Frame-LU motion seeds and expands them through a shared set of factors, such as intensity, route, speed, and object conditions. This replaces the blind accumulation of disconnected scripts with a systematic and traceable expansion of motion units and their physical variations. To our knowledge, HiPHI is the first MoCap dataset for robot learning to prospectively adopt such a linguistic scaffold for data construction.

![](images/2ceb2df0b0f757b5233e920be04c732e977f4d4a1dd4c634bdb08732cb6a2df1.jpg)  
Figure 3: Manual scripting versus FrameNet-guided construction. Manual collection grows by adding scripts one by one, whereas HiPHI organizes collection around Frame-LU motion seeds and expands them through shared factors. This turns dataset growth into a structured and scalable expansion of the motion space.

## 4 Dataset Composition and Statistics

Following this construction procedure, HiPHI is created as a LU-indexed motion dataset containing body-only and object-interaction sequences. This section summarizes its scale, relationship to existing datasets, Frame-LU composition, and object-interaction subset. The file organization and metadata are detailed in Appendix B.

## 4.1 Data Characteristics

HiPHI contains 617.5 hours of high-fidelity optical motion capture data in BVH format, corresponding to approximately 200.1 million frames. This is obtained by applying left-right mirroring to 308.7 hours of original motion captured with a high-precision optical MoCap system, following the convention used by BONES-SEED [25]. Human motion is captured at 90 Hz from 132 distinct performers, with detailed statistics in Appendix B. For object-interaction sequences, the data include synchronized human motion, object trajectories, and object meshes, making the object state an integral part of the motion record. Table 1 compares HiPHI with existing representative motion and human-object interaction datasets. Existing data often emphasize either large-scale body motion without object state or smaller-scale object-centric interaction. In contrast, HiPHI combines large-scale high-precision optical MoCap, Frame-LU indexing, synchronized object state, and a benchmark protocol designed for humanoid robot learning.

## 4.2 Frame-LU Composition and Long-Tail Coverage

Each clip in HiPHI is paired with a Frame-LU label and a natural-language description. The Frame-LU labels also serve as indices for retrieval, sampling, and analysis. The 214 Frame-LU labels across 22 FrameNet frames, placing each motion meaning within its event context rather than relying on ambiguous motion names. As shown in Figure 4, the dataset spans both frequent motion units, such as locomotion, posture changes, and body-part movements, and a long tail of more dynamic, irregular, and constrained patterns. The top 50 Frame-LUs account for 53.7% of the released duration, leaving 46.3% distributed across the remaining long-tail motion units. Motion scripts generated from each LU are performed by approximately 24 different actors on average, and 154 Frame-LUs are performed by at least 10 actors, providing substantial performer variation within individual motion units.

Table 1: Comparison with representative human motion and interaction datasets. HiPHI offers 617.5 hours of MoCap data, including 245.7 hours of object-state-aligned motion, that exceeds representative human motion and interaction datasets in scale and interaction coverage.
<table><tr><td>Dataset</td><td>Hours</td><td>Capture</td><td>Format</td><td>Motion index</td><td>Object motion</td></tr><tr><td>HiPHI</td><td>617.5</td><td>MoCap</td><td>BVH</td><td>Frame-LU</td><td>245.7 h</td></tr><tr><td>AMASS</td><td>&gt;40</td><td>MoCap</td><td>SMPL</td><td>一</td><td>一</td></tr><tr><td>BONES-SEED</td><td>288.3</td><td>MoCap</td><td>SOMA/G1</td><td>NL segments</td><td></td></tr><tr><td>Motion-X++ (orig.)†</td><td>40.4</td><td>Video</td><td>SMPL-X</td><td>text + pose</td><td>一</td></tr><tr><td>LAFAN1</td><td>4.6</td><td>MoCap</td><td>BVH</td><td>action themes</td><td></td></tr><tr><td>GRAB</td><td>3.8</td><td>MoCap</td><td>SMPL-X</td><td>intent labels</td><td>3.8 h</td></tr><tr><td>OMOMO</td><td>9.8</td><td>MoCap</td><td>SMPL-X</td><td></td><td>9.8 h</td></tr><tr><td>HIMO</td><td>9.4</td><td>MoCap</td><td>SMPL-X</td><td>text seg.</td><td>9.4 h</td></tr></table>

<sup>†</sup> Original Motion-X++ data only, where its integrated third-party sources are excluded. NL indicates Natural Language.

![](images/c60d64ea7a3586e90efe8934b3ce901a8a210446f355776d12af6ba8c980a435.jpg)  
Figure 4: Frame-LU composition and long-tail structure. The figure shows duration by FrameNet frame, the Frame-LU duration distribution, and cumulative duration share, illustrating that HiPHI combines common motion units with a broad long tail of whole-body patterns.

## 4.3 Body-Only and Object-Interaction Motions

HiPHI includes 371.8 hours of body-only motion data and 245.7 hours of human-object interaction motion data. The body-only subset covers self-motion, posture changes, body-part movement, dynamic motion, and coupled whole-body coordination. The object-interaction subset records motions shaped by real geometry, contact, load, and object motion, such as sitting, leaning, supporting, pushing, pulling, and carrying. It contains 40 objects across 12 categories, ranging from furniture and containers to cleaning tools and sports equipment, with masses from 0.45 to 6.25 kg. The subset spans 90 Frame-LUs across 15 FrameNet frames. Together, HiPHI provide a large-scale reference base for physically grounded interaction analysis and object-constrained humanoid learning.

## 5 Experiments

In this section, we create the HiPHI benchmark and evaluate HiPHI along two axes: dataset-level coverage and quality (Secs. 5.1, 5.2), and downstream utility through humanoid tracking in simulation and sim-to-real deployment (Secs. 5.3, 5.4).

## 5.1 Motion-Space Diversity

We use a dataset-balanced protocol to compare kinematic motion-space coverage. All datasets are first mapped to a unified 23-keypoint, 30 FPS, root-aligned, and body-scale-normalized representation, and each sequence is divided into one-second windows. For shared-encoder training, we randomly sample 5,000 clips from each dataset and draw one window from each sampled clip. One exception is LAFAN1 that its full dataset contains fewer clips than 5,000, and thus we use its all available clips for comparison. We then train a shared temporal-convolutional autoencoder using only a reconstruction objective and project its 16-D latent codes into a common t-SNE space. The visualization and occupancy analysis likewise use at most 5,000 windows per dataset (see Appendix E). Figure 5 shows that HiPHI spans the broadest kinematic region among the compared datasets, covering most regions occupied by the other datasets while extending into additional parts of the motion space. We further discretize the embedding and report three statistics (defined in Appendix E): occupied cells (how wide the coverage is), effective occupancy (how uniform the coverage is across occupied cells), and long-tail share (the fraction of samples in globally rare cells). HiPHI covers more grid cells than the closest baseline BONES-SEED (1620 vs. 1438), with larger effective occupancy (1443 vs. 1114) and higher long-tail share (14.1% vs. 10.7%), indicating broader and more uniform coverage rather than denser sampling of common motions. In other words, HiPHI not only reaches more regions of the kinematic space, but also distributes its motions more broadly across those regions, under the same randomly sampled size. We further repeat the coverage analysis across multiple random seeds, t-SNE perplexities, and grid resolutions, and refer the detailed results to Appendix E. HiPHI retains a positive occupied-cell margin over the strongest baseline in every tested setting, confirming that its coverage advantage is consistent across projection and discretization choices.

![](images/ff748fcc6b25e27b5416a7b897f9e2860ed2bc3fb813985c741e094bb6e01cba.jpg)  
Figure 5: Full kinematic motion-space visualization. All datasets are encoded by the same unsupervised body-motion encoder and projected into one shared t-SNE space. (a) Global projection with balanced sampled points per dataset. (b) Support-envelope comparison between HiPHI and BONES-SEED, the closest large-scale baseline; HiPHI’s support nearly encloses BONES-SEED. (c) Grid-based local-coverage statistics on the same embedding using a 55×55 grid.

## 5.2 Data Quality

Body-motion precision. We measure five quantities (lower is better; aggregation defined in Appendix F) that capture common failure modes in humanoid imitation: jerk $\mathcal { I } _ { \tau } ~ ( \mathrm { m / s ^ { 3 } } )$ , acceleration $\bar { \mathcal { A } } \bar { ( } \mathrm { m } / \mathrm { s } ^ { 2 } )$ , ground penetration $\delta _ { \mathrm { g r o u n d } }$ (mm), unsupported-floating share $\phi _ { \mathrm { f l o a t } } ( \% )$ , and support-point drift $\nu _ { \mathrm { f o o t } }$ (mm/s). $\mathbb { Q } _ { \tau }$ and E denote the upper-tail quantile and mean $\scriptstyle ( \tau = 0 . 9 5 ) ; \nabla _ { t }$ is a finitedifference operator divided by the source-specific frame interval $\Delta t ,$ so that $\nabla _ { t } ^ { k }$ x has units of $\mathrm { m } / \mathrm { s } ^ { k } ;$ ${ \bar { x } } _ { j }$ and $x _ { j }$ are smoothed and raw joint positions; C is the core-body joint set; $\mathscr { F }$ is the support-point set; $\mathcal { K } = \mathbf { \bar { \Phi } } \{ ( t , f ) : h _ { f } ( t ) - g < \epsilon \}$ is the contact set of (frame, support-point) pairs within $\epsilon = 3 0$ mm of the ground at height g; $h _ { f }$ and $p _ { f }$ are the height and position of support point $f ; u ( t )$ is the unsupported-floating indicator; and $\bar { \Pi } _ { x y } ( \cdot )$ is an operator that projects the input onto the ground plane. Formally, we have

$$
\begin{array} { r l } & { \mathcal { T } _ { \tau } = \mathbb { Q } _ { \tau } \big \{ \big \| \nabla _ { t } ^ { 3 } \bar { x } _ { j } ( t ) \big \| _ { 2 } : j \in \mathcal { C } \big \} , \ A = \mathbb { Q } _ { 1 / 2 } \big \{ \big \| \nabla _ { t } ^ { 2 } x _ { j } ( t ) \big \| _ { 2 } : j \in \mathcal { C } \big \} , \ \phi _ { \mathrm { f l o a t } } = \mathbb { E } [ u ( t ) ] , } \\ & { \delta _ { \mathrm { g r o u n d } } = \mathbb { Q } _ { \tau } \big \{ \operatorname* { m a x } \bigl ( g - h _ { f } ( t ) , 0 \bigr ) : f \in \mathcal { F } \big \} , \quad \nu _ { \mathrm { f o o t } } = \mathbb { E } \big [ \| \Pi _ { x y } \left( \nabla _ { t } p _ { f } ( t ) \right) \| _ { 2 } : ( t , f ) \in \mathcal { K } \big ] . } \end{array}\tag{1}
$$

Floor-related metrics $( \delta _ { \mathrm { g r o u n d } } , \phi _ { \mathrm { f l o a t } } , \nu _ { \mathrm { f o o t } } )$ require a comparable absolute ground convention. Motion-X++ does not provide one and is therefore excluded from these three metrics in Table 2(a).

Among datasets sharing the same floor-plane convention, HiPHI achieves the best value on every reported body-motion metric in Table $2 ( \mathrm { a } )$ . These metrics correspond to important physical failure modes in humanoid imitation: $\mathcal { I } _ { \tau }$ measures abrupt motion changes, $\mathcal { A }$ captures excessive acceleration, $\delta _ { \mathrm { g r o u n d } }$ indicates ground penetration, $\phi _ { \mathrm { f l o a t } }$ reflects unsupported floating, and $\nu _ { \mathrm { f o o t } }$ measures foot sliding during contact.

Table 2: Data quality comparison. (a) Body-motion smoothness and ground-contact metrics, lower is better: jerk ${ \mathcal { I } } _ { \tau } ,$ acceleration A, below-ground depth $\delta _ { \mathrm { g r o u n d } }$ , unsupported-floating share ϕ<sub>float</sub>, and support-point drift $\nu _ { \mathrm { f o o t } }$ . (b) Human-object geometric consistency, higher is better: non-conflict fraction $\eta _ { \mathrm { n c } }$ and near-surface grounding within 20 cm $\rho _ { \mathrm { n e a r } }$  
(a) Body motion precision
<table><tr><td>Dataset</td><td>Duration (h)</td><td> $\boldsymbol { \mathcal { T } } _ { \boldsymbol { \tau } }$   $\mathrm { ( m / s ^ { 3 } ) }$ </td><td> $\pmb { A }$   $\mathrm { ( m / s ^ { 2 } ) }$ </td><td> $\delta _ { \mathrm { g r o u n d } }$  (mm)</td><td> $\phi _ { \mathrm { f l o a t } }$   $( \% )$ </td><td> $\scriptstyle { \pmb { \nu } } _ { \mathrm { f o o t } }$  (mm/s)</td></tr><tr><td>HiPHI</td><td>617.5</td><td>173.9</td><td>10.7</td><td>8</td><td>0.015</td><td>64</td></tr><tr><td>AMASS</td><td>62.9</td><td>529.0</td><td>14.9</td><td>111</td><td>0.76</td><td>87</td></tr><tr><td>BONES-SEED</td><td>288.3</td><td>294.2</td><td>11.3</td><td>18</td><td>1.02</td><td>86</td></tr><tr><td>Motion-X++†</td><td>26.0</td><td>709.0</td><td>23.8</td><td>一</td><td>一</td><td></td></tr><tr><td>LAFAN1</td><td>4.6</td><td>383.1</td><td>23.4</td><td>29</td><td>7.22</td><td>157</td></tr></table>

(b) Object interaction geometry
<table><tr><td>Dataset</td><td>Duration (h)</td><td>ηnc (%)</td><td>ρnear (%)</td></tr><tr><td>HiPHI</td><td>245.7</td><td>98.1</td><td>95.7</td></tr><tr><td>HIMO*</td><td>21.6</td><td>97.6</td><td>79.0</td></tr><tr><td>OMOMO</td><td>9.8</td><td>90.6</td><td>50.4</td></tr><tr><td>HUMOTO‡</td><td>0.81</td><td>99.9</td><td>82.9</td></tr></table>

All metrics are aggregated over the full evaluated duration. <sup>∗</sup>HIMO contains 9.4 h of unique multi-object sequences. Since each object in a multi-object sequence is evaluated separately, the reported duration becomes 21.6 object-track hours. <sup>†</sup>Motion-X++ floor metrics require a comparable absolute floor convention and are therefore omitted. <sup>‡</sup>HUMOTO is evaluated on its public subset; the full licensed dataset is not directly downloadable as public data.

Object-interaction consistency. Two quantities (higher is better) measure whether human motion and object geometry remain physically consistent in a shared frame: non-conflict fraction $\eta _ { \mathrm { n c } }$ (%) and near-surface grounding $\rho _ { \mathrm { n e a r } }$ (%). Let ${ \mathbf { } } S ( t )$ be a set of points sampled along the human-skeleton segment centerlines (uniformly along each segment), Ω(t) the posed object as a closed 3D region, and d(S, ∂Ω) the minimum point-to-surface distance from $s$ to the object boundary ∂Ω. Below, 1[·] is the indicator function (1 if the condition holds, 0 otherwise):

$$
\eta _ { \mathrm { n c } } = \mathbb { E } \bigl [ \mathbf { 1 } \bigl [ S ( t ) \cap \Omega ( t ) = \emptyset \bigr ] \bigr ] , \qquad \rho _ { \mathrm { n e a r } } = \mathbb { E } \bigl [ \mathbf { 1 } \bigl [ d ( S ( t ) , \partial \Omega ( t ) ) < 0 . 2 0 \mathrm { m } \bigr ] \bigr ] .\tag{2}
$$

Table 2(b) reports $\eta _ { \mathrm { n c } }$ and $\rho _ { \mathrm { n e a r } }$ . HiPHI provides substantially longer object-interaction duration than existing full-body HOI datasets (245.7 h vs. 21.6 h for HIMO and 9.8 h for OMOMO) while maintaining strong geometric consistency. In particular, HiPHI achieves 98.1% non-conflict rate and 95.7% near-surface grounding, substantially improving interaction grounding over HIMO (79.0%) and OMOMO (50.4%).

## 5.3 Humanoid Learning

We evaluate the physical executability of HiPHI through physics-based humanoid tracking. For whole-body motion, we compare HiPHI with AMASS [4], LAFAN1 [20], Motion-X++ [24], and BONES-SEED [25]. All motions are retargeted to the Unitree G1 robot and trained with the same DeepMimic imitation learning pipeline [32]. For motion-with-object tracking, we compare with HUMOTO [7] and OMOMO [29], using Omni-Retarget [41] and BeyondMimic-style [39] to handle synchronized human-object demonstrations. This unified setup allows performance differences to mainly reflect the usability of the motion data. All experiments are conducted from unmirrored original motions.

## 5.3.1 Whole-Body Motion Tracking

Whole-body tracking tests how well each motion source can be physically imitated without the additional complexity of object interaction. Due to the limited usable duration of LAFAN1, we construct a balanced 3-hour subset for each data source and additionally evaluate a 20-hour setting for larger datasets, to ensure a fair comparison independent of HiPHI’s data-scale advantage. For each setting, subsets are sampled using the same random sampling procedure, and all policies are trained under the same optimization budget, with additional details provided in Appendix G. A rollout is considered successful only if its mean full-body position error remains below 0.5 m at every simulation step; otherwise, it is marked as failed.

As shown in Figure 6, HiPHI achieves the highest success rates and fastest convergence in both the 3-hour and 20-hour settings. The advantage remains consistent across five independent training runs, with HiPHI maintaining lower failure rates and smaller variance than the compared datasets. These results show that HiPHI provides motion references that are not only larger in scale, but also more directly executable for humanoid policy learning.

(b)  
![](images/968af11aeffe169a0df44001f3b46bf0972e1b5454b33b5d1675d1964fe63764.jpg)

![](images/21583eded1acdff2b8f927acd4158a7f9979f1f5396bc6c783fcc690069649cc.jpg)

Figure 6: Whole-body humanoid tracking performance across different motion data sources. (a) Success rates under matched 3-hour and 20-hour data budgets. (b) Training failure-rate curves averaged over five independent runs, with shaded regions indicating variance. HiPHI achieves faster convergence and lower final failure rates across different data scales.  
![](images/58978e33793b4048e4b27cdb21b95399a244a3d84041feddaa043323d2f1f03c.jpg)  
Figure 7: Scaling behavior of HiPHI for humanoid tracking. Increasing the amount of unmirrored HiPHI training data from 3 to 300 hours consistently reduces cross-dataset MPJPE on AMASS, BONES-SEED, Motion-X++, and LAFAN1, demonstrating continuous performance gains with increasing data scale.

## 5.3.2 Data Scaling

Beyond the matched-budget comparison, we further investigate how the scale of HiPHI affects downstream humanoid tracking. We progressively increase the amount of unmirrored HiPHI training data from 3 to 300 hours and evaluate the resulting policies on AMASS, BONES-SEED, Motion-X++, and LAFAN1. Consistent with the experiments above, we randomly sample 20 hours from each dataset as the evaluation set, except for LAFAN1, for which 3-hour setting is used. All experiments are repeated 10 times, and we report the mean curves with variance bands. As shown in Figure 7, increasing the amount of training data consistently reduces cross-dataset MPJPE, demonstrating that the large scale of HiPHI translates into continuous downstream performance gains.

## 5.3.3 Whole-Body Motion Tracking with Object

We further evaluate human-object interaction tracking on four categories: kick, carry, push, and lean. All datasets are processed using the same retargeting and physics-based tracking pipeline, allowing us to evaluate how effectively each dataset supports physically grounded human-object imitation. Some categories are unavailable in existing datasets: HUMOTO does not contain push motions, and OMOMO does not provide lean motions. These cases are marked as “N/A” in Table 3.

We report both body-level and object-level tracking errors. Mean per-joint position error (MPJPE) and velocity error (Vel.) measure humanoid pose and motion tracking quality, while object position error (Obj-Pos.) and object orientation error (Obj-Ori.) measure object trajectory and rotation accuracy. As shown in Table 3, HiPHI achieves the best overall body tracking performance across most categories. For object tracking, HiPHI obtains the best position and orientation accuracy on kick and push, and the best orientation accuracy on carry, indicating better interaction consistency between the humanoid and manipulated objects. Additional experimental details are provided in Appendix H.

Table 3: Whole-body motion tracking with object comparison (a) Body tracking errors are measured by MPJPE in mm and Vel. in mm/frame. (b) Object tracking errors are measured by Obj-Pos. in mm and Obj-Ori. in degrees.  
(a) Body tracking errors
<table><tr><td>Dataset</td><td colspan="2">Kick</td><td colspan="2">Carry</td><td colspan="2">Push</td><td colspan="2">Lean</td></tr><tr><td></td><td>MPJPE↓</td><td>Vel. ↓ |</td><td>MPJPE↓</td><td>Vel. ↓ |</td><td>MPJPE↓</td><td>Vel. ↓ |</td><td>MPJPE↓</td><td>Vel. ↓</td></tr><tr><td>HUMOTO [7]</td><td>47.77</td><td>4.75</td><td>55.37</td><td>11.67</td><td>N/A</td><td>N/A</td><td>38.83</td><td>4.67</td></tr><tr><td>OMOMO [29]</td><td>85.31</td><td>7.05</td><td>75.57</td><td>6.83</td><td>66.09</td><td>11.28</td><td>N/A</td><td>N/A</td></tr><tr><td>HiPHI (ours)</td><td>26.31</td><td>3.68</td><td>50.33</td><td>4.14</td><td>99.20</td><td>6.26</td><td>30.76</td><td>2.12</td></tr></table>

(b) Object tracking errors
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Kick</td><td colspan="2">Carry</td><td colspan="2">Push</td><td colspan="2">Lean</td></tr><tr><td>Obj-Pos. ↓</td><td>Obj-Ori. ↓ |</td><td>Obj-Pos. ↓</td><td>Obj-Ori. ↓ |</td><td>Obj-Pos. ↓</td><td>Obj-Ori. ↓ |</td><td>Obj-Pos. ↓</td><td>Obj-Ori. ↓</td></tr><tr><td>HUMOTO [7]</td><td>214.91</td><td>35.92</td><td>59.20</td><td>5.39</td><td>N/A</td><td>N/A</td><td>136.71</td><td>4.81</td></tr><tr><td>OMOMO [29]</td><td>128.62</td><td>104.28</td><td>62.46</td><td>4.87</td><td>257.82</td><td>46.24</td><td>N/A</td><td>N/A</td></tr><tr><td>HiPHI (ours)</td><td>68.17</td><td>3.27</td><td>89.13</td><td>2.64</td><td>60.51</td><td>2.23</td><td>69.16</td><td>9.63</td></tr></table>

## 5.4 Real-World Evaluation of HiPHI

To evaluate the practical usability of HiPHI beyond simulation, we deploy policies trained with HiPHI on Unitree G1. Despite actuation limits, sensing noise, and sim-to-real discrepancies, the robot successfully performs diverse behaviors, including locomotion, sitting, crawling, carrying, flipping, and pulling motions. As shown in Figure 8, these results demonstrate that HiPHI supports stable and physically plausible whole-body and interaction-rich control on real hardware.

![](images/3b3aaf4f0265841f31ba887f8f1eae0cce7e010dc74529d312bdb23dce3170d0.jpg)

![](images/82cb3060046065857de327f4eee52fd2aabb1b10c97766f60349bbb1bace3baf.jpg)  
Figure 8: Real-world deployment of policies trained with HiPHI. The humanoid successfully performs diverse whole-body and object-interaction behaviors, including (a) running, (b) sitting, (c) crawling, (d) carrying a box, (e) flipping, and (f) pulling a suitcase, showing the physical executability of our motion data.

## 6 Conclusion

We introduced HiPHI, a 617.5-hour high-precision optical MoCap dataset and benchmark for humanoid learning, including 245.7 hours of synchronized human-object interaction with object trajectories and meshes. Built upon a Frame-LU-guided motion-space construction pipeline, HiPHI provides a systematic and scalable approach for collecting broad whole-body motion and physically grounded interactions.

Through comprehensive evaluations, HiPHI demonstrates broader motion-space coverage, higher motion quality, stronger human-object geometric consistency, and improved downstream humanoid tracking performance. Policies trained with HiPHI achieve strong matched-budget tracking performance, continue to improve with increasing data scale, and successfully transfer to real humanoid hardware across diverse whole-body and object-interaction behaviors.

Together, these results establish HiPHI as a large-scale data foundation for learning physically grounded humanoid skills, bridging high-precision motion capture, object-aware interaction, and scalable policy learning.

## 7 Limitations

HiPHI currently captures only single-person motion. Multi-person interaction and human-human contact remain outside its scope and require complementary efforts. In addition, although HiPHI provides high-precision full-body motion and object trajectories, it focuses on kinematics rather than directly measuring contact forces or tactile signals. Finally, the dataset is collected in studio, and capturing accurate motion and object interaction with ego-vision in-the-wild could be a promising future data collection direction.

## Acknowledgments

We sincerely thank Roch Nakajima for generously sharing his extensive experience in motion capture and for his invaluable support in interpreting FrameNet lexical units and assessing their suitability for motion capture. We thank Alex Martinez for developing a practical data visualization tool that greatly facilitated data inspection and quality control. We also thank Baoze Du for his valuable advice on standardizing the BVH skeleton specification. Finally, we thank the many other staff members at Noitom Robotics who operated and maintained the motion-capture facilities and equipment, coordinated data-collection logistics and infrastructure, and supported the performers throughout the capture process. We are also deeply grateful to all performers who contributed their time and effort to the creation of HiPHI.

## References

[1] A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

[2] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. arXiv preprint arXiv:2402.10329, 2024.

[3] K. Grauman, A. Westbury, E. Byrne, Z. Chavis, A. Furnari, R. Girdhar, J. Hamburger, H. Jiang, M. Liu, X. Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022.

[4] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. J. Black. Amass: Archive of motion capture as surface shapes. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5442–5451, 2019.

[5] C. Guo, S. Zou, X. Zuo, S. Wang, W. Ji, X. Li, and L. Cheng. Generating diverse and natural 3d human motions from text. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5152–5161, 2022.

[6] B. L. Bhatnagar, X. Xie, I. A. Petrov, C. Sminchisescu, C. Theobalt, and G. Pons-Moll. Behave: Dataset and method for tracking human object interactions. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 15935–15946, 2022.

[7] J. Lu, C.-H. P. Huang, U. Bhattacharya, Q. Huang, and Y. Zhou. Humoto: A 4d dataset of mocap human object interactions. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 10886–10897, 2025.

[8] C. F. Baker, C. J. Fillmore, and J. B. Lowe. The berkeley framenet project. In Proceedings ofthe 36th Annual Meeting ofthe Associationfor Computational Linguistics and 17th International Conference on Computational Linguistics - Volume 1, ACL ’98/COLING ’98, page 86–90, USA, 1998. Association for Computational Linguistics. doi:10.3115/980845.980860. URL https://doi.org/10.3115/980845.980860.

[9] J. Ruppenhofer, M. Ellsworth, M. Schwarzer-Petruck, C. R. Johnson, and J. Scheffczyk. Framenet ii: Extended theory and practice. Technical report, International Computer Science Institute, 2016.

[10] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

[11] H. R. Walke, K. Black, T. Z. Zhao, Q. Vuong, C. Zheng, P. Hansen-Estruch, A. W. He, V. Myers, M. J. Kim, M. Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pages 1723–1736. PMLR, 2023.

[12] S. James, Z. Ma, D. R. Arrojo, and A. J. Davison. Rlbench: The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters, 5(2):3019–3026, 2020.

[13] C. Li, R. Zhang, J. Wong, C. Gokmen, S. Srivastava, R. Mart´ın-Mart´ın, C. Wang, G. Levine, M. Lingelbach, J. Sun, et al. Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation. In Conference on Robot Learning, pages 80–93. PMLR, 2023.

[14] O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard. Calvin: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334, 2022.

[15] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36: 44776–44791, 2023.

[16] S. Kareer, D. Patel, R. Punamiya, P. Mathur, S. Cheng, C. Wang, J. Hoffman, and D. Xu. Egomimic: Scaling imitation learning via egocentric video. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 13226–13233. IEEE, 2025.

[17] R. Hoque, P. Huang, D. J. Yoon, M. Sivapurapu, and J. Zhang. Egodex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025.

[18] G. Li, Y. Lyu, Z. Liu, C. Hou, J. Zhang, and S. Zhang. H2r: A human-to-robot data augmentation for robot pre-training from videos. arXiv preprint arXiv:2505.11920, 2025.

[19] L. Y. Zhu, P. Kuppili, R. Punamiya, P. Aphiwetsa, D. Patel, S. Kareer, S. Ha, and D. Xu. Emma: Scaling mobile manipulation via egocentric human data. IEEE Robotics and Automation Letters, 2026.

[20] F. G. Harvey, M. Yurick, D. Nowrouzezahrai, and C. Pal. Robust motion in-betweening. ACM Transactions on Graphics (TOG), 39(4):60–1, 2020.

[21] A. R. Punnakkal, A. Chandrasekaran, N. Athanasiou, A. Quiros-Ramirez, and M. J. Black. Babel: Bodies, action and behavior with english labels. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 722–731, 2021.

[22] M. Plappert, C. Mandery, and T. Asfour. The kit motion-language dataset. Big data, 4(4): 236–252, 2016.

[23] J. Lin, A. Zeng, S. Lu, Y. Cai, R. Zhang, H. Wang, and L. Zhang. Motion-x: A large-scale 3d expressive whole-body human motion dataset. Advances in Neural Information Processing Systems, 36:25268–25280, 2023.

[24] Y. Zhang, J. Lin, A. Zeng, G. Wu, S. Lu, Y. Fu, Y. Cai, R. Zhang, H. Wang, and L. Zhang. Motion-x++: A large-scale multimodal 3d whole-body human motion dataset. arXiv preprint arXiv:2501.05098, 2025.

[25] Bones Studio. BONES-SEED: Skeletal Everyday Embodiment Dataset. Hugging Face dataset, 2026. URL https://huggingface.co/datasets/bones-studio/seed.

[26] O. Taheri, N. Ghorbani, M. J. Black, and D. Tzionas. Grab: A dataset of whole-body human grasping of objects. In European conference on computer vision, pages 581–600. Springer, 2020.

[27] Y. Huang, O. Taheri, M. J. Black, and D. Tzionas. Intercap: joint markerless 3d tracking of humans and objects in interaction from multi-view rgb-d images. International Journal of Computer Vision, 132(7):2551–2566, 2024.

[28] Y. Liu, Y. Liu, C. Jiang, K. Lyu, W. Wan, H. Shen, B. Liang, Z. Fu, H. Wang, and L. Yi. Hoi4d: A 4d egocentric dataset for category-level human-object interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21013–21022, 2022.

[29] J. Li, J. Wu, and C. K. Liu. Object motion guided human motion synthesis. ACM Transactions on Graphics (TOG), 42(6):1–11, 2023.

[30] X. Lv, L. Xu, Y. Yan, X. Jin, C. Xu, S. Wu, Y. Liu, L. Li, M. Bi, W. Zeng, et al. Himo: A new benchmark for full-body human interacting with multiple objects. In European Conference on Computer Vision, pages 300–318. Springer, 2024.

[31] S. Xu, D. Li, Y. Zhang, X. Xu, Q. Long, Z. Wang, Y. Lu, S. Dong, H. Jiang, A. Gupta, et al. Interact: Advancing large-scale versatile 3d human-object interaction generation. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 7048–7060, 2025.

[32] X. B. Peng, P. Abbeel, S. Levine, and M. Van de Panne. Deepmimic: Example-guided deep reinforcement learning of physics-based character skills. ACM Transactions On Graphics (TOG), 37(4):1–14, 2018.

[33] X. B. Peng, Z. Ma, P. Abbeel, S. Levine, and A. Kanazawa. Amp: Adversarial motion priors for stylized physics-based character control. ACM Transactions on Graphics (ToG), 40(4):1–20, 2021.

[34] X. B. Peng, Y. Guo, L. Halper, S. Levine, and S. Fidler. Ase: Large-scale reusable adversarial skill embeddings for physically simulated characters. ACM Transactions On Graphics (TOG), 41(4):1–17, 2022.

[35] N. Wagener, A. Kolobov, F. Vieira Frujeri, R. Loynd, C.-A. Cheng, and M. Hausknecht. Mocapact: A multi-task dataset for simulated humanoid control. Advances in Neural Information Processing Systems, 35:35418–35431, 2022.

[36] Z. Luo, J. Cao, K. Kitani, W. Xu, et al. Perpetual humanoid control for real-time simulated avatars. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10895–10904, 2023.

[37] Q. Zhu, H. Zhang, M. Lan, and L. Han. Neural categorical priors for physics-based character control. ACM Transactions on Graphics (TOG), 2023.

[38] L. Han, Q. Zhu, J. Sheng, C. Zhang, T. Li, Y. Zhang, H. Zhang, Y. Liu, C. Zhou, R. Zhao, et al. Lifelike agility and play in quadrupedal robots using reinforcement learning and generative pre-trained models. Nature Machine Intelligence, 6(7):787–798, 2024.

[39] Q. Liao, T. E. Truong, X. Huang, Y. Gao, G. Tevet, K. Sreenath, and C. K. Liu. Beyondmimic: From motion tracking to versatile humanoid control via guided diffusion. arXiv preprint arXiv:2508.08241, 2025.

[40] Z. Luo, Y. Yuan, T. Wang, C. Li, S. Chen, F. Castaneda, Z.-A. Cao, J. Li, D. Minor, Q. Ben, et al. Sonic: Supersizing motion tracking for natural humanoid whole-body control. arXiv preprint arXiv:2511.07820, 2025.

[41] L. Yang, X. Huang, Z. Wu, A. Kanazawa, P. Abbeel, C. Sferrazza, C. K. Liu, R. Duan, and G. Shi. Omniretarget: Interaction-preserving data generation for humanoid whole-body locomanipulation and scene interaction. arXiv preprint arXiv:2509.26633, 2025.

[42] C. Tessler\*, Y. Jiang\*, X. B. Peng, E. Coumans, Y. Shi, H. Zhang, D. Rempe, G. Chechik†, and S. Fidler†. Protomotions3: An open-source framework for humanoid simulation and control. https://github.com/NVLabs/ProtoMotions/, 2025.

[43] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[44] G. A. Miller. WordNet: A lexical database for English. In Human Language Technology: Proceedings of a Workshop held at Plainsboro, New Jersey, March 8-11, 1994, 1994. URL https://aclanthology.org/H94-1111/.

[45] C. M. Kim, B. Yi, H. Choi, Y. Ma, K. Goldberg, and A. Kanazawa. Pyroki: A modular toolkit for robot kinematic optimization. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1312–1319. IEEE, 2025.

## Appendix

The appendix provides additional details that complement the main text. Appendix A-B describe FrameNet-guided motion-unit construction and the release composition, metadata, and demographics. Appendix C states the release license and ethics. Appendix D shows representative sample sequences. Appendix E-F provide the unsupervised motion-space embedding setup and the full-run quality metrics. Appendix G-H document training details for whole-body and motion-with-object tracking.

## A Additional Details for Dataset Construction

FrameNet-guided motion units. HiPHI uses FrameNet as a semantic scaffold rather than a closed action taxonomy. We identify motion-relevant frames and lexical units (LUs), and use each Frame-LU pair as a seed for a motion unit that can be translated into performer-facing capture instructions. This keeps the public index compact while allowing each semantic seed to expand into multiple concrete realizations.

Table 4: Examples of FrameNet-guided motion-unit construction. Frame-LU seeds are expanded by controlled realization factors rather than treated as single fixed actions.
<table><tr><td>Motion family</td><td>Frame-LU examples</td><td>Realization factors</td></tr><tr><td>tion</td><td>stride</td><td>Locomotion and direc- Self_motion: walk, run, jog, path, direction, speed, turning radius, step rhythm</td></tr><tr><td>Posture and support</td><td>Change -posture, crawl, kneel, lean</td><td>Posture: body height, support relation, amplitude, transi- tion speed</td></tr><tr><td>Body-part motion</td><td>shake, clap</td><td>Body_movement: bend, toss, body part, range of motion, rhythm, unilat- eral/bilateral pattern</td></tr><tr><td>Object actuation and Cause _motion, transfer</td><td>push, pull, lift, carry</td><td>Bringing: object category, contact mode, load condition, object trajectory</td></tr><tr><td>Object-local dynamics</td><td>tate, shake, swing</td><td>Cause_to_move_in_place: ro- axis of motion, contact point, amplitude, repeti- tion pattern</td></tr></table>

Performer-facing realization protocol. Each selected Frame-LU seed is converted into short, repeatable capture instructions by varying factors that are directly observable during MoCap: direction, path, speed, intensity, amplitude, body height, body-part involvement, object category, contact mode, and load condition. For instance, a push seed is realized with different object categories, loads, and directions; a lean seed varies support surface and body height; and a crawl seed varies path shape, speed, and limb coordination. This protocol turns a compact semantic index into a broader set of controllable whole-body motion realizations.

## B Additional Details for Dataset Composition and Metadata

Release duration convention. All release-scale durations in the paper follow the public mirrored release convention, where left-right mirrored sequences are included as normal released data. Under this convention, HiPHI contains 617.5 hours of motion, corresponding to approximately 200.1M frames at 90 Hz. The release contains 371.8 hours of body-only motion and 245.7 hours of strict object-interaction motion, where strict object-interaction denotes explicit physical interaction with a tracked object.

Frame-LU composition. The released semantic layer contains 22 FrameNet frames and 214 Frame-LU motion-unit labels. Its duration distribution follows a common-to-long-tail structure: the top 10 Frame-LUs account for 20.4% of duration, the top 20 for 31.8%, and the top 50 for 53.7%. Thus, nearly half of the release duration remains outside the top 50 Frame-LUs, supporting retrieval and analysis beyond a small set of frequent actions.

Table 5: Release-level composition used throughout the paper. Durations follow the mirrored-release convention.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Total release duration</td><td>617.5 h</td></tr><tr><td>Total frames</td><td>200.1M</td></tr><tr><td>Capture frame rate</td><td>90 Hz</td></tr><tr><td>Total performers</td><td>132</td></tr><tr><td>Body-only duration</td><td>371.8 h</td></tr><tr><td>Human object-interaction duration Human object-interaction share</td><td>245.7 h</td></tr><tr><td>FrameNet frames</td><td>39.8%</td></tr><tr><td>Frame-LU labels</td><td>22</td></tr><tr><td></td><td>214</td></tr><tr><td>Median actors per Frame-LU</td><td>24</td></tr><tr><td>Frame-LUs with at least 10 actors</td><td>154</td></tr><tr><td>Object categories in strict object-interaction subset</td><td>12</td></tr></table>

(a) Gender  
![](images/a9eb7a08eff56e04433daf926629f43ab9e0e23fe077b40720ece8671ab60e35.jpg)

(b) Height  
![](images/ff10858e69ca59348ebd36efbb2cdfac6446746677d352745f51383eb7ba3123.jpg)

(c) Weight  
![](images/b0f5442219b3a1d21518919de5601b0a77d26ee899f464bf2f9bd6ad361d33c5.jpg)  
Figure 9: Performer demographics of HiPHI. (a) Gender split among 132 performers. (b) Height distribution. (c) Weight distribution.

Performer demographics. HiPHI contains 132 anonymized performer profiles, including 76 male and 56 female profiles, with heights from 155 to 185 cm and weights from 40 to 82 kg.Figure 9 summarizes the distribution.

Object-interaction metadata. The HiPHI object-interaction subset contains 40 real-world objects from 12 categories, covering a mass range of 0.45-6.25 kg and spanning 15 FrameNet frames and 90 Frame-LU labels (Table 7). For strict object-interaction sequences, the release provides synchronized object trajectories and object meshes together with the human BVH motion.

## C Release Terms, License, and Ethics

License. HiPHI will be released under the ModalityNet Open Research License v1.0, a custom non-commercial license for scientific research, education, and evaluation. The complete license terms will accompany the public release of HiPHI.

Release schedule. The full dataset, including BVH motions, object trajectories, object meshes, and the Frame-LU index, will be released publicly soon. The supplementary material contains four representative sample sequences (three body-only and three object-interaction) that illustrate the release format and quality.

Ethics. Data collection was reviewed and approved by an institutional ethics review board before capture. All performers were adults who provided written informed consent describing the capture protocol, the intended research use, and the planned public release of the motion data and anonymized demographic attributes. Performers were compensated for each capture session and may request withdrawal of their data at any time. During capture, RGB video and audio were recorded as references for production review; the public release of HiPHI contains only BVH motion, object trajectories, and object meshes, and does not include any RGB video, audio, facial imagery, or voice recordings. Performer identities are released only as numeric actor identifiers, with anonymized height, weight, and gender attributes. Specific institutional and protocol identifiers are withheld for double-blind review and will be provided in the camera-ready version.

Table 6: Metadata fields exposed by the HiPHI release. Fields are provided through the global motion manifest, performer metadata, and per-package metadata.json files.
<table><tr><td>Field</td><td>Scope</td><td>Meaning</td></tr><tr><td>Global motion metadata</td><td></td><td></td></tr><tr><td>motion_id</td><td>Global / package</td><td>Unique motion identifier. Mirrored motions use the corresponding ID</td></tr><tr><td>frame</td><td>Global / package</td><td>with a _mirror suffix. FrameNet frame label.</td></tr><tr><td>lu</td><td>Global / package</td><td>FrameNet lexical-unit label.</td></tr><tr><td>frame_lu</td><td>Global / package</td><td>Frame-LU motion-unit label.</td></tr><tr><td>duration_sec</td><td>Global / package</td><td>Temporal duration of the motion, in seconds.</td></tr><tr><td>frame_count</td><td>Global / package</td><td>Number of motion frames in the sequence.</td></tr><tr><td>actor_id</td><td>Global / package</td><td>Anonymized performer identifier in the release.</td></tr><tr><td>text_annotation</td><td>Global / package</td><td>Natural-language annotation aligned with the motion sequence.</td></tr><tr><td>is_hoi</td><td>Global / package</td><td>Whether the sequence contains tracked human-object interaction.</td></tr><tr><td>object_categories</td><td>Global</td><td>Object-category information associated with the sequence, when applicable.</td></tr><tr><td>mirrored</td><td>Global / package</td><td>Boolean indicating whether the motion is the left-right mirrored counterpart of an original sequence.</td></tr><tr><td>Performer metadata</td><td></td><td></td></tr><tr><td>actor_id</td><td>Actor metadata</td><td>Anonymized performer identifier used to link performer metadata to</td></tr><tr><td>height_cm</td><td>Actor metadata</td><td>motion sequences. Performer height in centimeters.</td></tr><tr><td>weight_kg</td><td>Actor metadata</td><td>Performer weight in kilograms.</td></tr><tr><td>gender</td><td>Actor metadata</td><td>Performer gender metadata.</td></tr><tr><td>Additional per-package metadata</td><td></td><td></td></tr><tr><td>dataset</td><td>Package only</td><td>Dataset identifier stored in the package metadata.</td></tr><tr><td>fps</td><td>Package only</td><td>Frame rate associated with the released motion package.</td></tr><tr><td>actor_metadata</td><td>Package only</td><td>Nested performer metadata associated with the sequence.</td></tr><tr><td>objects</td><td>Package only</td><td>List of tracked objects associated with the sequence; empty for</td></tr><tr><td>objects[].object_id</td><td>Package only</td><td>body-only motion. Identifier of a tracked object instance.</td></tr><tr><td>objects[].object_category</td><td>Package only</td><td>Semantic category of the tracked object.</td></tr><tr><td>objects[].trajectory-path</td><td>Package only</td><td>Package-relative path to the synchronized object trajectory.</td></tr><tr><td>objects[].mesh_id</td><td>Package only</td><td>Identifier of the corresponding object mesh.</td></tr><tr><td>objects [] .mesh_path</td><td>Package only</td><td>Package-relative path to the corresponding object mesh.</td></tr></table>

Table 7: Object statistics of the HiPHI human-object interaction subset. The subset contains diverse real-world objects with synchronized human motion, object trajectories, and object meshes.

<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Object instances</td><td>40</td></tr><tr><td>Object categories</td><td>12</td></tr><tr><td>Mass range</td><td>0.45-6.25 kg</td></tr><tr><td>Covered FrameNet frames</td><td>15</td></tr><tr><td>Covered Frame-LU labels</td><td>90</td></tr><tr><td>Interaction duration</td><td>245.7 h</td></tr><tr><td>Object state modalities</td><td>Trajectory + Mesh</td></tr></table>

## D Representative Sample Sequences

We show six representative sequences from HiPHI. Per-sequence renderings are shown in Figure 10, and full motion playback is provided in the supplementary video.

## E Additional Details for Motion-Space Diversity

Unified motion representation. For the motion-space analysis in Sec. 5.1, all datasets are first mapped to a common body representation with 23 keypoints in meters, resampled to 30 FPS in a z-up coordinate frame. The keypoints are pelvis, spine, chest, neck, head, left/right hip, knee, ankle, foot, toe, shoulder, elbow, wrist, and hand. Each sequence is divided into 30-frame windows with a stride of 25 frames. The per-frame input contains heading-local, root-relative joint positions and root-relative joint velocities. We further apply body-scale normalization and subtract the per-window median pose, which reduces scale and skeleton-layout differences across datasets. The resulting feature dimension is 23 $\times 3 \times 2 = 1 3 8$ per frame.

![](images/26a9eb956b821f02f8283dc0b18471f348b3e1487e912321e096c53ce4688392.jpg)  
Figure 10: Representative sample sequences from HiPHI. We show six representative sequences from HiPHI: (a)-(c) body-only motion sequences and (d)-(f) object-interaction motion sequences.

Table 8: Embedding setup for the motion-space coverage analysis.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Keypoints</td><td>23 body keypoints, meter scale, z-up</td></tr><tr><td>Frame rate</td><td>30 FPS</td></tr><tr><td>Window / stride</td><td>30 frames / 25 frames</td></tr><tr><td>Feature type</td><td>Heading-local normalized joint dynamics and relative velocities</td></tr><tr><td>Feature dimension</td><td>138 per frame</td></tr><tr><td>Balanced sampling</td><td>Up to 5,000 clips and 5,000 windows per dataset</td></tr><tr><td>Encoder</td><td>Shared unsupervised temporal Conv1D autoencoder</td></tr><tr><td>Latent dimension</td><td>16</td></tr><tr><td>Objective</td><td>Reconstruction MSE</td></tr><tr><td>Checkpoint</td><td>Epoch 20</td></tr><tr><td>t-SNE</td><td>PCA dim. 5, perplexity 30, PCA init., auto learning rate, 1000 itera- tions, seed 42</td></tr><tr><td>Grid statistics</td><td>55 × 55 grid on the default 2-D embedding</td></tr></table>

Dataset-balanced sampling. To prevent dataset size from dominating either representation learning or coverage evaluation, we construct a balanced sample for each dataset. We randomly sample 5,000 motion clips from each dataset when available and draw one valid 30-frame window from each selected clip, yielding 5,000 training windows per dataset. LAFAN1 uses all available clips because it contains fewer than 5,000 samples. The t-SNE projection and grid-based occupancy analysis follow the same cap of at most 5,000 windows per dataset. Therefore, both the shared representation and the reported coverage statistics are dataset-balanced rather than proportional to raw dataset size.

Shared unsupervised encoder. We train one temporal convolutional autoencoder jointly on the balanced samples from all datasets and use its encoder as the motion descriptor. The model takes a tensor of size 30 × 138, uses temporal Conv1D layers with residual dilated blocks, and outputs a 16-D latent code. It is trained only with reconstruction loss; dataset identity, text labels, Frame-LU labels, and object labels are not used. The default analysis uses the epoch-20 checkpoint. Before t-SNE, latent features are standardized and reduced to 5 dimensions with PCA. We then run t-SNE with Euclidean distance, perplexity 30, PCA initialization, automatic learning rate, 1000 iterations, and random seed 42.

Grid-based coverage statistics. Let G be the $5 5 \times 5 5$ grid in the shared t-SNE plane and $n _ { D } ( c )$ the number of sampled points from dataset D in cell $c \in { \mathcal { G } }$ . Let $\mathcal { G } _ { D } ^ { + } = \{ c \in \mathcal { G } : n _ { D } ( c ) > 0 \}$ be the cells occupied by D. We report occupied grid cells, entropy-based effective grid cells, and long-tail share:

Δ Occ. is reported as grid size × grid size: occupied-cell difference (HiPHI − best baseline; positive and higher is better). Because Δ Occ. varies with grid resolution, we evaluate multiple grid sizes. HiPHI (ours) BONES-SEED Motion-X++ AMASS  
![](images/03b7d2cee81c77e32c371e8a679d05f657c84c11e926a709c6d875cb8f8b7520.jpg)  
Figure 11: Robustness of motion-space coverage. We repeat the coverage analysis using different random seeds, t-SNE perplexities, and grid resolutions. Each entry reports the occupied-cell margin between HiPHI and the strongest baseline under the corresponding setting. HiPHI maintains a positive margin across all tested configurations.

$$
\begin{array} { l } { { m _ { D } = | \mathcal { G } _ { D } ^ { + } | , \qquad p _ { D } ( c ) = \frac { n _ { D } ( c ) } { \sum _ { c ^ { \prime } \in \mathcal { G } } n _ { D } ( c ^ { \prime } ) } , } } \\ { { e _ { D } = \exp \Bigl ( - \displaystyle \sum _ { c \in \mathcal { G } _ { D } ^ { + } } p _ { D } ( c ) \log p _ { D } ( c ) \Bigr ) , \qquad \ell _ { D } = \displaystyle \sum _ { c \in \mathcal { T } } p _ { D } ( c ) . } } \end{array}\tag{3}
$$

Here, $m _ { D }$ is the number of occupied cells, measuring how broadly a dataset spans the shared motion space. The effective occupancy $e _ { D }$ measures how evenly its samples are distributed across the occupied cells. The long-tail share $\ell _ { D }$ measures the fraction of samples located in globally rare regions. We use the convention 0 log $0 = 0$ . The set $\tau \subseteq \mathcal G$ contains non-empty cells whose global occupancy is at or below the 25th percentile of all non-empty global cells; under the default 55 × 55 setting, this threshold corresponds to 7 points per cell.

Coverage robustness analysis. We further test whether the occupied-region advantage depends on a particular random seed, t-SNE perplexity, or grid resolution. Starting from the default configuration in Table 8, we repeat the analysis using random seeds 55, 73, and 112; perplexities 40, 55, and 65; and grid resolutions of $5 5 \times 5 5 , 8 8 \times 8 8$ , and $1 1 1 \times 1 1 1$

For every configuration, we report the occupied-cell margin

$$
\Delta \mathrm { O c c } = m _ { \mathrm { H i P H I } } - \operatorname* { m a x } _ { D \neq \mathrm { H i P H I } } m _ { D } ,\tag{4}
$$

where the second term is the occupied-cell count of the strongest baseline under the same configuration. A positive value means that HiPHI covers more cells than every compared dataset. Because occupiedcell counts naturally change with grid resolution, the margin is reported separately for each resolution.

As shown in Figure 11, HiPHI maintains a positive occupied-cell margin in every tested configuration. The coverage advantage therefore remains consistent across resampling, projection, and discretization choices.

Table 9: Summary of quality metrics used in Sec. 5.2.
<table><tr><td>Symbol</td><td>Meaning</td><td>Unit</td><td>Better</td></tr><tr><td> $\mathcal { I } _ { \tau }$ </td><td>Upper-tail fixed-window jerk over core-body joints</td><td> $\mathrm { m } / \mathrm { s } ^ { 3 }$ </td><td>lower</td></tr><tr><td> $\mathcal { A }$ </td><td>Median core-body acceleration</td><td> $\mathrm { { m } / \mathrm { { s } ^ { 2 } } }$ </td><td>lower</td></tr><tr><td> $\delta _ { \mathrm { g r o u n d } }$ </td><td>Upper-tail below-ground support-point depth</td><td>mm</td><td>lower</td></tr><tr><td>φfloat</td><td>Unsupported-floating duration share</td><td> $\%$ </td><td>lower</td></tr><tr><td> $\nu _ { \mathrm { f o o t } }$ </td><td>Support-point drift over contact frames</td><td>mm/s</td><td>lower</td></tr><tr><td> $\eta _ { \mathrm { n c } }$ </td><td>Non-conflict fraction under skeleton-point / object- % volume proxy</td><td></td><td>higher</td></tr><tr><td> $\rho _ { \mathrm { n e a r } }$ </td><td>Near-surface fraction within 20 cm of the object sur- face</td><td> $\%$ </td><td>higher</td></tr></table>

## F Additional Details for Data Quality and Precision

Aggregation across sequences. All quality statistics in Sec. 5.2 are aggregated over the full evaluated duration of each dataset, so that every recorded second contributes equally and dataset size does not bias the result. Let $r _ { k }$ index any observation at the relevant level (frame, window, or sequence aggregate) with non-negative duration weight $w _ { k } \left( \mathrm { i . e . , } w _ { k } = \Delta t \right.$ of the source for per-frame observations, the window length for per-window observations, and the sequence duration $T _ { i }$ for per-sequence aggregates). We report

$$
\mathbb { E } [ r ] = \frac { \sum _ { k } w _ { k } r _ { k } } { \sum _ { k } w _ { k } } , \qquad \mathbb { Q } _ { \tau } ( r ) = \operatorname* { i n f } \Bigl \{ a : \frac { \sum _ { k } w _ { k } { \bf 1 } [ r _ { k } \le a ] } { \sum _ { k } w _ { k } } \ge \tau \Bigr \} ,\tag{5}
$$

i.e., the time-weighted mean and quantile, with $\tau = 0 . 9 5$ for upper-tail statistics. For ratio-type metrics $( \eta _ { \mathrm { n c } } , \rho _ { \mathrm { n e a r } } , \phi _ { \mathrm { f l o a t } } )$ , each $r _ { i }$ is itself a frame-level ratio inside sequence $i ,$ and weighting by $T _ { i }$ recovers the frame-level aggregation across the dataset. For distribution-type metrics $( \mathcal { I } _ { \tau } , \mathcal { A } , \delta _ { \mathrm { g r o u n d } }$ $\nu _ { \mathrm { f o o t } } ) , r _ { k }$ is the per-frame (or per-pair) observation entering the corresponding equation in Sec. 5.2.

Body-motion metric sets and smoothing. Equation 1 uses one joint set, one point set, and one pair set. C is the core-body joint set, excluding fingers and face joints when present. $\mathcal { F }$ is the support-point set: foot, toe, and ankle proxies when available. $\mathcal { K } = \{ ( t , f ) : h _ { f } ( t ) - g < \epsilon \}$ is the contact set of (frame, support-point) pairs within $\epsilon = 3 0$ mm of the ground at height g; speed is then used to measure drift over those pairs. The finite-difference operator $\nabla _ { t }$ is divided by the source-specific frame interval $\Delta t ,$ so $\nabla _ { t } ^ { k } \boldsymbol { x }$ has units of $\mathrm { m } / \mathrm { s } ^ { k }$ . For jerk, trajectories are smoothed by a fixed physical window of $1 / 6 \ \mathrm { s } ,$ implemented as an odd-length moving average with a minimum size of three frames; raw positions are used for acceleration. Floor-related metrics require a comparable absolute ground convention, which is why Motion-X++ is excluded from $\delta _ { \mathrm { g r o u n d } } , \phi _ { \mathrm { f l o a t } }$ , and $\nu _ { \mathrm { f o o t } }$ in Table 2.

Object-interaction geometry. Equation 2 uses a sparse skeleton-point proxy. The human body is represented by canonical skeleton segments covering hands, arms, legs, torso, pelvis, and head/neck regions. For each evaluated frame, segment centerlines are uniformly sampled (32 points per segment in the released code) to produce ${ \mathbf { } } S ( t )$ . The object is represented as a closed 3D region $\Omega ( t )$ obtained by posing the released mesh; ∂Ω(t) is its boundary surface. Distances $d ( S , \partial \Omega )$ are computed in the shared coordinate frame. HUMOTO is evaluated on its publicly released GLB subset; the full licensed dataset is not directly downloadable as public data.

Applicability. The body-motion smoothness metrics require only temporally aligned human joint trajectories and native timing. Ground-contact metrics additionally require a comparable up-axis and absolute ground convention. Object-interaction metrics require synchronized human motion, object pose, and object mesh in a shared coordinate frame. HUMOTO is evaluated on its publicly released GLB subset; the full licensed dataset is not directly downloadable as public data.

## G Training Details for Whole-Body Motion Tracking

For whole-body motion tracking, we evaluate all datasets under the same physics-based imitation learning pipeline. All source motions are first retargeted to the Unitree G1 humanoid using Py-Roki [45], and the resulting robot motions are then used as reference trajectories for policy learning. We use the same observation design, reward formulation, optimization budget, and evaluation protocol for all data sources so that performance differences mainly reflect the physical usability of the underlying motion data. Each tracking experiment is repeated five times with different random seeds. We report the mean curve and variance across runs. The detailed observation and reward are summarized in Tables 10 and 11.

Figure 12 shows qualitative whole-body tracking results of HiPHI on the Unitree G1 humanoid. The learned policy produces stable and physically plausible motions across diverse behaviors, demonstrating that the motions in HiPHI can serve as effective executable references for humanoid control.

Table 10: Observation terms for whole-body motion tracking.
<table><tr><td>State</td><td>Dim.</td></tr><tr><td>(a) Motion Command</td><td></td></tr><tr><td>Reference Body Position (heading frame)</td><td>99</td></tr><tr><td>Reference Body Position (relative)</td><td>99</td></tr><tr><td>Reference Body Orientation (heading frame)</td><td>198</td></tr><tr><td>Reference Body Orientation (relative)</td><td>198</td></tr><tr><td>Reference Body Linear Velocity (relative)</td><td>99</td></tr><tr><td>Reference Body Angular Velocity (relative)</td><td>99</td></tr><tr><td>(b) Proprioceptive State</td><td></td></tr><tr><td>Root Height (above terrain)</td><td>1</td></tr><tr><td>Body Position (non-root, heading frame)</td><td>96</td></tr><tr><td>Body Orientation (heading frame, tan-norm)</td><td>198</td></tr><tr><td>Body Linear Velocity (heading frame)</td><td>99</td></tr><tr><td>Body Angular Velocity (heading frame)</td><td>99</td></tr><tr><td>Last Action</td><td>29</td></tr></table>

Table 11: Reward terms used for whole-body motion tracking.
<table><tr><td>Term</td><td>Expression</td><td>Weight</td><td>Remarks</td></tr><tr><td colspan="4">Tracking Rewards</td></tr><tr><td>Global Translation (gt)</td><td> $\begin{array} { r } { \exp \left( c _ { \mathrm { g t } } \frac { 1 } { n _ { b } } \sum _ { i } \| p _ { i } - \hat { p } _ { i } \| _ { 2 } ^ { 2 } \right) } \end{array}$ </td><td>0.5</td><td>cgt = −25</td></tr><tr><td>Global Rotation (gr)</td><td> $\begin{array} { r } { \exp \left( c _ { \mathrm { g r } } \frac { 1 } { n _ { b } } \sum _ { i } d _ { R } ( q _ { i } , \hat { q } _ { i } ) ^ { 2 } \right) } \end{array}$ </td><td>0.3</td><td>cgr = −5</td></tr><tr><td>Global Linear Velocity (gv)</td><td> $\begin{array} { r } { \exp \left( c _ { \mathrm { g v } } \frac { 1 } { n _ { b } } \sum _ { i } \| v _ { i } - \hat { v } _ { i } \| _ { 2 } ^ { 2 } \right) } \end{array}$ </td><td>0.1</td><td>cgv = −0.5</td></tr><tr><td>Global Angular Velocity (gav)</td><td> $\begin{array} { r } { \exp \left( c _ { \mathrm { g a v } } \frac { 1 } { n _ { b } } \sum _ { i } \| \omega _ { i } - \hat { \omega } _ { i } \| _ { 2 } ^ { 2 } \right) } \end{array}$ </td><td>0.2</td><td>Cgav = −0.1</td></tr><tr><td>Root Height (rh)</td><td> $\exp \left( { { c _ { \mathrm { r h } } } \left( { z _ { \mathrm { r o o t } } } - { \hat { z } } _ { \mathrm { r o o t } } \right) ^ { 2 } } \right)$ </td><td>0.2</td><td>Crh = −100</td></tr><tr><td colspan="4">Regularization and Safety Penalties</td></tr><tr><td>Action Smoothness</td><td>|āt− ăt-1∥2</td><td>-0.02</td><td></td></tr><tr><td>Power Consumption</td><td>∑j |τj ij|</td><td>-10⁻5</td><td></td></tr></table>

Here $d _ { R } ( \cdot , \cdot )$ denotes quaternion orientation error, n<sub>b</sub> is the number of tracked body links.

## H Training Details for Motion-with-Object Tracking

For motion-with-object tracking, we first process synchronized human-object demonstrations using Omni-Retarget [41], which retargets human motion to the Unitree G1 humanoid while preserving the corresponding object trajectories. Because existing open-source human-object motion datasets usually do not specify object physical parameters, we use a unified object configuration across datasets for fair comparison. Unless otherwise specified, each object is modeled as a free-joint mesh object with a total mass of 2.0 kg.

We then use a BeyondMimic-style tracking setup [39], following a similar observation design and reward formulation while extending the framework to explicitly model object interactions. Since the task requires the humanoid to reproduce both whole-body motion and the associated object trajectory, we additionally include the current object state in the policy observation. We also introduce object-tracking rewards by measuring the position and orientation errors between the simulated object and its reference trajectory. To improve policy robustness, we apply domain randomization to both the humanoid and the object, including object mass, friction, inertial properties, contact parameters, and external perturbations. The detailed observation, reward, and randomization settings are summarized in Tables 12, 13, and 14.

![](images/342d6eab7880e802900ee42986f99a637a48338da332b82228db22c3b90038ce.jpg)  
Figure 12: Qualitative visualization of diverse humanoid motion skills trained with HiPHI. The motions cover diverse whole-body behaviors, including standing, walking, sitting, lying down, crawling, kneeling, and recovery motions, demonstrating the coverage and diversity of the training data.

Table 12: Observation terms for motion-with-object tracking.
<table><tr><td>State</td><td>Dim.</td><td>Policy</td><td>Critic</td></tr><tr><td colspan="4">(a) Motion Command</td></tr><tr><td>Reference Joint Pos</td><td>29</td><td>√</td><td>√</td></tr><tr><td>Reference Joint Vel</td><td>29</td><td>√</td><td>V</td></tr><tr><td>Reference Anchor Position</td><td>3</td><td>√</td><td>√</td></tr><tr><td>Reference Anchor Orientation</td><td>6</td><td>√</td><td>1</td></tr><tr><td colspan="4">(b) Proprioceptive State</td></tr><tr><td>Base Linear Velocity</td><td>3</td><td>x</td><td></td></tr><tr><td>Base Angular Velocity</td><td>3</td><td>√</td><td>&gt;</td></tr><tr><td>Joint Position</td><td>29</td><td></td><td></td></tr><tr><td>Joint Velocity</td><td>29</td><td>&gt;&gt;~</td><td></td></tr><tr><td>Last Action</td><td>29</td><td></td><td></td></tr><tr><td colspan="4">(c) Object State</td></tr><tr><td>Object position in robot frame</td><td>3</td><td>√</td><td>√</td></tr><tr><td>Object orientation in robot frame</td><td>6</td><td>√</td><td>√</td></tr><tr><td colspan="4">(d) Critic-Only Privileged State</td></tr><tr><td>Body position</td><td>14 × 3</td><td>x</td><td>√</td></tr><tr><td>Body orientation</td><td>14 × 6</td><td>x</td><td>√</td></tr></table>

Table 13: Reward terms used for motion-with-object tracking.
<table><tr><td>Term</td><td>Expression</td><td></td><td>Remarks</td></tr><tr><td colspan="4">Tracking Rewards</td></tr><tr><td>Anchor position</td><td> $\exp \left( - \Vert p _ { a } - \hat { p } _ { a } \Vert _ { 2 } ^ { 2 } / \sigma ^ { 2 } \right)$ </td><td>0.5</td><td> $\sigma = 0 . 3$ </td></tr><tr><td>Anchor orientation</td><td> $\exp { \left( - d _ { R } ( q _ { a } , \hat { q } _ { a } ) ^ { \bar { 2 } } / \sigma ^ { 2 } \right) }$ </td><td>0.5</td><td> $\sigma = 0 . 4$ </td></tr><tr><td>Body position</td><td> $\begin{array} { r l } { \exp { \biggl ( - \frac { 1 } { n _ { b } } \sum _ { i } \| p _ { i } - \hat { p } _ { i } \| _ { 2 } ^ { 2 } / \sigma ^ { 2 } \biggr ) } } & { { } } \end{array}$ </td><td>1.0</td><td> $\sigma = 0 . 3$ </td></tr><tr><td>Body orientation</td><td> $\begin{array} { r l } { \exp \Bigl ( - \frac { 1 } { n _ { b } } \sum _ { i } d _ { R } ( q _ { i } , \hat { q } _ { i } ) ^ { 2 } / \sigma ^ { 2 } \Bigr ) } & { { } } \end{array}$ </td><td>1.0</td><td> $\sigma = 0 . 3$ </td></tr><tr><td>Body linear velocity</td><td> $\begin{array} { r l } { \exp { \biggl ( - \frac { 1 } { n _ { b } } \sum _ { i } \| v _ { i } - \hat { v } _ { i } \| _ { 2 } ^ { 2 } / \sigma ^ { 2 } \biggr ) } } & { { } } \end{array}$ </td><td>0.5</td><td> $\sigma = 1 . 0$ </td></tr><tr><td>Body angular velocity</td><td> $\begin{array} { r l } { \exp { \bigg ( - \frac { 1 } { n _ { b } } \sum _ { i } { \| \omega _ { i } - \hat { \omega } _ { i } \| _ { 2 } ^ { 2 } } / \sigma ^ { 2 } \bigg ) } } & { { } } \end{array}$ </td><td>0.5</td><td> $\sigma = 3 . 1 4$ </td></tr><tr><td>Object position</td><td> $\exp \big ( - \| \bar { p _ { o } } - \hat { p } _ { o } \| _ { 2 } ^ { 2 } / \sigma ^ { 2 } \big )$ </td><td>1.0</td><td> $\sigma = 0 . 3$ </td></tr><tr><td>Object orientation</td><td> $\exp \left( - d _ { R } ( q _ { o } , \hat { q } _ { o } ) ^ { 2 } / \sigma ^ { 2 } \right)$ </td><td>1.0</td><td> $\sigma = 0 . 3$ </td></tr><tr><td colspan="4">Regularization and Safety Penalties</td></tr><tr><td>Action rate</td><td> $\lVert a _ { t } - a _ { t - 1 } \rVert _ { 2 } ^ { 2 }$ </td><td>-0.1</td><td>Smooth action penalty</td></tr><tr><td>Joint limit</td><td> ${ \bf 1 } ( q \notin [ q _ { \mathrm { m i n } } , q _ { \mathrm { m a x } } ] )$ </td><td>-10.0</td><td>Joint limit violation</td></tr></table>

Here $d _ { R } ( \cdot , \cdot )$ denotes quaternion orientation error, $n _ { b }$ is the number of tracked body links, c and cˆ denote measured and reference contact states, and hatted variables denote reference motion states.

Table 14: Domain randomization and disturbance settings for motion-with-object tracking.
<table><tr><td>Term</td><td>Value</td></tr><tr><td colspan="2">External Disturbances</td></tr><tr><td>Push robot</td><td>interval = 1–3s, vx, vy ∼ U[−0.5, 0.5] m/s,  $v _ { z } \sim U [ - 0 . 2 , 0 . 2 ]$  m/s</td></tr><tr><td colspan="2">Robot Dynamics Randomization</td></tr><tr><td>Torso COM offset</td><td> $x \sim U [ - 0 . 0 2 5 , 0 . 0 2 5 ] { \mathrm { ~ m } } , \quad y , z \sim U [ - 0 . 0 5 , 0 . 0 5 ] { \mathrm { ~ m } }$ </td></tr><tr><td>Encoder bias</td><td> $U [ - 0 . 0 1 , 0 . 0 1 ]$ </td></tr><tr><td>Default joint position offset</td><td> $U [ - 0 . 0 1 , 0 . 0 1 ]$  rad</td></tr><tr><td>Rigid-body material</td><td> $\mathrm { s t a t i c ~ f r i c t i o n } ~ U [ 0 . 3 , 1 . 6 ]$  , dynamic friction U [0.3, 1.2], restitution  $U [ 0 . 0 , 0 . 5 ]$ </td></tr><tr><td colspan="2">Object Dynamics Randomization</td></tr><tr><td>Object mass</td><td> $U [ 0 . 3 , 2 . 0 ] \times$  default mass</td></tr><tr><td>Object COM offset</td><td> $U [ - 0 . 0 2 , 0 . 0 2 ] \mathrm { m }$ </td></tr></table>

Figure 13 presents qualitative comparisons of motion-with-object tracking across three datasets: HiPHI, OMOMO, and HUMOTO. We visualize two representative interaction categories, including carry (first row) and kick interactions (bottom row). Existing datasets exhibit limited interaction coverage and tracking robustness in certain categories. For example, HUMOTO fails to reproduce stable object-aware behaviors in the carry scenario, while OMOMO suffers from reduced interaction fidelity. In contrast, HiPHI produces more stable whole-body motions and more consistent humanobject interaction behaviors across diverse tasks.

![](images/6484e55929282b021b5bf6b257e9d4ac1e831a5c43abf23a086939e714070f9e.jpg)  
Figure 13: Qualitative comparisons of motion-with-object tracking across different datasets. The top row shows carry interactions, while the bottom row presents kick interaction behaviors. HiPHI demonstrates more stable whole-body tracking and better human-object interaction consistency across diverse tasks.
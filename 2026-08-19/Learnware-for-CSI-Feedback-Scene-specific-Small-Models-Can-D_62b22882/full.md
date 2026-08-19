# Learnware for CSI Feedback: Scene-specific Small Models Can Do Big

Xiangyi Li, Graduate Student Member, IEEE, Jiajia Guo, Member, IEEE, Chao-Kai Wen, Fellow, IEEE, Xin Geng, Senior Member, IEEE, Shi Jin, Fellow, IEEE, and Zhi-Hua Zhou, Fellow, IEEE

Abstract—Intelligent channel state information (CSI) feedback is essential for realizing the high capacity and spectral efficiency goals of future 6G systems, yet existing deep learning solutions face a trade-off between model generalization and scenariospecific performance. Large neural networks generalize well but incur high computational and tuning costs, while small models excel in particular environments but require repetitive costly end-to-end training for each base station (BS). To address these challenges, we introduce a model repository-based deployment framework in which a centralized AI data center maintains a catalog of scene-specific CSI models. The repository is enhanced with a Learnware-based framework, where each model is associated with a specification including semantic part (network architecture parameters) and statistical part (codebookfingerprint embeddings of training-data distributions). A BS submits only its local statistical specifications to retrieve the most relevant pre-trained model, enhancing data privacy by avoiding raw CSI transmission and drastically reducing retrieval latency and communication overhead. We further develop a data-driven search strategy that matches codebook fingerprints to model performance, achieving over 90% selection accuracy. In simulations, our scheme yields 18.8% and 57.7% performance improvements over the General Model in LOS and NLOS scenarios, respectively while reducing local fine-tuning by up to 1000 samples and 100 epochs. This Learnware-based approach minimizes redundant training, maximizes model reuse, and supports rapid, privacyenhancing deployment of CSI feedback models.

Index Terms—Massive MIMO, CSI feedback, deep learning, model repository, model selection.

## I. INTRODUCTION

CCORDING to the latest 6G vision released by the International Telecommunication Union (ITU) [1], future   
wireless communication systems will demand higher through  
put, lower latency, greater reliability, massive connectivity, and   
more efficient spectrum utilization [2], [3]. Massive multiple  
input multiple-output (MIMO) arrays are essential to achieving   
these goals, offering spatial multiplexing gains that improve

![](images/c6af24c76f5fafeb2cd3fbe1e18bc06ead9f41dbecc861923cbf9b48247cb20e.jpg)  
Fig. 1. Model repository-based CSI feedback deployment, where the respective functions of the AI data center, BS and UE are marked in different colors.

both capacity and spectral efficiency [4]. In frequency division duplexing (FDD) MIMO systems, user equipment (UE) estimates the downlink channel state information (CSI) and feeds it back to the base station (BS) for precoding. Since transmitting full CSI entails substantial overhead, efficient compression and reconstruction mechanisms are critical.

Conventional CSI compression techniques, such as compressed sensing [5] and codebook-based methods [6], provide theoretical guarantees but often underperform in complex, large-scale deployments [7]. Deep learning (DL)-based methods have emerged as effective alternatives by modeling CSI matrices as high-dimensional images and training autoencoders for end-to-end compression and reconstruction. This approach was first demonstrated in CsiNet [8], which achieved significant reductions in feedback overhead while maintaining reconstruction accuracy. Subsequent works have incorporated convolutional structures, attention mechanisms, and lightweight network designs to further improve performance and reduce inference latency [9]–[15]. In parallel with these research advances, the 3rd Generation Partnership Project (3GPP) initiated AI-driven CSI feedback as a study item in Release 18 (R18) and subsequently advanced it to a work item in Release 20 [16], [17]. More broadly, AI has been recognized as a key enabler of 6G and is being incorporated into air interface design across multiple 3GPP working groups [18]–[20].

Despite these advances, practical deployment of AI-based CSI feedback remains challenging. Current systems typically centralize model training in AI data centers, which either generate personalized models based on CSI uploaded from BSs or construct unified models trained on aggregated data [21]– [23]. Unified models facilitate broad applicability but may exhibit suboptimal performance in specialized environments, while also incurring high computational costs, long training times, and greater reliance on expert tuning. In contrast, scene-specific small models trained on localized CSI can achieve near-optimal performance in their target scenarios, supporting dynamic model switching or rapid fine-tuning during deployment [24]–[27]. For example, the 3GPP R18 Model Switch strategy introduces multiple predefined configurations (e.g., 38.901 UMi/UMa) for online selection [16] [28]. However, models trained for personalized deployment are typically discarded after use, resulting in wasted computational resources and storage. More importantly, existing literature predominantly focuses on model training and inference, with limited attention to systematic model storage, management, and reuse across deployments, which constitutes a critical gap that this work aims to address.

To address these limitations, we propose a model repositorybased deployment framework, illustrated in Fig. 1, which aligns with the 3GPP AI/ML lifecycle management framework [20]. Our framework focuses primarily on model storage and repository, designing its effective connection with other modules. A centralized repository maintains a catalog of scene-specific AI models, each annotated with standardized metadata. In management, when a BS submits a Delivery Request (retrieval request) by providing a small amount of local CSI or a brief scene description, the repository returns the most appropriate pre-trained model with model transfer for the BS's inference and further model updating, minimizing the need for local fine-tuning and avoiding redundant training.

Efficient model retrieval requires informative yet privacyaware metadata. To this end, we adopt and tailor the Learnware paradigm to CSI feedback deployment. Learnware is defined as Model plus Specification [29]-[32], where each model is accompanied by a specification that characterizes its capability without exposing the training data. In this work, we extend the Learnware specification framework to the CSI compression task. Each specification consists of two components. The semantic specification encodes architectural attributes such as input and output dimensions, compression ratios, and task types. The statistical specification summarizes the training data distribution through compact embeddings derived from codebook fingerprint frequency vectors. This design enables efficient model comparison and retrieval without sharing raw CSI. It has been shown in [33] that the specification mechanism in [29], [31] protects the original training data even against inference and linkage attacks under strong adversarial settings. Therefore, by uploading only statistical specifications rather than raw CSI, BSs avoid exposing raw channel data, mitigating a primary privacy risk, while enabling rapid model retrieval with low communication overhead. The main contributions of this work are summarized as follows:

• First repository-centric CSI feedback deployment architecture. We propose a principled model repositorybased deployment framework for AI-driven CSI feedback, aligned with the 3GPP AI and ML lifecycle management architecture. Unlike conventional retrain-anddiscard paradigms, the proposed framework enables systematic model reuse and adaptation across heterogeneous deployment scenarios. This architecture fundamentally transforms CSI feedback from isolated model training to lifecycle-aware model orchestration, substantially reducing redundant training cost and accelerating deployment.

• Learnware-enabled privacy-enhancing CSI model indexing. We introduce a Learnware-based repository design tailored to CSI compression tasks and develop a novel codebook fingerprint pdf as the statistical specification. The proposed representation captures angulardomain channel structure through DFT beamspace projection while avoiding raw CSI exposure. This design establishes a unified, privacy-enhancing, and computationefficient model indexing mechanism that enables scalable model comparison across diverse radio environments.

• Scalable sub-millisecond retrieval with controllable accuracy-latency tradeoff. We develop a high-precision and low-latency model retrieval framework based on pdf similarity matching, achieving over 90% selection accuracy. To meet stringent CSI feedback timing constraints, we further design a multi-level locality-sensitive hashing (LSH) structure that scales to billion-model repositories and enables 0.25–1.15 ms retrieval latency. The proposed mechanism provides a controllable tradeoff between retrieval accuracy and latency through adjustable search depth, making it suitable for real-time wireless systems.

The remainder of this paper is organized as follows. Section II describes the system and channel models. Section III introduces the Learnware-based repository structure and the specification designs. Section IV displays the online deployment framework with Learnware repository. Section V presents simulation settings, comparative evaluations, and ablation studies. Section VI concludes the paper and outlines future work.

## II. SYSTEM MODEL

## A. Massive MIMO FDD System

Consider a downlink massive MIMO system where a BS equipped with $N _ { \mathrm { t } }$ uniform linear array (ULA) antennas serves a UE with $N _ { \mathrm { r } }$ antennas. The received signal at the UE is given by [7], [8]:

$$
\mathbf { y } = \mathbf { H } \mathbf { v } s + \mathbf { n } ,\tag{1}
$$

where $\mathbf { H } \in \mathbb { C } ^ { N _ { \mathrm { r } } \times N _ { \mathrm { t } } }$ is the channel matrix, $\mathbf { v _ { \tau } } \in \mathbb { C } ^ { N _ { \mathrm { t } } }$ is the precoding vector with $\| \mathbf { v } \| _ { 2 } = 1 , s \in \mathbb { C }$ is the transmitted symbol with $\mathbb { E } [ | s | ^ { 2 } ] = 1$ , and n $\mathbf { \Psi } _ { \cdot } \in \mathbb { C } ^ { N _ { \mathrm { r } } }$ is the additive white Gaussian noise with $\mathbf { n } \sim \mathcal { C N } ( 0 , \sigma ^ { 2 } \mathbf { I } _ { N _ { \mathrm { r } } } )$ . The channel matrix can be decomposed via singular value decomposition (SVD) as:

$$
\mathbf { H } = \mathbf { U } \pmb { \Sigma } \mathbf { V } ^ { H } ,\tag{2}
$$

where $\mathbf { U } \in \mathbb { C } ^ { N _ { \mathrm { r } } \times N _ { \mathrm { r } } }$ and $\mathbf { V } \in \mathbb { C } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { t } } }$ are unitary matrices, and $\pmb { \Sigma } \in \mathbb { R } ^ { N _ { \mathrm { r } } \times N _ { \mathrm { t } } }$ is a diagonal matrix containing the singular values in descending order. For single-stream transmission, the BS selects the first column of V, corresponding to the largest singular value, as the precoding vector v [34].

![](images/725b5ffbbb5fd9730fb4035cdbd6e3b8044ce6eea30f5531a5a8a43af78f7971.jpg)  
(a) Simulation layout.

![](images/128edd39c91a7aa66549ed7c2cd35d353a1e3c7675b8d40f10617abc94792cc1.jpg)  
(b) Codeword fingerprint distribution pdf.  
Fig. 2. Sample frequency distribution of codeword fingerprints. (a) Layout of QuaDRiGa-simulated scenarios with UEs randomly distributed in 5 m × 5 m rectangle subregions. (b) Corresponding codeword fingerprint distributions pdf for LOS and NLOS conditions.

## B. DL-Based CSI Feedback

The DL-based CSI feedback framework adopts an autoencoder architecture. The input to the network is the precoding vector v, which is preprocessed by separating its real and imaginary parts and stacking them into a matrix of shape $( 2 , N _ { \mathrm { t } } )$ . The matrix is then normalized to the range [0, 1] and still denoted as v.

The autoencoder comprises an encoder enc(·) and a decoder dec(·), deployed at the UE and BS, respectively. The feedback bitstream is obtained through encoder compression followed by a uniform quantizer $f _ { \mathrm { Q } }$ with quantization level Q:

$$
{ \bf s } _ { \mathrm { Q } } = f _ { \mathrm { Q } } ( \mathtt { e n c } ( { \bf v } ; \Omega _ { \mathrm { e n c } } ) ) ,\tag{3}
$$

where $\Omega _ { \mathrm { e n c } }$ denotes the encoder network parameters. The $M -$ length bitstream $\mathbf { s } _ { \mathbf { Q } } .$ , with compression ratio $\mathrm { C R } = M / ( 2 Q \times$ $N _ { \mathrm { t } } )$ . At the BS, the decoder reconstructs the precoding vector:

$$
\hat { \mathbf { v } } = \mathsf { d e c } ( f _ { \mathrm { Q } } ^ { - 1 } ( \mathbf { s } _ { \mathrm { Q } } ) ; \boldsymbol { \Omega } _ { \mathrm { d e c } } ) ,\tag{4}
$$

where $\Omega _ { \mathrm { d e c } }$ are the decoder parameters.

To evaluate reconstruction fidelity, we adopt the squared generalized cosine similarity (SGCS) metric:

$$
\rho ^ { 2 } = \frac { 1 } { | \mathcal { D } | } \sum _ { { \bf v } \in \mathcal { D } } \left( \frac { | { \bf v } ^ { H } \hat { { \bf v } } | } { \| { \bf v } \| _ { 2 } \| \hat { { \bf v } } \| _ { 2 } } \right) ^ { 2 } .\tag{5}
$$

During end-to-end training, we minimize the cosine distance between the original and reconstructed precoding vectors:

$$
\mathcal { L } = \frac { 1 } { | \mathcal { D } | } \sum _ { \mathbf { v } \in \mathcal { D } } \left( 1 - \frac { | \mathbf { v } ^ { H } \hat { \mathbf { v } } | } { \| \mathbf { v } \| _ { 2 } \| \hat { \mathbf { v } } \| _ { 2 } } \right) ,\tag{6}
$$

where D denotes the training dataset and |D| its cardinality. This loss function is directly aligned with the SGCS metric, thereby ensuring consistency between the training objective and the evaluation criterion.

## C. Codeword Fingerprint

To characterize the statistical properties of CSI datasets, we adopt a codebook fingerprinting technique based on [35]. We utilize a discrete Fourier transform (DFT) codebook, which is well-suited to uniform linear arrays and captures long-term angular characteristics of wireless channels [36].

The DFT codebook of size $N _ { \mathrm { c o d e } } = 2 ^ { B }$ is defined as $\mathbf { W } =$ $\left[ \mathbf { w } _ { 0 } , \mathbf { w } _ { 1 } , \ldots , \mathbf { w } _ { N _ { \mathrm { c o d e } } - 1 } \right]$ , where the n-th codeword is given by:

$$
\mathbf { w } _ { n } = \frac { 1 } { \sqrt { N _ { \mathrm { t } } } } \left[ 1 , e ^ { j \frac { 2 \pi } { N _ { \mathrm { c o d e } } } n } , \dots , e ^ { j \frac { 2 \pi } { N _ { \mathrm { c o d e } } } ( N _ { \mathrm { t } } - 1 ) n } \right] ^ { T } .\tag{7}
$$

For a given precoding vector $\mathbf { v } ,$ its fingerprint in the angular domain is computed as:

$$
\mathbf { f } = \mathbf { v } ^ { H } \mathbf { W } = [ f _ { 1 } , f _ { 2 } , \dots , f _ { N _ { \mathrm { c o d e } } } ] .\tag{8}
$$

The index of the dominant beam direction is:

$$
\mathrm { i d x } = \operatorname { a r g m a x } _ { i \in \{ 1 , \dots , N _ { \mathrm { c o d e } } \} } | f _ { i } | .\tag{9}
$$

By aggregating the dominant directions over the dataset, we compute a histogram to approximate the angular probability density function (PDF):

$$
\begin{array} { r } { \mathbf p \mathbf d \mathbf f = \mathrm { h i s t } _ { \mathcal D } ( \mathrm { i d x } ) . } \end{array}\tag{10}
$$

This statistical distribution pdf reflects the angular spread of energy in the dataset, revealing environmental characteristics. In line-of-sight (LOS) cases, the dominant direction corresponds to the angle of departure (AoD) of the LOS path. In non-line-of-sight (NLOS) scenarios, it reflects the AoD of the strongest scattering path.

We illustrate this with CSI datasets generated using QuaDRiGa under the “3GPP-38.901-UMi-LOS/NLOS" settings. UEs are randomly distributed within 5 m × 5 m rectangle subregions, as shown in Fig. 2(a). The resulting pdf distributions are presented in Fig. 2(b), where LOS scenarios exhibit more concentrated angular energy, while NLOS cases show broader dispersion due to multipath scattering. The LOS case illustrates an extreme example where small sampling areas (5 m) and ULA configuration create concentrated angular distributions. This apparent correlation is an artifact of experimental design: expanding sampling areas yields diffuse distributions, different antenna arrays break the one-to-one mapping, and each dataset restarts obstacle layouts across independent maps. Unlike CSI-positioning with fixed environments, our framework prevents inferring the precise UE position from pdf.

## III. LEARNWARE FRAMEWORK

This section presents our Learnware-based CSI-feedback model repository. We first motivate the use of scene-specific small models and the integration of a centralized repository into the CSI-feedback workflow. We then outline the Learnware paradigm and its benefits for CSI-feedback applications.

## A. Motivation

Deploying AI-driven CSI-feedback models entails two primary challenges: (1) balancing generalization with per-scene performance, and (2) managing the computational and deployment costs of large models. Our Learnware Model Repository addresses these by enabling efficient reuse of scene-specific models identified via statistical specifications. We highlight three key insights: the efficacy of small models, the power of statistical specifications, and the accelerated fine-tuning convergence they afford.

1) Scene-Specific Small Models vs. General-Purpose Models: Training a single large-capacity model to generalize across diverse scenarios simplifies management but demands extensive compute, memory, and often yields suboptimal performance in specific environments. In contrast, scene-specific small models, trained on localized CSI data, capture finegrained propagation characteristics, require fewer resources, and consistently outperform general models within their target domains (see Table VI). Detailed experiments appear in Section V. This phenomenon is theoretically grounded in multi-task learning theory [37], [38]. A general model trained on heterogeneous scenes suffers from negative transfer: parameter sharing induces gradient interference across tasks, degrading performance when tasks are insufficiently related. Scene-specific models avoid this interference by dedicating capacity to a single distribution, yielding superior performance under capacity-limited regimes.

2) Statistical Specifications vs. Semantic Specifications: Accurate model retrieval requires more than coarse structural descriptors (e.g., network depth, codeword length). Semantic specifications alone cannot distinguish differences in data distributions due to environmental variations. Instead, we use statistical specifications, continuous descriptors of the trainingdata distribution (e.g., codeword fingerprint histograms), to index models in a continuous specification space. This approach generalizes to unseen environments and avoids the inefficiency of exhaustive validation.

3) Rapid Fine-Tuning Convergence: Retrieved models often require local fine-tuning. The closer a model's original training distribution is to the target domain, the fewer iterations and samples needed for convergence [39]. Fig. 3 illustrates convergence trajectories from (1) a mismatched model, (2) a general model, and (3) the nearest Learnware model. A densely populated repository ensures rapid adaptation or even direct deployment without further training.

## B. Learnware Paradigm and Specification Design

The motivations described in the previous subsection support our design of a Learnware-based repository that emphasizes scene-specific model reuse, retrieval using statistical specifications, and efficient adaptation through fine-tuning. Fig. 4 illustrates the framework of the Learnware-based repository, which consists mainly of two stages: submission and deployment. In the submission stage, developers package each model with its semantic and statistical specifications and upload it to the repository. In the deployment stage, users submit their task requirements, and the system matches semantic tags and locates the appropriate specification for model retrieval.

![](images/4acad323867b25544894137a200fb4c132868687308aec3ab6cc9c8c160397ff.jpg)  
Fig. 3. CSI feedback model solution space. Curves indicate convergence paths from (1) a mismatched model, (2) a general model, and (3) the closest Learnware model. TARIEI

TABLE I  
COMPARISON OF PARADIGMS
<table><tr><td>Paradigm</td><td>Resource requirements</td><td>Transmission cost</td><td>Lifelong learning</td><td>Robustness to forgetting</td><td>Data privacy</td></tr><tr><td>Aggregate datasets</td><td>High</td><td>High</td><td>×</td><td>×&gt;</td><td>×</td></tr><tr><td>Learnware Paradigm</td><td>Low</td><td>Low</td><td>√</td><td></td><td>√</td></tr></table>

The Learnware paradigm [29], [31] provides a marketplace of pre-trained models that users can reuse without sharing raw data. Each Learnware entry contains a model and its specification, allowing the system to match user requests while keeping each contributor's training data local. This paradigm is provided as a heterogeneous and evolving alternative to the traditional approach of training strong generalized models from aggregated datasets without collecting their training data. Instead of learning from a large centralized dataset [40], it accumulates knowledge through an ever-growing collection of distributed models. This model-centric approach supports on-demand access to task-relevant knowledge while avoiding catastrophic forgetting and the scalability limitations of data aggregation, as depicted in Tab. I. It also offers the future possibility of reassembling multiple models to address complicated or even unplanned tasks [29].

The remainder of this section describes the submission stage and the design of specifications. The deployment stage will be discussed in the next section.

1) Model Submission: The Learnware system supports large-scale operation, allowing many developers to contribute models, thus creating a rich, diverse repository. Future users submit requirements to the Learnware Dock system, which identifies or reassembles learnwares. Users can then reuse these models or refine them with their data. Although BSs are encouraged to contribute high-quality models voluntarily, most CSI-feedback models are trained centrally in an AI resource data center, which can maintain a broad collection of pre-trained models. Each model is stored as a Learnware entry that includes its specifications, describing functional characteristics, enabling efficient identification and retrieval without retaining the original training data.

![](images/e9231f354d292122fbd127b9951e6e0b57a2cbf1c0812f38667f521d59c6f7e4.jpg)  
Fig. 4. Learnware paradigm applied to CSI feedback neural networks.

2) Specification Design: Due to the diverse functionalities and training data distributions of Learnware models, the specification space is inherently complex and expansive. Without a well-structured design, users may struggle to locate suitable models efficiently. To address this, we introduce the concept of a specification world, where functionally similar models are grouped into specification islands. Within each island, models differ primarily in their training data distributions, referred to as the data statistical space.

Thus, the overall specification world can be viewed as a coupling of a discrete functional space and a continuous statistical space. For two models with identical functional specifications, a smaller difference in their training data distributions generally implies better generalization performance when transferred to similar environments. As shown in Fig. 4, users first identify the appropriate island based on semantic requirements, and then select the closest model by comparing statistical similarities between their local data and the Learnware training distribution.

Specifications are divided into two categories:

a) Semantic Specifications: These consist of string-based descriptors defining model functionality. They help identify the relevant specification island and cover attributes such as task type (e.g., full or implicit feedback, one-side or two-side network), model structure (input/output dimensions, codeword length), and system configurations (bandwidth, frequency, number of carriers, antenna array type). An illustrative example is provided in Tab. III.

b) Statistical Specifications: These characterize the distributional properties of the training data and help pinpoint a model's exact position within an island. Due to privacy concerns, the original training data are not shared. Instead, statistical specifications are generated offline at the model developer's side, and packaged with the model and semantic tags before uploading. In our framework, the codeword fingerprint distribution pdf, defined in Eq. 10, is used as the statistical specification. Cosine similarity is employed as the distance metric for retrieval.

3) Validation of Codeword Fingerprint PDF: We validate the effectiveness of the codeword fingerprint distribution pdf as a statistical specification. Following the dataset similarity evaluation framework established in [41], we investigate the correlation between dataset similarity and model generalization performance. The underlying principle is that if a distance metric meaningfully captures dataset characteristics, then datasets that are close in that metric space should yield similar model performance when models are transferred between them. This validation is essential for our Learnware retrieval mechanism, which relies on pdf similarity to select appropriate pre-trained models.

To this end, we use 144 scenario-specific simulated datasets (i.e., Learnwares), comprising 72 LOS and 72 NLOS environments. These datasets are arranged counterclockwise according to the azimuth angle relative to a central BS, ensuring full 360-degree coverage for both propagation conditions. Fig. 5(a) presents the Euclidean distance matrix of raw CSI vectors between dataset pairs. Each element (i, j) is calculated as:

$$
d _ { ( i , j ) } = \frac { 1 } { | \mathcal { D } _ { i } | \times | \mathcal { D } _ { j } | } \sum _ { \mathbf { v } ^ { ( i ) } \in \mathcal { D } _ { i } } \sum _ { \mathbf { v } ^ { ( j ) } \in \mathcal { D } _ { j } } \left\| \mathbf { v } ^ { ( i ) } - \mathbf { v } ^ { ( j ) } \right\| _ { 2 } .\tag{11}
$$

Fig. 5(b) shows the cosine similarity matrix between the pdf vectors of the datasets, where each element is given by:

$$
\mathrm { s i m } _ { ( i , j ) } = \frac { \vert \mathbf { p } \mathbf { d f } _ { i } ^ { H } \mathbf { p } \mathbf { d f } _ { j } \vert } { \vert \vert \mathbf { p } \mathbf { d f } _ { i } \vert \vert _ { 2 } \vert \vert \mathbf { p } \mathbf { d f } _ { j } \vert \vert _ { 2 } } .\tag{12}
$$

Fig. 5(c) displays the generalization performance matrix. The element $p _ { ( i , j ) }$ indicates the SGCS performance of a model trained on dataset $\mathcal { D } _ { i }$ and tested on $\mathcal { D } _ { j }$ . To ensure fairness, all values are normalized by the self-performance $p _ { \left( j , j \right) }$

A comparison of the three matrices reveals that Fig. 5(b) aligns more closely with Fig. 5(c) than does Fig. 5(a), suggesting that pdf is a more effective proxy for predicting model transferability than raw CSI distances. This confirms the suitability of pdf as a statistical specification for Learnware retrieval. Notably, the cosine similarity matrix in Fig. 5(b) exhibits greater sharpness than the performance matrix in Fig. 5(c). For example, the X-shaped spot in the upper-left corner of Fig. 5(c) is more diffuse than its counterpart in Fig. 5(b), indicating that pdf similarities serve as a reliable surrogate for functionality proximity. This X-pattern arises from the simulation configuration, in which the BS is a northsouth ULA and datasets are spatially distributed by angular sectors, yielding a periodic structure in beam directionality.

![](images/9385e1e431c80c00d6b1bd8a298499e9c38e88a5b560f8065694d663990efbb1.jpg)  
(a) Euclidean distance matrix of 144 datasets.

![](images/7a530e34ac247062842c699f3faba7639cd33d04023f0f83359d02049790b7df.jpg)  
(b) Cosine similarity matrix of pdf.

![](images/cfb241c42d8bbaef3159afabf47e5321ff2a64b77c0b890c4e112ebb85decd66.jpg)  
(c) Performance evaluation matrix.  
Fig. 5. Comparison between dataset-level distances and model generalization performance.

A closer analysis comparing LOS and NLOS subsets reveals that pdf is more sensitive to variations in LOS scenes, as evidenced by the different sharpness of the upper-left submatrix (LOS) and the lower-right submatrix (NLOS) in Fig. 5(b). This is expected since pdf in LOS conditions mainly captures the AoD of the direct path, which is sharply defined by the relative geometry between the UE and BS. In contrast, pdf in NLOS conditions reflects a broader scattering profile due to multipath propagation, resulting in a more diffuse distribution.

In the early stages of repository development, the diversity of available models may be limited. To mitigate this, initial configuration sets can be used to define semantic specification labels, thereby establishing a range of specification islands. Moreover, simulation-trained CSI models can be employed to populate the repository. As more models are added over time, both semantic and statistical specification coverage will naturally expand.

## IV. DEPLOYMENT STRATEGIES

This section describes the deployment stage of our Learnware framework, including the online retrieval mechanism, comparisons with alternative methods, and an optimized search strategy to accelerate model lookup.

## A. Deploying Stage

During deployment, the user-side BS first activates Model Management by monitoring for model mismatch. It then submits a model Delivery Request to the Learnware marketplace, which identifies and delivers the most suitable models. This process presents two main challenges: (1) effectively retrieving models that satisfy the user's task requirements; and (2) maximizing the utility of the returned models while minimizing adaptation overhead.

The baseline approach, denoted the Original Model Repository, uploads a subset of local CSI data as a validation set, evaluates every candidate model on this set, and returns the highest-performing model. Although this method achieves high retrieval accuracy, it suffers from several drawbacks:

it compromises data privacy and intellectual property, incurs substantial communication overhead, and scales poorly due to sequential model evaluations. These limitations render it impractical for real-time CSI deployment.

To overcome these issues, our Learnware system employs a specification space retrieval mechanism, as illustrated in Fig. 4. Local CSI data is only used in specification generation, and the specification is sent to the AI data center as the request. The semantic task specifications are used to locate the appropriate specification island. Within that island, statistical specifications are used to index a suitably similar model. Crucially, an exact statistical match is not required. As illustrated in Fig. 5, small discrepancies often result in highly interoperable models, enabling flexible and responsive retrieval. When multiple candidates share identical statistical specifications but differ in network architecture, the BS performs a lightweight ondevice evaluation. If the returned model meets the performance threshold, it is deployed directly; otherwise, the BS fine-tunes the model locally.

In the early stages of repository development, coverage of the specification space may be sparse, leading to suboptimal matches. Moreover, the BS may lack knowledge of the ideal local performance, making it difficult to assess the gap between the retrieved model and an optimally trained one. We address this by incorporating a fine-tuning step governed by a user-defined threshold, based on available samples, allowable training iterations, and resource constraints. Models that fail to improve after fine-tuning are discarded in favor of the original, while those that show improvement are adopted for deployment.

## B. Post-Retrieval Adaptation

Online deployment of CSI feedback models involves interactions among the AI data center, the BS, and the UE. Initially, the AI data center trains a two-sided neural network and deploys it to the BS for local adaptation. As illustrated in Fig. 6, when the BS detects performance degradation, it issues a model update request [42]. The UE collects new CSI samples and forwards them to the BS, which may either finetune its current model, incurring substantial computation and data requirements, or request a new pre-trained model whose training distribution more closely matches the current channel conditions.

![](images/c047317d361388501f8828a63faa831aee3dbdea4faf275d024f710a22a8237f.jpg)  
(a) Scheme with Original model repository.

![](images/da86c946668e92ae184ac966146b80ce1707fc65c618f9db90f4307cd4dc4152.jpg)  
(b) Scheme with Learnware model repository.  
Fig. 6. Online deployment schemes based on model repository, where the key differences are marked in orange.

TABLE II  
COMPARISON OF GENERAL MODEL, MODEL SWITCH, AND MODEL REPOSITORY-BASED SCHEMES ACROSS KEY DEPLOYMENT CONSIDERATIONS.
<table><tr><td colspan="2">Method</td><td>General Model</td><td>General Model with fine-tuning</td><td>Model Switch</td><td>Original Model Repository</td><td>Learnware Model Repository</td></tr><tr><td rowspan="4">Construction (Initialize)</td><td>AI data center</td><td colspan="2">Train one model using large-scale data</td><td rowspan="2">Train several models using large-scale data Store several models</td><td colspan="2">Train many models in parallel</td></tr><tr><td>BS</td><td colspan="2">Store a single model</td><td colspan="2">Store a single model</td></tr><tr><td>AI data center</td><td>None</td><td>None</td><td>None Switch model and</td><td>Model retrieval and return</td><td>Upload specifications,</td></tr><tr><td rowspan="2">Online deployment</td><td>BS</td><td>None</td><td rowspan="2">Fine-tune and transmit encoder NN</td><td rowspan="2">Upload local CSI samples, transmit encoder NN fine-tune, transmit encoder NN Upload local CSI samples to BS</td><td rowspan="2"></td><td rowspan="2">fine-tune, transmit encoder NN</td></tr><tr><td>UE</td><td>None</td></tr><tr><td>Merits</td><td>Accuracy</td><td>Low</td><td>High</td><td>Middle</td><td>High</td><td>High</td></tr><tr><td rowspan="2">and</td><td>Communication overhead</td><td>Zero</td><td>High</td><td>Low</td><td>Middle</td><td>Middle</td></tr><tr><td>Computation overhead</td><td>Zero</td><td>High</td><td>Low</td><td>High</td><td>Middle</td></tr><tr><td>Limitations</td><td>Data privacy</td><td>√</td><td></td><td></td><td>X</td><td></td></tr></table>

As illustrated in Fig.6(a), under the Original Model Repository, the BS must upload raw CSI data for exhaustive candidate evaluation, raising privacy and intellectual property concerns and introducing high latency. In contrast, the Learnware Model Repository requires only compact statistical specifications, reducing raw CSI exposure and significantly reducing retrieval time. Although this approach may slightly reduce selection accuracy, it enables near real-time updates. As illustrated in Fig.6(b), semantic specifications (e.g., antenna configuration and bandwidth) remain fixed during deployment, constraining the search to a single specification island. The BS then submits only the relevant statistical specification for rapid retrieval. For large repositories, the elastic search strategies presented in Section IV-D maintain fast and scalable performance.

## C. Comparison with Alternative Methods

We evaluate several baselines against our repository-based solutions:

• General Model: A single model trained on a large, diverse dataset and deployed directly to the BS.

• General Model with fine-tuning: The general model is fine-tuned at the BS to match local CSI conditions.

• Model Switch: Multiple pre-trained models with different configurations; the BS selects the best based on local performance.

• Original Model Repository: The BS uploads local data to the AI data center, which evaluates all candidates and returns the best for fine-tuning.

• Learnware Model Repository: The BS uploads statistical specifications; the AI data center retrieves the most appropriate pre-trained model based on semantic and statistical matching.

Table II summarizes these approaches across initialization and deployment phases, as well as trade-offs in accuracy communication cost, computation overhead, and data privacy. During initialization, all methods require substantial training the General Model and Model Switch focus on fewer, larger models, whereas repository approaches leverage many smaller models for parallelism and scalability. In deployment, General Model and Model Switch require no further AI data center interaction, while repository methods involve continuous communication. The Learnware Model Repository achieves a balanced trade-off, delivering high performance with moderate overhead and preserving user privacy, an equilibrium further validated by our experimental results.

## D. Adjustable-Speed Model Search Strategy

To support real-time CSI feedback deployment, we enhance the repository's internal structure to enable rapid and scalable retrieval. After locating the appropriate specification island using semantic attributes, brute-force computation of cosine similarities across all candidates remains impractical for large repositories. As demonstrated in Fig. 5, models with similar statistical specifications tend to be functionally interoperable, offering opportunities to accelerate retrieval. Therefore, we propose an adjustable-speed multi-level LSH strategy (Algorithm 1), which unifies coarse-grained anchor selection and hierarchical indexing into a K-level cascade with early termination capability.

Algorithm 1: Hierarchical LSH Retrieval   
Input: Repository $\mathcal { M } = \{ \mathbf { p } _ { i } \} _ { i = 1 } ^ { N } ( \mathbf { p } _ { i } ; $ normalized spec.);   
query q; levels K; LSH params $\{ ( k _ { \ell } , L _ { \ell } ) \} _ { \ell = 1 } ^ { \hat { K } }$ (ke:   
#hash bits, $L _ { \ell } { : }$ #tables); stop level kstop   
Output: Retrieved anchor model $\mathbf { \hat { m } ^ { * } } \in \mathcal { M }$   
// Offline: build K-level index.   
$\mathbf { W } _ { \ell } \in \mathbb { R } ^ { L _ { \ell } \times k _ { \ell } \times d } :$ random projection matrix;   
1 for $\ell = 1 , \ldots , K$ do   
2 Sample We with normalized rows;   
3 for $\hat { \mathbf { p } } _ { i } \in \mathcal { M } _ { \ell } , \ j = 1 , \ldots , L _ { \ell }$ do   
$\mathcal { T } _ { \ell . \ s } ^ { ( j ) } [ h _ { i } ^ { ( j ) } ] \gets \mathcal { T } _ { \ell } ^ { ( j ) } [ h _ { i } ^ { ( j ) } ] \cup \{ i \}$ with   
$\begin{array} { r } { \boldsymbol { h } _ { i } ^ { \left( j \right) } = \mathbf { P } \mathrm { a c k B i t s } ( \mathrm { s i g n } ( \mathbf { W } _ { \ell } ^ { \left( j \right) } \mathbf { p } _ { i } ) ) ; } \end{array}$   
4 for each bucket h with models B do   
$\begin{array} { r } { \mathcal { A } _ { \ell } ^ { ( j ) } [ h ]  \arg \operatorname* { m a x } _ { \mathbf { p } \in B } \mathbf { p } ^ { \top } \mathbf { c } \colon } \end{array}$ select model closest to   
centroid $\begin{array} { r } { \mathbf { c } = \frac { 1 } { | B | } \sum _ { \mathbf { p ^ { \prime } } \in B } \mathbf { p ^ { \prime } } ; } \end{array}$   
// Online: cascade.   
$h ^ { ( j ) } = \mathtt { P a c k B i t s } \big ( \mathtt { s i g n } ( \mathbf { W } _ { \ell } ^ { ( j ) } \mathbf { q } ) \big )$ : query   
signature;   
5 for $\mathbf { \bar { \ell } } = 1 , \dots , k _ { s t o p }$ do   
6 me ← arg ma $\mathbf { x } _ { j = 1 , \dots , L _ { \ell } } \mathbf { q } ^ { \top } \mathcal { A } _ { \ell } ^ { ( j ) } [ h ^ { ( j ) } ]$ // best   
anchor at level l   
7 if qTml $> \theta _ { \ell }$ (confidence threshold) then return me;   
8 return $\mathbf { m } ^ { * } \gets \mathbf { m } _ { k _ { s t o p } } ;$

The algorithm consists of two phases. Offline indexing builds K levels of LSH tables with increasing hash precision. For each level l, we generate $L _ { \ell }$ hash tables using $k _ { \ell } .$ -bit signatures $( k _ { 1 } < k _ { 2 } < \dots < k _ { K } , L _ { 1 } > L _ { 2 } > \dots > L _ { K } )$ . This configuration ensures that coarse levels use more tables for high recall to capture a broad candidate set, while finer levels use fewer tables with longer hash keys for precise filtering, balancing retrieval efficiency and accuracy. Each bucket precomputes an anchor model selected as the one closest to the centroid, enabling O(1) retrieval without enumerating all candidates. Online retrieval processes levels sequentially until a specified stopping level $k _ { \mathrm { s t o p } } .$ which can be determined by a latency budget $\tau \_ \mathrm { o r }$ confidence thresholds $\{ \theta _ { \ell } \}$ . At each level, the query specification q is hashed to retrieve $L _ { \ell }$ bucket anchors, and the best anchor $\mathbf { m } _ { \ell }$ is selected via cosine similarity. The system can terminate early if $\mathbf { q } ^ { T } \mathbf { m } _ { \ell } > \theta _ { \ell } .$ or proceed to finer levels for higher accuracy.

This design provides adjustable speed through two mechanisms: (i) level selection: stopping at coarse levels (small $k _ { \mathrm { s t o p } } )$ yields sub-millisecond latency suitable for urgent updates, while full cascade (large $k _ { \mathrm { s t o p } } )$ ensures near-optimal accuracy; (ii) confidence-based early exit: termination triggers when retrieved anchors already satisfy performance requirements, avoiding unnecessary computation. The per-level complexity is $O ( L _ { \ell } k _ { \ell } d { + } L _ { \ell } d )$ , independent of repository size N, enabling scalable deployment to billion-model repositories.

SEMANTIC SPECIFICATION LABELS USED IN THE EXPERIMENT.
<table><tr><td rowspan=1 colspan=1>CSI_Number_Domain</td><td rowspan=1 colspan=1>spatial domain</td></tr><tr><td rowspan=1 colspan=1>Feedback_Mode</td><td rowspan=1 colspan=1>implicit feedback</td></tr><tr><td rowspan=1 colspan=1>Structure</td><td rowspan=1 colspan=1>two-sided model</td></tr><tr><td rowspan=1 colspan=1>Input_Dimension</td><td rowspan=1 colspan=1>32 × 2</td></tr><tr><td rowspan=1 colspan=1>Output_Dimension</td><td rowspan=1 colspan=1>32 × 2</td></tr><tr><td rowspan=1 colspan=1>Compression_Ratio</td><td rowspan=1 colspan=1>1/16</td></tr><tr><td rowspan=1 colspan=1>Quantization_Bits</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>BS_Antenna_Array</td><td rowspan=1 colspan=1>ULA</td></tr><tr><td rowspan=1 colspan=1>Central_Frequency</td><td rowspan=1 colspan=1>2.6 GHz</td></tr><tr><td rowspan=1 colspan=1>Bandwidth</td><td rowspan=1 colspan=1>20 MHz</td></tr></table>

TABLE IV

BASIC PARAMETER SETTINGS USED IN THE SIMULATION.
<table><tr><td>Antenna setting</td><td>32 ULA antennas at BS with spacing of 0.5 wavelength, centered at (0,0) and arranged in a north-south direction (Y-axis) Single antenna at UE</td></tr><tr><td>Center frequency Bandwidth</td><td>2.6 GHz 20 MHz</td></tr><tr><td>Scene configuration</td><td>3GPP-38.901-UMi-LOS 3GPP-38.901-UMi-NLOS</td></tr><tr><td>Height</td><td>10 m for the BS 1.5 m for the UE</td></tr><tr><td>Position of sampling area</td><td>Randomly distributed in the 400 m × 400 m area centered relative to the BS</td></tr><tr><td>District range</td><td>Circle with a radius of 5 m for Learnware training set Rectangles of various lengths (5 m × [5,10, 20,30,40,50] m) for target test set</td></tr><tr><td>Sampling number per dataset Maps number / Learnware number</td><td>10,000 144</td></tr><tr><td>CSI Pretreatment</td><td>SVD Normalized to real values in the range [0, 1]</td></tr></table>

V. SIMULATION RESULTS AND DISCUSSIONS

## A. Learnware Model Repository Initiation

We initialized the Learnware model library using the implicit feedback CSI two-sided network as an example to construct a specification island. The corresponding semantic labels are shown in Tab. III.

1) CSI Simulation: Before collecting models, we generated simulation datasets using the configurations listed above to train Learnware models and enrich the specification island. We created 144 Learnwares using QuaDRiGa [43] under the 3GPP-38.901-UMi configuration (72 LOS + 72 NLOS, each with 10,000 samples). CSI sampling covered full 360° azimuth diversity with independent environment layouts per dataset (i.e., the environment layout was restarted for each dataset). Details are in Tab. IV and Fig. 2(a). The angular distribution characteristics are reflected in the pdf shown in Fig. 2(b).

2) NN Training and Evaluation: Each Learnware was trained using a lightweight CNN-based autoencoder (CsiNet [8] with 2D convolutions replaced by 1D, kernel size expanded from 3 to 7). The General Model follows the transformer-based EVCsiNet-T [44] (details in Tab. V), aligning with the 3GPP R18 AI/ML study [44], [45]. The dataset was partitioned into training (85%), validation (10%), and test (5%) sets. We used the Adam optimizer with a learning rate of 1e-4, training for 1,000 epochs and saving the model at the point of optimal validation performance.

For targeted testing, QuaDRiGa generated CSI data matching the semantic configurations. Test set sampling areas were rectangles of various lengths (5 m × [5, 10, 20, 30, 40, 50] m) with random centers and rotations under the 3GPP-38.901-UMi LOS/NLOS configuration. Our solution retrieves the best Learnware based on statistical specification similarity

TABLE V  
AUTOENCODER STRUCTURES FOR CNN AND TRANSFORMER-BASED CSI FEEDBACK
<table><tr><td>Layer</td><td>Type</td><td>Input</td><td>Output</td><td>Details</td></tr><tr><td colspan="5">CNN-based Autoencoder [8]</td></tr><tr><td>Input</td><td></td><td>(32, 2)</td><td>(32, 2)</td><td>Real and imaginary parts</td></tr><tr><td rowspan="4">Encoder</td><td>Conv1D</td><td>(32, 2)</td><td>(32, 2)</td><td>Kernel: 7 × 1, BN, ReLU</td></tr><tr><td>Reshape</td><td>(32, 2)</td><td>(64,)</td><td>Flatten</td></tr><tr><td>Dense+Sigmoid</td><td>(64,)</td><td>(4,)</td><td>Compression ratio C R = 1/16</td></tr><tr><td>Quantization</td><td>(4,)</td><td>(16,)</td><td>Q =4 bits, feedback_bits=16</td></tr><tr><td rowspan="6">Decoder</td><td>Dequantization</td><td>(16,)</td><td>(4,)</td><td>一</td></tr><tr><td>Dense</td><td>(4,)</td><td>(64,)</td><td></td></tr><tr><td>Reshape</td><td>(64,)</td><td>(32, 2)</td><td></td></tr><tr><td>RefineNet</td><td></td><td></td><td>Conv1D: (32, 2) → (32, 8) Conv1D: (32, 8) → (32, 16)</td></tr><tr><td>Block ×2</td><td>(32, 2)</td><td>(32, 2)</td><td>Conv1D: (32, 16) → (32, 2)</td></tr><tr><td>Conv1D</td><td>(32, 2)</td><td>(32, 2)</td><td>Skip Connection Kernel: 7 × 1, BN, ReLU</td></tr><tr><td colspan="5">Sigmoid (32, 2)</td></tr><tr><td colspan="5">Transformer-based Autoencoder [44]</td></tr><tr><td rowspan="2">Input</td><td>Input Linear Embedding</td><td>(32, 2) (32, 2)</td><td>(32, 2)</td><td>NToken = 32, feature_dim=2</td></tr><tr><td>Positional Encoding</td><td>(32, 81)</td><td>(32, 81) (32, 81)</td><td>Embedding_dim = 81, Dense Sinusoidal encoding</td></tr><tr><td rowspan="7">Encoder</td><td>Transformer Block ×LTF  $( L _ { \mathrm { T F } } = 1 2 )$ </td><td>(32, 81)</td><td>(32, 81)</td><td>Multi-Head Attn: Nhead = 3, Dropout(0.1) LayerNorm, Residual connection FFN: 324 → 81 (4×), ReLU, Dropout(0.1)</td></tr><tr><td></td><td>(32, 81)</td><td>(32, 81)</td><td>LayerNorm, Residual connection</td></tr><tr><td>LayerNorm Dense</td><td>(32, 81)</td><td>(32, 2)</td><td>Final normalization Project back to 2 channels</td></tr><tr><td>Flatten</td><td>(32, 2)</td><td>(64,)</td><td></td></tr><tr><td>Dense+Sigmoid</td><td>(64,)</td><td>(4,)</td><td> $C R = 1 / 1 6$ </td></tr><tr><td>Quantization</td><td>(4,)</td><td>(16,)</td><td>Q = 4 bits, feedback_bits=16</td></tr><tr><td>Dequantization</td><td></td><td></td><td>一</td></tr><tr><td rowspan="7">Decoder</td><td>Dense</td><td>(16,)</td><td>(4,) (64,)</td><td></td></tr><tr><td>Reshape</td><td>(4,) (64,)</td><td>(32, 2)</td><td></td></tr><tr><td>Linear Embedding</td><td>(32, 2)</td><td>(32, 81)</td><td>Dense(81), scale by √81</td></tr><tr><td>Positional Encoding</td><td>(32, 81)</td><td>(32, 81)</td><td>Sinusoidal encoding</td></tr><tr><td></td><td></td><td></td><td>Multi-Head Attn:  $\tilde { N } _ { \mathrm { h e a d } } = 3 ,$ </td></tr><tr><td>Transformer Block ×LTF (LTF = 12)</td><td>(32, 81)</td><td>(32, 81)</td><td>LayerNorm, Residual connection FFN: 324 → 81 (4×), ReLU, Dropout(0.1)</td></tr><tr><td>LayerNorm</td><td>(32, 81) (32, 81)</td><td>(32, 81)</td><td>LayerNorm, Residual connection Final normalization</td></tr></table>

and returns it for optional fine-tuning. We evaluated both the initial and fine-tuned performance.

## B. Performance Gain

Performance evaluations are conducted for the aforementioned schemes, all sharing the same architectural design but differing in their training processes:

• General Model (with/without fine-tuning): A transformer-based (architecture in Tab. V) model with high generalization capability is trained using a mixture of the 144 datasets from the training Learnwares.

• Model Switch: Four configuration-specific models are prepared with CNN-based architecture in Tab. V. The 144 datasets are divided into four groups based on LOS/NLOS conditions and the upper/lower sides of the ULA axis at the BS (north/south sides in Fig. 2(a), corresponding to the first/second halves in Fig. 2(b)). Each model is trained with a mix of 36 datasets from each group.

• Original Model Repository: The performance of the CNN-based 144 models in the repository is evaluated to select the best-performing model, which represents the upper bound of performance for model repository-based schemes.

• Learnware Model Repository: The codebook fingerprint sample frequency pdf is adopted as the statistical specification, with N set to 64. The model library computes the cosine similarity of pdf between all Learnwares within the specified island1 and the target scene test set, selecting the Learnware with the highest similarity.

• Upper Bound: With sufficient local data, smaller models often achieve better overfitting performance than generalized models, thus setting the performance upper bound. Due to the large sampling area in the test scenario, datasets ranging from 10,000 to 50,000 samples are used for training to obtain this upper bound.

1) Initial Performance Verification: We first evaluate performance without fine-tuning. Tab. VI first validates the strong performance of the scene-specific model compared to General Model, where the SGCS performance is the mean of the 10 cases included in each sampling area range. Specific CNN models (2k parameters, 0.11M FLOPs) are compared against transformer-based General Models with increasing complexity (TF-1 to TF-4, with millions of parameters). Results reveal that specific models consistently outperform all General Models across all scenarios, even when General Model capacity expands to 1,000× the parameter count and 5,600× the computational cost. This demonstrates that specific advantages persist over an extremely wide capacity range, with millionparameter models still failing to match small specialized models. The diminishing returns in General Model performance also highlight the inefficiency of simply scaling up model capacity. Moreover, deploying such large models imposes significant hardware demands and inference latency, making them impractical for resource-constrained edge devices, whereas lightweight scene-specific models enable efficient on-device deployment.

Tab. VII analyzes the sensitivity of all methods to different compression ratios and quantization bits. The Learnware Model Repository consistently outperforms the General Model across all configurations, with SGCS gains of 0.05– 0.45 (6.4%–128.2%) in LOS and 0.05–0.09 (12.4%–60.4%) in NLOS. Even under the most constrained feedback (CR=1/32, Q = 4), the Learnware achieves 0.794 (LOS) and 0.215 (NLOS), far exceeding the General Model (0.348 LOS, 0.134 NLOS), demonstrating robustness to feedback payload variations.

Fig. 7 displays representative individual cases randomly selected from the 10 LOS and 10 NLOS test scenarios with 5m × 5m sampling area. In each subfigure, the blue bars represent the SGCS performance of all 144 LOS Learnware models when tested on that specific LOS/NLOS test scenario, while orange bars represent NLOS Learnware models. The green-highlighted bar indicates the Learnware model selected by our codebook fingerprint method (i.e., the model whose training dataset's pdf has the highest cosine similarity with the test scenario's pdf). The horizontal dashed lines show the performance of five methods evaluated on the test scenario. The performance ranking consistently follows: Upper Bound > Original Model Repository > Learnware Model Repository > General Model/Model Switch. In both scenarios, repositorybased selections closely approach the Upper Bound, significantly outperforming General Model by approximately 18.8% (0.147/0.781) in LOS and 57.7% (0.169/0.293) in NLOS, demonstrating the effectiveness of pdf-based model selection across different propagation conditions. Extensive testing reveals that Learnware selection is robust to LOS/NLOS conditions: green bars (selected models) predominantly appear among blue bars (LOS Learnwares) for LOS test scenarios, and among orange bars (NLOS Learnwares) for NLOS scenarios. Hence, LOS/NLOS is excluded from semantic specifications.

TABLE VI  
AVERAGE SGCS PERFORMANCE OF SCENE-SPECIFIC SMALL MODELS AND LARGE GENERAL MODEL WITH VARIOUS COMPLEXITY. CR=1/16 AND FEEDBACK QUANTIZATION BITS $Q { = } 4$
<table><tr><td rowspan="2">Sample Range (m2)</td><td colspan="8">LOS</td><td colspan="8">NLOS</td></tr><tr><td>Scene-specific</td><td>TF-1</td><td>Learnw. Model  $\mathbf { R e p o s i t o r y }$ </td><td>TF-1</td><td>General Model (various complexity)1</td><td></td><td></td><td></td><td>Scene-specific</td><td></td><td>Learnw. Model Repository</td><td></td><td>General Model (various complexity)</td><td></td><td></td><td></td></tr><tr><td></td><td>CNN2 0.915</td><td>0.915</td><td></td><td>CNN 0.332</td><td></td><td>TF-2 0.725</td><td>TF-3 0.755</td><td>TF-43 0.777</td><td>CNN 0.350</td><td>TF-1 0.357</td><td>0.317</td><td>CNN</td><td>TF-1</td><td>TF-2</td><td>TF-3 0.214</td><td>TF-4 0.230</td></tr><tr><td>5× 5</td><td>0.881</td><td></td><td>0.889</td><td></td><td>0.628</td><td></td><td></td><td></td><td></td><td>0.343</td><td></td><td>0.181 0.225</td><td>0.208 0.213</td><td>0.207 0.220</td><td></td><td></td></tr><tr><td>5 × 10 5 × 20</td><td>0.879</td><td>0.883 0.886</td><td>0.855 0.808</td><td>0.340 0.297</td><td>0.701</td><td>0.719 0.710</td><td>0.746 0.713</td><td>0.715 0.715</td><td>0.347 0.330</td><td>0.347</td><td>0.321 0.314</td><td>0.162</td><td>0.205</td><td>0.205</td><td>0.245 0.216</td><td>0.263 0.221</td></tr><tr><td>5 × 30</td><td>0.868</td><td>0.881</td><td>0.721</td><td>0.292</td><td>0.703 0.651</td><td>0.684</td><td>0.691</td><td>0.660</td><td>0.324</td><td>0.339</td><td>0.303</td><td>0.187</td><td>0.213</td><td>0.207</td><td>0.214</td><td>0.213</td></tr><tr><td>5 × 40</td><td>0.848</td><td>0.858</td><td>0.703</td><td>0.290</td><td>0.521</td><td>0.579</td><td>0.598</td><td>0.613</td><td>0.334</td><td>0.350</td><td>0.285</td><td>0.189</td><td>0.217</td><td>0.219</td><td>0.230</td><td>0.238</td></tr><tr><td>5× 50</td><td>0.844</td><td>0.865</td><td>0.652</td><td>0.284</td><td>0.489</td><td>0.491</td><td>0.467</td><td>0.492</td><td>0.276</td><td>0.286</td><td>0.232</td><td>0.126</td><td>0.189</td><td>0.188</td><td>0.188</td><td>0.194</td></tr><tr><td>Parameters [103]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>58.66</td><td>424.29</td><td></td><td></td></tr><tr><td>Compare with CNN</td><td>2.01 1×</td><td>58.66 29×</td><td>2.01 1×</td><td>2.01 1×</td><td>58.66 29×</td><td>424.29 211×</td><td>1,069.56 532×</td><td>2,114.99 1,051×</td><td>2.01 1x</td><td>58.66 29×</td><td>2.01 1×</td><td>2.01 1×</td><td>29×</td><td>211×</td><td>1,069.56 532×</td><td>2,114.99 1,051×</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>12.64</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FLOPs [106] Compare with CNN</td><td>0.11 1×</td><td>12.64 115×</td><td>0.11 1×</td><td>0.11 1×</td><td>12.64 115×</td><td>104.03 945×</td><td>312.06 2,837×</td><td>624.09 5,673×</td><td>0.11 1×</td><td>115×</td><td>0.11 1×</td><td>0.11 1×</td><td>12.64 115×</td><td>104.03 945×</td><td>312.06 2,837×</td><td>624.09 5,673×</td></tr></table>

1 Architectures in Table V: CNN (same with Scene-specific and Learnware Model Repository); TF-1: Transformer with $L _ { \mathrm { T F } } = 2$ and $E m b e d d i n g \_ d i m = 2 7 ;$ TF-2: Transformer with $L _ { \mathrm { T F } } = 2$ and Embedding\_dim = 81; TF-3: Transformer with $\bar { L _ { \mathrm { T F } } } = 6$ and Embedding\_dim = 81; TF-4: Transformer with $L _ { \mathrm { T F } } = 1 .$ 2 and $E m b e d d i n g \_ { d i m } = 8 1$ 2 Thh1d h:\~h1ahtd CAN: datd f if da1 a thUL  
2 The bold-highlighted CNN is adopted for scene-specific models as the Upper Bound.  
3 The bold-highlighted TF-4 is adopted as the default NN architecture for the General Model in subsequent experiments.

TABLE VII  
AVERAGE SGCS PERFORMANCE OF METHODS WITH VARIOUS CR AND FEEDBACK QUANTIZATION BITS. SAMPLING RANGE: $5 \times 5 m ^ { 2 }$
<table><tr><td colspan="2">Configurations</td><td colspan="5">LOS scenes</td><td colspan="5">NLOS scenes</td></tr><tr><td>CR</td><td>Quantization bits (Q)</td><td>General Model</td><td>Model Switch</td><td>Learnware Model Repository</td><td>Original Model Repository</td><td>Upper Bound</td><td>General Model</td><td>Model Switch</td><td>Learnware Model Repository</td><td>Original Model Repository</td><td>Upper Bound</td></tr><tr><td rowspan="5">1/16</td><td>2345</td><td>0.681</td><td>0.371</td><td>0.832</td><td>0.841</td><td>0.878</td><td>0.219</td><td>0.197</td><td>0.296</td><td>0.309</td><td>0.336</td></tr><tr><td></td><td>0.715</td><td>0.453</td><td>0.850</td><td>0.859</td><td>0.881</td><td>0.230</td><td>0.222</td><td>0.313</td><td>0.328</td><td>0.356</td></tr><tr><td></td><td>0.777</td><td>0.477</td><td>0.889</td><td>0.891</td><td>0.915</td><td>0.230</td><td>0.230</td><td>0.317</td><td>0.333</td><td>0.350</td></tr><tr><td></td><td>0.781</td><td>0.482</td><td>0.899</td><td>0.907</td><td>0.921</td><td>0.232</td><td>0.233</td><td>0.320</td><td>0.335</td><td>0.356</td></tr><tr><td></td><td>0.348</td><td>0.249</td><td>0.794</td><td>0.816</td><td>0.852</td><td>0.134</td><td>0.144</td><td>0.215</td><td>0.239</td><td>0.251</td></tr><tr><td>1/32 1/16</td><td>4</td><td>0.777</td><td>0.477</td><td>0.889</td><td>0.891</td><td>0.915</td><td>0.230</td><td>0.230</td><td>0.317</td><td>0.333</td><td>0.350</td></tr><tr><td>1/8</td><td></td><td>0.843</td><td>0.681</td><td>0.897</td><td>0.906</td><td>0.920</td><td>0.420</td><td>0.385</td><td>0.472</td><td>0.489</td><td>0.510</td></tr></table>

![](images/6c3c8a59dab7dbbb12f33696009e73f54e4fc46907dd04a3820efcf00b1330f5.jpg)

![](images/6beb523ebf0496b0f235f02084431848e101e94d9a0a8adf7b9f4328155dd2ec.jpg)

![](images/151b0d99fccf031a6401b5565a2f73ffce69a10dceb08d58d2490591e0b18c25.jpg)

(a) Testing LOS scene.  
![](images/9ca350f89ee0169892a719cd2f7236b5cb9b19180ec08980af03384a8eaba289.jpg)  
(b) Testing NLOS scene.  
Fig. 7. SGCS performance without fine-tuning and codebook fingerprint cosine similarity. The LOS and NLOS target testing scenes are presented in (a) and (b). In each subfigure, the LOS and NLOS Learnwares are displayed in blue and orange, respectively. The Learnware selected by the codebook fingerprint (i.e., the one with the maximum cosine similarity) is marked in green. The CSI sampling area for the test set is set to 5 m × 5 m, the same as the Learnware training set. CR=1/16 and feedback quantization bits Q=4.

As shown in Fig. 7, it is hard to decide from a predefined threshold whether a Learnware requires fine-tuning, because the Upper Bound is relatively high for LOS test scenarios but relatively low for NLOS scenarios, making fine-tuning necessary in the latter. To improve Learnware performance in NLOS scenarios, one approach is to increase the feedback overhead by switching to a different specification island.

TABLE VIII  
SGCS PERFORMANCE BEFORE AND AFTER FINE-TUNING (500 SAMPLES, 100 EPOCHS) OF INDIVIDUAL REPRESENTATIVE CASE, CORRESPONDING TO FIG. 8.
<table><tr><td rowspan="2">Methods</td><td colspan="2">LOS Test Scene</td><td colspan="2">NLOS Test Scene</td></tr><tr><td>Without fine-tuning</td><td>With fine-tuning</td><td>Without fine-tuning</td><td>With fine-tuning</td></tr><tr><td>General Model</td><td>0.781</td><td>0.936</td><td>0.293</td><td>0.408</td></tr><tr><td>Model Switch</td><td>0.506</td><td>None</td><td>0.413</td><td>None</td></tr><tr><td>Original Model Repository</td><td>0.943</td><td>0.951</td><td>0.492</td><td>0.499</td></tr><tr><td>Learnware Model Repository</td><td>0.928</td><td>0.946</td><td>0.462</td><td>0.472</td></tr><tr><td>Upper Bound</td><td colspan="2">0.951</td><td colspan="2">0.507</td></tr></table>

![](images/b669b995f5dbac59c05d1c1d2fb715f4b9d666e32ca1384c77b6fb5fc264b479.jpg)  
(a) Testing LOS scene.  
(b) Testing NLOS scene.  
(c) Average sgcs performance.  
Fig. 8. Performance improvements on the two cases from Fig. 7 post-fine-tuning. The LOS and NLOS target test scenarios are shown in (a) and (b), respectively. In each subfigure, the upper and lower sections represent the number of samples used for fine-tuning and the number of fine-tuning epochs, respectively. Tested CSI sampling area: 5 m × 5 m. The corresponding fine-tuning results are summarized in Tab. VIII. (c) presents the average SGCS performance over multiple sampling areas, where each bar represents the average across 10 test scenarios. Fine-tuning overhead: 500 samples and 100 epochs.

2) Fine-Tuning Performance: The convergence and sample requirements for fine-tuning are depicted in Figs. 8(a) and 8(b) (same cases from Fig. 7), with detailed SGCS performance before and after fine-tuning provided in Tab. VIII. The results demonstrate that minimal fine-tuning of the selected Learnware can effectively boost performance, achieving parity with the Original model repository and, in some cases, even match the Upper bound. Learnware fine-tuning overhead is minimal, as it provides a better initialization point than General Model. For example, in the LOS Finetune-sample scenario, General Model requires 300 samples to match the initial performance of the Learnware after 100 fine-tuning epochs (indicated by the gray dashed lines), meaning that direct Learnware deployment saves at least 300 samples times 100 epochs of fine-tuning effort compared to General Model. In the NLOS scenario, the General Model needs up to 1000 samples over 100 epochs to catch up with the Learnware's initial performance. This demonstrates that direct Learnware deployment saves 300 to 1000 training samples (depending on propagation conditions) compared to fine-tuning the General Model from scratch, while achieving superior final performance after fine-tuning.

We also conducted experiments on multiple test scenarios across different sampling areas, each consisting of 10 LOS and 10 NLOS cases. The averaged performance is presented in Fig. 8(c). The fine-tuning overhead was set to 500 samples for 100 epochs, with yellow bars indicating the corresponding fine-tuning gain. The results confirm that repository-based approaches provide superior initialization, and in some cases, outperform General Model even after its fine-tuning. Additionally, the slower performance improvement observed for Learnware compared to General Model is due to its weaker generalization ability and its closer proximity to the upper bound. This proximity makes it more challenging to reach the optimal solution and slows down the gradient descent process.

## C. Adjustable-Speed LSH

We evaluate the proposed adjustable-speed multi-level LSH retrieval strategy, validating the accuracy-latency trade-offs and demonstrate the practical feasibility for real-time CSI feedback deployment. We create a large-scale specification space containing 100 million statistical specification vectors. First, we generate 1,000 CSI datasets using QuaDRiGa simulations under diverse urban microcell scenarios and compute their pdf. We then fit a Gaussian Mixture Model (GMM) with 50 components to these real pdf vectors and perform data augmentation by sampling 100 million normalized 64- dimensional specification vectors. The repository is organized into four LSH levels, with various L hash tables and k-bit signatures, yielding next-level's candidate set sizes of 6.25M, 195k, 3k, and 48 models. We test 1,000 query specifications (from the same GMM distribution) and measure average cosine similarity between retrieved anchors and queries (i.e., LSH score). The brute-force upper bound is computed similarly by exhaustive search over the entire repository (100 million vectors). Relative Accuracy is the ratio of LSH Score to Brute-force Upper Bound. Speedup Factor compares the cumulative LSH retrieval time against brute-force search (5,000 ms for 100 million vectors).

TABLE IX  
PERFORMANCE OF MULTI-LEVEL LSH WITH ADJUSTABLE SEARCH DEPTH
<table><tr><td>Level</td><td>k</td><td>L</td><td>Input Candidate Repository Size</td><td>LSH Score</td><td>Brute-force Upper Bound</td><td>Relative Accuracy</td><td>Per-Level Time (ms)</td><td>Cumulative Time (ms)</td><td>Speedup Factor</td></tr><tr><td>L1</td><td>4</td><td>25</td><td> $1 . 0 \times 1 0 ^ { 8 }$ </td><td>0.85</td><td>0.99</td><td>85.9%</td><td>0.25</td><td>0.25</td><td>20,000×</td></tr><tr><td>L2</td><td>5</td><td>18</td><td> $6 . 2 5 \times 1 0 ^ { 6 }$ </td><td>0.91</td><td>0.99</td><td>91.9%</td><td>0.35</td><td>0.60</td><td>8,333×</td></tr><tr><td>L3</td><td>6</td><td>12</td><td> $1 . 9 5 \times 1 0 ^ { 5 }$ </td><td>0.96</td><td>0.99</td><td>97.0%</td><td>0.30</td><td>0.90</td><td>5,556×</td></tr><tr><td>L4</td><td>7</td><td>6</td><td> $3 . 0 5 \times 1 0 ^ { 3 }$ </td><td>0.98</td><td>0.99</td><td>99.0%</td><td>0.20</td><td>1.10</td><td>4,545×</td></tr><tr><td>Final</td><td>-</td><td>-</td><td>48</td><td>0.99</td><td>0.99</td><td>100%</td><td>0.05</td><td>1.15</td><td>4,348×</td></tr></table>

TABLE X  
ONLINE DEPLOYMENT COST COMPARISON FOR THE GENERAL MODEL, MODEL SWITCH, AND PROPOSED MODEL REPOSITORY-BASED SCHEMES.
<table><tr><td rowspan="2">Deployment considerations</td><td colspan="2">Communication overhead</td><td colspan="2">Time complexity</td><td colspan="2">Communication overhead [KB]1</td><td colspan="2">Time complexity [s]2</td></tr><tr><td>Data center-BS</td><td>BS-UE</td><td>Fine-tuning</td><td>Model retrieval</td><td>Data center-BS</td><td>BS-UE</td><td>Fine-tuning</td><td>Model retrieval</td></tr><tr><td>General Model</td><td>None</td><td>None</td><td>None</td><td>None</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>General Model + fine-tuning</td><td>None</td><td>500 samples for fine-tuning; return 1 encoder</td><td>100 epochs, 500 samples</td><td>None</td><td>0</td><td> $2 4 0 + 3 . 3 7 \times 1 0 ^ { 4 }$ </td><td>373.5</td><td>0</td></tr><tr><td>Model Switch</td><td>None</td><td>500 samples for evaluation; return 1 encoder</td><td>None</td><td>Evaluate 4 models</td><td>0</td><td> $2 4 0 + 5 . 8$ </td><td>0</td><td>0.71</td></tr><tr><td>Original</td><td>500 samples for evaluation;</td><td>500 samples for fine-tuning;</td><td>100 epochs,</td><td>Evaluate 144 models</td><td> $2 4 0 + 3 1 . 7$ </td><td> $2 4 0 + 5 . 8$ </td><td>61.6</td><td>25.7</td></tr><tr><td>model repository Learnware</td><td>return 1 model Specification;</td><td>return 1 encoder 500 samples for fine-tuning;</td><td>500 samples 100 epochs,</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>model repository</td><td>return 1 model</td><td>return 1 encoder</td><td>500 samples</td><td>Retrieve from 144 models</td><td> $2 . 6 + 3 1 . 7$ </td><td> $2 4 0 + 5 . 8$ </td><td>61.6</td><td>1.48e-5</td></tr></table>

1 File sizes: 500 samples = 240 KB; CNN autoencoder = 31.7 KB; CNN encoder = 5.83 KB; TF-4 autoencoder = 67.5 MB; TF-4 encoder = 33.7 MB; semantic specification = 0.25 KB; statistical specification = 2.36 KB.  
2 All measurements are obtained using a Tesla V100 GPU. Fine-tuning 500 samples for 100 epochs averaged 373.5 seconds for TF-4 Transformer and 61.6 for CNN. Model evaluation took 0.178 seconds per model. Cosine similarity computation for all 144 Learnwares averaged 1.48e-5 seconds.

Tab. IX summarizes the performance of our multi-level LSH strategy across different search depths. Deeper search levels yield higher accuracy but incur greater cumulative latency, enabling users to select the appropriate trade-off based on their requirements. Shallower levels (L1-L2) provide ultra-fast retrieval (0.25–0.60 ms) with good accuracy (85.9–91.9%), while deeper levels (L3-L4) deliver near-optimal accuracy (97.0–99.0%) at moderately higher latencies (0.90–1.15 ms). All cumulative latencies (0.25-1.15 ms) remain well within the typical CSI feedback cycles of 10–80 ms in 5G NR systems [46], confirming the practical feasibility of our approach.

## D. Complexity Analysis

In Tab. X, we quantitatively compare the online deployment costs of General Model, Model Switch, and the proposed model repository-based schemes, including communication overhead and time complexity. Given a fine-tuning budget of 500 samples over 100 epochs, and with model performance evaluation also requiring 500 samples, communication overhead is categorized into two segments: from the AI data center to the BS, and from the BS to the UE. Only the model repository-based schemes incur communication costs in the former.

The Learnware system addresses the Original model repository's drawback of transmitting local data by significantly reducing transmission overhead from 240 KB to 2.6 KB. Notably, the TF-4 General Model exhibits substantially higher communication costs: its encoder size reaches 33.7 MB, three orders of magnitude larger than the CNN encoder (5.83 KB), making frequent model updates impractical for bandwidthconstrained scenarios.

Time complexity is critical for online deployment and is divided into fine-tuning time and model retrieval time. Finetuning the TF-4 Transformer requires 373.5 seconds, over six times longer than the CNN-based model (61.6 seconds), due to its massive parameter count (2.1 million). This becomes a significant disadvantage for the General Model + fine-tuning scheme, as the system must fall back to traditional feedback during this extended period, potentially degrading feedback quality. In contrast, the Learnware model repository requires only 1.48e-5 seconds for retrieval, which corresponds to the cost of computing inner products and is thus negligible. Even when scaled to billion-model repositories, our adjustable-speed multi-level LSH (Algorithm 1) maintains retrieval latency within 0.25-1.15 ms (Tab. IX), well below the 10–80 ms CSI feedback cycles specified in 3GPP standards [46]. The near-instantaneous response ensures that during fine-tuning, the Learnware model can be deployed immediately while the locally fine-tuned model trains in the background. This scalability, combined with the minimal communication overhead of statistical specifications, positions the Learnware framework as a practical solution for real-world deployments where both bandwidth and latency are constrained.

## E. Further Discussion

1) Validation for pdf Selection: To validate the accuracy of codebook fingerprint pdf selection, we analyzed both the ranking of the Learnware selected by pdf and its performance gap compared to the top-performing model, as illustrated in Fig. 9. We simulated 1,000 LOS and 1,000 NLOS test scenarios for each sampling area, with each scenario containing 500 samples (used to generate pdf and to evaluate performance). This extensive simulation ensures robust probability estimations.

Fig. 9(a) shows the probability that the test performance of the pdf-selected Learnware ranks among the top x positions in the model repository, which consists of 144 models. In over 90% of cases, the ranking falls within positions 25 to 27. Additionally, the probability of ranking in the top three is 45% to 55% for LOS scenarios and 25% to 35% for NLOS scenarios.

A more intuitive indicator is the SGCS performance gap between the pdf-selected Learnware and the top-performing model $( \mathrm { S G C S _ { O M R } - S G C S _ { L M R } } )$ . Fig. 9(b) presents a boxplot of the performance gap distribution. The lower boundary, median line, and upper boundary of each box represent the first quartile (Q1), median (Q2), and third quartile (Q3), respectively. The box length reflects the interquartile range of the data. In LOS scenarios, the shorter boxes indicate a more concentrated performance gap distribution, which becomes more dispersed as the sampling area increases. Conversely, in NLOS scenarios, the distribution becomes more concentrated with larger sampling areas, since the performance degradation of the top-performing model offsets the impact of less accurate pdf selection. The median line is consistently biased towards the lower end of the box, indicating that the performance gap is generally small. The whiskers represent the range of the data and help identify potential outliers. The results show that even in outlier cases, the performance gap does not exceed 0.5.

![](images/20e4a6e4db0bfc0d83e751a9d96a0a3e1d00b202cfc85d4a4ce371e1e40803c1.jpg)

![](images/9b0b36b5df1ca05ef1040c0a2c2133064690c03958cbd1234da2bc72aeb38b86.jpg)

![](images/600353d1cbd665778ea70ad4ad24b5b6266d5fc2569606c1d08eb2f0d2795252.jpg)

![](images/cb6032895e44851b3c7e4231f62d6896dec0e7d86a1ec57a8e6c47a06bd20968.jpg)  
(a) Performance ranking.

![](images/572c9d4c0e3bd6067dcf61aa78032958641923edfc50128c9f170bafe3237e0f.jpg)  
(b) Performance gap boxplot.

![](images/23445c87684fb7fb1793a7175c531e5c97d2bfc46cf036ac04069b1d829d3733.jpg)  
(c) Probability of meeting performance threshold  
Fig. 9. We simulated 1,000 LOS and 1,000 NLOS scenarios in each sampling area, with each scenario comprising 500 samples. (a) Performance ranking of the learnware selected by pdf within the model repository (comprising 144 models), i.e., the probability of being ranked in the top x positions; (b) box-plot of the performance gap between the pdf-selected learnware and the top-performing model (i.e., Original model repository); (c) Selection accuracy: probability that the performance gap of the pdf-selected learnware meets the expected threshold.

To quantitatively assess selection accuracy, we define it as the probability that the performance gap falls within a threshold €:

$$
\operatorname { A c c } _ { \mathrm { s e l e c t } } ( \epsilon ) = \operatorname* { P r } \left( \left| \operatorname { S G C S } _ { \mathrm { O M R } } - \operatorname { S G C S } _ { \mathrm { L M R } } \right| \le \epsilon \right) ,\tag{13}
$$

Fig. 9(c) presents selection accuracy with $\epsilon \ = \ 0 . 1$ . The light blue area indicates acceptable cases $( \mathrm { g a p } \leq 0 . 1 )$ , while the light red area indicates failure $( \mathrm { g a p } > 0 . 1 )$ . The results demonstrate approximately 95% accuracy for LOS and 90% for NLOS scenarios, significantly exceeding random selection.

2) Impact of Learnware Quantity: We further investigate the impact of the Learnware model library size on performance. As shown in Fig. 10(a), we randomly selected varying numbers of Learnwares from a pool of 144 to form model libraries of different sizes. This process was repeated 100 times to reduce the effects of randomness. The mean performance across 10 test scenarios was then computed for each sampling region. The last column in each group of bars represents the performance of the best Learnware among the 144, corresponding to the Original model repository.

The experimental results indicate that as the model library size increases, the performance of the selected Learnwares also improves. However, this improvement exhibits diminishing marginal returns, suggesting that the model library size should be kept within a reasonable range to balance performance with retrieval efficiency. Additionally, as the sampling region of the test scenarios expands, the performance gap between the Original model repository and the selected Learnwares increases, particularly in NLOS scenarios. This is attributed to the more dispersed angular distribution of the datasets, which reduces the effectiveness of pdf as a statistical specification criterion. Therefore, when deploying the system, the BS should carefully define local scene boundaries to avoid excessively dispersed angular distributions.

3) Impact of Sample Size on Model Selection Performance: We investigate how the number of CSI samples affects pdf estimation accuracy and subsequent model selection performance. In practical deployment, BSs may have limited samples for statistical specification generation, making this analysis essential for understanding real-world applicability.

Fig. 10(b) illustrates the impact of sample size on both PDF similarity and SGCS performance. The blue curves with error bars represent PDF similarity, the green curves show average SGCS, and the red dashed lines indicate the benchmark SGCS performance. The results demonstrate that the Learnware Model Repository model exhibits strong robustness to sample size variations, with performance degradation negligible across all tested scenarios. For LOS scenes, characterized by concentrated angular energy, even 25 samples yield near-perfect PDF similarity (> 0.99), and the SGCS performance stabilizes at the benchmark level, confirming that sample size has minimal impact on model selection in such environments. For NLOS scenes, while PDF estimation quality improves progressively with sample size due to diffuse multipath scattering, the impact on SGCS performance remains remarkably small. Remarkably, with only 50 samples, the SGCS already exceeds 98% of the benchmark, and with 500 samples, the performance gap becomes virtually indistinguishable. These findings confirm that BSs can collect as few as 500 samples to obtain reliable model selection performance regardless of propagation conditions, eliminating the need for extensive data collection in practical deployments.

![](images/a75dc973fd6eb0e76336f4e51f491eaed29a6335d3e5fbb80fe1c70882a991cf.jpg)  
(a) Learnware model repository performance with (b) Sample size impacts for pdf and SGCS in Learn- (c) SGCS performance without fine-tuning in various numbers of learnwares. ware Model Repository. OFDM scenario.

Fig. 10. (a) Average SGCS performance of Learnware Model repository, with each group of 10 cases. CR=1/16 and feedback quantization bits Q=4; (b) Test scenarios: 10 cases for LOS/NLOS, 5 m × 5 m sampling range, For each sample size, the estimated pdf is compared against the benchmark (full 10,000 samples), Performance loss is defined as $\mathrm { S G C S } _ { \mathrm { b e n c h m a r k } } - \mathrm { S } \mathrm { \bar { G } C S } _ { \mathrm { o t h e r } }$ PDF similarity measures the cosine similarity between the estimated pdf and the benchmark pdf. (c) Multi-carrier OFDM scenario without fine-tuning. The configuration and legend follow the same format as Fig. 7.  
TABLE XI  
SIMILARITY AND DISTANCE METRICS FOR PROBABILITY DISTRIBUTIONS
<table><tr><td>Metric</td><td colspan="2">Original Formula</td><td>Original Range</td><td>Normalized Similarity</td><td>Complexity1</td></tr><tr><td>Cosine Similarity</td><td> $s _ { \mathrm { c o s } } ( p , q ) = { \frac { \sum _ { i } p _ { i } q _ { i } } { \| p \| \cdot \| q \| } }$ </td><td></td><td>[0, 1]</td><td> $S _ { \mathrm { c o s } } = s _ { \mathrm { c o s } }$ </td><td>Low</td></tr><tr><td>Bhattacharyya Coefficient [47]</td><td> $s _ { \mathrm { B h a t t } } ( p , q ) = \sum _ { i } { \sqrt { p _ { i } q _ { i } } }$ </td><td></td><td>[0, 1]</td><td> $S _ { \mathrm { B h a t t } } = s _ { \mathrm { B h a t t } }$ </td><td>Medium</td></tr><tr><td>Hellinger Distance [48]</td><td></td><td> $d _ { \mathrm { H e l l } } ( p , q ) = \frac { 1 } { \sqrt { 2 } } \left. \sqrt { p } - \sqrt { q } \right. _ { 2 }$ </td><td>[0, 1]</td><td> $S _ { \mathrm { H e l l } } = 1 - d _ { \mathrm { H e l l } }$ </td><td>Medium</td></tr><tr><td>Total Variation</td><td></td><td> $d _ { \mathrm { T V } } ( p , q ) = \frac { 1 } { 2 } \sum _ { i } \left| p _ { i } - q _ { i } \right|$ </td><td>[0, 1]</td><td> $S _ { \mathrm { T V } } = 1 - d _ { \mathrm { T V } }$ </td><td>Low</td></tr><tr><td>Jensen-Shannon Divergence [49]</td><td></td><td> $d _ { \mathrm { J S } } ( p , q ) = \frac { 1 } { 2 } D _ { \mathrm { K L } } ( p \Vert m ) + \frac { 1 } { 2 } D _ { \mathrm { K L } } ( q \Vert m )$ </td><td>[0, 1]</td><td> $S _ { \mathrm { J S } } = 1 - d _ { \mathrm { J S } }$ </td><td>High</td></tr><tr><td>KL Divergence [50]</td><td></td><td> $d _ { \mathrm { K L } } ( p , q ) = \sum _ { i } p _ { i } \log { \frac { p _ { i } } { q _ { i } } }$ </td><td>[0, +∞)</td><td> $S _ { \mathrm { K L } } = \exp ( - d _ { \mathrm { K L } } )$ </td><td>High</td></tr><tr><td>Euclidean Distance</td><td></td><td> $d _ { \mathrm { E u c } } ( p , q ) = \sqrt { \sum _ { i } ( p _ { i } - q _ { i } ) ^ { 2 } }$ </td><td>[0,√2]</td><td> $\begin{array} { r } { S _ { \mathrm { E u c } } = 1 - \frac { d _ { \mathrm { E u c } } } { \sqrt { 2 } } } \end{array}$ </td><td>Low</td></tr></table>

1 Complexity levels: Low (basic arithmetic operations), Medium (element-wise square roots), High (element-wise logarithmic or exponential operations).

4) Comparison of Distributional Similarity Metrics: Given that codebook fingerprints are represented as probability distributions, selecting an appropriate similarity metric is critical for accurate learnware retrieval. We systematically evaluate seven metrics: Cosine similarity, Bhattacharyya coefficient [47], Hellinger distance [48], Total Variation, KL divergence [50], Jensen-Shannon divergence [49], and Euclidean distance (details in Tab. XI). The retrieval performance was measured by average SGCS performance of selected models across various scenarios (6 LOS and 6 NLOS sampling ranges, each with 10 cases), with reference to the Original Model Repository (Top-1 performance, upper bound for Model Repository).

TABLE XII  
AVERAGE SGCS PERFORMANCE OF LEARNWARE MODEL REPOSITORY WITH VARIOUS DISTRIBUTIONAL SIMILARITY METRICS. CR=1/16 AND FEEDBACK QUANTIZATION BITS $Q { = } 4 .$
<table><tr><td>Similarity Metrics</td><td colspan="6">LOS</td><td colspan="6">NLOS</td></tr><tr><td>Sampling Range  $( \mathrm { m } ^ { 2 } )$ </td><td> $5 \times 5$ </td><td> $5 \times 1 0$ </td><td> $5 \times 2 0$ </td><td>5 × 30</td><td> $5 \times 4 0$ </td><td> $5 \times 5 0$ </td><td> $5 \times 5$ </td><td> $5 \times 1 0$ </td><td> $5 \times 2 0$ </td><td> $5 \times 3 0$ </td><td> $5 \times 4 0$ </td><td> $5 \times 5 0$ </td></tr><tr><td>Cosine  $( s _ { \mathrm { c o s } } )$ </td><td>0.886</td><td>0.855</td><td>0.839</td><td>0.721</td><td>0.659</td><td>0.599</td><td>0.324</td><td>0.318</td><td>0.319</td><td>0.303</td><td>0.285</td><td>0.233</td></tr><tr><td>Bhattacharyya  $\boldsymbol { [ 4 7 ] } ~ ( s _ { \mathrm { B h a t t } } )$ </td><td>0.883</td><td>0.857</td><td>0.855</td><td>0.746</td><td>0.666</td><td>0.612</td><td>0.327</td><td>0.309</td><td>0.312</td><td>0.287</td><td>0.266</td><td>0.209</td></tr><tr><td>Hellinger [48]  $( S _ { \mathrm { H e l l } } )$ </td><td>0.883</td><td>0.857</td><td>0.855</td><td>0.746</td><td>0.666</td><td>0.612</td><td>0.327</td><td>0.309</td><td>0.312</td><td>0.287</td><td>0.266</td><td>0.209</td></tr><tr><td>Total Variation  $( S _ { \mathrm { T V } } )$ </td><td>0.884</td><td>0.856</td><td>0.854</td><td>0.721</td><td>0.669</td><td>0.625</td><td>0.324</td><td>0.307</td><td>0.302</td><td>0.300</td><td>0.283</td><td>0.226</td></tr><tr><td>KL [50]  $( S _ { \mathrm { K L } } )$ </td><td>0.881</td><td>0.847</td><td>0.855</td><td>0.786</td><td>0.719</td><td>0.657</td><td>0.327</td><td>0.296</td><td>0.318</td><td>0.286</td><td>0.270</td><td>0.209</td></tr><tr><td>Jensen-Shannon  $\left[ 4 9 \right] ( S \mathbf { J } \mathbf { S } )$ </td><td>0.883</td><td>0.857</td><td>0.854</td><td>0.746</td><td>0.666</td><td>0.612</td><td>0.327</td><td>0.309</td><td>0.312</td><td>0.292</td><td>0.289</td><td>0.209</td></tr><tr><td>Euclidean  $( S _ { \mathrm { E u c } } )$ </td><td>0.873</td><td>0.856</td><td>0.759</td><td>0.678</td><td>0.634</td><td>0.528</td><td>0.325</td><td>0.316</td><td>0.315</td><td>0.299</td><td>0.278</td><td>0.217</td></tr><tr><td>Original Model Repository</td><td>0.891</td><td>0.864</td><td>0.855</td><td>0.786</td><td>0.737</td><td>0.667</td><td>0.346</td><td>0.333</td><td>0.330</td><td>0.317</td><td>0.309</td><td>0.261</td></tr><tr><td>Upper Bound</td><td>0.915</td><td>0.881</td><td>0.903</td><td>0.868</td><td>0.844</td><td>0.848</td><td>0.372</td><td>0.356</td><td>0.347</td><td>0.324</td><td>0.334</td><td>0.277</td></tr></table>

Tab. XII reveals distinct performance patterns. KL divergence achieves optimal accuracy in LOS large-range environments, matching the repository upper bound. However, it suffers catastrophic failure in NLOS diffuse conditions (0.209 at $5 \times 5 0$ versus 0.261 upper bound). In contrast, Cosine similarity maintains stable performance across all NLOS scenarios (0.324–0.233) without extreme degradation. This contrast stems from fundamental differences in handling lowmagnitude diffuse distributions. KL divergence's logarithmic form $\sum p _ { i } \log ( p _ { i } / q _ { i } )$ amplifies relative errors when absolute probabilities are small (characteristic of NLOS multipath scattering), causing similar diffuse patterns to be misjudged as dissimilar. Cosine similarity's L2-normalization cos $\theta \ =$ $\frac { \mathbf { p } ^ { T } \mathbf { q } } { \| \mathbf { p } \| \| \mathbf { q } \| }$ eliminates sensitivity to absolute energy concentration, focusing on relative angular patterns rather than absolute power levels.

We select Cosine similarity for learnware retrieval due to its robust performance across both LOS concentrated and NLOS scattered distributions, combined with computational efficiency (Low complexity versus High for KL), enabling real-time retrieval in large-scale repositories.

5) Multi-carrier OFDM Scenario: To verify the extensibility of the proposed framework beyond single-carrier systems, we conduct additional experiments on a multi-carrier OFDM scenario with explicit CSI feedback, where the 512-subcarrier channel is transformed via 2D-DFT to the angle-delay domain and pruned to the first 32 delay taps; the real-valued and normalized version of $\mathbf { H } \in \mathbb { C } ^ { 3 2 \times 3 2 }$ is then fed into the autoencoder. Following the same design principle as the angulardomain codebook fingerprint, we construct a dual-domain statistical specification $\mathbf { \bar { p } d f } _ { \mathrm { j o i n t } } = [ \mathbf { p d f } _ { \mathrm { a n g l e } } ; \mathbf { p d f } _ { \mathrm { d e l a y } } ] \in \mathbb { R } ^ { 6 4 }$ Specifically, for each channel sample, we extract the angular domain by projecting each subcarrier's channel vector onto the DFT codebook with $B = 5$ and taking a majority vote across all subcarriers to obtain the dominant angular index $\mathrm { i d x _ { a n g l e } , }$ which filters out subcarrier-level fluctuations and identifies the path consistently observed across the entire bandwidth. For the delay domain, since H is already in the delay domain, we directly average the energy across antennas to obtain the power delay profile, from which the dominant delay index $\mathrm { i d x _ { d e l a y } }$ is identified. Aggregating these indices across the dataset yields $\mathbf { p d f } _ { \mathrm { a n g l e } } = \mathrm { h i s t } _ { \mathcal { D } } ( \mathrm { i d x } _ { \mathrm { a n g l e } } )$ and $\mathbf { p d f } _ { \mathrm { d e l a y } } = \mathrm { h i s t } _ { \mathcal { D } } ( \mathrm { i d x } _ { \mathrm { d e l a y } } )$ We construct a repository of 200 Learnware models (100 $\mathrm { L O S } ~ + ~ 1 0 0$ NLOS) using TF-1 architecture $\begin{array} { r } { ( L _ { \mathrm { T F } } ~ = ~ 2 . } \end{array}$ feature $d i m { = } 6 4 ,$ Embedding $. d i m = 6 4 )$ , each trained on 10,000 samples. A TF-4 $( L _ { \mathrm { T F } } \ = \ 1 2 , $ Embedding\_dim = 192) General Model is trained on a mixture of 2 million samples. Fig. 10(c) presents the performance on representative LOS/NLOS test cases $\mathrm { ( 5 m \times 5 m ) }$ . The selected Learnware consistently outperforms the General Model and approaches the Original Model Repository, confirming the framework's effectiveness in multi-carrier OFDM systems.

## VI. CONCLUSION

This paper proposes a Learnware-based framework for intelligent deployment of CSI feedback models in wireless communication systems. Hosted in a centralized AI data center, the framework supports model reuse across diverse deployment scenarios, reducing training overhead and improving adaptability. To address key limitations of conventional model repositories, such as data privacy concerns, communication costs, and inefficient retrieval, the proposed approach introduces semantic and statistical specifications to enable accurate and lightweight model selection. The Learnware Model Repository eliminates the need for transmitting raw CSI data and exhaustive model evaluations by leveraging codebook fingerprint distributions for retrieval. An adjustable-speed retrieval strategy ensures scalability, while fine-tuning based on a user-defined threshold enables efficient model adaptation. Extensive simulations verify that the Learnware model repository consistently outperforms baseline methods. Compared to the general model with fine-tuning, the Learnware approach achieves higher SGCS performance, with representative gains of 18.8% in LOS and 57.7% in NLOS scenarios, while reducing fine-tuning overhead by 300–1000 samples (depending on propagation conditions) and 100 training epochs. Furthermore, hierarchical LSH model retrieval and complexity analysis confirms that the retrieval latency is negligible, validating the repository's practicality for real-time deployment.

## REFERENCES

[1] ITU-R WP5D, “Framework and overall objectives of the future development of IMT for 2030 and beyond," Radio Communication Division of the International Telecommunication Union (ITU-R), Tech. Rep., Jun. 2023. [Online]. Available: https://www.itu.int/md/ R19-WP5D-230612-TD-0905/en

[2] S. Dang, O. Amin, B. Shihada, and M.-S. Alouini, "What should 6G be?" Nature Electron., vol. 3, no. 1, pp. 20–29, Jan. 2020.

[3] A. Alhammadi, I. Shayea, A. A. El-Saleh, M. H. Azmi, Z. H. Ismail, L. Kouhalvandi, and S. A. Saad, “Artificial intelligence in 6G wireless networks: Opportunities, applications, and challenges," Int. J. Intell. Syst., vol. 2024, no. 1, p. 8845070, Mar. 2024.

[4] T. L. Marzetta, "Massive MIMO: an introduction," Bell Labs Tech. J., vol. 20, pp. 11–22, Mar. 2015.

[5] M. E. Eltayeb, T. Y. Al-Naffouri, and H. R. Bahrami, "Compressive sensing for feedback reduction in MIMO broadcast channels," IEEE Trans. Commun., vol. 62, no. 9, pp. 3209–3222, Sept. 2014.

[6] M. S. Rahman, Y.-H. Nam, J. Zhang, and J.-Y. Seol, "Linear combination codebook based CSI feedback scheme for FD-MIMO systems," in Proc. 2015 IEEE Globecom Workshops (GC Wkshps), Feb. 2015, pp. 1–6.

[7] J. Guo, C.-K. Wen, S. Jin, and G. Y. Li, “Overview of deep learningbased CSI feedback in massive MIMO systems," IEEE Trans. Commun., vol. 70, no. 12, pp. 8017–8045, Dec. 2022.

[8] C.-K. Wen, W.-T. Shih, and S. Jin, "Deep learning for massive MIMO CSI feedback," IEEE Wirel. Commun. Lett., vol. 7, no. 5, pp. 748–751, Oct. 2018.

[9] Y. An, S. Lu, H. Cai, and Z. Ji, “A deep learning-based approach to lightweight CSI feedback," Phys. Commun., vol. 68, p. 102538, Feb. 2025.

[10] S. Jiang and A. Alkhateeb, "Digital twin aided massive MIMO: CSI compression and feedback," in Proc. ICC 2024-IEEE Int. Conf. Commun. IEEE, Aug. 2024, pp. 3586–3591.

[11] J. Chen, W. J. Hillery, K. R. Mestav, C. Nuzman, I. Saniee, and Y. Xing, "CSI compression for massive MIMO: Model-based or data-driven?" IEEE Wirel. Commun., vol. 32, no. 1, pp. 22–27, Feb. 2025.

[12] X. Liang, Z. Jia, X. Gu, and L. Zhang, “Toward better low-rate deep learning-based CSI feedback: A test channel-based approach," IEEE Trans. Wirel. Commun., vol. 23, no. 8, pp. 8773–8786, Aug. 2024.

[13] J. Shin, Y. Kang, and Y.-S. Jeon, "Vector quantization for deep-learningbased CSI feedback in massive MIMO systems,"IEEE Wirel. Commun. Lett., vol. 13, no. 9, pp. 2382–2386, Sep. 2024.

[14] Y.-C. Lin, T.-S. Lee, and Z. Ding, “A scalable deep learning framework for dynamic CSI feedback with variable antenna port numbers," IEEE Trans. Wirel. Commun., vol. 23, no. 4, pp. 3102–3116, Apr. 2024.

[15] Z. Peng, R. Liu, Z. Li, C. Pan, and J. Wang, "Deep learning-based CSI feedback for XL-MIMO systems in the near-field domain," IEEE Wirel. Commun. Lett., vol. 13, no. 12, pp. 3613–3617, Dec. 2024.

[16] 3GPP TR 38.843, “Study on artificial intelligence (AI)/machine learning (ML) for NR air interface," 3rd Generation Partnership Project (3GPP), Tech. Rep., Dec. 2023. [Online]. Available: https: //ftp.3gpp.org//Specs/archive/38\_series/38.843/38843-200.zip

[17] X. Lin, “A tale of two mobile generations: 5G-Advanced and 6G in 3GPP release 20," IEEE Commun. Stand. Mag., pp. 1–9, Oct. 2025.

[18] J. Hoydis, F. A. Aoudia, A. Valcarce, and H. Viswanathan, "Toward a 6G AI-native air interface," IEEE Commun. Mag., vol. 59, no. 5, pp. 76–81, May 2021.

[19] N. Ye, S. Miao, J. Pan, Q. Ouyang, X. Li, and X. Hou, "Artificial intelligence for wireless physical-layer technologies (AI4PHY): A comprehensive survey," IEEE Trans. Cognit. Commun. Netw., vol. 10, no. 3, pp. 729–755, Jun. 2024.

[20] C.-H. Huang, C.-K. Wen, and G. Y. Li, “AI/ML life cycle management for interoperable AI native RAN," arXiv preprint arXiv:2507.18538, jul. 2025.

[21] J. Guo, Y. Cui, C.-K. Wen, and S. Jin, “"Prompt-enabled large AI models for CSI feedback," IEEE J. Sel. Areas Commun., vol. 44, pp. 2654–2668, 2026.

[22] S. Alikhani, G. Charan, and A. Alkhateeb, “"Large wireless model (LWM): A foundation model for wireless channels," arXiv preprint arXiv:2411.08872, 2024.

[23] M. Chen, M. Liu, Z. Zhang, Z. Xu, and L. Wang, "Task-oriented semantic communication with foundation models," China Commun., vol. 21, no. 7, pp. 65–77, Jul. 2024.

[24] Y. Li, G. Li, Z. Wen, S. Han, S. Gao, G. Liu, and J. Wang, "Channel modeling aided dataset generation for AI-enabled CSI feedback: Advances, challenges, and solutions," IEEE Commun. Stand. Mag., vol. 8, no. 4, pp. 72–78, Dec. 2024.

[25] M. Inoue, T. Ohtsuki, K. Yamamoto, and G. Gui, "Evaluation of source data selection for DTL based CSI feedback method in FDD massive MIMO systems," in Proc. 2023 IEEE 20th Consumer Commun. Netw. Conf. (CCNC), Mar. 2023, pp. 182–187.

[26] J. Sun, Y. Zhang, G. Gui, H. Zhao, H. Gacanin, and H. Sari, "Interacting federated and transfer learning-aided CSI prediction for intelligent

cellular networks," IEEE Trans. Veh. Technol., vol. 72, no. 12, pp. 15 776–15 787, Dec. 2023.

[27] Y. Zhang and A. Alkhateeb, “Zone-specific CSI feedback for massive MIMO: A situation-aware deep learning approach," IEEE Wirel. Commun. Lett., vol. 13, no. 12, pp. 3320–3324, Dec. 2024.

[28] J. Liu, M. Bouazizi, and T. Ohtsuki, "Source channel model selection for downlink CSI feedback estimation using transfer learning in massive MIMO system," in Proc. 2025 Int. Conf. Comput., Netw. Commun. (ICNC), May 2025, pp. 276–281.

[29] Z.-H. Zhou, “Learnware: on the future of machine learning." Frontiers Comput. Sci., vol. 10, no. 4, pp. 589–590, Aug. 2016.

[30] Y.-J. Zhang, Y.-H. Yan, P. Zhao, and Z.-H. Zhou, “Towards enabling learnware to handle unseen jobs," in Proc. AAAI Conf. Artificial Intelligence, vol. 35, no. 12, May 2021, pp. 10 964–10 972.

[31] Z.-H. Zhou and Z.-H. Tan, "Learnware: Small models do big," Sci. China Inf. Sci., vol. 67, no. 1, p. 112102, Oct. 2024.

[32] Z.-H. Tan, J.-D. Liu, X.-D. Bi, P. Tan, Q.-C. Zheng, H.-T. Liu, Y. Xie, X.-C. Zou, Y. Yu, and Z.-H. Zhou, "Beimingwu: A learnware dock system," in Proc. 30th ACM SIGKDD Conf. Knowledge Discovery Data Mining, Aug. 2024, pp. 5773–5782.

[33] H.-Y. Lei, Z.-H. Tan, and Z.-H. Zhou, "On the ability of developers' training data preservation of learnware," in Proc. 38th Advances Neural Inf. Process. Syst. (NeurIPS), vol. 37. Curran Associates, Inc., Dec. 2024, pp. 36471–36 513.

[34] M. Chen, J. Guo, C.-K. Wen, S. Jin, G. Y. Li, and A. Yang, "Deep learning-based implicit CSI feedback in massive MIMO," IEEE Trans. Commun., vol. 70, no. 2, pp. 935–950, Feb. 2022.

[35] M. Baur, N. Turan, S. Wallner, and W. Utschick, "Evaluation metrics and methods for generative models in the wireless PHY layer," IEEE Trans. Machine Learning Commun. Netw., vol. 3, pp. 677–689, May 2025.

[36] J. Suh, C. Kim, W. Sung, J. So, and S. W. Heo, "Construction of a generalized DFT codebook using channel-adaptive parameters," IEEE Commun. Lett., vol. 21, no. 1, pp. 196–199, Jan. 2017.

[37] J. Baxter, “A model of inductive bias learning," J. Artif. Intell. Res. (JAIR), vol. 12, pp. 149–198, Mar. 2000.

[38] A. Maurer, M. Pontil, and B. Romera-Paredes, “The benefit of multitask representation learning," J. Artif. Intell. Res. (JAIR), vol. 17, no. 81, pp. 1–32, 2016.

[39] K. Dwivedi and G. Roig, "Representation similarity analysis for efficient task taxonomy & transfer learning," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.(CVPR), 2019, pp. 12 387–12 396.

[40] J. Guo, Y. Cui, S. Jin, and J. Zhang, "Large AI models for wireless physical layer," IEEE Commun. Mag., vol. 64, no. 5, pp. 148–155, May 2026.

[41] J. Morais, S. Alikhani, A. Malhotra, S. Hamidi-Rad, and A. Alkhateeb, “A dataset similarity evaluation framework for wireless communications and sensing," in 2024 58th Asilomar Conf. Signals Syst. Comput., 2024, pp. 1144–1149.

[42] R. Ahmed, E. Visotsky, and T. Wild, “Explicit CSI feedback design for 5G new radio phase II," in Proc. 22nd Int. ITG Workshop Smart Antennas (WSA), Jun. 2018, pp. 1–5.

[43] S. Jaeckel, L. Raschkowski, K. Börner, and L. Thiele, "QuaDRiGa: A 3- D multi-cell channel model with time evolution for enabling virtual field trials," IEEE Trans. Antennas Propag., vol. 62, no. 6, pp. 3242–3256, Jun. 2014.

[44] H. Xiao, Z. Wang, D. Li, W. Tian, X. Liu, W. Liu, S. Jin, J. Shen, Z. Zhang, and N. Yang, “AI enlightens wireless communication: A transformer backbone for CSI feedback," China Commun., vol. 21, no. 12, pp. 243–256, Dec. 2024.

[45] 3GPP, "New SI: Study on artificial intelligence (AI)/machine learning (ML) for NR air interface," 3GPP, Technical Report RP-213599, December 2021, 3GPP TSG RAN Meeting #94e, Electronic Meeting.

[46] “Technical Specification Group Radio Access Network; NR; Physical layer procedures for data (Release 17)," 3rd Generation Partnership Project (3GPP), Technical Specification 3GPP TS 38.214 V17.0.0, Mar. 2022.

[47] A. Bhattacharyya, “On a measure of divergence between two statistical populations defined by their probability distributions," Bull. Calcutta Math. Soc., vol. 35, pp. 99–110, 1943.

[48] E. Hellinger, “Neue Begründung der Theorie quadratischer Formen von unendlichvielen Veränderlichen," Crelle's J., vol. 136, pp. 210–271, 1909.

[49] J. Lin, “"Divergence measures based on the Shannon entropy," IEEE Trans. Inf. Theory, vol. 37, no. 1, pp. 145–151, 1991.

[50] S. Kullback and R. A. Leibler, "On information and sufficiency," Ann. Math. Statist., vol. 22, no. 1, pp. 79–86, 1951.
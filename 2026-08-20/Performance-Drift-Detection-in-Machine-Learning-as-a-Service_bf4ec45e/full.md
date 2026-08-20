# Performance Drift Detection in Machine Learning as a Service (MLaaS) for IoT Environments

Deepak Kanneganti , Sajib Mistry , Sheik Mohammad Mostakim Fattah , Erik Elmroth , Aneesh Krishna , and Monowar Bhuyan, Senior Member, IEEE

Abstract—Machine Learning as a Service (MLaaS) is a powerful cloud paradigm enabling data-driven intelligent applications in Internet of Things (IoT) environments, widely adopted across healthcare, smart homes, and industry due to its costeffectiveness. However, the dynamic nature of IoT frequently alters data distributions, affecting MLaaS stability, while periodic MLaaS updates further introduce performance drift. Unlike traditional ML systems, MLaaS clients operate as black-box users without access to internal data or parameters, making drift detection particularly challenging. To address this, we propose a novel MLaaS Performance Drift Detection framework for IoT environments. The framework first employs an MLaaS extraction model that learns service behavior from input–output pairs and identifies prediction-influenced features. Building on this, the proposed MLaaS Performance Drift Detection (MPDD) model jointly captures variations in input data and MLaaS behavior. We further design an Adaptive-Temporal Performance Drift Detection Mechanism (APDDM) that dynamically adjusts monitoring frequency based on behavioral and data variations, enabling timely drift detection for effective service management. Extensive experiments on real-world datasets demonstrate that MPDD achieves up to 22–25% accuracy improvement over baseline drift detection methods. APDDM provides an average accuracy gain of approximately 4% and reduces the miss detection rate by around 9% compared to fixed-interval monitoring.

Index Terms—Machine Learning as a Service, IoT, Performance Drift, Drift Detection, Model Monitoring

## I. INTRODUCTION

Machine Learning as a Service (MLaaS) is a popular cloudbased service that provides scalable infrastructure and tools to build, train, and deploy ML models [1]. These services eliminate the complexity of managing hardware and manually designing ML models. MLaaS is typically accessed through web interfaces, application programming interfaces (APIs), or software development kits (SDKs). Major providers such as Microsoft Azure<sup>1</sup>, AWS SageMaker<sup>2</sup>, and OpenAI ChatGPT API<sup>3</sup> offer comprehensive MLaaS solutions. MLaaS services are generally categorized as platform-based MLaaS (PMLaaS) for centralized training and storage (e.g., AWS SageMaker<sup>2</sup>), and inference-based MLaaS (IMLaaS) for local prediction using pretrained models (e.g., Google Activity Recognition<sup>4</sup>).

Internet of Things (IoT) companies often rely on IMLaaS to add intelligent capabilities without training models inhouse [2]. For example, Medtronic, a healthcare IoT provider, integrates IMLaaS from IBM IQ cast into its application to predict low blood sugar events [3]. Clients typically evaluate these services using ground truth data from their environments to assess functional attributes (e.g., model and data specifications) and quality-of-service metrics (e.g., efficiency and scalability). However, the performance of an IMLaaS model is not static and may change over time. This occurs mainly due to two factors: 1) the dynamic nature ofIoT environments, where data patterns evolve [4], and 2) MLaaS updates, where providers periodically update services beyond client control [3]. As a result, clients may experience performance drift over time.

Performance drift refers to the gradual degradation of the predictive ability of IMLaaS over time, reflected in metrics such as accuracy, precision, or F1 score [5]. It occurs when the underlying relationship between input data X and output labels y changes, a phenomenon known as concept drift [5], [6]. IoT environments are highly dynamic due to evolving user behavior, sensor distributions, and system requirements, making it difficult for clients to maintain the reliability of MLaaS services. For example, an IoT environment using Google’s Activity Recognition API for healthcare monitoring may face reduced accuracy as patient age, health condition, or mobility alter sensor data over time, leading to performance drift. Additionally, MLaaS providers periodically update or retrain their models, which may further contribute to performance drift. Hence, IoT environments need to collect ground truth periodically to monitor MLaaS performance. However, collecting ground truth data requires substantial time and effort, making it costly and often impractical. The key challenge is the timely detection of performance drift without relying on continuous ground truth collection. To address this, we propose an MLaaS performance drift detection model that operates without requiring continuous ground truth data.

Traditional ML drift detection methods typically rely on monitoring error-related measures (e.g., accuracy) [5]–[8]. These approaches assume timely access to ground-truth labels, which is often not feasible in real-world deployments. Consequently, label-free drift detection techniques focus on monitoring internal model parameters, training data characteristics, or input data distributions to identify shifts. For example, methods such as DRIFTLENS [9] and Type-LDD [10] analyze training data properties and internal model parameters (e.g., weights and gradients) to infer distributional changes, while other approaches rely solely on observable input data and use statistical tests or discriminative models to detect shifts in input data distributions [11]–[13]. However, these techniques may not be directly applied in MLaaS settings due to the black-box nature of MLaaS, where service providers do not expose internal parameters or training data because of security and proprietary constraints. Moreover, approaches that rely only on changes in input data characteristics mainly capture distributional variation and cannot determine whether such changes actually alter the predictive behavior of the deployed MLaaS service under evolving IoT streams. As a result, the relationship between changing IoT data and MLaaS output behavior remains unmodeled. Therefore, IoT environments require an effective mechanism that explicitly captures the interaction between input data evolution and observable MLaaS outputs to accurately identify performance drift and support reliable service management. To the best of our knowledge, performance drift detection in MLaaS remains an underexplored problem. We identify two key challenges in detecting performance drift in MLaaS for IoT environments:

• How to detect the MLaaS performance drift without ground truth?: The dynamic nature of IoT environments can cause changes in input data patterns, potentially affecting MLaaS performance [14], [15]. IoT environments require continuous collection of ground-truth data to evaluate these services, which is time-consuming and often impractical. Existing studies often analyze statistical properties of input data distributions to detect drifts but distributional changes do not necessarily imply a performance change in MLaaS [9]. We consider this phenomenon as pseudo drift (see Definition 3). In addition, MLaaS providers periodically update or retrain their models, whereas IoT environments lack access to internal training data, feature distributions, or parameter updates. This highlights the need for mechanisms that jointly capture input data changes and MLaaS behavior to distinguish performance drift without relying on ground truth.

• How frequently should the MLaaS be re-evaluated?: The second critical challenge is determining the appropriate time period for monitoring MLaaS. Since MLaaS performance may degrade due to changes in input data distribution and MLaaS behavior, using fixed or inappropriate intervals can lead to oversensitivity (short intervals) or under-sensitivity (long intervals) in detecting performance drift. For example, if the MLaaS is evaluated every five minutes, the drift detection technique may continuously generate pseudo-drift notifications, reflecting an oversensitive setting that incurs unnecessary computational cost and resource consumption. Conversely, if evaluation occurs only every few hours, the system may fail to capture timely variations and detect real drift too late. This highlights the need for an adaptive timevariable mechanism that can adjust its evaluation period to detect performance drift and support effective service management decisions in IoT environments.

To address this challenge, we propose an MLaaS performance drift detection framework that identifies performance degradation without continuously collecting ground truth. The framework is built upon an MLaaS Performance Drift Detection Model (MPDD), which determines when the MLaaS should be re-evaluated to detect potential drift. First, we design an MLaaS Extraction Model that leverages IoT input data and MLaaS outputs to mimic MLaaS behavior and identify key input features without requiring access to black-box MLaaS parameters or training data. Next, we design two complementary scores, the Frechet Data Drift Score (FDDS) ´ and the MLaaS-Aware Drift Exposure Score (MDES), to capture changes in input data and their impact on MLaaS behavior. We then propose the MLaaS Performance Drift Detection Model, which captures the relationship between input data and MLaaS behavior, enabling detection of both real and pseudo drift without relying on ground-truth labels. Finally, we propose an Adaptive-Temporal Performance Drift Detection Mechanism (APDDM) that dynamically adjusts the evaluation interval to mitigate oversensitivity and undersensitivity, supporting effective service management decisions in IoT environments. Our contributions are summarized as follows:

• We propose an MPDD framework designed for IoT environments that identifies performance degradation in black-box MLaaS models without relying on ground truth data.

• We develop an MLaaS Extraction Model to approximate MLaaS behavior using input–output pairs.

• We propose a novel MPDD model that jointly captures the relationship between input data and MLaaS behavior, allowing accurate identification of performance drift.

• We propose an APDDM that dynamically adjusts evaluation intervals to balance oversensitivity and undersensitivity.

## II. PRIOR WORK

## A. Error-Rate-Based Drift Detection

ML services deployed in IoT environments often suffer from performance degradation over time due to changes in the underlying data-generating process, commonly referred to as concept drift [16]. Prior research in IoT data stream learning has proposed several drift detection techniques that mainly rely on observable error-rate signals [5], [7], [17]. Classical error-monitoring detectors, such as Drift Detection Method (DDM) and Early Drift Detection Method (EDDM) [5], along with adaptive extensions including Reactive Drift Detection Method (RDDM) and Adaptive Windowing (ADWIN) [8], [18], identify drift by monitoring variations in prediction errors within sliding or adaptive windows. More advanced variants, including Hoeffding-based approaches such as HDDM and FHDDM [6], [19], improve sensitivity through statistical testing based on Hoeffding bounds [7]. Recent studies have also explored drift-aware analytics in IoT streaming environments. Yang and Shami [20] proposed a lightweight IoT anomaly detection framework using sliding-window monitoring and dynamic retraining for evolving sensor and network traffic streams. Similarly, FedConD [21] introduced a federated IoT sensor learning framework that detects drift by monitoring historical prediction behavior from distributed local models. Despite addressing IoT-oriented streaming scenarios, these approaches still depend on observable prediction errors and continuous access to ground-truth labels [5]. Such assumptions are often impractical in real-world IoT deployments due to the cost, delay, and difficulty of obtaining reliable labeled feedback from distributed sensor devices [22], [23].

## B. Black-Box, Label-Independent Drift Detection Methods

Input data and distribution-based detection methods focus on analyzing changes in the input data stream rather than output errors. These techniques rely on statistical divergence measures such as Maximum Mean Discrepancy (MMD), Kolmogorov–Smirnov [24], and Frechet (Wasserstein-2) dis-´ tance [25] to identify distributional shifts. MMD-based approaches compare data segments over time but often require large observation windows, limiting responsiveness in blackbox environments [26]. Similarly, DRIFTLENS [9] detects drift using training data representations and internal model parameters for unsupervised latent feature analysis, yet its effectiveness depends on access to model internals and training information, which is typically unavailable in service-oriented settings. The Type-LDD framework [10] formulates drift detection using a multi-task sharing-loss function pre-trained on synthetic data to capture drift timing and type, but assumes availability of training data and semantic knowledge aligned with the model structure.

Recent studies have also explored label-independent drift handling in IoT and edge environments under limited observability. For example, [27] detected concept drift by analyzing IoT traffic flow behavior and classification score distributions without requiring true labels, while [28] combined KSWINbased drift detection with unsupervised anomaly monitoring over IoT traffic streams. Similarly, A-Detection [29] identified drift using reliability data streams collected from edge and IoT services. However, these approaches assume access to historical traffic distributions or reference data that are typically unavailable to clients. Existing black-box drift detection methods mainly focus on input-stream distributional changes rather than behavioral performance variations of deployed services. D3 [11] detects drift by training a discriminative classifier to separate historical and recent input samples, while SCSD [12] models drift detection as a sequential classifier two-sample test to identify covariate shift through calibrated confidence intervals between training and live inputs. The STUDD framework [13] adopts a student–teacher paradigm, where drift is inferred by monitoring imitation loss as the student replicates teacher predictions under evolving inputs. Although these methods operate under limited observability, they primarily capture input-space variations rather than behavioral performance changes of deployed ML services. In MLaaS environments, clients only observe input–output interactions without access to training data or internal model states [8], [18], [23]. This limitation can lead to pseudo drift, where input distribution changes do not necessarily indicate actual service degradation. This gap motivates the need for methods that directly relate observable inputs to MLaaS behavioral performance, which we address through a performance drift detection framework designed for black-box MLaaS settings.

## III. MOTIVATION SCENARIO

Let us consider an IoT healthcare environment that aims to integrate Human Activity Recognition (HAR) capabilities to monitor and analyze patient activities such as walking, sitting, and sleeping. However, the environment lacks the expertise and infrastructure required to develop, train, and maintain an HAR model. Therefore, the environment leverages an inference-based MLaaS provider to obtain HAR predictions. Major cloud platforms such as Google Cloud offer pre-trained activity recognition models that can be deployed without inhouse ML expertise. In practice, the IoT environment submits collected sensor data to the provider in periodic batches through batch inference channels such as Google Cloud Vertex AI Batch Prediction<sup>5</sup> or Azure Machine Learning batch endpoints<sup>6</sup>. Fig. 1 illustrates the motivating scenario of MLaaS integration within the IoT.

Let us consider that the IoT environment is subject to changes influenced by user behavior, device configurations, and environmental factors. As a result, the current IoT environment may no longer reflect the conditions assumed during the original deployment of the MLaaS service. This phenomenon may lead to a noticeable degradation in performance reflected in metrics such as accuracy, precision, and recall. For instance, in healthcare, variations in patient demographics (e.g., pediatric to elderly), health conditions (e.g., recovery stages or mobility impairments), or sensor placements (e.g., wrist versus ankle) can alter sensor data and reduce prediction accuracy from 93% to 68%. Hence, the IoT environment requires a performance drift detection strategy for effective service management, enabling informed service-level decision making such as notifying the MLaaS provider or switching to a more suitable service.

The primary challenge is to detect performance drift in MLaaS without the ground truth. IoT environments are dynamic and subject to changes influenced by user behavior, device configurations, and environmental factors. However, a change in input data distribution does not necessarily indicate performance drift, as some MLaaS models may be robust enough to handle such variations, leading to pseudo drift. Therefore, detecting input changes alone does not solve the problem of identifying real performance drift. Moreover, MLaaS providers’ behavior must also be considered. Due to security restrictions, clients lack access to internal parameters or training data, making MLaaS a black-box model. This highlights the need for an effective mechanism to capture the underlying relationship between input data and MLaaS to detect real and pseudo drift without relying on ground truth.

Let us assume that an IoT environment designs a performance drift detection model to monitor MLaaS. Identifying an appropriate time interval for re-evaluation is a key challenge, as inappropriate choices can lead to oversensitivity (short intervals) or undersensitivity (long intervals) in detecting performance drift. For example, frequent checks (e.g., every minute) may trigger repeated pseudo-drift notifications, while infrequent checks (e.g., hourly) may delay the detection of real drift, allowing true performance degradation to persist unnoticed. Such inappropriate monitoring frequencies also increase computational and operational costs. Hence, IoT environments require a time-variable mechanism that can adaptively adjust the evaluation interval to detect real performance drift accurately and promptly.

![](images/6cbf210f09aad1b7e1bee25f3d6308765cbdaf86a29332377cec130252bb08da.jpg)  
Fig. 1. Illustration of the Motivation Scenario

## IV. KEY DEFINITIONS AND PROBLEM STATEMENT

Definition 1: MLaaS. An MLaaS service is represented as a tuple ⟨F, QoS⟩, where:

$F$ is the functional specification offered by M, including model specifications ${ \mathcal { N } } _ { \Phi } \ ( \mathrm { e . g . } $ , model type $M _ { t } )$ and data specifications $D _ { s }$ , where $D _ { s } ~ = ~ \{ D _ { V } , D _ { F } \}$ denotes data volume $( D _ { V } )$ , and data features $( D _ { F } )$

• QoS is the non-functional specification offered by M, encompassing evaluation metrics $( E _ { f } , \mathbf { e } . \mathbf { g } .$ ., accuracy, $R ^ { 2 }$ , taskspecific scores), quality factor (Q, historical performance trends), and response time $( R _ { t }$ , latency).

Definition 2: Real Drift. Consider an MLaaS service M. The stream of input–prediction pairs observed over the interval [0, t] is defined as

$$
S _ { t } = \{ ( X _ { 0 } , \hat { Y } _ { 0 } ) , ( X _ { 1 } , \hat { Y } _ { 1 } ) , \ldots , ( X _ { t } , \hat { Y } _ { t } ) \} ,\tag{1}
$$

where $X _ { i }$ denotes the input instance at time $i ,$ and $\hat { Y } _ { i } ~ =$ $M ( X _ { i } )$ represents the prediction generated by the MLaaS. Let $P _ { t } ^ { \dot { M } } ( \dot { X } , \hat { Y } )$ denote the joint probability distribution of the input data and the corresponding MLaaS predictions over the interval [0, t]. A real drift is said to occur at time t + 1 if

$$
P _ { t } ^ { M } ( X , { \hat { Y } } ) \neq P _ { t + 1 } ^ { M } ( X , { \hat { Y } } ) ,\tag{2}
$$

indicating a fundamental change in the relationship between the input data and the MLaaS predictions, thereby reflecting a degradation in predictive behaviour.

Definition 3 (Pseudo Drift). A pseudo drift occurs when the input data distribution changes while the underlying predictive

TABLE I NOTATION SUMMARY
<table><tr><td>Notation</td><td>Description</td></tr><tr><td>M</td><td>Black-box ML service (MLaaS model).</td></tr><tr><td> $F$ </td><td>Functional specification of the ML service.</td></tr><tr><td> $\mathrm { Q o S }$ </td><td>Non-functional (quality) attributes of the service.</td></tr><tr><td> $D _ { s }$ </td><td>Data specification of the service.</td></tr><tr><td> $X$ </td><td>Input feature vector.</td></tr><tr><td> $d$ </td><td>Number of input features.</td></tr><tr><td> $x _ { i }$ </td><td>i-th input feature.</td></tr><tr><td> $^ { b }$ </td><td>Baseline (reference) window.</td></tr><tr><td> $_ w$ </td><td>Current observation window.</td></tr><tr><td> $I ( x _ { i } )$ </td><td>Feature importance score of  $x _ { i } .$ </td></tr><tr><td> $I$ </td><td>Feature importance vector.</td></tr><tr><td> $\delta$ </td><td>Small perturbation applied to input features.</td></tr><tr><td> $\theta _ { m }$ </td><td>Drift decision threshold.</td></tr><tr><td> $m _ { t }$ </td><td>Drift score at block t.</td></tr><tr><td> $B$ </td><td>Buffer of recent drift scores.</td></tr><tr><td> $W _ { s }$ </td><td>Adaptive window size.</td></tr><tr><td> $W _ { \mathrm { m i n } } , W _ { \mathrm { m a x } }$ </td><td>Minimum and maximum window bounds.</td></tr><tr><td> $\Delta W$ </td><td>Window adjustment step.</td></tr><tr><td> $L$ </td><td>Adaptation trend length.</td></tr><tr><td> $c _ { u }$ </td><td>Counter for drift sensitivity (oversensitivity).</td></tr><tr><td> $c _ { o }$ </td><td>Counter for non-drift sensitivity (undersensitivity).</td></tr><tr><td> $r$ </td><td>Drift ratio within the buffer.</td></tr><tr><td> $\theta _ { r }$ </td><td>drift ratio threshold.</td></tr></table>

mapping remains stable, such that MLaaS performance does not experience sustained degradation.

$$
P _ { t } ^ { M } ( \boldsymbol { X } ) \neq P _ { t + 1 } ^ { M } ( \boldsymbol { X } ) \quad \mathrm { a n d } \quad P _ { t } ^ { M } ( \boldsymbol { \hat { Y } } \mid \boldsymbol { X } ) \approx P _ { t + 1 } ^ { M } ( \boldsymbol { \hat { Y } } \mid \boldsymbol { X } ) ,\tag{3}
$$

where $P _ { t } ^ { M } ( X )$ denotes the input data distribution at time $t ,$ and $\dot { P } _ { t } ^ { M } ( \hat { Y } | X )$ represents the conditional predictive distribution of the MLaaS. A pseudo drift reflects a temporary fluctuation in input data that does not alter the predictive mapping of the service and therefore does not lead to longterm performance degradation.

MLaaS Performance Drift Detection Problem. The MPDD problem concerns automatically distinguishing between real drift and pseudo drift in an MLaaS service when changes in the operational environment are observed. To this end, the drift detection model outputs a drift classification (DC) and is defined as the following function

$$
\mathrm { M P D D } ( M , S _ { t } ) = \left\{ 1 , \quad \mathrm { i f ~ } P _ { t } ^ { M } ( X , \hat { Y } ) \neq P _ { t + 1 } ^ { M } ( X , \hat { Y } ) , \right.\tag{4}
$$

Here, M denotes the MLaaS service in operation, and $S _ { t } =$ $\{ ( X _ { i } , \hat { Y } _ { i } ) \} _ { i = 0 } ^ { t }$ represents the stream of recent input–prediction pairs up to time t. A classification of Real Drift requires immediate intervention (e.g., model retraining or service reconfiguration), whereas Pseudo Drift indicates that the service can continue operating without corrective action.

## V. PROPOSED MLAAS PERFORMANCE DRIFT DETECTION FRAMEWORK FOR IOT ENVIRONMENTS

In this section, we present the MLaaS Performance Drift Detection (MPDD) framework for identifying performance drift in MLaaS within IoT environments. The framework comprises five key components, as illustrated in Fig. 2. The first component is the edge and database layer, where IoT devices such as smartwatches, fitness trackers, cameras, and smart rings generate real-time data streams stored in a centralized database. The second component, the MLaaS layer, integrates a pre-trained model provided by the MLaaS provider into the IoT edge for local predictions (see Definition 1). The third component, the MLaaS extraction layer, trains an extraction model to capture latent factors influencing predictions. The extraction model M<sup>′</sup> is constructed using queried MLaaS input–output interactions collected from the IoT environment and operates within the cloud infrastructure. In the current implementation, the extraction model is initialized through an offline training phase prior to runtime monitoring. The performance drift detection module compares input-data statistics with extraction models–derived features over time to distinguish real and pseudo drift. Finally, the adaptive temporal performance drift detection mechanism dynamically adjusts the monitoring interval, ensuring reliable drift detection while reducing oversensitivity to minor fluctuations.

![](images/f6c37178726c7bbc32d75aa4050e2ef0b00826412658320546d02dc8fb88c005.jpg)  
Fig. 2. Machine Learning as a Service (MLaaS) Performance Drift Detection Framework for IoT environments

## A. MLaaS Extraction model

We propose an MLaaS Extraction Model to approximate MLaaS behaviour using unlabeled input instances and the corresponding predictions returned by the service, as illustrated in Algorithm 1. We assume that the IoT provider maintains a historical repository of unlabeled inputs collected during normal system operation. Accordingly, Phase 1 (Input– Output Query Log Collection) constructs the input–output query log from these historical inputs for offline training of the MLaaS Extraction Model. Once trained, the proposed framework requires no additional MLaaS queries beyond the application’s normal inference requests. Traditional modelextraction approaches often rely on prediction probabilities, which can be affected by protection mechanisms such as output rounding, quantization, and probability perturbation [30], [31].To mitigate the impact of these protection mechanisms, the proposed MLaaS Extraction Model relies exclusively on predicted class labels. The resulting input–label pairs capture the observable decision behaviour of the MLaaS and are used to train the extraction model. Since surrogate-based methods (e.g., SHAP) incur high computational overhead in MLaaS settings, Phase 2 (Extraction Model Training) trains a lightweight decision-tree model M<sup>′</sup> using the collected query log to approximate the observable input–output behaviour (lines 8–9). In our implementation, $M ^ { \prime }$ is a CART decisiontree classifier trained using the Gini impurity criterion and best-split strategy. The maximum tree depth is set to four, with a minimum split size of two samples and a minimum leaf size of one sample. The model is trained using MLaaSpredicted labels from the query log. To capture both stable and drifted operating conditions, the query pool consists of 60% clean samples and 40% drift samples, with drift data uniformly collected from sudden, gradual, incremental, and recurrent drift scenarios. The detailed configuration and training setup of M<sup>′</sup> are summarized in Table II. In Phase 3 (Feature-Importance Extraction), the trained model perturbs each input feature individually and measures the output deviation to compute feature-importance scores as the average deviation across samples (lines 10–17). Finally, the algorithm returns the aggregated feature-importance vector (line 18), which forms the basis for subsequent performance drift analysis.

Lemma 1 Let M be the black-box MLaaS service and M<sup>′</sup> the extraction model. Assume that M<sup>′</sup> satisfies a bounded fidelity condition such that $\mathbb { E } [ | M ( X ) - M ^ { \prime } ( X ) | ] \leq \epsilon$ . Then, the feature importance vector ${ \bf \cal I } ^ { \ddot { M } ^ { \prime } }$ extracted from $M ^ { \prime }$ consistently approximates the true behavioural importance ${ \bf \cal I } ^ { M }$ of M, and the deviation remains bounded as

Algorithm 1 MLaaS Extraction Model   
1: Input: Unlabeled inputs $X = \{ X ^ { ( 1 ) } , \ldots , X ^ { ( N ) } \}$ , black  
box MLaaS $M ( \cdot )$   
2: Output: Feature-importance vector I   
$\{ I ( x _ { 1 } ) , \ldots , I ( x _ { d } ) \}$   
Phase 1: Input–Output Query Log Collection   
3: Initialize $\mathcal { D }  \emptyset$   
4: for $j  1$ to N do   
5: $\hat { y } ^ { ( j ) } \gets M \big ( X ^ { ( j ) } \big )$   
6: $\mathcal { D }  \mathcal { D } \cup \{ ( X ^ { ( j ) } , \hat { y } ^ { ( j ) } ) \}$   
7: end for   
Phase 2: Train Extraction Model $M ^ { \prime }$   
8: Initialize $M ^ { \prime }$ as a lightweight decision-tree model   
9: Train $M ^ { \prime }$ on D to approximate the mapping $X  { \hat { y } }$   
Phase 3: Feature Importance Extraction   
10: for i ← 1 to d do   
11: $I ( x _ { i } ) \gets 0$   
12: for $j  1$ to N do   
13: $\overset { \circ } X _ { + i } ^ { ( j ) }  ( x _ { 1 } ^ { ( j ) } , \ldots , x _ { i } ^ { ( j ) } + \delta , \ldots , x _ { d } ^ { ( j ) } )$   
14: $I ( x _ { i } )  I ( x _ { i } ) + \| M ^ { \prime } ( X _ { + i } ^ { ( j ) } ) - M ^ { \prime } ( X ^ { ( j ) } ) \| _ { 1 }$   
15: end for   
16: $\begin{array} { r } { I ( x _ { i } ) \gets \frac { 1 } { N } I ( x _ { i } ) } \end{array}$   
17: end for   
18: return I

$$
\| \boldsymbol { \mathbf { I } } ^ { M ^ { \prime } } - \boldsymbol { \mathbf { I } } ^ { M } \| \leq f ( \epsilon , \delta ) ,
$$

where ϵ denotes the fidelity error and δ denotes the perturbation sensitivity in the importance estimation process.

Proof. Please refer to Appendix A.

## B. Frechet Data Drift Score (FDDS)´

Frechet distance´ (also known as the Wasserstein-2 distance), has been widely used to measure the distance between distributions of models’ features in deep learning studies [32]. In our context, we use it to measure the statistical difference between the historical (baseline) data used to select the MLaaS and the new data windows in the stream, and we call this metric the Frechet Data Drift Score (FDDS)´ . The proposed framework incorporates a preprocessing and feature transformation layer prior to FDDS computation. This layer transforms categorical and raw event-log data into structured numerical feature representations, enabling FDDS to operate on the transformed feature space. Given a multivariate normal baseline distribution b characterized by mean vector $\mu _ { b }$ and covariance matrix $\Sigma _ { b }$ , and a new data window w with mean vector $\mu _ { w }$ and covariance matrix $\Sigma _ { w }$ , the FDDS is computed as

$$
\begin{array} { r } { \mathrm { F D D S } ( b _ { i } , w _ { i } ) = \underbrace { \Vert \mu _ { b } - \mu _ { w } \Vert _ { 2 } ^ { 2 } } _ { \mathrm { M e a n ~ S h i f t } } + \underbrace { \mathrm { T r } \Big ( \Sigma _ { b } + \Sigma _ { w } - 2 \sqrt { \Sigma _ { b } \Sigma _ { w } } \Big ) } _ { \mathrm { C o v a r i a n c e ~ S h i f t } } , } \end{array}\tag{5}
$$

where $\| \mu _ { b } - \mu _ { w } \| _ { 2 } ^ { 2 }$ measures the squared difference between the mean vectors (shift in center), and the trace term measures the difference in the covariance structure (spread) between the two distributions. A higher FDDS indicates a larger divergence and a higher likelihood of data drift.

Lemma 2 Let b and w denote the baseline and the current data window as defined in the FDDS formulation in Eq. (2). Since FDDS decomposes the statistical difference into mean shift and covariance shift, and the covariance shift term is non-negative, FDDS is lower bounded by the mean shift, i.e.,

$$
\mathrm { F D D S } ( b _ { i } , w _ { i } ) \geq \| \mu _ { b } - \mu _ { w } \| _ { 2 } ^ { 2 } .
$$

When the covariance structures are approximately unchanged, FDDS remains stable under small mean perturbations. Conversely, larger mean and/or covariance changes yield higher FDDS. Thus, FDDS stays low under minor deviations and increases under significant drift, ensuring effective drift differentiation between the baseline batch and the current window. Proof. Please refer to Appendix A.

## C. MLaaS-Aware Drift Exposure Score (MDES)

MDES is designed to capture the relationship between changes in input data and MLaaS behavior. MDES quantifies the degree of exposure of an MLaaS service to performance drift by combining the MLaaS extraction feature importance with the FDDS. This integration allows the framework to capture changes in input data characteristics and variations in MLaaS behavior inferred through the MLaaS extraction model. Unlike traditional drift detectors that rely solely on distributional change, MDES considers how these changes interact with the influential features identified by the MLaaS extraction model, providing a deeper understanding of drift causes. Formally, let $\mathbf { I } = \{ I ( x _ { 1 } ) , I ( x _ { 2 } ) , \ldots , I ( x _ { d } ) \}$ represent the importance weights obtained from the MLaaS extraction model for d input features, and let $\mathrm { F D D S } ( b , w )$ denote the Frechet Data Drift Score between the baseline window´ b and the current window w. The MDES is defined as

$$
\mathrm { M D E S } = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } \left( \sum _ { i = 1 } ^ { d } I ( x _ { i } ) { \mathrm { ~ F D D S } } ( b _ { i } , w _ { i } ) \right) > \theta _ { m } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right.\tag{6}
$$

Here, $\theta _ { m }$ is an automated drift-exposure threshold derived from the mean (µ) and standard deviation (σ) of the driftexposure scores, defined as $\theta _ { m } = \mu + \beta \sigma$ . The sensitivity coefficient $\beta$ controls the threshold level and is empirically determined. When the weighted combination of feature importance and data drift exceeds $\theta _ { m }$ , the MLaaS service is classified as experiencing real drift. Changes that do not exceed $\theta _ { m }$ are classified as pseudo drift

Lemma 3 Let b be the baseline window and w the current window, and let $\mathrm { F D D S } ( b , w )$ denote the statistical shift between them. Let $\textbf { I } = \{ I ( x _ { 1 } ) , \ldots , I ( x _ { d } ) \}$ be the bounded feature-importance vector obtainedfrom the MLaaS extraction model, and let MDES be defined as in Eq. (3). If the statistical shift captured by FDDS(b, w) is primarily concentrated on features with low importance weights, then the cumulative weighted exposure remains bounded and the drift is characterized as pseudo drift. Conversely, if the shift aligns with highly influential features, the weighted exposure increases proportionally and the drift is characterized as real drift. Proof. Please refer to Appendix A.

Algorithm 2 MLaaS Performance Drift Detection Model Algorithm 3 Adaptive-Temporal Performance Drift Detection   
1: Input: Input data $( X )$ , baseline statistics b Mechanism (APDDM)   
2: Output: MLaaS-Aware Drift Exposure Score $S _ { \mathrm { M D E S } }$ 1: Input: $\left. \{ X _ { t } \} , b , W _ { s } , \theta _ { r } , L , \Delta W , W _ { m i n } , W _ { m a x } \right.$   
3: Step 1: MLaaS Extraction Model Training 2: Output: drift triggers, W   
4: Train lightweight extraction model $M ^ { \prime }$ to approximate M: 3: $B  [ ] , c _ { u }  0 , c _ { o }  0$   
$M ^ { \prime } ( X ) \approx M ( X )$ 4: for each block $X _ { t }$ do   
5: Compute feature importance scores: 5: $m _ { t } \gets \mathbf { M P D D } ( X _ { t } , b )$ ▷ $m _ { t } \in \{ 0 , 1 \}$ , where 1   
$\begin{array} { r } { \dot { I ( x _ { i } ) } ~ = ~ \frac { 1 } { N } \sum _ { j = 1 } ^ { N ^ { \star } } | M ^ { \prime } ( x _ { 1 } ^ { ( j ) } , \dots , x _ { i } ^ { ( j ) } + \delta , \dots , x _ { d } ^ { ( j ) } ) - } \end{array}$ indicates real drif   
$M ^ { \prime } ( X ^ { ( j ) } ) |$ 6: append $m _ { t } \mathrm { t o } B ; \mathrm { i f } | B | > W _ { s } ,$ , remove oldest   
6: Form importance vector $\mathbf { I } = \{ I ( x _ { 1 } ) , I ( x _ { 2 } ) , \ldots , I ( x _ { d } ) \}$ 7: $\mathbf { i f } \ | \boldsymbol { B } | = W _ { s }$ then   
8: $\begin{array} { r } { r \gets \frac { 1 } { W _ { s } } \sum _ { m \in { \mathcal { B } } } m } \end{array}$ ▷ drift ratio in recent window   
7: Step 2: Compute Frechet Data Drift Score´   
$\mu _ { b } , \Sigma _ { b }  \mathrm { S t a t s } ( X _ { b } ) , \quad \mu _ { w } , \Sigma _ { w }  \mathrm { S t a t s } ( X _ { w } )$ 9: if $r > \breve { \theta } _ { r }$ then   
$S _ { \mathrm { F D D S } } = \Vert \mu _ { b } - \mu _ { w } \Vert _ { 2 } ^ { 2 } + \operatorname { T r } \bigl ( \Sigma _ { b } + \Sigma _ { w } - 2 \sqrt { \Sigma _ { b } \Sigma _ { w } } \bigr )$ 10: trigger drift; $c _ { u } \gets c _ { u } + 1 ; c _ { o } \gets 0$   
8: Step 3: Compute MLaaS-Aware Drift Exposure Score 11: else   
9: Compute feature-weighted exposure: 12: $c _ { o } \gets c _ { o } + 1 ; c _ { u } \gets 0$   
$\begin{array} { r } { S _ { \mathrm { w e i g h t e d } } = \sum _ { i = 1 } ^ { d } I ( x _ { i } ) \times S _ { \mathrm { F D D S } } } \end{array}$ 13: end if   
14: if $c _ { u } \geq L$ then   
10: Compute automated threshold:   
$\theta _ { m } = \mu + \beta \sigma$ 15: $W _ { s } \gets \operatorname* { m a x } ( W _ { m i n } , W _ { s } - \Delta W ) ; c _ { u } \gets 0$   
11: if $S _ { \mathrm { w e i g h t e d } } > \theta _ { m }$ then 16: end if   
17: if $c _ { o } \geq L$ then   
12: $S _ { \mathrm { M D E S } }  1$ ▷ Real drift   
13: else 18: $W _ { s } \gets \operatorname* { m i n } ( W _ { m a x } , W _ { s } + \Delta W ) ; c _ { o } \gets 0$   
14: $S _ { \mathrm { M D E S } }  0$ ▷ Pseudo drift 19: end if   
20: end if   
15: end if   
21: end for

## D. MLaaS Performance Drift Detection Model

We propose a novel MPDD model to detect real and pseudo drift, specifically designed for IoT environments. It explicitly addresses two critical challenges: (i) detecting changes in dynamic input data properties, and (ii) monitoring concurrent changes in MLaaS behavior. Algorithm 2 describes the MLaaS Performance Drift Detection process. The inputs are the input data $X _ { w } ,$ , the baseline window b, and the current window w. The output is the MLaaS-Aware Drift Exposure Score S<sub>MDES</sub> (Algorithm 2, Lines 1–2). The algorithm trains the MLaaS extraction model (M<sup>′</sup>) on input–output pairs (X, Y ) to approximate the underlying MLaaS model. This enables changes in input data to be mapped to shifts in the model’s decision patterns. It then computes feature-importance scores $I ( x _ { i } )$ for each input characteristic and collects them as $\mathbf { I } = \{ I ( x _ { 1 } ) , \ldots , I ( x _ { d } ) \}$ (Algorithm 2, Lines 3–6). Next, the algorithm computes the Frechet Data Drift Score (FDDS) to´ compare the current input statistics with the baseline statistics. This produces a single score that quantifies the distributional change (Algorithm 2, Line 7). The MLaaS-Aware Drift Exposure Score (MDES) is then evaluated using Eq. 6, which combines I with FDDS to weight distributional shifts by feature influence (Algorithm 2, Lines 8–10). Finally, the score is compared against a user-defined threshold $\theta _ { m }$ to classify the event as real drift $( S _ { \mathrm { M D E S } } = 1 )$ or pseudo drift $( S _ { \mathrm { M D E S } } = 0 )$ (Algorithm 2, Lines 10–15), and the procedure returns S<sub>MDES</sub>.

## E. Adaptive-Temporal Performance Drift Detection Mechanism (APDDM)

We propose an APDDM to adaptively adjust the window size $W _ { s }$ based on recent drift patterns, ensuring stable

$$
W _ { s }
$$

performance-drift detection in MLaaS environments. The algorithm takes as inputs a stream of blocks $\{ X _ { t } \}$ , buffer length $b ,$ initial window size $W _ { s }$ , drift ratio threshold $\theta _ { r } .$ , trend length $L ,$ adaptation step $\Delta W ,$ and window bounds $[ W _ { \mathrm { m i n } } , W _ { \mathrm { m a x } } ]$ (Algorithm 3, line 1), and outputs the detected drift triggers and final adapted window size (Algorithm 3, line 2). The procedure initializes a buffer B to store recent drift scores along with two counters $c _ { u }$ and $c _ { o }$ that track consecutive drift and non-drift windows (Algorithm 3, lines 3–4). For each incoming block $X _ { t } ,$ the MPDD model computes a drift score $m _ { t }$ and appends it to the buffer while retaining only the most recent observations (Algorithm 3, lines 4–6). Once the buffer reaches capacity, the algorithm evaluates the proportion of scores exceeding the threshold to classify the current window as drift or stable (Algorithm 3, lines 7–8). Accordingly, $c _ { u }$ is incremented and $c _ { o }$ reset under drift conditions, whereas stable states increment $c _ { o }$ and reset $c _ { u }$ (Algorithm 3, lines 9–12), capturing shortterm temporal trends in detection outcomes. If drift persists for at least $L$ consecutive windows, the method interprets this as under-sensitivity and reduces $W _ { s }$ by $\Delta W$ within the allowable bounds (Algorithm 3, lines 14–15). Conversely, prolonged stability for L windows indicates over-sensitivity, leading to an increase in the window size to avoid unnecessary evaluations (Algorithm 3, lines 17–18). After processing all blocks, the algorithm returns the final adapted window size (Algorithm 3, line 22). Through this feedback-driven strategy, APDDM balances responsiveness and stability, reducing false alarms while ensuring timely detection of performance drift.

## VI. COMPUTATIONAL COMPLEXITY ANALYSIS

The computational complexity of Algorithm 3 (APDDM) is primarily governed by the per-block drift score computation and the adaptive window management operations. For each incoming data block $X _ { t }$ of size n with d features, the algorithm invokes the MPDD scoring function once and then performs lightweight buffer updates and ratio evaluation. Assuming the extraction model and baseline statistics are precomputed offline (i.e., no online retraining), the online MPDD computation involves feature-wise statistical comparison and weighted aggregation, requiring a single pass over the block and yielding a time complexity of $O ( n d )$ under a diagonal/statistical representation, or up to $O ( n d ^ { 2 } )$ if full covariance-based distance measures are employed. The APDDM control logic consists of inserting the drift score into a bounded buffer of size $b ,$ computing the drift ratio over the buffer, and updating counters and window size, resulting in $O ( W _ { s } )$ per block (or $O ( 1 )$ if a running count is maintained). Therefore, the overall per-block complexity of the algorithm can be expressed as $O ( n d + W _ { s } )$ and over $T$ processed blocks, the total time complexity of the proposed adaptive drift prediction mechanism is given by

$$
\mathcal { T } ( T ) = O \left( \sum _ { t = 1 } ^ { T } \left( n d + W _ { s } ( t ) \right) \right)\tag{7}
$$

which simplifies to $O ( T n d )$ in typical streaming settings where $W _ { s } \ll n .$ . The space complexity is modest, requiring $O ( W _ { s } )$ memory to store the adaptive drift-score buffer and $O ( d )$ (or $O ( d ^ { 2 } )$ for full covariance) for maintaining baseline statistical summaries, making the algorithm computationally efficient and scalable for continuous MLaaS monitoring scenarios without incurring any retraining overhead.

## VII. EXPERIMENTS AND RESULTS

In this section, we evaluate the proposed MPDD for IoT environments. First, we assess the effectiveness of the MPDD model in distinguishing real drift from pseudo drift by comparing it with representative baseline techniques, including ADWIN, MMD, D3, STUDD, and SCSD, using accuracy, precision, and recall. Second, we evaluate the timeliness of the proposed APDDM across multiple datasets. This experiment investigates the effectiveness of adaptive temporal monitoring in improving drift responsiveness compared with fixed-interval monitoring and baseline drift detectors. Performance is evaluated using detection accuracy, accuracy gain, miss detection ratio, and false negatives. Third, we evaluate the MLaaS extraction model by analyzing its fidelity and predictive behavior relative to the original black-box MLaaS models. In addition, we assess the impact of surrogate approximation on the overall detection performance. The MLaaS servies are extracted from the MDA data generation framework [33]. All experiments were conducted on an Intel Core i7 machine with 16 GB RAM using Python. The source code is publicly available in the repository<sup>7</sup>.

## A. Experiment Setup and Dataset

In this section, we discuss the datasets, drift setup, and baseline techniques used to evaluate the proposed framework.

## 1) Dataset:

• Human Activity Recognition (HAR) [34]: We use the PAMAP2 dataset containing 3,850,505 samples across 52 sensor channels collected from multiple daily activities.

• Electricity [35]: The Electricity dataset has 45,325 samples with nine features from the New South Wales electricity market, capturing price changes every five minutes.

• Weather [36]: The NOAA Weather dataset includes 96,454 records with nine features describing weather attributes.

• Airline [37]: The Airline Delay dataset contains 539,395 instances with 8 features, including arrival and departure records of U.S. commercial flights.

• Poker [38]: The Poker dataset consists of 1,000,000 samples with 11 attributes, where each instance represents a hand of five cards encoded by rank and suit.

2) Drift Data Generation: Traditional studies simulate different forms of concept drift, including sudden, gradual, incremental, and recurrent changes in data streams. To comprehensively evaluate the proposed framework, we generate representative drift scenarios covering all four drift categories using the widely adopted SEA generator [39]. The generated scenarios introduce controlled changes in the data distribution, allowing the effectiveness of the framework to be assessed under diverse drift behaviors without being restricted to a specific drift type. Examples of the generated drift scenarios across datasets are provided in Appendix B.

3) Baseline Techniques: Existing literature does not explicitly address the challenge of MLaaS performance drift, particularly in black-box IoT settings. However, several related drift detection methods partially capture distributional or modelrelated changes and can serve as reasonable baselines for comparison. To ensure a fair and consistent evaluation, we compare the proposed MPDD framework with established benchmark techniques, including ADWIN [18] and MMD [26], which represent error-based and statistical data-driven drift detection approaches. In addition, we include representative blackbox unsupervised methods that align with realistic MLaaS deployment constraints, namely STUDD [13], SCSD [12], and D3 [11]. All baseline models are implemented using their recommended drift thresholds and evaluated consistently across the selected datasets.

4) Configuration of the MLaaS Extraction Model: The MLaaS extraction model $( M ^ { \prime } )$ plays a key role in the proposed framework by approximating the behaviour of the blackbox MLaaS service. This model enables the extraction of feature-importance information used for performance drift detection. Table II summarizes the architecture and parameter configuration of the extraction model, including the decisiontree structure, splitting criterion, training target, and model settings. Table III presents the training-data configuration used to construct $M ^ { \prime } .$ , including the query-pool size, clean and drift sample composition, and train–test split across datasets. These details provide additional transparency regarding the design and training process of the extraction model.

TABLE IV  
COMPARISON OF MLAAS PERFORMANCE DRIFT DETECTION (MPDD) UNDER PSEUDO DRIFT AND REAL DRIFT ACROSS DATASETS.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Drift</td><td colspan="3">HAR [34]</td><td colspan="3">Electricity [35]</td><td colspan="3">Weather [36]</td><td colspan="3">Poker [38]</td><td colspan="3">Airline [37]</td></tr><tr><td>Acc</td><td>Prec</td><td>Rec</td><td>Acc</td><td>Prec</td><td>Rec</td><td>Acc</td><td>Prec</td><td>Rec</td><td>Acc</td><td>Prec</td><td>Rec</td><td>Acc</td><td>Prec</td><td>Rec</td></tr><tr><td>MPDD</td><td>Pseudo Real</td><td>0.91</td><td>0.88 0.98</td><td>0.98 0.71</td><td>0.90</td><td>0.87 0.99</td><td>0.98 0.72</td><td>0.89</td><td>0.85 0.98</td><td>0.98 0.71</td><td>0.88</td><td>0.89 0.85</td><td>0.94 0.75</td><td>0.90</td><td>0.87 0.98</td><td>0.98 0.69</td></tr><tr><td>ADWIN [18]</td><td>Pseudo Real</td><td>0.62</td><td>0.66 0.53</td><td>0.74 0.44</td><td>0.77</td><td>0.77 0.77</td><td>0.83 0.69</td><td>0.50</td><td>0.46 0.57</td><td>0.65 0.38</td><td>0.54</td><td>0.61 0.42</td><td>0.64 0.39</td><td>0.65</td><td>0.71 0.56</td><td>0.71 0.56</td></tr><tr><td>MMD [26]</td><td>Pseudo Real</td><td>0.52</td><td>0.59 0.39</td><td>0.64 0.34</td><td>0.69</td><td>0.69 0.71</td><td>0.82 0.53</td><td>0.54</td><td>0.49 0.62</td><td>0.70 0.40</td><td>0.54</td><td>0.61 0.42</td><td>0.65 0.38</td><td>0.65</td><td>0.71 0.56</td><td>0.71 0.56</td></tr><tr><td>D3 [11]</td><td>Pseudo Real</td><td>0.46</td><td>0.98 0.43</td><td>0.10 0.98</td><td>0.72</td><td>0.81 0.65</td><td>0.64 0.82</td><td>0.57</td><td>0.51 0.67</td><td>0.74 0.43</td><td>0.53</td><td>0.59 0.38</td><td>0.69 0.28</td><td>0.60</td><td>0.60 0.10</td><td>0.98 0.10</td></tr><tr><td>STUDD [13]</td><td>Pseudo Real</td><td>0.69</td><td>0.69 0.68</td><td>0.99 0.04</td><td>0.66</td><td>0.66 0.57</td><td>0.98 0.05</td><td>0.61</td><td>0.62 0.37</td><td>0.95 0.05</td><td>0.68</td><td>0.68 0.10</td><td>0.98 0.10</td><td>0.68</td><td>0.68 0.10</td><td>0.98 0.10</td></tr><tr><td>SCSD [12]</td><td>Pseudo Real</td><td>0.65</td><td>0.99 0.48</td><td>0.50 0.98</td><td>0.56</td><td>0.98 0.44</td><td>0.33 0.98</td><td>0.69</td><td>0.67 0.95</td><td>0.99 0.21</td><td>0.70</td><td>0.70 0.97</td><td>0.98 0.06</td><td>0.41</td><td>0.98 0.35</td><td>0.15 0.90</td></tr></table>

CONFIGURATION OF THE MLAAS EXTRACTION MODEL M<sup>′</sup>.  
TABLE II
<table><tr><td>Parameter</td><td>Configuration</td><td>Parameter</td><td>Configuration</td></tr><tr><td>Model</td><td>CART decision tree</td><td>Criterion</td><td>Gini impurity</td></tr><tr><td>Split strategy</td><td>Best split</td><td>Maximum depth</td><td>4</td></tr><tr><td>Min samples split</td><td>2</td><td>Min samples leaf</td><td>1</td></tr><tr><td>Maximum features</td><td>All variables</td><td>Training target</td><td>MLaaS labels</td></tr><tr><td>Train/Test split</td><td>50% /  50%</td><td>Training data</td><td>60% clean + 40% drift</td></tr></table>

TABLE III  
TRAINING DATA CONFIGURATION OF THE MLAAS EXTRACTION MODEL
<table><tr><td>Dataset</td><td>Query Pool</td><td>Clean</td><td>Drift</td><td>Train</td><td>Test</td></tr><tr><td>HAR</td><td>100,000</td><td>60,000</td><td>40,000</td><td>50,000</td><td>50,000</td></tr><tr><td>Electricity</td><td>75,000</td><td>45,000</td><td>30,000</td><td>37,500</td><td>37,500</td></tr><tr><td>Weather</td><td>160,000</td><td>96,000</td><td>64,000</td><td>80,000</td><td>80,000</td></tr><tr><td>Poker</td><td>160,000</td><td>96,000</td><td>64,000</td><td>80,000</td><td>80,000</td></tr><tr><td>Airline</td><td>16,649</td><td>9,989</td><td>6,660</td><td>8,324</td><td>8,325</td></tr></table>

B. Experiment 1: Evaluation of MLaaS Performance Drift Detection approach

The proposed MPDD model aims to detect real and pseudo drift in IoT environments. To evaluate its effectiveness, we use standard metrics including accuracy, precision, and recall. Table IV compares MPDD against benchmark drift detection techniques, including ADWIN, MMD, STUDD, SCSD, and D3, across multiple datasets. Identifying an appropriate threshold to distinguish real and pseudo drift is a critical challenge in drift detection. Since trial-and-error threshold selection may not generalize well across different IoT environments, we adopt an automated threshold adjustment mechanism inspired by DDM and EDDM, where the drift-exposure threshold $( \theta _ { m } )$ is automatically determined from the observed drift-exposure scores. Table IV shows that MPDD consistently achieves the highest overall detection accuracy across all datasets while maintaining balanced precision and recall for both real and pseudo drift scenarios. For pseudo drift, MPDD achieves accuracy values of 0.91 on HAR, 0.90 on Electricity, 0.89 on Weather, 0.88 on Poker, and 0.90 on Airline, with precision ranging from 0.85 to 0.89 and recall values consistently above 0.94. More importantly, MPDD maintains strong performance under real drift conditions, where competing approaches often degrade substantially. Across all datasets, it achieves precision values of up to 0.99 and recall values between 0.69 and 0.75, demonstrating reliable identification of genuine MLaaS performance degradation. In contrast, traditional datadistribution-based methods such as ADWIN and MMD show lower and less balanced performance. While they achieve moderate pseudo-drift detection in some datasets, their realdrift recall ranges from only 0.34 to 0.69. This suggests that methods relying primarily on data distribution changes may fail when variations in the input data do not directly affect MLaaS prediction behaviour, indicating that data drift does not always translate to performance drift. Black-box unsupervised approaches such as D3, STUDD, and SCSD exhibit mixed performance. D3 achieves competitive results on some datasets, such as Electricity with a real-drift recall of 0.82, but shows severe precision-recall imbalance on others, including HAR where pseudo-drift recall falls to 0.10. Similarly, STUDD achieves high pseudo-drift recall across all datasets (0.95–0.99) but performs poorly for real drift detection, with recall values dropping to as low as 0.04– 0.10. SCSD achieves high precision or recall for specific drift classes, but suffers from substantial precision-recall imbalance, resulting in inconsistent detection performance. Overall, these results confirm the robustness and generalization of MPDD in detecting both pseudo and real drift while maintaining a more balanced performance than existing data-based and black-box unsupervised drift detection approaches.

![](images/9196670e47b70ba9f4e63cbe2f061fa6b1138094de288bc370df12fe7c0f962a.jpg)  
(a)

![](images/a04fe3c6f5c736789ef0edf0962a881a98e16d855b918e543a4c6d94f9ef1965.jpg)  
(b)

![](images/b5d7b7d96e0fcb1ca85a096bfdeb5cf593f41e6a88cce221ec74ce6c06d5abcc.jpg)  
(c)

![](images/f526cb18808bee3fa9632fb53289b66f72de01060a959e1ae2b0fb618793a03e.jpg)  
(d)

![](images/449fc43e9aca27a72abb8fd59a1ae0daed89551abccb5ac96ee8c9f0ba44d7be.jpg)  
(e)

![](images/89a9e2aa38a310529effd4d41f0075f8153103b8277ed1cefa2a8a1a8d6b0084.jpg)  
(f)  
Fig. 3. Comparative accuracy analysis of drift detection techniques across datasets: (a) HAR, (b) Electricity, (c) Weather, (d) Poker, (e) Airline, and (f) threshold sensitivity analysis, evaluated using the proposed MPDD model.

Figure 3 compares the detection accuracy of MPDD and baseline methods across all datasets. Figures 3(a)–(e) show that MPDD consistently achieves the highest accuracy on HAR, Electricity, Weather, Poker, and Airline. The performance gap between MPDD and the strongest competing method remains substantial across all datasets, highlighting the effectiveness of incorporating both feature-importance variations and data-drift characteristics into the drift-detection process. These results further demonstrate that MPDD maintains robust and consistent performance across diverse IoT environments and drift scenarios. Figure 3(f) evaluates the efficiency of the automated threshold-selection mechanism. The results show that the proposed approach determines suitable driftexposure thresholds significantly faster than the trial-and-error method, achieving an approximately threefold reduction in threshold-selection time. In addition to reducing computational overhead, the automated mechanism eliminates the need for dataset-specific threshold tuning for new IoT environments.

## C. Experiment 2: Timeliness Evaluation of Adaptive-Temporal Performance Drift Detection Mechanism (APDDM)

Figure 4 presents the temporal accuracy comparison between MPDD and APDDM across multiple datasets. Each row corresponds to a dataset and shows accuracy over evaluation chunks (approximately 250 samples per chunk). The shaded green regions (AG) represent the accuracy gain achieved by APDDM over MPDD, illustrating the benefit of adaptive window adjustment under changing drift conditions. As shown in

Fig. 4(a), the HAR dataset highlights the advantage of adaptive monitoring under sudden and recurring drift. During major drift events, MPDD experiences substantial accuracy degradation, dropping from near-perfect performance to approximately 0.20, whereas APDDM maintains accuracy around 0.75 and recovers more rapidly. The corresponding AG regions indicate substantial accuracy improvements during drift intervals. The baseline methods exhibit larger fluctuations and less consistent behaviour across the stream. The Electricity results in Fig. 4(b) demonstrate the benefits of adaptation under mixed sudden and incremental drift. While MPDD accuracy falls to approximately 0.2 and later approaches 0.0 during major drift transitions, APDDM maintains higher accuracy and returns to stable performance more quickly, resulting in larger AG regions during the most severe drift periods. For Weather, Fig. 4(c) presents a smoother but under-sensitive drift pattern. MPDD accuracy drops to approximately 0.15 during major drift periods, whereas APDDM limits the degradation to around 0.4–0.6 and recovers earlier. Although the performance gap is smaller than in HAR and Electricity, the adaptive strategy consistently reduces both the magnitude and duration of accuracy loss, resulting in a smaller but consistent accuracy gain. The Airline dataset, shown in Fig. 4(d), contains stronger sudden-drift behaviour and therefore represents a challenging setting for fixed monitoring. MPDD experiences severe performance drops, reaching nearly 0.0 during major drift events, while APDDM maintains near-perfect accuracy throughout most of the stream. Accordingly, the AG regions are most prominent during abrupt drift events, demonstrating the effectiveness of adaptive monitoring under severe drift conditions. Finally, Fig. 4(e) highlights the recurring and incremental drift characteristics of the Poker stream. MPDD exhibits several accuracy fluctuations throughout the stream, with noticeable drops to approximately 0.65 during drift intervals before recovering. In contrast, APDDM maintains more stable performance, remaining above approximately 0.85 for most of the evaluation period and recovering faster following drift events.

![](images/131ce0f39b6b3fd946fcd3c18b44132b8f2237619d9bd3a3eb4a95fe70a241fd.jpg)  
(a)

![](images/162108005fc1f6434b6d8592d721793be66884911388b652bb0b3ec0a81ae017.jpg)  
(b)

![](images/0db9ffa42ac2e5266a1a8dd7eb6e1ddfcd1754cfcdd6d88d2a3d205cbe2a9277.jpg)  
(c)

![](images/a28743a1d0d843144a493720ca4a0d3541c64931a76c0648e7ab62edbea9f354.jpg)  
(d)

![](images/be7a003fa00e72b70e2d632337a297d75ee798c872a2e41f5189ea97c944902f.jpg)  
(e)

Fig. 4. Timeliness evaluation of APDDM across datasets: (a) HAR, (b) Electricity, (c) Weather, (d) Airline, and (e) Poker, showing accuracy comparisons for each scenario.  
![](images/dee54436633d36c2976a859f06f93183703fcd519e4ff4de50875e86dfef42ab.jpg)  
Fig. 5. False negative count comparison across the HAR, Airline, Electricity, Weather, and Poker datasets.

We further analyze the number of false negatives across different drift-detection techniques to evaluate their ability to identify real performance-drift events. Fig. 5 presents the comparison across datasets, where lower values indicate fewer missed drift detections. Overall, APDDM and MPDD achieve relatively low false-negative counts across all datasets. Although data-distribution-based techniques such as ADWIN, MMD, and D3 show lower false-negative counts for some datasets, particularly HAR and Electricity, these methods also produce significantly higher false-positive rates. Similarly, STUDD and SCSD achieve low false-negative counts in a few cases but exhibit substantially larger values on Weather and Poker, reducing their overall effectiveness. In contrast, MPDD and APDDM provide more consistent performance across different datasets. APDDM achieves the lowest or nearlowest false-negative counts in most cases, demonstrating that adaptive monitoring can reduce missed drift events while maintaining a better balance between false positives and false negatives. These results highlight the effectiveness of the proposed approaches for reliable MLaaS performance-drift detection in dynamic IoT environments.

Table V presents the comparative performance of the proposed APDDM framework against the fixed-interval MPDD baseline and five representative drift detection techniques, namely ADWIN, MMD, D3, STUDD, and SCSD, across five datasets. The comparison is conducted using Accuracy↑, Accuracy Gain↑, Miss Detection Ratio (MDR)↓, and False Negatives (FN)↓. Accuracy reflects the overall correctness of drift detection decisions, while Accuracy Gain quantifies the relative improvement achieved by APDDM over competing approaches. MDR and FN are used to evaluate the ability of each method to identify drift events without missing significant changes in the data stream. Lower MDR and FN values indicate better drift detection capability and improved responsiveness to evolving data distributions. Overall, APDDM consistently achieves the highest detection accuracy across all datasets while simultaneously reducing the number of missed drift events. For the HAR dataset, APDDM achieves the highest accuracy of 0.92, representing an improvement of 1.25% over the MPDD baseline. Furthermore, APDDM reduces the number of false negatives from 440 under MPDD to 380, corresponding to a reduction of approximately 13.6%, while also lowering the MDR from 0.28 to 0.25. These results indicate that adaptive monitoring improves drift-detection accuracy while reducing missed drift events. A similar trend is observed in the Airline dataset, where APDDM achieves an accuracy of 0.96 compared to 0.90 for MPDD, representing a 6.0% improvement. In addition, the number of false negatives is reduced from 80 to 32, corresponding to a 60.0% reduction, while the MDR decreases from 0.31 to 0.17.

TABLE V  
DETECTION-QUALITY COMPARISON ACROSS DRIFT DETECTION TECHNIQUES.
<table><tr><td>Data</td><td>Method</td><td>Acc.↑</td><td>Acc. Gain↑</td><td>MDR↓</td><td>FP↓</td><td>FN↓</td></tr><tr><td>HAR</td><td>ADWIN</td><td>0.62</td><td>30.42</td><td>0.56</td><td>190</td><td>270</td></tr><tr><td></td><td>MMD</td><td>0.52</td><td>40.00</td><td>0.66</td><td>260</td><td>315</td></tr><tr><td></td><td>D3</td><td>0.30</td><td>62.50</td><td>0.73</td><td>495</td><td>350</td></tr><tr><td></td><td>STUDD</td><td>0.69</td><td>23.23</td><td>0.96</td><td>30</td><td>1465</td></tr><tr><td></td><td>SCSD</td><td>0.65</td><td>26.98</td><td>0.02</td><td>1650</td><td>25</td></tr><tr><td></td><td>MPDD</td><td>0.91</td><td>1.25</td><td>0.29</td><td>0</td><td>440</td></tr><tr><td></td><td>APDDM</td><td>0.92</td><td></td><td>0.25</td><td>0</td><td>380</td></tr><tr><td>Air</td><td>ADWIN</td><td>0.65</td><td>31.00</td><td>0.44</td><td>35</td><td>35</td></tr><tr><td></td><td>MMD</td><td>0.65</td><td>31.00</td><td>0.44</td><td>35</td><td>35</td></tr><tr><td></td><td>D3</td><td>0.60</td><td>36.00</td><td>1.00</td><td>0</td><td>80</td></tr><tr><td></td><td>STUDD</td><td>0.68</td><td>27.87</td><td>1.00</td><td>0</td><td>255</td></tr><tr><td></td><td>SCSD</td><td>0.42</td><td>54.12</td><td>0.00</td><td>465</td><td>0</td></tr><tr><td></td><td>MPDD</td><td>0.90</td><td>6.00</td><td>0.31</td><td>0</td><td>80</td></tr><tr><td></td><td>APDDM</td><td>0.96</td><td></td><td>0.18</td><td>0</td><td>32</td></tr><tr><td>Elec.</td><td>ADWIN</td><td>0.77</td><td>15.35</td><td>0.31</td><td>85</td><td>125</td></tr><tr><td></td><td>MMD</td><td>0.69</td><td>23.05</td><td>0.47</td><td>90</td><td>190</td></tr><tr><td></td><td>D3</td><td>0.72</td><td>20.30</td><td>0.19</td><td>180</td><td>75</td></tr><tr><td></td><td>STUDD</td><td>0.66</td><td>26.21</td><td>0.95</td><td>45</td><td>1185</td></tr><tr><td></td><td>SCSD</td><td>0.56</td><td>36.00</td><td>0.00</td><td>1585</td><td>0</td></tr><tr><td></td><td>MPDD</td><td>0.90</td><td>2.34</td><td>0.29</td><td>10</td><td>355</td></tr><tr><td></td><td>APDDM</td><td>0.92</td><td></td><td>0.24</td><td>0</td><td>280</td></tr><tr><td>Weath.</td><td>ADWIN</td><td>0.50</td><td>39.19</td><td>0.62</td><td>305</td><td>655</td></tr><tr><td></td><td>MMD</td><td>0.54</td><td>35.82</td><td>0.60</td><td>260</td><td>635</td></tr><tr><td></td><td>D3</td><td>0.57</td><td>32.72</td><td>0.58</td><td>225</td><td>610</td></tr><tr><td></td><td>STUDD</td><td>0.61</td><td>28.92</td><td>0.95</td><td>235</td><td>2810</td></tr><tr><td></td><td>SCSD</td><td>0.69</td><td>20.30</td><td>0.80</td><td>35</td><td>2345</td></tr><tr><td></td><td>MPDD</td><td>0.89</td><td>0.79</td><td>0.30</td><td>5</td><td>870</td></tr><tr><td></td><td>APDDM</td><td>0.90</td><td></td><td>0.27</td><td>20</td><td>794</td></tr><tr><td>Poker</td><td>ADWIN</td><td>0.54</td><td>43.25</td><td>0.61</td><td>430</td><td>485</td></tr><tr><td></td><td>MMD</td><td>0.54</td><td>43.25</td><td>0.62</td><td>420</td><td>495</td></tr><tr><td></td><td>D3</td><td>0.53</td><td>45.00</td><td>0.72</td><td>375</td><td>575</td></tr><tr><td></td><td>STUDD</td><td>0.68</td><td>29.69</td><td>1.00</td><td>30</td><td>2545</td></tr><tr><td></td><td>SCSD</td><td>0.70</td><td>27.50</td><td>0.94</td><td>5</td><td>2395</td></tr><tr><td></td><td>MPDD</td><td>0.88</td><td>9.81</td><td>0.25</td><td>345</td><td>640</td></tr><tr><td></td><td>APDDM</td><td>0.98</td><td>一</td><td>0.06</td><td>40</td><td>160</td></tr></table>

This demonstrates the effectiveness of adaptive monitoring in identifying evolving drift patterns and minimizing missed detections. For the Electricity dataset, APDDM achieves the highest accuracy of 0.92 and reduces false negatives from 355 under MPDD to 280, corresponding to a reduction of approximately 21.1%. Similarly, the MDR decreases from 0.285 to 0.23. Although D3 achieves a lower MDR of 0.18 with only 75 false negatives, its overall detection accuracy remains substantially lower at 0.72, indicating a less balanced trade-off between drift coverage and overall detection quality. In the Weather dataset, APDDM again provides the highest accuracy of 0.89 while reducing false negatives from 870 to 794 and lowering the MDR from 0.29 to 0.26 compared with MPDD. The performance gap is even more pronounced when compared with traditional drift-detection techniques, which exhibit substantially higher MDR values and considerably larger numbers of missed drift events. Similarly, for the Poker dataset, APDDM achieves the highest accuracy of 0.97, representing a 9.81% improvement over MPDD. Moreover, the number of false negatives is reduced dramatically from 640 under MPDD to only 160, corresponding to a 75.0% reduction. This improvement is accompanied by a substantial decrease in MDR from 0.25 to 0.06, demonstrating the effectiveness of adaptive monitoring in reducing missed drift detections. In contrast, methods such as STUDD and SCSD exhibit MDR values close to 1.0 and very large false-negative counts, indicating that a substantial proportion of drift events remain undetected. Across all datasets, the results demonstrate that APDDM consistently provides a more favourable balance between detection accuracy and drift coverage.

## D. Experiment 3: Comprehensive Evaluation of the MLaaS Extraction Model

The MLaaS extraction model is key in the proposed MPDD framework to approximate the behaviour of the original blackbox MLaaS service. The effectiveness of the framework depends on the quality of this approximation. To evaluate this, we first analyze the fidelity and prediction behavior analysis of the MLaaS extracted model. Next, we then investigate the impact of MLaaS extraction model error on the MPDD and APDDM techniques performance.

1) Fidelity and Predictive Behavior Analysis of the MLaaS extraction Model: In this experiment, we evaluate the fidelity and predictive behavior of the proposed MLaaS extraction model, which aims to approximate the decision patterns of black-box MLaaS services. Fidelity analysis is a standard evaluation strategy in model extraction research. It measures the agreement between the predictions of the extracted model and those of the original inaccessible MLaaS service, rather than the accuracy with respect to the ground-truth labels [40]. Figure 6(a) compares the predictive accuracy of both the original MLaaS service and the extraction model against the ground truth across all datasets after performance drift has occurred. As expected, the predictive accuracy of the original MLaaS service decreases under drifted conditions. The extraction model exhibits nearly the same degradation trend, closely tracking the predictive behaviour of the original MLaaS service. The deviation between the MLaaS service and the extraction model remains small across all datasets, at approximately 3% for HAR, 1% for Electricity, 3% for Weather, and nearly identical performance for Poker and Airline. Consequently, the extraction model achieves an average fidelity of approximately 95.2%, indicating a high level of agreement with the predictions of the original MLaaS service. Specifically, the extraction model achieves predictive accuracies against the ground truth of approximately 0.61 on HAR, 0.72 on Electricity, 0.82 on Weather, 0.78 on Poker, and 0.63 on Airline, closely matching the corresponding accuracies of the original MLaaS services under the same drift conditions. These results demonstrate that the extraction model effectively captures original MLaaS service. To further investigate behavioral consistency, Fig. 6(b) presents a predictive outcome decomposition that categorizes predictions into four cases: both MLaaS and extraction models correct, MLaaS correct but extraction incorrect, extraction correct but MLaaS incorrect, and both models incorrect. This analysis provides a more detailed understanding of the relationship between the extracted model and the original MLaaS service beyond overall accuracy.

![](images/551eb9a8cdbdf373c05cea0ad293d4bb1baf654d7d2a7def9d83bea1f27df1e8.jpg)  
(a)

![](images/3c38eded6706d6fc1d36fd2d2e8e7f88bcc8bf37672c22d46cb73b9a5dc3e71a.jpg)  
(b)  
Fig. 6. Fidelity and predictive behavior analysis of the MLaaS extraction model across datasets: (a) predictive accuracy comparison, and (b) predictive behavior analysis

Across datasets, the dominant portion corresponds to the both-correct category, reaching approximately 0.80 for Weather, 0.78 for Poker, 0.70 for Electricity, 0.63 for Airline, and 0.59 for HAR, indicating substantial agreement between the models. The disagreement regions remain relatively small, with the MLaaS-correct/extraction-wrong and MLaaSwrong/extraction-correct categories generally below 5% across datasets. The both-wrong category is primarily influenced by the inherent difficulty of each dataset. However, the small disagreement fractions indicate that the extraction model closely follows the predictive behavior of the original MLaaS service.

2) Impact of MLaaS Extraction Model Errors on MPDD and APDDM Performance: A key concern in black-box MLaaS monitoring is that the extracted model may not perfectly reproduce the decision behaviour of the original service, particularly when the underlying model is a complex non-linear model. Such approximation errors may occur in disagreement regions and could potentially affect downstream drift-detection decisions. Therefore, we quantify how much MLaaS extraction-model errors propagate to the final driftdetection outcome. Figure 7 compares the drift-detection accuracy obtained using the MLaaS extracted model and the original black-box MLaaS model for both MPDD and APDDM. For MPDD, the results show only small differences between the two models, with accuracy differences ranging from 0.00 to 1.67 percentage points across datasets. No difference is observed for Airline and Weather, while HAR, Electricity, and Poker exhibit only minor variations. For APDDM, the results remain highly consistent with those obtained using the original MLaaS model. The accuracy remains unchanged for Airline, Electricity, and Weather, with only a small difference for HAR and a moderate decrease for Poker. Overall, APDDM is even less sensitive to extraction-model approximation errors, with deviations ranging from -2.58 to +0.42 percentage points, and in most cases showing no measurable difference. This robustness stems from APDDM’s adaptive parameter-selection mechanism, which adjusts the monitoring interval based on recent drift-score trends rather than relying on the output of any single extraction-model prediction. Consequently, small approximation errors have only a limited influence on the final drift-detection decision. These results indicate that MLaaS extraction-model errors have limited impact on the final MPDD and APDDM decisions across the evaluated scenarios. Hence, the extracted model provides a reliable approximation of the original MLaaS service for performance-drift detection.

![](images/01da01b9a9d90c5893f206072d56becb4c7803152c797affb99f659891a66fbf.jpg)  
Fig. 7. Comparison of MPDD and APDDM accuracy when using the MLaaS extracted model and the original MLaaS

## VIII. DISCUSSIONS AND IMPLICATIONS

Our framework enables clients to identify potential drifts in MLaaS, allowing them to notify providers, opt for alternative services, or perform service composition. Since some drift points in the experiments were generated randomly, the results may be sensitive to their placement. To mitigate this, we repeated the random generation multiple times and reported aggregated metrics to ensure robustness. In this context, the proposed approach functions as an effective service management tool for MLaaS consumers by enabling proactive drift awareness, informed provider notification, and timely service switching or composition decisions. This capability is especially important in cost-sensitive and safety-critical domains, such as smart healthcare, industrial IoT, and intelligent transportation, where undetected performance drift can lead to incorrect decisions, operational risks, and financial losses.

## IX. CONCLUDING REMARKS

In this paper, we proposed an MLaaS Performance Drift Detection (MPDD) framework for IoT environments where ground-truth labels and internal model parameters are inaccessible. To address this limitation, we designed an MLaaS extraction model that captures observable behavioral patterns of black-box MLaaS services, enabling reliable monitoring of feature preference variations. We further introduced a Frechet-based drift measurement mechanism´ to quantify distributional changes in evolving data streams and developed an MLaaS-aware Drift Exposure Score within MPDD to predict performance drift. Extensive experiments across multiple real-world datasets demonstrate that the framework improves drift detection effectiveness, achieving approximately 22–25% accuracy improvement over representative baseline drift detection methods while maintaining stability under diverse drift patterns. Additionally, the adaptive-interval APDDM dynamically adjusts monitoring frequency based on drift intensity, yielding around 4% accuracy improvement and approximately 9% reduction in miss detection rate compared to fixed-interval monitoring, thereby enhancing detection reliability and timeliness in non-stationary IoT streams. Overall, the framework provides a robust and timely solution for continuous MLaaS performance monitoring without requiring manual drift inspection or continuous ground-truth collection.

## REFERENCES

[1] Mauro Ribeiro and et al. Mlaas: Machine learning as a service. In 2015 IEEE 14th ICMLA, pages 896–902. IEEE, 2015.

[2] Robert Philipp and et al. Machine learning as a service: Challenges in research and applications. In IIWAS, 2020.

[3] Medtronic plc. New Prediction Feature for Hypoglycemia Now Available in Sugar.IQ<sup>TM</sup> Personal Diabetes Assistant App, Developed by Medtronic and IBM Watson Health, 2019.

[4] Zubair Md Fadlullah and et al. On delay-sensitive healthcare data analytics at the network edge based on deep learning. In 2018 14th IWCMC. IEEE, 2018.

[5] Manuel Baena-Garcıa and et al. Early drift detection method. In Fourth international workshop on knowledge discovery from data streams, 2006.

[6] Isvani Frias-Blanco and et al. Online and non-parametric drift detection methods based on hoeffding’s bounds. IEEE TKDE, 2014.

[7] Myuu Myuu Wai Yan. Accurate detecting concept drift in evolving data streams. ICT Express, 2020.

[8] Roberto SM Barros and et al. Rddm: Reactive drift detection method. ESA, 2017.

[9] Salvatore Greco and et al. Unsupervised concept drift detection from deep learning representations in real-time. IEEE TKDE, 2025.

[10] Hang Yu and et al. Type-ldd: A type-driven lite concept drift detector for data streams. IEEE TKDE, 2023.

[11] Omer G <sup>¨</sup> oz¨ uac¸ık and et al. Unsupervised concept drift detection with a¨ discriminative classifier. In ACM CIKM, 2019.

[12] Sooyong Jang and et al. Sequential covariate shift detection using classifier two-sample tests. In ICML, 2022.

[13] Vitor Cerqueira and et al. Studd: A student–teacher method for unsupervised concept drift detection. Machine Learning, 2023.

[14] Firas Bayram and et al. From concept drift to model degradation: An overview on performance-aware drift detectors. Knowledge-Based Systems, 2022.

[15] Deepak Kanneganti and et al. Adaptive composition of machine learning as a service (mlaas) for iot environments. In 2025 IEEE ICWS. IEEE, 2025.

[16] Alexey Tsymbal. The problem of concept drift: definitions and related work. Computer Science Department, Trinity College Dublin, 2004.

[17] Sheik Mohammad Mostakim Fattah and Athman Bouguettaya. Eventbased detection of changes in iaas performance signatures. In 2020 IEEE SCC. IEEE, 2020.

[18] Albert Bifet and Ricard Gavalda. Learning from time-changing data with adaptive windowing. In SIAM, pages 443–448. SIAM, 2007.

[19] Ali Pesaranghader and et al. Fast hoeffding drift detection method for evolving data streams. In Joint European conference on machine learning and knowledge discovery in databases. Springer, 2016.

[20] Li Yang and Abdallah Shami. A lightweight concept drift detection and adaptation framework for iot data streams. IEEE IoTM, 2021.

[21] Yujing Chen and et al. Asynchronous federated learning for sensor data with concept drift. In IEEE Big Data. IEEE, 2021.

[22] Behshid Shayesteh and et al. Automated concept drift handling for fault prediction in edge clouds using reinforcement learning. IEEE TNSM, 2022.

[23] Pin-Hsuan Chiang and Shi-Chun Tsai. Detection of malicious domains with concept drift using ensemble learning. IEEE TNSM, 2024.

[24] Frank J Massey Jr. The kolmogorov-smirnov test for goodness of fit. JASAA, 1951.

[25] Leonid Nisonovich Vaserstein. Markov processes over denumerable products of spaces, describing large systems of automata. PPI, 1969.

[26] Mathieu Sinn, Ali Ghodsi, and Karsten Keller. Detecting changepoints in time series by maximum mean discrepancy of ordinal pattern distributions. arXiv preprint arXiv:1210.4903, 2012.

[27] Arman Pashamokhtari and et al. Dynamic inference from iot traffic flows under concept drifts in residential isp networks. IEEE IoTJ, 2023.

[28] Vincenzo Agate and et al. Enhancing iot network security with concept drift-aware unsupervised threat detection. In 2024 IEEE ISCC. IEEE, 2024.

[29] Lei Wang and et al. Concept drift-based runtime reliability anomaly detection for edge services adaptation. IEEE TKDE, 2021.

[30] Florian Tramer and et al. Stealing machine learning models via\` prediction apis. In 25th USENIX Security 16, pages 601–618, 2016.

[31] Minxue Tang and et al. Modelguard: Information-theoretic defense against model extraction attacks. In 33rd USENIX Security 24, 2024.

[32] DC Dowson and BV Landau. The frechet distance between multivariate´ normal distributions. JMA, 1982.

[33] Deepak Kanneganti, Sajib Mistry, Sheik Fattah, Joshua Boland, and Aneesh Krishna. Machine learning as a service (mlaas) dataset generator framework for iot environments. In Proceedings of the ACM Web Conference 2026, pages 8553–8556, 2026.

[34] Attila Reiss. PAMAP2 Physical Activity Monitoring. UCI Machine Learning Repository, 2012. DOI: https://doi.org/10.24432/C5NW2H.

[35] Artur Trindade. ElectricityLoadDiagrams20112014. UCI Machine Learning Repository, 2015. DOI: https://doi.org/10.24432/C58C86.

[36] National Oceanic and Atmospheric Administration (NOAA). Noaa global surface summary of the day. https://www.kaggle.com/datasets/ noaa/noaa-global-surface-summary-of-the-day, 2025.

[37] Data Expo 2009: Airline on time data, 2008.

[38] Ryan Cattral and Franz Oppacher. Poker hand dataset. UCI Machine Learning Repository, 2002.

[39] Fabian Pedregosa and et al. Scikit-learn: Machine learning in python. Journal of Machine Learning Research, 12:2825–2830, 2011.

[40] Matthew Jagielski and et al. High accuracy and high fidelity extraction of neural networks. In USENIX Security 20, pages 1345–1362, 2020.

## APPENDIX A PROOF OF LEMMAS

## Proof of Lemma 1

Let M be the black-box MLaaS and M<sup>′</sup> be the extraction model trained to approximate M from query logs. Assume the

bounded fidelity condition holds:

$$
\mathbb { E } \big [ | M ( X ) - M ^ { \prime } ( X ) | \big ] \leq \epsilon .
$$

Fix a feature $x _ { i }$ . In Algorithm 1, feature importance is defined operationally (i.e., from input–output behavior) as the average change in the model output under a controlled perturbation of magnitude δ applied to $x _ { i } .$ . Let $X _ { + i }$ denote the perturbed version of X along $x _ { i } .$ . Consider the (per-sample) importance contributions

$$
\begin{array} { c } { { \Delta _ { i } ^ { M } ( X ) = \left| M ( X _ { + i } ) - M ( X ) \right| , } } \\ { { \Delta _ { i } ^ { M ^ { \prime } } ( X ) = \left| M ^ { \prime } ( X _ { + i } ) - M ^ { \prime } ( X ) \right| . } } \end{array}
$$

By the triangle inequality,

$$
\begin{array} { r l } & { \left| \Delta _ { i } ^ { M ^ { \prime } } ( X ) - \Delta _ { i } ^ { M } ( X ) \right| \leq \left| M ^ { \prime } ( X _ { + i } ) - M ( X _ { + i } ) \right| } \\ & { \qquad + \left| M ^ { \prime } ( X ) - M ( X ) \right| . } \end{array}
$$

Taking expectation and using the fidelity assumption on both X and $X _ { + i }$ gives

$$
\mathbb { E } \big [ | \Delta _ { i } ^ { M ^ { \prime } } ( X ) - \Delta _ { i } ^ { M } ( X ) | \big ] \ \leq \ 2 \epsilon .
$$

Averaging over the samples used by Algorithm 1 yields the same bound for the estimated importance of feature $x _ { i } \colon$

$$
\mathbb { E } \big [ | I ^ { M ^ { \prime } } ( x _ { i } ) - I ^ { M } ( x _ { i } ) | \big ] \ \leq \ 2 \epsilon .
$$

Therefore, the extracted importance vector $\mathbf { I } ^ { M ^ { \prime } }$ deviates from the behavioural importance vector ${ \bf \cal I } ^ { M }$ by a bounded amount. Since the measured importance is induced by perturbations of magnitude δ, the estimator sensitivity is governed by $\delta ,$ yielding

$$
\| \mathbf { I } ^ { M ^ { \prime } } - \mathbf { I } ^ { M } \| \leq f ( \epsilon , \delta ) ,
$$

for some bounded function $f ( \epsilon , \delta )$ that is monotone in ϵ and dependent on the perturbation scale δ and the chosen norm. Hence, under bounded fidelity, the extraction model preserves the feature-importance behaviour of the black-box MLaaS up to a bounded deviation, establishing Lemma 1. □

## Proof of Lemma 2

Recall that FDDS between the baseline window b and current window w is defined as in Eq. (5) of the paper:

$$
\begin{array} { r } { \mathrm { F D D S } ( b , w ) = \underbrace { | | \mu _ { b } - \mu _ { w } | | _ { 2 } ^ { 2 } } _ { m e a n s h i f t } + \underbrace { \mathrm { T r } \Big ( \Sigma _ { b } + \Sigma _ { w } - 2 \sqrt { \Sigma _ { b } \Sigma _ { w } } \Big ) } _ { c o v a r i a n c e s h i f t } . } \end{array}
$$

(i) Lower bounded by the mean shift. Let $\mathcal { C } ( \Sigma _ { b } , \Sigma _ { w } )$ denote the covariance shift term. Since $\Sigma _ { b }$ and $\Sigma _ { w }$ are symmetric positive semi-definite, the trace-based covariance discrepancy is non-negative and cannot reduce the FDDS below the mean shift component. Hence,

$$
\mathrm { F D D S } ( b , w ) = \| \mu _ { b } - \mu _ { w } \| _ { 2 } ^ { 2 } + \mathcal { C } ( \Sigma _ { b } , \Sigma _ { w } ) \geq \| \mu _ { b } - \mu _ { w } \| _ { 2 } ^ { 2 } ,
$$

which shows that FDDS is lower bounded by the mean shift. (ii) Under significant covariance change Consider that the current window undergoes a substantial covariance drift case where the current covariance differs substantially from the baseline, i.e., $\| \Sigma _ { w } - \Sigma _ { b } \|$ is large. Then the product $\Sigma _ { b } \Sigma _ { w }$ changes accordingly, and the square-root interaction term $\sqrt { \Sigma _ { b } \Sigma _ { w } }$ no longer aligns with $\Sigma _ { b } .$ . As a result, the gap

$$
\Sigma _ { b } + \Sigma _ { w } - 2 \sqrt { \Sigma _ { b } \Sigma _ { w } }
$$

becomes larger in the trace sense, so $\mathcal { C } ( \Sigma _ { b } , \Sigma _ { w } )$ increases, which directly increases $\mathrm { F D D S } ( b , w )$ even if $\mu _ { b } \approx \mu _ { w }$

(iii) Under minor distributional change Let us consider the case where the distributional change between the baseline and current windows is minor, meaning $\| \mu _ { b } - \mu _ { w } \|$ is small and $\begin{array} { r } { \Sigma _ { w } ~ \approx ~ \Sigma _ { b } , } \end{array}$ , then the mean shift term is small and $\sqrt { \Sigma _ { b } \Sigma _ { w } } \approx \sqrt { \Sigma _ { b } ^ { 2 } } = \Sigma _ { b }$ , making $\mathcal { C } ( \Sigma _ { b } , \Sigma _ { w } )$ ≈ 0. Therefore, FDDS $( b , w )$ stays small under minor fluctuations. Combining $( \mathrm { i } ) { - } ( \mathrm { i } \mathrm { v } )$ , FDDS is lower bounded by the mean shift and increases with either substantial mean drift or substantial covariance drift, while remaining low under minor changes. This proves Lemma 2. □

## Proof of Lemma 3

Proof. As defined in Eq. (6), Algorithm 2 flags drift when the MDES score exceeds the threshold $\theta _ { m } ,$ i.e.,

$$
\mathrm { M D E S } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } \displaystyle \sum _ { i = 1 } ^ { d } I ( x _ { i } ) \cdot \mathrm { F D D S } ( b _ { i } , w _ { i } ) > \theta _ { m } , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

Consider the MDES defined in Eq. (6), where the cumulative exposure is given by $\begin{array} { r } { \sum _ { i = 1 } ^ { d } I ( x _ { i } ) \cdot \mathrm { \bar { F } D D S } ( b _ { i } , w _ { i } ) } \end{array}$ , which jointly captures the feature importance vector I and the feature-wise statistical shift between b and w. If the statistical drift is predominantly concentrated on features with high importance weights, then the corresponding terms $I ( x _ { i } ) \cdot \mathrm { F D D S } ( b _ { i } , w _ { i } )$ become large, leading to a significant increase in the cumulative weighted exposure. Consequently, the exposure is more likely to exceed the threshold $\theta _ { m }$ , which activates the condition $\mathrm { \ : M D E S ~ = ~ 1 }$ and indicates that the drift is aligned with influential features, thereby characterizing it as real drift. In contrast, if the statistical shift mainly occurs on features with low importance weights, the same $\mathrm { { F D D S } } ( b _ { i } , w _ { i } )$ contributions are attenuated by small $I ( x _ { i } )$ values, keeping the cumulative exposure bounded and unlikely to surpass $\theta _ { m }$ . Under this condition, ${ \mathrm { M D E S } } = 0 .$ , and the drift is interpreted as pseudodrift. Therefore, the MLaaS Drift Exposure Score distinguishes real drift from pseudo-drift based on the alignment between drift magnitude and feature importance. This proves Lemma 3. □

## APPENDIX B DRIFT GENERATION

TABLE 6 DRIFT GENERATION STATISTICS ACROSS DATASETS.
<table><tr><td>Dataset</td><td>No Drift</td><td>Sudden</td><td>Incremental</td><td>Gradual</td><td>Recurrent</td><td>Total Drift</td></tr><tr><td>HAR</td><td>60,000</td><td>24,000</td><td>14,400</td><td>14,000</td><td>24,000</td><td>76,400</td></tr><tr><td>Airline</td><td>9,989</td><td>3,996</td><td>2,398</td><td>2,332</td><td>3,998</td><td>12,724</td></tr><tr><td>Electricity</td><td>45,312</td><td>20,391</td><td>12,235</td><td>11,329</td><td>18,126</td><td>62,081</td></tr><tr><td>Weather</td><td>96,453</td><td>53,050</td><td>31,830</td><td>24,109</td><td>38,583</td><td>147,572</td></tr><tr><td>Poker</td><td>100,000</td><td>40,000</td><td>24,000</td><td>23,336</td><td>40,000</td><td>127,336</td></tr></table>

![](images/44912dd26a39f46f6286d7ed25b85e54574f3843fd01aec9ad7727ffaa8b5dc6.jpg)  
(a)

![](images/94ae37ed1dfdfd4a719e93fd81d87f84e0cd2c6f71457191392b6a5b8f2f4be1.jpg)  
(b)

![](images/2e8724c48d9fc58fb3dc8605ea615c9abf708e382a8a0560cc1cd372535cead0.jpg)  
(c)

![](images/5b79359c911d6a8ad5eec1058c047e11d37d6fd94e6139c500d30f3b612cb41c.jpg)  
(d)

![](images/7ebb44d403a7d569b97ce58f85cf14e9d8bfaae87877244c8a7e4740415eccf0.jpg)  
(e)  
Fig. 8. Drift generation across datasets: (A) HAR, (B) Airline, (C) Weather, (D) Electricity, and (E) Poker.

1) Drift Data Generation: We use the SEA generator [39] to create drifted data streams across the datasets used in our experiments, modelling four common drift categories: sudden, incremental, gradual, and recurrent drift. For each dataset, four full-length drifted streams are generated, one for each drift type. Consequently, the combined generated stream contains four times the number of samples in the original dataset. Table 6 summarizes the stream sizes and drift statistics across all datasets. Specifically, the generated streams contain 240,000 samples for HAR, 39,956 for Airline, 181,248 for Electricity, 385,812 for Weather, and 400,000 for Poker. The generated drift regions comprise 76,400 samples for HAR, 12,724 for Airline, 62,081 for Electricity, $^ { 1 4 7 , 5 7 2 }$ for Weather, and 127,336 for Poker. These drift samples are distributed across sudden, incremental, gradual, and recurrent drift scenarios. For each dataset, we generate one sudden drift event, one gradual drift event, one incremental drift event consisting of five progressively stronger stages, and two recurrent drift events, resulting in five drift episodes per dataset and 25 drift episodes across all datasets. Figure 8 illustrates the drift behaviour across the generated streams and highlights how different drift types evolve over time. For evaluation, each stream is processed using chunks of 250 samples, where each chunk consists of five consecutive blocks of 50 samples. A chunk is labelled as a real-drift chunk when more than 50% of its samples belong to a generated drift region. The visualizations presented in Appendix B show uniformly selected representative chunks from the complete streams and are intended to illustrate drift progression rather than indicate the total number of generated drift events.

![](images/13e85d69c94c76943e2d9abc421ce0b2363da282369a2b4e3b53b3e73d9697fd.jpg)

Deepak Kanneganti is an Associate Lecturer at Curtin University, Australia. He received his Master’s degree in Predictive Analytics from Curtin University, Bentley, Perth, WA, Australia, and his Bachelor’s degree in Electronics and Communication Engineering. He previously served as a Data Science Intern at the Pawsey Supercomputing Research Centre. His research interests include distributed machine learning, federated learning, and explainable artificial intelligence.

![](images/34c8fd6b511cdc86add76fa0a90839e820e20d7d512b000fab9059cfa14aaa6e.jpg)

Sajib Mistry is an Associate Professor at Curtin University, Australia. He received his Ph.D. degree from RMIT University, Australia. Prior to joining Curtin University, he was a Postdoctoral Fellow with the School of Computer Science, University of Sydney, Australia. His research interests include edge and cloud computing, big data, and the Internet of Things. He has published articles in international journals and conferences, including IEEE Transactions on Services Computing, IEEE Transactions on Knowledge and Data Engineering, Communications of the ACM, ICSOC, WISE, and IEEE ICWS. He received the Best Paper Award at ICSOC 2016.

![](images/4c294979e90a0fc2f0ffbb44a9d33519cd1372cb4749120ebbd2a42b0cf106c9.jpg)

Sheik Mohammad Mostakim Fattah is a Lecturer at Curtin University, Australia. He received his Ph.D. degree in Computer Science from the University of Sydney, Australia, an M.Eng. degree in Computer and Information Communication Engineering from Hankuk University of Foreign Studies, South Korea, and a B.Sc. (Hons.) degree in Computer Science and Engineering from the University of Dhaka, Bangladesh. Prior to joining Curtin University, he held academic and research positions at the University of Adelaide, Torrens University, and the

University of Sydney, as well as industry positions at the Korea Electronics Technology Institute and Monist IT Ltd. His research interests include cloud computing, services computing, edge computing, the Internet of Things, and Semantic Web technologies. His work has appeared in leading venues such as IEEE Transactions on Services Computing, ACM Transactions on the Web, ICSOC, and IEEE ICWS. He serves as a reviewer and program committee member for several international conferences and journals.

![](images/82f529c979c3f6e03a3c4c274b1b175d05e0d1f3ce12cf5db067a82dcafec9d8.jpg)

Erik Elmroth (Member, IEEE) is a Professor of Computing Science at Umea University, Sweden.˚ He has served as Head and Deputy Head of the Department of Computing Science for 13 years and as Deputy Director of the National Supercomputer Centre for another 13 years. He established Umea˚ University’s research activities in distributed systems. His experience in management and executive groups of large-scale research initiatives includes the EUR 550 million Wallenberg AI, Autonomous Systems and Software Program and the strategic re-

search area eSSENCE. He has developed two international research strategies for the Nordic Council of Ministers. His international experience includes one year at NERSC, Lawrence Berkeley National Laboratory, University of California, Berkeley, USA, and one semester at the Massachusetts Institute of Technology, Cambridge, MA, USA. He has served as a member of the Swedish Research Council’s Committee for Research Infrastructure, Chair of its expert panel on eScience, and Chair of the Board of the Swedish National Infrastructure for Computing. He is a Lifetime Member of the Royal Swedish Academy of Engineering Sciences and has served as Vice Chair of its Division for Information Technology.

![](images/e94eff99ecca5412c7f1ca48c1ab88cdc9025be7fc5628c54f2dfc60cecff3c3.jpg)

Aneesh Krishna is a Full Professor in the School of Electrical Engineering, Computing and Mathematical Sciences at Curtin University, Australia. He received his Ph.D. degree in Computer Science from the University of Wollongong, Australia. He has held several academic positions, including Lecturer in Software Engineering with the School of Computer Science and Software Engineering at the University of Wollongong from February 2006 to June 2009. His research interests include artificial intelligence for software engineering, model-driven development and evolution, requirements engineering, agent systems, formal methods, data mining, computer vision, machine learning, bioinformatics, and renewable energy systems. He has published more than 200 articles in reputed journals and international conference proceedings.

![](images/c4effc9fd19d9256586557e4e378e8739646d8a5f3356044bd96d4a3bbf716be.jpg)

Monowar Bhuyan (WASP Fellow and Senior Member, IEEE) received his Ph.D. degree in Computer Science and Engineering from Tezpur University, Assam, India. He is currently an Associate Professor in the Department of Computing Science at Umea˚ University, Sweden. He established and leads the Cyber Analytics and Learning Group and is a Senior Member of the Autonomous Distributed Systems Lab. Prior to this, he held academic and research positions at several institutions, including the Nara Institute of Science and Technology, Japan; Assam

Kaziranga University, India; and Umea University, Sweden, at various levels˚ from Junior Scientist to Associate Professor between January 2009 and December 2019. He has published more than 100 papers in leading peerreviewed international journals and conference proceedings and authored the advanced textbook “Network Traffic Anomaly Detection and Prevention” with Springer. His experience in leading and co-leading research projects has attracted more than SEK 40 million in funding from national, European Commission, and international funding agencies. His research interests include machine learning, anomaly detection, systems and AI security, and distributed systems.
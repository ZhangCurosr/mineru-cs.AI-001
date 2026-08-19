# Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection

Bin Li<sup>1∗</sup>, Dongdong Wang<sup>2∗</sup>, Siyang Lu<sup>1†</sup>

<sup>1</sup>Beijing Jiaotong University, Beijing, China

<sup>2</sup>University of Florida, Gainesville, FL, USA

24120434@bjtu.edu.cn, dongdongwang@ufl.edu, sylu@bjtu.edu.cn

Abstract—Online log anomaly detection is critical for maintaining the reliability of large-scale computing systems. Although recent language model-based log anomaly detectors achieve strong detection performance, their confidence estimates remain poorly calibrated. We show that these detectors frequently assign excessive confidence to incorrect predictions, particularly for anomalous logs under severe class imbalance. Moreover, confidence on erroneous predictions remains persistently high even when conventional calibration metrics indicate good calibration, creating a critical reliability gap for operational monitoring systems. To address this issue, we propose Log Reconstruction and Distance (LoRD), a lightweight post-hoc calibration framework for reliable log anomaly detection. LoRD learns predictionroute-specific reliability models from latent representations of correctly classified validation samples and estimates prediction reliability through route-wise reconstruction distances. Based on the estimated reliability, LoRD selectively recalibrates high-risk predictions to suppress overconfident errors while preserving reliable predictions. Extensive experiments on four large-scale log benchmark datasets and multiple language model-based detectors demonstrate that LoRD consistently improves confidence reliability and substantially reduces overconfident anomaly-related errors without sacrificing anomaly detection performance.

Index Terms—Log anomaly detection, model calibration, confi dence reliability, overconfidence

## I. INTRODUCTION

Large-scale data systems are increasingly critical in the era of artificial intelligence and high-performance computing. Maintaining system reliability relies heavily on continuous log monitoring, making log anomaly detection essential for identifying abnormal events and preventing system failures. Existing log anomaly detection approaches have achieved substantial progress in semantic log analysis and anomalous event detection [1]–[4].

Current methods for log anomaly detection can generally be categorized into two paradigms [5]. The first models normal log patterns and identifies deviations from expected behavior as anomalies. The other formulates log anomaly detection as a supervised classification problem, where models directly predict whether a log sequence is normal or anomalous. While the former typically requires less labeled data, the latter often achieves stronger benchmark [6], [7] performance but relies heavily on high-quality annotations and reliable preprocessing pipelines. In this paper, we focus on supervised classificationbased log anomaly detectors and investigate the reliability of their confidence estimation.

However, strong classification performance does not guarantee reliable decision-making. In log anomaly detection, prediction errors are asymmetric in their operational impact [8]. A normal log sequence that is incorrectly classified as anomalous may trigger a false alarm, causing additional inspection costs and potentially contributing to alert fatigue. While such errors are undesirable, they still expose suspicious cases to operators or downstream monitoring modules. By contrast, an anomalous log sequence that is incorrectly classified as normal constitutes a missed anomaly, allowing the underlying abnormal behavior to bypass the detection pipeline. This risk becomes more severe when the detector assigns high confidence to the incorrect normal prediction, as the system may treat the decision as reliable and suppress further intervention. Therefore, high confidence false negatives constitute a central reliability risk for supervised log anomaly detectors.

We first examine the confidence behavior of representative supervised log anomaly detectors. The results reveal a recurring reliability issue: detectors can be confidently wrong. Both false positives and false negatives often receive confidence scores comparable to correctly classified samples, suggesting that detector confidence is not consistently aligned with prediction correctness. This issue is not simply resolved by increasing model complexity. More complex detectors may obtain higher accuracy and more favorable aggregate calibration scores, yet their confidence distributions often remain highly concentrated near one, leaving incorrect predictions overconfident. We also find that anomalous samples tend to exhibit larger calibration errors than normal samples. These observations indicate that strong benchmark performance can still conceal important reliability risks, particularly when anomalous log sequences are confidently predicted as normal.

To mitigate this risk, we design a selector-based calibration framework for supervised log anomaly detectors. Given the output of a pre-trained log anomaly detector, the selector predicts whether the detector’s current prediction is correct or incorrect. Specifically, for each input, we use the detector’s hidden representation as the basis for reliability estimation. Since predictions assigned to the normal class and those assigned to the anomalous class exhibit different error patterns, we construct two prediction routes according to the detector’s output label. For each route, an autoencoder is trained on the hidden representations of correctly classified validation samples, and its reconstruction error is used as a reliability score. Although the two routes follow the same autoencoder-based scoring principle, their calibration objectives are asymmetric. The predicted-normal route prioritizes potential false negatives, while the predicted-anomalous route is treated more conservatively to avoid weakening reliable anomaly predictions. This design reflects the asymmetric risk of log anomaly detection, where missed anomalies are typically more critical than false alarms.

In this work, we first show that supervised log anomaly detectors can remain overconfident on misclassified samples despite strong benchmark performance. We then propose a selector-based calibration framework that predicts whether a detector’s output is correct or incorrect using its predictions and hidden representations. Finally, we demonstrate across multiple datasets and representative detectors that our method achieves a task-oriented calibration trade-off, yielding more reliable confidence estimates for anomalous samples, especially those confidently predicted as normal, while keeping the calibration degradation on normal samples within an acceptable range.

Our contributions are summarized as follows:

• To the best of our knowledge, this is the first work to systematically investigate reliability and confidence calibration in language model-based log anomaly detection. We reveal a persistent overconfidence phenomenon in which state-of-the-art detectors assign excessively high confidence to erroneous anomaly predictions despite exhibiting strong performance under conventional calibration metrics. Through extensive analyses across history window length, LoRA rank, model complexity, and classimbalance mitigation strategies, we show that Confidence on Error (CoE) remains consistently high across models and settings, exposing a critical reliability gap that is overlooked by existing calibration metrics.

• We propose LoRD (Log Reconstruction and Distance), a lightweight post-hoc calibration framework for log anomaly detection. LoRD learns route-specific reconstruc tion models that characterize the reliable conditional latent distributions of normal and anomalous predictions, and leverages reconstruction distances as reliability signals for tri-region confidence calibration. By selectively preserving reliable predictions while suppressing overconfident errors, LoRD directly targets the reliability challenge identified in modern log anomaly detectors.

• We conduct extensive experiments on four large-scale log benchmark datasets using diverse language modelbased log anomaly detectors with varying architectures and model scales. Experimental results demonstrate that LoRD consistently improves reliability and calibration quality across datasets and detectors, effectively reducing overconfident mispredictions while preserving detection performance. These findings highlight the robustness, effectiveness, and generalizability of LoRD for reliabilityaware log anomaly detection.

## II. RELATED WORK

## A. Log Anomaly Detection

Researchers have extensively employed sequence modeling techniques for system log analysis with notable success. Among these applications, log anomaly detection has emerged as a critical task, often formulated as a text classification problem in natural language processing. Driven by advances in deep neural networks, increasingly powerful text classification models have substantially improved the accuracy and robustness of log anomaly detection.

Log anomaly detection aims to identify abnormal system behaviors from log data. Existing studies mainly follow two lines [5]: i) semi-supervised methods and ii) supervised methods. Semi-supervised methods usually learn normal log patterns from historical logs and identify samples that deviate from the learned normality as anomalies. DeepLog [9] is a representative method in this line, which models normal log sequences with an LSTM and detects anomalies based on deviations from predicted execution patterns. LogAnomaly [10] extends this idea by considering both sequential and quantitative anomalies, and further incorporates semantic representations of log templates to better handle unstructured logs. PLELog [11] reduces the reliance on manual labeling by estimating probabilistic labels and learning semantic representations for log anomaly detection.

Supervised methods treat log anomaly detection as a binary classification task, learning discriminative patterns from labeled normal and anomalous sequences. Inspired by TextCNN, Lu et al. [12] apply multiple convolutional filters with different kernel sizes in parallel to capture local patterns at different granularities, achieving strong performance in supervised log anomaly detection. LogRobust [13] combines semantic log-template representations with an attention-based recurrent model to improve robustness to unstable log events. LightLog [14] improves efficiency by applying PCA-based dimensionality reduction to log semantic representations before using a lightweight temporal convolutional network to model sequential dependencies. NeuralLog [15] uses pretrained language models to encode the semantics of log events and adopts a Transformerbased encoder to capture contextual dependencies within log sequences, thereby improving anomaly classification.

Recent studies have also begun to explore the potential of large language models in log anomaly detection. LogLLM [16] follows a semantic-enhanced framework similar to NeuralLog: it first uses pretrained language models such as BERT to encode log sequences, then employs a projector to align the extracted log representations with the input space of LLaMA, and finally applies parameter-efficient fine-tuning techniques such as LoRA for anomaly classification. Another line of work adapts GPT-2 [17] to log anomaly detection through a simpler preprocessing pipeline, directly feeding the processed log-text sequences into GPT-2.

![](images/f48c88e76cdb9bf8acffbccce9eb6879f1cfef2517dcaaa6608942fcc2833c30.jpg)  
Fig. 1. Overview of existing log anomaly detection frameworks. Logs are preprocessed, embedded into semantic representations, encoded using sequential models such as Bi-LSTM or Transformer, and finally used for anomaly prediction.

![](images/c72116e2f318f6ae6ffa5fb862e8b20645872c8533a2cc55dfb10014d4b6a7c1.jpg)  
(a)

![](images/83e471ca96672ac6c2a2c6a4910454a7582cbbcd04a28c5bab99998c2c76f2bf.jpg)  
(b)

![](images/eb726602682a99695baae7fa63563151ea130ea7f9236325722f3edbbf95c1c9.jpg)  
(c)

![](images/d108942d62995eeaaa75583bc2d62a252e3d0530b0b55743036bdf774396ea00.jpg)  
(d)  
Fig. 2. Ablation analysis of factors affecting confidence calibration and error confidence in log anomaly detection. Panels (a)–(c) report Acc, F1, and CoE on the left axis and ECE, NLL, and Brier score on the right axis. Panel (d) reports abnormal-class ECE, Brier score, CoC, and CoE for imbalance handling strategies. Green bars denote higher-is-better metrics, while red bars denote lower-is-better metrics. Label Sm., Over Samp., Under Samp., Bal. Samp., and Class Wtd. denote Label Smoothing, Over-Sampler, Under-Sampler, Balanced Sampler, and Class-Weighted Loss, respectively.

![](images/31ee3ab07296ae01f753f56e1ccb272589258581c953db9c6f6837f2411ea1f2.jpg)  
(a)

![](images/fc25a7c7273206c9877b62b1536b844d9b8dfcc9c22fcd1460f1517e0e6a3bd0.jpg)  
(b)

![](images/c2989fb786dfffdea1e2926a320b41e0dbd28d551a076d8f7c1523982ea0e1f6.jpg)  
(c)

![](images/b56920afea56f73440403527326da2f2420d4b2d87e720654e807555f8024470.jpg)  
(d)  
Fig. 3. Class-wise calibration comparison on the BGL dataset. Panels (a)–(c) compare normal-class and abnormal-class ECE, NLL, and Brier score using a logarithmic axis. Panel (d) compares class-wise confidence on correct predictions and erroneous predictions.

## B. Model Calibration

Model calibration aims to make predictive confidence better reflect empirical correctness. Existing methods can be broadly grouped into three categories: regularization-based training, uncertainty estimation, and post-hoc calibration [18].

Regularization-based methods improve calibration by reducing overconfidence during training. Typical examples include label smoothing [19], confidence penalty [20], mixup [21], focal loss [22], and class-balanced loss [23]. These methods can improve calibration, but require detector retraining and may alter the original detection behavior.

Uncertainty estimation methods quantify predictive uncertainty through approaches such as Monte Carlo dropout [24], deep ensembles [25], and Bayesian neural networks [26]. However, they often require extra computation or multiple forward passes, making them less suitable for lightweight posthoc calibration.

Post-hoc calibration methods adjust the output probabilities of a trained model without modifying its parameters. Representative methods include Temperature Scaling and Logistic Scaling [27], Beta Scaling [28], and Selective Scaling [29]. These methods improve confidence estimation through probabilitylevel mappings or auxiliary calibration models, and serve as strong post-hoc calibration baselines.

## III. PRELIMINARIES

## A. Problem Formulation

We study supervised log anomaly detection. The problem is formulated as a binary confidence calibration problem. Given an input log instance $x _ { i }$ with binary indicator label $y _ { i } \in \{ 0 , 1 \}$ where $y _ { i } = 0$ denotes a normal sample and $y _ { i } = 1$ denotes an anomalous sample, a detector $f _ { \theta }$ produces a scalar anomaly confidence score:

$$
p _ { i } = \sigma ( f _ { \theta } ( x _ { i } ) ) , \quad p _ { i } \in ( 0 , 1 ) ,\tag{1}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function and $p _ { i }$ represents the predicted confidence that $x _ { i }$ is anomalous. The final prediction is obtained by applying a decision threshold τ:

$$
\hat { y } _ { i } = \left\{ { \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } p _ { i } \geq \tau , } \\ { 0 , } & { \mathrm { i f ~ } p _ { i } < \tau . } \end{array} } \right.\tag{2}
$$

Based on the relationship between the predicted label $\hat { y } _ { i }$ and the ground-truth label $y _ { i } ,$ , predictions are partitioned into four categories: true negatives (TN), false positives (FP), true positives (TP), and false negatives (FN). This formulation enables systematic analysis of prediction confidence and calibration behavior across correct and incorrect anomaly predictions.

## B. Calibration Metrics

Besides accuracy and F1 score, we evaluate model reliability using several widely adopted confidence calibration metrics for log anomaly detection. For example, negative log-likelihood (NLL) evaluates the quality of probabilistic predictions by measuring the likelihood assigned to the ground-truth labels. Lower NLL indicates better calibrated confidence estimates. To provide a comprehensive assessment of calibration quality, we further consider the following metrics that capture confidence– accuracy alignment, probabilistic prediction quality, and confidence behavior on both correct and incorrect predictions.

Expected Calibration Error (ECE). ECE [30] measures the discrepancy between prediction confidence and empirical accuracy by partitioning predictions into M fixed-width confidence bins:

$$
\mathrm { E C E } = \sum _ { m = 1 } ^ { M } \frac { | B _ { m } | } { N } \left| \operatorname { a c c } ( B _ { m } ) - \operatorname { c o n f } ( B _ { m } ) \right| ,\tag{3}
$$

where $B _ { m }$ denotes the m-th confidence bin, N is the total number of samples, and $\operatorname { a c c } ( B _ { m } )$ and $\operatorname { c o n f } ( B _ { m } )$ represent the empirical accuracy and average confidence within the bin, respectively.

Brier Score (BS). BS measures the mean squared error between predicted probabilities and ground-truth labels:

$$
\mathrm { B S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( p _ { i } - y _ { i } ) ^ { 2 } .\tag{4}
$$

Unlike ECE, which evaluates calibration by comparing confidence with empirical accuracy, BS measures overall probability quality by directly penalizing probability estimation errors. As a result, BS reflects both calibration and class discrimination. However, similar to ECE, BS aggregates over all predictions and may therefore overlook confidence behavior on rare but critical mispredictions.

Confidence on Error (CoE). CoE measures the average confidence assigned to misclassified samples:

$$
\mathrm { C o E } = \frac { 1 } { | \mathcal { E } | } \sum _ { i \in \mathcal { E } } c _ { i } ,\tag{5}
$$

where $\mathcal { E } = \{ i \mid \hat { y } _ { i } \neq y _ { i } \}$ denotes the set of incorrectly classified samples and $c _ { i }$ is the confidence score of sample i. A high CoE indicates that the model is systematically overconfident on its mistakes — a particularly concerning failure mode in anomaly detection, where missed abnormal cases carry high operational cost. Unlike ECE and BS, which aggregate over all predictions, CoE isolates confidence behavior specifically on failure cases, providing a targeted diagnostic that the other metrics do not directly expose. Conversely, Confidence on Correct (CoC) measures the average confidence assigned to correctly classified samples.

## IV. EMPIRICAL ANALYSIS OF CONFIDENCE CALIBRATION IN LOG ANOMALY DETECTION

## A. History Length

We analyze the influence of sliding-window history size while fixing the stride to 1. As shown in Figure 2(a), increasing the history length produces only modest variations in predictive performance, with Acc remaining above 98% and F1 remaining above 0.84 across all settings. Although ECE and Brier score exhibit noticeable changes as the context length varies, CoE remains consistently high, ranging from 0.90 to 0.98. Interestingly, larger history windows reduce CoE more substantially than they improve conventional calibration metrics. While ECE increases from 0.13 to 0.17 as the history length grows from 5 to 30, CoE decreases from 0.98 to 0.90. This discrepancy suggests that average calibration metrics do not adequately reflect confidence behavior on erroneous predictions. Even when additional contextual information improves reliability, the model continues to assign excessively high confidence to incorrect predictions. Therefore, longer temporal context alone is insufficient to resolve the overconfidence problem.

## B. LoRA Rank

Figure 2(b) presents the impact of LoRA rank on calibration behavior. Increasing the LoRA rank substantially improves predictive performance, with F1 increasing from 0.61 at rank 4 to 0.94 at rank 32. Accuracy exhibits a similar trend, indicating that larger adaptation capacity enables the model to capture richer task-specific patterns. However, calibrationrelated metrics remain largely unchanged. ECE and Brier score stay nearly constant across all LoRA ranks, while CoE remains extremely high, approaching 1.0 regardless of adaptation capacity. These results suggest that increasing model expressiveness primarily improves predictive accuracy but provides limited benefit for mitigating overconfident errors. Even highly accurate models continue to assign near-maximal confidence to misclassified samples. This observation further demonstrates that improving predictive performance alone is insufficient for reliable confidence estimation.

## C. Model Complexity

Figure 2(c) investigates the effect of model complexity by varying the number of trainable parameters in a multilayer perceptron. Performance metrics remain nearly identical across all model sizes, with Acc and F1 consistently exceeding 0.999. Likewise, ECE and Brier score remain extremely small, suggesting excellent calibration according to conventional metrics. Surprisingly, CoE increases monotonically as model size grows, rising from 0.90 to nearly 0.99. This finding indicates that larger models become increasingly confident on their mistakes despite exhibiting near-perfect conventional calibration scores. The result highlights a potential trade-off between model capacity and confidence reliability, where additional representational power may amplify overconfidence on rare erroneous predictions. Consequently, model scaling alone does not improve confidence quality and may even exacerbate error confidence.

## D. Solutions for Imbalances

Figure 2(d) compares several commonly used strategies for addressing class imbalance. Traditional sampling-based methods, including over-sampling, under-sampling, and balanced sampling, produce only modest improvements in calibration behavior. Although these approaches slightly reduce CoE rela tive to the baseline, error confidence remains consistently high. Among all methods, class-weighted loss achieves the largest reduction in CoE, decreasing error confidence from 0.996 to 0.718 while simultaneously reducing ECE and Brier score. Focal loss also improves calibration relative to the baseline but remains less effective than class weighting. These results suggest that imbalance-aware optimization is more effective than data-level resampling for reducing overconfidence. Nevertheless, even the strongest baseline continues to exhibit substantial error confidence, indicating that class imbalance treatment alone cannot fully address the overconfidence problem.

## E. Class-wise Calibration

Figure 3 compares calibration behavior between normal and abnormal logs. A consistent class-wise disparity is observed across all detectors. In terms of calibration metrics, abnormal logs exhibit substantially larger ECE, NLL, and Brier scores than normal logs, indicating that confidence estimates for anomaly predictions are considerably less reliable. The gap is particularly pronounced for LightLog, NeuralLog, and GPT2, where abnormal-class calibration errors exceed those of the normal class by more than an order of magnitude. The confidence decomposition in Figure 3(d) further reveals the source of this disparity. While both normal and abnormal correctly classified samples (Nor. CoC and Abn. CoC) generally receive high confidence, incorrectly classified abnormal samples (Abn. CoE) also receive surprisingly high confidence scores, often exceeding 0.9. In contrast, confidence on incorrectly classified normal samples (Nor. CoE) remains substantially lower and closer to the uncertainty region. This indicates that language model-based log anomaly detectors tend to be particularly overconfident when making anomaly-related mistakes, especially false negatives. These observations suggest that calibration errors are not uniformly distributed across classes. Instead, the primary calibration challenge arises from overconfident mispredictions involving anomalous logs. Consequently, improving confidence reliability for anomaly predictions is likely to yield the greatest gains in overall calibration quality, motivating the anomaly-focused route-wise calibration strategy adopted by LoRD.

## F. Characterizing Error Confidence

The preceding analyses reveal a consistent pattern across different architectures, hyperparameters, and training strategies. Although ECE and Brier score often indicate moderate calibration performance, confidence on erroneous predictions (CoE) remains persistently high, frequently exceeding 0.9. This suggests that models can appear well calibrated on average while still assigning near-maximal confidence to incorrect predictions. This discrepancy arises because conventional calibration metrics are dominated by correctly classified samples and therefore provide limited insight into confidence behavior on rare prediction errors. For log anomaly detection, such overconfident failures are particularly problematic because they may lead to missed anomalies or incorrect operational decisions. These observations consistently show that CoE remains a persistent challenge regardless of model architecture, history window length, adaptation rank, model complexity, or imbalancehandling strategy. This finding suggests that confidence on erroneous predictions is an important yet largely overlooked aspect of reliability in log anomaly detection. Consequently, we introduce CoE as a complementary reliability metric and develop a reliability-aware calibration framework that explicitly optimizes confidence quality on misclassified samples.

## V. LORD SCALING

In this section, we present Log Reconstruction and Distance (LoRD), a lightweight calibration framework designed to improve reliability in log anomaly detection. Building on the observation that confidence on erroneous predictions remains persistently high across different detectors and training settings, LoRD estimates prediction reliability from latent representations and uses the resulting reliability scores to recalibrate confidence. Specifically, LoRD combines route-wise reconstruction modeling, asymmetric threshold and margin selection, and selective confidence adjustment to better align prediction confidence with underlying prediction reliability.

Given an input log sequence $x _ { i } ,$ a trained anomaly detector produces a hidden representation $\mathbf { h } _ { i } \in \mathbb { R } ^ { d } ,$ , a predicted probability vector $\mathbf { p } _ { i }$ , and a predicted label $\hat { y } _ { i } \in \{ 0 , 1 \}$ . The true label is denoted as y , where 0 indicates normal and 1 indicates anomalous. Our goal is to recalibrate the detector confidence so that high-risk predictions, especially anomalous samples incorrectly predicted as normal, receive lower confidence while preserving the detector’s original prediction labels.

![](images/807b5f5d157322ab99b11ac3b890abbe9902c781980a62d5cc6f363855fc32dd.jpg)  
Fig. 4. Overview of LoRD with AE. Correctly classified validation samples are separated by prediction route to train route-specific AE. Reliability distances are then used to partition predictions into route-aware regions, which are subsequently mapped to confidence calibration policies.

## A. Representation Learning

LoRD estimates prediction reliability through route-specific conditional representation learning in the detector latent space. Let $\mathbf { h } _ { i }$ denote the hidden representation of sample i, and let $\hat { y } _ { i } , y _ { i } \in \{ 0 , 1 \}$ denote the predicted and ground-truth labels, respectively. The key observation is that reliable normal predictions and reliable anomaly predictions follow distinct latent distributions,

$$
p ( \mathbf { h } \mid \hat { y } = 0 , y = 0 ) \neq p ( \mathbf { h } \mid \hat { y } = 1 , y = 1 ) .\tag{6}
$$

Consequently, fitting a single reliability model to all reliable samples may obscure route-specific reliability structures and reduce the separability of prediction errors. To address this issue, LoRD learns route-specific reliability models that capture the conditional latent distributions of reliable predictions:

$$
\mathcal { M } _ { r } \sim p ( \mathbf { h } \mid \hat { y } = r , y = r ) , \qquad r \in \{ 0 , 1 \} ,\tag{7}
$$

where $\mathcal { M } _ { r }$ denotes the reliable latent manifold associated with prediction route $^ { r } \cdot$ For each route, LoRD learns a representation function $f _ { r } ( \cdot )$ and a reconstruction function $g _ { r } ( \cdot )$ . Given a sample assigned to route $\hat { y } _ { i }$ , the corresponding route-specific model generates a reconstructed latent representation:

$$
\hat { \mathbf { h } } _ { i } = g _ { \hat { y } _ { i } } \big ( f _ { \hat { y } _ { i } } ( \mathbf { h } _ { i } ) \big ) .\tag{8}
$$

In this work, $f _ { r } ( \cdot )$ and $g _ { r } ( \cdot )$ are implemented as encoder– decoder networks, although alternative representation learning

architectures can be incorporated. We estimate the reliability distance of a sample using its reconstruction error:

$$
d _ { i } = \left\| \mathbf { h } _ { i } - \hat { \mathbf { h } } _ { i } \right\| _ { 2 } ^ { 2 } ,\tag{9}
$$

where $d _ { i }$ measures the deviation of $\mathbf { h } _ { i }$ from the route-specific reliable manifold $\mathcal { M } _ { \hat { y } _ { i } }$ . A small reliability distance indicates that the sample is well aligned with the reliable conditional distribution of its prediction route, whereas a large distance suggests atypical latent behavior. Consequently, false negatives tend to deviate from the reliable normal manifold, while false positives tend to deviate from the reliable anomaly manifold. The resulting reliability distance provides a unified measure of prediction reliability and serves as the basis for subsequent confidence calibration.

## B. Route-Aware Reliability Partitioning

LoRD constructs route-specific calibration regions based on the reliability distance from the corresponding reliable manifold. To capture route-specific error patterns, LoRD learns two thresholds, $\tau _ { r } ^ { ( 1 ) } < \tau _ { r } ^ { ( 2 ) }$ , for each prediction route and partitions samples into low-, mid-, and high-distance regions:

$$
\begin{array} { r l } & { \mathcal { R } _ { r } ^ { \mathrm { l o w } } = \{ d _ { i } \leq \tau _ { r } ^ { ( 1 ) } \} , } \\ & { \mathcal { R } _ { r } ^ { \mathrm { m i d } } = \{ \tau _ { r } ^ { ( 1 ) } < d _ { i } \leq \tau _ { r } ^ { ( 2 ) } \} , } \\ & { \mathcal { R } _ { r } ^ { \mathrm { h i g h } } = \{ d _ { i } > \tau _ { r } ^ { ( 2 ) } \} . } \end{array}\tag{10}
$$

Here, $\tau _ { r } ^ { ( 1 ) }$ and $\tau _ { r } ^ { ( 2 ) }$ are selected on the validation set according to the route-specific target recall $R _ { r }$ and flagged rate $\rho _ { r }$ . In practice, $R _ { r }$ and $\rho _ { r }$ are selected according to the operational risk tolerance of the application. Larger $R _ { r }$ increases the likelihood of identifying unreliable predictions, whereas smaller $\rho _ { r }$ limits unnecessary calibration of reliable samples. The normal route $( r = 0 )$ prioritizes false-negative coverage, while the abnormal route $\mathbf { \Phi } ( r \mathbf { \Phi } = \mathbf { \Phi } 1 )$ prioritizes preserving reliable anomaly alarms.

This partition reflects the route-dependent reliability structure. In the normal route, samples with small reliability distance are more likely to be reliable true negatives, whereas samples with large distance are more likely to be false negatives. In the abnormal route, samples with small reliability distance are more likely to be reliable true positives, whereas samples with large distance are more likely to be false positives. Samples in the mid-distance region are treated as uncertain and are left for the reject option in the subsequent calibration policy.

## C. Calibration Policies

Based on the route-aware calibration map, LoRD assigns a calibration action to each route-region combination, as summarized in Table I. Predictions in the uncertain region are left unchanged through a reject option, while predictions in the reliable and high-risk regions undergo confidence adjustment according to their estimated log risk. The policy is designed to increase confidence for samples that are more likely to be safe logs and suppress confidence for samples that are more likely to be anomalous logs.

Based on Table I, LoRD defines a route-region calibration policy $\pi ( r , z )$ , where r denotes the prediction route and z denotes the reliability-distance region. Here, $p _ { i }$ denotes the confidence assigned to the detector’s original predicted label. LoRD is label-preserving: it recalibrates the confidence score but does not recompute the predicted label after calibration.

The Identity policy leaves $p _ { i }$ unchanged. For the remaining regions, LoRD either directly assigns a target confidence through HardAssign or gradually moves $p _ { i }$ toward a target confidence through SoftPull. The target confidence $q _ { r , z }$ is determined by the selected route-region policy. Specifically, the normal low-distance region uses $q _ { r , z } = 1$ , the normal highdistance region uses $q _ { r , z }  0 . 5 + \epsilon .$ , the abnormal low-distance region uses $q _ { r , z }  1$ , and the abnormal high-distance region uses $q _ { r , z } = 0 . 5 + \epsilon$

For SoftPull, the adjustment strength is determined by the distance to the nearest calibration boundary:

$$
\alpha _ { i } = 1 - \exp \left( - \frac { \Delta _ { i } } s \right) , \qquad \alpha _ { i } \in [ 0 , 1 ] ,\tag{11}
$$

where $\Delta _ { i }$ denotes the boundary distance and s is a scale parameter determined by the corresponding margin. The recalibrated confidence is computed as

$$
p _ { i } ^ { \mathrm { n e w } } = ( 1 - \alpha _ { i } ) p _ { i } + \alpha _ { i } q _ { r , z } ,\tag{12}
$$

where $q _ { r , z }$ is specified by the selected calibration policy. Thus, samples close to the boundary are only mildly adjusted, while samples farther away are pulled more strongly toward the target confidence. For HardAssign, LoRD directly sets $p _ { i } ^ { \mathrm { n e w } } = q _ { r , z }$

## D. Algorithm

We summarize the calibration framework in Algorithm 1. LoRD performs route-specific confidence calibration by modeling the latent representations of reliable predictions separately for the normal and abnormal routes. First, the base detector is applied to a validation set to obtain predicted labels, confidence scores, and hidden representations. For each route $r \in \{ 0 , 1 \}$ , correctly classified samples are treated as routereliable instances, while misclassified samples form the routeerror set. A route-specific autoencoder is then trained on the latent representations of reliable samples to learn the characteristic feature distribution of trustworthy predictions. Reconstruction distances are computed for both reliable and error samples, and these distances are used to construct low-, medium-, and high-reliability regions according to predefined route-specific targets. During inference, each test sample is assigned to its predicted route, and its reconstruction distance is evaluated using the corresponding route autoencoder. The resulting distance serves as a reliability indicator that is mapped through a calibration policy to adjust the detector’s original confidence score, producing calibrated probabilities that better reflect prediction reliability.

TABLE I  
ROUTE-REGION CALIBRATION POLICY MAP π.
<table><tr><td>Route</td><td> $\mathcal { R } _ { r } ^ { \mathrm { l o w } }$ </td><td> $\mathcal { R } _ { r } ^ { \mathrm { m i d } }$ </td><td> $\mathcal { R } _ { r } ^ { \mathrm { h i g h } }$ </td></tr><tr><td>Normal  $( r = 0 )$ </td><td>HardAssign:  $p = 1$ </td><td>Identity</td><td>SoftPull:  $p  0 . 5 + \epsilon$ </td></tr><tr><td>Abnormal  $( r = 1 )$ </td><td>SoftPull:  $p  1$ </td><td>Identity</td><td>HardAssign:  $p = 0 . 5 + \epsilon$ </td></tr></table>

Algorithm 1 LoRD Calibration   
Require: Detector $f _ { \theta } ,$ validation set $\mathcal { D } _ { s e l } .$ test set $\mathcal { D } _ { t e s t } ,$ , route targets   
$\{ ( R _ { r } , \rho _ { r } ) \} _ { r \in \{ 0 , 1 \} }$ , calibration policy π   
Ensure: Calibrated probabilities $\mathbf { p } _ { i } ^ { n e w }$   
1: Obtain $\left( \hat { y } _ { i } , y _ { i } , \mathbf { p } _ { i } , \mathbf { h } _ { i } \right)$ from $f _ { \theta } ( \mathcal { D } _ { s e l } )$   
2: $\hat { y } _ { i } , y _ { i } \in \{ 0 , 1 \}$ , where 0 is normal and 1 is abnormal   
3: for $r \in \{ 0 , 1 \}$ do   
4: Define route-reliable set $\mathcal { T } _ { r } = \{ i : \hat { y } _ { i } = r , ~ y _ { i } = r \}$   
5: Define route-error set $\mathcal { E } _ { r } = \{ i : \dot { y } _ { i } = r , ~ y _ { i } \neq r \}$   
6: Train route autoencoder $( f _ { r } , \mathbf { \dot { g } } _ { r } ) \operatorname { o n } \left\{ \mathbf { h } _ { i } : i \in \dot { \mathcal { T } } _ { r } \right\}$   
7: Compute $d _ { i } ^ { r } = \lVert \mathbf { h } _ { i } - g _ { r _ { * } } ( \mathbf { \dot { f } } _ { r } ( \mathbf { \check { h } } _ { i } ) ) \rVert _ { 2 } ^ { 2 }$ for $i \in \mathcal { T } _ { r } \cup \mathcal { E } _ { r }$   
8: Update $\mathcal { R } _ { r } ^ { l _ { o w } } , \ddot { \mathcal { R } } _ { r } ^ { m i d } , \dot { \mathcal { R } } _ { r } ^ { h i g h }$ under $( R _ { r } , \rho _ { r } )$ with ${ \mathcal { E } } _ { r }$   
9: end for   
10: for $i \in \mathcal { D } _ { t e s t }$ do   
11: Obtain $\left( \hat { y } _ { i } , \mathbf p _ { i } , \mathbf h _ { i } \right)$ from $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { i } )$   
12: Set route $r  \hat { y } _ { i }$   
13: Compute $d _ { i } ^ { r } = \lVert  { \mathbf { h } } _ { i } - g _ { r } ( f _ { r } (  { \mathbf { h } } _ { i } ) ) \rVert _ { 2 } ^ { 2 }$   
14: Calibrate $\mathbf { p } _ { i }$ with π given $d _ { i } ^ { r }$   
15: end for

## VI. EXPERIMENTS

## A. Datasets

We evaluate our method on four large-scale supercomputing log datasets: BGL [31], Spirit [31], Liberty [31], and Thunderbird [31], covering anomaly ratios ranging from 0.49% to 32.01%. Following prior work, we use 4.7M log messages from BGL, the first 5M log messages from Spirit and Liberty, and 10M log messages from Thunderbird. To prevent information leakage, each dataset is chronologically partitioned into nonoverlapping training, detector-validation, selector-validation, and test splits with a ratio of $7 : 0 . 5 : 0 . 5 : 2$ . The training split is used to optimize the base detector, the detector-validation split is used for early stopping and checkpoint selection, and the selector-validation split is used for calibrator training and threshold selection. When route-specific validation errors are unavailable in the selector-validation split, the detectorvalidation split is used as a fallback. The test split is reserved exclusively for final evaluation.

## B. Log Anomaly Detectors

We evaluate five widely used supervised log anomaly detectors covering both conventional deep learning and language model-based approaches: TextCNN [12], which employs multi-scale one-dimensional convolutions to capture local log patterns; LogRobust [13], which models log sequences using an attention-enhanced bidirectional recurrent network; LightLog [14], which combines compressed semantic log representations with a lightweight temporal convolutional architecture; NeuralLog [15], which leverages pretrained language-model embeddings and a Transformer encoder to capture contextual dependencies; and GPT2 [17], which adapts a pretrained large language model for log anomaly detection through direct sequence modeling of log text. Unless otherwise specified, all detectors are evaluated at the log-sequence level using a sliding-window configuration with history length 10 and stride 1.

## C. Baseline Calibration Methods

We compare LoRD against five representative post-hoc calibration methods. Temperature Scaling (TempS) [27] calibrates confidence using ${ \bf p } _ { i } ^ { \mathrm { T S } } = \mathrm { s o f t m a x } ( { \bf z } _ { i } / \bar { T } )$ , where T is optimized on a validation set. Logistic Scaling (LogS) [27] generalizes TS through a learnable affine transformation, $\mathbf { p } _ { i } ^ { \mathrm { L S } } = \mathrm { s o f t m a x } ( \mathbf { w } \odot$ ${ \bf z } _ { i } + { \bf b } )$ . Beta Scaling (BetaS) [28] performs probabilitylevel calibration using $p _ { i } ^ { \mathrm { B S } } = \sigma ( a \log p _ { i } + b \log ( 1 - p _ { i } ) + c )$ Selective Scaling (SeleS) [29] applies separate temperatures to selected high-risk and low-risk predictions based on outputlevel signals such as confidence, entropy, or logit margin. Ensembling (Ens.) [29] averages predictions from multiple independently trained detectors, $\begin{array} { r } { { \bf p } _ { i } ^ { \mathrm { E n s } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } { \bf p } _ { i } ^ { ( k ) } } \end{array}$ . All calibration parameters are optimized using the validation set.

## D. Results

1) Performance: The main experiments show that LoRD consistently improves calibration reliability for anomalous log detection across datasets and base detectors. In Table II, LoRD obtains the lowest CoE in nearly every dataset–model combination, indicating that it effectively suppresses overconfident misclassifications without degrading the detector’s overall discrimination ability. Compared to standard post-hoc methods such as Temperature Scaling, Logistic Scaling, Beta Scaling, Selective Scaling, and Ensembling, LoRD yields substantially lower error confidence while preserving reliable anomaly alarms. This behavior is pronounced for language-model-based and high-capacity detectors, where LoRD reduces CoE from values near one to values close to 0.5 in many settings.

LoRD also maintains high confidence on correctly classified anomalous samples, as reflected by its robustness in CoC values.

This demonstrates that the proposed route-wise reconstruction mechanism selectively lowers confidence on risky predictions while retaining trustworthy anomaly detections. These results confirm that LoRD provides a task-driven calibration trade-off that is well suited for supervised log anomaly detection.

2) Ablation study: Table III first evaluates the necessity of route-specific reliability modeling in LoRD. Replacing the route-specific autoencoders with a single shared autoencoder (SingleAE) consistently degrades abnormal calibration across all datasets and detectors. For example, with LogRobust on Spirit, Abn. CoE increases from 0.508 to 0.516, while with NeuralLog on Liberty it increases from 0.680 to 0.788. These results suggest that predicted-normal and predictedabnormal samples follow distinct reliability patterns in the latent space. A shared reconstruction model must simul taneously characterize both distributions, which may blur route-dependent reliability structures and reduce its ability to distinguish reliable predictions from potential errors. In contrast, route-specific modeling allows each autoencoder to learn a more homogeneous reliability manifold, resulting in improved separation between reliable and unreliable predictions. This observation is consistent with our hypothesis that normal and abnormal prediction routes exhibit different latent reliability characteristics and therefore benefit from separate modeling.

We next evaluate the contributions of the two remaining components of LoRD: the reject region mechanism and the distance aware soft calibration strategy. Because confidence calibration involves multiple objectives, including preserving confidence on correctly classified samples while suppressing confidence on misclassified samples, we introduce two aggregate measures to summarize overall calibration quality. Let $\mathbf { x } = ( x _ { 1 } , \dots , x _ { m } )$ denote a set of calibration objectives. For each objective $x _ { j }$ , we define a target value $t _ { j }$ , where $t _ { j } = 1$ for correct prediction confidence metrics and $t _ { j } = 0 . 5$ for error confidence metrics. The objective deviation vector is defined as $\mathbf { z } = \left( | x _ { 1 } - t _ { 1 } | , \ldots , | x _ { m } - t _ { m } | \right)$ . We then define the aggregate calibration distance as $D = \| \mathbf { z } \| _ { 2 }$ . To obtain a normalized relative score, we further define $\begin{array} { r } { C = \frac { \| \mathbf { z } - \mathbf { z } ^ { - } \| _ { 2 } } { \| \mathbf { z } \| _ { 2 } + \| \mathbf { z } - \mathbf { z } ^ { - } \| _ { 2 } } } \end{array}$ ,where $\mathbf { z } ^ { - }$ denotes the worst achievable deviation vector. Smaller D and larger C indicate better overall calibration quality.

Tables IV and V show that removing either the reject region or the soft calibration component consistently degrades overall calibration quality. For example, on Spirit with LogRobust, LoRD achieves the best performance with $D = 0 . 4 0 5$ and $C = 0 . 7 8 3$ , outperforming both w/o Reject $( D = 0 . 4 6 8 , C =$ 0.754) and w/o Soft $( D = 0 . 4 1 5 , C = 0 . 7 7 6 )$ . Similar trends are observed on Liberty and with NeuralLog, indicating that the two components provide complementary benefits. These results demonstrate that route specific reliability modeling, the reject region mechanism, and distance aware soft calibration work together to improve confidence reliability without sacrificing anomaly detection performance.

## VII. CONCLUSION

We propose LoRD, a lightweight post-hoc calibration framework for supervised log anomaly detection. By training separate autoencoders for predicted-normal and predictedanomalous routes, LoRD captures route-specific reliability signals from detector latent representations and selectively recalibrates high-risk predictions while preserving the original detection labels. Experiments on multiple benchmark datasets and representative detectors show that LoRD effectively reduces confidence on misclassified anomalous samples, maintains trustworthy anomaly alarms, and outperforms standard scaling and ensemble-based calibration baselines. These results demonstrate the effectiveness of task-oriented, route-aware calibration for improving the reliability of real-world log anomaly detection

TABLE II  
CALIBRATION PERFORMANCE MEASURED BY COE ON ANOMALOUS LOG DETECTION ACROSS BENCHMARK DATASETS AND REPRESENTATIVE DETECTORS.REPORTED VALUES DENOTE MEAN COE WITH STANDARD DEVIATION OVER MULTIPLE RUNS, WHERE LOWER VALUES INDICATE BETTER PERFORMANCE.THE BEST AND SECOND-BEST RESULTS IN EACH ROW ARE HIGHLIGHTED IN BOLD AND UNDERLINED ITALICS.
<table><tr><td>Dataset</td><td>Model</td><td>Uncal</td><td>TempS</td><td>LogS</td><td>BetaS</td><td>SeleS</td><td>Ens.</td><td>LoRD</td></tr><tr><td rowspan="5">BGL</td><td>TextCNN</td><td> $0 . 9 7 7 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 9 7 6 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 9 8 8 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 9 8 9 _ { \pm 0 . 0 0 3 }$ </td><td> $0 . 9 6 8 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 9 6 6 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 5 4 0 { \scriptstyle \pm 0 . 0 0 6 } }$ </td></tr><tr><td>LogRobust</td><td> $0 . 9 8 7 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 9 7 3 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $0 . 9 8 8 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 9 8 8 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 9 0 7 { \scriptstyle \pm 0 . 0 9 8 }$ </td><td> $\overline { { 0 . 7 4 8 _ { \pm 0 . 1 6 3 } } }$ </td><td> $\mathbf { 0 . 5 7 7 { \scriptstyle \pm 0 . 0 9 2 } }$ </td></tr><tr><td>LightLog</td><td> $0 . 8 2 9 _ { \pm 0 . 0 2 2 }$ </td><td> $0 . 9 8 3 _ { \pm 0 . 0 0 7 }$ </td><td> $0 . 9 6 7 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 9 7 9 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 8 3 1 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $0 . 9 9 8 _ { \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 7 0 2 { \scriptstyle \pm 0 . 0 3 6 } }$ </td></tr><tr><td>NeuralLog</td><td> $\overline { { 0 . 9 9 0 { \scriptstyle \pm 0 . 0 0 2 } } }$ </td><td> $0 . 9 7 5 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 9 7 4 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 9 8 8 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 8 4 7 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $0 . 9 8 5 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 5 0 9 { \scriptstyle \pm 0 . 0 0 2 } }$ </td></tr><tr><td>GPT2</td><td> $0 . 9 9 8 _ { \pm 0 . 0 0 0 }$ </td><td> $0 . 9 9 2 _ { \pm 0 . 0 0 0 }$ </td><td> $0 . 9 9 3 _ { \pm 0 . 0 0 0 }$ </td><td> $0 . 9 9 1 _ { \pm 0 . 0 0 0 }$ </td><td> $\overline { { 1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 } } }$ </td><td> $0 . 9 9 8 _ { \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 5 6 6 _ { \pm 0 . 0 0 2 } }$ </td></tr><tr><td rowspan="5">Spirit</td><td>TextCNN</td><td> $0 . 8 9 0 { \scriptstyle \pm 0 . 0 5 5 }$ </td><td> $0 . 8 6 2 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $0 . 7 7 6 { \scriptstyle \pm 0 . 0 9 1 }$ </td><td> $0 . 8 3 3 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td> $0 . 8 5 8 { \scriptstyle \pm 0 . 0 3 9 }$ </td><td> $0 . 9 2 6 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $\mathbf { 0 . 5 5 6 { \scriptstyle \pm 0 . 0 2 2 } }$ </td></tr><tr><td>LogRobust</td><td> $0 . 9 5 7 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 9 3 6 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $\overline { { 0 . 9 2 9 _ { \pm 0 . 0 0 9 } } }$ </td><td> $0 . 9 5 0 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 9 3 0 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $0 . 9 5 1 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $\mathbf { 0 . 5 0 6 _ { \pm 0 . 0 0 4 } }$ </td></tr><tr><td>LightLog</td><td> $0 . 9 5 9 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 9 8 3 _ { \pm 0 . 0 0 7 }$ </td><td> $\overline { { 0 . 9 6 7 _ { \pm 0 . 0 0 9 } } }$ </td><td> $0 . 9 7 9 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 9 2 9 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 9 4 8 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $\mathbf { 0 . 5 1 4 { \scriptstyle \pm 0 . 0 1 4 } }$ </td></tr><tr><td>NeuralLog</td><td> $0 . 9 9 6 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 9 7 5 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 9 7 4 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 9 8 8 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $\overline { { 0 . 9 7 2 _ { \pm 0 . 0 0 9 } } }$ </td><td> $0 . 9 9 1 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 5 0 8 { \scriptstyle \pm 0 . 0 0 7 } }$ </td></tr><tr><td>GPT2</td><td> $0 . 9 3 6 _ { \pm 0 . 0 0 0 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 9 8 2 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 9 4 2 _ { \pm 0 . 0 0 0 }$ </td><td> $\overline { { 0 . 9 9 8 _ { \pm 0 . 0 0 1 } } }$ </td><td> $\underline { { 0 . 9 3 6 _ { \pm 0 . 0 0 1 } } }$ </td><td> $\mathbf { 0 . 5 0 0 { \scriptstyle \pm 0 . 0 0 0 } }$ </td></tr><tr><td rowspan="5">Liberty</td><td>TextCNN</td><td> $0 . 9 7 9 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 9 7 2 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 8 8 1 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $0 . 9 3 3 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 9 4 9 _ { \pm 0 . 0 1 6 }$ </td><td> $0 . 9 6 7 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 5 0 6 { \scriptstyle \pm 0 . 0 0 2 } }$ </td></tr><tr><td>LogRobust</td><td> $0 . 9 3 7 { \scriptstyle \pm 0 . 0 5 0 }$ </td><td> $0 . 9 3 0 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td> $\overline { { 0 . 9 4 4 _ { \pm 0 . 0 2 0 } } }$ </td><td> $0 . 9 7 9 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 9 5 8 { \scriptstyle \pm 0 . 0 3 6 }$ </td><td> $0 . 9 5 2 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td> $\mathbf { 0 . 5 0 2 _ { \pm 0 . 0 0 2 } }$ </td></tr><tr><td>LightLog</td><td> $0 . 9 5 9 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $\overline { { 0 . 9 6 0 _ { \pm 0 . 0 2 0 } } }$ </td><td>0.925±0.044</td><td> $0 . 9 7 9 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 8 6 1 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 9 1 6 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $\mathbf { 0 . 5 0 5 { \scriptstyle \pm 0 . 0 0 1 } }$ </td></tr><tr><td>NeuralLog</td><td> $0 . 9 9 5 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 9 9 4 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 9 8 2 _ { \pm 0 . 0 0 7 }$ </td><td> $0 . 9 9 5 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $\overline { { 0 . 9 9 5 _ { \pm 0 . 0 0 4 } } }$ </td><td> $0 . 9 9 5 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 6 8 3 { \scriptstyle \pm 0 . 0 1 8 } }$ </td></tr><tr><td>GPT2</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 9 9 3 _ { \pm 0 . 0 0 3 }$ </td><td> $\overline { { 0 . 9 9 4 _ { \pm 0 . 0 0 0 } } }$ </td><td> $\underline { { 0 . 9 8 8 _ { \pm 0 . 0 0 1 } } }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 5 9 0 { \scriptstyle \pm 0 . 0 0 4 } }$ </td></tr><tr><td rowspan="5">Thunderbird</td><td>TextCNN</td><td> $0 . 8 5 2 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 8 8 4 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 9 1 0 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 9 9 9 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 7 0 8 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td> $0 . 7 3 7 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $\mathbf { 0 . 5 6 1 { \scriptstyle \pm 0 . 0 4 0 } }$ </td></tr><tr><td>LogRobust</td><td> $0 . 9 2 6 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 9 2 9 { \scriptstyle \pm 0 . 0 4 8 }$ </td><td> $0 . 9 0 2 { \scriptstyle \pm 0 . 0 9 3 }$ </td><td> $0 . 9 9 5 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $\overline { { 0 . 8 9 8 { \scriptstyle \pm 0 . 0 3 6 } } }$ </td><td> $0 . 7 0 6 { \scriptstyle \pm 0 . 0 7 6 }$ </td><td> $\mathbf { 0 . 6 1 9 { \scriptstyle \pm 0 . 1 2 3 } }$ </td></tr><tr><td>LightLog</td><td> $0 . 6 5 6 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $0 . 9 7 3 { \scriptstyle \pm 0 . 0 3 6 }$ </td><td> $0 . 9 7 5 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td> $0 . 9 9 9 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 9 0 5 { \scriptstyle \pm 0 . 0 6 3 }$ </td><td> $\overline { { 0 . 7 2 3 _ { \pm 0 . 0 4 8 } } }$ </td><td> $\mathbf { 0 . 5 0 0 { \scriptstyle \pm 0 . 0 0 0 } }$ </td></tr><tr><td>NeuralLog</td><td> $\overline { { 0 . 8 5 2 _ { \pm 0 . 0 8 7 } } }$ </td><td> $0 . 9 4 7 { \scriptstyle \pm 0 . 0 6 1 }$ </td><td> $0 . 9 5 1 { \scriptstyle \pm 0 . 0 6 2 }$ </td><td> $0 . 9 9 1 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 9 0 2 { \scriptstyle \pm 0 . 0 8 7 }$ </td><td> $0 . 8 8 1 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td> $\mathbf { 0 . 5 1 6 { \scriptstyle \pm 0 . 0 2 1 } }$ </td></tr><tr><td>GPT2</td><td> $\overline { { 1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 } } }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $\underline { { 0 . 9 9 4 { \scriptstyle \pm 0 . 0 1 0 } } }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 5 3 4 { \scriptstyle \pm 0 . 0 0 8 } }$ </td></tr></table>

TABLE III

ABLATION STUDY OF LORD ON BGL, SPIRIT, AND LIBERTY WITH LOGROBUST AND NEURALLOG AS BASE DETECTORS. ABN. COC AND ABN. COE DENOTE THE AVERAGE CONFIDENCE ON CORRECTLY AND INCORRECTLY CLASSIFIED ABNORMAL SAMPLES, RESPECTIVELY.
<table><tr><td>Detector</td><td>Architecture</td><td colspan="2">BGL</td><td colspan="2">Spirit</td><td colspan="2">Liberty</td></tr><tr><td></td><td></td><td>Abn. CoC↑</td><td> $\mathbf { A b n . \ C o E \downarrow }$ </td><td> $\mathbf { A b n . { C o C \uparrow } }$ </td><td> $\mathbf { A b n . \ C o E \downarrow }$ </td><td> $\mathbf { A b n . \ C o C \uparrow }$ </td><td>Abn. CoE↓</td></tr><tr><td>LogRobust</td><td>SingleAE</td><td> $0 . 5 6 2 { \scriptstyle \pm 0 . 0 6 6 }$ </td><td> $0 . 5 8 2 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td> $0 . 5 5 6 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td> $0 . 5 1 6 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 7 4 0 { \scriptstyle \pm 0 . 0 4 9 }$ </td><td> $0 . 5 0 6 { \scriptstyle \pm 0 . 0 0 2 }$ </td></tr><tr><td></td><td>Route-0-Only</td><td> $\mathbf { 0 . 9 9 3 { \scriptstyle \pm 0 . 0 0 5 } }$ </td><td> $0 . 5 4 2 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 9 9 9 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 5 8 3 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $\mathbf { 0 . 9 9 9 \pm 0 . 0 0 1 }$ </td><td> $0 . 6 1 7 { \scriptstyle \pm 0 . 0 5 0 }$ </td></tr><tr><td></td><td>LoRD</td><td> $0 . 9 9 0 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 0 . 5 4 1 { \scriptstyle \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 0 . 9 9 9 } { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 5 0 8 _ { \pm 0 . 0 0 3 } }$ </td><td> $0 . 9 8 6 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $\mathbf { 0 . 5 0 2 _ { \pm 0 . 0 0 1 } }$ </td></tr><tr><td></td><td>SingleAE</td><td> $0 . 5 4 9 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td> $0 . 6 1 0 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td> $0 . 6 4 7 { \scriptstyle \pm 0 . 1 1 1 }$ </td><td> $0 . 5 4 3 { \scriptstyle \pm 0 . 0 5 9 }$ </td><td> $0 . 7 8 3 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 7 8 8 { \scriptstyle \pm 0 . 2 0 0 }$ </td></tr><tr><td>NeuralLog</td><td>Route-0-Only</td><td> $0 . 9 9 8 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 5 0 6 { \scriptstyle \pm 0 . 0 0 2 } }$ </td><td> $0 . 9 9 8 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 5 7 5 { \scriptstyle \pm 0 . 0 5 9 }$ </td><td> $\mathbf { 1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 } }$ </td><td> $0 . 7 1 6 { \scriptstyle \pm 0 . 0 4 4 }$ </td></tr><tr><td></td><td>LoRD</td><td> $\mathbf { 0 . 9 9 8 _ { \pm 0 . 0 0 1 } }$ </td><td> $\mathbf { 0 . 5 0 6 _ { \pm 0 . 0 0 2 } ^ { - } }$ </td><td> $\mathbf { 0 . 9 9 9 } \pm \mathbf { 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 5 1 1 { \scriptstyle \pm 0 . 0 0 8 } }$ </td><td> $\mathbf { 1 . 0 0 0 } _ { \pm 0 . 0 0 0 } ^ { - }$ </td><td> $\mathbf { 0 . 6 8 0 { \scriptstyle \pm 0 . 0 4 7 } }$ </td></tr></table>

TABLE IV

ABLATION STUDY OF LORD ON SPIRIT AND LIBERTY WITH LOGROBUST AS THE BASE DETECTOR. COC AND COE DENOTE THE AVERAGE CONFIDENCE ON CORRECTLY AND INCORRECTLY CLASSIFIED SAMPLES, RESPECTIVELY. LOWER D AND HIGHER C INDICATE BETTER OVERALL CALIBRATION QUALITY.
<table><tr><td></td><td colspan="6">Spirit</td><td colspan="6">Liberty</td></tr><tr><td>Method</td><td>Nor. CoC↑</td><td>Nor. CoE↓</td><td>Abn. CoC↑</td><td>Abn. CoE↓</td><td>D↓</td><td>C↑</td><td> $\mathrm { N o r . ~ C o C \uparrow }$ </td><td>Nor. CoE↓</td><td>Abn. CoC↑</td><td>Abn. CoE↓</td><td>D↓</td><td> $C \uparrow$ </td></tr><tr><td>w/o Reject</td><td>0.899</td><td>0.957</td><td>1.000</td><td>0.504</td><td>0.468</td><td>0.754</td><td>0.856</td><td>0.922</td><td>0.994</td><td>0.501</td><td>0.446</td><td>0.759</td></tr><tr><td>w/o Soft</td><td>0.908</td><td>0.905</td><td>1.000</td><td>0.503</td><td>0.415</td><td>0.776</td><td>0.879</td><td>0.843</td><td>0.988</td><td>0.501</td><td>0.364</td><td>0.796</td></tr><tr><td>LoRD</td><td>0.942</td><td>0.901</td><td>1.000</td><td>0.509</td><td>0.405</td><td>0.783</td><td>0.916</td><td>0.812</td><td>0.986</td><td>0.502</td><td>0.323</td><td>0.817</td></tr></table>

## VIII. LIMITATIONS

systems. An anonymous demo implementation of LoRD is available at: https://anonymous.4open.science/r/LoRD\_code-E10B.

LoRD has several limitations. First, its threshold and margin selection depend on route-specific validation errors, so the calibration boundary may be less stable when false negatives or false positives are rare. Second, LoRD is a task-oriented calibration method. It focuses on reducing the confidence of high-risk errors rather than improving all aggregate calibration metrics, which may lead to trade-offs in metrics such as ECE, NLL, or Brier score. Third, LoRD requires access to detector hidden representations and is therefore more suitable for whitebox or gray-box detectors. Future work will extend LoRD to stronger distribution shifts and multi-class diagnosis.

TABLE V  
ABLATION STUDY OF LORD ON ABNORMAL LOG CALIBRATION USING LOGROBUST AND NEURALLOG AS BASE DETECTORS. “W/O REJECT” AND “W/O SOFT” REMOVE THE REJECT REGION AND SOFTPULL CALIBRATION, RESPECTIVELY. LOWER D AND HIGHER C INDICATE BETTER OVERALL CALIBRATION.
<table><tr><td></td><td colspan="4">Spirit</td><td colspan="4">Liberty</td></tr><tr><td>Method</td><td>CoC↑</td><td>CoE↓</td><td>D↓</td><td>C↑</td><td>CoC↑</td><td>CoE↓</td><td>D↓</td><td>C↑</td></tr><tr><td colspan="9">LogRobust</td></tr><tr><td>w/o Reject</td><td>1.000</td><td>0.504</td><td>0.468</td><td>0.754</td><td>0.994</td><td>0.501</td><td>0.446</td><td>0.759</td></tr><tr><td>w/o Soft LoRD</td><td>1.000</td><td>0.503</td><td>0.415</td><td>0.776</td><td>0.988</td><td>0.501</td><td>0.364</td><td>0.796</td></tr><tr><td></td><td>1.000</td><td>0.509</td><td>0.405</td><td>0.783</td><td>0.986</td><td>0.502</td><td>0.323</td><td>0.817</td></tr><tr><td colspan="9">NeuralLog</td></tr><tr><td>w/o Reject</td><td>1.000</td><td>0.507</td><td>0.508</td><td>0.736</td><td>1.000</td><td>0.572</td><td>0.450</td><td>0.757</td></tr><tr><td>w/o Soft</td><td>1.000</td><td>0.507</td><td>0.506</td><td>0.741</td><td>1.000</td><td>0.602</td><td>0.395</td><td>0.790</td></tr><tr><td>LoRD</td><td>1.000</td><td>0.513</td><td>0.495</td><td>0.747</td><td>1.000</td><td>0.680</td><td>0.354</td><td>0.814</td></tr></table>

## REFERENCES

[1] N. Han, S. Lu, D. Wang, M. Wang, X. Tan, and X. Wei, “Skdlog: self-knowledge distillation-based cnn for abnormal log detection,” in 2022 IEEE Smartworld, Ubiquitous Intelligence & Computing, Scalable Computing & Communications, Digital Twin, Privacy Computing, Metaverse, Autonomous & Trusted Vehicles (Smart-World/UIC/ScalCom/DigitalTwin/PriComp/Meta). IEEE, 2022, pp. 796– 805.

[2] S. Lu, N. Han, M. Wang, X. Wei, Z. Lin, and D. Wang, “Ssdlog: a semi-supervised dual branch model for log anomaly detection,” World Wide Web, vol. 26, no. 5, pp. 3137–3153, 2023.

[3] S. Lu, M. Wang, D. Wang, X. Wei, S. Xiao, Z. Wang, N. Han, and L. Wang, “Black-box attacks against log anomaly detection with adversarial examples,” Information Sciences, vol. 619, pp. 249–262, 2023.

[4] S. Nedelkoski, J. Bogatinovski, A. Acker, J. Cardoso, and O. Kao, “Self attentive classification-based anomaly detection in unstructured logs,” in 2020 IEEE international conference on data mining (ICDM). IEEE, 2020, pp. 1196–1201.

[5] V.-H. Le and H. Zhang, “Log-based anomaly detection with deep learning: How far are we?” in Proceedings of the 44th international conference on software engineering, 2022, pp. 1356–1367.

[6] M. Landauer, F. Skopik, and M. Wurzenberger, “A critical review of common log data sets used for evaluation of sequence-based anomaly detection techniques,” Proceedings of the ACM on Software Engineering, vol. 1, no. FSE, pp. 1354–1375, 2024.

[7] J. Zhu, S. He, P. He, J. Liu, and M. R. Lyu, “Loghub: A large collection of system log datasets for ai-driven log analytics,” in 2023 IEEE 34th International Symposium on Software Reliability Engineering (ISSRE). IEEE, 2023, pp. 355–366.

[8] L. Wu, B. Lei, D. Xu, and D. Zhou, “Towards reliable rare category analysis on graphs via individual calibration,” in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023, pp. 2629–2638.

[9] M. Du, F. Li, G. Zheng, and V. Srikumar, “Deeplog: Anomaly detection and diagnosis from system logs through deep learning,” in Proceedings of the 2017 ACM SIGSAC conference on computer and communications security, 2017, pp. 1285–1298.

[10] W. Meng, Y. Liu, Y. Zhu, S. Zhang, D. Pei, Y. Liu, Y. Chen, R. Zhang, S. Tao, P. Sun et al., “Loganomaly: Unsupervised detection of sequential and quantitative anomalies in unstructured logs.” in Ijcai, vol. 19, no. 7, 2019, pp. 4739–4745.

[11] L. Yang, J. Chen, Z. Wang, W. Wang, J. Jiang, X. Dong, and W. Zhang, “Plelog: Semi-supervised log-based anomaly detection via probabilistic label estimation,” in 2021 IEEE/ACM 43rd International Conference on Software Engineering: Companion Proceedings (ICSE-Companion). IEEE, 2021, pp. 230–231.

[12] S. Lu, X. Wei, Y. Li, and L. Wang, “Detecting anomaly in big data system logs using convolutional neural network,” in 2018 IEEE 16th Intl Conf on Dependable, Autonomic and Secure Computing, 16th Intl Conf on Pervasive Intelligence and Computing, 4th Intl Conf on Big

Data Intelligence and Computing and Cyber Science and Technology Congress (DASC/PiCom/DataCom/CyberSciTech), 2018, pp. 151–158.

[13] X. Zhang, Y. Xu, Q. Lin, B. Qiao, H. Zhang, Y. Dang, C. Xie, X. Yang, Q. Cheng, Z. Li et al., “Robust log-based anomaly detection on unstable log data,” in Proceedings of the 2019 27th ACM joint meeting on European software engineering conference and symposium on the foundations of software engineering, 2019, pp. 807–817.

[14] Z. Wang, J. Tian, H. Fang, L. Chen, and J. Qin, “Lightlog: A lightweight temporal convolutional network for log anomaly detection on the edge,” Computer Networks, vol. 203, p. 108616, 2022.

[15] V.-H. Le and H. Zhang, “Log-based anomaly detection without log parsing,” in 2021 36th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2021, pp. 492–504.

[16] W. Guan, J. Cao, S. Qian, J. Gao, and C. Ouyang, “Logllm: Logbased anomaly detection using large language models,” arXiv preprint arXiv:2411.08561, 2024.

[17] Y. F. Lim, J. Zhu, and G. Pang, “Adapting large language models for parameter-efficient log anomaly detection,” in Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, 2025, pp. 325–337.

[18] J. Gawlikowski, C. R. N. Tassi, M. Ali, J. Lee, M. Humt, J. Feng, A. Kruspe, R. Triebel, P. Jung, R. Roscher et al., “A survey of uncertainty in deep neural networks,” Artificial intelligence review, vol. 56, no. Suppl 1, pp. 1513–1589, 2023.

[19] R. Müller, S. Kornblith, and G. E. Hinton, “When does label smoothing help?” Advances in neural information processing systems, vol. 32, 2019.

[20] G. Pereyra, G. Tucker, J. Chorowski, Ł. Kaiser, and G. Hinton, “Regularizing neural networks by penalizing confident output distributions,” arXiv preprint arXiv:1701.06548, 2017.

[21] S. Thulasidasan, G. Chennupati, J. A. Bilmes, T. Bhattacharya, and S. Michalak, “On mixup training: Improved calibration and predictive uncertainty for deep neural networks,” Advances in neural information processing systems, vol. 32, 2019.

[22] J. Mukhoti, V. Kulharia, A. Sanyal, S. Golodetz, P. Torr, and P. Dokania, “Calibrating deep neural networks using focal loss,” Advances in neural information processing systems, vol. 33, pp. 15 288–15 299, 2020.

[23] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, “Class-balanced loss based on effective number of samples,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 9268– 9277.

[24] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation: Representing model uncertainty in deep learning,” in international conference on machine learning. PMLR, 2016, pp. 1050–1059.

[25] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” Advances in neural information processing systems, vol. 30, 2017.

[26] A. G. Wilson and P. Izmailov, “Bayesian deep learning and a probabilistic perspective of generalization,” Advances in neural information processing systems, vol. 33, pp. 4697–4708, 2020.

[27] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in International conference on machine learning. PMLR, 2017, pp. 1321–1330.

[28] M. Kull, T. Silva Filho, and P. Flach, “Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers,” in Artificial intelligence and statistics. PMLR, 2017, pp. 623–631.

[29] D. Wang, B. Gong, and L. Wang, “On calibrating semantic segmentation models: Analyses and an algorithm,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 23 652–23 662.

[30] M. P. Naeini, G. Cooper, and M. Hauskrecht, “Obtaining well calibrated probabilities using bayesian binning,” in Proceedings of the AAAI conference on artificial intelligence, vol. 29, no. 1, 2015.

[31] A. Oliner and J. Stearley, “What supercomputers say: A study of five system logs,” in 37th annual IEEE/IFIP international conference on dependable systems and networks (DSN’07). IEEE, 2007, pp. 575–584.
# Confidence Calibration of Deep Learning Systems

Yacob (Coby) Penso

Faculty of Engineering

Ph.D. Thesis

Submitted to the Senate of Bar-Ilan University

Ramat Gan, Israel

May 2025

The Faculty of Engineering,

Bar-Ilan University

Prof. Jacob Goldberger,

This work was carried out under the supervision of

## Acknowledgment

I would like to express my deepest gratitude to my supervisor, Prof. Jacob Goldberger, for his invaluable guidance, support, and encouragement throughout the course of my doctoral studies. His insight, patience, and high standards of research have shaped not only this thesis but also my approach to science as a whole.

I am also grateful to my collaborators and co-authors, Ethan Fetaya, Bar Mahpud, and Lior Frenkel, whose ideas, feedback, and dedication have greatly enriched this work. It has been a privilege to learn and work alongside them.

Finally, I extend my heartfelt thanks to my family and friends for their unwavering support, understanding, and encouragement during this journey. Without their love and patience, this thesis would not have been possible.

## Contents

Abstract   
1 Introduction   
2 Background 6   
2.1 Confidence Calibration 6   
2.2 Conformal Prediction 7   
2.3 Unsupervised Target Domain Calibration 9   
2.4 Noisy Labels 10   
2.5 Local Diferential Privacy 12   
3 Confidence Calibration under Noisy Labels 15   
3.1 Problem Statement . 15   
3.2 Confidence Calibration with Noisy Labels 17   
3.3 Network Training with Noisy Labels 21   
3.4 Experiments . 22   
4 Conformal Prediction under Noisy Labels 29   
4.1 Problem Statement 29   
4.2 Score That Is Robust to Label Noise 30   
4.3 Procedure of Threshold Estimation That Is Robust to Label Noise 32   
4.3.1 Prediction Size Comparison 35   
4.3.2 Coverage Guarantees . 36   
4.3.3 A More General Noise Model 38   
4.4 Related Work 39   
4.5 Experiments . 41   
5 Local Diferential Private Conformal Prediction 47   
5.1 Problem Statement . 47   
5.2 Local-DP Conformal Prediction on Labels 49   
5.3 Local-DP Conformal Prediction on Scores 51   
5.4 Theoretical Guarantees 53   
5.5 Practical Considerations 54   
5.5.1 LDP-CP-L vs. LDP-CP-S 54   
5.5.2 The Shufle Model of Diferential Privacy 55   
5.6 Experiments . 56   
6 Unsupervised Target Domain Confidence Calibration 60   
6.1 Problem Statement . 60   
6.2 Calibration on the Target Domain in Unsupervised Domain Adaptation 62   
6.3 Experiments . 64   
6.4 Analysis 66   
7 Discussion 71   
7.1 Summary of Contributions . 71   
7.2 Key Insights and Implications 72   
7.3 Future Research Directions 73   
7.4 Conclusion 74   
Bibliography 75   
Hebrew Abstract ℵ

## List of Figures

3.1 Schema of the proposed model that includes the full pipeline of network training and   
calibration based on data with noisy labels. 17   
3.2 Comparative calibration results on several datasets that were trained with ResNet-50.   
The top row shows the ECE results on the (clean) test set. The bottom row shows   
the optimal temperature found on the noisy validation set. 23   
3.3 Standard deviation of ECE scores on the test set after the NTS calibration as a   
function of the size of the noisy validations set. 24   
3.4 Calibration results for several noise transition matrices on ChestX-ray14-bal and   
ResNet-18. In each example we show the noise transition matrix (Left) and the   
adaECE measure on the test set for the compared calibration methods (right). . 27   
3.5 Calibration performance, measured by adaECE (left), and the corresponding temper  
ature (right) as a function of the noise ratio used by the NTS algorithm. 27   
4.1 Correction terms $\Delta$ of NACP, ACNL and CRCP as a function of the validation set   
size n given ϵ = 0.2. We show results for 3 numbers of classes, 10, 100 and 1000. . 43   
4.2 Noisy labels conformal prediction on ImageNet with diferent calibration set sizes. (a)   
Mean size (b) Coverage (%), and (c) Correction terms $\Delta$ as a function of calibration   
set size. 44   
5.1 Local Diferential Private Conformal Prediction (LDP-CP-L) Pipeline (Best viewed   
in color) 49   
5.2 Local Diferential Private Conformal Prediction (LDP-CP-S) Pipeline (Best viewed   
in color). 52   
5.3 Comparison of $\Delta _ { l }$ and $\Delta _ { s }$ as a function of the number of classes k and dataset size   
n, for ϵ = 2, 4, 8. 57   
5.4 CP correction terms $\Delta _ { L } , \Delta _ { S }$ as a function of ϵ privacy parameter across diferent   
dataset configurations of n and k without the shufle model. 57   
5.5 CP correction terms $\Delta _ { L } , \Delta _ { S }$ as a function of $\epsilon ^ { \mathrm { e f f } }$ privacy parameter across diferent   
dataset configurations of n and k with the shufle model. 57   
5.6 Size of prediction set (left) and coverage (right) as a function of the privacy ϵ (bot  
tom x-axis) and efective privacy $\epsilon ^ { \mathrm { e f f } }$ (top x-axis). We show the $( \mathrm { m e a n } \pm \mathrm { s t d } )$ on   
TissueMNIST and APS score. 59   
6.1 Average accuracy on Ofice-home tasks for the three UDA techniques (DANN, DANN+E,   
CDAN+E). 67   
6.2 adaECE results as a function of the correction ratio R on Ofice-Home, $A  C$ task. 68   
6.3 Accuracy of k-th percentile source images based on their probability of being classified   
as target [97], compared to target accuracy (Ofice-home, A → C). . 69   
6.4 Accuracy per bin for source and target images. The results are shown on the Ofice  
home $C  P$ task. 69

## List of Tables

3.1 Adaptive ECE for top-1 predictions (in %) using 15 bins. on various medical imaging   
classification datasets and models with diferent calibration methods with a noise level   
ϵ = 0.2. The lowest is highlighted in bold. 23   
3.2 ECE for top-1 predictions (in %) using 15 bins. on various medical imaging classifica  
tion datasets and models with diferent calibration methods with a noise level ϵ = 0.2.   
The lowest is highlighted in bold. 23   
3.3 Adaptive ECE for top-1 predictions (in %) using 15 bins (with the lowest in bold) on   
various medical imaging classification datasets and models with diferent calibration   
methods with varying noise levels. Model training and noise matrix estimation used   
[52]. 24   
3.4 ECE for top-1 predictions (in %) using 15 bins (with the lowest in bold) on various   
medical imaging classification datasets and models with diferent calibration methods   
with varying noise levels. Model training and noise matrix estimation used [52]. . 25   
4.1 Conformal Prediction methods for validation sets with noisy labels. Given an image   
x, y is the true label, yˆ is its noisy version and ϵ is the noise level. S is a conformal   
score and S<sup>ˆ</sup> is its noise robust variant. 31   
4.2 CP calibration results for 1−α = 0.9 and noise level ϵ = 0.2. We report the mean and   
the std over 1000 diferent splits. We show the best result with theoretical guarantees   
in bold. 42   
4.3 Finite sample correction terms ∆ of NACP, ACNL [86] and CRCP [13], for several   
datasets and two noise levels, n is the size of the validation set 42   
4.4 Rand-APS calibration results for 1−α = 0.9 on CIFAR-100 dataset and two noise   
models. We report the mean and the std over 1000 diferent splits. 44   
4.5 CP calibration results on ImageNet and various model architectures for 1−α = 0.9   
and ϵ = 0.2. We report the mean and the std over 1000 diferent splits. Bold for best   
result with theoretical guarantees. 45   
4.6 CP calibration results on CIFAR-10N for $1 - \alpha = 0 . 9$ and $\epsilon = 0 . 2 .$ We report the mean   
and the std over 1000 diferent splits. Bold for best result with theoretical guarantees. 46   
5.1 Calibration results for HPS and APS conformal scores across various datasets, using   
$\epsilon = 4 ,$ $\epsilon ^ { \mathrm { e f f } } = \frac { \epsilon } { \sqrt { n } }$ , and $\alpha = 0 . 1$ on 100 diferent seeds. 58   
6.1 Comparison of calibration methods for unsupervised domain adaptation (UDA). . 61   
6.2 AdaECE results on Ofice-home (with the lowest in bold) on various UDA classifica   
tion tasks and models with diferent calibration methods. 65   
6.3 AdaECE results on Ofice-31 (with the lowest in bold) on various UDA classification   
tasks and models with diferent calibration methods. 66   
6.4 adaECE results on VisDA Task S → R, for various calibration methods. 66   
6.5 adaECE results on DomainNet for various UDA classification tasks and models with   
diferent calibration methods. 67   
6.6 Calibration metrics results of various UDA calibration methods on the Ofice-home   
tasks. 67   
6.7 Computed temperature on various UDA Ofice-home tasks, and calibration methods   
using CDAN+E. 68   
6.8 AdaECE results for variations of UTDC based on diferent methods of domain accu  
racy estimation 70   
6.9 Comparison of several target domain accuracy estimation methods measured by $| A C C ( T r u e ) -$   
$A C C ( E s t ) |$ 70

## List of Algorithms

1 Noisy Temperature Scaling (NTS) - Uniform noise 18   
2 Noisy Temperature Scaling (NTS) - General Noise Matrix 20   
3 Noise-Robust Score Conformal Prediction (NRSCP) 32   
4 Noise-Aware Conformal Prediction (NACP) for uniform noise 36   
5 Noise-Aware Conformal Prediction (NACP) for a noise matrix model . . 37   
6 LDP-CP-L - User-side Procedure 49   
7 LDP-CP-L - Aggregator-side Procedure 51   
8 LDP-CP-S - Both sides 53   
9 Unsupervised Target Domain Calibration (UTDC) 63

# List of Abbreviations and Notations

In this thesis ϵ used as the noise rate, except for Chapter 5 when ϵ is the privacy in ϵ-LDP and β used as the noise rate.

CE Conformal Prediction

CE Cross-Entropy

CNN Convolutional Neural Network

DL Deep Learning

DNN Deep Neural Network

ECE Expected Calibration Error

adaECE Adaptive Expected Calibration Error

LLMs Large-Language Models

MCE Maximum Calibration Error

MSE Mean Square Error

NN Neural Network

DA Domain Adaptation

UDA Unsupervised Domain Adaptation

OOD Out-of-Distribution

UTDC Unsupervised Target Domain Calibration

NRCP Noise Robust Conformal Prediction

NACP Noise Aware Conformal Prediction

NTS Noisy Temperature Scaling

TS Temperature Scaling

DP Diferential Privacy

LDP Local Diferential Privacy

LDP-CP Local Diferential Private Conformal Prediction

LDP-CP-L Local Diferential Private Conformal Prediction On Labels

LDP-CP-S Local Diferential Private Conformal Prediction On Scores

## Abstract

In high-stakes applications such as medical imaging, the reliability of a model’s confidence in its predictions is as crucial as the predictions themselves. Confidence calibration ensures that a model’s predicted probabilities accurately reflect its likelihood of correctness, making it a critical component for safe and efective deployment of deep learning models in medical diagnostics. However, existing calibration techniques assume access to clean validation data, which is often unrealistic in medical imaging settings due to the prevalence of label noise and domain shifts. This thesis explores novel methods for improving confidence calibration under these challenging conditions. First, we address confidence calibration in the presence of label noise. When calibration methods are applied to data with unreliable labels, they may yield misleading confidence estimates that undermine the trustworthiness of model predictions. We propose a calibration framework that accounts for label noise by leveraging an estimated noise model. Specifically, we demonstrate how to reconstruct noise-free confidence estimates by modeling the relationship between noisy and clean label distributions. We extend this idea to Conformal Prediction (CP), a framework that provides set-valued predictions with a guaranteed level of coverage. We introduce a noise-aware conformal prediction approach that estimates the true conformity scores despite label noise, allowing us to maintain eficient and reliable uncertainty quantification. Next, we investigate confidence calibration in unsupervised domain adaptation (UDA), where a model trained on a labeled source domain is adapted to an unlabeled target domain. Traditional calibration methods require labeled validation data from the target domain, which is unavailable in this setting. To overcome this limitation, we develop an approach that estimates the target domain accuracy based on the model’s performance in the source domain and known domain discrepancies. This allows us to directly calibrate model confidence without access to target domain labels. Furthermore, we extend our study to privacy-preserving settings, where individual user labels and model outputs must be protected. We propose a locally diferentially private conformal prediction framework that ensures valid uncertainty quantification while maintaining rigorous privacy guarantees. Our approach balances the trade-ofs between privacy, computational feasibility, and prediction reliability, making it applicable to sensitive medical data applications. Through extensive experiments on natural and medical imaging datasets, we demonstrate that our proposed methods significantly improve calibration robustness under both label noise and domain shift conditions. We provide theoretical guarantees and empirical validations that bridge the gap between theoretical calibration guarantees and practical deployment in safety-critical environments.

Our findings contribute to the development of reliable, privacy-preserving, and noise-resilient calibration frameworks, enhancing the trustworthiness of neural network predictions in real-world medical and high-stakes applications.

## Chapter 1

## Introduction

In high-stakes domains like medical imaging, the accuracy of a model’s confidence in its predictions can be just as important as the predictions themselves. Confidence calibration is the process of ensuring that a model’s predicted probabilities align with its actual accuracy, ofering reliable confidence estimates for each prediction. Confidence calibration is defined as the ability of a classifier network to provide an accurate probability of correctness for any of its predictions. Neural networks have been shown to be more overconfident in their predictions than their predecessors even though their generalization accuracy is higher, partly due to the fact that they can overfit on the negative log-likelihood loss without overfitting on the classification error [33, 49, 36]. In medical imaging applications, images for which the model makes low-confidence predictions are sent to a physician for review. Skipping the human review based on confident but incorrect predictions can have disastrous consequences [62]. The gap between the model’s predicted probabilities and its accuracy is one of the key obstacles to the applicability of neural network models to fully automatic medical diagnosis.

Calibration methods can generally be divided into two main approaches. The first focuses on calibrating the confidence in a predicted class, while the second addresses the problem by generating a prediction set - a collection of possible classes - with a specified probability that the true class is included in this set. These approaches will be referred to as Confidence Calibration and Conformal Prediction, respectively, throughout this thesis.

Within the framework of confidence calibration, various methods have recently been developed to address the issue of excessive overconfidence in predictions. Network calibration can either be performed alongside training (e.g., [63, 64, 115, 104]) or applied as a post-hoc procedure (e.g., Platt scaling [80], isotonic regression [110], and temperature scaling [33]). Post-hoc methods improve calibration by applying it as a post-processing step, using hold-out validation data to create a calibration map that adjusts the model’s predictions. Among these, temperature scaling stands out as a practical and widely adopted approach due to its simplicity and ease of implementation. Despite the critical role of network calibration in automating medical reporting, only a limited number of studies have specifically addressed the calibration of medical imaging systems (e.g., [23, 25, 83, 111]).

In conformal prediction, the goal is to return a (preferably small) set of potential class candidates that includes the true class with a predefined level of confidence. This approach is particularly wellsuited for medical imaging, where safety is paramount and the final decision is made by a human. By reducing the number of possible diagnoses a practitioner needs to consider, conformal prediction helps streamline decision-making while maintaining a controlled risk of error. The general method of producing a prediction set without making assumptions about the data distribution (aside from i.i.d. samples) is known as Conformal Prediction (CP) [1, 93]. CP guarantees that the probability of the correct class being included in the set meets or exceeds a specified confidence level, while aiming to return the smallest set possible that still maintains this guarantee. With the increasing use of neural networks in safety-critical applications like medical imaging, CP has emerged as a crucial calibration tool [57, 58, 67]. It is important to note that CP is a general framework rather than a single algorithm, with the most common implementations constructing the prediction set based on a conformity score. Diferent algorithms mainly difer in how this conformity score is defined.

Confidence calibration and conformal prediction are extensively studied in settings where clean data and labels are provided. However, our research focuses on more complex and realistic scenarios, specifically when labels are noisy and in the context of unsupervised domain adaptation, where the goal is to calibrate the target model without access to labeled data.

Deep neural networks have been highly successful in various natural image and medical image computing tasks. However, these achievements depend on having accurate annotated training data. Neural networks require massive amounts of carefully labeled data to succeed, but acquiring such data is expensive and time-consuming. Non-expert sources, like Amazon’s Mechanical Turk, have been used to reduce labeling costs, but their labels can be unreliable. Experienced domain experts may also struggle with complex labeling tasks. Medical imaging datasets often have problems with noisy labels due to ambiguous images that can confuse clinical experts. Physicians may disagree on the diagnosis of the same medical image, resulting in variability in the ground truth label. Furthermore, using Natural Language Processing (NLP) tools to extract labels from radiological reports can also introduce label noise [42]. Therefore, addressing annotation noise is a crucial topic in medical image analysis.

Training neural networks with noisy labels is problematic because the models can easily overfit to the corrupted labels, resulting lack of generalizability when evaluated on a separate test dataset. While popular regularization techniques have been used to address overfitting, they do not entirely solve the problem. Even when these techniques are applied, there is a significant gap in test accuracy between models trained on clean vs. noisy data, and the accuracy decreases with label noise. Noisy labels are dificult to avoid, and studies indicate that Deep Neural Networks (DNNs) can memorize entire datasets. Consequently, errors in datasets may result in erroneous predictions, which can impact medical diagnoses. Therefore, efectively managing noisy labels is crucial for automated medical image classification. A review of network training methods for noisy labels can be found in [88] and an excellent up-to-date discussion of training medical image classification networks from data with noisy labels can be found in [105].

Numerous studies have examined the problem of training networks that are resilient to label noise, which can also disrupt the network calibration process. Our findings suggest that network calibration methods are more susceptible to label noise compared to network training. Nevertheless, we have not come across any previous research that tackles the challenge of network calibration using a validation set containing noisy labels.

In addition to the challenges posed by noisy labels, another critical issue in real-world applications of deep learning is the performance degradation that occurs when a network trained on data from one domain is applied to data from a diferent domain, where the feature distribution difers - a phenomenon known as domain shift (see e.g., [61]). In the context of Unsupervised Domain Adaptation (UDA), the goal is to adapt a model to a target domain where labeled data is unavailable, though data from the target domain itself is accessible. Our findings indicate that existing calibration methods for unsupervised domain adaptation often fail in practice, particularly when the domain gap is large, further complicating the task of achieving reliable network calibration.

Lastly, in many critical settings, the calibration procedure is performed by a centralized component, referred to as an aggregator, which may be untrusted. In such scenarios, exposing sensitive data directly to this untrusted aggregator poses significant privacy risks. A promising approach to mitigating these risks is Local Diferentially Private (LDP) Conformal Prediction, where individual data contributors apply noise to their calibration data before sharing it with the aggregator. This ensures that the aggregator can perform calibration without directly accessing private or sensitive information from individual sources. However, the introduction of noise through LDP presents new challenges in maintaining both the validity and eficiency of conformal prediction methods. We explore strategies to adapt conformal calibration techniques to function efectively under diferential privacy constraints, ensuring that predictions remain reliable while preserving user privacy.

In our research, we tackle these challenges in three key areas: first, by addressing the calibration of neural networks using validation sets with inaccurate labels, focusing on both confidence calibration and conformal prediction; second, by exploring confidence calibration in systems facing unsupervised domain shift scenarios; and third, by investigating privacy-preserving calibration methods using local diferentially private conformal prediction, ensuring robust calibration without compromising data privacy. Through these eforts, we aim to enhance the reliability, robustness, and privacy of neural networks in real-world applications where label noise, domain shifts, and privacy concerns are significant challenges.

The remainder of this thesis is structured as follows:

Chapter 2, Background, provides an overview of the foundational concepts and related work, including confidence calibration, conformal prediction, unsupervised target domain calibration, noisy labels, and local diferential privacy.

Chapter 3, Confidence Calibration under Noisy Labels, introduces our proposed methods for improving confidence calibration when dealing with noisy labels. We explore both the challenges of confidence calibration with noisy labels and network training strategies, followed by a comprehensive set of experiments to validate our approach.

Chapter 4, Conformal Prediction under Noisy Labels, extends our focus to conformal prediction, presenting a robust scoring method and a threshold estimation procedure tailored to noisy label scenarios. We provide theoretical insights and experimental evaluations, including prediction size comparisons, coverage guarantees, and adaptations to more general noise models.

Chapter 5, Local Diferential Private Conformal Prediction, delves into privacy-preserving techniques, introducing methods to apply local diferential privacy (LDP) to both labels and prediction scores. We discuss theoretical guarantees, practical considerations, and experimental results, comparing diferent approaches under privacy constraints.

Chapter 6, Unsupervised Target Domain Confidence Calibration, addresses the challenge of calibrating confidence when transitioning models to a new, unlabeled target domain. We describe our proposed calibration methods, present experimental findings, and analyze the performance under unsupervised domain adaptation settings.

Chapter 7, Discussion, summarizes our contributions, highlights the key insights and implications of our work, and outlines potential directions for future research. The chapter concludes by reafirming the significance of our research in advancing confidence calibration and conformal prediction in challenging real-world scenarios.

As part of the thesis, the following papers have been published in various conferences and journals:

• Confidence calibration of a medical imaging classification system that is robust to label noise - Coby Penso, Lior Frenkel, Jacob Goldberger, IEEE Transactions on Medical Imaging (TMI), vol. 43(6), pp. 2050-2060, 2024, [74].

• A joint training and confidence calibration procedure that is robust to label noise - Coby Penso, Jacob Goldberger, IEEE International Symposium on Biomedical Imaging (ISBI), 2024, [77].

• A conformal prediction score that is robust to label noise - Coby Penso, Jacob Goldberger, MICCAI, Machine Learning for Medical Imaging Workshop, 2024, [76].

• Network calibration under domain shift based on estimating the target domain accuracy - Coby • Conformal Prediction of Classifiers with Many Classes based on Noisy Labels - Coby Penso, Jacob Goldberger, Eitan Fetaya, accepted to the Symposium on Conformal and Probabilistic Prediction with Applications (COPA), 2025, [79].

Penso, Jacob Goldberger, ECCV, Uncertainty in Computer Vision Workshop, 2024, [75].

• Privacy-Preserving Conformal Prediction Under Local Diferential Privacy - Coby Penso, Bar Mahpud, Jacob Goldberger, Or Shefet, accepted to Symposium on Conformal and Probabilistic Prediction with Applications (COPA), 2025, [73].

# Chapter 2

## Background

## 2.1 Confidence Calibration

In this section we review the definition of confidence calibration. Consider a network that classifies an input image x into k pre-defined categories. The last layer of the network architecture is comprised of k real numbers $\boldsymbol { z } = ( z _ { 1 } , . . . , z _ { k } )$ known as logits. Each of these numbers is the score for one of the k possible classes. The logits are then converted into a soft decision distribution using a softmax layer: $\begin{array} { r } { p ( y = i | x ) = \frac { \exp ( z _ { i } ) } { \sum _ { j } \exp ( z _ { j } ) } } \end{array}$ where x is the input image and y is the image class. Despite having the mathematical form of a distribution, the output of the softmax layer does not necessarily represent the true posterior distribution of the classes, and the network often tends to have overconfidence in its predictions.

The predicted class is calculated from the output distribution by $\hat { y } ~ = ~ \arg \operatorname* { m a x } _ { i } p ( y ~ = ~ i | x ) ~ =$ arg $\operatorname* { m a x } _ { i } { z _ { i } }$ . The network confidence for this sample is defined by $\begin{array} { r } { \hat { p } = p ( y = \hat { y } | x ) = \operatorname* { m a x } _ { i } p ( y = i | x ) } \end{array}$ The network accuracy is defined by the probability that the most probable class yˆ is indeed correct. The network is said to be calibrated if the estimated confidence coincides with the actual accuracy.

The Expected Calibration Error (ECE) [65] stands as the conventional metric employed for quantifying the calibration of a model. It is characterized by the expected absolute disparity between the model’s accuracy and its level of confidence. In practice, we only have a validation set with a finite number of samples $( x _ { 1 } , y _ { 1 } ) , . . . , ( x _ { n } , y _ { n } )$ thus an approximation is used. Denote the predictions and confidence values of the validation set by $( \hat { y } _ { 1 } , \hat { p } _ { 1 } ) , . . . , ( \hat { y } _ { n } , \hat { p } _ { n } )$ . To compute the ECE measure we first divide the unit interval [0, 1] into m equal size bins $b _ { 1 } , . . . , b _ { m }$ and let $B _ { i } = \{ t | \hat { p } _ { t } \in b _ { i } \}$ be the set of samples whose confidence values belong to bin $b _ { i }$ . The network average accuracy at this bin is computed as:

$$
A _ { i } = \frac { 1 } { | B _ { i } | } \sum _ { t \in B _ { i } } \mathbb { 1 } _ { \{ \hat { y } _ { t } = y _ { t } \} } ,\tag{2.1}
$$

where 1 is the indicator function, and $y _ { t }$ and $\hat { y } _ { t }$ are the correct and predicted labels for $x _ { t }$ respectively. $A _ { i }$ is the relative number of correct predictions of instances that were assigned to $B _ { i }$ based on their

confidence value. The average confidence at bin $b _ { i }$ is computed as:

$$
C _ { i } = \frac { 1 } { | B _ { i } | } \sum _ { t \in B _ { i } } \hat { p } _ { t } .\tag{2.2}
$$

If the network is under-confident at bin $b _ { i }$ then $A _ { i } > C _ { i }$ and vice-versa. The ECE is defined as follows:

$$
\mathrm { E C E } = \sum _ { i = 1 } ^ { m } { \frac { | B _ { i } | } { n } } \left| A _ { i } - C _ { i } \right| .\tag{2.3}
$$

The ECE is based on a uniform bin width. If the model is well-trained, most of the samples should lie within the highest confidence bins. Hence, the low confidence bins should be almost empty and therefore have no influence on the computed value of the ECE. For this reason, we can consider another metric, Adaptive ECE (adaECE) where bin sizes are taken into account so as to evenly distribute samples between bins [66]:

$$
{ \mathrm { a d a E C E } } = { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } \left| A _ { i } - C _ { i } \right|\tag{2.4}
$$

such that each bin contains $1 / m$ of the data points with similar confidence values. AdaECE is considered a better and more resilient method than ECE for assessing network calibration. In this study we used the adaECE for both calibration and evaluation.

Temperature Scaling (TS) is a standard highly efective technique for calibrating the output distribution of a classification network [33]. It uses a single parameter $T > 0$ to rescale logit scores before applying the softmax function to compute the class distribution. Temperature scaling is expressed as follows:

$$
p _ { T } ( y = i | x ) = \frac { \exp ( z _ { i } / T ) } { \sum _ { j = 1 } ^ { k } \exp ( z _ { j } / T ) } , \quad i = 1 , \ldots , k\tag{2.5}
$$

s.t. $z _ { 1 } , . . . , z _ { k }$ are the logit values derived from the application of the network to the input vector $x .$ The optimal temperature T for a trained model can be found by maximizing the log-likelihood $\begin{array} { r } { \sum _ { t } \log p _ { T } ( y _ { t } | x _ { t } ) } \end{array}$ for the held-out validation dataset. Studies show that finding the optimal T by directly minimizing the ECE or adaECE measures yields better calibration results [63].

## 2.2 Conformal Prediction

Consider a setup involving a classification network that categorizes an input x into k predetermined classes. Given a coverage level of $1 - \alpha$ , we aim to identify the smallest possible prediction set (a subset of these classes) ensuring the correct class is within the set with a probability of at least $1 - \alpha$ A straightforward strategy to achieve this objective involves sequentially incorporating classes from the highest to the lowest probabilities until their cumulative sum exceeds the threshold of $1 - \alpha$

Despite the network’s output adopting a mathematical distribution format, it does not inherently reflect the actual class distribution. Typically, the network will not be calibrated and it tends to be overly optimistic [33]. Consequently, this straightforward approach doesn’t assure the inclusion of the correct class with the desired probability.

The first step of the CP algorithm involves forming a conformity score $S ( x , y )$ that measures the network’s uncertainty between x and its true label y (larger scores indicate worse agreement). The Homogeneous Prediction Sets (HPS) score [93] is $S _ { \scriptscriptstyle { \mathrm { H } P S } } ( x , y ) = 1 - p ( y | x ; \theta )$ , s.t. θ is the network parameter set. The Adaptive Prediction Score (APS) [82] is the sum of all class probabilities that are not lower than the probability of the true class:

$$
\displaystyle S _ { A P S } ( x , y ) = \sum _ { \{ i | p _ { i } \geq p _ { y } \} } p _ { i } ,\tag{2.6}
$$

such that $p _ { i } = p ( y = i | x ; \theta )$ and $p _ { y }$ is the probability of the label y. The RAPS score [2] is a variant of APS, which is defined as follows:

$$
S _ { R A P S } ( x , y ) = \sum _ { \{ i | p _ { i } \geq p _ { y } \} } p _ { i } + a \cdot \operatorname* { m a x } ( 0 , ( N C - b ) )\tag{2.7}
$$

s.t. $N C = | \{ i | p _ { i } \geq p _ { y } \} |$ and a, b are parameters that need to be tuned. RAPS is especially efective in the case of a large number of classes where it explicitly encourages small prediction sets.

We can also define a randomized version of a conformity score. For example in the case of APS we define:

$$
S _ { r a n d - A P S } ( x , y , u ) = \sum _ { \{ i | p _ { i } > p _ { y } \} } p _ { i } + u \cdot p _ { y } , \qquad u \sim U [ 0 , 1 ] .\tag{2.8}
$$

The random version tends to yield the required coverage more precisely and thus it produces smaller prediction sets [1]. The CP prediction set of a data point x is defined as $C ( x ) = \{ y | S ( x , y ) \leq q \}$ where q is a threshold that is found using a labeled validation set $( x _ { 1 } , y _ { 1 } ) , . . . , ( x _ { n } , y _ { n } )$ . The CP theorem states that if we set q to be the (1−α) quantile of the conformal scores $S ( x _ { 1 } , y _ { 1 } ) , . . . , S ( x _ { n } , y _ { n } )$ we can guarantee that $\begin{array} { r } { 1 - \alpha \le p ( y \in C ( x ) ) \le 1 - \alpha + \frac { 1 } { n + 1 } } \end{array}$ , where x is a test point and y is its the unknown true label [93]. In the random case there is still a coverage guarantee, which is defined by marginalizing over all test points x and samplings u from the uniform distribution [82]. Note that the coverage guarantee is for a marginal probability over all possible test points and coverage may be worse or better for diferent points. It can be proved that obtaining a conditional coverage guarantee is impossible [24].

## 2.3 Unsupervised Target Domain Calibration

When a network trained on data from one domain is applied to data from a diferent domain, the distribution of features often changes between domains, leading to what is known as the domain shift problem (see e.g. [61]). In an Unsupervised Domain Adaptation (UDA) setup, we assume the availability of data from the target domain without any annotations. Numerous UDA methods have been developed to address this issue, employing strategies such as adversarial training to align the distributions of the source and target domains [27], or self-training algorithms that compute pseudo labels for the target domain data [120]. Studies show that present-day UDA methods are prone to learning improved accuracy at the expense of deteriorated prediction confidence [97].

This brings us to the challenge of calibrating the network’s confidence on the target domain data. In UDA, adapting a network to the target domain typically focuses on accuracy improvements, but accurately calibrating the model’s confidence is equally important. Without calibration, the model may become overconfident in its predictions despite reduced performance on the target domain.

Next, we formulate the problem of unsupervised target domain calibration. Assume a network was trained on the source domain. We are given a labeled source domain validation-set dataset, denoted as $\boldsymbol { S } = \{ ( x _ { s } ^ { i } , y _ { s } ^ { i } ) \} _ { i = 1 } ^ { n _ { s } }$ with $n _ { s }$ samples, and an unlabeled target domain dataset $\mathcal { T } = \{ x _ { t } ^ { i } \} _  i =  \bar  $ 1 with $n _ { t }$ samples. Adapting the network trained on the source domain to the target domain in an unsupervised manner without access to the labels can be achieved using various methods. Here, our goal is to calibrate the confidence of the adapted network prediction on samples from the target domain.

Calibrating the confidence of the adapted model on data from the target domain is challenging due to the coexistence of the domain gap and the lack of target labels. Current UDA calibration methods use the labeled validation set from the source domain to approximate the target domain statistics in certain aspects. Some studies [85, 90] propose to modify the calibration set to represent a generic distribution shift. Other methods [69, 97, 68] apply Importance Weighting (IW) by assigning higher weights to source examples that resemble those in the target domain. In practice, even after the domain adaptation process, the accuracy on the source domain, where labels are available, remains greater than the accuracy on the target domain. Hence, the accuracy estimation when calibrating the target domain using the source data is still too optimistic. Calibrating neural networks is necessary because they are often overconfident in their predictions compared to their actual accuracy [33, 49, 36]. If the accuracy is overestimated, it conceals the overconfidence issue, leading to a suboptimal temperature scaling value in the case of temperature scaling. Another drawback of IW methods is that they only use the unlabeled target data to train a binary source/target classifier, but the actual calibration is done on the source domain data while the target domain data are ignored. The network confidence, however, is independent of the true labels and can thus be directly computed

## 2.4 Noisy Labels

In supervised classification tasks a dataset is defined as pairs of input and labels $\mathcal { D } = \{ ( x ^ { i } , y ^ { i } ) \} _ { i = 1 } ^ { n _ { D } }$ In the setting of label noise, we only observe the corrupted labels $\tilde { y } _ { i } = g ( y _ { i } )$ for some corruption function g : $Y \times [ 0 , 1 ] \to Y$

One notable label noise function follows a uniform distribution, where with a probability of ϵ, the correct label is replaced by a randomly selected label 2.9. The noise is applied to each sample independently. This noise model is commonly referred to as uniform noise. The uniform noise may be formulated also such that with a probability of $\epsilon ,$ the correct label is replaced by a randomly selected label from the remaining (k − 1) classes 2.10. In this thesis, we adopt both definitions interchangeably depending on the context, as the transition between them is straightforward.

A more general noise model assumes that the true label is corrupted by a label noise matrix P, where $P ( i , j ) = p ( \tilde { y } = j | y = i )$ is the probability of the true label i being flipped to a corrupted label j. In the simpler uniform noise model, P takes the form of:

$$
P ( i , j ) = ( 1 - \epsilon ) \mathbb { 1 } _ { \{ i = j \} } + \frac { \epsilon } { k } .\tag{2.9}
$$

$$
P ( i , j ) = ( 1 - \epsilon ) \mathbb { 1 } _ { \{ i = j \} } + \frac { \epsilon } { k } \mathbb { 1 } _ { \{ i \neq j \} } .\tag{2.10}
$$

In future chapters, we concentrate on calibrating the network confidence based on noisy labels. A preliminary step before confidence calibration is training the network using a training set with noisy labels (see Figure 3.1). We next provide a brief overview of current training methods that are resilient to label noise and describe the training method we used in our experiments.

There is a plethora of recent works on learning with noisy labels, which include estimating the noise matrix [31, 17, 52, 37, 103, 102, 10], reweighting examples [55, 81, 87, 94], selecting confident examples [40, 106, 8, 50], designing robust loss functions [118, 30, 11, 100], introducing regularization [112, 38, 9] and generating pseudo labels [89, 119, 116, 34, 51]. Zhang et al. [114] addressed the problem of learning from noisy labels in the context of inconsistent annotation collected from several medical experts. They also relied on the fact that the confusion matrix of the noisy labels can be expressed as the matrix product between the confusion matrix of the clean labels and the label noise. Our focus in this context is on learning techniques that deal with noisy data by estimating the label noise matrix. These methods have been successful in producing cutting-edge results, and our calibration approach utilizes the estimated noise matrix to obtain a noise-robust calibration measure.

Noise robust training methods which estimate the noise matrix, are all based on the following observation. The clean class-posterior $p ( y | x )$ can be inferred by utilizing the noisy class-posterior $p ( \tilde { y } | x )$ and the class-dependent noise matrix $P ,$ where $P _ { i j } = P ( \tilde { y } = j | y = i )$ , as follow: $p ( y | x ) =$ $P ^ { - 1 } p ( \tilde { y } | x )$ . While this approach theoretically guarantees statistical consistency, it relies heavily on the success of estimating the noise matrix. Several methods have been developed to estimate the noise matrix under the so-called anchor-point assumption. Anchor points are instances belonging to a specific class with a probability of one [55]. This assumption is reasonable in certain applications but typically, we cannot assume the availability of anchor points. This has motivated the development of noise-robust training algorithms that do not exploit anchor points. Several studies have implemented modifications to the classification network’s architecture to better represent the label noise matrix in noisy datasets [6, 31, 52]. These adjustments encompass the inclusion of a noise adaptation layer on top of the softmax layer and the creation of a specialized architecture. The noise adaptation layer is intended to mimic the label transition behavior in learning a network. These changes have led to enhanced generalization by altering the network output according to the estimated label transition probability. In our implementation, of training of noisy labels we follow the approach in Li et al. [52] which yields state-of-the-art results.

Denote the network’s soft-max label prediction by $p _ { \theta } ( y | x )$ where θ is the network’s parameter set. Given training data $x _ { 1 } , . . . , x _ { n }$ with corresponding noisy labels $\tilde { y } _ { 1 } , . . . , \tilde { y } _ { n }$ , the standard loss function is:

$$
L ( \theta ) = \sum _ { t = 1 } ^ { n } \mathrm { C E } ( \tilde { y } _ { t } , p _ { \theta } ( y _ { t } | x _ { t } ) ) ,\tag{2.11}
$$

such that CE is the cross-entropy loss. Li et al. [52] proposed minimizing the following loss function:

$$
L ( \theta , P ) = \sum _ { t = 1 } ^ { n } { \mathrm { C E } } ( { \tilde { y } } _ { t } , P \cdot p _ { \theta } ( y _ { t } | x _ { t } ) ) + \lambda \log \operatorname* { d e t } ( P )\tag{2.12}
$$

such that $\lambda > 0$ is a regularization coeficient that trades of distribution fidelity with the complexity of the matrix P. P is enforced to be a diagonally dominant stochastic matrix (i.e. $P _ { i , i } > P _ { i , j }$ for every $i \neq j )$ . We first create a matrix A s.t. $A _ { i , i } = 1$ and $A _ { i , j } = \sigma ( w _ { i , j } )$ for all $i \neq j$ where $\sigma$ is the sigmoid function and each $w _ { i , j }$ is a real-valued variable that is updated throughout training. Then we normalize each row to obtain a stochastic matrix: $\begin{array} { r } { P _ { i j } = \frac { A _ { i , j } } { \sum _ { l } A _ { i , l } } } \end{array}$ . Once we finish the training phase, we eliminate the noise adaptation layer defined by matrix $P ,$ , because our objective is to predict the clean label. As a by-product of the training process, we also obtain an approximation of the label noise matrix $P ,$ which we can utilize in our network calibration approach.

Current label-noise learning methods generally assume that the class distribution of the training data is balanced, i.e., that each class is represented by almost the same number of samples. However, data in real-world applications are often imbalanced. In cases where the training labels are both noisy and imbalanced it is dificult to distinguish between clean and noisy samples in rare classes because the clean samples are overwhelmed by noisy labels from frequent classes. Several recent attempts have been made to find noise-robust training procedures for imbalance data [59, 41, 43]. However, all these methods are focused on extracting confident examples. We are not aware of any methods for estimating the noise matrix with unbalanced data.

Additional noise types. Next, we define two common general noise matrices that will be used through out this thesis. The Neighborhood noise as:

$$
P _ { i , j } = p ( \tilde { y } = j | y = i ) = \left\{ \begin{array} { l l } { \xi } & { \mathrm { ~ i f ~ } i = j } \\ { 1 - \xi } & { \mathrm { ~ i f ~ } | i - j | = 1 \mathrm { ~ a n d ~ } i \in ( 1 , k ) } \\ { ( 1 - \xi ) / 2 } & { \mathrm { ~ i f ~ } | i - j | = 1 \mathrm { ~ a n d ~ } i \notin ( 1 , k ) } \\ { 0 } & { \mathrm { ~ o t h e r w i s e } } \end{array} \right.\tag{2.13}
$$

The Random noise is defined as: first, on the diagonal, we have $\xi .$ next, for each line (aka ∀i) the rest of the values (i.e. k − 1 items) are sampled from a random distribution $u _ { i }$ vector of size $k - 1$ and then normalized to sum up to $1 - \xi$ to keep the matrix a probability matrix.

$$
P _ { i , j } = p ( \tilde { y } = j | y = i ) = \left\{ \begin{array} { l l } { \xi \qquad } & { \mathrm { i f ~ } i = j } \\ { \big ( 1 - \xi \big ) \cdot \frac { u _ { i } [ j ] } { \sum _ { z \ne i } u _ { i } [ z ] } \qquad } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{2.14}
$$

In this study, we tackle the challenge of applying calibration on a validation set with noisy labels. Calibration methods, particularly for neural networks, are known to be highly sensitive to label noise. In the context of conformal prediction, Einbinder et al. [21] proposed ignoring the label noise and directly applying the standard CP algorithm to the noisy-labeled validation set. However, this approach tends to result in overly large prediction sets. In contrast, within the field of confidence calibration, we have not encountered any prior research that specifically addresses the issue of calibrating networks using validation sets with noisy labels.

## 2.5 Local Diferential Privacy

Traditional (central) diferential privacy [20] presumes a trusted curator who sees the raw data and then adds noise before publishing. In contrast, LDP [19, 45, 46] treats the aggregator as untrusted:

individual users randomize their own data locally before sending it to the aggregator, thus ensuring strong privacy. LDP is considered a harder setting since noise insertion is done on the user side in a distributed manner, whereas in the centralized DP model the curator holds the entire data and can apply operations on the clean data.

Definition 2.5.1. A discrete randomized mechanism $Q ( \cdot )$ is ε-LDP if for any pair of input labels y, $y ^ { \prime } \in \mathcal { V }$ and any output z,

$$
Q ( z \mid y ) \leq e ^ { \varepsilon } Q ( z \mid y ^ { \prime } ) .
$$

This definition ensures that any two possible labels are (roughly) indistinguishable from the aggregator’s perspective. A common mechanism is the k-ary randomized response (k-RR) [98, 95]. For a label $y \in \{ 1 , \ldots , k \}$ , it outputs:

$$
\tilde { y } = \left\{ \begin{array} { l l } { y , } & { \mathrm { w . p ~ } \frac { e ^ { \varepsilon } } { ( k - 1 ) + e ^ { \varepsilon } } , } \\ { \mathrm { a n y ~ o t h e r ~ l a b e l ~ ( u n i f o r m l y ) , } } & { \mathrm { w . p ~ } \frac { 1 } { ( k - 1 ) + e ^ { \varepsilon } } . } \end{array} \right.
$$

This preserves label privacy, preventing the aggregator from easily inferring the user’s true label from the reported y˜. The parameter ε, known as the privacy loss, controls the privacy-utility trade of: lower ε enforces stronger privacy guarantees but introduces more noise, potentially degrading the utility of downstream applications. Note that if $k = 2$ , we recover Warner’s binary randomized response (RR) [98], flipping the label with some probability.

LDP has gained popularity as a strong privacy paradigm that enables data owners to randomize their data locally before sharing it with an untrusted aggregator, thus ensuring that sensitive information remains protected (e.g. Google’s RAPPOR [22] and Apple’s locally private data collection of emojis and usage patterns [4]).

Challenges of LDP for statistical learning and conformal prediction. While LDP ensures strong privacy guarantees, its main challenge lies in the significant increase in variance due to local randomization. Unlike central DP, where controlled noise can be added post-aggregation, in LDP, the noise is introduced at the user level, leading to a loss of information before any statistical inference is performed. This introduces several key challenges in machine learning and uncertainty quantification:

• Challenges in calibration: Many traditional statistical methods assume access to clean calibration data. However, in an LDP setting, the observed data is randomized, afecting the empirical coverage of conformal prediction intervals.

• Impact on distribution-free guarantees: Conformal prediction provides finite-sample marginal coverage guarantees without assumptions on the underlying data distribution. However, when predictions are made using noisy, privatized data, the standard conformal prediction framework may require adaptation to account for the added uncertainty.

Chapter 3

# Confidence Calibration under Noisy Labels

In this chapter, we explore the critical challenge of confidence calibration in neural networks, particularly under the influence of noisy labels. Confidence calibration is vital in applications such as medical imaging, where overly confident yet incorrect predictions can have serious consequences. We review existing calibration methods, including post-hoc techniques like Temperature Scaling, and discuss the unique dificulties posed by label noise in medical datasets. Finally, we introduce a novel method that leverages noisy validation data to achieve robust calibration, demonstrating its efectiveness across various medical imaging scenarios.

## 3.1 Problem Statement

Confidence calibration is defined as the ability of a classifier network to provide an accurate probability of correctness for any of its predictions. Neural networks have been shown to be more overconfident in their predictions than their predecessors even though their generalization accuracy is higher, partly due to the fact that they can overfit on the negative log-likelihood loss without overfitting on the classification error [33, 49, 36]. In medical imaging applications, images for which the model makes low-confidence predictions are sent to a physician for review. Skipping the human review based on confident but incorrect predictions can have disastrous consequences [62]. The gap between the model’s predicted probabilities and its accuracy is one of the key obstacles to the applicability of neural network models to fully automatic medical diagnosis.

Various confidence calibration methods have recently emerged, aiming to address the issue of excessive overconfidence. Network calibration can be performed in conjunction with training (see e.g. [63, 64, 115, 104]). Post-hoc scaling approaches to calibration (e.g. Platt scaling [80], isotonic regression [110], and temperature scaling [33]) are widely used. In order to enhance calibration, they incorporate calibration as a post-processing step, utilizing hold-out validation data to acquire a calibration map that modifies the model’s predictions. Temperature scaling, which is currently the widely accepted practical calibration method, is a straightforward approach that can be easily implemented. Despite the importance of network calibration for automating medical reports, there are only a few studies that have addressed the problem of calibrating medical imaging systems (see e.g. [23, 25, 83, 111]).

Deep neural networks have been highly successful in various natural image and medical image computing tasks. However, these achievements depend on having accurate annotated training data. Neural networks require massive amounts of carefully labeled data to succeed, but acquiring such data is expensive and time-consuming. Non-expert sources, like Amazon’s Mechanical Turk, have been used to reduce labeling costs, but their labels can be unreliable. Experienced domain experts may also struggle with complex labeling tasks. Medical imaging datasets often have problems with noisy labels due to ambiguous images that can confuse clinical experts. Physicians may disagree on the diagnosis of the same medical image, resulting in variability in the ground truth label. Furthermore, using Natural Language Processing (NLP) tools to extract labels from radiological reports can also introduce label noise [42]. Therefore, addressing annotation noise is a crucial topic in medical image analysis. Training neural networks with noisy labels is problematic because the models can easily overfit to the corrupted labels, resulting lack of generalizability when evaluated on a separate test dataset. While popular regularization techniques have been used to address overfitting, they do not entirely solve the problem. Even when these techniques are applied, there is a significant gap in test accuracy between models trained on clean vs. noisy data, and the accuracy decreases with label noise. Noisy labels are dificult to avoid, and studies indicate that Deep Neural Networks (DNNs) can memorize entire datasets. Consequently, errors in datasets may result in erroneous predictions, which can impact medical diagnoses. Therefore, efectively managing noisy labels is crucial for automated medical image classification. A review of network training methods for noisy labels can be found in [88] and an excellent up-to-date discussion of training medical image classification networks from data with noisy labels can be found in [105].

In the following chapter, we address the challenge of calibrating medical networks with a validation set that has inaccurate labels. Numerous studies have examined the problem of training networks that are resilient to label noise, which can also disrupt the network calibration process. Our findings suggest that network calibration methods are more susceptible to label noise compared to network training. Nevertheless, we have not come across any previous research that tackles the challenge of network calibration using a validation set containing noisy labels. The findings reveal that the Temperature Scaling method [33], which is commonly used, is highly susceptible to label noise and can even result in worse calibration than the original model. We present a simple method that uses data with noisy labels to calibrate a network by taking advantage of the fact that in calibration, we only need to estimate the average accuracy at pre-determined confidence bins, rather than determining the correctness of each label. Testing the method on various medical imaging datasets, network architectures, and noise levels, showed that the calibration results were comparable to those obtained using a noise-free validation set.

The study described in this chapter was published in [74, 77].

## 3.2 Confidence Calibration with Noisy Labels

Consider a multi-class classification task with k classes. Suppose we have a validation set with labels that are potentially inaccurate. Let $y _ { 1 } , . . . , y _ { n }$ be the correct labels of the validation set and let $\tilde { y } _ { 1 } , . . . , \tilde { y } _ { n }$ be the corresponding observed corrupted labels. We assume that the label noise follows a uniform distribution, where with a probability of ϵ, the correct label is replaced by a randomly selected label from the remaining (k − 1) classes. The noise is applied to each sample independently. This noise model is commonly referred to as uniform noise. Our objective is to calibrate the network using these noisy labels.

![](images/77d421c30b546956aa5918c6c6b7bb526f091f667cd9ba34958ae5431acddbb5.jpg)  
Figure 3.1: Schema of the proposed model that includes the full pipeline of network training and calibration based on data with noisy labels.

It can be verified from the adaECE definition (2.4) that only the accuracy terms $\{ A _ { i } \} _ { i = 1 } ^ { m } .$ are afected by the noise, whereas the confidence terms $\{ C _ { i } \} _ { i = } ^ { m }$ remain the same. Let ${ \tilde { A } } _ { i }$ be the average accuracy at bin i that is computed using the corrupted labels. Note that a prediction is considered correct if the label is not corrupted and the network’s prediction matches it, or if the label is corrupted and the network’s (incorrect) prediction matches the corrupted label. This implies that:

$$
\tilde { A } _ { i } = \frac { 1 } { | B _ { i } | } \sum _ { t \in B _ { i } } \mathbb { 1 } _ { \{ \hat { y } _ { t } = \tilde { y } _ { t } \} } = \frac { 1 } { | B _ { i } | } \sum _ { \{ t \in B _ { i } | \tilde { y } _ { t } = y _ { t } \} } \mathbb { 1 } _ { \{ \hat { y } _ { t } = y _ { t } \} } + \frac { 1 } { | B _ { i } | } \sum _ { \{ t \in B _ { i } | \tilde { y } _ { t } \neq y _ { t } \} } \mathbb { 1 } _ { \{ \hat { y } _ { t } = \tilde { y } _ { t } \} } .
$$

The law of large numbers implies that as the size of the validation set increases, the empirical average noisy accuracy at each adaECE bin becomes increasingly close to the mean noisy accuracy. Therefore:

$$
\tilde { A } _ { i } \approx ( 1 - \epsilon ) A _ { i } + { \frac { \epsilon } { k - 1 } } ( 1 - A _ { i } ) .\tag{3.1}
$$

We can thus obtain an estimation ${ \hat { A } } _ { i }$ of the correct accuracy $A _ { i }$ from the noisy accuracy ${ \tilde { A } } _ { i }$ as follows:

$$
\tilde { A } _ { i } = ( 1 - \epsilon ) \hat { A } _ { i } + { \frac { \epsilon } { k - 1 } } ( 1 - \hat { A } _ { i } ) .\tag{3.2}
$$

Rearranging (3.2) we finally obtain:

$$
\hat { A } _ { i } = \frac { \tilde { A } _ { i } - \epsilon / ( k - 1 ) } { ( 1 - \epsilon ) - \epsilon / ( k - 1 ) } .\tag{3.3}
$$

Substituting the estimated accuracy term, based on data with noisy labels (3.3) into the adaECE definition (2.4), yields the following noise-robust adaECE measure:

$$
\mathrm { N o i s y - a d a E C E } ( \epsilon ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left| \frac { \tilde { A } _ { i } - \epsilon / ( k { - } 1 ) } { ( 1 { - } \epsilon ) - \epsilon / ( k { - } 1 ) } - C _ { i } \right| .\tag{3.4}
$$

When training a network with noisy labels we need to make individual decisions for each sample regarding the corruption of its label. The key feature of the Noisy-adaECE metric is that here we only require an estimation of the average label noise within each bin, which is a significantly simpler task.

Algorithm 1 Noisy Temperature Scaling (NTS) - Uniform noise   
input: A validation set $( x _ { 1 } , \tilde { y } _ { 1 } ) , . . . , ( x _ { n } , \tilde { y } _ { n } )$ whose labels are corrupted with noise level $\epsilon .$   
- Feed each $x _ { t }$ into the classifier network to produce class distribution $p _ { t 1 } , . . . , p _ { t k }$ . Compute   
the confidence values $\hat { p } _ { t } = \operatorname* { m a x } _ { j } p _ { t j }$ and the predictions $\hat { y } _ { t } = \arg \operatorname* { m a x } _ { j } p _ { t j } .$   
- Order the points based on their confidence and divide them into equal-sized sets $B _ { 1 } , . . . , B _ { m }$   
- Find $T$ that minimizes the Noisy-adaECE score:   
$\mathrm { N o i s y - a d a E C E } ( T ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left| \operatorname* { m a x } ( 0 , \operatorname* { m i n } ( 1 , \frac { \tilde { A } _ { i } - \epsilon / ( k - 1 ) } { ( 1 - \epsilon ) - \epsilon / ( k - 1 ) } ) ) - C _ { i } ( T ) \right|$   
s.t.   
$C _ { i } ( T ) = \frac { 1 } { | B _ { i } | } \sum _ { t \in B _ { i } } \frac { \exp ( \log ( \hat { p } _ { t } / T ) ) } { \sum _ { l = 1 } ^ { k } \exp ( \log ( p _ { t l } / T ) ) } \qquad \mathrm { ~ a n d ~ } \qquad \tilde { A } _ { i } = \frac { 1 } { | B _ { i } | } \sum _ { t \in B _ { i } } \mathbb { 1 } _ { \{ \hat { y } _ { t } = \tilde { y } _ { t } \} }$   
output: $\hat { T } = \mathop { \mathrm { a r g } } \mathop { \operatorname* { m i n } } _ { \mathrm { \mathbf { N o i s y - a d a E C E } } ( T ) }$

This Noisy-adaECE calibration measure assumes knowledge of the noise level ϵ. If ϵ is not known, we can estimate it from the noisy data (see e.g. [70, 52, 117]). In the next section, we describe the noise level estimation that was used in our experiments. For each calibration method whose parameters can be found by minimizing the adaECE measure, we can form a noise-robust variant in which Noisy-adaECE (3.4) is minimized instead of adaECE (2.4). Examples of these calibration methods include Temperature Scaling (TS), Vector Scaling, Matrix Scaling [33], Mix-n-Match [113], Wight Scaling [25], and others. We next present the noise-robust calibration measure in the case of the TS method. The optimal temperature is obtained by finding the temperature $T$ that minimizes the Noisy-adaECE (3.4) calibration measure. The proposed robust variant of TS which we dub the

Noise-robust Temperature Scaling (NTS) algorithm, is summarized in Algorithm Box 1.

So far we have considered the simplest uniform label noise model. A more general noise model assumes that the true label is corrupted by a label noise matrix $P ,$ where $P ( i , j ) = p ( \tilde { y } = j | y = i )$ is the probability of the true label i being flipped to a corrupted label $j .$ In the simpler uniform noise model, P takes the form of:

$$
P ( i , j ) = ( 1 - \epsilon ) \mathbb { 1 } _ { \{ i = j \} } + \frac { \epsilon } { k - 1 } \mathbb { 1 } _ { \{ i \neq j \} } .\tag{3.5}
$$

We next extend the noise-robust calibration measure Noisy-adaECE defined above to the case of a general label noise matrix P. Denote

$$
M _ { i } ( r , s ) = p _ { i } ( \boldsymbol { \hat { y } } = r , \boldsymbol { y } = s ) = \frac { 1 } { | B _ { i } | } \sum _ { t \in B _ { i } } \mathbb { 1 } _ { \{ \boldsymbol { \hat { y } } _ { t } = r , \boldsymbol { y } _ { t } = s \} } .
$$

$M _ { i }$ is the classifier confusion matrix computed on the validation data from the i-th bin using clean labels. The adaECE accuracy term (2.1) is thus:

$$
A _ { i } = \sum _ { j = 1 } ^ { k } M _ { i } ( j , j ) = \operatorname { T r } ( M _ { i } ) .\tag{3.6}
$$

In a similar manner, we define a confusion matrix based on the available noisy labels:

$$
\tilde { M } _ { i } ( \boldsymbol { r } , s ) = \frac { 1 } { | B _ { i } | } \sum _ { t \in B _ { i } } \mathbb { 1 } _ { \{ \hat { y } _ { t } = \boldsymbol { r } , \tilde { y } _ { t } = s \} } .\tag{3.7}
$$

According to our noise model, given a sample from the validation set along with its true label, the corresponding noisy label and the network soft prediction are conditionally independent. This implies that:

$$
\begin{array} { l } { \displaystyle \tilde { M } _ { i } ( \boldsymbol { r } , s ) = p _ { i } ( \hat { y } = \boldsymbol { r } , \tilde { y } = s ) = \sum _ { j } p _ { i } ( \hat { y } = \boldsymbol { r } , \tilde { y } = s , y = j ) } \\ { \displaystyle \qquad = \sum _ { j } p _ { i } ( \hat { y } = \boldsymbol { r } , y = j ) p ( \tilde { y } = s | \hat { y } = \boldsymbol { r } , y = j ) } \\ { \displaystyle \qquad = \sum _ { j } p _ { i } ( \hat { y } = \boldsymbol { r } , y = j ) p ( \tilde { y } = s | y = j ) = \sum _ { j } M _ { i } ( \boldsymbol { r } , j ) P ( j , s ) . } \end{array}\tag{3.8}
$$

We can write (3.8) as a matrix multiplication: $\tilde { M } _ { i } = M _ { i } P$ . This implies that

$$
M _ { i } = \tilde { M _ { i } } P ^ { - 1 } .\tag{3.9}
$$

By substituting (3.9) in (3.6) we obtain an estimation of the adaECE accuracy term of the clean

data $A _ { i }$ as a function of the confusion matrix of the noisy data $\tilde { M _ { i } }$ and the label noise matrix $P \colon$

$$
\hat { A } _ { i } = \mathrm { T r } ( M _ { i } ) = \mathrm { T r } ( \tilde { M } _ { i } P ^ { - 1 } ) .\tag{3.10}
$$

We note that, by applying this derivation to the case of uniform noise (3.5), we obtain:

$$
\begin{array} { l } { \tilde { A } _ { i } = \displaystyle \sum _ { r } \tilde { M } _ { i } ( r , r ) = \displaystyle \sum _ { r } ( M _ { i } P ) ( r , r ) = \displaystyle \sum _ { s , r } M _ { i } ( r , s ) P ( s , r ) } \\ { = \displaystyle \sum _ { s } ( M _ { i } ( s , s ) ( 1 - \epsilon ) + \displaystyle \sum _ { r \neq s } M _ { i } ( r , s ) ( \frac { \epsilon } { k - 1 } ) ) = ( 1 - \epsilon ) A _ { i } + \frac { \epsilon } { k - 1 } ( 1 - A _ { i } ) . } \end{array}\tag{3.11}
$$

This coincides with the direct derivation of binwise average accuracy for the case of uniform noise.

Algorithm 2 Noisy Temperature Scaling (NTS) - General Noise Matrix   
input: A validation set $( x _ { 1 } , \tilde { y } _ { 1 } ) , . . . , ( x _ { n } , \tilde { y } _ { n } )$ whose labels are corrupted by a noise matrix $P .$   
- Feed each $x _ { t }$ into the classifier network to produce class distribution $p _ { t 1 } , . . . , p _ { t k }$ . Compute   
the confidence values $\hat { p } _ { t } = \operatorname* { m a x } _ { j } p _ { t j }$ and the predictions $\hat { y } _ { t } = \arg \operatorname* { m a x } _ { j } p _ { t j }$   
- Order the points based on their confidence and divide them into equal-sized sets $B _ { 1 } , . . . , B _ { m }$   
- Find $T$ that minimizes the Noisy-adaECE score:   
Noisy-adaECE(T) = ${ \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } { \bigg | }$ max(0, min $( 1 , \mathrm { T r } ( \tilde { M } _ { i } P ^ { - 1 } ) ) ) - C _ { i } ( T ) \Big |$   
s.t.   
$C _ { i } ( T ) = \frac { 1 } { | B _ { i } | } \sum _ { t \in B _ { i } } \frac { \exp ( \log ( \hat { p } _ { t } / T ) ) } { \sum _ { l = 1 } ^ { k } \exp ( \log ( p _ { t l } / T ) ) }$ and $\tilde { M } _ { i } ( \boldsymbol { r } , s ) = \frac { 1 } { | B _ { i } | } \sum _ { t \in B _ { i } } \mathbb { 1 } _ { \{ \hat { y } _ { t } = \boldsymbol { r } , \tilde { y } _ { t } = s \} }$   
output: $\hat { T } = \mathop { \mathrm { a r g } \mathrm { m i n } } \mathrm { N o i s y - a d a E C E } ( T )$

The Noise-adaECE in the case of a general noise matrix P is defined by:

$$
\mathrm { N o i s y - a d a E C E } ( P ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left| \mathrm { T r } ( \tilde { M } _ { i } P ^ { - 1 } ) - C _ { i } \right| ,\tag{3.12}
$$

such that $\tilde { M }$ is the confusion matrix of the noisy validation set data (3.7) and we used (3.10) to estimate the binwise average clean data accuracy. To apply TS calibration in the case of a general noise model, we need to find a temperature $T$ that minimizes the Noise-adaECE expression (3.12). In the case where the label noise matrix P is not known, there is a plethora of methods for estimating $P$ without accessing clean labels [70, 52, 117]. The Noisy Temperature Scaling (NTS) for the case of a general noise matrix, is summarized in Algorithm Box 2. Figure 3.1 illustrates the entire pipeline composed of the noisy label training followed by the noisy label calibration process.

## 3.3 Network Training with Noisy Labels

To achieve robust confidence calibration, it is crucial to first train the network in a manner that accounts for label noise. Various approaches have been developed to mitigate the adverse efects of noisy labels, ranging from reweighting and selecting reliable samples to modifying the network architecture to model label noise explicitly. Among these, a particularly efective strategy involves estimating the label noise matrix and incorporating it into the training process (see Background Section 2.4 for details).

Our approach follows the provably consistent method proposed by Li et al. [52], where a noise adaptation layer is introduced to learn the label transition probabilities. This method optimizes a loss function that balances prediction fidelity with a regularization term enforcing the structural properties of the estimated noise matrix. Once training is complete, the noise adaptation layer is removed, allowing the network to predict clean labels while retaining an estimated noise matrix as a by-product. This estimated noise matrix plays a key role in our confidence calibration procedure.

By leveraging noise-robust training, we ensure that the network maintains reliable predictive confidence even when exposed to datasets with significant label noise. In the following sections, we detail how this estimated noise matrix enhances the calibration process.

Figure 3.1 provides a flow diagram of our method in conjunction with model training. A noiserobust training method [52] is applied to the noisy training data, yielding a trained network and an estimation of the noise matrix. Then, given a noisy validation set, we apply our NTS method to calibrate the network confidence.

## 3.4 Experiments

We implemented the proposed NTS noisy calibration method on various medical imaging classification tasks to evaluate its performance. We share our code for reproducibility <sup>1</sup>. The experimental setup included the following medical imaging classification datasets:

• ChestX-ray14 [96]: A huge dataset that contains 112,120 frontal-view X-ray images of 30,805 unique patients of size 1024×1024, individually labeled with up to 14 diferent thoracic diseases. The original dataset is multi-label. The problem is treated as a multi-class task by choosing the samples containing only one annotated positive label or the ”No-finding” case without any positive label. More than 60% of the images belong to this class, which makes the dataset highly imbalanced. We used a train/validation/test split of 89,696/11,212/11,212 images. We used a balanced variant of the dataset denoted by ChestX-ray14+bal which only included images with exactly one annotated positive label.

• HAM10000 [91]: This dataset contains 10,015 dermatoscopic images of size 800 × 600. Cases include a representative collection of 7 diagnostic categories in the realm of pigmented lesions. We used a train/validation/test split of 8,013/1,001/1,001 images.

• PathMNIST [108]: A dataset that contains 97,176 images of Colon Pathology with nine classes. The images’ size is 28×28. Here, we used a train/validation/test split of 89,996/3,590/3,590 images.

Each dataset was fine-tuned on pre-trained ResNet-18, ResNet-50 [35], and DenseNet-121 [39] networks. The models were taken from the PyTorch site <sup>2</sup>. These network architectures were selected because of their widespread use in classification problems. The last fully-connected layer output size of each was adjusted to fit the corresponding number of classes for each dataset. All the models were fine-tuned using the Adam optimizer [47].

The compared calibration methods. (1) $\mathrm { T S } _ { n o i s y } \ - \ \mathrm { A }$ standard TS that was applied to the validation set with noisy labels. (2) ${ \mathrm { V S } } _ { n o i s y }$ and ${ \mathrm { M S } } _ { n o i s y }$ - Vector Scaling (VS) and Matrix Scaling (MS) calibration methods [33] were applied to the validation set with noisy labels. (3) The proposed Noisy Temperature Scaling (NTS). (4) $\mathrm { T S } _ { c l e a n }$ - Temperature Scaling that was applied to the clean validation set. The main comparison was between the $\mathrm { T S } _ { n o i s y }$ and NTS that were applied to the noisy validation dataset with the same label corruption. $\mathrm { T S } _ { c l e a n }$ is served as an upper bound on the performance of NTS. We are not aware of any other noise robust calibration strategies. For each method, we report the adaECE and ECE scores (computed using 15 bins) on the test set. Although adaECE was used as the objective function in our algorithm and is a more robust calibration measure,

![](images/3d95897b0ea36862fe6d0db8fa59188b379c71a2b366e9db4bd23c8acb0198c1.jpg)

![](images/348e357b218f42cd66cf9c47e739bf60b16f0281b8d0ddaf684723cfee80919c.jpg)

![](images/705aa371263b57ce0ad0b0550649c351ed1186c835cd22e72ccd8d0ccdef5db4.jpg)

![](images/d41241345dbb6c0fadf56c8d2d2981bf8e3232abcf2c89f3ee34f69cae8bb250.jpg)

![](images/43d5258130e5ed6af3d58dd5ac3c455b52b1ee14e726ce6bbbb991cdd41c7797.jpg)  
(a) ChestX-ray14  
(b) HAM-10000

![](images/69170ec65e4e475fa7dd67b36a6ebd3ee6ba2042c4de687a7f8edd1c1492a182.jpg)  
(c) PathMNIST  
Figure 3.2: Comparative calibration results on several datasets that were trained with ResNet-50. The top row shows the ECE results on the (clean) test set. The bottom row shows the optimal temperature found on the noisy validation set.

Table 3.1: Adaptive ECE for top-1 predictions (in %) using 15 bins. on various medical imaging classification datasets and models with diferent calibration methods with a noise level $\epsilon = 0 . 2 .$ . The lowest is highlighted in bold.
<table><tr><td>Dataset</td><td>Architecture</td><td>Acc (%)</td><td> $\mathrm { U n c a l i b r a t e d }$ </td><td> $\mathrm { T S } _ { c l e a n }$ </td><td> ${ \mathrm { V S } } _ { n o i s y }$ </td><td> ${ \mathrm { M S } } _ { n o i s y }$ </td><td> $\mathrm { T S } _ { n o i s y }$ </td><td>NTS</td></tr><tr><td rowspan="3">ChestX-ray14</td><td>ResNet-18</td><td>65.80 ± 0.01</td><td> $1 . 9 2 \pm 0 . 4 3$ </td><td> $1 . 5 2 \pm 0 . 1 9$ </td><td> $1 2 . 2 5 \pm 0 . 1 7$ </td><td> $1 2 . 2 7 \pm 0 . 1 7$ </td><td> $1 1 . 9 0 \pm 0 . 4 6$ </td><td> ${ \bf 1 . 4 5 \pm 0 . 1 5 }$ </td></tr><tr><td>ResNet-50</td><td>65.84 ± 0.02</td><td> $4 . 5 9 \pm 0 . 6 2$ </td><td> $1 . 6 3 \pm 0 . 2 5$ </td><td> $1 2 . 2 6 \pm 0 . 2 1$ </td><td> $1 2 . 2 2 \pm 0 . 1 8$ </td><td> $1 1 . 8 1 \pm 0 . 2 8$ </td><td> ${ \bf 1 . 5 2 \pm 0 . 1 9 }$ </td></tr><tr><td>Densenet-121</td><td> $6 5 . 6 7 \pm 0 . 0 1$ </td><td> $3 . 6 1 \pm 0 . 7 5$ </td><td> $1 . 4 9 \pm 0 . 1 0$ </td><td> $1 2 . 2 7 \pm 0 . 1 8$ </td><td> $1 2 . 2 4 \pm 0 . 1 6$ </td><td> $1 2 . 1 5 \pm 0 . 4 9$ </td><td> $\mathbf { 1 . 5 0 \ : \pm { \ : 0 . 0 6 } }$ </td></tr><tr><td rowspan="3">HAM-10000</td><td>ResNet-18</td><td>88.16 ± 0.17</td><td> $4 . 3 8 \pm 1 . 5 2$ </td><td> $3 . 1 2 \pm 0 . 7 4$ </td><td> $1 9 . 7 5 \pm 2 . 4 0$ </td><td> $1 7 . 1 7 \pm 0 . 9 1$ </td><td> $1 4 . 2 2 \pm 5 . 8 8$ </td><td> ${ \bf 3 . 6 7 \pm 0 . 5 9 }$ </td></tr><tr><td>ResNet-50</td><td>91.06 ± 0.01</td><td> $3 . 5 1 \pm 0 . 6 9$ </td><td> $1 . 9 8 \pm 0 . 3 4$ </td><td> $1 9 . 9 5 \pm 0 . 7 3$ </td><td> $1 8 . 0 3 \pm 0 . 5 4$ </td><td> $1 8 . 7 5 \pm 1 . 4 9$ </td><td> ${ \bf 3 . 0 4 } \pm { \bf 1 . 0 8 }$ </td></tr><tr><td>Densenet-121</td><td> $8 9 . 4 3 \pm 0 . 1 8$ </td><td> $3 . 7 5 \pm 1 . 5 3$ </td><td> $2 . 6 7 \pm 0 . 1 1$ </td><td> $1 8 . 8 5 \pm 1 . 2 3 $ </td><td> $1 7 . 5 6 \pm 1 . 0 9$ </td><td> $1 7 . 5 0 \pm 2 . 0 4$ </td><td> $\mathbf { 2 . 7 4 \ : \pm { \ : 0 . 6 8 } }$ </td></tr><tr><td rowspan="3">PathMNIST</td><td>ResNet-18</td><td> $8 4 . 6 8 \pm 0 . 0 5$ </td><td> $1 3 . 1 4 \pm 0 . 5 1$ </td><td> $2 . 1 9 \pm 0 . 2 0$ </td><td> $2 3 . 1 9 \pm 0 . 5 7$ </td><td> $2 2 . 9 9 \pm 0 . 0 5$ </td><td> $1 1 . 9 4 \pm 1 . 3 0$ </td><td> ${ \bf 2 . 3 7 \pm 0 . 5 2 }$ </td></tr><tr><td>ResNet-50</td><td> $8 5 . 9 7 \pm 0 . 0 4$ </td><td> $1 1 . 5 0 \pm 0 . 4 4$ </td><td> $2 . 2 3 \pm 0 . 3 5$ </td><td>21.47 ± 3.71</td><td> $2 7 . 3 5 \pm 3 . 9 6$ </td><td> $1 3 . 0 0 \pm 0 . 3 2$ </td><td> ${ \bf 2 . 4 2 \pm 0 . 4 8 }$ </td></tr><tr><td>Densenet-121</td><td> $8 6 . 5 4 \pm 0 . 0 5$ </td><td> $1 1 . 1 2 \pm 0 . 4 3$ </td><td> $1 . 8 3 \pm 0 . 4 9$ </td><td> $2 5 . 4 7 \pm 0 . 7 1$ </td><td> $2 5 . 5 8 \pm 0 . 4 2$ </td><td> $1 2 . 3 1 \pm 1 . 0 0$ </td><td> ${ \bf 2 . 5 5 \pm 0 . 3 5 }$ </td></tr></table>

Table 3.2: ECE for top-1 predictions (in %) using 15 bins. on various medical imaging classification datasets and models with diferent calibration methods with a noise level $\epsilon = 0 . 2 .$ . The lowest is highlighted in bold.
<table><tr><td>Dataset</td><td>Architecture</td><td> Uncalibrated</td><td> $\mathrm { T S } _ { c l e a n }$ </td><td> ${ \mathrm { V S } } _ { n o i s y }$ </td><td> ${ \mathrm { M S } } _ { n o i s y }$ </td><td> $\mathrm { T S } _ { n o i s y }$ </td><td>NTS</td></tr><tr><td rowspan="3">ChestX-ray14</td><td>ResNet-18</td><td> $1 . 9 3 \pm 0 . 4 9$ </td><td> $1 . 2 4 \pm 0 . 1 4$ </td><td> $1 2 . 2 5 \pm 0 . 1 7$ </td><td> $1 2 . 2 7 \pm 0 . 1 8$ </td><td> $1 1 . 9 2 \pm 0 . 4 6$ </td><td> ${ \bf 1 . 2 1 \pm 0 . 1 2 }$ </td></tr><tr><td>ResNet-50</td><td> $4 . 5 1 \pm 0 . 3 0$ </td><td> $1 . 3 5 \pm 0 . 3 2$ </td><td> $1 2 . 2 4 \pm 0 . 2 2$ </td><td> $1 2 . 2 1 \pm 0 . 2 0$ </td><td> $1 1 . 8 4 \pm 0 . 2 8$ </td><td> ${ \bf 1 . 2 6 \pm 0 . 2 9 }$ </td></tr><tr><td>Densenet-121</td><td> $3 . 7 1 \pm 0 . 8 0$ </td><td> $1 . 2 2 \pm 0 . 1 9$ </td><td> $1 2 . 2 5 \pm 0 . 1 8$ </td><td> $1 2 . 2 4 \pm 0 . 1 7$ </td><td> $1 2 . 1 0 \pm 0 . 4 9$ </td><td> ${ \bf 1 . 2 6 \pm 0 . 0 6 }$ </td></tr><tr><td rowspan="3">HAM-10000</td><td>ResNet-18</td><td> $2 . 7 2 \pm 1 . 4 1$ </td><td> $0 . 7 9 \pm 0 . 3 5$ </td><td> $1 8 . 8 4 \pm 1 . 6 1$ </td><td> $1 4 . 8 7 \pm 4 . 0 1$ </td><td> $1 2 . 5 3 \pm 6 . 5 1$ </td><td> ${ \bf 2 . 5 2 \pm 0 . 6 9 }$ </td></tr><tr><td>ResNet-50</td><td> $1 . 6 9 \pm 0 . 6 1$ </td><td> $0 . 6 1 \pm 0 . 1 2$ </td><td> $1 9 . 0 0 \pm 0 . 1 4$ </td><td> $1 7 . 5 0 \pm 0 . 8 8$ </td><td> $1 6 . 5 0 \pm 2 . 4 1$ </td><td> ${ \bf 1 . 4 1 \pm 1 . 1 0 }$ </td></tr><tr><td>Densenet-121</td><td> $3 . 0 4 \pm 1 . 6 7$ </td><td> $1 . 2 3 \pm 0 . 3 4$ </td><td> $1 8 . 3 9 \pm 1 . 7 4$ </td><td> $1 7 . 1 3 \pm 1 . 2 4$ </td><td> $1 6 . 2 3 \pm 2 . 1 3$ </td><td> ${ \bf 1 . 7 2 \pm 0 . 7 4 }$ </td></tr><tr><td rowspan="3">PathMNIST</td><td>ResNet-18</td><td> $1 3 . 1 2 \pm 0 . 5 2$ </td><td> $1 . 8 9 \pm 0 . 1 1$ </td><td> $2 3 . 2 2 \pm 0 . 5 9$ </td><td> $2 3 . 0 0 \pm 0 . 0 6$ </td><td> $1 1 . 9 1 \pm 1 . 3 5$ </td><td> $\mathbf { 2 . 0 9 \ : \pm { \ : 0 . 4 6 } }$ </td></tr><tr><td>ResNet-50</td><td> $1 1 . 5 0 \pm 0 . 5 2$ </td><td> $2 . 3 0 \pm 0 . 3 4$ </td><td> $2 1 . 3 1 \pm 3 . 9 3$ </td><td> $2 7 . 1 4 \pm 4 . 7 5$ </td><td> $1 2 . 9 0 \pm 0 . 1 1$ </td><td> ${ \bf 2 . 5 2 \pm 0 . 4 9 }$ </td></tr><tr><td>Densenet-121</td><td> $1 1 . 0 2 \pm 0 . 4 0$ </td><td> $1 . 8 2 \pm 0 . 5 4$ </td><td> $2 5 . 2 9 \pm 0 . 6 7$ </td><td> $2 5 . 5 8 \pm 0 . 4 4$ </td><td> $1 2 . 0 2 \pm 0 . 7 9$ </td><td> ${ \bf 2 . 5 8 \pm 0 . 4 6 }$ </td></tr></table>

![](images/c1e33d4a5feb433fb93d887e1b2679e3ae2a0a6c2cc7426c54bbe731ac8bd75e.jpg)  
(a) ResNet-18

![](images/55fe8a2873fa0a8a8772d8ba112ac48e7f88fe7a1a72c69ad783caba03d4c69b.jpg)  
(b) ResNnet-50

![](images/e28fae165e770220d139d5cf771026812f7e6878ab2209183e00741afbc21c12.jpg)  
(c) Densenet-121  
Figure 3.3: Standard deviation of ECE scores on the test set after the NTS calibration as a function of the size of the noisy validations set.

Table 3.3: Adaptive ECE for top-1 predictions (in %) using 15 bins (with the lowest in bold) on various medical imaging classification datasets and models with diferent calibration methods with varying noise levels. Model training and noise matrix estimation used [52].
<table><tr><td>Dataset</td><td>Noise level (%)</td><td></td><td>Acc (%) | Uncalibrated</td><td> $\mathrm { T S } _ { c l e a n }$ </td><td>NTS(P)</td><td> $\mathrm { T S } _ { n o i s y }$ </td><td>NTS(P)</td></tr><tr><td rowspan="4">ChestX-ray14-bal</td><td>0</td><td> $4 4 . 3 7 \pm 0 . 1 2$ </td><td> $2 6 . 2 1 \pm 0 . 5 6$ </td><td> $2 . 8 8 \pm 0 . 4 8$ </td><td></td><td></td><td></td></tr><tr><td>5</td><td> $4 4 . 1 0 \pm 0 . 0 8$ </td><td> $2 5 . 5 2 \pm 0 . 7 9$ </td><td> $3 . 1 8 \pm 0 . 4 4$ </td><td> $3 . 2 0 \pm 0 . 4 6$ </td><td> $5 . 9 1 \pm 0 . 4 6$ </td><td> ${ \bf 5 . 2 7 \pm 1 . 4 1 }$ </td></tr><tr><td>10</td><td> $4 3 . 7 6 \pm 0 . 3 1$ </td><td> $2 3 . 8 5 \pm 1 . 0 2$ </td><td> $3 . 4 1 \pm 0 . 3 0$ </td><td> $3 . 4 0 \pm 0 . 5 4$ </td><td> $7 . 3 5 \pm 1 . 1 8$ </td><td> ${ \bf 6 . 0 8 \pm 1 . 5 2 }$ </td></tr><tr><td>20</td><td> $4 3 . 0 2 \pm 0 . 2 8$ </td><td> $2 1 . 3 4 \pm 0 . 3 3$ </td><td> $3 . 3 0 \pm 0 . 3 6$ </td><td> $3 . 3 3 \pm 0 . 4 4$ </td><td> $1 0 . 4 8 \pm 0 . 6 3$ </td><td> ${ \bf 6 . 4 3 \pm 0 . 9 6 }$ </td></tr><tr><td rowspan="4">ChestX-ray14</td><td>0</td><td> $6 5 . 1 3 \pm 0 . 0 9$ </td><td> $2 1 . 5 0 \pm 0 . 1 8$ </td><td> $5 . 7 0 \pm 0 . 6 3$ </td><td></td><td></td><td></td></tr><tr><td>5</td><td> $6 4 . 9 5 \pm 0 . 0 5$ </td><td> $1 8 . 7 1 \pm 0 . 2 0$ </td><td> $4 . 4 6 \pm 0 . 6 7$ </td><td> $4 . 2 9 \pm 0 . 4 4$ </td><td> ${ \bf 5 . 8 6 \pm 0 . 4 9 }$ </td><td> $1 2 . 8 \pm 1 . 3 4$ </td></tr><tr><td>10</td><td> $6 4 . 7 6 \pm 0 . 3 6$ </td><td> $1 9 . 0 4 \pm 2 . 8 5$ </td><td> $4 . 3 4 \pm 0 . 5 5$ </td><td> $4 . 4 2 \pm 0 . 3 3$ </td><td> ${ \bf 6 . 8 1 \pm 0 . 6 4 }$ </td><td> $9 . 6 5 \pm 0 . 7 8$ </td></tr><tr><td>20</td><td> $6 4 . 7 1 \pm 0 . 0 9$ </td><td> $1 8 . 2 2 \pm 3 . 2 6$ </td><td> $4 . 6 4 \pm 0 . 5 4$ </td><td> $4 . 6 4 \pm 0 . 5 4$ </td><td> $1 2 . 0 2 \pm 1 . 7 4$ </td><td> ${ \bf 7 . 2 8 \pm 1 . 7 5 }$ </td></tr><tr><td rowspan="4">HAM-10000</td><td>0</td><td> $9 0 . 9 4 \pm 0 . 7 8$ </td><td> $3 . 9 5 \pm 0 . 6 7$ </td><td> $2 . 4 8 \pm 1 . 0 2$ </td><td></td><td></td><td></td></tr><tr><td>5</td><td> $9 0 . 6 4 \pm 0 . 8 3$ </td><td> $3 . 4 7 \pm 1 . 1 6$ </td><td> $3 . 1 9 \pm 1 . 0 3$ </td><td> $2 . 9 8 \pm 0 . 4 9$ </td><td> $4 . 3 4 \pm 0 . 5 7$ </td><td> ${ \bf 3 . 4 1 \pm 1 . 2 1 }$ </td></tr><tr><td>10</td><td> $9 0 . 1 6 \pm 0 . 8 1$ </td><td> $2 . 9 5 \pm 0 . 3 2$ </td><td> $2 . 7 0 \pm 0 . 3 9$ </td><td> $2 . 7 1 \pm 0 . 2 9$ </td><td> $8 . 0 8 \pm 1 . 1 2$ </td><td> ${ \bf 2 . 7 6 \pm 0 . 3 1 }$ </td></tr><tr><td>20</td><td> $8 9 . 7 9 \pm 1 . 1 1$ </td><td> $4 . 7 8 \pm 0 . 6 7$ </td><td> $3 . 5 6 \pm 0 . 7 2$ </td><td> $4 . 4 4 \pm 0 . 9 7$ </td><td> $1 9 . 1 2 \pm 1 . 2 8$ </td><td> ${ \bf 6 . 5 2 \pm 2 . 3 5 }$ </td></tr><tr><td rowspan="4">PathMNIST</td><td>0</td><td> $8 5 . 3 3 \pm 0 . 9 9$ </td><td> $4 . 5 2 \pm 0 . 7 5$ </td><td> $1 . 3 9 \pm 0 . 6 1$ </td><td></td><td></td><td></td></tr><tr><td>5</td><td> $8 5 . 3 0 \pm 0 . 6 7$ </td><td> $3 . 7 1 \pm 0 . 4 9$ </td><td> $1 . 7 3 \pm 0 . 2 8$ </td><td> $1 . 6 5 \pm 0 . 2 6$ </td><td> $4 . 0 8 \pm 0 . 3 7$ </td><td> ${ \bf 1 . 8 6 \pm 0 . 3 7 }$ </td></tr><tr><td>10</td><td> $8 5 . 2 6 \pm 0 . 5 5$ </td><td> $3 . 7 3 \pm 0 . 7 1$ </td><td> $1 . 6 7 \pm 0 . 6 2$ </td><td> $1 . 5 6 \pm 0 . 5 9$ </td><td> $7 . 0 9 \pm 0 . 6 1$ </td><td> ${ \bf 2 . 4 4 \pm 0 . 9 2 }$ </td></tr><tr><td>20</td><td> $8 4 . 4 6 \pm 0 . 5 7$ </td><td> $3 . 3 6 \pm 0 . 6 3$ </td><td> $1 . 7 3 \pm 0 . 1 6$ </td><td> $2 . 3 3 \pm 0 . 8 2$ </td><td> $1 2 . 3 0 \pm 1 . 3 4$ </td><td> ${ \bf 2 . 6 3 \pm 0 . 1 9 }$ </td></tr></table>

ECE is still a standard measure to evaluate calibration results, so we also used it to compare our calibration results to previous studies.

Results of calibration with uniform noise. We first applied the calibration methods to a validation set corrupted by noise with noise level $\epsilon = 0 . 2$ . Table 3.1 reports the calibration results using adaECE. We report the mean and standard deviation over 3 diferent trained models and noise labels sampling. The results indicate that when the validation set contains noise, the standard $\mathrm { T S } _ { n o i s y }$ method fails to properly calibrate the network and may make the calibration worse. The same behavior was observed for ${ \mathrm { V S } } _ { n o i s y }$ and $\mathrm { N T S } _ { n o i s y }$ . By contrast, the proposed NTS method produces calibration results that are similar to those obtained by the $\mathrm { T S } _ { c l e a n }$ method that has access to the clean data. we tested the assumption that (in the case of noise level $\epsilon = 0 . 2 )$ NTS has no efect compared to $\mathrm { T S } _ { n o i s y }$ and its p-value was 0.005. Note that since NTS works well, the temperatures computed by NTS and $\mathrm { T S } _ { c l e a n }$ are similar so that the performance of NTS can even

Table 3.4: ECE for top-1 predictions (in %) using 15 bins (with the lowest in bold) on various medical imaging classification datasets and models with diferent calibration methods with varying noise levels. Model training and noise matrix estimation used [52].
<table><tr><td>Dataset</td><td>Noise level (%)</td><td>Uncalibrated</td><td> $\mathrm { T S } _ { c l e a n }$ </td><td>NTS(P)</td><td> $\mathrm { T S } _ { n o i s y }$ </td><td>NTS(P)</td></tr><tr><td rowspan="4">ChestX-ray14-bal</td><td>0</td><td> $2 6 . 2 1 \pm 0 . 5 6$ </td><td> $2 . 6 3 \pm 0 . 3 2$ </td><td></td><td></td><td></td></tr><tr><td>5</td><td> $2 5 . 5 3 \pm 0 . 8 0$ </td><td> $3 . 0 3 \pm 0 . 3 7$ </td><td> $3 . 0 8 \pm 0 . 3 8$ </td><td> $4 . 8 9 \pm 0 . 8 2$ </td><td> ${ \bf 3 . 8 3 \pm 0 . 3 6 }$ </td></tr><tr><td>10</td><td> $2 3 . 8 7 \pm 1 . 0 1$ </td><td> $3 . 0 0 \pm 0 . 6 1$ </td><td> $2 . 9 2 \pm 0 . 4 6$ </td><td> $7 . 4 9 \pm 0 . 7 3$ </td><td> ${ \bf 4 . 1 3 \pm 0 . 3 9 }$ </td></tr><tr><td>20</td><td> $2 1 . 3 7 \pm 0 . 3 5$ </td><td> $3 . 3 3 \pm 0 . 3 7$ </td><td> $3 . 2 6 \pm 0 . 2 8$ </td><td> $9 . 7 8 \pm 0 . 9 8$ </td><td> ${ \bf 4 . 9 4 \pm 0 . 1 9 }$ </td></tr><tr><td rowspan="4">ChestX-ray14</td><td>0</td><td> $2 1 . 0 5 \pm 0 . 5 5$ </td><td> $5 . 8 6 \pm 0 . 8 2$ </td><td></td><td></td><td></td></tr><tr><td>5</td><td> $1 8 . 7 3 \pm 0 . 1 8$ </td><td> $4 . 4 3 \pm 0 . 7 1$ </td><td> $4 . 2 5 \pm 0 . 4 9$ </td><td> ${ \bf 5 . 7 4 \pm 0 . 6 8 }$ </td><td> $1 2 . 9 \pm 1 . 4 4$ </td></tr><tr><td>10</td><td> $1 9 . 1 1 \pm 2 . 8 5$ </td><td> $4 . 2 1 \pm 0 . 7 5$ </td><td> $4 . 3 0 \pm 0 . 6 2$ </td><td> ${ \bf 7 . 0 6 \pm 0 . 2 2 }$ </td><td> $1 0 . 0 \pm 1 . 2 7$ </td></tr><tr><td>20</td><td> $1 6 . 2 4 \pm 2 . 0 2$ </td><td> $4 . 9 5 \pm 1 . 0 1$ </td><td> $4 . 9 8 \pm 1 . 0 8$ </td><td> $1 1 . 6 \pm 0 . 9 3$ </td><td> ${ \bf 5 . 6 2 \pm 0 . 7 9 }$ </td></tr><tr><td rowspan="4">HAM-10000</td><td>0</td><td> $1 . 8 3 \pm 0 . 2 7$ </td><td> $1 . 1 9 \pm 0 . 5 9$ </td><td></td><td></td><td></td></tr><tr><td>5</td><td> $1 . 5 4 \pm 0 . 3 8$ </td><td> $1 . 3 1 \pm 0 . 3 8$ </td><td> $2 . 0 3 \pm 0 . 9 4$ </td><td> $4 . 1 0 \pm 0 . 8 7$ </td><td> ${ \bf 1 . 3 1 \pm 0 . 3 3 }$ </td></tr><tr><td>10</td><td> $1 . 5 3 \pm 0 . 1 3$ </td><td> $1 . 4 1 \pm 0 . 4 3$ </td><td> $2 . 3 5 \pm 0 . 4 7$ </td><td> $6 . 6 1 \pm 1 . 5 6$ </td><td> ${ \bf 1 . 4 0 \pm 0 . 6 8 }$ </td></tr><tr><td>20</td><td> $3 . 4 9 \pm 0 . 8 0$ </td><td> $0 . 9 4 \pm 0 . 3 0$ </td><td> $3 . 3 0 \pm 1 . 6 8$ </td><td> $1 2 . 3 \pm 8 . 8 2$ </td><td> ${ \bf 5 . 3 7 \pm 2 . 0 2 }$ </td></tr><tr><td rowspan="4">PathMNIST</td><td>0</td><td> $4 . 4 8 \pm 0 . 7 6$ </td><td> $1 . 5 8 \pm 0 . 0 4$ </td><td></td><td></td><td></td></tr><tr><td>5</td><td> $4 . 0 0 \pm 0 . 0 1$ </td><td> $1 . 4 0 \pm 0 . 0 0$ </td><td> $1 . 6 5 \pm 0 . 0 1$ </td><td> $3 . 9 1 \pm 0 . 0 0$ </td><td> ${ \bf 1 . 3 4 } \pm \mathbf { 0 . 0 1 }$ </td></tr><tr><td>10</td><td> $3 . 8 2 \pm 0 . 4 5$ </td><td> $1 . 7 3 \pm 0 . 3 7$ </td><td> $1 . 6 5 \pm 0 . 1 7$ </td><td> $6 . 4 1 \pm 0 . 8 0$ </td><td> $\mathbf { 1 . 6 8 \ : \pm { \ : 0 . 3 9 } }$ </td></tr><tr><td>20</td><td> $3 . 4 2 \pm 0 . 4 4$ </td><td> $1 . 9 7 \pm 0 . 5 1$ </td><td> $2 . 4 2 \pm 0 . 4 1$ </td><td> $1 2 . 5 \pm 1 . 3 6$ </td><td> ${ \bf 2 . 5 7 \pm 0 . 2 9 }$ </td></tr></table>

be slightly better than $\mathrm { T S } _ { c l e a n }$ on the test set.

In this experiment we assume that the noise level is known and the network was trained using clean data. Below we show the results of end-to-end experiments where the noise matrix is estimated during the training procedure.

We next show that our noise-tolerant NTS method works well across various levels of noise. Figure 3.2 displays the adaECE calibration results for various classification tasks with noise levels ϵ that ranged from 0% to 40%. While higher noise levels negatively impacted the calibration of $\mathrm { T S } _ { n o i s y } ,$ NTS demonstrated resilience even in the presence of high levels of noise. Figure 3.2 also illustrates the optimal temperature obtained for each experiment. It shows that the optimal temperature found by $\mathrm { T S } _ { n o i s y }$ increased in a linear manner with the noise level. The presence of noisy labels led to an underestimation of accuracy in each adaECE bin, causing the $\mathrm { T S } _ { n o i s y }$ algorithm to incorrectly conclude that the network was over-confident, thus resulting in an aggressive calibration using a high temperature. Our proposed NTS algorithm avoids this misinterpretation, by computing an accurate estimation of the network accuracy. It is worth mentioning that typical neural network training techniques can handle a limited amount of incorrectly labeled data in the training set, typically less than 10% (see e.g. [105]). However, $\mathrm { T S } _ { n o i s y }$ does not perform well under these conditions.

Our method is justified by the law of large numbers (see (3.1)). Therefore, we expected a correlation between the size of the validation set used for calibration and the stability of the NTS method when dealing with diferent noise samples. To verify this, we selected ChestX-ray14 as a dataset from our experiments and generated various validation set sizes (using the sklearn split function). For each sample size, and noise level ϵ, we created multiple noisy versions of the validation set. For each noisy version we applied the NTS algorithm and computed the adaECE score on the test set. Finally, we computed the standard deviation (STD) of all the adaECE scores. The results, as shown in Figure 3.3, indicate that as the validation set size increased, the standard deviation of the NTS algorithm decreased, thus resulting in more stable model accuracy estimates for each bin.

End-to-End training and calibration with Noisy labels. In the following experiment, we combined our noise-robust calibration method with the method for network training using training data with noisy labels described in the previous section [52]. Thus, we simulated a real-world scenario in which the labels of both the training set and the validation set had the same noise level. Making it even more realistic, we assumed here that the label noise matrix $P$ is unknown and was estimated during the training step [52]. We trained models for the three medical-imaging datasets (ChestXray14, HAM-10000, PathMNIST), using training data with label noise level $\epsilon \in \{ 0 \% , 5 \% , 1 0 \% , 2 0 \% \}$ In addition to training the network, we also estimated the noise matrix $P ,$ denoted as ${ \hat { P } } .$ We report the results of two variants of our calibration method NTS; namely, $\mathrm { N T S } ( P )$ where $P$ is known and $\mathrm { N T S } ( \hat { P } )$ where P is estimated during the training phase from the noisy training set. Tables 3.3 and 3.4 report the calibration results using adaECE and ECE respectively. The results show that the noisy labels corrupted the calibration of the network, whereas our method, even in the case where $P$ was estimated achieved a much better calibration result. We can also observe that the network training procedure is much more resilient to label noise compared to calibration. A noise level of 20% only results in a slight degradation of the network’s accuracy. However, the calibration process does not provide any assistance and instead significantly reduces the network’s calibration. The ChestX-ray14 task contains a dominant ”no-finding” class in addition to the 14 pathology-related classes and is heavily unbalanced. In that case, when the true noise matrix P was used we obtained good calibration results but when the estimated noise matrix was used, the calibration procedure failed. The reason for that is that in ChestX-ray14, the classes are unbalanced, and in that case there is currently no reliable way to estimate the noise matrix. When The No-finding class is removed (dataset ChestX-ray1-bal) the dataset is relatively balanced and the training algorithm manages to estimate the noise matrix. Finally, results satisfy the statistical significance requirement by a large margin, with an average p-value across the 4 diferent datasets and ϵ = 0.2 of $P - v a l u e = 0 . 0 0 5 0 8$

Calibration results for a general noise matrix. So far, we conducted experiments with uniform noise. Next, we evaluate our NTS method on a validation set that was corrupted by a general label noise matrix P, which allows all types of noise to afect the labels. We tested four cases that were diferent in terms of their label noise matrix P: (1) Symmetric noise where labels are converted to any other label with equal probability, (2) Neighbor noise where labels can only be flipped to adjacent labels, (3) Decreasing noise where labels are flipped to adjacent labels with decreasing probability and finally (4) Random noise - a general noise matrix whose of-diagonal probabilities were randomly selected. The same noise model was applied to the training and validation sets.

![](images/4554be7a88ea6b36bdf6c244c00be4c55efddc0e5f98cfc406bba8bd326e113c.jpg)

![](images/2c0b364c642f59c1767afddd310ed44a8895f02ea1fa949587ace2899c69fec3.jpg)  
(1) symmetric noise

![](images/1acbc39af54e1b07050427cf8b763618be0ef2565709ce597292a7f7e30e007c.jpg)

![](images/5c0bed6fbb96b60dfed95546b46077549a8a575a03f1409084e186e385b086d0.jpg)  
(2) neighbor noise

![](images/56c8bd274f1c548bc15cb44c9d145ad8385911459d9402800af5fea425fb92bf.jpg)

![](images/00d83f58689719342ad68da4f650be8ab829d0c6d9e670e71824d905682258c4.jpg)  
(3) decreasing noise

![](images/cdce30892661e8d0373dfa51c1ab1bdd8e99e3a2507df3316a0c6148894ac450.jpg)

![](images/94fc2d95b2f488c5891902dc15433cd2403f5993a13fb20431133f8c670a8996.jpg)  
(4) random noise

Figure 3.4: Calibration results for several noise transition matrices on ChestX-ray14-bal and ResNet-18. In each example we show the noise transition matrix (Left) and the adaECE measure on the test set for the compared calibration methods (right).  
![](images/0fce6b65b72cb2c4fd0c6e3dfefbad66a2272e44a582a69688f2345b1996cc43.jpg)

![](images/0180a1b98c9ac06cf78eedb6a1ea314d51d6e1feab60d3f0e16a55fe34853bab.jpg)  
Figure 3.5: Calibration performance, measured by adaECE (left), and the corresponding temperature (right) as a function of the noise ratio used by the NTS algorithm.

Figure 3.4 shows the noise transition matrices and the corresponding adaECE calibration results on the test set for the baseline calibration method $\mathrm { T S } _ { n o i s y }$ and NTS. We report the results of two variants of our calibration method NTS; namely, $\mathrm { N T S } ( P )$ where the noise matrix P is known and $\mathrm { N T S } ( \hat { P } )$ where P is estimated during the training phase from the noisy training set. We also report the results of $\mathrm { T S } _ { c l e a n }$ where calibration was applied to the clean validation set. In all cases, regardless of the noise matrix, NTS achieved calibration results that were on par with the results obtained by applying $\mathrm { T S } _ { c l e a n }$ on the clean validation set and much better than the results obtained by $\mathrm { T S } _ { n o i s y }$

Sensitivity to the noise estimation accuracy. Our NTS method is based on estimating the noise level, which is carried out as part of the network training. Crucially, the NTS method must not be highly sensitive to variations in the noise estimation, i.e., there is a range of values around the correct noise level in which NTS performs well. In the following experiment, we corrupted the validation set using noise level true-ϵ = 0.2. Figure 3.5 shows the adaECE and the temperature obtained by the NTS as a function of the actual ϵ used in the calibration process. We denote this method NTS(ϵ). We also report the results of NTS(est-ϵ) and $\mathrm { N T S } ( \mathrm { t r u e } { - } \epsilon )$ , which are variants of NTS using the estimated noise matrix and the exact noise matrix, respectively. The experiments were run on the test set of PathMNIST with the ResNet18 architecture. The results show that there is indeed a large range of ϵ values in which NTS performs much better than $\mathrm { T S } _ { n o i s y }$

# Conformal Prediction under Noisy Labels

In this chapter, we address the application of Conformal Prediction (CP) methods in classification networks under the challenge of noisy labels. Conformal Prediction is a powerful tool for safetycritical applications, such as medical imaging, where maintaining a predefined level of certainty in model predictions is essential. The ability to generate a set of possible class candidates, ensuring the true class is included with high confidence, provides valuable support for clinical practitioners by narrowing down the possible diagnoses while controlling the risk of mistakes. We introduce novel methods that enhance the robustness of CP under label noise, demonstrating their efectiveness on medical and natural image datasets.

## 4.1 Problem Statement

In machine learning for safety-critical applications, the model must only make predictions it is confident about. One way to achieve this is by returning a (hopefully small) set of possible class candidates that contain the true class with a predefined level of certainty. This is a natural approach for medical imaging, where safety is of the utmost importance and a human makes the final decision. This allows us to aid the practitioner, by reducing the number of possible diagnoses he needs to consider, with a controlled chance of mistake. The general approach to return a prediction set without any assumptions on the data distribution (besides i.i.d. samples) is called Conformal Prediction (CP) [1, 93]. It creates a prediction set with the guarantee that the probability of the correct class being within this set meets or exceeds a specified confidence threshold. The goal is to return the smallest set possible while maintaining the confidence level guarantees. Recently, with the growing use of neural network systems in safety-critical applications such as medical imaging, CP has become an important calibration tool [57, 58, 67]. We note that CP is a general framework rather than a specific algorithm. The most common approach builds the prediction set using a conformity score, and diferent algorithms mostly vary in terms of how the conformity score is defined.

When dealing with conformal predictions, a critical challenge arises in applications such as medical imaging due to label noise. In these domains, datasets frequently contain noisy labels stemming from ambiguous data that can confuse even clinical experts. Furthermore, physicians may disagree on the diagnosis for the same medical image, leading to inconsistencies in the ground truth labeling. Noisy labels also occur when applying diferential privacy techniques to overcome privacy issues [29]. While significant eforts have been devoted to the problem of noise-robust network training [88, 105], the challenge of calibrating the models has only recently begun to receive attention.

In this chapter, we tackle the challenge of applying CP to classification networks using a validation set with noisy labels. (author?) [21] suggested ignoring label noise and simply applying the standard CP algorithm on the noisy labeled validation set. This strategy results in large prediction sets especially when there are many classes. The most related studies to ours are [86, 13] which present a noisy CP algorithm using conservative coverage guarantee bounds which can result in large prediction sets, failing in classification tasks with many classes. In the following sections, we present two novel approaches and algorithms for CP on noisy data. The first approach propose a new score that is robust to label noise. The second approach propose to estimate the threshold in the presence of noisy labels, similar to (author?) [86] and [13], but yields an efective coverage guarantee even in tasks with a large number of classes in the uniform noise setting. We applied the algorithms to several standard medical and scenery imaging classification datasets and show that the latter method outperformed previous methods by a significant margin and achieved results comparable to those obtained by using a clean validation set. The greatest value and novelty of our approach lies in tasks with many classes, such as CIFAR-100, TinyImageNet, and ImageNet, where all other methods fail.

The study described in this chapter was published in [78] and [79].

## 4.2 Score That Is Robust to Label Noise

Let $( x _ { 1 } , \tilde { y } _ { 1 } ) , . . . , ( x _ { n } , \tilde { y } _ { n } )$ be a noisy validation set where the labels were corrupted by uniform noise with a noise level ϵ. We aim to find a noise-robust conformal score that can be applied to the noisy labeled data. Since $y _ { t }$ is not observed, we cannot directly compute the score $S ( x _ { t } , y _ { t } )$ . Instead, we can estimate it using its noisy version $\tilde { y } _ { t }$ :

$$
{ \bf E } ( S ( x _ { t } , y _ { t } ) | \tilde { y } _ { t } ) = \sum _ { i = 1 } ^ { k } p ( y _ { t } = i | \tilde { y } _ { t } ) S ( x _ { t } , i ) .\tag{4.1}
$$

Assuming a non-informative uniform prior on the correct label $y _ { t } , \mathrm { i . e . , } p ( y _ { t } = i ) = 1 / k )$ , we obtain:

$$
p ( y _ { t } = i | \tilde { y } _ { t } ) = ( 1 - \epsilon ) 1 _ { \{ \tilde { y } _ { t } = i \} } + \frac { \epsilon } { k } .\tag{4.2}
$$

Substituting (4.2) in (4.1), yields an estimate $\hat { S } ( x _ { t } , \tilde { y } _ { t } , \epsilon )$ of the noise-free score:

$$
\hat { S } ( x _ { t } , \tilde { y } _ { t } , \epsilon ) = \mathbf { E } ( S ( x _ { t } , y _ { t } ) | \tilde { y } _ { t } ) = ( 1 - \epsilon ) S ( x _ { t } , \tilde { y } _ { t } ) + \epsilon S ( x _ { t } )\tag{4.3}
$$

s.t. $\begin{array} { r } { S ( x _ { t } ) = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } S ( x _ { t } , i ) } \end{array}$ . Note that to obtain the score estimation $\hat { S } ( x _ { t } , \tilde { y } _ { t } , \epsilon )$ , we need to either know the noise level ϵ or estimate it from the noisy-label data. We elaborate further on this issue in the next section.

Table 4.1: Conformal Prediction methods for validation sets with noisy labels. Given an image $x ,$ $y$ is the true label, $\hat { y }$ is its noisy version and ϵ is the noise level. S is a conformal score and $\hat { S }$ is its noise robust variant.
<table><tr><td rowspan=1 colspan=1>Stage</td><td rowspan=1 colspan=1>CP (Oracle)</td><td rowspan=1 colspan=1>Noisy-CP [21]</td><td rowspan=1 colspan=1>NRESCP</td><td rowspan=1 colspan=1>NRSCP</td></tr><tr><td rowspan=1 colspan=1>Learning phaseInference phase</td><td rowspan=1 colspan=1> $S ( x , y ) \to q$  $\{ y | S ( x , y ) \leq q \}$ </td><td rowspan=1 colspan=1> $S ( x , \tilde { y } )  q _ { n o i s e }$  $\{ y | S ( x , y ) \le q _ { n o i s e } \}$ </td><td rowspan=1 colspan=1> $\hat { S } ( x , \tilde { y } , \epsilon ) \to q _ { \epsilon }$  $\{ y | \hat { S } ( x , y , \epsilon ) \le q _ { \epsilon } \}$ </td><td rowspan=1 colspan=1> $\hat { S } ( x , \tilde { y } , \epsilon ) \to q _ { \epsilon }$  $\{ y | S ( x , y ) \le q _ { \epsilon } \}$ </td></tr></table>

We next apply the CP algorithm on the estimated conformal scores and set $q _ { \epsilon }$ to be the $( 1 - \alpha )$ quantile of $\hat { S } ( x _ { 1 } , \tilde { y } _ { 1 } , \epsilon ) , . . . , \hat { S } ( x _ { n } , \tilde { y } _ { n } , \epsilon )$ . According to the general CP theory, the prediction set of a given test sample x is:

$$
\hat { C } _ { \epsilon } ( x ) = \{ y | \hat { S } ( x , y , \epsilon ) \leq q _ { \epsilon } \} = \{ y | S ( x , y ) \leq \frac { q _ { \epsilon } - \epsilon S ( x ) } { 1 - \epsilon } \} .\tag{4.4}
$$

Let $x$ be a test point and y and $\tilde { y }$ be its true label and noisy label respectively. The general $\mathrm { C P }$ theory guarantees that $1 - \alpha \le p ( \tilde { y } \in \hat { C } _ { \epsilon } ( x ) )$ . This guarantee, however, is for the noisy labeled data. Assume that for every x the order of the network class predictions $p ( y = i | x ; \theta )$ coincides with the order of the true probabilities $p ( y = i | x )$ . In that case, the same argument that appears in [21] for Noisy-CP, implies a coverage guarantee in the noise-free case. However, the prediction set obtained by the estimated score (4.4), is usually still too large.

In the case of noisy labels, during the CP learning phase, we need to estimate the score of the correct class. However, to form the prediction set at test time we need to compute the scores of all the possible classes. Hence, in a way similar to the Noisy-CP algorithm, we can thus construct the prediction set of a test sample x using the exact score $S ( x , y )$ instead of the estimated score $\hat { S } ( x , \tilde { y } , \epsilon )$ :

$$
C _ { \epsilon } ( x ) = \{ y \mid S ( x , y ) \leq q _ { \epsilon } \} .\tag{4.5}
$$

We denote the algorithm variants based on Eqs. (4.4) and (4.5) as the Noise Robust Estimated Score CP (NRESCP) and the Noise Robust Score CP (NRSCP) respectively. The only diference between them lies in how the test-time prediction set is formed. The various noisy label CP methods discussed above are summarized in Table 4.1. We empirically show below that NRSCP satisfies the coverage requirement and yields an average size that is much smaller than the one obtained by NRESCP. The NRSCP is summarized in Algorithm Box 3.

Algorithm 3 Noise-Robust Score Conformal Prediction (NRSCP)   
1: Input: A conformal score $S ( x , y )$ , a coverage level $1 - \alpha$ and a validation set $( x _ { 1 } , y _ { 1 } ) , . . . , ( x _ { n } , y _ { n } )$   
s.t. the labels are corrupted by uniform noise with parameter ϵ.   
2: Compute the estimated scores:   
$s _ { t } = \hat { S } ( x _ { t } , y _ { t } , \epsilon ) = ( 1 - \epsilon ) S ( x _ { t } , y _ { t } ) + \frac { \epsilon } { k } \sum _ { i = 1 } ^ { k } S ( x _ { t } , i ) , t = 1 , . . . , n$   
3: Set q to be the $\lceil ( n + 1 ) ( 1 - \alpha ) / n \rceil$ quantile of $s _ { 1 } , . . . , s _ { n }$   
4: The prediction set of a test sample x is $C ( x ) = \{ y \mid S ( x , y ) < q \}$

We next analyze the proposed noise robust conformal score. It is easy to verify that the prediction set obtained by NRSCP is smaller than the one obtained by NRESCP, i.e., $C _ { \epsilon } ( x ) \subset \hat { C } _ { \epsilon } ( x )$ if and only if $S ( x ) \le q _ { \epsilon }$ . In the case of HPS, $\begin{array} { r } { S ( x ) = \frac { 1 } { k } \sum _ { i } ( 1 - p ( y = i | x ) ) = ( k - 1 ) / k } \end{array}$ and therefore $\hat { S } ( x , \tilde { y } , \epsilon ) = ( 1 - \epsilon ) S ( x , \tilde { y } ) + \epsilon \frac { k - 1 } { k }$ . This implies that $\begin{array} { r } { q _ { \epsilon } = ( 1 - \epsilon ) q _ { n o i s e } + \epsilon \frac { k - 1 } { k } } \end{array}$ such that $q _ { n o i s e }$ and $q _ { \epsilon }$ are the thresholds computed by Noisy-CP [21] and NRESCP (4.4) respectively. It is easy to verify that Noisy-CP and NRESCP yield the same prediction set, i.e. $\{ y | S ( x , y ) \leq q _ { n o i s e } \} = \{ y | \hat { S } ( x , y , \epsilon ) \leq q _ { \epsilon } \}$ In the case of the APS conformal score, it is easy to see that: $( { \hat { p } } + 1 ) / 2 \leq S ( x ) \leq ( { \hat { p } } + k - 1 ) / k .$ s.t. $\hat { p } = \operatorname* { m a x } _ { i } p ( y = i | x ; \theta )$ is the network confidence on its single-class prediction. Hence, when the prediction sets obtained by NRSCP are smaller than those obtained by NRESCP, the prediction confidence satisfies $( \hat { p } + 1 ) / 2 \le S ( x ) \le q _ { \epsilon }$

## 4.3 Procedure of Threshold Estimation That Is Robust to Label Noise

In the previous section we presented a new conformal score that takes into account the noise rate making it robust to the label noise. Next, we tackle the problem of conformal prediction with noisy labels from a diferent angle by defining a conformal prediction procedure that is robust to noisy labels along with theoretic and coverage guarantees. Here we show how, given a simple noise model and a known noise level, we can get the correct CP threshold based on noisy data. We will generalize this beyond the simple noise model in the following section. Consider a network that classifies an input x into k pre-defined classes. Given a conformity score $S ( x , y )$ and a specified coverage $1 - \alpha .$ the goal of the conformal prediction algorithm is to find a minimal q such that $p ( y \in C _ { q } ( x ) ) \geq 1 - \alpha$ Let $( x _ { 1 } , \tilde { y } _ { 1 } ) , . . . , ( x _ { n } , \tilde { y } _ { n } )$ be a validation set with noisy labels and let $y _ { i }$ be the unknown correct label of $x _ { i }$ . Let $s _ { i } = S ( x _ { i } , \tilde { y } _ { i } )$ be the conformity score of $( x _ { i } , \tilde { y } _ { i } )$ . We assume that the label noise follows a uniform distribution, where with a probability of ϵ, the correct label is replaced by a label that is randomly sampled from the k classes:

$$
p ( \tilde { y } = j | y = i ) = \mathbb { 1 } _ { \{ i = j \} } ( 1 - \epsilon ) + \frac { \epsilon } { k } .\tag{4.6}
$$

Uniform noise is relevant, for example, when applying diferential privacy techniques to overcome privacy issues [29]. In that setup the noise level ϵ is usually known. In other applications such as medical imaging, where the noise parameter ϵ is not given, it can be estimated with suficient accuracy from the noisy-label data during training [117, 52, 54]. We can write y˜ as $\tilde { y } = ( 1 - z ) \cdot y + z \cdot u ,$ s.t. u is a random label uniformly sampled from $\{ 1 , . . . , k \}$ and z is a binary random variable $( p ( z = 1 ) = \epsilon )$ indicating whether the label of the sample $( x , y )$ was replaced by a random label or not. For each candidate threshold, q denote:

$$
F ^ { c } ( q ) = p ( y \in C _ { q } ( x ) ) , \qquad F ^ { n } ( q ) = p ( \tilde { y } \in C _ { q } ( x ) ) ,
$$

$$
F ^ { r } ( q ) = p ( u \in C _ { q } ( x ) ) ,
$$

where $F ^ { c } , F ^ { n }$ , and $F ^ { r }$ represent the clean, noisy and random labels. Note as well that each one is the CDF of the appropriate conformal score function, e.g., $F ^ { c } ( q ) = p ( y \in C _ { q } ( x ) ) = p ( S ( x , y ) \leq q )$

It is easily verified that

$$
\begin{array} { c } { { F ^ { n } ( q ) = p ( z = 0 ) F ^ { c } ( q ) + p ( z = 1 ) F ^ { r } ( q ) } } \\ { { { } } } \\ { { { } = ( 1 - \epsilon ) F ^ { c } ( q ) + \epsilon F ^ { r } ( q ) . } } \end{array}\tag{4.7}
$$

For each value q, we can estimate $F ^ { n } ( q )$ from the noisy validation set:

$$
\hat { F } ^ { n } ( q ) = \frac { 1 } { n } \sum _ { i } \mathbb { 1 } _ { \{ \tilde { y } _ { i } \in C _ { q } ( x _ { i } ) \} } = \frac { 1 } { n } \sum _ { i } \mathbb { 1 } _ { \{ s _ { i } \leq q \} } .\tag{4.8}
$$

Note that $q$ is the $\hat { F } ^ { n } ( q )$ -quantile of $s _ { 1 } , . . . , s _ { n }$ . Similarly we can also estimate $F ^ { r } ( q )$

$$
{ \hat { F } } ^ { r } ( q ) = { \frac { 1 } { n } } \sum _ { i } p ( u _ { i } \in C _ { q } ( x _ { i } ) ) = { \frac { 1 } { n } } \sum _ { i } { \frac { | C _ { q } ( x _ { i } ) | } { k } } ,\tag{4.9}
$$

s.t. $u _ { i }$ is uniformly sampled from $\{ 1 , . . . , k \}$

By substituting (4.8) and (4.9) in (4.7) we obtain an estimation of $F ^ { c } ( q ) = p ( y \in C _ { q } ( x ) )$ based on the noisy validation set and the noise level ϵ:

$$
\hat { F } ^ { c } ( q ) = \frac { \hat { F } ^ { n } ( q ) - \epsilon \hat { F } ^ { r } ( q ) } { 1 - \epsilon } .\tag{4.10}
$$

For each candidate q we first compute ${ \hat { F } } ^ { n } ( q )$ and $\hat { F } ^ { r } ( q )$ and then by using (4.10) obtain the coverage estimation ${ \hat { F } } ^ { c } ( q )$ . Given a coverage requirement $( 1 - \alpha )$ , we can thus use the noisy validation set to find a threshold q such that $\hat { F } ^ { c } ( q ) = 1 - \alpha$ . Note that since $F ^ { c } ( q )$ is monotonous, it seems reasonable to search for the threshold $q$ using the bisection method. However, as ${ \hat { F } } ^ { c } ( q )$ is an approximation based on the diference between two monotonic functions, it might not be exactly monotonous. We therefore find the threshold $q$ using an exhaustive grid search. If there are several solutions we select the largest value. (In practice selecting one of the solutions has almost no efect on the results.) We note that even with an exhaustive search the entire runtime is negligible compared to the training time.

We can narrow the threshold search domain as follows:

Lemma 4.3.1. For every threshold $q$ we have: $\hat { F } _ { q } ^ { n } / k \le \hat { F } ^ { r } ( q )$

Proof. Denote $A = \{ i | \hat { y } _ { i } \in C _ { q } ( x _ { i } ) \}$ and $B = \{ i | 1 \leq | C _ { q } ( x _ { i } ) | \}$ . Note that $\hat { F } ^ { n } ( q ) = | A | / n$

$$
| B | = \sum _ { i \in B } 1 \leq \sum _ { i \in B } | C _ { q } ( x _ { i } ) | \leq \sum _ { i = 1 } ^ { n } | C _ { q } ( x _ { i } ) | = n k \hat { F } ^ { r } ( q ) .
$$

Finally $A \subset B$ implies that: $\hat { F } ^ { n } ( q ) = | A | / n \leq | B | / n \leq k \hat { F } ^ { r } ( q )$

Theorem 4.3.2. Let $q _ { 1 }$ be the $\begin{array} { r } { ( 1 - \alpha ) ( 1 - \epsilon ) / ( 1 - \frac { \epsilon } { k } ) } \end{array}$ quantile of $s _ { 1 } , . . . , s _ { n }$ and let $q _ { 2 }$ be the $( 1 - \alpha ) + \alpha \epsilon$ quantile. ${ \mathit { I f q } }$ satisfies $\hat { F } ^ { c } ( q ) = 1 - \alpha$ then $q _ { 1 } \leq q \leq q _ { 2 }$

Proof. Assume q satisfies $\hat { F } ^ { c } ( q ) = 1 - \alpha$ . Eq. (4.10) implies that

$$
1 - \alpha = \hat { F } ^ { c } ( q ) = \frac { \hat { F } ^ { n } ( q ) - \epsilon \hat { F } ^ { r } ( q ) } { 1 - \epsilon }\tag{4.11}
$$

$$
\Rightarrow \hat { F } ^ { n } ( q ) = ( 1 - \alpha ) ( 1 - \epsilon ) + \epsilon \hat { F } ^ { r } ( q ) .
$$

Since $0 \leq \hat { F } ^ { r } ( q ) \leq 1$ we get that:

$$
( 1 - \alpha ) ( 1 - \epsilon ) \le \hat { F } ^ { n } ( q ) \le ( 1 - \alpha ) + \alpha \epsilon = \hat { F } ^ { n } ( q _ { 2 } ) .\tag{4.12}
$$

For every q we have $\hat { F } ^ { n } ( q ) / k \le \hat { F } ^ { r } ( q )$ (Lemma 3.1). Hence, $( 1 - \alpha ) ( 1 - \epsilon ) \leq \hat { F } ^ { n } ( q )$ (4.12) implies that $( 1 - \alpha ) ( 1 - \epsilon ) / k \leq \hat { F } ^ { r } ( q )$ . Combining this inequality with Eq. (4.11) yields a better lower bound: $( 1 - \alpha ) ( 1 - \epsilon ) ( 1 + \epsilon / k ) \leq \hat { F } ^ { n } ( q )$ . Iterating this process yields:

$$
( 1 - \alpha ) ( 1 - \epsilon ) \left( 1 + \frac { \epsilon } { k } + \left( \frac { \epsilon } { k } \right) ^ { 2 } + \ldots \right)
$$

$$
= ( 1 - \alpha ) \frac { 1 - \epsilon } { 1 - \frac { \epsilon } { k } } = \hat { F } ^ { n } ( q _ { 1 } ) \leq \hat { F } ^ { n } ( q ) .
$$

Finaly, $\hat { F } ^ { n } ( q )$ is a monotonically increasing function of q which implies that $q _ { 1 } \leq q \leq q _ { 2 }$

As an alternative to the grid search we can sort the noisy conformity scores $s _ { i } = S ( x _ { i } , \tilde { y } _ { i } )$ and look for the minimal i such that ${ \hat { F } } ^ { c } ( s _ { i } ) \geq 1 - \alpha$ . In the noise-free case $\hat { F } ^ { c }$ is piece-wise constant, with jumps determined exactly by the order statistics $s _ { i } ,$ namely, ${ \hat { F } } ^ { c } ( s _ { i } ) = i / n$ and thus this algorithm coincides with the standard CP algorithm. In the noisy case ${ \hat { F } } ^ { c } ( q )$ depends on the conformity scores of all the k classes and thus its structure is more complicated. We dub our algorithm Noise-Aware Conformal Prediction (NACP), and summarize it in Algorithm Box 4. Note that in the noise-free case $( \epsilon = 0 )$ the NACP algorithm coincides with the standard CP algorithm and selects q that satisfies $\hat { F } ^ { c } ( q ) = \hat { F } ^ { n } ( q ) = 1 - \alpha$ , i.e., q is the 1 − α quantile of the validation set conformity scores.

## 4.3.1 Prediction Size Comparison

We next compare our NACP approach analytically to Noisy-CP [21] in terms of the average size of the prediction set.

Theorem 4.3.3. Let q and q˜ be the thresholds computed by the NACP and the Noisy-CP algorithms respectively. Then $q \leq \tilde { q }$ if and only $i f \hat { F } ^ { r } ( \tilde { q } ) \leq ( 1 - \alpha )$

Proof. The threshold $\tilde { q }$ computed by the Noisy-CP algorithm (by applying standard CP on the noisy validations set) satisfies $\hat { F } ^ { n } ( \tilde { q } ) = ( 1 - \alpha )$ . The true threshold q satisfies $\hat { F } ^ { n } ( q ) = ( 1 - \alpha ) ( 1 - \epsilon ) + \epsilon \hat { F } ^ { r } ( q )$ (4.11). Looking at the diference

$$
\hat { F } ^ { n } ( \tilde { q } ) - \hat { F } ^ { n } ( q ) = 1 - \alpha - ( 1 - \alpha ) ( 1 - \epsilon ) - \epsilon \hat { F } ^ { r } ( q )\tag{4.13}
$$

$$
= \epsilon ( 1 - \alpha - \hat { F } ^ { r } ( q ) ) .
$$

Hence from the monotonicity of ${ \hat { F } } ^ { n } ( q )$ we have $q \leq \tilde { q }$ if $\hat { F } ^ { n } ( q ) \leq \hat { F } ^ { n } ( \tilde { q } )$ if $\hat { F } ^ { r } ( q ) \leq 1 - \alpha$

The theorem above states that if the size of the prediction set obtained by NACP is less than $k ( 1 - \alpha )$ , NACP is more efective than Noisy-CP. For example, assume $k = 1 0 0$ and $1 - \alpha = 0 . 9$ . In this case, if the average size of the NACP prediction set is less than 90, NACP is more efective than Noisy-CP. We also see from eq. (4.13) that the smaller ${ \hat { F } } ^ { r }$ is the larger the gap between the two methods. Since ${ \hat { F } } ^ { r }$ is inversely proportional to the number of classes, we expect the diference to be substantial when there is a large number of classes to consider, which is exactly where CPs’ ability to reliably exclude possible classes is very useful. In our experiments, we indeed found a considerable gap between the two methods when we experimented on classification tasks with a large number of classes.

Algorithm 4 Noise-Aware Conformal Prediction (NACP) for uniform noise   
1: Input: A conformity score $S ( x , y )$ , a coverage level $_ { 1 - \alpha }$ and a validation set $( x _ { 1 } , \tilde { y } _ { 1 } ) , . . . , ( x _ { n } , \tilde { y } _ { n } )$   
s.t. the labels are corrupted by a uniform noise with parameter ϵ.   
2: Set $q _ { 1 }$ to be the $\begin{array} { r } { ( 1 - \alpha ) ( 1 - \epsilon ) / ( 1 - \frac { \epsilon } { k } ) } \end{array}$ quantile of $S ( x _ { 1 } , \tilde { y } _ { 1 } ) , . . . , S ( x _ { n } , \tilde { y } _ { n } )$ and set $q _ { 2 }$ to be   
$( ( 1 - \alpha ) + \alpha \epsilon )$ quantile.   
3: For each candidate threshold $q$ compute:   
$\hat { F } ^ { n } ( q ) = \frac { 1 } { n } \sum _ { i } \mathbb { 1 } _ { \{ \tilde { y } _ { i } \in C _ { q } ( x _ { i } ) \} } ,$   
$\hat { F } ^ { r } ( q ) = \frac { 1 } { n } \sum _ { i } \frac { | C _ { q } ( x _ { i } ) | } { k } ,$   
$\hat { F } ^ { c } ( q ) = \frac { \hat { F } ^ { n } ( q ) - \epsilon \hat { F } ^ { r } ( q ) } { 1 - \epsilon }$   
4: Apply a grid search to find $q \in [ q _ { 1 } , q _ { 2 } ]$ that satisfies $\hat { F } ^ { c } ( q ) = { 1 - \alpha }$   
5: The prediction set of a test sample x is:   
$C _ { q } ( x ) = \{ y \mid S ( x , y ) < q \} .$   
6: Coverage guarantee: $p ( y \in C _ { q } ( x ) ) \geq 1 - \alpha - \Delta ( n , \epsilon , \delta )$ with probability $( 1 - \delta )$ over the noisy   
validation set sampling (see Theorem 4.3.5).

## 4.3.2 Coverage Guarantees

We next provide a coverage guarantee for NACP. We show that if we apply the NACP to find a threshold $q$ for $1 - \alpha + \Delta$ , then $P ( y \in C _ { q } ( x ) ) \geq 1 - \alpha$ were $\Delta$ depends on the validation set size. $\Delta$ is a finite-sample term that is needed to approximate the CDF to set the threshold instead of simply picking a predefined quantile. Because $\Delta$ can be computed, one can adjust the α used in the NACP algorithm to get the desired coverage guarantee. However, we note that we empirically found this bound to be over-conservative, and that the un-adjusted method does reach the desired coverage.

Lemma 4.3.4. Given $\delta > 0$ , define $\Delta = \sqrt { \frac { \log ( 4 / \delta ) } { 2 n h ^ { 2 } } }$ such that $\textstyle h = { \frac { 1 - \epsilon } { 1 + \epsilon } }$ and n is the size of the noisy validation set. Then

$$
\begin{array} { r } { p ( \underset { q } { \operatorname* { s u p } } | F ^ { c } ( q ) - \hat { F } ^ { c } ( q ) | > \Delta ) \leq \delta , } \end{array}\tag{4.14}
$$

such that the probability is over the validation set.

Proof. The Dvoretzky–Kiefer–Wolfowitz (DKW) inequality [60] states that if we estimate a CDF $F$ from n samples using the empirical CDF $F _ { n }$ then $\begin{array} { r } { p ( \operatorname* { s u p } _ { x } | F _ { n } ( x ) - F ( x ) | > \Delta ) \le 2 \exp ( - 2 n \Delta ^ { 2 } ) } \end{array}$ . Eq. (4.10) defines ${ \hat { F } } ^ { c } ( q )$ using $\hat { F } ^ { n } ( q )$ and $\hat { F } ^ { r } ( q )$ . Both are empirical CDF, so from the DKW theorem and the union bound we get that:

$$
\begin{array} { r l } & { p ( \underset { q } { \operatorname* { s u p } } | F ^ { r } ( q ) - \hat { F } ^ { r } ( q ) | > h \Delta \mathrm { ~ o r } } \\ & { \quad \underset { q } { \operatorname* { s u p } } | F ^ { n } ( q ) - \hat { F } ^ { n } ( q ) | > h \Delta ) \leq 4 \exp ( - 2 n h ^ { 2 } \Delta ^ { 2 } ) = \delta . } \end{array}\tag{4.15}
$$

```powershell
Algorithm 5 Noise-Aware Conformal Prediction (NACP) for a noise matrix model
1: Input: A conformity score $S ( x , y )$ , a coverage level $_ { 1 - \alpha }$ and a validation set $( x _ { 1 } , \tilde { y } _ { 1 } ) , . . . , ( x _ { n } , \tilde { y } _ { n } )$
s.t. the labels are corrupted by a noise matrix $P .$
2: For each candidate threshold q compute:
$\hat { M } _ { q } ( \ell , i ) = \frac { 1 } { n } \sum _ { j } \mathbb { 1 } _ { \{ \tilde { y } _ { j } = i , \ell \in C _ { q } ( x _ { j } ) \} } , ~ i , \ell = 1 , . . , k .$
$\hat { F } ^ { c } ( q ) = \mathrm { T r } ( \hat { M } _ { q } P ^ { - 1 } ) .$
3: Apply a grid search to find q that satisfies $\hat { F } ^ { c } ( q ) = 1 - \alpha .$
4: The prediction set of a test sample x is:
$C _ { q } ( x ) = \{ y \mid S ( x , y ) < q \} .$
```

Using eq. (4.10) we get that with probability at least $1 - \delta$ for every q:

$$
\begin{array} { l } { \displaystyle \hat { F } ^ { c } ( q ) = \frac { \hat { F } ^ { n } ( q ) - \epsilon \hat { F } ^ { r } ( q ) } { 1 - \epsilon } \leq } \\ { \displaystyle \frac { ( F ^ { n } ( q ) + h \Delta ) - \epsilon ( F ^ { r } ( q ) - h \Delta ) } { 1 - \epsilon } F ^ { c } ( q ) + \frac { h \Delta + \epsilon h \Delta } { 1 - \epsilon } } \\ { = F ^ { c } ( q ) + h \Delta \frac { 1 + \epsilon } { 1 - \epsilon } = F ^ { c } ( q ) + \Delta . } \end{array}\tag{4.16}
$$

Similarly, we can show that $\hat { F } ^ { c } ( q ) \ge F ^ { c } ( q ) - \Delta$ which completes the proof.

The proof of the main theorem now follows the standard CP proof, taking the inaccuracy in estimating $F ^ { c } ( q )$ into account.

Theorem 4.3.5. Assume you have a noisy validation set of size n with noise level ϵ and set $\begin{array} { r } { \Delta ( n , \epsilon , \delta ) = \sqrt { \frac { \log ( 4 / \delta ) } { 2 n h ^ { 2 } } } } \end{array}$ s.t. $\textstyle h = { \frac { 1 - \epsilon } { 1 + \epsilon } }$ and that you pick q such that $\hat { F } ^ { c } ( q ) = 1 - \alpha + \Delta$ . Then with probability at least $1 - \delta$ (over the validation set), we have that $i f \left( x , y \right)$ are sampled from the clear label distribution we get:

$$
1 - \alpha \leq p ( y \in C _ { q } ( x ) ) \leq 1 - \alpha + 2 \Delta .
$$

Proof. Given a clean test pair $( x , y )$ , with probability δ over the validation set, we have:

$$
\begin{array} { l } { { p ( y \in C _ { q } ( x ) ) = p ( S ( x , y ) < q ) } } \\ { { \ } } \\ { { = F ^ { c } ( q ) \geq \hat { F } ^ { c } ( q ) - \Delta = 1 - \alpha . } } \end{array}
$$

In a similar way: $p ( y \in C _ { q } ( x ) ) = F ^ { c } ( q ) \leq \hat { F } ^ { c } ( q ) + \Delta = 1 - \alpha + 2 \Delta .$

As the size of the noisy validation set, $n ,$ tends to infinity, $\Delta$ converges to zero and thus the noisy threshold converges to the noise-free threshold.

## 4.3.3 A More General Noise Model

Next, we will extend our approach to a more general noise model. We will assume that the noisy label $\tilde { y }$ is independent of x given $y .$ We also assume that the noise matrix $P ( i , j ) = p ( \tilde { y } = j | y = i )$ is known and that the matrix P is invertible. For each q define the following matrices for the clear and the noisy data: $M _ { q } ^ { c } ( \ell , i ) = p ( \ell \in C _ { q } ( x ) , y = i )$ and $M _ { q } ( \ell , i ) = p ( \ell \in C _ { q } ( x ) , \tilde { y } = i )$ . Assuming that, given the true label y, the r.v. x and y˜ are independent, we obtain:

$$
\begin{array} { l } { M _ { q } ( \ell , i ) = p ( \ell \in C _ { q } ( x ) , \tilde { y } = i ) } \\ { \displaystyle = \sum _ { j } p ( \ell \in C _ { q } ( x ) , \tilde { y } = i , y = j ) } \\ { \displaystyle = \sum _ { j } p ( \ell \in C _ { q } ( x ) , y = j ) p ( \tilde { y } = i | y = j ) } \\ { \displaystyle = \sum _ { j } M _ { q } ^ { c } ( \ell , j ) P ( j , i ) . } \end{array}\tag{4.17}
$$

We can write (4.17) in matrix notation: $M _ { q } = M _ { q } ^ { c } P$ . Eq. (4.17) implies that:

$$
F ^ { c } ( q ) = p ( y \in C _ { q } ( x ) ) = p ( y \in C _ { q } ( x ) )\tag{4.18}
$$

$$
= \sum _ { i } p ( i \in C _ { q } ( x ) , y = i ) = \sum _ { i } M _ { q } ^ { c } ( i , i ) = \operatorname { T r } ( M _ { q } P ^ { - 1 } ) .
$$

We can estimate matrix $M _ { q }$ from the noisy samples:

$$
\hat { M } _ { q } ( \ell , i ) = \frac { 1 } { n } \sum _ { j } \mathbb { 1 } _ { \{ \tilde { y } _ { j } = i , \ell \in C _ { q } ( x _ { j } ) \} } , \qquad i , \ell = 1 , . . , k .\tag{4.19}
$$

Substituting (4.19) in (4.18) yields an estimation of the probability $F ^ { c } ( q ) = p ( y \in C _ { q } ( x ) )$ :

$$
\hat { F } ^ { c } ( q ) = \mathrm { T r } ( \hat { M } _ { q } P ^ { - 1 } ) .\tag{4.20}
$$

The final step is applying a grid search to find a threshold $q$ such that $\hat { F } ^ { c } ( q ) = 1 - \alpha$

In the case that $P$ is a uniform noise matrix (4.6), the Sherman-Morison formula implies that $\begin{array} { r } { P ^ { - 1 } = ( \frac { 1 } { 1 - \epsilon } I - \frac { \epsilon } { ( 1 - \epsilon ) k } \mathbf { 1 } ^ { \top } ) } \end{array}$ . Therefore,

$$
\hat { F } ^ { c } ( q ) = \mathrm { T r } ( \hat { M } _ { q } P ^ { - 1 } ) = \frac { 1 } { 1 - \epsilon } \sum _ { i } \hat { M } _ { q } ( i , i )
$$

$$
- \frac { \epsilon } { ( 1 - \epsilon ) k } \sum _ { \ell , i } \hat { M } _ { q } ( \ell , i ) = \frac { \hat { F } ^ { n } ( q ) - \epsilon \hat { F } ^ { r } ( q ) } { 1 - \epsilon } .
$$

Thus in the case of a uniform noise the coverage estimation (4.20) coincides with (4.10). If the noise

matrix is unknown, it can be estimated from the noisy-label data during training [117, 52, 54]. The NACP method for a noise matrix model is summarized in Algorithm box 5.

We can extend the finite sample term ∆ that was developed for a uniform noise to obtain a theoretical coverage guarantee for a noise matrix model (4.3.6).

Theorem 4.3.6. Let P be a general noise matrix. Given $\delta > 0 ,$ define $\begin{array} { r } { \Delta = \| P ^ { - 1 } \| _ { \infty } k \sqrt { \frac { \log ( 2 k ^ { 2 } / \delta ) } { 2 n } } } \end{array}$ where , k is the number of classes and n is the size of the noisy validation set. Then

$$
p ( \operatorname* { s u p } _ { q } | \hat { F } ^ { c } ( q ) - F ^ { c } ( q ) | > \Delta ) < \delta .
$$

Proof. From $\operatorname { E q } .$ . (4.20) we have $| \hat { F } ^ { c } ( q ) - F ^ { c } ( q ) | = | \operatorname { T r } ( P ^ { - 1 } \hat { M } _ { q } ) - \operatorname { T r } ( P ^ { - 1 } M _ { q } ) | = | \operatorname { T r } ( P ^ { - 1 } \Delta \hat { M } _ { q } ) |$ where $\Delta M _ { q } = \hat { M } _ { q } - M _ { q }$ . We first note that $M _ { q } [ i , j ] = p ( j \in C _ { q } ( x ) , \tilde { y } = i )$ is not a CDF but we can define one that agrees with it for $q \in ( - \infty , C )$ which is the range of interest where C is a constant that bound the score function $S ( x , y )$ from above. We define $\tilde { S } _ { i j } ( x , \tilde { y } ) = \left\{ \begin{array} { l l } { S ( x , j ) , } & { { \mathrm { i f ~ } } \tilde { y } = i } \\ { C , } & { { \mathrm { i f ~ } } \tilde { y } \neq i } \end{array} \right.$ so $M _ { q } [ i , j ] = p ( \tilde { S } _ { i j } ( x , \tilde { y } ) \leq q )$ for $q \in ( - \infty , C )$ . Now from the DKW theorem, we know that if we estimate a CDF using n samples then with probability at least $1 - \delta$ we get a uniform bound on the error of size $\sqrt { \frac { \log ( 2 / \delta ) } { 2 n } }$ . As we are estimating $k ^ { 2 }$ matrix elements we can use the union bound to get that with probability $1 - \delta$ the $\begin{array} { r } { \forall i , j , q \in ( - \infty , C ) : | \Delta M _ { q } | \leq \sqrt { \frac { \log ( 2 k ^ { 2 } / \delta ) } { 2 n } } } \end{array}$ . Now if we look at the infinity norm of $P ^ { - 1 }$ , then $\begin{array} { r } { | ( P ^ { - 1 } \Delta M _ { q } ) _ { i , j } | \leq | | P ^ { - 1 } | | _ { \infty } \sqrt { \frac { \log ( 2 k ^ { 2 } / \delta ) } { 2 n } } } \end{array}$ . As the trace is the sum of k such matrix entries, the total bound is $\kappa _ { \infty } k \sqrt { \frac { \log ( 2 k ^ { 2 } / \delta ) } { 2 n } }$ for $q \in ( - \infty , C )$ . Since we know $F ^ { c } ( q ) = 1$ for $q \geq C$ , we can set $\hat { F } ^ { c } ( q ) = 1$ for $q \geq C$ and get a bound for all $q \in \mathbb { R }$ □

However, this approach yields large prediction sets especially in tasks with many classes and thus is inefective. In the experiment section we show that in practice, even without adding finite sample terms, we obtain the required coverage probability.

## 4.4 Related Work

In this section, we review two closely related works that address the same problem of calibration with noisy labels [86, 13]. The derivation of the noisy conformal threshold in these two works is similar to ours. These two methods compute the same threshold q that satisfies ${ \hat { f } } ^ { c } ( q ) = 1 - \alpha$ (4.20). The only minor diference is that in these two studies they use the distribution of correct labels given the noisy labels, while we use the more natural distribution of the noisy labels given the correct label. As a result, they need to know the marginal class frequencies for both the clean and noisy labels, whereas we do not. Each one of the two methods provides a diferent finite coverage

guarantee in the form of:

$$
p ( y \in C _ { q } ( x ) ) \geq 1 - \alpha - \Delta
$$

where $\Delta$ depends on the validation set size $n ,$ the number of classes $k ,$ and the noise model, but it doesn’t depend on the validation dataset itself.

We first review the bound $\Delta$ derived in [86]. Let $\rho _ { i } = p ( y = i )$ and $\tilde { \rho } _ { i } = p ( \tilde { y } = i )$ be the marginal true and noisy label distributions. Let $M ( y | \tilde { y } )$ be the noise conditional distribution and let $V = M ^ { - 1 }$ Let $\begin{array} { r } { c ( n ) \ = \ \mathbb { E } \left[ \operatorname* { m a x } _ { i \in [ n ] } \left( \frac { i } { n } - u _ { ( i ) } \right) \right] } \end{array}$ , such that $\{ u _ { ( i ) } \} _ { i = 1 } ^ { n }$ order statistics of $\{ u _ { i } \} _ { i = 1 } ^ { n }$ i.i.d. uniform random variables on [0, 1]. The size of the least common class is $\begin{array} { r } { n _ { * } = \operatorname* { m i n } _ { i \in [ k ] } n _ { i } } \end{array}$ s.t. $n _ { i }$ is the number of samples of noisy label i. Finally, the finite sample correction is:

$$
\begin{array} { r } { \Delta = c ( n ) + \frac { 2 \operatorname* { m a x } _ { i \in [ k ] } \sum _ { l \neq i } | V _ { i l } | + \sum _ { i = 1 } ^ { k } | \rho _ { i } - \tilde { \rho } _ { i } | } { \sqrt { n _ { * } } } } \\ { \cdot \operatorname* { m i n } \left( k ^ { 2 } \sqrt { \frac { \pi } { 2 } } , \frac { 1 } { \sqrt { n ^ { * } } } + \sqrt { \frac { \log ( 2 k ^ { 2 } ) + \log ( n ^ { * } ) } { 2 } } \right) . } \end{array}\tag{4.21}
$$

It can be easily verified that $\Delta = O ( \log k )$ and therefore the bound becomes less efective for large values of k.

(author?) [86] also suggested a boosted version under additional assumptions. This version takes a hybrid approach that adaptively chooses between its algorithm and Noisy-CP depending on which approach leads to a lower (less conservative) calibrated threshold.

The finite sample term $\Delta$ derived in [13] is:

$$
\Delta = \sum _ { i = 1 } ^ { k } ( | w _ { i } ^ { ( 1 ) } | b ( n , i ) + \sum _ { i \neq j } | w _ { i j } ^ { ( 2 ) } | b ( n , j ) )\tag{4.22}
$$

s.t. $k$ is the number of classes, $w _ { i } ^ { ( 1 ) } = P _ { i , i } ^ { - 1 } \rho _ { i } - \tilde { \rho } _ { i } , w _ { i j } ^ { ( 2 ) } = \rho _ { i } P _ { j i } ^ { - 1 }$ , and $\begin{array} { r } { b ( n , j ) = ( 1 - \tilde { \rho } _ { j } ) ^ { n } + \sqrt { \frac { \pi } { n \tilde { \rho } _ { j } } } . } \end{array}$ $\rho _ { i } = p ( y = i )$ and $\tilde { \rho } _ { i } = p ( \tilde { y } = i )$ are the marginal clean and contaminated label probabilities and $P _ { j i } = p ( y = j | \tilde { y } = i )$ is the conditional label noise distribution. It can be easily verified that $\Delta = O ( { \sqrt { k } } )$ and therefore the bound becomes less efective for large values of k.

We note that our finite sample term $\Delta$ (see Lemma 4.3.4) does not depend on the number of classes k. Therefore, unlike the algorithms of (author?) [86] and (author?) [13], it remains efective even in tasks with many classes. A further distinction between us and the previous works is that their finite sample coverage guarantee is established for the average of all the noisy validation sets. In contrast, our approach provides an individual coverage guarantee for nearly all $( 1 - \delta )$ of the sampled noisy validation sets. In Section 4.5 we show that the average coverage guarantee obtained by (author?) [86] and (author?) [13] implies that in tasks with a large number of classes, the prediction set should include all the classes and therefore it is useless. In contrast, our individual finite set coverage guarantee, on to (1 − δ) portion of the noisy validation sets, remains efective for tasks with many classes.

We note that our observation that the finite sample correction term does not depend on the number of classes, applies to the case of a uniform label noise. In the case of a general noise matrix, all finite sample correction terms are not efective.

## 4.5 Experiments

In this section, we evaluate the capabilities of our NRSCP and NACP algorithms on various medical and scenery imaging datasets.

Compared methods. Our method takes an existing conformity score S and computes a threshold q that takes into account the label noise level. We implemented three popular conformal prediction scores, namely APS [82], RAPS [2] and HPS [93]. For each score, we implemented the following CP methods: (1) CP (oracle) - using a validation set with clean labels, (2) Noisy-CP - applying a standard CP on noisy labels without any modifications [21], (3) NR-CP (w/o ∆) - Noise-Robust CP approach without the finite sample coverage guarantee ∆, see Eq. (4.20) and [86, 13]. We also implemented three methods that add finite sample coverage guarantee terms to the NR-CP method. (4) Adaptive Conformal Classification with Noisy labels (ACNL) [86], (5) Boosted Adaptive Conformal Classification with Noisy labels (ACNL<sup>+</sup>) [86] (6) Contamination Robust Conformal Prediction (CRCP) [13], (7) NRSCP - our first approach (4.2) and (8) NACP - our second approach (4.3). For methods (4), (5), and (6), we used their oficial codes <sup>1</sup> <sup>2</sup> and we share our code for reproducibility<sup>3</sup>.

Evaluation Measures. We evaluated each CP method based on the average size of the prediction sets (where a small value means high eficiency) and the fraction of test samples for which the prediction sets contained the ground-truth labels. The two evaluation metrics are formally defined as:

$$
{ \mathrm { s i z e } } = { \frac { 1 } { n } } \sum _ { i } | C ( x _ { i } ) | , { \mathrm { c o v e r a g e } } = { \frac { 1 } { n } } \sum _ { i } \mathbf { 1 } ( y _ { i } \in C ( x _ { i } ) )
$$

such that n is the size of the test set. We report the mean and standard deviation over 1000 random splits.

Datasets. We show results on four standard scenery image datasets CIFAR-10, CIFAR-100 [48], Tiny-ImageNet, and ImageNet [15].

Table 4.2: CP calibration results for $1 - \alpha = 0 . 9$ and noise level ϵ = 0.2. We report the mean and the std over 1000 diferent splits. We show the best result with theoretical guarantees in bold.
<table><tr><td colspan="2">CP Method</td><td colspan="2">APS</td><td colspan="2">RAPS</td><td colspan="2">HPS</td></tr><tr><td colspan="2">Dataset</td><td>size ↓</td><td>coverage(%)</td><td>size ↓</td><td>coverage(%)</td><td>size ↓</td><td>coverage(%)</td></tr><tr><td rowspan="8">CIFAR-10 (10 classes)</td><td>CP (oracle) Noisy-CP</td><td>1.1 ± 0.01</td><td>90.0 ± 0.62</td><td>1.1 ± 0.01</td><td>90.0 ± 0.61</td><td>0.9 ± 0.01</td><td>90.0 ± 0.59</td></tr><tr><td></td><td>5.1 ± 0.18</td><td>99.9 ± 0.04</td><td>5.1 ± 0.18</td><td>99.9 ± 0.04</td><td> $5 . 1 \pm 0 . 1 8$ </td><td> $9 9 . 8 \pm 0 . 0 4$ </td></tr><tr><td>NR-CP (w/o ∆)</td><td>1.1 ± 0.02</td><td> $9 0 . 1 \pm 0 . 7 0$ </td><td>1.1 ± 0.02</td><td>90.1 ± 0.69</td><td> $0 . 9 \pm 0 . 0 2$ </td><td> $9 0 . 0 \pm 0 . 7 5$ </td></tr><tr><td>ACNL</td><td> $1 . 5 \pm 0 . 0 6$ </td><td> $9 6 . 0 \pm 0 . 6 1$ </td><td>1.3 ± 0.03</td><td> $9 4 . 6 \pm 0 . 6 5$ </td><td> $1 . 1 \pm 0 . 0 3$ </td><td> $9 6 . 0 \pm 0 . 5 9$ </td></tr><tr><td>ACNL+</td><td> $1 . 4 \pm 0 . 0 6$ </td><td> $9 5 . 8 \pm 0 . 5 9$ </td><td> $1 . 3 \pm 0 . 0 2$ </td><td> $9 4 . 8 \pm 0 . 6 1$ </td><td> $1 . 1 \pm 0 . 0 3$ </td><td> $9 6 . 1 \pm 0 . 5 7$ </td></tr><tr><td>CRCP</td><td> ${ \bf 1 . 2 \pm 0 . 0 3 }$ </td><td> ${ \bf 9 3 . 7 \pm 0 . 6 2 }$ </td><td> ${ \bf 1 . 2 \pm 0 . 0 3 }$ </td><td> ${ \bf 9 3 . 7 \pm 0 . 6 2 }$ </td><td> $\mathbf { 1 . 1 \ : \pm { \ : 0 . 0 1 } }$ </td><td> ${ \bf 9 5 . 7 \pm 0 . 1 8 }$ </td></tr><tr><td>NRSCP</td><td> $2 . 2 \pm 0 . 0 7$ </td><td> $9 8 . 9 \pm 0 . 1 2 $ </td><td>3.3 ± 0.20</td><td> $9 9 . 6 \pm 0 . 0 8$ </td><td> $1 . 2 \pm 0 . 0 1$ </td><td> $9 7 . 8 \pm 0 . 1 2$ </td></tr><tr><td>NACP</td><td> $1 . 3 \pm 0 . 0 4$ </td><td> $9 4 . 4 \pm 0 . 6 2$ </td><td> $1 . 3 \pm 0 . 0 4$ </td><td> $9 4 . 5 \pm 0 . 6 2$ </td><td> $1 . 1 \pm 0 . 0 1$ </td><td> $9 5 . 9 \pm 0 . 1 8$ </td></tr><tr><td rowspan="8">CIFAR-100 (100 classes)</td><td>CP (oracle)</td><td> $6 . 5 \pm 0 . 2 0$ </td><td>90.0 ± 0.43</td><td> $\overline { { 4 . 0 \pm 0 . 0 8 } }$ </td><td> $\overline { { 9 0 . 0 \pm 0 . 4 3 } }$ </td><td> $\overline { { 2 . 0 \pm 0 . 0 3 } }$ </td><td>90.0 ± 0.43</td></tr><tr><td>Noisy-CP</td><td> $5 0 . 5 \pm 1 . 2 9$ </td><td> $9 9 . 8 \pm 0 . 0 4$ </td><td> $5 0 . 5 \pm 1 . 3 3$ </td><td> $9 9 . 8 \pm 0 . 0 3$ </td><td> $5 0 . 1 \pm 1 . 3 4$ </td><td> $9 9 . 9 \pm 0 . 0 2$ </td></tr><tr><td> $\mathrm { N R - C P ~ } ( \mathrm { w / o } ~ \Delta )$ </td><td> $6 . 4 \pm 0 . 2 8$ </td><td> $8 9 . 9 \pm 0 . 5 4$ </td><td> $4 . 0 \pm 0 . 1 1$ </td><td> $8 9 . 9 \pm 0 . 5 5$ </td><td> $2 . 0 \pm 0 . 0 6$ </td><td> $8 9 . 9 \pm 0 . 5 6$ </td></tr><tr><td>ACNL</td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td></tr><tr><td>ACNL+</td><td> $5 0 . 4 \pm 1 . 1 3$ </td><td> $9 9 . 8 \pm 0 . 0 3$ </td><td> $5 0 . 4 \pm 1 . 0 1$ </td><td> $9 9 . 9 \pm 0 . 0 3$ </td><td> $5 0 . 1 \pm 1 . 2 3$ </td><td> $9 9 . 9 \pm 0 . 0 2$ </td></tr><tr><td>CRCP</td><td> $2 5 . 7 \pm 3 . 7 1$ </td><td> $9 8 . 7 \pm 0 . 3 9$ </td><td> $8 . 5 \pm 0 . 4 1$ </td><td> $9 8 . 3 \pm 0 . 1 6$ </td><td> $1 1 . 1 \pm 3 . 4 6$ </td><td> $9 8 . 7 \pm 0 . 4 2 $ </td></tr><tr><td>NRSCP</td><td> $3 7 . 6 \pm 0 . 8 6$ </td><td> $9 9 . 6 \pm 0 . 0 6$ </td><td> $5 0 . 1 \pm 1 . 0 8$ </td><td> $9 9 . 8 \pm 0 . 0 3$ </td><td> $1 1 . 7 \pm 0 . 3 7$ </td><td> $9 8 . 9 \pm 0 . 0 8$ </td></tr><tr><td>NACP</td><td> ${ \bf 9 . 0 \pm 0 . 4 6 }$ </td><td> ${ \bf 9 3 . 0 \pm 0 . 4 9 }$ </td><td> ${ \bf 4 . 8 \pm 0 . 1 3 }$ </td><td> ${ \bf 9 3 . 0 \pm 0 . 4 8 }$ </td><td> $\mathbf { 2 . 5 \ : \pm { \ : 0 . 0 9 } }$ </td><td> ${ \bf 9 3 . 0 \pm 0 . 5 2 }$ </td></tr><tr><td rowspan="8">TinyImagenet (200 classes)</td><td>CP (oracle)</td><td> $\overline { { 1 4 . 9 \pm 0 . 6 0 } }$ </td><td> $\overline { { 9 0 . 0 \pm 0 . 6 1 } }$ </td><td> $\overline { { 6 . 9 \pm 0 . 1 9 } }$ </td><td> $\overline { { 9 0 . 0 \pm 0 . 6 2 } }$ </td><td> $\overline { { { 3 . 8 \pm 0 . 1 3 } } }$ </td><td>90.02 ± 0.58</td></tr><tr><td>Noisy-CP</td><td> $9 9 . 7 \pm 3 . 6 7$ </td><td> $9 9 . 7 \pm 0 . 0 8$ </td><td> $1 0 1 . 4 \pm 3 . 5 8$ </td><td> $9 9 . 5 \pm 0 . 0 9$ </td><td> $9 8 . 3 \pm 3 . 8 0 $ </td><td> $9 9 . 8 \pm 0 . 0 5$ </td></tr><tr><td> $\mathrm { N R - C P ~ } ( \mathrm { w / o } ~ \Delta )$ </td><td> $1 4 . 0 \pm 0 . 9 1$ </td><td> $8 9 . 4 \pm 0 . 8 1$ </td><td> $6 . 7 \pm 0 . 2 7$ </td><td> $8 9 . 3 \pm 0 . 8 0$ </td><td> $3 . 5 \pm 0 . 2 4$ </td><td> $8 9 . 3 \pm 0 . 8 0$ </td></tr><tr><td>ACNL</td><td> $2 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $2 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $2 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td></tr><tr><td>ACNL+</td><td> $9 9 . 7 \pm 3 . 6 7$ </td><td> $9 9 . 7 \pm 0 . 0 8$ </td><td> $1 0 1 . 4 \pm 3 . 5 8$ </td><td> $9 9 . 5 \pm 0 . 0 9$ </td><td> $9 8 . 3 \pm 3 . 8 0 $ </td><td> $9 9 . 8 \pm 0 . 0 5$ </td></tr><tr><td>CRCP</td><td> $2 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td>200.0 ± 0.00</td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $2 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td></tr><tr><td>NRSCP</td><td> $7 9 . 6 \pm 2 . 8 2$ </td><td> $9 9 . 3 \pm 0 . 1 1 $ </td><td> $1 0 0 . 6 \pm 2 . 8 8$ </td><td> $9 9 . 5 \pm 0 . 0 9$ </td><td> $2 8 . 1 \pm 1 . 2 0$ </td><td> $9 8 . 1 \pm 0 . 1 5$ </td></tr><tr><td>NACP</td><td> ${ \bf 2 2 . 6 \pm 1 . 8 7 }$ </td><td> ${ \bf 9 3 . 7 \pm 0 . 7 1 }$ </td><td> ${ \bf 9 . 0 \pm 0 . 5 0 }$ </td><td> ${ \bf 9 3 . 6 \pm 0 . 7 0 }$ </td><td> $\mathbf { 7 . 0 \pm 0 . 8 7 }$ </td><td> ${ \bf 9 3 . 6 \pm 0 . 7 2 }$ </td></tr><tr><td rowspan="8">ImageNet (1000 classes)</td><td>CP (oracle)</td><td> $\overline { { 1 6 . 6 \pm 0 . 3 3 } }$ </td><td> $\overline { { 9 0 . 0 \pm 0 . 2 6 } }$ </td><td> $\overline { { 6 . 3 \pm 0 . 0 6 } }$ </td><td> $\overline { { 9 0 . 0 \pm 0 . 2 7 } }$ </td><td> $\overline { { 0 . 6 \pm 0 . 0 7 } }$ </td><td> $\overline { { 9 0 . 0 \pm 0 . 2 8 } }$ </td></tr><tr><td>Noisy-CP</td><td> $5 0 2 . 6 \pm 8 . 5 6$ </td><td> $9 9 . 9 \pm 0 . 0 1$ </td><td> $5 0 1 . 6 \pm 8 . 5 1$ </td><td> $9 9 . 9 \pm 0 . 0 1$ </td><td> $5 0 1 . 3 \pm 1 0 . 2$ </td><td> $1 0 0 . 0 \pm 0 . 0 1$ </td></tr><tr><td> $\mathrm { N R - C P ~ } ( \mathrm { w / o } ~ \Delta )$ </td><td> $1 6 . 7 \pm 0 . 5 1$ </td><td> $9 0 . 0 \pm 0 . 3 4$ </td><td> $6 . 3 \pm 0 . 1 0$ </td><td> $9 0 . 0 \pm 0 . 3 6$ </td><td> $3 . 6 \pm 0 . 1 4$ </td><td> $9 0 . 0 \pm 0 . 3 8$ </td></tr><tr><td>ACNL</td><td> $1 0 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td></tr><tr><td>ACNL+</td><td> $5 0 2 . 6 \pm 8 . 5 6$ </td><td> $9 9 . 9 \pm 0 . 0 1$ </td><td> $5 0 1 . 6 \pm 8 . 5 1$ </td><td> $9 9 . 9 \pm 0 . 0 1$ </td><td> $5 0 1 . 3 \pm 1 0 . 2$ </td><td> $1 0 0 . 0 \pm 0 . 0 1$ </td></tr><tr><td>CRCP</td><td> $1 0 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td></tr><tr><td>NRSCP</td><td> $2 7 5 . 6 \pm 2 7 . 1$ </td><td> $9 9 . 7 \pm 0 . 0 6$ </td><td> $4 5 5 . 2 \pm 2 0 . 7$ </td><td> $9 9 . 9 \pm 0 . 0 2$ </td><td> $5 5 . 9 \pm 0 . 8 7$ </td><td> $9 9 . 1 \pm 0 . 0 2$ </td></tr><tr><td>NACP</td><td> ${ \bf 2 0 . 9 \pm 0 . 7 2 }$ </td><td> ${ \bf 9 1 . 9 \pm 0 . 3 2 }$ </td><td> ${ \bf 7 . 1 \pm 0 . 1 3 }$ </td><td> ${ \bf 9 1 . 9 \pm 0 . 3 4 }$ </td><td> ${ \bf 4 . 8 \pm 0 . 2 3 }$ </td><td> ${ \bf 9 1 . 9 \pm 0 . 3 6 }$ </td></tr></table>

Table 4.3: Finite sample correction terms $\Delta$ of NACP, ACNL [86] and CRCP [13], for several datasets and two noise levels, n is the size of the validation set.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">n</td><td rowspan="2">#classes</td><td colspan="2">NACP</td><td colspan="2">ACNL</td><td colspan="2">CRCP</td></tr><tr><td>€ = 0.1</td><td> $\epsilon = 0 . 2$ </td><td> $\epsilon = 0 . 1$ </td><td> $\epsilon = 0 . 2$ </td><td> $\epsilon = 0 . 1$ </td><td> $\epsilon = 0 . 2$ </td></tr><tr><td>CIFAR-10</td><td>5000</td><td>10</td><td>0.035</td><td>0.043</td><td>0.031</td><td>0.059</td><td>0.016</td><td>0.036</td></tr><tr><td>CIFAR-100</td><td>10000</td><td>100</td><td>0.025</td><td>0.030</td><td>0.077</td><td>0.163</td><td>0.039</td><td>0.088</td></tr><tr><td>TinyImagenet</td><td>5000</td><td>200</td><td>0.035</td><td>0.043</td><td>0.175</td><td>0.382</td><td>0.078</td><td>0.176</td></tr><tr><td>ImageNet</td><td>25000</td><td>1000</td><td>0.016</td><td>0.019</td><td>0.194</td><td>0.466</td><td>0.079</td><td>0.177</td></tr></table>

Implementation details. Each task was trained by fine-tuning on a pre-trained ResNet-18 [35] network. The models were taken from the PyTorch site<sup>4</sup>. We selected this network architecture because of its widespread use in classification problems. The last fully connected layer output size was modified to fit the corresponding number of classes for each dataset. For the standard dataset evaluated in Table 4.2 we used publicly available checkpoints. For each dataset, we combined the validation and test sets and then constructed 1000 diferent splits where 50% was used for the calibration phase and 50% was used for testing. In all our experiments we used $\delta = 0 . 0 0 1$ . In other words, the computed coverage guarantee is applied to the sampled noisy validation set with probability 0.999

Conformal prediction results. Table 4.2 reports the noisy label calibration results across 3 diferent conformal prediction scores, HPS, APS, and RAPS for four standard publicly available datasets, CIFAR-10, CIFAR-100, Tiny-ImageNet, and ImageNet. In all cases, we used $1 - \alpha = 0 . 9$ and a noise level of $\epsilon = 0 . 2 .$ . The results indicate that in the case of a validation set with noisy labels, the Noisy-CP threshold became larger to facilitate the uncertainty induced by the noisy labels. This yielded larger prediction sets and the coverage was higher than the target coverage which was set to 90%. We can see that NACP outperformed the ACNL, and CRCP methods for all datasets except for CIFAR-10 with fewer classes. Following Theorem 4.3.3, we expect the gain in performance when using NACP versus Noisy-CP to increase with the number of classes, indeed validated empirically in Table 4.2. Here for CIFAR-100, Tiny-ImageNet, and ImageNet the ACNL and CRCP methods failed due to the large number of classes and the relatively small number of samples per class. For TinyImagenet and Imagenet, $\mathrm { A C N L ^ { + } }$ falls back to Noisy-CP. A direct comparison of the finite sample correction terms $\Delta$ obtained by NACP, ACNL and CRCP is shown in Table 4.3. Note that if $1 - \alpha + \Delta > 1$ , the prediction set includes all the classes and thus it becomes useless. We can see in Table 4.3 that this is the case for ACNL and CRCP in datasets with a large number of classes.

![](images/cb8f7eb708bb2414a1bff6fbda1c5269c89ef53507cfbad64f9d06e88e8e6277.jpg)

![](images/e8ee145b31765e6a25ec31919467daa9219acf9d455e21e8d03a7ab3a3044881.jpg)

![](images/40d540f47bd992d1f086e8f5349810610f4f6be940ef8e2fa8dfb3bc62015f38.jpg)  
Figure 4.1: Correction terms $\Delta$ of NACP, ACNL and CRCP as a function of the validation set size n given $\epsilon = 0 . 2$ . We show results for 3 numbers of classes, 10, 100 and 1000.

Correction term analysis. Following the theoretical and empirical results, the efectiveness of our method and baselines can be fully explained by the correction terms $\Delta$ each method guarantees as practitioners require coverage guarantee and therefore will use $1 - \alpha + \Delta$ . Note that, as explained in Section 4.4, our finite sample coverage guarantee is diferent from the one provided by the baseline method. Figure 4.1 presents the correction term as a function of calibration set size and the number of classes. Note that the NACP curve remains exactly the same across the 3 plots. Our main contribution is grounded in the fact that NACP is not dependent on the number of classes k, clearly shown in plots as the number of classes grows.

General noise transition matrix. Finally, we evaluate NRSCP and NACP on two common general noise matrices: Neighborhood noise and Random noise (see details in Section 2.4). While existing final sample terms bounds are not efective, in practice NACP (without a finite sample correction) achieves the required coverage guarantee and the average prediction size is similar to the one obtained by the noise-free CP. We observe the same pattern when using uniform noise. This indicates that the current coverage guarantee bounds are too conservative. Table 4.4 shows the results on the CIFAR-100 dataset and the rand-APS technique when using NACP without finite sample correction term ∆. Results show a clear dominance of $\mathrm { N A C P }$ over Noisy-CP and NRCP on the two diferent noise models, presenting the robustness of NACP across various noise models. ACNL (without the finite sample term) achieves here similar results.

Table 4.4: Rand-APS calibration results for $1 - \alpha = 0 . 9 \mathrm { o n }$ CIFAR-100 dataset and two noise models. We report the mean and the std over 1000 diferent splits.
<table><tr><td></td><td>Neighborhood noise</td><td colspan="2">Random noise</td></tr><tr><td>CP Method</td><td>size ↓ coverage (%)</td><td></td><td>size ↓ coverage (%)</td></tr><tr><td> $\overline { { \mathrm { C P ~ } ( \mathrm { o r a c l e } ) } }$ </td><td> $\overline { { 6 . 4 8 \pm 0 . 1 9 } }$   $\overline { { 9 0 . 0 1 \pm 0 . 4 1 } }$ </td><td> $\overline { { 6 . 4 8 \pm 0 . 1 9 } }$ </td><td> $\overline { { 9 0 . 0 1 \pm 0 . 4 1 } }$ </td></tr><tr><td> $\mathrm { N o i s y - C P }$ </td><td> $4 8 . 8 9 \pm 1 . 1 3$   $9 9 . 8 0 \pm 0 . 0 4$ </td><td> $5 0 . 2 5 \pm 1 . 3 7$ </td><td> $9 9 . 8 2 \pm 0 . 0 4$ </td></tr><tr><td>NRSCP</td><td> $1 2 . 8 2 \pm 0 . 3 6$   $9 5 . 6 2 \pm 0 . 2 1$ </td><td> $3 7 . 0 1 \pm 0 . 8 8$ </td><td> $9 9 . 5 3 \pm 0 . 0 6$ </td></tr><tr><td> $\mathrm { N R - C P ~ } ( \mathrm { w / o } ~ \Delta )$ </td><td> ${ \bf 6 . 5 2 \pm 0 . 2 2 }$   $9 0 . 0 3 \pm 0 . 4 7$ </td><td> ${ \bf 6 . 4 5 \pm 0 . 3 0 }$ </td><td> $8 9 . 9 7 \pm 0 . 5 7$ </td></tr></table>

![](images/b34b424269613b74dd6483676617e77c13de6a8e451ed6dafa79a9229857d89d.jpg)  
(a)

![](images/3110d26be71a8767fb71c8cb7d1d3b967c7da798674975eb3b2a2df7600735a7.jpg)  
(b)

![](images/51ad63870a1a3bff35628a20baa3706995abf92954f5095eaf734fe64d02f7d8.jpg)  
(c)  
Figure 4.2: Noisy labels conformal prediction on ImageNet with diferent calibration set sizes. (a) Mean size (b) Coverage (%), and (c) Correction terms ∆ as a function of calibration set size.

Next, we show the results of the following experiments. In Calibration set size we test the performance of various conformal prediction methods under noisy labels as a function of the calibration set size. In NACP Agnostic to Diferent model architectures we show that NACP is agnostic to different classification network architectures. Finally, in Experiments on real noisy datasets we report experiments on real noisy datasets where the noise is due to manual annotation error. We show that in this case, by imposing a uniform noise model, we get better results than the one obtained by ignoring the noise and applying CP directly on the noisy validation set.

Calibration set size. In the following experiment, we test the performance of various conformal prediction methods under noisy labels as a function of the calibration set size on the ImageNet dataset. Figure 4.2 shows the mean size and coverage as a function of the calibration set size. In addition, the correction term $\Delta$ is depicted for ImageNet for each calibration set size. Results show that even with as little as 2500 images that correspond to 2.5 images per class the calibration results are almost on par with the oracle calibration given clean labels.

NACP Agnostic to Diferent model architectures. Conformal prediction in general and our method NACP specifically has no assumption and is agnostic to the underlying model architecture. In the following section, we verify that by experimenting with ImageNet across diferent model architectures. Table 4.5 presents the results of applying conformal prediction with and without noisy labels on ResNet18, ResNet50, DenseNet121, ViT-B16 (Vision transformer).

Table 4.5: CP calibration results on ImageNet and various model architectures for $1 - \alpha = 0 . 9$ and ϵ = 0.2. We report the mean and the std over 1000 diferent splits. Bold for best result with theoretical guarantees.
<table><tr><td></td><td></td><td colspan="2">ResNet-18</td><td colspan="2">ResNet-50</td><td colspan="2">DenseNet121</td><td colspan="2">ViT-B16</td></tr><tr><td>Dataset</td><td>CP Method</td><td>size ↓</td><td> $\overline { { \mathrm { c o v e r a g e } ( \% ) } }$ </td><td>size↓</td><td> $\underline { { \mathrm { c o v e r a g e } ( \% ) } }$ </td><td>size↓</td><td> $\overline { { \mathrm { c o v e r a g e } ( \% ) } }$ </td><td>size ↓</td><td>coverage(%)</td></tr><tr><td>APS</td><td>CP (oracle)</td><td>16.6 ± 0.33</td><td> $\overline { { 9 0 . 0 \pm 0 . 2 6 } }$ </td><td> $\overline { { 1 3 . 9 \pm 0 . 3 4 } }$ </td><td> $\overline { { 9 0 . 0 \pm 0 . 2 8 } }$ </td><td>12.0 ± 0.28</td><td> $\overline { { 9 0 . 0 \pm 0 . 2 7 } }$ </td><td> $\overline { { 1 0 . 7 \pm 0 . 3 8 } }$ </td><td>90.0 ± 0.25</td></tr><tr><td></td><td>Noisy-CP</td><td> $5 0 2 . 6 \pm 8 . 5 6$ </td><td> $9 9 . 9 \pm 0 . 0 1 $ </td><td> $5 0 5 . 5 \pm 8 . 1 1$ </td><td> $9 9 . 9 \pm 0 . 0 1$ </td><td>502.8 ± 8.46</td><td> $9 9 . 9 \pm 0 . 0 1$ </td><td> $5 0 6 . 8 \pm 8 . 1 4$ </td><td> $9 9 . 8 \pm 0 . 0 2$ </td></tr><tr><td></td><td>NR-CP (w/o ∆) ACNL</td><td> $1 6 . 7 \pm 0 . 5 1$ </td><td> $9 0 . 0 \pm 0 . 3 4$ </td><td> $1 3 . 9 \pm 0 . 4 7$ </td><td> $9 0 . 0 \pm 0 . 3 7$ </td><td> $1 2 . 0 \pm 0 . 3 8$ </td><td> $9 0 . 0 \pm 0 . 3 4$ </td><td> $1 0 . 7 \pm 0 . 5 5$ </td><td> $9 0 . 0 \pm 0 . 3 5$ </td></tr><tr><td></td><td>ACNL+</td><td>1000.0 ±0.00</td><td>100.0 ± 0.00</td><td> $\begin{array} { c } { { 1 0 0 0 . 0 \pm 0 . 0 0 } } \\ { { . . . . } } \end{array}$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $\begin{array} { c } { 1 0 0 0 . 0 \pm 0 . 0 0 } \\ { . - . - . } \end{array}$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td></tr><tr><td></td><td></td><td>502.6 ± 8.56</td><td> $9 9 . 9 \pm 0 . 0 1 $ </td><td>505.5 ± 8.11</td><td> $9 9 . 9 \pm 0 . 0 1$ </td><td>502.8 ± 8.46</td><td>99.9 ± 0.01</td><td>506.8 ± 8.14</td><td> $9 9 . 8 \pm 0 . 0 2$ </td></tr><tr><td></td><td>CRCP</td><td>1000.0 ±0.00</td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 0 . 0 \pm 0 . 0 0$ </td><td>100.0 ± 0.00</td><td>1000.0 ±0.00</td><td>100.0 ± 0.00</td><td> $1 0 0 0 . 0 \pm 0 . 0 0$ </td><td>100.0 ± 0.00</td></tr><tr><td>RAPS</td><td>NACP</td><td>20.9 ± 0.72</td><td> ${ \bf 9 1 . 9 \pm 0 . 3 2 }$ </td><td> $\mathbf { 1 7 . 4 \ : \pm { \ : 0 . 6 2 } }$ </td><td>91.9 ± 0.36</td><td>15.1 ± 0.55</td><td>91.9 ± 0.34</td><td>15.5 ± 0.81</td><td>91.9 ± 0.31</td></tr><tr><td></td><td>CP (oracle)</td><td>6.3 ± 0.06</td><td>90.0 ± 0.27</td><td> $\overline { { \phantom { - } 4 . 5 \pm 0 . 0 5 } }$ </td><td>89.9 ± 0.29</td><td>4.7 ± 0.06</td><td>90.0 ± 0.26</td><td>2.6 ± 0.04</td><td>90.0 ± 0.25</td></tr><tr><td></td><td>Noisy-CP</td><td>501.6 ± 8.51</td><td>99.9 ± 0.01</td><td>501.1 ± 8.85</td><td>99.9 ± 0.01</td><td>501.9 ± 8.80</td><td>99.9 ± 0.01</td><td>505.8 ± 7.90</td><td>99.9 ± 0.01</td></tr><tr><td></td><td>NR-CP (w/o ∆)</td><td>6.3 ± 0.10</td><td>90.0 ± 0.36</td><td>4.5 ± 0.06</td><td>90.0 ± 0.35</td><td>4.7 ± 0.08</td><td>90.0 ± 0.36</td><td>2.6 ± 0.05</td><td>90.0 ± 0.36</td></tr><tr><td></td><td>ACNL</td><td>1000.0 ±0.00</td><td>100.0 ± 0.00</td><td>1000.0 ±0.00</td><td>100.0 ± 0.00</td><td>1000.0 ±0.00</td><td>100.0 ± 0.00</td><td>1000.0 ±0.00</td><td>100.0 ± 0.00</td></tr><tr><td></td><td>ACNL+</td><td>501.6 ± 8.51</td><td>99.9 ± 0.01</td><td>501.1 ± 8.85</td><td>99.9 ± 0.01</td><td>501.9 ± 8.80</td><td>99.9 ± 0.01</td><td>505.8 ± 7.90</td><td>99.9 ± 0.01</td></tr><tr><td></td><td>CRCP</td><td>1000.0 ±0.00</td><td>100.0 ± 0.00</td><td>1000.0 ±0.00</td><td>100.0 ± 0.00</td><td>1000.0 ±0.00</td><td>100.0 ± 0.00</td><td>1000.0 ±0.00</td><td>100.0 ± 0.00</td></tr><tr><td>HPS</td><td>NACP</td><td>7.1 ± 0.13</td><td>91.9 ± 0.34</td><td>5.0 ± 0.08</td><td>91.9 ± 0.34</td><td>5.3 ± 0.10</td><td>92.0 ± 0.35</td><td>2.9 ± 0.07</td><td>92.0 ± 0.30</td></tr><tr><td></td><td>CP (oracle)</td><td>3.6 ± 0.07</td><td>90.0 ± 0.28</td><td>2.0 ± 0.03</td><td>90.0 ± 0.28</td><td>2.4 ± 0.03</td><td>90.0 ± 0.25</td><td>1.5 ± 0.02</td><td> $\overline { { 9 0 . 0 \pm 0 . 2 6 } }$ </td></tr><tr><td></td><td>Noisy-CP</td><td>501.3 ± 10.2</td><td>100.0 ± 0.01</td><td>502.4 ± 9.50</td><td>99.9 ± 0.01</td><td>502.3 ± 10.3</td><td>99.9 ± 0.20</td><td>504.3 ± 8.19</td><td> $9 9 . 9 \pm 0 . 0 1$ </td></tr><tr><td></td><td>NR-CP (w/o ∆)</td><td>3.6 ± 0.14</td><td> $9 0 . 0 \pm 0 . 3 8$ </td><td>2.1 ± 0.06</td><td>90.0 ± 0.38</td><td>2.4 ± 0.07</td><td>90.0 ± 0.34</td><td>1.5 ± 0.03</td><td> $9 0 . 0 \pm 0 . 3 5$ </td></tr><tr><td></td><td>ACNL</td><td>1000.0 ±0.00</td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td>1000.0 ±0.00</td><td>100.0 ± 0.00</td><td> $1 0 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td></tr><tr><td></td><td>ACNL+</td><td>501.3 ± 10.2</td><td>100.0 ± 0.01</td><td> $5 0 2 . 4 \pm 9 . 5 0$ </td><td> $9 9 . 9 \pm 0 . 0 1$ </td><td>502.3 ± 10.3</td><td>99.9 ± 0.20</td><td> $5 0 4 . 3 \pm 8 . 1 9$ </td><td> $9 9 . 9 \pm 0 . 0 1$ </td></tr><tr><td></td><td>CRCP</td><td>1000.0 ±0.00</td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td><td>1000.0 ±0.00</td><td>100.0 ± 0.00</td><td> $1 0 0 0 . 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 \pm 0 . 0 0$ </td></tr><tr><td></td><td>NACP</td><td>4.8 ± 0.23</td><td> ${ \bf 9 1 . 9 \pm 0 . 3 6 }$ </td><td> $\mathbf { 2 . 6 \ : \pm { \ : 0 . 1 0 } }$ </td><td> ${ \bf 9 1 . 9 \pm 0 . 3 7 }$ </td><td>3.1 ± 0.12</td><td> ${ \bf 9 1 . 9 \pm 0 . 3 4 }$ </td><td> $\mathbf { 1 . 7 \pm 0 . 0 4 }$ </td><td> ${ \bf 9 1 . 9 \pm 0 . 3 3 }$ </td></tr></table>

Experiments on real noisy datasets. We evaluate our methods on real-world noisy datasets, focusing on the CIFAR-10N dataset, which contains human annotation errors and was introduced in [99]. Specifically, we analyze four variations of CIFAR-10N: CIFAR-10-aggregate and CIFAR-10- random-1,2,3. CIFAR-10-aggregate combines three noisy labels using majority voting. If the three submitted labels difer, the aggregated label is randomly selected from the three options. CIFAR-10-random-i (i ∈ {1, 2, 3}) refers to the i-th submitted label for each image. Importantly, the data collection process ensures that no image is labeled multiple times by the same annotator.

While this noise model realistically reflects human annotation behavior, it does not adhere to a strict uniform noise distribution. For our experiments with NACP and baseline methods, we adopt the noise ratio reported in [99] to compute ϵ for the noise-aware conformal prediction algorithm. Notably, [99] defines the noise rate with ϵ such that $T _ { i , i } = 1 - \epsilon$ In contrast, our notation uses ϵ with the formulation $\begin{array} { r } { T _ { i , i } = 1 - \epsilon + \frac { \epsilon } { k } } \end{array}$ . Consequently, we adjust ϵ values from the original paper to align with our approach.

The purpose of this experiment is to demonstrate that even when the true noise—arising from annotators—is not exactly uniform, approximating it as such can still yield efective performance in real-world datasets and scenarios.

Table 4.6 summarizes the results for the four CIFAR-10N variations. In this real-world scenario, clean labels (CP Oracle) are unavailable. Instead, Noisy-CP results reflect calibration using annotatorprovided labels as-is. Our noise-aware approach demonstrates a consistent improvement over this baseline.

Table 4.6: CP calibration results on CIFAR-10N for 1−α = 0.9 and ϵ = 0.2. We report the mean and the std over 1000 diferent splits. Bold for best result with theoretical guarantees.
<table><tr><td colspan="2"></td><td colspan="2">CIFAR-10N-aggregate (10.0%)</td><td colspan="2">CIFAR-10N-random-1 (19.1%)</td><td colspan="2">CIFAR-10N-random-2 (20.1%)</td><td colspan="2">CIFAR-10N-random-3 (19.6%)</td></tr><tr><td rowspan="2">Dataset rand-APS</td><td>CP Method</td><td>size ↓</td><td>coverage(%)</td><td>size ↓ coverage(%)</td><td>size ↓</td><td>coverage(%)</td><td>size ↓</td><td>coverage(%)</td></tr><tr><td>Noisy-CP</td><td>1.76</td><td>91.73</td><td>2.00</td><td>93.18</td><td>2.13 93.39</td><td>2.44</td><td>96.45</td></tr><tr><td rowspan="2">HPS</td><td>NACP</td><td>1.58</td><td>88.49</td><td>1.82</td><td>91.38</td><td>1.93 91.67</td><td>2.13</td><td>94.50</td></tr><tr><td>Noisy-CP</td><td>1.32</td><td>91.19</td><td>1.67</td><td>92.97</td><td>1.79 93.15</td><td>2.16</td><td>96.31</td></tr><tr><td rowspan="2"></td><td>NACP</td><td>1.38</td><td>91.85</td><td>1.45</td><td>91.00</td><td>1.47 90.41</td><td>1.53</td><td>92.18</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Chapter 5

## Local Diferential Private Conformal Prediction

In this chapter, we explore the intersection of conformal prediction (CP) and privacy-preserving techniques, introducing a Local Diferentially Private Conformal Prediction (LDP-CP) framework. While traditional CP methods rely on cleanly labeled calibration sets, data privacy concerns often prevent access to true labels, particularly in sensitive domains like medical or personal data. Our proposed framework addresses the challenge of maintaining valid prediction set coverage while protecting user data through Local Diferential Privacy (LDP). We present two complementary approaches—LDP-CP-L and LDP-CP-S—that cater to varying privacy goals, computational resources, and data scenarios, ofering a robust solution for secure, decentralized data handling.

## 5.1 Problem Statement

Conformal prediction typically relies on having access to a cleanly labeled calibration set to set the CP threshold. However, in many real-world scenarios, such clean labels may be unavailable due to data privacy concerns. For instance, in scenarios involving medical or personal data, or if the data aggregator (e.g. a cloud-based ML service) is considered untrusted, privacy constraints may prohibit access to true labels and data, requiring a privacy-preserving mechanism that provides Local Diferential Privacy (LDP). In such situations, the aggregator might only be allowed to view noisy versions of the labels.

Next we tackle the challenge of applying conformal prediction when validation-set labels (or conformity scores) must be protected through privacy-preserving mechanisms. Specifically, we introduce a Local Diferentially Private Conformal Prediction (LDP-CP) framework that balances privacy with real-world considerations such as user-side computational capacity, aggregator trustworthiness, and intellectual property.

We present two complementary LDP-CP solutions. LDP-CP-L locally perturbs labels using a randomized response, that shifts all score-related computations to the aggregator. This design suits cases where users have minimal computational resources, or the model’s internal structure is not disclosed to them, while not sharing their sensitive labels. However, it only achieves label-DP [5, 29]. In contrast, LDP-CP-S allows users to generate and locally randomize their own conformity scores, which is ideal for scenarios where both feature and label privacy are paramount and the user can handle additional computational tasks. We also ofer guidelines on choosing which method aligns better with specific privacy goals, resource constraints, and per-dataset properties such as sample size and the number of classes.

By considering both score-based and label-based perturbations, we provide a flexible framework that adapts to various privacy budgets and computational setups. This framework guarantees valid coverage for the true labels asymptotically and in finite samples. As a result, LDP-CP aligns well with modern demands for secure, decentralized data handling.

Relationship to Past Work. Conformal prediction in the presence of label noise has received growing attention [21, 86, 13] and Chapter 4, with methods that adjust the calibration threshold when a known noise matrix corrupts labels. Simultaneously, LDP has become a leading approach for protecting sensitive data at the user end [98, 45, 95], allowing users to randomize their own inputs (e.g., labels) before sharing with an untrusted aggregator. We bridge these lines of work by recognizing that local DP can be seen as a known noisy channel on the labels or conformal scores, we can plug that channel into a “noise-aware” conformal procedure. The end result is a conformal predictor whose coverage remains valid, while also protecting each user’s label or score via ε-LDP. To the best of our knowledge, there is no prior work on LDP conformal prediction. Thus, we are the first to research local diferential private conformal prediction. The closest work to ours is that of (author?) [3], which suggests a centrally diferentially private conformal prediction procedure where a trusted aggregator has access to raw data. In our setting, the aggregator never observes true labels directly, hence, closing a key gap in privacy-preserving conformal prediction research.

This dissertation makes three key contributions:

• We introduce two complementary LDP-CP methods, LDP-CP-S and LDP-CP-L, that accommodate diferent privacy constraints and computational setups. (Figures 5.1 and 5.2).

• We prove coverage guarantees for both methods under LDP constraints. (Theorems 5.4.1 and 5.4.2).

• We demonstrate the feasibility and efectiveness of our approaches in privacy-sensitive applications, such as medical data analysis and untrusted cloud-based ML services (see Section 5.6).

![](images/069127cf781e8fada00ff4e469a651f4e99f7a20353dd6eb3681173ef9a32f6b.jpg)  
Figure 5.1: Local Diferential Private Conformal Prediction (LDP-CP-L) Pipeline (Best viewed in color).

This work bridges the gap between privacy-preserving mechanisms and conformal prediction, by providing a foundation for robust and private uncertainty quantification in real-world scenarios. Unlike other methods that rely on trusted aggregators, our approaches ensure privacy directly at the user level, which aligns with modern demands for decentralized privacy protection.

## 5.2 Local-DP Conformal Prediction on Labels

In many real-world contexts, users lack the ability—or permission—to compute model-based scores on their own devices. This may be due to limited computational resources, a restricted API that only provides predictions, or a proprietary model architecture. To address these cases, we propose an LDP mechanism that randomizes each user’s label and then applies a noise-aware conformal calibration at the aggregator. This design protects sensitive labels while remaining model-agnostic, since the aggregator performs all the scoring steps. Figure 5.1 illustrates the overall pipeline.

Algorithm 6 LDP-CP-L - User-side Procedure   
1: Input: Calibration set $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } ,$ privacy parameter ϵ, number of labels k   
2: Output: Noisy calibration set $\{ ( x _ { i } , \tilde { y } _ { i } ) \} _ { i = 1 } ^ { n }$   
3: Set $\begin{array} { r } { \bar { \beta } = \frac { k } { k - 1 + e ^ { \epsilon } } } \end{array}$   
4: for each user $i \in \{ 1 , \ldots , n \}$ do   
5: Apply k-ary Randomized Response (RR) to label y to obtain ${ \tilde { y } } _ { i } { \mathrm { : } }$   
6:   
$p ( \tilde { y } = t ^ { \prime } \mid y = t ) = 1 _ { \{ t = t ^ { \prime } \} } ( 1 - \beta ) + \frac { \beta } { k }$   
7: Send $( x _ { i } , \tilde { y } _ { i } )$ to the aggregator

We next unpack LDP-CP-L by first describing the user’s side procedure that applies to its data point, and then the procedure done by the untrusted aggregator to compute the conformal prediction threshold.

User’s side Procedure. For each user i with pair $( x _ { i } , y _ { i } )$

1. User applies the LDP mechanism - k-ary RR to their label. This yields $\tilde { y } _ { i }$

2. Report $( x _ { i } , \tilde { y } _ { i } )$ to the aggregator.

Aggregator’s Procedure. Once the aggregator collects $\{ x _ { i } , \tilde { y } _ { i } \} _ { i = 1 } ^ { n }$ (it does not see $y _ { i } )$ , it runs a “noise-aware” conformal method on the pairs $( x _ { i } , \tilde { y } _ { i } )$ . Since the aggregator knows the noise model, it can estimate the coverage for the true label.

A key advantage of this label-perturbation strategy is that the aggregator never observes raw labels, thus meeting label-LDP guarantees. Meanwhile, by modeling a k-ary Randomized Response as a known noise channel, the aggregator recovers the necessary calibration adjustments to preserve near-correct coverage on the true labels. In what follows, we outline the procedure in more detail, leading to our first main theorem and contribution culminating in Theorem 5.4.1.

For k-RR, each label is replaced by a random label with probability $\beta ,$ known channel from $y$ to ${ \tilde { y } } .$ Then,

$$
p ( \tilde { y } = j \mid y = i ) = 1 _ { \{ i = j \} } ( 1 - \beta ) + \frac { \beta } { k } ; \quad \beta = \frac { k } { k - 1 + e ^ { \epsilon } }
$$

For each threshold $q ,$ let $F ^ { c } ( q ) = p { \big ( } S ( x , y ) \leq q { \big ) } , F ^ { n } ( q ) = p { \big ( } S ( x , \tilde { y } ) \leq q { \big ) } , F ^ { r } ( q ) = p { \big ( } S ( x , u ) \leq q { \big ) } .$ $q )$ 2 $u \sim U n i f ( 1 , \dots , k )$ coverage on true labels, noisy labels, and uniform respectively. One can see that

$$
F ^ { n } ( q ) = ( 1 - \beta ) F ^ { c } ( q ) + \beta F ^ { r } ( q ) .\tag{5.1}
$$

Given a calibration set $\{ ( x _ { i } , \tilde { y } _ { i } ) \}$ , we can estimate $F ^ { n } ( q )$ and $F ^ { r } ( q )$ by

$$
\hat { F } ^ { n } ( q ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } 1 \big ( S ( x _ { i } , \tilde { y } _ { i } ) \leq q \big ) ; \hat { F } ^ { r } ( q ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { | C _ { q } ( x _ { i } ) | } { k } ,
$$

Thus, rearranging Eq. (5.1) we obtain:

$$
\hat { F } ^ { c } ( q ) = \frac { \hat { F } ^ { n } ( q ) - \beta \hat { F } ^ { r } ( q ) } { 1 - \beta } .
$$

Hence, to find a threshold $q$ that yields coverage 1 − α on the true labels in a private manner, we solve $\hat { F } ^ { c } ( q ) = { 1 - \alpha }$ . In practice, we do a binary search over candidate thresholds which continues until either the estimate $\hat { Z } ^ { ( j ) }$ satisfies $| \hat { Z } ^ { ( j ) } - ( 1 - \alpha ) | \le \Delta$ or the interval length $s _ { h i g h } - s _ { l o w }$ becomes smaller than a predefined threshold τ . This is exactly the “noise-aware” threshold that corrects for the k-RR noise mechanism. User’s side and aggregator’s side procedures depicted in Algorithm boxes 6 and 7 respectively.

Algorithm 7 LDP-CP-L - Aggregator-side Procedure   
1: Input: Noisy calibration set $\{ ( x _ { i } , \tilde { y } _ { i } ) \} _ { i = 1 } ^ { n }$ , privacy parameter ϵ, number of labels k, target   
coverage 1 − α   
2: Output: Threshold $q$ ensuring private coverage $1 - \alpha - \Delta$ on true labels   
3: Collect the noisy calibration set $\{ ( x _ { i } , \tilde { y } _ { i } ) \} _ { i = 1 } ^ { n }$   
4: Compute $\begin{array} { r } { \beta = \frac { k } { k - 1 + e ^ { \epsilon } } } \end{array}$   
5: Estimate ${ \hat { F } } ^ { n } ( q )$ and $\hat { F } ^ { r } ( q )$ for candidate thresholds $q \colon$   
$\hat { F } ^ { n } ( q ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { 1 } ( S ( x _ { i } , \tilde { y } _ { i } ) \leq q ) ; \quad \hat { F } ^ { r } ( q ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { | C _ { q } ( x _ { i } ) | } { k }$   
6: Compute ${ \hat { F } } ^ { c } ( q )$   
$\hat { F } ^ { c } ( q ) = \frac { \hat { F } ^ { n } ( q ) - \beta \cdot \hat { F } ^ { r } ( q ) } { 1 - \beta }$   
7: Initialize $s _ { l o w } = 0 , s _ { h i g h } = 1$   
8: for $j = 1 , . . . , T$ do   
9: Set $\begin{array} { r } { q ^ { ( j ) } = \frac { s _ { l o w } + s _ { h i g h } } { 2 } } \end{array}$   
10: Obtain $Z ^ { ( j ) } = { \bar { \hat { F } } } ^ { c } ( q ^ { ( j ) } )$   
11: if $\begin{array} { r } { Z ^ { ( j ) } > ( 1 - \alpha ) \overset { \cdot \cdot } { + } \frac { \Delta } { 2 } } \end{array}$ then $s _ { h i g h } = q ^ { ( j ) }$   
12: else $\begin{array} { r } { \mathbf { i f } \ Z ^ { ( j ) } < ( 1 - \alpha ) - \frac { \Delta } { 2 } } \end{array}$ then $s _ { l o w } = q ^ { ( j ) }$   
13: else break   
14: Return the threshold $q ^ { ( j ) }$

In Chapter 4 we considered this exact scenario and referred to this procedure as NACP (Noise-Aware Conformal Prediction). However, while we were motivated by problems related to a general noisy channel (e.g. experts’ mistakes/disagreements as to the true label), here we use the noisy channel as a privacy protection for the labels in the calibration set. As it turns out, using k-RR falls neatly into their paradigm and in turn yields Theorem 5.4.1. Note that the k-RR mechanism is implemented here, instead of more advanced LDP methods, such as RAPPOR, since it aligns well with the NACP framework.

## 5.3 Local-DP Conformal Prediction on Scores

In some scenarios, users may be able to compute the full conformity score locally. Concretely, the aggregator (or model provider) sends the neural network’s parameters or logits to each user, who then computes the conformity score $S ( x _ { i } , y _ { i } )$ for the ground-truth label $y _ { i }$ on their own. This approach leverages the ability to estimate quantiles of a distribution using noisy scores that are locally privatized by the users. We employ the LDP-binary search algorithm described in [26] to estimate the (1 − α)-quantile of the scores. This method is incorporated into the conformal prediction framework by estimating the (1 − α)-quantile of the conformity scores derived locally by the users. Once the quantile is estimated, it is used as the threshold for constructing prediction sets. The use of LDP ensures that the process is privacy-preserving, whereas the quantile estimation guarantees accurate calibration of prediction intervals. $\mathrm { B y }$ combining the strengths of binary search and randomized response, this method ofers a robust approach to privacy-preserving conformal prediction that is both practical and theoretically sound. The settings are depicted in Figure 5.2. User’s side Procedure. For each user i with pair $( x _ { i } , y _ { i } )$

![](images/fa3e32dd372b063ca4cdf2ae22f1003bcb3e1d48a108d7b49c8a70e26b45b4ec.jpg)  
Figure 5.2: Local Diferential Private Conformal Prediction (LDP-CP-S) Pipeline (Best viewed in color).

1. Compute the conformity score $S _ { i } = S ( x _ { i } , y _ { i } )$ locally.

2. Compare $S _ { i }$ with a threshold $q ( j )$ provided by the aggregator. (Note that each user has a single interaction with the aggregator – see details in Aggregator’s procedure).

3. Return a binary response using randomized response (RR), ensuring ε-LDP.

Aggregator Procedure. The aggregator collects the binary responses from users over a binary search procedure. The process starts by defining an initial range that is guaranteed to contain the desired (1 − α)-quantile. The range is repeatedly divided in half through a binary search procedure. At each step $j ,$ a midpoint $q ( j )$ is calculated, and a subset of the users is used to privately estimate how many data points fall below this midpoint. This estimation is done using randomized response, which ensures that the algorithm complies with privacy guarantees. Depending on the results of this estimation, the algorithm updates the range: if too many data points are estimated to fall below the midpoint, the upper boundary is adjusted; if too few, the lower boundary is adjusted. Note that the subsets of users used at each step are disjoint, which assures that each user has at most one interaction with the aggregator. This process continues until either the estimate $\hat { Z } ^ { ( j ) }$ satisfies $| \hat { Z } ^ { ( j ) } - ( 1 - \alpha ) | \leq \Delta$ or the interval length $s _ { h i g h } - s _ { l o w }$ becomes smaller than a predefined threshold τ. In Theorem 5.4.2 we give the concrete sample complexity, under which we can estimate $\hat { Z } ^ { ( j ) }$ both privately and accurately in all iterations of the binary search. Once the aggregator finds the desired estimation, denoted ${ \hat { q } } ,$ the aggregator can now construct the prediction set as: $C _ { \hat { q } } ( x ) \ = \ \{ y \mid S ( x , y ) \leq \hat { q } \}$ . User’s side and aggregator’s side procedures depicted in Algorithm box

Algorithm 8 LDP-CP-S - Both sides   
1: Input: Calibration set $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } ,$ privacy parameter $\epsilon ,$ number of labels $k ,$ desired coverage   
$1 - \alpha , \Delta \quad$ , T   
2: Output: Threshold q ensuring private coverage $1 - \alpha - \Delta$ on true labels   
3: Initialize $\begin{array} { r } { j = 0 , n ^ { \prime } = \frac { n } { T } , s _ { h i g h } = Q _ { m a x } , s _ { l o w } = Q _ { m i n } } \end{array}$   
4: for $j = 1 , . . . , T$ do   
5: Select users $\mathcal { U } _ { . } ^ { ( j ) } = \{ j \cdot n ^ { \prime } + 1 , j \cdot n ^ { \prime } + 2 , . . . , ( j + 1 ) \cdot n ^ { \prime } \}$   
6: Set $\begin{array} { r } { q ^ { ( j ) } = \frac { s _ { l o w } + s _ { h i g h } } { 2 } } \end{array}$   
$7 { : }$ Individual users $i \in \mathcal { U } ^ { ( j ) } \colon ( 1 )$ compute their score $s = s ( x _ { i } , y _ { i } )$ , (2) sets $b _ { i } = 1 ( s < q ^ { ( j ) } )$ , (3)   
sends the aggregator $R R ( b _ { i } )$   
8: Obtain $\begin{array} { r } { \begin{array} { r } { \left. \sum ^ { \prime ( j ) } = \frac { e ^ { \epsilon } + 1 } { e ^ { \epsilon } - 1 } \cdot \frac { 1 } { n } \sum _ { i \in \mathcal { U } ^ { ( j ) } } R R ( b _ { i } ) - \frac { 1 } { e ^ { \epsilon } - 1 } \right.} \end{array}  } \end{array}$   
9: if $\begin{array} { r } { Z ^ { ( j ) } > ( 1 - \alpha ) + \frac { \Delta } { 2 } } \end{array}$ then $s _ { h i g h } = q ^ { ( j ) }$   
10: else if $\begin{array} { r } { Z ^ { ( j ) } < ( 1 - \bar { \alpha } ) - \frac { \Delta } { 2 } } \end{array}$ then $s _ { l o w } = q ^ { ( j ) }$   
11: else break   
12: Return $q ^ { ( j ) }$

## 5.4 Theoretical Guarantees

We now present our main theoretical results.

Theorem 5.4.1 (LDP-CP-L). Fix $\alpha , \delta , \Delta \ > \ 0$ . There exists an ϵ-local diferentially private algorithm that draws $\begin{array} { r } { n = O \left( \frac { \log \left( 1 / \delta \right) } { \Delta ^ { 2 } h ^ { 2 } } \right) } \end{array}$ exchangeable samples from any admissible distribution ${ \mathcal { D } } ,$ where $\begin{array} { r } { h \ = \ \frac { 1 - \beta } { 1 + \beta } } \end{array}$ , and $\begin{array} { r } { \beta = \frac { k } { k - 1 + e ^ { \epsilon } } } \end{array}$ , and, within at most $T = \lceil \log ( 1 / \tau ) \rceil$ iterations, produces an estimate $\hat { q }$ that satisfies $\mathrm { P r } \bigl ( y \in C _ { \hat { q } } ( x ) \bigr ) \ \geq \ 1 - \alpha - \Delta$ with probability at least $1 - \delta$ , where $1 - \alpha$ is the desired coverage and $\tau$ is an a-priori bound on the length of an interval that can hold ∆-probability mass.

Proof. The LDP-CP mechanism applies k-RR to the input data, ensuring ϵ-local diferential privacy [45]. Additionally, using the post-processing property of diferential privacy [20], it is safe to perform arbitrary computations on the output of a diferentially private mechanism - which maintains the privacy guarantees of the mechanism. Therefore, since $\mathrm { k - R R } ( \{ x _ { i } , y _ { i } \} _ { i = 1 } ^ { n _ { c a l } } )$ satisfies ϵ-local-diferential privacy, and because NACP is a deterministic or randomized post-processing function, it follows that $\mathrm { N A C P } ( \mathrm { k \mathrm { - } R R } ( \{ x _ { i } , y _ { i } \} _ { i = 1 } ^ { n _ { c a l } } ) )$ satisfies ϵ-local-diferential privacy.

The remainder of the proof focuses on the conformal prediction coverage guarantee bound. Given our $\mathrm { k } { \cdot } \mathrm { R R } ( \epsilon ) \ \epsilon { \mathrm { - } } \mathrm { L D P }$ mechanism, we derive $\begin{array} { r } { \beta = \frac { k } { k - 1 + e ^ { \epsilon } } } \end{array}$ . Substituting $\beta ,$ and $n _ { c a l }$ into 4.3.5 we obtain $\Delta ( n , \beta , \delta )$ such that $\mathrm { P r } \bigl ( y \in C _ { \tilde { q } } ( x ) \bigr ) \ \geq \ 1 - \alpha - \Delta$ □

Theorem 5.4.2 (LDP-CP-S). Fix $\alpha , \delta , \Delta \ > \ 0$ . There exists an ϵ-local diferentially private algorithm that draws $\begin{array} { r } { n = O \left( \frac { T } { \Delta ^ { 2 } } ( \frac { e ^ { \epsilon } + 1 } { e ^ { \epsilon } - 1 } ) ^ { 2 } \log ( T / \delta ) ) \right) } \end{array}$ exchangeable samples from any admissible distribution D and, within at most $T = \lceil \log ( 1 / \tau ) \rceil$ iterations, produces an estimate qˆ that satisfies

Pr $\left( y \in C _ { \hat { q } } ( x ) \right) \ \geq \ 1 - \alpha - \Delta$ with probability at least 1 − δ, where 1 − α is the desired coverage and τ is an a-priori bound on the length of an interval that can hold ∆-probability mass.

Proof. The privacy proof of a ϵ-LDP quantile binary search algorithm can be found in (author?) [26]. Users start by computing scores locally and then a local diferentially private quantile binary search algorithm is taken place [26]. ϵ-LDP follows immediately from the fact that the only time we access the data is via randomized response. The output of the algorithm is the (1 − α)’th quantile that one would obtain by applying conformal prediction to the clean data, yet it is recovered solely from the privatized (noisy) scores. □

## 5.5 Practical Considerations

## 5.5.1 LDP-CP-L vs. LDP-CP-S

The proposed methods, LDP-CP-L and LDP-CP-S, ofer distinct approaches for achieving local diferential privacy in conformal prediction, each is tailored to specific privacy setups and has diferent coverage guarantees. An in-depth understanding of their trade-ofs is essential to determine their suitability for various scenarios.

LDP-CP-L focuses on achieving local diferential privacy by perturbing the labels (y) while exposing the features (x) to the untrusted aggregator. This approach ensures privacy for the labels, which are typically considered more sensitive in many applications. This mechanism is particularly advanta geous in scenarios where users are resource-constrained, since they only need to perturb their labels locally before sending (x, y˜) to the aggregator. This design also keeps the model parameters and scoring functions secret from the users, because all computations related to the score are performed centrally. Consequently, LDP-CP-L is well-suited for real-world calibration datasets of moderate size, such as those containing several thousand records, where the additional noise introduced by the mechanism remains manageable. However, a notable drawback of LDP-CP-L is its sensitivity to the number of classes k in the dataset. As k increases, the calibration error (∆) grows, potentially compromising coverage guarantees for datasets with a large number of classes. Additionally, while the exposure of x provides practical utility by allowing centralized score computation, it introduces privacy concerns. This drawback can be partially mitigated by employing the shufle model of differential privacy, which adds an extra layer of anonymity to the users’ data (See further discussion in Section 5.5.2). Furthermore, privatizing x would result in a substantial insertion of noise, thereby leading to a significant degradation in the accuracy of the algorithms.

By contrast, LDP-CP-S achieves privacy at the score level by having users compute scores locally and perturb their responses before submitting them to the aggregator. This design ensures that both x and y remain private and are never exposed to the aggregator, making LDP-CP-S particularly suitable for scenarios where feature privacy is paramount. Unlike LDP-CP-L, the performance of LDP-CP-S is independent of the number of classes k. However, this approach imposes additional computational requirements on the users, who must perform local computations involving the model and scoring function. This requirement necessitates sharing the model with users, which might raise concerns about intellectual property or model misuse. Furthermore, LDP-CP-S requires a larger calibration dataset (n) to achieve a suficiently small calibration error, potentially limiting its applicability in scenarios with limited data availability.

In summary, the choice between LDP-CP-L and LDP-CP-S depends on the specific privacy and computational constraints of the application. LDP-CP-L is better suited for scenarios where label privacy is the primary concern, datasets are of moderate size (and larger). Its design minimizes user-side computations and protects the model from exposure. Conversely, LDP-CP-S is preferable when both feature and label privacy are critical, and when users have the computational resources to perform local scoring. Its robustness to the number of classes makes it a strong candidate for applications with a large class set, provided a suficiently large calibration dataset is available. By carefully considering these trade-ofs, practitioners can select the most appropriate method for their privacy-preserving conformal prediction tasks. In the experiment section, we report a numerical comparison of methods accuracy $( \Delta _ { L } , \Delta _ { S } )$ as a function of the calibration set size n and the number of classes k (Figure 5.3).

## 5.5.2 The Shufle Model of Diferential Privacy

The shufle model of diferential privacy enhances privacy guarantees by introducing an additional layer of anonymization between users and the data aggregator. In this model, each user applies a local randomizer to their data and then sends the output to a secure shufler, which permutes the messages uniformly at random before forwarding them to the aggregator. This anonymization mechanism breaks the association between individual users and their messages, thereby amplifying privacy guarantees beyond those attainable in the purely local model.

A notable benefit of the shufle model is its capacity for privacy amplification [12]: if each user applies an ε-LDP mechanism before sending their message, the efective privacy loss can be reduced to approximately $\epsilon ^ { \mathrm { e f f } } = \varepsilon / \sqrt { n }$ in the shufled output. This amplification enables stronger privacy guarantees with the same local noise or, conversely, allows for reduced noise to achieve a given privacy target—thereby improving utility in downstream tasks. This is particularly beneficial in regimes where moderately high local privacy levels $( \mathrm { e . g . } , \varepsilon > 1 )$ would otherwise impose significant performance degradation.

Given these advantages, our approach incorporates the shufle model to improve the privacy-utility trade-of of our mechanisms. While earlier versions of our framework considered exclusively the local model, we found that adopting the shufle model allows us to retain decentralization and local control while achieving significantly improved accuracy through amplification. Importantly, this integration does not alter the algorithmic structure of our mechanisms but rather augments their privacy analysis and performance guarantees under realistic assumptions of an honest-but-curious shufler. As such, the shufle model serves not only as a technical enhancement but also as a practical enabler of more efective private learning in our setting.

## 5.6 Experiments

In this section, we evaluate the capabilities of our LDP-CP algorithms on various medical and scenery imaging datasets, and address the utility-coverage tradeof.

Compared Methods. Our method takes an existing conformity score S and computes a threshold q that considers the injected noise level. We implemented two popular conformal prediction scores, namely APS [82] and HPS [93]. For each score S, we compared the following CP methods: (1) Not-Private-CP with coverage guarantee 1−α - using a validation set with clean labels (2) LDP-CP-{S,L} and (3) LDP-CP-{S,L}\* with coverage guarantee $1 - \alpha + \Delta$ . We share our code for reproducibility<sup>1</sup>.

Datasets. We present results on several publicly available medical imaging classification datasets [108]. TissuMNIST [107, 108]: This dataset contains 236,386 human kidney cortex cells, organized into 8 categories. Each gray-scale image is $3 2 \times 3 2 \times 7$ pixels. The 2D projections were obtained by taking the maximum pixel value along the axial-axis of each pixel, and were resized into 28 × 28 gray-scale images [101]. OrganSMNIST [108]: This dataset contains 25,221 images of abdominal CT in eleven classes. The images are 28 × 28 in size. Here, we used a train/validation/test split of 13,940/2,452/8,829 images. OrganAMNIST and OrganCMNIST are similar datasets, the diferences of Organ{A,C,S}MNIST are the views and dataset size. Lastly, OCTMNIST Retina OCT images dataset.

![](images/4fb4ec6e41ee581a416d7307bcce713f1e99cbc8983c9682b7868de339a8b3f9.jpg)

![](images/ff0fe323c2d41118722376caf7e30db17deb470078af65399af8e0599ad384c1.jpg)

![](images/2c16a8a6aa7d1e6268e855bf3abc5317216f17c35b1c798aa64e933f9d93bc8a.jpg)  
Figure 5.3: Comparison of $\Delta _ { l }$ and $\Delta _ { s }$ as a function of the number of classes k and dataset size $n ,$ for ϵ = 2, 4, 8.

![](images/efa4f1a1099a066c009bf5c54754d4a773a602affe11c376bb98b6b9d01fec7c.jpg)

![](images/7da1653879c9339790e3e63e98109849ed25e3a7673448f7d0a30aae54e24b30.jpg)  
(a) LDP-CP-L - ∆<sub>L</sub>

![](images/1c7c80e1405b94cae43ba975040c3455183a02c8d3c21cb51b20fedc4f9c7ee5.jpg)

![](images/fbbc8877c18f906c229fddc7e8222bc49aabefa5bdadc6f9f7b39226259e9391.jpg)  
(b) $\mathrm { L D P - C P - S \ - \ } \Delta _ { S }$

Figure 5.4: CP correction terms $\Delta _ { L } , \Delta _ { S }$ as a function of ϵ privacy parameter across diferent dataset configurations of n and k without the shufle model.  
![](images/2086071054cc0871b6d702f264ce9e8e9a5256f96c3beb8d8ef5d064ac307c1e.jpg)  
(a) LDP-CP-L - ∆<sub>L</sub>

![](images/3ab696927159e244620e174c21ab37d947273e98a59f1f2ad812894a476d4120.jpg)

![](images/a8910ed92a7f7b9a0acec72e668735d06bf9c06135c334cc3997e10906431efb.jpg)

![](images/36451668dcacdf38d528c3e004025f3876387af8af751197cf8a90760e4a6428.jpg)  
(b) LDP-CP-S - ∆<sub>S</sub>  
Figure 5.5: CP correction terms $\Delta _ { L } , \Delta _ { S }$ as a function of $\epsilon ^ { \mathrm { e f f } }$ privacy parameter across diferent dataset configurations of n and k with the shufle model.

Utility-Coverage Tradeof. Theorems 5.4.1 and 5.4.2 state that LDP-CP-L and LDP-CP-S are $\mathrm { \epsilon - L D P }$ with a conformal prediction coverage guarantee correction term of $\Delta _ { L } ( n , \beta ( \epsilon , k ) , \delta )$ and $\Delta _ { S } ( n , \epsilon , \delta )$ , which depend on the number of samples n, the privacy ϵ, δ, and $\Delta _ { L }$ also depends on the number of classes k. Figure 5.4 explores the trade-ofs of $\epsilon , n ,$ k on the conformal prediction correction terms $\Delta _ { L } , \Delta _ { S }$ , on diferent configurations and various medical image datasets [108] respectively. The results show both the finite-sample practicality and theoretical asymptotic performance of LDP-CP on existing medical datasets that are medium in size and simulated scenarios with growing n. For the majority of the evaluated datasets with $\epsilon = 3 .$ , LDP-CP-L maintains correction terms that become negligible for $\alpha = 0 . 1$ and higher (1 − α is the coverage). $\Delta _ { L } , \Delta _ { S }  0$ as $n \to \infty$ While as the number of samples grows $\Delta _ { S }$ goes to 0, for the evaluated medical datasets (with n fixed) $\Delta _ { S } > \Delta _ { L }$ , i.e. being inferior to $\Delta _ { L }$ , and is therefore more applicable to datasets with larger calibration sets. Figure 5.5 covers the same experiment setup, this time when the shufle model is deployed and presents the correction terms as a function of the efective privacy. The efective privacy $\epsilon ^ { \mathrm { e f f } }$ ranges between 0 and 0.2, which is much more practical then the $\epsilon \geq 3$ acquired without the shufle model.

Figure 5.3 compares $\Delta _ { S }$ and $\Delta _ { L }$ as a function of n, k, and ϵ. Results show that roughly speaking for $n \geq 1 0 ^ { 5 }  \Delta _ { S } \leq \Delta _ { L }$ . In addition, as the number of classes k increases (particularly for large values of k, e.g., 100 or 1000), LDP-CP-S dominates, whereas LDP-CP-L exhibits deteriorating performance.

Conformal Prediction Results. Table 5.1 reports the mean size and coverage when applying conformal prediction in a non ϵ-LDP setting (Not-Private-CP), and when satisfying the local differentially private property using $_ { \mathrm { L D P - C P - \{ S , L \} } }$ . The LDP-CP-{S-L} method has two variations where the first consists of calibration using 1 − α and getting $p ( y \in C _ { q } ( x ) ) \ \geq \ 1 - \alpha - \Delta$ (Theorems 5.4.1,5.4.2). The second variation needs to satisfy p(y $\in C _ { q } ( x ) ) \geq 1 - \alpha$ and therefore uses $1 - \alpha + \Delta$ in the calibration phase. For the experiment in Table 5.1 we used $\epsilon = 4$ and $\alpha = 0 . 1$ over 100 diferent data splits (seeds). Table 5.1 also provides the efective privacy $\epsilon ^ { \mathrm { e f f } }$ per dataset to show increased practicality when the shufle model is incorporated.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">HPS</td><td colspan="2">APS</td></tr><tr><td>size ↓</td><td>coverage (%)</td><td>size ↓</td><td>coverage (%)</td></tr><tr><td rowspan="5">OCTMNIST  $( \epsilon ^ { \mathrm { e f f } } = 0 . 0 3 8 )$ </td><td>Not-Private-CP</td><td> $2 . 5 7 \pm 0 . 0 3$ </td><td> $9 0 . 0 6 \pm 0 . 9 9$ </td><td> $2 . 6 1 \pm 0 . 0 3$ </td><td> $9 0 . 0 6 \pm 0 . 9 7$ </td></tr><tr><td>LDP-CP-L</td><td> $2 . 5 6 \pm 0 . 0 4$ </td><td> $8 9 . 9 9 \pm 1 . 0 1$ </td><td> $2 . 6 1 \pm 0 . 0 3$ </td><td> $9 0 . 0 2 \pm 1 . 0 1$ </td></tr><tr><td>LDP-CP-S</td><td> $2 . 5 8 \pm 0 . 0 4$ </td><td> $9 0 . 2 1 \pm 0 . 6 3$ </td><td> $2 . 6 7 \pm 0 . 0 8$ </td><td> $9 0 . 8 4 \pm 1 . 4 5$ </td></tr><tr><td>LDP-CP-L*</td><td> $2 . 7 6 \pm 0 . 0 4$ </td><td> $9 2 . 2 2 \pm 0 . 9 2$ </td><td> $2 . 7 9 \pm 0 . 0 3$ </td><td> $9 2 . 2 8 \pm 0 . 8 7$ </td></tr><tr><td>LDP-CP-S*</td><td> $2 . 9 7 \pm 0 . 0 7$ </td><td> $9 4 . 3 8 \pm 0 . 8 1$ </td><td> $2 . 9 9 \pm 0 . 0 6$ </td><td> $9 4 . 3 5 \pm 0 . 7 0$ </td></tr><tr><td rowspan="5">TissueMNIST  $( \epsilon ^ { \mathrm { e f f } } = 0 . 0 2 6 )$ </td><td>Not-Private-CP LDP-CP-L</td><td> $\overline { { 5 . 5 5 \pm 0 . 0 2 } }$   $5 . 5 4 \pm 0 . 0 2$ </td><td> $\overline { { 9 0 . 0 0 \pm 0 . 2 4 } }$ </td><td> $\overline { { 5 . 5 8 \pm 0 . 0 2 } }$   $5 . 5 8 \pm 0 . 0 2$ </td><td> $\overline { { 8 9 . 9 6 \pm 0 . 2 4 } }$   $8 9 . 9 7 \pm 0 . 2 7$ </td></tr><tr><td>LDP-CP-S</td><td> $6 . 1 2 \pm 0 . 0 1$ </td><td> $8 9 . 9 7 \pm 0 . 2 9$ </td><td></td><td></td></tr><tr><td>LDP-CP-L*</td><td> $5 . 7 1 \pm 0 . 0 2$ </td><td> $9 5 . 3 5 \pm 0 . 0 7$ </td><td> $5 . 6 1 \pm 0 . 0 9$ </td><td> $9 0 . 3 2 \pm 0 . 9 1$ </td></tr><tr><td>LDP-CP-S*</td><td></td><td> $9 1 . 6 8 \pm 0 . 2 7$ </td><td> $5 . 7 6 \pm 0 . 0 2$ </td><td> $9 1 . 7 0 \pm 0 . 2 5$ </td></tr><tr><td>Not-Private-CP</td><td> $6 . 1 2 \pm 0 . 0 1$   $\overline { { 1 . 9 3 \pm 0 . 0 5 } }$ </td><td> $9 5 . 3 5 \pm 0 . 0 7$ </td><td> $5 . 8 3 \pm 0 . 0 5$ </td><td> $9 2 . 3 2 \pm 0 . 4 5$ </td></tr><tr><td rowspan="5">OrganSMNIST  $( \epsilon ^ { \mathrm { e f f } } = 0 . 0 8 0 )$ </td><td>LDP-CP-L</td><td> $1 . 8 8 \pm 0 . 0 7$ </td><td> $\overline { { 9 0 . 0 9 \pm 0 . 6 6 } }$   $8 9 . 4 9 \pm 0 . 9 3$ </td><td> $\overline { { 2 . 3 5 \pm 0 . 0 5 } }$   $2 . 3 0 \pm 0 . 0 9$ </td><td> $\overline { { 9 0 . 1 0 \pm 0 . 5 5 } }$   $8 9 . 6 3 \pm 0 . 9 1$ </td></tr><tr><td>LDP-CP-S</td><td> $1 . 6 1 \pm 0 . 2 1$ </td><td></td><td></td><td> $8 7 . 3 8 \pm 1 . 3 7$ </td></tr><tr><td>LDP-CP-L*</td><td> $2 . 7 7 \pm 0 . 2 2$ </td><td> $8 4 . 8 1 \pm 4 . 4 8$ </td><td> $2 . 0 9 \pm 0 . 0 7$ </td><td></td></tr><tr><td>LDP-CP-S*</td><td> $3 . 9 0 \pm 0 . 0 3$ </td><td> $9 5 . 3 5 \pm 0 . 7 4$ </td><td> $3 . 3 5 \pm 0 . 2 2$ </td><td> $9 5 . 4 5 \pm 0 . 7 4$ </td></tr><tr><td>Not-Private-CP</td><td> $\overline { { 1 . 1 9 \pm 0 . 0 2 } }$ </td><td> $9 7 . 7 5 \pm 0 . 0 6$ </td><td> $4 . 7 5 \pm 0 . 0 3$ </td><td> $9 8 . 4 0 \pm 0 . 0 7$ </td></tr><tr><td rowspan="5">OrganAMNIST  $( \epsilon ^ { \mathrm { e f f } } = 0 . 0 4 9 )$ </td><td>LDP-CP-L</td><td> $1 . 3 1 \pm 0 . 0 0$ </td><td> $\overline { { 8 9 . 9 9 \pm 0 . 4 6 } }$ </td><td> $\overline { { 1 . 6 1 \pm 0 . 0 2 } }$ </td><td> $\overline { { 9 0 . 0 2 \pm 0 . 3 8 } }$ </td></tr><tr><td>LDP-CP-S</td><td></td><td> $9 2 . 1 7 \pm 0 . 0 9$ </td><td> $1 . 6 0 \pm 0 . 0 3$ </td><td> $8 9 . 9 4 \pm 0 . 5 4$ </td></tr><tr><td></td><td> $1 . 1 5 \pm 0 . 0 5$ </td><td> $8 8 . 8 9 \pm 0 . 9 6$ </td><td> $1 . 6 7 \pm 0 . 2 1$ </td><td> $9 0 . 3 5 \pm 3 . 1 0$ </td></tr><tr><td>LDP-CP-L*</td><td> $1 . 4 3 \pm 0 . 0 3$ </td><td> $9 3 . 6 2 \pm 0 . 3 4$ </td><td> $1 . 8 9 \pm 0 . 0 5$ </td><td> $9 3 . 5 1 \pm 0 . 5 1$ </td></tr><tr><td> $\mathrm { L D P - C P - S ^ { \ast } }$ </td><td> $1 . 8 8 \pm 0 . 1 9$ </td><td> $9 6 . 5 2 \pm 0 . 8 2$ </td><td> $2 . 4 4 \pm 0 . 1 9$ </td><td> $9 6 . 8 0 \pm 0 . 6 0$ </td></tr><tr><td rowspan="5">OrganCMNIST  $( \epsilon ^ { \mathrm { e f f } } = 0 . 0 8 1 )$ </td><td> $\overline { { { \mathrm { N o t } } { \mathrm { - P r i v a t e } } { \mathrm { - } } { \mathrm { C P } } } }$ </td><td> $\overline { { 1 . 1 8 \pm 0 . 0 3 } }$ </td><td> $\overline { { 8 9 . 9 9 \pm 0 . 7 1 } }$ </td><td> $\overline { { 1 . 5 6 \pm 0 . 0 3 } }$ </td><td> $\overline { { 9 0 . 0 2 \pm 0 . 6 5 } }$ </td></tr><tr><td>LDP-CP-L</td><td> $1 . 3 0 \pm 0 . 0 0$ </td><td> $9 1 . 9 6 \pm 0 . 1 3$ </td><td> $1 . 5 2 \pm 0 . 0 4$ </td><td> $8 9 . 3 6 \pm 0 . 9 1$ </td></tr><tr><td>LDP-CP-S</td><td> $0 . 9 5 \pm 0 . 0 8$ </td><td> $8 2 . 4 4 \pm 2 . 7 4$ </td><td> $1 . 3 9 \pm 0 . 0 7$ </td><td> $8 6 . 5 9 \pm 2 . 4 4$ </td></tr><tr><td> $\mathrm { L D P \mathrm { - C P \mathrm { - } L ^ { \ast } } }$ </td><td> $1 . 6 3 \pm 0 . 1 0$ </td><td> $9 5 . 2 1 \pm 0 . 7 4$ </td><td> $2 . 0 5 \pm 0 . 1 3$ </td><td> $9 5 . 2 1 \pm 0 . 8 0$ </td></tr><tr><td> $\mathrm { L D P - C P - S ^ { \ast } }$ </td><td> $2 . 4 7 \pm 0 . 0 2$ </td><td> $9 8 . 1 8 \pm 0 . 0 6$ </td><td> $2 . 9 0 \pm 0 . 0 4$ </td><td> $9 8 . 1 5 \pm 0 . 1 0$ </td></tr></table>

Table 5.1: Calibration results for HPS and APS conformal scores across various datasets, using $\epsilon = 4 , \epsilon ^ { \mathrm { e f f } } = \frac { \epsilon } { \sqrt { n } }$ , and $\alpha = 0 . 1$ on 100 diferent seeds.

![](images/0c85f26597c7c5f7eee5288260459160c371177e535d7eab7a295ba88806f32a.jpg)

![](images/11545d6d4c943688f4f16233381bb789ceb0a30f19d80762613eb762c28eaa41.jpg)  
Figure 5.6: Size of prediction set (left) and coverage (right) as a function of the privacy ϵ (bottom x-axis) and efective privacy $\epsilon ^ { \mathrm { e f f } }$ (top x-axis). We show the (mean ± std) on TissueMNIST and APS score.

Next, we experimented with diferent values of $\epsilon ,$ namely $\epsilon \in [ 1 , 8 ]$ . Recall that with the shufle model the efective privacy is $\epsilon ^ { \mathrm { e f f } } = \frac { \epsilon } { \sqrt { n } }$ . Figure 5.6 shows the coverage, size of the prediction set, and efective privacy on the TissueMNIST dataset. As expected as ϵ grows, $\Delta$ decreases and CP results get closer to the Non-Private-CP. In addition, LDP-CP-L performs on a par with the Non-Private-CP for all $\epsilon { \mathrm { ^ { \circ } s } } ,$ , and LDP-CP-S for $\epsilon ^ { \mathrm { e f f } } \geq 0 . 0 3 \ ( \epsilon \geq 4 )$ , showcasing their true applicability.

Chapter 6

## Unsupervised Target Domain Confidence Calibration

In this chapter, we explore the challenge of confidence calibration in Deep Neural Networks (DNNs) when transferring models from a source domain to an unlabeled target domain. While DNNs exhibit high accuracy in classification and detection tasks under well-supervised conditions, real-world applications often involve domain shifts where target domain labels are unavailable. Traditional calibration methods rely on labeled data to align model confidence with true probabilities, but these approaches fall short under unsupervised domain adaptation (UDA) scenarios. We introduce a novel method for directly calibrating model predictions on target domain data, demonstrating its efectiveness in addressing the pitfalls of existing approaches.

## 6.1 Problem Statement

Deep Neural Networks (DNN) have shown remarkable accuracy in tasks such as classification and detection when suficient data and supervision are present. In practical applications, it is crucial for models not only to be accurate, but also to indicate how much confidence users can have in their predictions. DNNs generate confidence scores that can serve as a rough estimate of the likelihood of correct classification, but these scores do not guarantee a match with the actual probabilities [33]. Neural networks tend to be overconfident in their predictions, despite having higher generalization accuracy, due to the possibility of overfitting on negative log-likelihood loss without afecting classification error [33, 49, 36]. A classifier is said to be calibrated with respect to a dataset sampled from a given distribution if its predicted probability of being correct matches its true probability. Various methods have been introduced to address the issue of over-confidence. Network calibration can be performed in conjunction with training (see e.g. [63, 64, 115]). Post-hoc scaling methods for calibration, such as Platt scaling [80], isotonic regression [110], and temperature scaling [33], are commonly employed. These techniques apply calibration as post-processing, using a hold-out validation set to learn a calibration map that adjusts the model’s confidence in its predictions to become better calibrated.

Table 6.1: Comparison of calibration methods for unsupervised domain adaptation (UDA).
<table><tr><td>Calibration Method</td><td>Designed for domain shift</td><td>Works without target label</td><td>Works on target data</td><td>Approach</td><td>Granularity</td></tr><tr><td>Temp. Scaling [33]</td><td>X</td><td>X</td><td>X</td><td></td><td>Instance level</td></tr><tr><td>CPCS[69], TransCal[97]</td><td>√</td><td>√</td><td>X</td><td>Importance weight estimation</td><td>Instance level</td></tr><tr><td>UTDC (proposed)</td><td>√</td><td>√</td><td>√</td><td>Estimates target accuracy</td><td>Dataset level</td></tr></table>

The implementation of deep learning systems on real-world problems is hindered by the decrease in performance when a network trained on data from one domain is applied to data from a diferent domain where the distribution of features changes across domains (see e.g. [61]). This is known as the domain shift problem. In an Unsupervised Domain Adaptation (UDA) setup we assume the availability of data from the target domain but without annotation. There is a plethora of UDA methods based on strategies such as adversarial training methods that aim to align the distributions of the source and target domains [27], or self-training algorithms based on computing pseudo labels for the target domain data [120].

In the following section we tackle the problem of calibrating predicted probabilities when transferring a trained model from a source domain to a target domain without any given labels.

Our major contributions include the following:

• We show that current UDA calibration methods which are all based on the source domain data, rely on an overly optimistic estimation of the target accuracy. Thus they can’t well handle the domain shift problem.

• We propose a calibration method that is directly applied to the target domain data, based on a realistic estimation of the accuracy of the adapted model on the target domain.

• We show that previously proposed UDA calibration methods don’t work at all and thus in this study we propose the first efective method for calibrating a network obtained by an unsupervised domain adaptation.

The study described in this chapter was published in [75].

## 6.2 Calibration on the Target Domain in Unsupervised Domain Adaptation

Our method involves calibrating the adapted network directly on the target data. While applying the network on the target domain data allows us to compute its confidence, we cannot determine its accuracy. Thus, the challenge is to find a reliable estimate of the network accuracy on the target domain. Our approach is based on the observation that when calibrating by minimizing the adaECE score, we do not need to know whether each individual prediction is correct. Instead, we only need to determine the mean accuracy for each bin. Fortunately, there are techniques which given a trained network, can estimate the network accuracy on data samples from a new domain without access to their labels [16, 32, 28, 109].

We next suggest a simple, intuitive, and very efective method that calibrates the network directly on the target domain. We first compute the overall network accuracy on the source data $A _ { \mathrm { s o u r c e } }$ and estimate the network accuracy on the target domain (e.g, using [16]). Denote the estimated target accuracy by $\tilde { A } _ { \mathrm { t a r g e t } }$ . Next, we divide the source data into M equal-size bins according to their confidence values and compute the corresponding network accuracy $A _ { \mathrm { s o u r c e } , m }$ at each bin m. We also divide the target data into M equal-size bins according to their confidence values and estimate the binwise accuracy of the target $A _ { \mathrm { t a r g e t } , m }$ by rescaling the binwise accuracy on the source domain in the following way:

$$
\tilde { A } _ { \mathrm { t a r g e t } , m } = A _ { \mathrm { s o u r c e } , m } \cdot \frac { \tilde { A } _ { \mathrm { t a r g e t } } } { A _ { \mathrm { s o u r c e } } } , \qquad m = 1 , . . . , M .\tag{6.1}
$$

In the next section, we empirically show that the accuracy ratio between source and target is indeed similar across the calibration bins. The estimated network accuracy on the target data $\tilde { A } _ { \mathrm { t a r g e t } }$ obtained by an unsupervised adaptation is usually lower than its accuracy on the source data $A _ { \mathrm { s o u r c e } }$ Thus, this accuracy rescaling provides a more realistic estimation of the bin-wise network average accuracy on the target data. The accuracy ratio $\tilde { A } _ { \mathrm { t a r g e t } } / A _ { \mathrm { s o u r c e } }$ indicates the size of the domain gap or the dificulty of the adaptation task [121].

Let $\boldsymbol { C } _ { t a r g e t , m }$ be the bin-wise network average confidence values computed on the target data. Substituting the estimated accuracy term, based on the source labeled data (6.1) into the adaECE definition, yields the following adaECE measure for the target domain in a UDA setup:

$$
\mathrm { U D A - a d a E C E } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left| \tilde { A } _ { \mathrm { t a r g e t } , m } - C _ { \mathrm { t a r g e t } , m } \right| .\tag{6.2}
$$

For each calibration method whose parameters can be found by minimizing the adaECE measure,

Algorithm 9 Unsupervised Target Domain Calibration (UTDC)   
input: A labeled validation set from the source domain, an unlabeled dataset from the target   
domain, and a k-class classifier that was adapted to the target domain.   
- Compute the source accuracy $A _ { \mathrm { s o u r c e } }$ and estimate the target accuracy $\tilde { A } _ { \mathrm { t a r g e t } }$   
using a target accuracy estimation technique.   
- Divide the source points into M equal size sets based on their confidence and   
compute the binwise mean accuracy: $A _ { \mathrm { s o u r c e } , m } , m = 1 , . . . , M$   
- Divide the target points into M equal size sets $B _ { 1 } , . . . , B _ { M }$ based on their confidence.   
for each candidate value of $T$ do   
- Compute the binwise mean confidence on the target:   
$C _ { \mathrm { t a r g e t } , m } ( T ) = \frac { 1 } { | B _ { m } | } \sum _ { x \in B _ { m } } \operatorname* { m a x } _ { i = 1 } ^ { k } \frac { \exp ( z _ { x , i } / T ) } { \sum _ { j = 1 } ^ { k } \exp ( z _ { x , j } / T ) }$ $m = 1 , . . . , M .$   
s.t. $z _ { x , 1 } , . . . , z _ { x , k }$ are the logits computed by the network that is fed by $x \in B _ { m }$   
- Compute the adaECE score as a function of $T \colon$   
$\mathrm { U D A - a d a E C E } ( T ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left| A _ { \mathrm { s o u r c e } , m } \times \frac { \tilde { A } _ { \mathrm { t a r g e t } } } { A _ { \mathrm { s o u r c e } } } - C _ { \mathrm { t a r g e t } , m } ( T ) \right|$   
output: The optimal temperature: $\hat { T } = \arg \operatorname* { m i n } _ { T } \mathrm { U D A - a d a E C E } ( T )$

we can form a UDA variant in which UDA-adaECE is minimized instead of adaECE. Examples of these calibration methods include Temperature Scaling (TS), Vector Scaling, Matrix Scaling [33], Mix-n-Match [113], Weight Scaling [25], and others.

We next demonstrate the UDA calibration principle in the case of TS calibration. We can determine the temperature that minimizes the UDA-adaECE measure (6.2) by conducting a grid search on the possible values.

Given the division of the target data into bins, we can compute the binwise average confidence after temperature calibration by T on the target $C _ { \mathrm { t a r g e t } , m } ( T )$ . We can then define the following temperature-dependent adaECE scores:

$$
\mathrm { U D A \mathrm { - } a d a E C E } ( T ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left| \tilde { A } _ { \mathrm { t a r g e t } , m } - C _ { \mathrm { t a r g e t } , m } ( T ) \right| .\tag{6.3}
$$

The optimal temperature is thus obtained by applying a grid search to find T that minimizes UDA-adaECE(T). The proposed Unsupervised Target Domain Calibration (UTDC) algorithm is summarized in Algorithm Box 9.

A major component of the UTDC is estimating the target domain accuracy. based on unlabeled target domain data. We next describe several recently suggested estimation algorithms. Deng et al. [16] suggested learning a dataset-level regression problem. The first step is to augment the source domain validation set, denoted by $D _ { s } .$ , using various visual transformations such as resizing, cropping, horizontal and vertical flipping, Gaussian blurring, and others. We then create n metadatasets, denoted as $D _ { 1 } , . . . , D _ { n }$ (in our implementation we set $n = 5 0 )$ . This process preserves the labels so we can compute the model’s accuracy on these datasets, denoted by $a _ { 1 } , . . . , a _ { n }$ Each dataset $D _ { i }$ is represented as a Gaussian distribution using its mean vector $\mu _ { i }$ and its diagonal covariance matrix $\Sigma _ { i }$ . Let $F _ { i }$ be the Fréchet distance [18] between the Gaussian representations of $D _ { s }$ and $D _ { i }$ $F _ { i }$ measures the domain gap between the original dataset $D _ { s }$ and $D _ { i }$ . Next, a linear regression model is fitted to the dataset $( F _ { 1 } , a _ { 1 } ) , . . . , ( F _ { n } , a _ { n } )$ in the form of $\hat { a } = w \cdot F + b$ . Finally, the linear regression model is employed to predict the accuracy of the network on the unlabeled data from the target domain. Another method is Average Thresholded Confidence (ATC) [28] which first selects a threshold t whose error in the source domain matches the expected number of points whose confidence is below t. Next, ATC predicts the error on the target domain which is expressed as the fraction of unlabeled points that obtain a confidence value below that threshold t. Let ${ \hat { p } } ( x ) = \operatorname* { m a x } _ { i } ( p = i | x )$ be the network confidence and let $\hat { y }$ be the network prediction. A threshold t is calculated to satisfy the equality $\begin{array} { r } { E _ { x \sim s o u r c e } 1 _ { \{ \hat { p } ( x ) \leq t \} } = E _ { ( x , y ) \sim s o u r c e } 1 _ { \{ \hat { y } \neq y \} } } \end{array}$ . The estimated target accuracy is the expectation $E _ { x \sim t a r g e t } 1 _ { \{ \hat { p } ( x ) \leq t \} }$ . Finally, the Projection Norm (PN) method [109] uses the model predictions to pseudo-label the test samples and then trains a new model on the pseudo-labels. The discrepancy between the parameters of the new and original models yields the predicted error of the target domain data. In Section 6.3 we compare the UTDC’s calibration performance when using each of the target accuracy prediction methods described above.

## 6.3 Experiments

In this section, we evaluate the capabilities of our UTDC technique to calibrate a network on a target domain after applying a UDA procedure.

Compared methods. We compared our method to six baselines: (1) Uncalibrated - The adapted classifier as is, without any post-hoc calibration; (2-4) Source-TS, Source-VS and source-MS - The adapted network was calibrated by either Temperature Scaling (TS), Vector Scaling (VS) or Matrix scaling (MS) [33] using the labeled validation set of the source domain; (5) CPCS [69], and (6) TransCal [97], importance weighted UDA calibrators. We also report Oracle results where TS calibration was applied to the labeled data from the target domain (denoted by Target-TS) and an Oracle version of our approach (denoted by UTDC\*) where we used the exact accuracy of the adapted model on the target data instead of estimating it.

Datasets. We report experiments on four standard real-world domain adaptation benchmarks, Ofice-home [92], Ofice-31 [84], VisDa-2017 [72], and DomainNet [71]. Ofice-home includes four domains - Art, Real-World, Clipart and Product, represented as A, R, C, and P in the experiments. Ofice-31 contains three domains - Amazon, Webcam and DSLR, denoted A, W, and D. VisDa-2017 is a simulation-to-real dataset for domain adaptation with over 280,000 images across 12 categories. DomainNet has six domains - Clipart, Infograph, Painting, Quickdraw, Real and Sketch, denoted C, I, P, Q, R, and S.

Table 6.2: AdaECE results on Ofice-home (with the lowest in bold) on various UDA classification tasks and models with diferent calibration methods.
<table><tr><td>UDA</td><td>Method</td><td>A → R</td><td>A → C</td><td> $A  P$ </td><td>C → R</td><td>C → P</td><td>C → A</td><td>P → R</td><td>P → C</td><td>P → A</td><td>Avg</td></tr><tr><td rowspan="11">CDAN+E</td><td>Uncalibrated</td><td>22.23</td><td>42.62</td><td>30.49</td><td>25.18</td><td>28.25</td><td>33.69</td><td>20.32</td><td>40.46</td><td>38.85</td><td>31.34</td></tr><tr><td>Source-TS</td><td>8.09</td><td>24.43</td><td>14.89</td><td>10.00</td><td>14.17</td><td>13.85</td><td>11.14</td><td>27.42</td><td>26.60</td><td>16.73</td></tr><tr><td>Source-VS</td><td>10.54</td><td>27.54</td><td>19.51</td><td>12.12</td><td>14.65</td><td>15.78</td><td>11.27</td><td>31.55</td><td>27.46</td><td>18.94</td></tr><tr><td>Source-MS</td><td>28.62</td><td>47.87</td><td>35.74</td><td>31.62</td><td>31.54</td><td>40.43</td><td>23.59</td><td>43.90</td><td>40.56</td><td>35.99</td></tr><tr><td>CPCS</td><td>15.84</td><td>49.78</td><td>23.42</td><td>14.02</td><td>16.60</td><td>18.45</td><td>6.31</td><td>49.21</td><td>25.62</td><td>24.36</td></tr><tr><td>TransCal</td><td>6.01</td><td>27.30</td><td>9.46</td><td>16.67</td><td>16.81</td><td>21.69</td><td>19.90</td><td>41.23</td><td>39.71</td><td>22.09</td></tr><tr><td>UTDC</td><td>4.46</td><td>9.74</td><td>7.53</td><td>8.36</td><td>5.91</td><td>8.08</td><td>10.45</td><td>7.46</td><td>9.37</td><td>7.93</td></tr><tr><td>UTDC*</td><td>4.30</td><td>5.93</td><td>7.41</td><td>7.85</td><td>4.62</td><td>10.16</td><td>10.76</td><td>4.55</td><td>9.54</td><td>7.24</td></tr><tr><td>Target-TS</td><td>3.97</td><td>5.05</td><td>7.19</td><td>4.07</td><td>4.39</td><td>7.07</td><td>2.32</td><td>4.39</td><td>8.57</td><td>5.22</td></tr><tr><td>Uncalibrated</td><td>19.90</td><td>39.19</td><td>26.75</td><td>24.47</td><td>26.33</td><td>33.53</td><td>20.25</td><td>40.06</td><td>39.25</td><td>29.97</td></tr><tr><td rowspan="8">DANN+E</td><td>Source-TS</td><td>6.90</td><td>19.80</td><td>7.93</td><td>6.54</td><td>7.01</td><td>16.01</td><td>15.68</td><td>27.87</td><td>30.97</td><td>15.41</td></tr><tr><td>Source-VS</td><td>10.15</td><td>25.83</td><td>15.31</td><td>12.13</td><td>10.70</td><td>17.90</td><td>14.69</td><td>32.40</td><td>31.64</td><td>18.97</td></tr><tr><td>Source-MS</td><td>30.78</td><td>52.03</td><td>38.39</td><td>35.44</td><td>35.45</td><td>44.21</td><td>26.40</td><td>45.87</td><td>43.33</td><td>39.10</td></tr><tr><td>CPCS</td><td>13.90</td><td>50.16</td><td>21.32</td><td>3.62</td><td>7.25</td><td>34.74</td><td>25.86</td><td>22.66</td><td>27.97</td><td>23.05</td></tr><tr><td>TransCal</td><td>7.21</td><td>27.42</td><td>12.36</td><td>17.81</td><td>15.43</td><td>29.93</td><td>24.64</td><td>46.61</td><td>45.83</td><td>25.25</td></tr><tr><td>UTDC</td><td>4.14</td><td>5.86</td><td>5.47</td><td>10.28</td><td>3.89</td><td>6.67</td><td>15.33</td><td>5.70</td><td>12.65</td><td>7.78</td></tr><tr><td>UTDC*</td><td>2.68</td><td>4.70</td><td>4.37</td><td>8.55</td><td>4.00</td><td>4.53</td><td>14.60</td><td>3.97</td><td>6.16</td><td></td></tr><tr><td>Target-TS</td><td>2.68</td><td>2.76</td><td>3.67</td><td>2.24</td><td>3.16</td><td>2.99</td><td>1.15</td><td>1.62</td><td>4.55</td><td>5.95 2.76</td></tr><tr><td rowspan="8">DANN</td><td>Uncalibrated</td><td>16.82</td><td>31.28</td><td>23.11</td><td>17.22</td><td>20.46</td><td>27.38</td><td>15.88</td><td>33.81</td><td>30.13</td><td>24.01</td></tr><tr><td>Source-TS</td><td>6.33</td><td></td><td></td><td></td><td></td><td>15.82</td><td></td><td>29.09</td><td></td><td></td></tr><tr><td>Source-VS</td><td>10.03</td><td>16.41</td><td>13.22</td><td>2.83</td><td>5.00</td><td></td><td>10.91</td><td></td><td>23.61</td><td>13.69</td></tr><tr><td>Source-MS</td><td>31.61</td><td>25.58</td><td>15.86</td><td>8.10</td><td>8.23 36.48</td><td>15.18 44.23</td><td>11.86 25.49</td><td>33.08</td><td>27.24</td><td>17.24</td></tr><tr><td>CPCS</td><td>8.89</td><td>50.68 33.56</td><td>41.31 19.99</td><td>34.23 25.29</td><td>9.62</td><td>12.82</td><td>16.87</td><td>44.75 27.49</td><td>40.17 45.93</td><td>38.77</td></tr><tr><td>TransCal</td><td>7.63</td><td>29.15</td><td>22.20</td><td>22.64</td><td>22.97</td><td>37.66</td><td>26.11</td><td>50.85</td><td>47.53</td><td>22.27</td></tr><tr><td>UTDC</td><td>5.15</td><td>4.87</td><td>11.24</td><td>8.63</td><td>5.23</td><td>15.08</td><td>18.62</td><td></td><td></td><td>29.64</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>12.62</td><td>11.23</td><td>10.30</td></tr><tr><td></td><td>UTDC* Target-TS</td><td>2.80 2.45</td><td>5.49 2.38</td><td>6.21 4.65</td><td>6.20 2.08</td><td>3.38 1.73</td><td>3.44 2.16</td><td>12.61 1.22</td><td>5.00 2.35</td><td>4.67 2.92</td><td>5.53 2.44</td></tr></table>

Implementation details. We followed the experiment setup described in [97] and used their code to implement CPCS and TransCal baselines. Following [97], we implemented three diferent UDA techniques; namely, DANN [27], DANN+E and CDAN+E [56]. The performance of more recent UDA models (e.g. [53, 44, 14]) on the target domain of the evaluated datasets is slightly better but is still much worse than the performance on the source domain. In most experiments we used the Meta target domain accuracy estimation [16] unless stated otherwise. We provide a code implementation of our method for reproducibility<sup>1</sup>.

Calibration results. Tables 6.2, 6.3, 6.4 and 6.5 report the calibration results (computed by adaECE with 15 bins) on Ofice-home, Ofice-31, VisDA, and DomainNet respectively. The results show that UTDC achieved significantly better results than the baseline methods on all tasks. The calibration obtained by previous IW-based methods was slightly better (but in some cases even worse) than a network with no calibration or a network that was calibrated on the source domain. In contrast, the adaECE score obtained by UTDC was almost as good as the adaECE obtained by an oracle that had access to the labels of the domain samples. In addition to the adaECE evaluation measure, Table 6.6 reports the average calibration results over all Ofice-home tasks, using three other calibration metrics: ECE, Negative Log-Likelihood (NLL) and Brier Score (BS) [7]. The same trends as above were observed.

Table 6.3: AdaECE results on Ofice-31 (with the lowest in bold) on various UDA classification tasks and models with diferent calibration methods.
<table><tr><td>UDA Method</td><td>Method</td><td>A→W</td><td>A→D</td><td>W→A</td><td> $W  D$ </td><td>D→A</td><td> $D \to W$ </td><td> $\operatorname { A v g }$ </td></tr><tr><td rowspan="10">CDAN+E</td><td>Uncalibrated</td><td>11.5</td><td>10.53</td><td>29.63</td><td>1.21</td><td>29.08</td><td>1.33</td><td>13.88</td></tr><tr><td>Source-TS</td><td>6.03</td><td>7.43</td><td>33.21</td><td>0.86</td><td>27.25</td><td>2.12</td><td>12.82</td></tr><tr><td>Source-VS</td><td>3.74</td><td>7.10</td><td>33.75</td><td>1.52</td><td>32.98</td><td>1.42</td><td>13.42</td></tr><tr><td>Source-MS</td><td>12.15</td><td>16.72</td><td>30.76</td><td>1.02</td><td>29.99</td><td>1.38</td><td>15.34</td></tr><tr><td>CPCS</td><td>9.67</td><td>12.66</td><td>33.47</td><td>1.11</td><td>28.16</td><td>2.18</td><td>14.54</td></tr><tr><td>TransCal</td><td>3.78</td><td>9.45</td><td>34.43</td><td>1.27</td><td>33.68</td><td>1.56</td><td>14.03</td></tr><tr><td>UTDC</td><td>4.19</td><td>5.18</td><td>5.15</td><td>1.20</td><td>5.14</td><td>2.18</td><td>3.84</td></tr><tr><td>UTDC*</td><td>3.82</td><td>5.18</td><td>5.09</td><td>1.13</td><td>5.36</td><td>2.18</td><td>7.13</td></tr><tr><td>Target-TS</td><td>3.44</td><td>4.67</td><td>3.32</td><td>0.75</td><td>3.20</td><td>0.89</td><td>2.71</td></tr><tr><td>Uncalibrated</td><td>13.05</td><td>13.55</td><td>28.29</td><td>0.87</td><td>27.15</td><td>1.68</td><td>14.10</td></tr><tr><td rowspan="8">DANN+E</td><td>Source-TS</td><td>5.18</td><td>9.29</td><td>26.93</td><td>1.31</td><td>26.44</td><td>2.44</td><td>11.93</td></tr><tr><td>Source-VS</td><td>4.63</td><td>8.24</td><td>36.64</td><td>0.87</td><td>31.35</td><td>1.55</td><td>13.88</td></tr><tr><td>Source-MS</td><td>18.01</td><td>14.02</td><td>31.10</td><td>1.09</td><td>28.51</td><td>1.51</td><td>15.71</td></tr><tr><td>CPCS</td><td>15.58</td><td>6.81</td><td>33.97</td><td>1.99</td><td>32.69</td><td>1.14</td><td>15.36</td></tr><tr><td>TransCal</td><td>7.98</td><td>5.63</td><td>34.53</td><td>1.57</td><td>31.12</td><td>1.59</td><td>13.74</td></tr><tr><td>UTDC</td><td>5.25</td><td>5.33</td><td>8.99</td><td>1.40</td><td>12.26</td><td>2.41</td><td>5.94</td></tr><tr><td>UTDC*</td><td>4.87</td><td>6.10</td><td>6.86</td><td>1.40</td><td>6.53</td><td>2.44</td><td>4.70</td></tr><tr><td>Target-TS</td><td>3.98</td><td>4.77</td><td>2.87</td><td>0.85</td><td>2.80</td><td>0.82</td><td>2.68</td></tr><tr><td rowspan="10">DANN</td><td>Uncalibrated</td><td>10.66</td><td>12.59</td><td>23.03</td><td>1.77</td><td>24.43</td><td>2.93</td><td>12.57</td></tr><tr><td>Source-TS</td><td>3.89</td><td>7.17</td><td>29.58</td><td>0.98</td><td>30.71</td><td>4.43</td><td>12.79</td></tr><tr><td>Source-VS</td><td>3.88</td><td>7.64</td><td></td><td>1.44</td><td></td><td>2.84</td><td></td></tr><tr><td>Source-MS</td><td></td><td></td><td>34.50</td><td></td><td>32.31</td><td></td><td>13.77</td></tr><tr><td>CPCS</td><td>21.06</td><td>24.70</td><td>28.81</td><td>1.35</td><td>28.45</td><td>1.30</td><td>17.61</td></tr><tr><td></td><td>16.96</td><td>10.10</td><td>33.69</td><td>2.61</td><td>35.39</td><td>4.80</td><td>17.26</td></tr><tr><td>TransCal</td><td>10.36</td><td>15.62</td><td>87.02</td><td>2.31</td><td>45.79</td><td>6.00</td><td>27.85</td></tr><tr><td>UTDC</td><td>3.71</td><td>8.70</td><td>5.14</td><td>2.61</td><td>9.26</td><td>5.23</td><td>5.78</td></tr><tr><td>UTDC*</td><td>5.04</td><td>7.52</td><td>5.54</td><td>2.61</td><td>12.25</td><td>6.54</td><td>6.58</td></tr><tr><td>Target-TS</td><td>3.53</td><td>4.12</td><td>2.79</td><td>0.97</td><td>3.19</td><td>1.94</td><td>2.76</td></tr></table>

Table 6.4: adaECE results on VisDA Task $S  R ,$ for various calibration methods.
<table><tr><td>Method</td><td>DANN</td><td>DANN+E</td><td>CDAN+E</td><td> $\operatorname { A v g }$ </td></tr><tr><td>Uncalibrated</td><td>33.23</td><td>31.79</td><td>29.88</td><td>31.63</td></tr><tr><td>Source-TS</td><td>26.54</td><td>18.66</td><td>23.38</td><td>34.29</td></tr><tr><td>Source-VS</td><td>38.22</td><td>36.96</td><td>28.48</td><td>34.55</td></tr><tr><td>Source-MS</td><td>41.19</td><td>38.17</td><td>30.87</td><td>36.74</td></tr><tr><td>CPCS</td><td>31.86</td><td>11.08</td><td>26.88</td><td>23.27</td></tr><tr><td>TransCal</td><td>43.52</td><td>35.93</td><td>36.71</td><td>38.72</td></tr><tr><td>UTDC</td><td>13.07</td><td>6.61</td><td>3.85</td><td>7.84</td></tr><tr><td>UTDC*</td><td>2.31</td><td>1.94</td><td>2.57</td><td>2.27</td></tr><tr><td> $\mathrm { T a r g e t - T S }$ </td><td>2.02</td><td>1.84</td><td>2.21</td><td>2.02</td></tr></table>

## 6.4 Analysis

We next illustrate and analyze several key features of the proposed method.

Table 6.5: adaECE results on DomainNet for various UDA classification tasks and models with diferent calibration methods
<table><tr><td>UDA</td><td>Method</td><td>S→R</td><td>S→P</td><td>P→R</td><td>P→S</td><td>R→S</td><td>R→P</td><td>Avg</td></tr><tr><td rowspan="9">CDAN+E</td><td>Uncalibrated</td><td>14.65</td><td>18.70</td><td>18.06</td><td>22.98</td><td>19.13</td><td>13.77</td><td>17.88</td></tr><tr><td>Source-TS</td><td>12.68</td><td>14.48</td><td>11.51</td><td>12.76</td><td>13.56</td><td>9.60</td><td>12.39</td></tr><tr><td>Source-VS</td><td>10.70</td><td>9.56</td><td>11.49</td><td>14.94</td><td>13.35</td><td>9.31</td><td>11.56</td></tr><tr><td>Source-MS</td><td>22.24</td><td>25.28</td><td>23.43</td><td>30.93</td><td>22.55</td><td>18.07</td><td>23.75</td></tr><tr><td>CPCS</td><td>9.41</td><td>11.20</td><td>13.26</td><td>17.06</td><td>17.16</td><td>11.86</td><td>13.32</td></tr><tr><td>TransCal</td><td>12.50</td><td>20.82</td><td>16.41</td><td>28.85</td><td>36.70</td><td>28.23</td><td>23.92</td></tr><tr><td>UTDC</td><td>6.06</td><td>5.17</td><td>6.48</td><td>4.75</td><td>8.85</td><td>8.32</td><td>6.61</td></tr><tr><td>UTDC*</td><td>5.07</td><td>6.78</td><td>4.86</td><td>3.56</td><td>5.19</td><td>6.86</td><td>5.38</td></tr><tr><td>Target-TS</td><td>1.31</td><td>1.35</td><td>2.18</td><td>1.39</td><td>1.25</td><td>1.07</td><td>1.42</td></tr><tr><td rowspan="9">DANN+E</td><td>Uncalibrated</td><td>15.03</td><td>17.77</td><td>17.57</td><td>24.54</td><td>21.08</td><td>16.63</td><td>18.77</td></tr><tr><td>Source-TS</td><td>10.12</td><td>12.20</td><td>10.31</td><td>11.75</td><td>11.76</td><td>10.69</td><td>11.14</td></tr><tr><td>Source-VS</td><td>9.71</td><td>14.25</td><td>11.85</td><td>19.42</td><td>16.88</td><td>12.15</td><td>14.04</td></tr><tr><td>Source-MS</td><td>23.68</td><td>28.77</td><td>24.18</td><td>35.03</td><td>24.94</td><td>20.91</td><td>26.25</td></tr><tr><td>CPCS</td><td>13.20</td><td>6.41</td><td>12.51</td><td>12.81</td><td>7.73</td><td>10.95</td><td>10.60</td></tr><tr><td>TransCal</td><td>14.56</td><td>19.85</td><td>16.14</td><td>29.19</td><td>34.98</td><td>28.96</td><td>23.95</td></tr><tr><td>UTDC</td><td>6.39</td><td>6.07</td><td>6.54</td><td>6.84</td><td>11.24</td><td>11.94</td><td>8.17</td></tr><tr><td>UTDC*</td><td>3.97</td><td>5.72</td><td>5.23</td><td>6.64</td><td>6.73</td><td>8.32</td><td>6.10</td></tr><tr><td>Target-TS</td><td>1.24</td><td>1.19</td><td>1.60</td><td>1.03</td><td>1.10</td><td>0.84</td><td>1.17</td></tr><tr><td rowspan="9">DANN</td><td>Uncalibrated</td><td>10.98</td><td>13.52</td><td>12.65</td><td>18.04</td><td>15.42</td><td>10.96</td><td>13.59</td></tr><tr><td>Source-TS</td><td>7.33</td><td>8.63</td><td>9.50</td><td>10.11</td><td>10.99</td><td>9.15</td><td>9.29</td></tr><tr><td>Source-VS</td><td>8.92</td><td>14.43</td><td>11.21</td><td>16.90</td><td>15.86</td><td>10.86</td><td>13.03</td></tr><tr><td>Source-MS</td><td>22.51</td><td>27.48</td><td>21.97</td><td>31.46</td><td>24.53</td><td>19.72</td><td>24.61</td></tr><tr><td>CPCS</td><td>7.02</td><td>7.37</td><td>14.60</td><td>15.83</td><td>15.42</td><td>8.88</td><td>11.52</td></tr><tr><td>TransCal</td><td>14.83</td><td>22.09</td><td>16.38</td><td>30.37</td><td>37.84</td><td>29.92</td><td>25.24</td></tr><tr><td>UTDC</td><td>5.82</td><td>5.84</td><td>6.30</td><td>9.24</td><td>5.80</td><td>7.53</td><td>6.76</td></tr><tr><td>UTDC*</td><td>4.34</td><td>5.46</td><td>4.71</td><td>7.34</td><td>6.53</td><td>6.81</td><td>5.87</td></tr><tr><td>Target-TS</td><td>1.07</td><td>1.25</td><td>1.06</td><td>0.90</td><td>1.53</td><td>1.61</td><td>1.24</td></tr></table>

Table 6.6: Calibration metrics results of various UDA calibration methods on the Ofice-home tasks.
<table><tr><td></td><td colspan="3">CDAN+E</td><td colspan="3">DANN+E</td><td colspan="3">DANN</td></tr><tr><td>method</td><td>BS</td><td>NLL</td><td>ECE</td><td>BS</td><td>NLL</td><td>ECE</td><td>BS</td><td>NLL</td><td>ECE</td></tr><tr><td>Uncalibrated</td><td>0.74</td><td>3.40</td><td>31.32</td><td>0.76</td><td>3.07</td><td>29.92</td><td>0.75</td><td>2.75</td><td>24.08</td></tr><tr><td>Source-TS</td><td>0.65</td><td>2.18</td><td>16.79</td><td>0.67</td><td>2.21</td><td>15.40</td><td>0.71</td><td>2.37</td><td>13.71</td></tr><tr><td>CPCS</td><td>0.71</td><td>3.48</td><td>24.46</td><td>0.72</td><td>3.08</td><td>23.12</td><td>0.76</td><td>2.87</td><td>22.37</td></tr><tr><td>TransCal</td><td>0.69</td><td>2.70</td><td>22.12</td><td>0.73</td><td>3.08</td><td>25.22</td><td>0.81</td><td>3.72</td><td>29.71</td></tr><tr><td>UTDC</td><td>0.62</td><td>1.95</td><td>8.01</td><td>0.64</td><td>2.01</td><td>7.81</td><td>0.69</td><td>2.26</td><td>10.35</td></tr><tr><td>UTDC*</td><td>0.62</td><td>1.95</td><td>7.21</td><td>0.63</td><td>1.99</td><td>5.94</td><td>0.68</td><td>2.18</td><td>5.53</td></tr><tr><td>Target-TS</td><td>0.61</td><td>1.92</td><td>5.41</td><td>0.63</td><td>1.96</td><td>2.72</td><td>0.68</td><td>2.14</td><td>2.78</td></tr></table>

![](images/a842eefdb8194b4e02ffa9c61624ecea9b87408692feba86820a85a1f86871f6.jpg)  
Figure 6.1: Average accuracy on Ofice-home tasks for the three UDA techniques (DANN, DANN+E, CDAN+E).

Accuracy gap between source and target. To gain a better understanding of the reasons why our method performs better than IW based methods, we first discuss the accuracy of the adapted models on the source and target domains. Figure 6.1 presents the accuracy on the source and target domains for three UDA techniques. It shows that even after adaptation to the target, the model’s performance on the source samples is consistently better than its performance on the target samples, especially in cases of large domain gaps.

Table 6.7: Computed temperature on various UDA Ofice-home tasks, and calibration methods using CDAN+E.
<table><tr><td>UDA</td><td>Method</td><td> $A  R$ </td><td> $A {  } C$ </td><td> $A \to P$ </td><td> $C {  } R$ </td><td> $C {  } P$ </td><td> $C {  } A$ </td><td> $P {  } R$ </td><td> $P {  } C$ </td><td> $P {  } A$ </td><td> $\operatorname { A v g }$ </td></tr><tr><td rowspan="6">CDAN+E</td><td>Source-TS</td><td>1.96</td><td>2.02</td><td>2.02</td><td>1.87</td><td>1.90</td><td>2.06</td><td>1.63</td><td>1.72</td><td>1.68</td><td>1.87</td></tr><tr><td>CPCS</td><td>1.46</td><td>0.57</td><td>1.49</td><td>1.68</td><td>1.75</td><td>2.05</td><td>1.93</td><td>0.50</td><td>1.73</td><td>1.46</td></tr><tr><td>TransCal</td><td>2.12</td><td>1.86</td><td>2.39</td><td>1.50</td><td>1.74</td><td>1.62</td><td>1.03</td><td>0.96</td><td>0.95</td><td>1.57</td></tr><tr><td>UTDC</td><td>2.27</td><td>2.90</td><td>2.91</td><td>1.97</td><td>2.44</td><td>2.54</td><td>1.67</td><td>2.93</td><td>2.89</td><td>2.50</td></tr><tr><td> $\overline { { \mathrm { U T D C ^ { * } } } }$ </td><td>2.29</td><td>3.21</td><td>2.68</td><td>2.00</td><td>2.62</td><td>2.30</td><td>1.65</td><td>3.41</td><td>2.90</td><td>2.56</td></tr><tr><td>Target-TS</td><td>2.36</td><td>3.61</td><td>2.73</td><td>2.42</td><td>2.73</td><td>2.81</td><td>2.24</td><td>3.49</td><td>3.37</td><td>2.86</td></tr></table>

![](images/4f1f0d35b9c509d30f86abcfde8f5c129b7fb606ff67de70a2290d46b400d6c5.jpg)  
(a) $\mathrm { C D A N + E }$

![](images/f383bffdcba9adecb737c84369351ad0a9238ef0c770e6c25a039006aa6e7cac.jpg)  
(b) DANN+E

![](images/37e4ded7a9d82f4778b1f6dce22b6042a704976fce0a2800ed069eed2e6ef583.jpg)  
(c) DANN  
Figure 6.2: adaECE results as a function of the correction ratio R on Ofice-Home, $A  C$ task.

Hence, using the network accuracy on the source to estimate the network’s accuracy on the target while minimizing the ECE measure is misleading because the over-optimistic accuracy estimation leads to a scaling temperature that is too small. Table 6.7 compares the optimal temperatures computed by the calibration methods. In all the baseline methods the computed calibration temperature was lower than the optimal value. This results in poorer calibration performance, as seen in Tables 6.2, 6.3, 6.4, and 6.5. By contrast, the temperature computed by all the UTDC variants was much closer to the optimal temperature computed by the Oracle method that had access to the target labels. Figure 6.1 also presents the estimated accuracy of the adapted model on the target domain. This estimation is close to the true accuracy. Thus, when it is combined with the confidence computed on the target domain, we obtain a calibrated mode.

Sensitivity of UTDC to the target accuracy prediction. UTDC is based on estimating the binwise average network accuracy on the target domain data from the labeled source domain data. This estimation is done by computing the ratio $\tilde { A } _ { \mathrm { t a r g e t } } / A _ { \mathrm { s o u r c e } }$ between the estimated target accuracy and the source accuracy. We next analyze the sensitivity of our calibration method to errors in estimating $A _ { \mathrm { t a r g e t } }$ . Let $R ( \mathrm { t r u e } ) = A _ { \mathrm { t a r g e t } } / A _ { \mathrm { s o u r c e } }$ and $R ( \mathrm { e s t i m a t e d } ) = \tilde { A } _ { \mathrm { t a r g e t } } / A _ { \mathrm { s o u r c e } }$ be the true and estimated ratio used by UTDC\* and UTDC respectively. In principle, any number

![](images/0ffeaede60dd582d132be1abe43c72ec4e1cb5f09d24bcd80f63a4ac92797d57.jpg)  
(a) CDAN+E

![](images/3dbe4931e97d6c07ea0be19108dbb22de59bc6627b7b4da5c1a85094cbc6089e.jpg)  
(b) DANN+E

![](images/8ef1077fb8c06bb1d0c38412da31d65a3074b54a118ee9571567102ace409a32.jpg)  
(c) DANN

Figure 6.3: Accuracy of k-th percentile source images based on their probability of being classified as target [97], compared to target accuracy (Ofice-home, $A  C )$  
![](images/eea37ded3bc30f25821b1451c045375460f5f2dc55c9aef6722131744ed52310.jpg)  
(a) CDAN+E

![](images/4893b8486fec1740ed3ba7c585ed5e2a69d8e55cca9cb2d3f91e819a7466647d.jpg)  
(b) DANN+E

![](images/62f1ce92475c4648d880c30f6da61474d8530cbeafcb4864860eee4daf640142.jpg)  
(c) DANN  
Figure 6.4: Accuracy per bin for source and target images. The results are shown on the Ofice-home $C  P$ task.

$0 < R$ can be used to obtain an estimation of the binwise target accuracy: $\tilde { A } _ { \mathrm { t a r g e t } , m } = A _ { \mathrm { s o u r c e } , m } \cdot R .$ We can thus find the temperature that minimizes the adaECE function on the target data as a function of $R \colon { \hat { T } } ( R ) = \arg$ min adaECE (T) where

$$
\mathrm { a d a E C E } _ { R } ( T ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } | A _ { \mathrm { s o u r c e } , m } \cdot R - C _ { \mathrm { t a r g e t } , m } ( T ) | .
$$

Figure 6.2 shows the adaECE measure on the target data after temperature scaling by ${ \hat { T } } ( R )$ as a function of the ratio R for the task Ofice-home $A  C$ . It shows that with the appropriate choice of R we can achieve the calibration level of the Oracle TS-target algorithm (the case where target labels are known). This means that the diference in accuracy is indeed the main reason for the calibration degradation caused by methods that try to calibrate the target domain using the source data. Specifically, as the ratio R drops towards $R ( \mathrm { t r u e } )$ , the adaECE improves and approaches the Oracle TS-target calibration. In addition, the adaECE reaches a minimum near R(true) and R(estimated). Finally, there is a range of correction ratios where UTDC is better by a large margin than other baselines, thus providing a tolerance for error and resilience in estimating $\tilde { A } _ { \mathrm { t a r g e t } }$

The problem with the IW assumption. We showed that our method achieves better results by explicitly addressing the accuracy gap between the source and target domains caused by the domain shift. Previous methods based on importance weights [69, 97] rely on re-weighting the source data based on their proximity to the target data, i.e., concentrating on source samples that resemble the target and attributing less weight to others. We computed the target similarity weights associated with each sample in the source validation set and divided them into 20% percentile subsets. Figure 6.3 shows the average accuracy of each group and the average target accuracy. It shows that the source accuracy is similar in all bins regardless of the similarity to the target. Thus the IW assumption that source samples that are classified as targets are more relevant for calibrating the target prediction is wrong.

Table 6.8: AdaECE results for variations of UTDC based on diferent methods of domain accuracy estimation.
<table><tr><td>Method</td><td>Office-home</td><td>Office-31</td><td>VisDA</td><td>DomainNet</td></tr><tr><td>Uncalibrated</td><td>28.44</td><td>13.51</td><td>31.63</td><td>16.74</td></tr><tr><td>UTDC-Meta[16]</td><td>8.67</td><td>6.96</td><td>7.84</td><td>7.18</td></tr><tr><td>UTDC-ATC[28]</td><td>10.12</td><td>7.47</td><td>5.68</td><td>8.01</td></tr><tr><td>UTDC-PN [109]</td><td>11.55</td><td>7.83</td><td>10.20</td><td>8.63</td></tr><tr><td>UTDC*</td><td>6.24</td><td>6.13</td><td>2.27</td><td>5.78</td></tr></table>

Table 6.9: Comparison of several target domain accuracy estimation methods measured by $| A C C ( T r u e ) - A C C ( E s t ) |$ .
<table><tr><td>Method</td><td>Office-home</td><td>Office-31</td><td>VisDA</td><td>DomainNet</td></tr><tr><td>Meta[28]</td><td>3.31</td><td>2.81</td><td>4.96</td><td>3.10</td></tr><tr><td>ATC[28]</td><td>5.05</td><td>3.37</td><td>3.48</td><td>4.25</td></tr><tr><td>PN [109]</td><td>6.26</td><td>4.85</td><td>6.30</td><td>5.91</td></tr></table>

Accuracy ratio across bins. Our method computes $\tilde { A } _ { \mathrm { t a r g e t } , m }$ by re-scaling $A _ { \mathrm { s o u r c e } , m }$ with the same ratio for all bins, as defined in 6.1. This estimation is based on the assumption that the accuracy ratio between the source and the target is similar across the bins. To illustrate the validity of this assumption, Figure 6.4 shows the accuracy of the adapted network at each bin, for the source and target data.

Diferent target accuracy estimation methods. Our UTDC method requires an estimation step of the target domain accuracy without labels. In all the experiments reported above we used the Meta method [16]. We next examine combining UTDC with two other methods for target domain accuracy estimation: ATC [28] and PN [109]. We implemented 3 variations of UTDC, dubbed UTDC-Meta, UTDC-ATC, and UDTC-PN based on the estimated target accuracy that was used. We also report results for $\mathrm { U T D C ^ { * } }$ based on the true target accuracy. Tables 6.8 and 6.9 present the average calibration results and the discrepancy between the estimated and actual accuracy, respectively. The results indicate that UTDC achieved the best calibration performance out of all the three target accuracy estimation methods examined, thus reinforcing the observed low sensitivity of UTDC to the precision of target accuracy predictions. This underscores the compatibility of UTDC with existing methods for network calibration under unsupervised domain shift. We also found that using UTDC-Meta yields better results, while UTDC-ATC exhibits improved performance and ease of implementation, since the ATC method is much simpler to implement and requires a small computational efort.

## Chapter 7

## Discussion

This chapter brings together the key findings and contributions of this dissertation, highlighting their implications for the broader field of machine learning. Throughout our research, we addressed criti cal challenges related to confidence calibration and uncertainty quantification, focusing on complex scenarios involving label noise, privacy constraints, and domain shifts. By developing novel methodologies and demonstrating their efectiveness across diverse settings, our work advances the reliability and interpretability of machine learning models in real-world applications. Here, we summarize our contributions, reflect on key insights, and outline promising directions for future research.

## 7.1 Summary of Contributions

In this dissertation, we explored multiple facets of confidence calibration and uncertainty quantification in machine learning, particularly under challenging real-world conditions such as label noise, privacy constraints, and domain shift. Our work contributes novel methodologies that enhance the robustness of existing calibration and conformal prediction (CP) frameworks, thereby addressing key limitations in the current literature.

First, we investigated confidence calibration in classification models trained with noisy labels, a critical issue in domains like medical imaging where obtaining accurate labels is often impractical. We demonstrated that traditional calibration methods are highly sensitive to label noise, even when network training itself remains relatively robust. To address this, we proposed a noise-aware calibration framework that efectively models label corruption using a noise transition matrix. Our results show that this approach achieves calibration performance comparable to that of clean labels, provided the noise model is well estimated.

Second, we extended the conformal prediction framework to handle label noise. We introduced a procedure that adjusts the calibration threshold based on a given noise model, allowing CP to maintain valid coverage while minimizing the size of prediction sets. We derived finite-sample coverage guarantees for the uniform noise case and showed that our method significantly improves over existing noisy CP approaches in terms of eficiency and prediction set size.

Third, we addressed confidence calibration under privacy constraints by developing two complementary CP approaches for locally diferentially private (LDP) settings. These methods, LDP-CP-L (label perturbation) and LDP-CP-S (score perturbation), provide valid uncertainty quantification while ensuring strong privacy guarantees for individual user data. Our results highlight the trade-ofs between privacy, computational feasibility, and model performance, ofering guidance for selecting the appropriate method in real-world applications.

Lastly, we studied network calibration in the context of unsupervised domain adaptation, where a model trained on a labeled source domain is deployed in a target domain with a diferent distribution. We showed that existing importance-weighting approaches fail to correct for domain shift in calibration. Instead, we proposed a method that directly calibrates using target-domain examples, leading to substantial improvements in calibration performance.

## 7.2 Key Insights and Implications

Sensitivity of Calibration to Label Noise. Our findings reveal that calibration procedures are far more sensitive to label noise than the training process itself. This observation underscores the need for noise-resilient calibration methods, as even a small fraction of incorrect labels in the validation set can severely degrade calibration quality. Our proposed approach, which explicitly accounts for label noise via a noise transition matrix, mitigates these efects and provides reliable confidence estimates despite label corruption.

Conformal Prediction Under Noisy Labels. We demonstrated that CP can be adapted to handle noisy labels by estimating a noise-free calibration threshold. Our method preserves valid coverage and improves eficiency compared to existing approaches. However, our analysis suggests that current coverage guarantees may be overly conservative, indicating room for further theoretical refinement. Additionally, we focused on noise models where label corruption is independent of input features; extending our method to more complex, feature-dependent noise processes remains an open challenge.

Privacy-Preserving Uncertainty Quantification. Our work on LDP-conformal prediction bridges the gap between uncertainty quantification and privacy protection. By leveraging randomized response mechanisms, we ensured valid CP coverage while preserving user privacy. A notable challenge for future work is integrating more sophisticated LDP techniques, such as RAPPOR, into our framework to enhance robustness and scalability. Furthermore, developing a method that preserves privacy without requiring local score computation remains an important open problem.

Confidence Calibration in Domain Adaptation. We showed that calibration methods relying on source-domain accuracy fail under domain shift due to diferences in true label distributions. Our approach, which calibrates directly on target-domain examples, outperforms existing methods. Extending this idea to non-parametric calibration techniques such as CP, as well as to regression and segmentation tasks, is a promising avenue for future research. Additionally, addressing confidence calibration in source-free adaptation—where access to source data is not available—remains an open challenge.

## 7.3 Future Research Directions

While our research has addressed several key challenges, many promising directions remain for future exploration:

Refining Theoretical Guarantees for Noisy Conformal Prediction. Our results suggest that existing theoretical bounds for CP under noisy labels may be overly conservative. Further research into tighter finite-sample guarantees could improve the eficiency of CP methods in noisy settings.

Advanced Noise Models in Calibration. Our methods assume that the noise transition matrix can be estimated with reasonable accuracy. However, in cases of highly unbalanced class distributions or feature-dependent noise, estimating this matrix remains challenging. Developing robust estimation techniques for such cases is an important open problem.

Extending Privacy-Preserving CP Methods. Our current LDP-CP framework is limited by the assumption that users can either perturb labels or scores locally. A more stringent privacy setting, where both the input features and labels must remain private, presents a fundamental challenge. Designing methods that operate under such conditions while maintaining valid coverage is a crucial area for future work.

Calibration in Source-Free Adaptation. In domain adaptation, we demonstrated that calibration benefits from access to target-domain examples. However, a more challenging setting is source-free adaptation, where only unlabeled target data is available. Developing efective calibration methods under such constraints is a key research direction.

Applying Calibration Strategies to Other Tasks. Our work focused on classification tasks, but the principles developed here could extend to regression and segmentation problems. Exploring calibration techniques for these settings, particularly under domain shift and noisy labels, could have significant practical impact.

## 7.4 Conclusion

This dissertation advances the understanding and application of confidence calibration and uncertainty quantification in machine learning. By addressing challenges in label noise, privacy, and domain shift, we developed robust methodologies that improve calibration performance across diverse settings. Our work not only enhances the reliability of machine learning models but also opens new avenues for research in uncertainty estimation under real-world constraints. We hope that our contributions will inspire further studies in these areas, ultimately leading to more trustworthy and interpretable AI systems.

## Bibliography

[1] Anastasios N Angelopoulos, Stephen Bates, et al. Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning, 16(4):494–591, 2023.

[2] Anastasios N. Angelopoulos, Stephen Bates, Jitendra Malik, and Michael I Jordan. Uncertainty sets for image classifiers using conformal prediction. International Conference on Learning Representations (ICLR), 2021.

[3] Anastasios N Angelopoulos, Stephen Bates, Tijana Zrnic, and Michael I Jordan. Private prediction sets. arXiv preprint arXiv:2102.06202, 2022.

[4] Apple. Learning with privacy at scale, 2017. Accessed: [Insert Access Date].

[5] Amos Beimel, Kobbi Nissim, and Uri Stemmer. Private learning and sanitization: Pure vs. approximate diferential privacy. In International Workshop on Approximation Algorithms for Combinatorial Optimization, pages 363–378. Springer, 2013.

[6] Alan Joseph Bekker and Jacob Goldberger. Training deep neural-networks based on unreliable labels. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2682–2686, 2016.

[7] Glenn W Brier. Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1):1–3, 1950.

[8] Ling-Hao Chen, He Li, and Wenhao Yang. Anomman: Detect anomaly on multi-view attributed networks. arXiv preprint arXiv:2201.02822, 2022.

[9] Pengfei Chen, Guangyong Chen, Junjie Ye, Pheng-Ann Heng, et al. Noise against noise: stochastic label noise helps combat inherent label noise. In International Conference on Learning Representations (ICLR), 2021.

[10] De Cheng, Tongliang Liu, Yixiong Ning, Nannan Wang, Bo Han, Gang Niu, Xinbo Gao, and Masashi Sugiyama. Instance-dependent label-noise learning with manifold-regularized transition matrix estimation. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[11] Hao Cheng, Zhaowei Zhu, Xingyu Li, Yifei Gong, Xing Sun, and Yang Liu. Learning with instance-dependent label noise: A sample sieve approach. In International Conference on Learning Representations (ICLR), 2021.

[12] Albert Cheu. Diferential privacy in the shufle model: A survey of separations, 2022.

[13] Jase Clarkson, Wenkai Xu, Mihai i Cucuringu, and Gesine Reinert. Split conformal prediction under data contamination. In Proceedings of the Thirteenth Symposium on Conformal and Probabilistic Prediction with Applications, 2024.

[14] Shuhao Cui, Shuhui Wang, Junbao Zhuo, Liang Li, Qingming Huang, and Qi Tian. Towards discriminability and diversity: Batch nuclear-norm maximization under label insuficient situations. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[15] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 248–255. Ieee, 2009.

[16] Weijian Deng and Liang Zheng. Are labels always necessary for classifier accuracy evaluation? In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[17] Yair Dgani, Hayit Greenspan, and Jacob Goldberger. Training a neural network based on unreliable human annotation of medical images. In The IEEE International Symposium on Biomedical Imaging (ISBI), 2018.

[18] D.C. Dowson and B. V. Landau. The Fréchet distance between multivariate normal distributions. Journal of Multivariate Analysis, 12(3):450–455, 1982.

[19] John C Duchi, Michael I Jordan, and Martin J Wainwright. Local privacy and statistical minimax rates. In 2013 IEEE 54th annual symposium on foundations of computer science, pages 429–438. IEEE, 2013.

[20] Cynthia Dwork. Diferential privacy. In International colloquium on automata, languages, and programming, pages 1–12. Springer, 2006.

[21] Bat-Sheva Einbinder, Stephen Bates, Anastasios N Angelopoulos, Asaf Gendler, and Yaniv Romano. Conformal prediction is robust to label noise. arXiv preprint arXiv:2209.14295, 2022.

[22] Úlfar Erlingsson, Vasyl Pihur, and Aleksandra Korolova. Rappor: Randomized aggregatable privacy-preserving ordinal response. In Proceedings of the 2014 ACM SIGSAC conference on computer and communications security, pages 1054–1067, 2014.

[23] K Ruwani M Fernando and Chris P Tsokos. Dynamically weighted balanced loss: class imbalanced learning and confidence calibration of deep neural networks. IEEE Transactions on Neural Networks and Learning Systems, 33(7):2940–2951, 2021.

[24] Rina Foygel Barber, Emmanuel J Candes, Aaditya Ramdas, and Ryan J Tibshirani. The limits of distribution-free conditional predictive inference. Information and Inference: A Journal of the IMA, 10(2):455–482, 2021.

[25] Lior Frenkel and Jacob Goldberger. Calibration of medical imaging classification systems with weight scaling. In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2022.

[26] Marco Gaboardi, Ryan Rogers, and Or Shefet. Locally private mean estimation: Z-test and tight confidence intervals, 2019.

[27] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural networks. The Journal of Machine Learning Research, 17(1):2096–2030, 2016.

[28] Saurabh Garg and Sivaraman Balakrishnan. Leveraging unlabeled data to predict out-ofdistribution performance. International Conference on Learning Representations (ICLR), 2022.

[29] Badih Ghazi, Noah Golowich, Ravi Kumar, Pasin Manurangsi, and Chiyuan Zhang. Deep learning with label diferential privacy. Advances in Neural Information Processing Systems (NeurIPs), 34:27131–27145, 2021.

[30] Aritra Ghosh, Himanshu Kumar, and P. S. Sastry. Robust loss functions under label noise for deep neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1919–1925, 2017.

[31] Jacob Goldberger and Ehud Ben-Reuven. Training deep neural-networks using a noise adaptation layer. In International Conference on Learning Representations (ICLR), 2017.

[32] Devin Guillory, Vaishaal Shankar, Sayna Ebrahimi, Trevor Darrell, and Ludwig Schmidt. Predicting with confidence on unseen distributions. In Proc. of the IEEE International Conference on Computer Vision (ICCV), 2021.

[33] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning (ICML), pages 1321–1330. PMLR, 2017.

[34] Jiangfan Han, Ping Luo, and Xiaogang Wang. Deep self-learning from noisy labels. In Proc. of the IEEE International Conference on Computer Vision (ICCV), pages 5138–5147, 2019.

[35] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.

[36] Matthias Hein, Maksym Andriushchenko, and Julian Bitterwolf. Why relu networks yield high-confidence predictions far away from the training data and how to mitigate the problem. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 41–50, 2019.

[37] Dan Hendrycks, Mantas Mazeika, Duncan Wilson, and Kevin Gimpel. Using trusted data to train deep networks on labels corrupted by severe noise. In Advances in Neural Information Processing Systems (NeurIPs), pages 10477–10486, 2018.

[38] Wei Hu, Zhiyuan Li, and Dingli Yu. Simple and efective regularization methods for training on noisily labeled data with generalization guarantee. In International Conference on Learning Representations (ICLR), 2020.

[39] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 4700–4708, 2017.

[40] Jinchi Huang, Lie Qu, Rongfei Jia, and Binqiang Zhao. O2u-net: A simple noisy label detection approach for deep neural networks. In Proc. of the IEEE International Conference on Computer Vision (ICCV), pages 3326–3334, 2019.

[41] Yingsong Huang, Bing Bai, Shengwei Zhao, Kun Bai, and Fei Wang. Uncertainty-aware learning against label noise on imbalanced datasets. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 6960–6969, 2022.

[42] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 590–597, 2019.

[43] Shenwang Jiang, Jianan Li, Ying Wang, Bo Huang, Zhang Zhang, and Tingfa Xu. Delving into sample loss curve to embrace noisy and imbalanced data. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 7024–7032, 2022.

[44] Ying Jin, Ximei Wang, Mingsheng Long, and Jianmin Wang. Minimum class confusion for versatile domain adaptation. In Proc. of the European Conference on Computer Vision (ECCV), 2020.

[45] Peter Kairouz, Keith Bonawitz, and Daniel Ramage. Discrete distribution estimation under local privacy. In International Conference on Machine Learning (ICML), pages 2436–2444. PMLR, 2016.

[46] Shiva Prasad Kasiviswanathan, Homin K Lee, Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith. What can we learn privately? SIAM Journal on Computing, 40(3):793–826, 2011.

[47] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[48] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, Department of Computer Science, University of Toronto, 2009.

[49] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems (NeurIPs), 30, 2017.

[50] Shikun Li, Shiming Ge, Yingying Hua, Chunhui Zhang, Hao Wen, Tengfei Liu, and Weiqiang Wang. Coupled-view deep classifier learning from multiple noisy annotators. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4667–4674, 2020.

[51] Shikun Li, Tongliang Liu, Jiyong Tan, Dan Zeng, and Shiming Ge. Trustable co-label learning from multiple noisy annotators. IEEE Transactions on Multimedia, 25:1045–1057, 2021.

[52] Xuefeng Li, Tongliang Liu, Bo Han, Gang Niu, and Masashi Sugiyama. Provably end-to-end label-noise learning without anchor points. In International Conference on Machine Learning (ICML), pages 6403–6413. PMLR, 2021.

[53] Jian Liang, Dapeng Hu, and Jiashi Feng. Domain adaptation with auxiliary target domainoriented classifier. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[54] Yong Lin, Renjie Pi, Weizhong Zhang, Xiaobo Xia, Jiahui Gao, Xiao Zhou, Tongliang Liu, and Bo Han. A holistic view of label noise transition matrix in deep learning and beyond. In International Conference on Learning Representations (ICLR), 2023.

[55] Tongliang Liu and Dacheng Tao. Classification with noisy labels by importance reweighting. IEEE Trans. on Pattern Analysis and Machine Intelligence (PAMI), 38(3):447–461, 2015.

[56] Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Michael I Jordan. Conditional adversarial domain adaptation. Advances in Neural Information Processing Systems (NeurIPs), 2018.

[57] Charles Lu, Anastasios N Angelopoulos, and Stuart Pomerantz. Improving trustworthiness of AI disease severity rating in medical imaging with ordinal conformal prediction sets. In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2022.

[58] Charles Lu, Andréanne Lemay, Ken Chang, Katharina Höbel, and Jayashree Kalpathy-Cramer. Fair conformal predictors for applications in medical imaging. In Proceedings of the AAAI Conference on Artificial Intelligence, 2022.

[59] Yang Lu, Yiliang Zhang, Bo Han, Yiu-ming Cheung, and Hanzi Wang. Label-noise learning with intrinsically long-tailed data. arXiv e-prints, pages arXiv–2208, 2022.

[60] Pascal Massart. The tight constant in the Dvoretzky-Kiefer-Wolfowitz inequality. The Annals of Probability, pages 1269–1283, 1990.

[61] John P Miller, Rohan Taori, Aditi Raghunathan, Shiori Sagawa, Pang Wei Koh, Vaishaal Shankar, Percy Liang, Yair Carmon, and Ludwig Schmidt. Accuracy on the line: on the strong correlation between out-of-distribution and in-distribution generalization. In International Conference on Machine Learning (ICML), 2021.

[62] Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, and Mario Lucic. Revisiting the calibration of modern neural networks. Advances in Neural Information Processing Systems (NeurIPs), 34:15682–15694, 2021.

[63] Jishnu Mukhoti, Viveka Kulharia, Amartya Sanyal, Stuart Golodetz, Philip Torr, and Puneet Dokania. Calibrating deep neural networks using focal loss. Advances in Neural Information Processing Systems (NeurIPs), 33:15288–15299, 2020.

[64] Rafael Müller, Simon Kornblith, and Geofrey Hinton. When does label smoothing help? arXiv preprint arXiv:1906.02629, 2019.

[65] Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 29, 2015.

[66] Khanh Nguyen and Brendan O’Connor. Posterior calibration and exploratory analysis for natural language processing models. arXiv preprint arXiv:1508.05154, 2015.

[67] Henrik Olsson, Kimmo Kartasalo, Nita Mulliqi, et al. Estimating diagnostic uncertainty in artificial intelligence assisted pathology using conformal prediction. Nature Communications, 13(1):7761, 2022.

[68] Anusri Pampari and Stefano Ermon. Unsupervised calibration under covariate shift. arXiv preprint arXiv:2006.16405, 2020.

[69] Sangdon Park, Osbert Bastani, James Weimer, and Insup Lee. Calibrated prediction with covariate shift via unsupervised domain adaptation. In International Conference on Artificial Intelligence and Statistics, 2020.

[70] Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, and Lizhen Qu. Making deep neural networks robust to label noise: A loss correction approach. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1944–1952, 2017.

[71] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In Proc. of the IEEE International Conference on Computer Vision (ICCV), pages 1406–1415, 2019.

[72] Xingchao Peng, Ben Usman, Neela Kaushik, Judy Hofman, Dequan Wang, and Kate Saenko. Visda: The visual domain adaptation challenge. arXiv preprint arXiv:1710.06924, 2017.

[73] Coby Penso, , Bar Mahpud, Jacob Goldberger, and Or Shefet. Privacy-preserving conformal prediction under local diferential privacy. Symposium on Conformal and Probabilistic Prediction with Applications (COPA 2025), 2025.

[74] Coby Penso, Lior Frenkel, and Jacob Goldberger. Confidence calibration of a medical imaging classification system that is robust to label noise. IEEE Transactions on Medical Imaging, 43(6):2050–2060, 2024.

[75] Coby Penso and Jacob Goldberger. Calibration of network confidence for unsupervised domain adaptation using estimated accuracy. In ECCV, Uncertainty in Computer Vision Workshop, 2024.

[76] Coby Penso and Jacob Goldberger. A conformal prediction score that is robust to label noise. In MICCAI, Machine Learning for Medical Imaging Workshop, 2024.

[77] Coby Penso and Jacob Goldberger. A joint training and confidence calibration procedure that is robust to label noise. In The IEEE International Symposium on Biomedical Imaging (ISBI), 2024.

[78] Coby Penso, Jacob Goldberger, and Ethan Fetaya. Estimating the conformal prediction thresh old from noisy labels. arXiv preprint arXiv:2501.12749, 2024.

[79] Coby Penso, Jacob Goldberger, and Ethan Fetaya. Conformal prediction of classifiers with many classes based on noisy labels. Symposium on Conformal and Probabilistic Prediction with Applications (COPA 2025), 2025.

[80] John Platt et al. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers, 10(3):61–74, 1999.

[81] Mengye Ren, Wenyuan Zeng, Bin Yang, and Raquel Urtasun. Learning to reweight examples for robust deep learning. In International Conference on Machine Learning (ICML), pages 4331–4340, 2018.

[82] Yaniv Romano, Matteo Sesia, and Emmanuel Candes. Classification with valid and adaptive coverage. Advances in Neural Information Processing Systems (NeurIPs), 2020.

[83] Axel-Jan Rousseau, Thijs Becker, Jeroen Bertels, Matthew B Blaschko, and Dirk Valkenborg. Post training uncertainty calibration of deep networks for medical image segmentation. In The IEEE International Symposium on Biomedical Imaging (ISBI), pages 1052–1056. IEEE, 2021.

[84] Kate Saenko, Brian Kulis, Mario Fritz, and Trevor Darrell. Adapting visual category models to new domains. In Proc. of the European Conference on Computer Vision (ECCV), 2010.

[85] Tiago Salvador, Vikram Voleti, Alexander Iannantuono, and Adam Oberman. Improved predictive uncertainty using corruption-based calibration. Stat, 1050:7, 2021.

[86] Matteo Sesia, YX Wang, and Xin Tong. Adaptive conformal classification with noisy labels. arXiv preprint arXiv:2309.05092, 2023.

[87] Jun Shu, Qi Xie, Lixuan Yi, Qian Zhao, Sanping Zhou, Zongben Xu, and Deyu Meng. Metaweight-net: Learning an explicit mapping for sample weighting. In Advances in Neural Information Processing Systems (NeurIPs), pages 1917–1928, 2019.

[88] Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. Learning from noisy labels with deep neural networks: A survey. IEEE transactions on neural networks and learning systems, 34(11):8135–8153, 2022.

[89] Daiki Tanaka, Daiki Ikami, Toshihiko Yamasaki, and Kiyoharu Aizawa. Joint optimization framework for learning with noisy labels. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5552–5560, 2018.

[90] Christian Tomani, Sebastian Gruber, Muhammed Ebrar Erdem, Daniel Cremers, and Florian Buettner. Post-hoc uncertainty calibration for domain drift scenarios. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[91] Philipp Tschandl, Clif Rosendahl, and Harald Kittler. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific data, 5(1):1–9, 2018.

[92] Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep hashing network for unsupervised domain adaptation. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[93] Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic learning in a random world, volume 29. Springer, 2005.

[94] Ruijia Wang, Shuai Mou, Xiao Wang, Wanpeng Xiao, Qi Ju, Chuan Shi, and Xing Xie. Graph structure estimation neural networks. In WWW, pages 342–353, 2021.

[95] Tianhao Wang, Jeremiah Blocki, Ninghui Li, and Somesh Jha. Locally diferentially private protocols for frequency estimation. In 26th USENIX Security Symposium (USENIX Security 17), pages 729–745, 2017.

[96] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M Summers. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2097–2106, 2017.

[97] Ximei Wang, Mingsheng Long, Jianmin Wang, and Michael Jordan. Transferable calibration with lower bias and variance in domain adaptation. In Advances in Neural Information Processing Systems (NeurIPs), 2020.

[98] Stanley L Warner. Randomized response: A survey technique for eliminating evasive answer bias. Journal of the American statistical association, 60(309):63–69, 1965.

[99] Jiaheng Wei, Zhaowei Zhu, Hao Cheng, Tongliang Liu, Gang Niu, and Yang Liu. Learning with noisy labels revisited: A study using real-world human annotations. arXiv preprint arXiv:2110.12088, 2021.

[100] Tong Wei, Jiang-Xin Shi, Wei-Wei Tu, and Yu-Feng Li. Robust long-tailed learning under label noise. arXiv preprint arXiv:2108.11569, 2021.

[101] Andre Woloshuk, Suraj Khochare, Aljohara F Almulhim, Andrew T McNutt, Dawson Dean, Daria Barwinska, Michael J Ferkowicz, Michael T Eadon, Katherine J Kelly, Kenneth W Dunn, et al. In situ classification of cell types in human kidney tissue using 3D nuclear staining. Cytometry Part A, 99(7):707–721, 2021.

[102] Xiaobo Xia, Tongliang Liu, Bo Han, Nannan Wang, Mingming Gong, Haifeng Liu, Gang Niu, Dacheng Tao, and Masashi Sugiyama. Part-dependent label noise: Towards instancedependent label noise. In Advances in Neural Information Processing Systems (NeurIPs), 2020.

[103] Xiaobo Xia, Tongliang Liu, Nannan Wang, Bo Han, Chen Gong, Gang Niu, and Masashi Sugiyama. Are anchor points really indispensable in label-noise learning? In Advances in Neural Information Processing Systems (NeurIPs), pages 6835–6846, 2019.

[104] Mou-Cheng Xu, Yukun Zhou, Chen Jin, Marius De Groot, Daniel C Alexander, Neil P Oxtoby, and Joseph Jacob. Mismatch: Calibrated segmentation via consistency on diferential morphological feature perturbations with limited labels. IEEE transactions on medical imaging, 42(10):2988–2999, 2023.

[105] Cheng Xue, Lequan Yu, Pengfei Chen, Qi Dou, and Pheng-Ann Heng. Robust medical image classification from noisy labeled data with global and local representation guided co-training. IEEE Transactions on Medical Imaging, 41(6):1371–1382, 2022.

[106] Hansi Yang, Quanming Yao, Bo Han, Gang Niu, Hansi Yang, Bo Han, Gang Niu, and James Kwok. Searching to exploit memorization efect in learning from corrupted labels. In International Conference on Machine Learning (ICML), 2020.

[107] Jiancheng Yang, Rui Shi, and Bingbing Ni. MedMNIST classification decathlon: A lightweight automl benchmark for medical image analysis. In The IEEE International Symposium on Biomedical Imaging (ISBI), 2021.

[108] Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, and Bingbing Ni. Medmnist v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification. Scientific Data, 10(1):41, 2023.

[109] Yaodong Yu, Zitong Yang, Alexander Wei, Yi Ma, and Jacob Steinhardt. Predicting out-of distribution error with the projection norm. In International Conference on Machine Learning (ICML), 2022.

[110] Bianca Zadrozny and Charles Elkan. Transforming classifier scores into accurate multiclass probability estimates. In International Conference on Knowledge Discovery and Data Mining (KDD), pages 694–699, 2002.

[111] Fan Zhang, Nicha Dvornek, Junlin Yang, Julius Chapiro, and James Duncan. Layer embedding analysis in convolutional neural networks for improved probability calibration and classification. IEEE Transactions on Medical Imaging, 39(11):3331–3342, 2020.

[112] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In International Conference on Learning Representations (ICLR), 2018.

[113] Jize Zhang, Bhavya Kailkhura, and T Yong-Jin Han. Mix-n-match: Ensemble and compositional methods for uncertainty calibration in deep learning. In International Conference on Machine Learning (ICML), pages 11117–11128. PMLR, 2020.

[114] Le Zhang, Ryutaro Tanno, Mou-Cheng Xu, Chen Jin, Joseph Jacob, Olga Cicarrelli, Frederik Barkhof, and Daniel Alexander. Disentangling human error from ground truth in segmentation of medical images. Advances in Neural Information Processing Systems (NeurIPs), 33:15750– 15762, 2020.

[115] Linjun Zhang, Zhun Deng, Kenji Kawaguchi, and James Zou. When and how mixup improves calibration. In International Conference on Machine Learning (ICML), pages 26135–26160. PMLR, 2022.

[116] Yikai Zhang, Songzhu Zheng, Pengxiang Wu, Mayank Goswami, and Chao Chen. Learning with feature-dependent label noise: A progressive approach. In International Conference on Learning Representations (ICLR), 2021.

[117] Yivan Zhang, Gang Niu, and Masashi Sugiyama. Learning noise transition matrix from only noisy labels via total variation regularization. In International Conference on Machine Learning (ICML), pages 12501–12512. PMLR, 2021.

[118] Zhilu Zhang and Mert Sabuncu. Generalized cross entropy loss for training deep neural networks with noisy labels. In Advances in Neural Information Processing Systems (NeurIPs), pages 8778–8788, 2018.

[119] Songzhu Zheng, Pengxiang Wu, Aman Goswami, Mayank Goswami, Dimitris Metaxas, and Chao Chen. Error-bounded correction of noisy labels. In International Conference on Machine Learning (ICML), pages 11447–11457, 2020.

[120] Yang Zou, Zhiding Yu, Xiaofeng Liu, BVK Kumar, and Jinsong Wang. Confidence regularized self-training. In Proc. of the IEEE International Conference on Computer Vision (ICCV), 2019.

[121] Yuli Zou, Weijian Deng, and Liang Zheng. Adaptive calibrator ensemble: Navigating test set dificulty in out-of-distribution scenarios. In Proc. of the IEEE International Conference on Computer Vision (ICCV), 2023.

## תקציר

ביישומים קריטיים כגון דימות רפואי, מידת האמון של מודל בעצמו חשובה לא פחות מהתחזיות שהוא   
מספק. כיול ביטחון מבטיח שההסתברויות שמודל מנבא משקפות באופן מדויק את הסיכוי שהוא צודק, ובכך מהווה רכיב קריטי לשימוש בטוח ויעיל של מודלים ללמידה עמוקה באבחון רפואי. עם זאת, טכניקותהכיולהקיימותמניחותגישהלנתוניולידציהנקיים,הנחהשאינהריאליתלעיתיםקרובות   
בסביבות דימות רפואי בשל נוכחות רעש תוויות ושינויים בין תחומיים. מחקר זה מציע שיטות חדשניות לשיפור כיול הביטחון בתנאים מאתגרים אלה. ראשית, אנו מתמודדים עם בעיית כיול הביטחון בנוכחות רעש תוויות. כאשר שיטות כיול מוכלות על נתונים עם תוויות בלתי אמינות, הן עלולות להניב אומדנים שגויים של ביטחון, ובכך לפגוע באמינות   
התחזיותשלהמודל.אנומציעיםשיטתכיולהלוקחתבחשבוןאתרעשהתוויותבאמצעותהערכתמודל רעש. באופן ספציפי, אנו מראים כיצד לשחזר אומדנים נטולי רעש על ידי מידול הקשר בין התפלגויות   
תוויות נקיות ומרועשות. אנו מרחיבים רעיון זה גם לחיזוי קונפורמי, מסגרת המספקת תחזיות קבוצתיות   
עם רמת כיסוי מובטחת. אנו מציגים גישה לחיזוי קונפורמי עמיד לרעש, אשר מעריכה את ציוני ההתאמה האמיתייםלמרותרעשהתוויות,ובכךמאפשרתלנולשמורעלהערכתאי-ודאותיעילהואמינה.

בהמשך,אנוחוקריםכיולביטחוןבהסתגלותתחומיתבלתימפוקחת,שבהמודלשאומןעלתחוםמקור מתויגמוחלעלתחוםיעדבלתימתויג.שיטותכיולמסורתיותדורשותנתוניולידציהמתויגיםמתחום היעד,שאינםזמיניםבסביבהזו.כדילהתגברעלמגבלהזו,אנומפתחיםגישההמעריכהאתדיוק המודל בתחום היעד על סמך ביצועיו בתחום המקור וביצועיו בתחומי יעד סינתטים ידועים. גישה זו מאפשרתלנולכיילאתרמותהביטחוןשלהמודלישירותללאצורךבגישהלתוויותמתחוםהיעד.

כמו כן, אנו מרחיבים את המחקר שלנו לסביבות השומרות על פרטיות, שבהן יש להגן על תוויות משתמשים ופלטי מודלים. אנו מציעים מסגרת לחיזוי קונפורמי עם פרטיות דיפרנציאלית מקומית, המבטיחה הערכת אי-ודאות תקפה תוך שמירה על הבטחות פרטיות מחמירות. גישה זו מאזנת בין דרישות פרטיות, חישוביות ואמינות תחזיות, ובכך מתאימה במיוחד ליישומים רפואיים רגישים.

באמצעות ניסויים נרחבים על מערכי נתונים טבעיים ורפואיים, אנו מראים כי השיטות המוצעות משפרות באופן משמעותי את חוסן הכיול הן בנוכחות רעש תוויות והן בתנאי שינוי תחומי. אנו מספקים הוכחות   
תיאורטיות ואישושים אמפיריים המגשרים על הפער בין ערבויות הכיול התיאורטיות לבין הפעלה מעשית   
בסביבות קריטיות. ממצאינו תורמים לפיתוח מסגרות כיול אמינות, שומרות פרטיות ועמידות לרעש, ובכך משפרים את האמינות של תחזיות רשתות נוירונים ביישומים רפואיים ויישומים ברמת סיכון גבוהה בעולםהאמיתי.

## תוכן עיניינים

תקציר   
הקדמה.   
6.. רקע .   
6. כיולבטחוןפרמטרי   
7. חיזויקונפורמי   
9.. כיולבטחוןשלמועדיעד.   
10.. תיוגרועש.   
13.. פרטיותדיפרנציאליתמקומית.   
15.. כיולבטחוןפרמטריתחתתיוגרועש.   
15... הגדרתהבעיה   
17.. כיולבטחוןתחתתיוגרועש.   
21. אימוןמודלתחתתיוגרועש.   
22. ניסויים   
29. חיזויקונפורמיתחתתיוגרועש   
29. הגדרתהבעיה   
30.. מדדחיזויקונפורמיעמידלרעש   
32. שערוךהסףהקונפורמיעמידלרעש   
35.. גודלהקבוצה.   
36... הוכחותכיסויתיאורטיות.   
38.. מודלרעשכללי   
40. השוואתשיטות   
41.. ניסויים   
47.. חיזויקונפורמיעםפרטיותדיפרנציאליתלוקאלית   
47.. הגדרתהבעיה   
49.. חיזויקונפורמיעםפרטיותדיפרנציאליתלוקאליתעלידיהרעשתתוויות.   
51. חיזויקונפורמיעםפרטיותדיפרנציאליתלוקאליתעלידיהרעשתמדדים   
53.. הוכחותכיסויופרטיות.   
54.. דיון...   
54... הרעשתתוויותמולהרעשתמדדים   
55... שיטתהערבולשלפרטיותדיפרנציאלית..   
56... ניסויים   
60... כיולבטחוןפרמטרישלמודליעדללאתיוג..   
60... הגדרתהבעיה.   
62... כיולבטחוןמודלהיעדבתרחישהתאמהללאתיוג..   
64... ניסויים   
67... דיון ואנליזה..   
71.. דיון   
71... סיכוםתרומותהתזה   
72.. תובנותוהשלכות.   
73... כיוונימחקרעתידיים   
74... סיכום.   
75 רשימת מקורות   
א תקציר בעברית

## הבעת תודה

ברצוני להביע את תודתי העמוקה למנחה שלי, פרופ' יעקב גולדברגר, על ההכוונה, התמיכה והעידוד לאורךכלתקופתלימודיהדוקטורט.תובנותיו,סבלנותווהסטנדרטיםהגבוהיםשלותרמורבות לעבודהזוולעיצובדרכיכחוקר.

אנימודהגםלשותפיילדרךולכותביםבצוות,איתןפתיה,ברמחפודוליאורפרנקל,עלהרעיונות, המשובוההשקעהשהעשירומאודאתהמחקר.היהליתענוגאמיתיללמודולעבודלצדם.

ולבסוף, תודה גדולה למשפחתי ולחבריי על התמיכה הבלתי מתפשרת, ההבנה והעידוד לאורך כל הדרך. בלעדיהם, עבודת דוקטורט זו לא הייתה מתאפשרת.

עבודה זו נעשתה בהדרכתו של   
פרופ׳ יעקב גולדברגר,   
הפקולטה להנדסה,   
אוניברסיטת בר-אילן

# כיול הבטחון בהחלטה שלמערכותלמידהעמוקה

חיבורלשםקבלתהתואר״דוקטורלפילוסופיה״ מאת: יעקב )קובי( פנסו

הפקולטה להנדסה

הוגשלסנטשלאוניברסיטתבראילן

אייר, תשפ״ה

רמת גן
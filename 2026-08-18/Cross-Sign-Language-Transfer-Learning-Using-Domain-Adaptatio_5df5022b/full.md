# Cross-Sign Language Transfer Learning Using Domain Adaptation with Multi-scale Temporal Alignment

Keren Artiaga, Yang Li, Ercan Engin Kuruoglu and Wai Kin (Victor) Chan

Tsinghua-Berkeley Shenzhen Insitute, Tsinghua Shenzhen International Graduate School, The University Town, Nanshan District, Shenzhen, 518055, P.R.China.

Contributing authors: kerenartiaga@outlook.com; yangli@sz.tsinghua.edu.cn; kuruoglu@sz.tsinghua.edu.cn; chanw@sz.tsinghua.edu.cn;

## Abstract

Sign language serves as a vital means of communication for individuals with hearing impairments, yet recognition resources for the over 100 distinct sign languages are severely lacking. In response, we present our work on sign language recognition using transfer learning and the domain adaptation method TA3N, which utilizes the Temporal Relational Network (TRN) module for aligning multi-scale temporal relations. Our findings highlight the superior performance of Domain Adaptation to neural network-based transfer learning, particularly in improving recognition of American Sign Language (ASL). Our research also identifies the efectiveness of aligning shorter-term temporal features between source and target domains. In addition to using RGB, we conducted experiments using Optical Flow mode for the sign language samples, ultimately determining that RGB outperforms Optical Flow in the majority of cases. Our work aims to improve accessibility and communication for individuals who rely on sign language as their primary mode of communication.

Keywords: sign language recognition, domain adaptation, temporal relations, optical flow

## 1 Introduction

The World Health Organization predicts that by 2050, one out of ten people will have some degree of hearing loss. Translation between sign languages and spoken languages is crucial for facilitating communication between sign language users and those who do not use sign language. However, Sign Language Recognition (SLR) is not as extensively researched as other types of action recognition, and SLR studies tend to focus on only a few of the 135 sign languages used worldwide due to the lack of high-resource datasets for most of these sign languages. It is common knowledge that low-resource datasets are prone to overfitting in classification or recognition tasks. Our solution to the low-resource dataset problem for SLR is cross-sign language transfer learning, where we transfer knowledge from higher-resource sign language datasets. This method is domain-specific and difers from state-of-the-art SLR methods that use non-sign language-specific datasets, such as ImageNet. By applying sign language to sign language transfer learning, we can improve SLR by learning high-level, specific visual features that cannot be provided by general largescale non-sign language datasets, which are limited to low-level, superficial features.

By definition, transfer learning is a machine learning technique that works by applying the knowledge gained in one domain to another. Specifically, this paper aims at achieving efective cross-sign language transfer learning using Domain Adaptation. Our study focuses on Argentine Sign Language (LSA) and Chinese Sign Language (CSL) as the source domains, with American Sign Language (ASL) as the target domain. However, the methodology presented in this paper can be universally applied to any sign language recognition task. We decided on ASL to be the target domain as being a low-resource dataset, it has a less number of samples per class than LSA and CSL, as seen in Table 2. Domain adaptation is a subset of transfer learning that brings the target task’s distribution closer to that of the source. There are studies [1–4] on neural network-based transfer learning, specifically pre-training applied to SLR, and a study on domain adaptation between ASL copy-sign data and ASL native data [5]. However, there are currently no studies on the application of domain adaptation between diferent sign languages. Moreover, as evidenced by the results of our pre-training experiments 4.3, it is clear that pre-training is not suficient for domains with diferent distributions, such as sign languages. Therefore, we deem it advantageous to utilize Domain Adaptation. Our major contribution, specifically, is the use of multi-scale temporal alignment to align the domain of the source sign language to that of the target.

We furthered our domain adaptation with multi-scale temporal alignment work by determining the diferent efects of aligning the source and target domains’ temporal relations at both short-term and long-term timescales. Temporal relations can be defined as a group of select time-ordered frames of a video. From the results of our experiment, we were able to prove our hypothesis that aligning shorter-term relations instead of longer-term ones is the better approach for improving the accuracy of the target SLR model in all adaptations (from LSA to ASL and from CSL to ASL) and in all input modalities we used in this study.

We experimented in two learning settings: full-scale transfer learning and few-shot transfer learning. In full-scale transfer learning, we use an 80:20 ratio of target training data to target test data for each k-fold cross-validation. Few-shot transfer learning aims to create an accurate model with a test base using less target training data, mimicking human learning. Compared to fullscale learning, few-shot learning requires less training data and computational resources. We implement few-shot transfer learning with a 20:80 ratio of target training data to target test data.

We used RGB and Optical Flow as input modalities, which can easily be generated from video frames, eliminating the need for additional equipment such as a specialized camera used in several SLR works, such as [4]. We specifically implemented the Gunner Farneback optical flow algorithm[6], which has the advantage of being invariant to appearance[7], making generalization easier for low-resource sign language datasets.

The datasets we used in this work are LSA64: A Dataset of Argentinian Sign Language [8] , Chinese isolated SLR dataset [9–11] and WLASL300 [12] datasets for LSA, CSL, and ASL, respectively.

To summarize, our contributions are as follows:

(1) We implemented multi-scale temporal alignment (Sec. 3.2.1) to conduct domain adaptation between diferent sign languages

(2) We determined the diferent efects of aligning temporal relations of the sign languages at both short-term and long-term timescales

(3) We conducted domain adaptation in two learning settings: full-scale transfer learning (Sec. 4.2.1) and few-shot transfer learning (Sec. 4.2.2)

(4) We used two input modalities for the samples: RGB and Optical Flow (Sec. 3.1), and provided a comparison of their performance for each experiment (Sec. 4.2).

This paper is organized as follows: In Section 2, related works on Sign Language Recognition (SLR) and Domain Adaptation are discussed. Section 3 outlines the technical approach used in the study, including pre-processing of the videos and details of the sign language models. Section 4 presents the dataset, experimentation setup, results, and analysis. Finally, Section 6 provides conclusions and suggestions for future research in this field.

## 2 Related Works

## 2.1 Related Works on Sign Language Recognition

In SLR, transfer learning is commonly done by pre-training on large-scale datasets like ImageNet of a diferent domain and using its weights to initialize the target’s weights.

One of the earliest applications of transfer learning in SLR is the creation of ASL word models by Farhadi et al. [13] that transfer between diferent signers as well as diferent aspects. For these ASL word models, the authors used one subject in the source dataset and one in the target dataset. The subject for the source is an avatar.

Nishat et al. [3] conducted unsupervised transfer learning to the Bangla sign language dataset (BsDL) to recognize Bangla character sign language such as digits, vowels, and consonants using VGG16 architecture pre-trained on ImageNet. Another work on BsDL is by Das et al. [14] wherein a hybrid approach using deep transfer learning model with random forest classifier is used.

Cayamcela et al. [2] used a convolutional neural network (CNN) pre-trained on the ImageNet dataset to help translate American Sign Language Alphabet in real time. Both [2] and [3] are dealing with static signs where the inputs to the network are still images and not videos. Recently, Halvardsson et al. [15], developed a Swedish Sign Language Alphabet model that uses an InceptionV3 network that is also pre-trained on ImageNet. This study involves 8 subjects and similar to [2], [3] and [14], does not utilize any videos as inputs.

Recent research has shown that pre-training neural networks on large-scale datasets like ImageNet can improve the accuracy of sign language recognition (SLR). This method is currently considered the state-of-the-art in transfer learning for SLR, as demonstrated in [1–3, 14–20]. Meanwhile, a study by Suharjito et al. [21] uses both ImageNet as well as kinetic dataset - an action recognition dataset, as source for training their Indonesian Signal System recognition model.

However, domain-specific transfer learning is gaining popularity in improving neural network models for SLR. Bird et al. [4] used transfer learning by pre-training with British Sign Language (BSL) to improve recognition of American Sign Language (ASL) using VGG16 architecture. The study included 18-class BSL and ASL datasets, with ASL consisting of one-handed signs from two subjects. The authors found that the late fusion of RGB and Leap Motion approaches (multimodality) in SLR improved their ASL recognition model compared to using RGB or LM alone. They suggested that similarities in movement between the selected BSL and ASL gestures transferred useful knowledge between the two sign languages.

Another domain-specific transfer learning study was conducted by Abdullayeva et al. [22], where they pre-trained their Azerbaijian fingerspelling (alphabet) recognition model on both the ImageNet dataset and an ASL alphabet dataset. To the best of our knowledge, no previous research has explored few-shot transfer learning in SLR. Existing studies are limited by the type of sign language they can recognize (i.e. static signs such as alphabet, characters, and digits), the number of subjects in their datasets, or the type of gestures used. Our work is particularly challenging as we focus on ASL, a low-resource language with two-handed gestures performed by diferent subjects, with an average of only 1.3 repetitions per sign in the dataset. Additionally, the ASL dataset we used is sourced from various online platforms with diferent lighting, backgrounds, aspects, and subject distances from the camera. The most relevant work to our study [4] only included one-handed gestures from two signers. See Table 11 for a detailed comparison with other studies.

## 2.2 Related Works on Domain Adaptation

Domain Adaptation is a type of Transfer Learning that aims to generalize a model trained in one domain to perform well in another domain. This is necessary because models trained in one domain may not work well in another domain due to domain shift. Domain shift occurs when the source and target domains have diferent probability distributions, even though they share the same task. To address this issue, domain adaptation adjusts the target data’s distribution to match that of the source data.

While both neural network-based transfer learning and domain adaptation can help networks learn discriminative features for low-resource settings, their execution difers. Neural network-based transfer learning reuses the source task’s features and weights in the target task’s network. Domain Adaptation, on the other hand, attempts to reduce shift between two diferent domains by re-weighting source features.

Similar to video classification tasks, Sign Language classification can sufer from domain discrepancy in both spatial and temporal aspects. Hence, there is a need for domain adaptation approaches, specifically for video recognition and classification. Currently, there are only a few studies that apply Domain Adaptation to video classification [23–26]. Only Temporal Attentive Adversarial Adaptation Network (TA3N) [27], Temporal Attentive Moment Alignment Network (TAMAN) [24], and Contrast and Mix [26] are tested on large-scale datasets. TAMAN is created for multi-source video domain adaptation, while Contrast and Mix are designed for unsupervised domain adaptation. Although TAMAN and Contrast and Mix may be applied to a single-source problem, we decided to use TA3N as it incorporates within its architecture an extensible, plug-and-play module called Temporal Relation Network (TRN) [28] that generates multiscale temporal relations. TRN is based on humans’ ability to connect meaningful transformations in an object or event without observing all changes, known as temporal relational reasoning. TRN aims to discover important temporal relations between frames in videos when applied to neural networks. Technical implementation of TRN is described rigorously in Section 3.2.1

Our aim is to determine the optimal timescale (short-term or long-term) for aligning the temporal relations of the source and target domains. To achieve this goal, we concluded that TA3N, which includes the TRN module, is the most suitable architecture for our cross-sign language domain adaptation task.

## 3 Technical Approach

Our technical approach is a two-step process that consists of a pre-processing step and a domain adaptation step as shown in Figure 1. In the pre-processing step, we convert videos into RGB frames and Optical Flow frames. After generating the frames, we proceeded with the transfer learning step, where we use the Domain Adaptation approach. Domain Adaptation difers from neural network-based transfer learning in that it is utilized when the source and target tasks share a similar feature space but have distinct probability distributions due to having unrelated datasets. This is particularly relevant in cases where we are adapting the domains of CSL and LSA to ASL.

We implemented Domain Adaptation in five diferent multiscale temporal relations: 3, 5, 7, 10, and 15. The shortest term is 3 and the highest term is 15. Section 3.2.1 describes multiscale temporal relations in detail. The experiment results in Section 4.2 show in detail the efects of N diferent multi-scale TRN on the recognition accuracy of the target domain.

![](images/9f650211c40aee77bb8f690d3319fcbb0aafffcf6302daa85faa964465381da9.jpg)  
Fig. 1 The process involved two steps: pre-processing and domain adaptation, to generate an SLR model. In the pre-processing step, we convert videos into both Optical Flow and RGB frames. The features of these frames are then fed into the Domain Adaptation network five times, each time utilizing a diferent N-multiscale temporal relations network (TRN) to determine the optimal timescale for aligning the temporal relations of the sign language domains.

## 3.1 Pre-processing of Videos

To facilitate cross-sign language transfer learning, we first convert selected individual gesture videos into Optical Flow and RGB frames. To capture the sequential diferences between gestures, we experimented with the Optical Flow input mode, which tracks the motion of pixels between frames. We extracted a maximum of 200 frames per sample, as most samples have less than 100 frames.

The Figures 2, 3, 4 and 5 show the subsampled frames from a collection of frames for the gesture Copy in CSL and ASL. For Optical Flow, the backgrounds are changed to white from black for the readers of this paper to see the hand movements more clearly.

![](images/57b5cce26b58c3b0a09a872278ad5b292e1398293345172d714e8439b544bd38.jpg)  
Fig. 2 Subsampled RGB frames (starting from left to right) for the gesture Copy in ASL from WLASL dataset

![](images/892ce0464f53c6aaedb5855cfceeb8202987804b7162328c767bd3c466e3c967.jpg)  
Fig. 3 Subsampled optical flow frames (starting from left to right) for the gesture Copy in ASL from WLASL dataset

It can be observed that for the optical flow mode of the ASL gesture Copy in Figure 3, the resulting images are pixelated in comparison to its LSA counterpart in Figure 5. The video quality of the samples in ASL is on average poorer than those of CSL and LSA. This is also a reason for selecting ASL as the target domain to improve using Domain Adaptation, in addition to having fewer samples per class.

![](images/a9554eb8efc1dc9c7c046d49661618ce6eaa9ea2aac81e5077304da80bdf3883.jpg)

![](images/01c16492747ab944850c1c8580cc430732c6ac4aa73ef79c00c6ca04ab966a84.jpg)  
Fig. 4 Subsampled RGB frames (starting from left to right) for the gesture Copy in LSA from LSA64 dataset  
Fig. 5 Subsampled optical Flow frames (starting from left to right) for the gesture Copy in LSA from LSA64 dataset

## 3.2 Domain Adaptation for Sign Classification

## 3.2.1 Temporal Relations Network

We are using the Temporal Attentive Adversarial Adaptation Network (TA3N) [27] to efectively align two diferent sign language domains. TA3N aligns multiscale temporal relation networks of source and target samples by assigning weights to them. The more domain discriminative the temporal relations are, the higher weights they will receive. Section 3.2.2 explains the implementation of TA3N. The multi-scale temporal relations network used in TA3N is from a separate module called Temporal Relation Network (TRN) [28] by Zhou et al.

These temporal relation features are generated by combining the CNN features of N equidistant frames that are sparsely sampled. A single relation feature that represents N number of frames is referred to as an N-frame temporal relation. The function below defines a 2-frame temporal relation:

$$
T _ { 2 } ( V ) = h _ { \Phi } ( \sum _ { i < j } g \Theta ( f i , f j ) )
$$

wherein $T _ { 2 } ( V )$ means 2-frame temporal relation of the video V with n chosen time-ordered frames $f _ { 1 } , f _ { 2 } , . . . , f _ { n }$ . The function $g _ { \Theta }$ and $g _ { \Phi }$ fuse the features of diferent time-ordered frames. The uniformly sampled i and $j$ frames need not be consecutive so as long as they are sorted chronologically. For a 3-frame temporal relation, below is the corresponding composite function:

$$
T _ { 3 } ( V ) = h _ { \Theta } ( \sum _ { i < j < k } g \Theta ( f i , f j , f k ) )
$$

The final output of the TRN module that is implemented in TA3N is a multiscale TRN which is a fusion of diferent N-frame temporal relations at multiple time scales. The function below shows the accumulation of N-frame temporal relations at N-time scales:

$$
M T _ { N } ( V ) = T _ { 2 } ( V ) + T _ { 3 } ( V ) . . . + T _ { N } ( V )
$$

![](images/531d047eebbe3ce023cae7d558542a8c98212e7ec04abeb9291bc4eb13e7c9e2.jpg)  
Fig. 6 TRN showing the process of generating N-frame temporal relations and finally a multiscale temporal relation for an ASL gesture

The TRN module determines the representative frames to be used by converting each frame into features and fusing them into diferent N-frame temporal relation features, which are then passed into the TRNs. For instance, for 2-frame relations, the TRN module will vote for the combination of two frames that best classify the video.

TA3N focuses on aligning all N-frame temporal relation features $T _ { N } ( V )$ of the source domain’s multi-scale TRN $M T _ { N } ( V _ { s } )$ and the target domain’s multiscale TRN $M T _ { N } ( V _ { t } )$ , whereas our objective is to learn the best N of multi-scale TRN $M T _ { N }$ for adapting the domain of one sign language to another.

Further details of the TA3N architecture are described below in Section 3.2.2.

## 3.2.2 Detailed TA3N Architecture

In Figure 7, TA3N applied to our SLR task is shown. The process begins with raw video frames inputted into a convolutional network (CNN) to convert them into frame-level feature vectors, which are then passed to the Spatial Module. This module contains multilayer perceptrons to extract the spatial information of objects, such as hands, in a video, and convert the feature vectors into task-driven feature vectors for video classification.

The task-driven feature vectors are generated and then sent to the Temporal Relation Network (TRN) Module. The TRN module extracts N-frame relations to generate a multiscale TRN $M T _ { N }$ , as shown in Figure 7. The process of the TRN module is described in detail in Section 3.2.1.

The next step involves Domain Adversarial Training for class and domain prediction, where two networks are used: the Temporal Classification Network for class prediction and the Temporal Adversarial Discriminator for domain prediction. Each network consists of a multi-layer perceptron (MLP) with a fully connected layers network that attends to the N-frame temporal relations of the multiscale TRN generated in the previous step.

The network will assign higher weights to N-frame temporal relations that will score low in Temporal Domain Loss which is calculated by the Temporal Adversarial Discriminator. This discriminator will calculate the loss based on the domain prediction. If the domain prediction is correct, the Temporal Adversarial Discriminator will output a low domain loss. Through backpropagation, the Domain Attention block will update the weights of the N-frame temporal relations $T _ { N } ( V )$ of the target domain’s multi-scale TRN $M T _ { T N } ( V _ { t } )$ where t means target domain in an attempt to increase the domain loss.

A fully connected layer as seen in Figure 7 converts the N-frame relations into class predictions to which Prediction Loss is computed. Prediction Loss must be calculated as well because, as discussed in Section 2.2, Domain Adaptation works by minimizing the class prediction loss and then maximizing the domain prediction loss.

The final loss to be calculated is called Attentive Entropy Loss which is a product of domain entropy and class entropy. Domain entropy and class entropy are calculated from domain loss and class loss, respectively. TA3N added this type of loss in the architecture to enhance the certainty of videos that have low domain discrepancy to enable it to focus more on videos, specifically their N-frame relations, that have high domain discrepancy by assigning higher weights to them during backpropagation. When the model experiences low domain discrepancies, indicated by lower domain prediction loss and low attentive entropy loss, it implies confusion between the source and target, whereas high domain discrepancies are observed when the model accurately predicts the domain, indicating high discrepancies between the source and target. The network needed to enhance the certainty of low-domain discrepant videos as their N-frame temporal relations do not need attention weights as much as those of high-domain discrepant videos. In Section 2.2, we elaborate on why we opted for TA3N for our SLR tasks, as well as its distinguishing features from other related domain adaptation architectures.

![](images/de7d31eb88984b3d73106856eecba877c100691bd055a704cf217121d1ff280d.jpg)  
Fig. 7 The architecture of TA3N applied on Sign Language Recognition task.

## 3.2.3 Fine-Tuning TA3N for Sign Language Recognition

N-multiscale TRN $M T _ { N } ( V _ { s } )$ is based on Temporal Relation Network (TRN) [28] which is an essential parameter for the TA3N architecture as it is used to determine the number of multiple temporal relations to align between the source and the target domain. Section 3.2.1 describes TRN in detail. In the TA3N architecture, the multi-scale TRN parameter is set to 5 by default; therefore, all of TA3N’s experiments with large-scale action recognition datasets are conducted in 5-multiscale TRN. Therefore, we argue that we can fine-tune this value for SLR by using a smaller N-multiscale TRN. For our study, we compared the performance of 3, 5, 7, 10, and 15 multi-scale TRNs to prove our hypothesis that short-term multi-scale TRNs would produce better results for SLR domain adaptation. We refer to 3, 5, and 7 multi-scale TRNs as shorterterm multiscale TRNs while 10 and 15 as longer-term multiscale TRNs. The reason shorter-term multiscale TRNs would perform better is that they would prevent overfitting, especially in the case where sign languages have significant diferences from each other. It is worth noting that LSA, CSL, and ASL are from diferent sign language families. In our experiment results and analysis section ( see Section 4.2 ), we would select the best multi-scale TRN for RGB and Optical Flow in each LSA to ASL and CSL to ASL adaptations.

Lastly, we implemented TA3N in a supervised manner by setting the use target parameter to sV (i.e. supervised). The default learning implementation of TA3N is unsupervised, that is, the classification loss is calculated solely from the labeled source data. By modifying the use target parameter to sV, the classification loss in our experiments would now be based on the concatenation of the labeled source and target data.

## 3.3 Few-Shot Transfer Learning

We conducted experiments using both full-scale transfer learning and few-shot transfer learning settings. Few-shot transfer learning is a type of Few-Shot Learning (FSL) that aims to achieve good learning performance with a limited training set in a supervised manner [29]. We used few-shot transfer learning to train our SLR model because we wanted it to classify sign language gestures with just a few examples, similar to how humans learn. Our target domain, ASL, has an average of about 14 samples per class. For full-scale transfer learning, we split the target training and test data in an 80/20 ratio. For few-shot transfer learning, we used a 20/80 split, resulting in a limited target training data of approximately 2 samples per class.

## 4 Experiments

TA3N was originally tested on large-scale action recognition datasets, HMDB51 [30] and UCF101 [31], using a ResNet-101 architecture. However, its default implementation showed poor performance on sign language recognition. We hypothesize that this is mainly due to the lower number of samples in comparison to HMDB51 and UCF101 datasets, which is a common characteristic among sign language datasets. Hence, we needed to modify the default implementation to fit a more domain-specific, low-resource task of sign language recognition.

We experimented to determine the efects of using shorter-term and longerterm multiscale TRNs on the accuracy of our target sign language recognition model. We also wanted our model to learn from few-shot transfer learning settings. Additionally, we aim to compare RGB inputs against Optical Flow inputs. Section 4.2 will show how our SLR models performed in the following scenarios: (1) short-term versus long-term multiscale TRNs, (2) full-scale versus few-shot transfer learning, (3) RGB vs Optical Flow input mode.

## 4.1 Experiment Setup

This study utilizes the CSL Isolated Chinese Sign Language Dataset [9–11] for CSL, the Word-Level American Sign Language (WLASL300) [12] dataset for ASL, and LSA64:A Dataset of Argentinian Sign Language [8] for LSA. The table presented below displays the classes that we used for the domain adaptation from LSA to ASL and CSL to ASL.

Table 1 List of classes used in all transfer learning experiments.
<table><tr><td>CSL and WLASL300 labels</td><td>LSA and WLASL300 labels</td></tr><tr><td>deaf</td><td>deaf</td></tr><tr><td>red</td><td>red</td></tr><tr><td>green</td><td>green</td></tr><tr><td>yellow</td><td>yellow</td></tr><tr><td>colors</td><td>colors</td></tr><tr><td>son</td><td>son</td></tr><tr><td>call</td><td>call</td></tr><tr><td>milk</td><td>milk</td></tr><tr><td>none</td><td>none</td></tr><tr><td>name</td><td>name</td></tr><tr><td>chair</td><td>water</td></tr><tr><td>thin</td><td>man</td></tr><tr><td>white</td><td>women</td></tr><tr><td>work</td><td>learn</td></tr><tr><td>blanket</td><td>country</td></tr><tr><td>coffee</td><td>where</td></tr><tr><td>friend</td><td>birthday</td></tr><tr><td>future</td><td>music</td></tr><tr><td>husband</td><td>candy</td></tr><tr><td>light</td><td>catch</td></tr><tr><td>new</td><td>help</td></tr><tr><td>student</td><td>dance</td></tr><tr><td>ugly</td><td>buy</td></tr><tr><td rowspan="3"></td><td>copy</td></tr><tr><td>run</td></tr><tr><td>give</td></tr><tr><td>Total of 23 classes</td><td>Total of 26 classes</td></tr></table>

To ensure suficient training and test sets, we used a subset of the larger WLASL dataset, namely WLASL300, which comprises the top 300 classes with the most samples. We selected classes for the experiments based on the labels present in both the source datasets (LSA64 and Isolated Chinese Sign Language) and the target dataset (WLASL300). We identified 26 mutual labels for LSA and ASL datasets, and 23 mutual labels for CSL and ASL datasets.

For the subset of the WLASL300 dataset used in this study, a subject only repeats a gesture on average 1.3 times compared to LSA64 and Chinese isolated SLR, where each subject repeats a gesture 5 times. This demonstrates the generalization ability of the WLASL dataset over other sign language datasets. However, we believe that this ability can cause underfitting, as the average number of samples per class in this dataset is less than 20. In comparison, LSA64 has 50 samples per class, and the Chinese isolated SLR has 250.

Table 2 Training sample sizes and testing sample sizes video-level
<table><tr><td></td><td>LSA</td><td>CSL</td><td>ASL for LSA to ASL DA</td><td>ASL for CSL to ASL DA</td></tr><tr><td>Training Sample Size</td><td>1040</td><td>4600</td><td>284</td><td>257</td></tr><tr><td>Testing Sample Size</td><td>260</td><td>1,150</td><td>71</td><td>64</td></tr></table>

DA stands for Domain Adaptation

Tables 3 and 4 show the number of learnable parameters in our Domain Adaptation models per N-multiscale TRN.

Table 3 Complexity of the Domain Adaptation Models for LSA to ASL. The number of parameters for each N-multiscale TRN is indicated in the first row, while the number of multiscale TRN for each model is listed in the second row.
<table><tr><td>Number of Parameters</td><td>2582844</td><td>3895616</td><td>5732676</td><td>9471306</td><td>18323796</td></tr><tr><td>N-multiscale TRN</td><td>3</td><td>5</td><td>7</td><td>10</td><td>15</td></tr></table>

Table 4 Complexity of the Domain Adaptation Models for CSL to ASL. The number of parameters for each N-multiscale TRN is indicated in the first row, while the number of multiscale TRN for each model is listed in the second row.
<table><tr><td>Number of Parameters</td><td>2580534</td><td>3893306</td><td>5730366</td><td>9468996</td><td>18321486</td></tr><tr><td>N-multiscale TRN</td><td>3</td><td>5</td><td>7</td><td>10</td><td>15</td></tr></table>

All of our experiments are benchmarked with randomized 5-fold crossvalidation, and training time is up to 100 epochs. Our training-to-testing sample size ratio is 80:20 for full-scale domain adaptation, and 20:80 for few-shot domain adaptation. We also have selected a batch size of 20.

For both full-scale and few-shot domain adaptation, we have decided to compare the results from utilizing 3, 5, 7, 10, and 15 multiscale TRNs. Aside from calculating the mean accuracy from each 5-fold cross-validation, we also computed their standard deviations.

## 4.2 Experiment Results and Analysis

For our experiment, we aim to compare the performance between (1) domain adaptation and neural network-based transfer learning, (2)full-scale and fewshot domain adaptation, and (3) N-diferent multi-scale TRNs. Our analysis will first highlight the best N-multiscale TRN for each group of the experiment: (1) full-scale domain adaptation between LSA to ASL, (2) full-scale domain adaptation between CSL to ASL, (3) few-shot domain adaptation between LSA to ASL, and (4) few-shot domain adaptation between CSL to ASL. Then, in Section 4.3, we will discuss how domain adaptation outperforms neural network-based transfer learning.

We are not only considering improvements from the baseline or non-transfer learning base model in determining the best N-multiscale TRN. We are also identifying the N-multiscale TRN that produced the maximum accuracy for each group of experiments. This approach ensures that improvements over weak baselines or baselines with the lowest classification accuracy are not misleading. The best N-multiscale TRN should produce a domain-adapted model with the highest classification accuracy and a significant improvement over its baseline.

Although the resulting accuracy from our experiments may appear low compared to other studies on Sign Language Recognition (SLR), our transfer learning task is particularly challenging. We experiment with both one-handed and two-handed dynamic word-level signs from the most diverse American Sign Language (ASL) dataset. This diversity includes not only signers or subjects but also video recording conditions such as aspect, lighting, and distance from the camera. These conditions persist because we collect WLASL samples from various sources, including YouTube and educational websites. To our knowledge, no cross-sign language transfer learning study has focused on two-handed dynamic signs. Table 11 summarizes the main diferences between our target domain ASL and the sign language datasets used in other relevant SLR studies. One critical factor to note is that signers in the WLASL subset we used appeared almost only once per class, with each class containing around 14 video samples. This is in contrast to Bird et al.’s study, the most relevant literature to our work, where their target ASL dataset features only two signers for all classes.

## 4.2.1 Full-scale Domain Adaptation Results

The bar charts in Figures 8 and 9 shows how the classification accuracies of the baseline ASL models and the domain-adapted ASL models change across diferent N-multiscale TRNs. Conversely, Table 5 shows all the results of conducting full-scale domain adaptation.

## Full-scale LSA to ASL domain adaptation

The bar chart in Figure 8 shows that using the 5-multiscale TRN resulted in the highest classification accuracy of 12.93% for the domain adapted model in RGB mode, in the LSA to ASL Domain Adaptation group of experiments. This improvement represents a 59.62% increase over the baseline. Among all RGB domain adaptations for this group, this improvement is the highest, making it easy to conclude that the 5-multiscale TRN is the best N-multiscale TRN for RGB mode in this group.

![](images/38b4648773f157937742b4ff0e33bebe24aadef64b95669c1293be453fb0b105.jpg)  
Fig. 8 Bar chart showing the Full-scale Domain Adaptation experiment between LSA to ASL across diferent N-multiscale TRNs

When analyzing the results of our LSA to ASL full-scale domain adaptation in Optical Flow mode, we found that the 3-multiscale TRN domain adaptation achieved a maximum classification accuracy of 10.08%, representing a 15.15% improvement from its baseline. On the other hand, the 7-multiscale TRN domain adaptation showed the highest improvement of approximately 92.017% from its corresponding baseline. In addition, the 7-multiscale TRN obtained the second-highest classification accuracy of 9.958%. Given these results, we conclude that the 7-multiscale TRN is the best N-multiscale TRN for LSA to ASL full-scale domain adaptation in Optical Flow mode.

## Full-scale CSL to ASL domain adaptation

In the bar chart shown in Figure 9, it can be observed that the 7-multiscale TRN yielded the highest classification accuracies for CSL to ASL domain adaptation in both RGB (11.86%) and optical flow mode (10.08%). Compared to the corresponding RGB baseline, this represents an improvement of 24.19%, while for optical flow, the improvement is 10.31%.

The 15-multiscale TRN helped the RGB baseline improve the most at 57.38% compared to other N-multiscale TRNs. However, the resulting domainadapted model has the second-lowest accuracy among other domain-adapted models in RGB mode at 9.906%. Therefore, for RGB mode, 7-multiscale TRN is the best for this group of experiments.

In contrast, for optical flow mode, the 7-multiscale TRN was not the most efective as the 3-multiscale TRN produced an improvement of nearly 20% from the baseline. Although the 7-multiscale TRN achieved the highest classification accuracy of 10.08%, the 3-multiscale TRN came in a close second with an accuracy of 10.01%. Therefore, for this set of experiments, the 3-multiscale TRN is the most efective in domain adaptation for optical flow mode.

![](images/86db2223f91ac85687a6eef742097965422e9a1dbe76d61f51f20a7c50344698.jpg)  
Fig. 9 Bar chart showing the Full-scale Domain Adaptation experiment between CSL to ASL across diferent N-multiscale TRNs

It is worth noting that a negative transfer is observed in the optical flow input mode under 15-multiscale TRN, where the domain-adapted model’s accuracy of 8.618% is lower than its baseline accuracy of 10.112%.

Overall, shorter-term multiscale TRNs such as 3, 5, and 7, on average, outperform longer-term multiscale TRNs in terms of improvement over strong baselines in Full-scale Domain Adaptation.

Table 5 Overall results of the Full-scale Domain Adaptation experiment. The second column of the table corresponds to the relevant subsets of ASL, which were selected based on their shared classes with LSA and CSL, as shown in Table 1. The third column specifies the type of domain adaptation that was performed where Non-transfer learning refers to the baseline.
<table><tr><td>N-multiscale TRN</td><td></td><td></td><td></td><td>3</td><td>5</td><td>7</td><td>10</td><td>15</td></tr><tr><td rowspan="6">RGB</td><td>ASL with</td><td>Non-transfer learning</td><td>mean std.</td><td>8.32% 1.399</td><td>8.1% 2.502</td><td>9.364% 1.26</td><td>9.174% 1.716</td><td>8.308% 1.696</td></tr><tr><td>LSA classes</td><td></td><td>mean</td><td>11.75%</td><td>12.93%</td><td>9.564%</td><td>9.318%</td><td>9.024%</td></tr><tr><td></td><td>LSA to ASL</td><td>std.</td><td>1.501</td><td>1.821</td><td>1.91</td><td>2.583</td><td>2.571</td></tr><tr><td></td><td>Non-transfer</td><td>mean</td><td>9.524%</td><td>9.418%</td><td>9.55%</td><td>7.59%</td><td>6.294%</td></tr><tr><td>ASL with</td><td>learning</td><td>std.</td><td>1.64</td><td>1.261</td><td>4.922</td><td>0.396</td><td>0.5</td></tr><tr><td>CSL classes</td><td>CSL to ASL</td><td>mean</td><td>10.46%</td><td>10.55%</td><td>11.86%</td><td>8.56%</td><td>9.906%</td></tr><tr><td></td><td></td><td>Non-transfer</td><td>std. mean</td><td>0.602 8.754%</td><td>3.321</td><td>1.772</td><td>3.00</td><td>2.688</td></tr><tr><td rowspan="5"></td><td rowspan="5">ASL with LSA classes</td><td>learning</td><td>std.</td><td></td><td>7.79%</td><td>5.186%</td><td>7.618%</td><td>7.16%</td></tr><tr><td></td><td></td><td>3.242</td><td>2.919</td><td>0.451</td><td>3.173</td><td>2.452</td></tr><tr><td>LSA to ASL</td><td>mean</td><td>10.08%</td><td>9.412%</td><td>9.958%</td><td>9.756%</td><td>8.45%</td></tr><tr><td></td><td>std.</td><td>1.33</td><td>2.45</td><td>1.99</td><td>2.129</td><td>1.123</td></tr><tr><td>Non-transfer</td><td>mean</td><td>8.342%</td><td>7.538%</td><td>9.138%</td><td>6.672%</td><td>10.112%</td></tr><tr><td rowspan="4"></td><td rowspan="4">ASL with CSL classes</td><td>learning</td><td>std.</td><td>1.341</td><td>1.546</td><td>2.33</td><td>0.585</td><td>1.3</td></tr><tr><td></td><td>mean</td><td>10.01%</td><td>9.076%</td><td>10.08%</td><td>6.6832%</td><td>8.618%</td></tr><tr><td>CSL to ASL</td><td>std.</td><td>1.309</td><td>2.039</td><td>1.047</td><td>1.047</td><td>1.047</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Standard deviations correspond to the mean of the 5-fold cross-validation results of each N-multiscale TRN domain adaptation

## 4.2.2 Few-shot Domain Adaptation

The bar charts in Figures 10 and 11 show how the classification accuracy changes between diferent N-multiscale TRNs in a few-shot learning setting. Conversely, Table 6 shows all the results of our few-scale domain adaptations.

## Few-shot LSA to ASL domain adaptation

The bar chart from Figure 10 shows that for RGB input mode, 3-multiscale TRN achieved the maximum classification accuracy of 6.85%. Its corresponding baseline is the strongest among all the other comparable baselines, with a classification accuracy of 6.4%. The percentage diference between this baseline and the domain-adapted model is 7.03%. From this figure, it is not dificult to conclude that 3-multiscale TRN is the most efective N-multiscale TRN for LSA to ASL few-shot domain adaptation in RGB mode.

Meanwhile, for the Optical Flow mode, 10-multiscale TRN produced the maximum classification accuracy of 7.34% for the domain-adapted model. This is a 26.48% increase from the corresponding baseline, which shows the highest improvement in the group. We consider 10-multiscale TRN as the best N-multiscale TRN for this domain adaptation, as it generated the highest classification accuracy as well as the highest improvement over its baseline in this group of experiments.

![](images/d9e286f49d7f4408bba8ed1f45297d95169eb787fbd5147114094f303a1be7b0.jpg)  
Fig. 10 Bar chart showing the Few-shot Domain Adaptation experiment between LSA to ASL across diferent N-multiscale TRNs

## Few-shot CSL to ASL domain adaptation

Figure 11 shows that for CSL to ASL few-shot domain adaptation, the 7- multiscale TRN produced the maximum classification accuracies for all input modes (RGB and Optical Flow). For RGB, the maximum classification accuracy is 9.266% which is an improvement of 3.41% from its corresponding baseline, whose accuracy is 8.96%. For Optical Flow, the maximum classification accuracy achieved is 7.948%. This accuracy is a 4.55% increase from the baseline of 7.602%.

![](images/7d10cd60bd68353f8b67735df5432fa79336b044337952b473da45057e98a817.jpg)  
Fig. 11 Bar chart showing the Few-shot Domain Adaptation experiment between CSL to ASL across different N-multiscale TRNs

The 5-multiscale TRN yielded the most improvement of approximately 19.32% from the RGB baseline. However, its corresponding domain-adapted model has the second-lowest classification accuracy among the group at 8.572%. Therefore, for RGB mode, the 7-multiscale TRN is considered best.

For optical flow mode, we consider the 3-multiscale TRN to be the most efective. The reason is that its domain-adapted model has the second-highest classification accuracy in the group at 7.788% which translated to an improvement of 12.54%. Having the second-highest classification accuracy and the highest baseline improvement made 3-multiscale TRN the best for CSL to ASL few-shot domain adaptation.

Similar to full-scale CSL to ASL Domain Adaptation (see Figure 9), the 15-multiscale TRN produced domain-adapted models with lower classification accuracy than its baseline (8.796% vs 8.95%). Also akin to full-scale domain adaptation, shorter-term multiscale TRNs such as 3, 5, and 7, on average, outperformed longer-term multiscale TRNs in terms of improvement over strong baselines in Few-scale Domain Adaptation as seen in Figures 9 and 8.

Conducting full-scale learning has resulted in domain-adapted models with higher accuracy. Moreover, based on the results of these experiments, it can be concluded that RGB is still superior to optical flow in terms of producing better domain-adapted models. Although LSA and CSL are not related sign languages to ASL, it can be observed that LSA and ASL in general share more similarities in gestures than CSL and ASL. For instance, we discovered that signs such as “Deaf”, “Call”, “Candy”, “Catch”, and “Copy” in our LSA and ASL subsets share notable similarities. Therefore, it is not surprising that

Table 6 Overall results of the Few-shot Domain Adaptation experiment. The second column of the table corresponds to the relevant subsets of ASL, which were selected based on their shared classes with LSA and CSL, as shown in Table 1. The third column specifies the type of domain adaptation that was performed where Non-transfer learning refers to the baseline.
<table><tr><td>N-multiscale TRN</td><td></td><td></td><td></td><td>3</td><td>5</td><td>7</td><td>10</td><td>15</td></tr><tr><td rowspan="6">RGB</td><td>ASL with</td><td>Non-transfer learning</td><td>mean std.</td><td>6.4% 0.998</td><td>6.386% 0.929</td><td>6.078% 0.77</td><td>5.88% 0.543</td><td>6.32% 0.48</td></tr><tr><td>LSA classes</td><td></td><td>mean</td><td>6.85%</td><td>6.764%</td><td>6.24%</td><td>6.022%</td><td>6.702%</td></tr><tr><td></td><td>LSA to ASL</td><td>std.</td><td>0.821</td><td>0.686</td><td>0.537</td><td>0.588</td><td>0.267</td></tr><tr><td></td><td>Non-transfer</td><td>mean</td><td>7.816%</td><td>7.184%</td><td>8.96%</td><td>8.792%</td><td>8.958%</td></tr><tr><td>ASL with</td><td>learning</td><td>std.</td><td>1.54</td><td>1.3</td><td>2.234</td><td>2.099</td><td>1.998</td></tr><tr><td>CSL classes</td><td>CSL to ASL</td><td>mean</td><td>8.056%</td><td>8.572%</td><td>9.266%</td><td>8.842%</td><td>8.796%</td></tr><tr><td rowspan="6"></td><td>ASL with</td><td>Non-transfer</td><td>std. mean</td><td>0.57 6.268%</td><td>1.245 6.58%</td><td>0.881 7.108%</td><td>0.792 5.806%</td><td>1.584 5.714%</td></tr><tr><td>LSA classes</td><td>learning</td><td>std.</td><td>1.132</td><td>1.166</td><td>0.884</td><td>0.906</td><td>1.122</td></tr><tr><td></td><td>LSA to ASL</td><td>mean</td><td>7.184%</td><td>6.654%</td><td>7.262%</td><td>7.344%</td><td>6.738%</td></tr><tr><td></td><td></td><td>std.</td><td>0.288</td><td>1.357</td><td>0.236</td><td>1.1</td><td>0.788</td></tr><tr><td>ASL with</td><td>Non-transfer</td><td>mean</td><td>6.92%</td><td>7.364%</td><td>7.602%</td><td>7.248%</td><td>7.004%</td></tr><tr><td>CSL classes</td><td>learning</td><td>std.</td><td>2.688</td><td>0.018</td><td>1.437</td><td>1.999</td><td>0.872</td></tr><tr><td></td><td></td><td>CSL to ASL</td><td>mean std.</td><td>7.788% 0.267</td><td>7.684% 0.548</td><td>7.948% 0.601</td><td>7.536% 0.619</td><td>7.34% 0.257</td></tr></table>

Standard deviations correspond to the mean of the 5-fold cross-validation results of each N-multiscale TRN domain adaptation

LSA as a source helped ASL to reach the highest accuracy in all domain adaptation experiments (see Table 5). However, LSA underperformed in our few-shot domain adaptation experiments. The reason is that, since we are only using 20% of the ASL samples for target training in few-shot learning settings, LSA, being a smaller source domain than CSL, was unable to provide suficient representations for ASL to learn.

## 4.3 Pre-training

As part of our study, we also experimented with the pre-training approach wherein the weights learned by a network from the source domain are used as initial weights for training the target domain. This transfer learning method is used by Bird et al. [4], the first paper to study transfer learning between two sign languages, and other studies that use transfer learning from large-scale datasets such as ImageNet to improve their SLR models [1–3, 15].

To apply this transfer learning technique in our task, we needed to add a Long short-term memory layer on top of the fully-connected layer of the convolutional neural network (CNN). For this method, we utilized a Resnet-50 architecture [32]. We have decided to use this architecture as it is the most popular deep residual network.

Similar to our domain adaptation experiment detailed in Section 4, all of our pre-training experiments are benchmarked with randomized 5-fold crossvalidation, and training and testing are continued until there is no more improvement in accuracy after 100 epochs. We also have selected a batch size of 2. For full-scale learning settings, the training-to-test sample size ratio is 80:20 while for few-shot learning settings, the sample size ratio is 20:80.

Table 7 5-fold classification accuracies of RGB to RGB and optical flow to optical flow of ASL pre-trained with LSA and CSL as well as the ASL baseline in optical flow (OF) and RGB modes in the full-scale learning setting.
<table><tr><td colspan="2"></td><td>ASL Baseline for LSA to ASL</td><td>LSA to ASL</td><td>ASL Baseline for CSL to ASL</td><td>CSL to ASL</td></tr><tr><td rowspan="3">RGB</td><td>mean</td><td>6.926%</td><td>10.676%</td><td>9.832%</td><td>9.518%</td></tr><tr><td>std.</td><td>2.129</td><td>7.209</td><td>4.645</td><td>4.807</td></tr><tr><td>mean</td><td>10.344%</td><td>6.094%</td><td>15.998%</td><td>11.748%</td></tr><tr><td rowspan="2">OF</td><td>std.</td><td>7.409</td><td>1.601</td><td>8.425</td><td>7.489</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/12e30e083aa9f43c3b9b60365533ded81526a5d272843167db23a1fd51de74a7.jpg)  
Fig. 12 Bar chart showing the accuracies of ASL when pre-trained with LSA and CSL in optical flow (OF) and RGB modes in the full-scale learning setting. Baseline results are also indicated

From Table 7 and Figure 12, we can observe that only ASL pre-trained with LSA in RGB mode produced a positive transfer outcome in full-scale learning settings. On the other hand, for few-shot learning settings, only ASL pre-trained with CSL in optical flow mode resulted in positive transfer as seen in Table 8 and Figure 13.

Table 8 5-fold classification accuracies of RGB to RGB and optical flow to optical flow of ASL pre-trained with LSA and CSL as well as the ASL baseline in optical flow (OF) and RGB modes in the few-shot learning setting.
<table><tr><td colspan="2"></td><td>ASL Baseline for LSA to ASL</td><td>LSA to ASL</td><td>ASL Baseline for CSL to ASL</td><td>CSL to ASL</td></tr><tr><td rowspan="3">RGB</td><td>mean</td><td>6.686%</td><td>6.602%</td><td>7.818%</td><td>7.59%</td></tr><tr><td>std.</td><td>1.422</td><td>1.257</td><td>2.201</td><td>2.322</td></tr><tr><td>mean</td><td>6.84%</td><td>6.086%</td><td>7.198%</td><td>7.504%</td></tr><tr><td rowspan="2">OF</td><td>std.</td><td>1.826</td><td>1.609</td><td>1.828</td><td>2.615</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/34d5e10b333114d2899f44374237df78b7a7869ff7278930d7e85991c58edbb0.jpg)  
Fig. 13 Bar chart showing the accuracies of ASL when pre-trained with LSA and CSL in optical flow (OF) and RGB modes in the few-shot learning setting. Baseline results are also indicated

Table 9 Performance matrix of Full-scale Neural network-based Transfer Learning and Domain Adaptation
<table><tr><td>Tasks</td><td colspan="6">Accuracies</td></tr><tr><td>Baseline(LSA to ASL) RGB</td><td>6.926</td><td>8.32</td><td>8.1</td><td>9.364</td><td>9.174</td><td>8.308</td></tr><tr><td>LSA to ÁSL RGB</td><td>10.676</td><td>11.75</td><td>12.93</td><td>9.564</td><td>9.318</td><td>9.024</td></tr><tr><td>Baseline(CSL to ASL) RGB</td><td>9.832</td><td>9.524</td><td>9.418</td><td>9.55</td><td>7.59</td><td>6.294</td></tr><tr><td>CSL to ÁSL RGB</td><td>9.518</td><td>10.46</td><td>10.55</td><td>11.86</td><td>8.56</td><td>9.906</td></tr><tr><td>Baseline(LSA to ASL) OF</td><td>10.344</td><td>8.754</td><td>7.79</td><td>5.186</td><td>7.618</td><td>7.16</td></tr><tr><td>LSA to ÁSL OF</td><td>6.094</td><td>10.08</td><td>9.412</td><td>9.958</td><td>9.756</td><td>8.45</td></tr><tr><td>Baseline(CSL to ASL) OF</td><td>15.998</td><td>8.342</td><td>7.538</td><td>9.138</td><td>6.672</td><td>10.112</td></tr><tr><td>CSL to ÁSL OF</td><td>11.748</td><td>10.01</td><td>9.076</td><td>10.08</td><td>6.6832</td><td>8.618</td></tr><tr><td></td><td>NN</td><td>DA 3-TRN</td><td>DA 5-TRN</td><td>DA 7-TRN</td><td>DA 10-TRN</td><td>DA 15-TRN</td></tr></table>

NN: Neural Network

Table 10 Performance matrix of Few-shot Neural network-based Transfer Learning and Domain Adaptation
<table><tr><td>Tasks</td><td colspan="6">Accuracies</td></tr><tr><td>Baseline(LSA to ASL) RGB</td><td>6.686</td><td>6.4</td><td>6.386</td><td>6.078</td><td>5.88</td><td>6.32</td></tr><tr><td>LSA to ÀSL RGB</td><td>6.602</td><td>6.85</td><td>6.764</td><td>6.24</td><td>6.022</td><td>6.702</td></tr><tr><td>Baseline(CSL to ASL) RGB</td><td>7.818</td><td>7.816</td><td>7.184</td><td>8.96</td><td>8.792</td><td>8.958</td></tr><tr><td>CSL to ASL RGB</td><td>7.59</td><td>8.056</td><td>8.572</td><td>9.266</td><td>8.842</td><td>8.796</td></tr><tr><td>Baseline(LSA to ASL) OF</td><td>6.84</td><td>6.268</td><td>6.58</td><td>7.108</td><td>5.806</td><td>5.714</td></tr><tr><td>LSA to ÀSL OF</td><td>6.086</td><td>7.184</td><td>6.654</td><td>7.262</td><td>7.344</td><td>6.738</td></tr><tr><td>Baseline(CSL to ASL) OF</td><td>7.198</td><td>6.92</td><td>7.364</td><td>7.602</td><td>7.248</td><td>7.004</td></tr><tr><td>CSL to ÀSL OF</td><td>7.504</td><td>7.788</td><td>7.684</td><td>7.948</td><td>7.536</td><td>7.34</td></tr><tr><td></td><td>NN</td><td>DA 3-TRN</td><td>DA 5-TRN</td><td>DA 7-TRN</td><td>DA 10-TRN</td><td>DA 15-TRN</td></tr></table>

NN: Neural Network

Pre-training in both full-scale and few-shot learning settings has resulted in more negative transfers or lower accuracy than positive transfers compared to their respective baselines. In contrast, for domain adaptation, only two instances of 15-multiscale TRN resulted in negative adaptations, while 18 out of 20 N-multiscale TRNs showed improvements in their baselines. Thus, for pre-training, improvements occurred only 25% of the time, compared to 90% for domain adaptation.

The standard deviations in this experiment are consistent with those observed in the domain adaptation experiments presented in Section 4.2, where the test results in the few-shot learning setting were closer to the mean than those in the full-scale learning setting.

<sub>parison</sub> <sub>of</sub> <sub>our</sub> m<sup>odel</sup> <sup>to</sup> <sup>other</sup> <sup>related</sup> <sup>Sign</sup> <sup>Language</sup> <sup>Recog</sup>
<table><tr><td>Study</td><td>Source Domain</td><td>Target Domain</td><td>Level</td><td>Dynamic</td><td>Two-handed</td><td>Signers</td><td>Number of times a subject repeat a sign</td><td>Classes</td><td>Accuracy (Highest Score Achieved)</td></tr><tr><td>Bird et al.[4]</td><td>BSL</td><td>ASL</td><td>Word</td><td>Yes</td><td>No</td><td>25</td><td>28</td><td>18</td><td>94.44%</td></tr><tr><td>Halvardsson et al.[15]</td><td>ImageNet</td><td>SSL</td><td>Alphabet</td><td>No</td><td>No</td><td>Implied as many.</td><td></td><td>26</td><td>85%</td></tr><tr><td>Cayamcela et al.[2]</td><td>ImageNet</td><td>ASL</td><td>Alphabet</td><td>No</td><td>No</td><td>Exact number not indicated</td><td>Not indicated</td><td>26</td><td>99.39%</td></tr><tr><td>Nishat et al.[3]</td><td>ImageNet</td><td>BSDL</td><td>digits, vowel,</td><td>No</td><td>Yes</td><td>Implied as many. Exact number</td><td>Not indicated</td><td>46</td><td>96.57%</td></tr><tr><td>Farhadi et al.[13]</td><td>ASL Avatar</td><td>ASL</td><td>consonants Word</td><td>Yes</td><td>Yes</td><td>not indicated 1</td><td></td><td>50</td><td>64.17%</td></tr><tr><td>Vazquez-Enriquez et al.[33]</td><td>ASL</td><td>Turkish and Spanish Sign Language</td><td>Word</td><td>Yes</td><td>Yes</td><td>43</td><td>Not indicated</td><td>226</td><td>95.24% (Turkish), 93.91% (Spanish)</td></tr><tr><td>Zakariah et al. [16]</td><td>Imagenet</td><td>Arabic Sign Language</td><td>Alphabet</td><td>No</td><td>No</td><td>40</td><td>Not indicated</td><td>32</td><td>95%</td></tr><tr><td>Shania et al. [17]</td><td>Imagenet Imagenet,</td><td>Indonesian Sign Language</td><td>Word</td><td>Yes</td><td>Yes</td><td>¿2</td><td>Not Indicated</td><td>11</td><td>98.5%</td></tr><tr><td>Abdullayeva et al. [22]</td><td>ASL letters</td><td>Azerbaijan Sign Language</td><td>Alphabet</td><td>Yes</td><td>No</td><td>Not indicated</td><td>Not indicated</td><td>32</td><td>88%</td></tr><tr><td>Thakar et al. [18] Das et al. [14]</td><td>Imagenet</td><td>ASL</td><td>Alphabet</td><td>No</td><td>No</td><td>Not indicated</td><td>Not indicated</td><td>29</td><td>98.7%</td></tr><tr><td></td><td>Imagenet</td><td>BsDL</td><td>Digit and characters Finger Signs</td><td>No</td><td>Yes</td><td>Not indicated</td><td>Not indicated</td><td>36</td><td>91.67% (character), 97.33% (digit)</td></tr><tr><td>Jiang et al. [19]</td><td>Imagenet</td><td>CSL</td><td>(A to Z,</td><td>No</td><td>No</td><td>Not indicated</td><td>Not indicated</td><td>30</td><td>91.48%</td></tr><tr><td>Sharma et al. [20] Suharjito et al. [21]</td><td>Imagenet</td><td>Indian Sign Language</td><td>ZH, CH, SH, and NG) Alphabet and digits</td><td>No</td><td>Not indicated</td><td>Not indicated</td><td>Not indicated</td><td>35</td><td>100%</td></tr><tr><td></td><td>Imagenet and Kinetic</td><td>Indonesian Signal System</td><td>Not indicated</td><td>Yes</td><td>Not indicated</td><td>2 109 for WLASL300.</td><td>10</td><td>10</td><td>97.50%</td></tr><tr><td>Our study</td><td>LSA, CSL</td><td>ASL</td><td>Word</td><td>Yes</td><td>Yes</td><td>For our subsets, on average 11 signers</td><td>Average 1.3</td><td>23</td><td>12.93% (LSA), 11.86% (CSL)</td></tr></table>

<sub>ourstudyreferstotheaveragenumberofsignersforeachclassofoursubsets</sub>.<sub>Thenu</sub>m<sup>beroftim</sup> <sub>mes</sub> <sub>a</sub> <sub>participant</sub> <sub>recorded</sub> <sub>a</sub> <sub>par</sub><sup>ticular</sup> <sup>sign</sup>. <sup>In</sup> <sup>the</sup> <sup>classes</sup> <sup>column</sup>, <sup>we</sup> <sup>wrote</sup> <sup>23</sup> <sup>and</sup> <sup>26</sup> <sup>as</sup> <sup>we</sup> <sup>hav</sup> <sub>nd</sub> <sub>one</sub> <sub>for</sub> <sub>CSL</sub> <sub>to</sub> <sub>ASL</sub> <sub>transfer</sub>. <sub>The</sub> <sub>average</sub> <sub>number</sub> o<sup>f</sup> <sup>samples</sup> <sup>for</sup> <sup>each</sup> <sup>class</sup> <sup>of</sup> <sup>our</sup> <sup>subsets</sup> <sup>is</sup> <sup>14</sup>. <sub>ain</sub>, <sub>consists</sub> <sub>of</sub> <sub>109</sub> <sub>signers</sub> <sub>and</sub> <sub>a</sub> <sub>mean</sub> <sub>of</sub> <sub>17</sub> <sub>sa</sub>m<sup>ples</sup>. <sup>However</sup>, <sup>by</sup> <sup>the</sup> <sup>time</sup> <sup>of</sup> <sup>this</sup> <sup>study</sup>, <sup>severa</sup> <sub>Tube</sub> <sub>and</sub> <sub>other</sub> <sub>educa</sub>t<sup>ional</sup> <sup>sign</sup> <sup>language</sup> <sup>websites</sup> <sup>are</sup> <sup>no</sup> <sup>lo</sup>

## 5 Conclusion

Our research on transfer learning approaches between diferent sign languages provided insights on how to improve classification and recognition in lowresource conditions by aligning the source sign language’s spatiotemporal features with the target sign language’s in a multiple-timescale fashion. From our best knowledge, we are the first to study the knowledge adaptability between LSA (Argentine Sign Language) to ASL (American Sign Language) and CSL (Chinese Sign Language) to ASL in both RGB and optical flow modes. We are also the first to conduct a few-shot domain adaptation between sign languages. Applying few-shot learning in SLR is important to advance the field and place it at the levels of other fields such as computational linguistics, natural language processing, and other action recognition areas where many studies are being conducted on the application of few-shot learning.

The empirical evidence resulting from our study showed that multi-scale temporal Domain Adaptation between LSA and ASL, and CSL and ASL performed significantly better than pre-training ASL with CSL and LSA. The pre-training approach yielded more negative transfers than positive ones. We were also able to prove our hypothesis that aligning shorter-term multiscale TRNs such as 3, 5, or 7 would be more beneficial for the domain adaptation between these sign languages than aligning longer-term multi-scale TRNs such as 10 or 15. This hypothesis holds for all of our experiment settings, such as full-scale and few-shot domain adaptation, and RGB and optical flow input modes (see Section 4.2). This is a piece of great news for the fields of SLR and action recognition in general as training longer-term multi-scale TRNs consume more GPU resources and time to complete. Moreover, we believe that longer-term multi-scale TRNs will only outperform shorter-term ones in cases where the source and target domains bear more similarities than diferences.

Our study also proved our main hypothesis that adapting the domain of one sign language to another - even if they are not from the same sign language family would make a better SLR model than not doing any domain adaptation at all or only pre-training the target domain with another sign language or with a generic large-scale dataset such as ImageNet.

Although our experiment showed that RGB performed better than optical flow (see Section 4.2), we believe that our study opened more possibilities for the application of optical flow in the field of SLR. In the future, we plan to conduct studies that aim to discover more potential in this type of motion representation.

Lastly, we envision that our study will be a valuable resource in the realtime translation of sign gestures into words or phrases, or in transcribing videos that feature sign languages. However, to efectively model sign languages, a language modeling technique must be utilized due to diferences in grammar structure between spoken languages and sign languages. Nonetheless, these grammar diferences can be addressed through domain adaptation. One example is adapting from a sentence-level sign language dataset with spoken language transcription and gloss-level annotation to a dataset with only gloss annotation, to automatically convert gloss-level annotations into transcriptions. Finding similar gloss annotations across datasets is a significant challenge. Alternatively, adapting a non-sign language video with transcriptions to sentence-level sign language videos with gloss-level annotations may be explored in future work.

## Declarations

• Conflict of interest/Competing interests

The authors have no relevant financial or non-financial interests to disclose.

• Funding

This research was funded by the Shenzhen Science and Technology Innovation Commission (JCYJ20210324135011030), Science and Technology Innovation Committee of Shenzhen-Platform and Carrier (International Science and Technology Information Center), High-end Foreign Expert Talent Introduction Plan (G2021032022L), Guangdong Pearl River Plan (2019QN01X890), and National Natural Science Foundation of China (Grant No. 71971127).

• Consent to participate Not applicable

• Consent for publication

Not applicable

• Availability of data and materials

All data generated or analyzed during this study are included in these published articles [8–12] (and its supplementary information files). The subsets we used are detailed in Section 4.1. For additional guidance on extracting the subsets from their originating datasets, please contact the authors.

• Code availability

The codes used for domain adaptation are based on TA3N [27]. Our modification includes setting the batch size to 20, the mode of learning to supervised learning, and the value of num segments to the N-multiscale TRN. The codes for converting videos into RGB and Optical Flow frames are available from this repository, https://doi.org/10.6084/m9.figshare.20223444 . For additional guidance, please contact the authors.

All authors contributed to the study’s conception and design. Material preparation, data collection, and analysis were performed by Keren Artiaga, Yang Li, Ercan Engin Kuruoglu, and Wai Kin (Victor) Chan. The first draft of the manuscript was written by Keren Artiaga and all authors commented on previous versions of the manuscript. All authors read and approved the final manuscript.

## References

[1] Rathi, D.: Optimization of transfer learning for sign language recognition targeting mobile platform. In: International Journal on Recent and Innovation Trends in Computing and Communication, vol. 6, pp. 198–203 (2018)

[2] Morocho-Cayamcela, M.E., Lim, W.: Fine-tuning a pre-trained convolutional neural network model to translate american sign language in real-time. 2019 International Conference on Computing, Networking and Communications (ICNC), 100–104 (2019)

[3] Nishat, Z.K., Shopon, M.: Unsupervised pretraining and transfer learningbased bangla sign language recognition. In: Proceedings of International Joint Conference on Computational Intelligence Algorithms for Intelligent Systems, pp. 529–540 (2020). https://doi.org/10.1007/ 978-981-15-3607-6 42

[4] Bird, J.J., Ek´art, A., Faria, D.R.: British sign language recognition via late fusion of computer vision and leap motion with transfer learning to american sign language. Sensors 20 (2020)

[5] Rahman, M.M., Mdrafi, R., Gurbuz, A.C., Malaia, E., Crawford, C., Grifin, D., Gurbuz, S.Z.: Word-level sign language recognition using linguistic adaptation of 77 ghz fmcw radar data. In: 2021 IEEE Radar Conference (RadarConf21), pp. 1–6 (2021). https://doi.org/10.1109/ RadarConf2147009.2021.9455190

[6] Farneb¨ack, G.: Two-frame motion estimation based on polynomial expansion. In: SCIA, pp. 363–370 (2003)

[7] Sevilla-Lara, L., Liao, Y., G¨uney, F., Jampani, V., Geiger, A., Black, M.J.: On the integration of optical flow and action recognition. In: GCPR, pp. 281–297 (2018)

[8] Ronchetti, F., Quiroga, F., Estrebou, C., Lanzarini, L., Rosete, A.: Lsa64: A dataset of argentinian sign language. XX II Congreso Argentino de Ciencias de la Computaci´on (CACIC), 794–803 (2016)

[9] Zhang, J., Zhou, W., Xie, C., Pu, J., Li, H.: Chinese sign language recognition with adaptive hmm. In: 2016 IEEE International Conference on Multimedia and Expo (ICME), pp. 1–6 (2016). https://doi.org/10.1109/ ICME.2016.7552950

[10] Pu, J., Zhou, W., Li, H.: Sign language recognition with multi-modal features. In: PCM, pp. 252–261 (2016)

[11] Huang, J., Zhou, W., Zhang, Q., Li, H., Li, W.: Video-based sign language recognition without temporal segmentation. In: Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence and Thirtieth Innovative Applications of Artificial Intelligence Conference and Eighth AAAI Symposium on Educational Advances in Artificial Intelligence. AAAI’18/IAAI’18/EAAI’18, pp. 2257–2264 (2018)

[12] Li, D., Rodriguez, C., Yu, X., Li, H.: Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison. In: The IEEE Winter Conference on Applications of Computer Vision, pp. 1459–1469 (2020)

[13] Farhadi, A., Forsyth, D., White, R.: Transfer learning in sign language. 2007 IEEE Conference on Computer Vision and Pattern Recognition, 1–8 (2007). https://doi.org/10.1109/cvpr.2007.383346

[14] Das, S., Imtiaz, M.S., Neom, N., Siddique, N., Wang, H.: A hybrid approach for bangla sign language recognition using deep transfer learning model with random forest classifier. Expert Syst. Appl. 213, 118914 (2022)

[15] Halvardsson, G., Peterson, J., Soto-Valero, C., Baudry, B.: Interpretation of swedish sign language using convolutional neural networks and transfer learning, p. 207 (2021). https://doi.org/10.1007/s42979-021-00612-w

[16] Zakariah, M., Alotaibi, Y.A., Koundal, D., Guo, Y., Elahi, M.M.: Sign language recognition for arabic alphabets using transfer learning technique. Computational Intelligence and Neuroscience 2022 (2022)

[17] Shania, S., Naufal, M.F., Prasetyo, V.R., Azmi, M.S.B.: Translator of indonesian sign language video using convolutional neural network with transfer learning. Indonesian Journal of Information Systems (2022)

[18] Thakar, S., Shah, S., Shah, B., Nimkar, A.V.: Sign language to text conversion in real time using transfer learning. 2022 IEEE 3rd Global Conference for Advancement in Technology (GCAT), 1–5 (2022)

[19] Jiang, X., Hu, B., Satapathy, S.C., Wang, S., Zhang, Y.: Fingerspelling identification for chinese sign language via alexnet-based transfer learning and adam optimizer. Sci. Program. 2020, 3291426–1329142613 (2020)

[20] Sharma, C.M., Tomar, K., Mishra, R.K., Chariar, V.M.: Indian sign language recognition using fine-tuned deep transfer learning model. SSRN Electronic Journal (2021)

[21] Suharjito, Thiracitta, N., Gunawan, H.: Sibi sign language recognition using convolutional neural network combined with transfer learning and

non-trainable parameters. Procedia Computer Science 179, 72–80 (2021)

[22] Abdullayeva, G.G., Alishzade, N.O.: Transfer learning for azerbaijani sign language recognition. Informatics and Control Problems (2022)

[23] Sultani, W., Saleemi, I.: Human action recognition across datasets by foreground-weighted histogram decomposition. 2014 IEEE Conference on Computer Vision and Pattern Recognition, 764–771 (2014)

[24] Xu, T., Zhu, F., Wong, E.K., Fang, Y.: Dual many-to-one-encoder-based transfer learning for cross-dataset human action recognition. Image Vis. Comput. 55, 127–137 (2016)

[25] Jamal, A., Namboodiri, V.P., Deodhare, D., Venkatesh, K.S.: Deep domain adaptation in action space. In: BMVC (2018)

[26] Sahoo, A., Shah, R., Panda, R., Saenko, K., Das, A.: Contrast and mix: Temporal contrastive video domain adaptation with background mixing. In: Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P.S., Vaughan, J.W. (eds.) Advances in Neural Information Processing Systems, vol. 34, pp. 23386–23400 (2021)

[27] Chen, M.H., Kira, Z., Al-Regib, G., Yoo, J., Chen, R., Zheng, J.: Temporal attentive alignment for large-scale video domain adaptation. 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 6320–6329 (2019)

[28] Zhou, B., Andonian, A., Oliva, A., Torralba, A.: Temporal relational reasoning in videos. European Conference on Computer Vision, 831–846 (2018)

[29] Wang, Y., Yao, Q., Kwok, J.T.-Y., Ni, L.M.: Generalizing from a few examples. ACM Computing Surveys (CSUR) 53, 1–34 (2020)

[30] Kuehne, H., Jhuang, H., Garrote, E., Poggio, T., Serre, T.: Hmdb: A large video database for human motion recognition. In: 2011 International Conference on Computer Vision, pp. 2556–2563 (2011). https://doi.org/ 10.1109/ICCV.2011.6126543

[31] Soomro, K., Zamir, A.R., Shah, M.: UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild (2012)

[32] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778 (2016)

[33] V´azquez-Enr´ıquez, M., Alba-Castro, J.L., Doc´ıo-Fern´andez, L.,

Rodr´ıguez-Banga, E.: Isolated sign language recognition with multi-scale spatial-temporal graph convolutional networks. In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 3457–3466 (2021). https://doi.org/10.1109/CVPRW53098.2021.00385
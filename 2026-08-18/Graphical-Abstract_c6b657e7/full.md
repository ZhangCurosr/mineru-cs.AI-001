## Graphical Abstract

Degradation-Aligned Self-Supervised Learning for State of Health Estimation of Lithium-Ion Batteries under Label Sparsity

Jiaqi Yao, Julia Kowal

![](images/df38c0ac1c135baa3e9f850ccbb45e2cad16c25dcc57b4526608c6ccb2ed5ed8.jpg)

![](images/6ff2b5c41a41ce13a27c7fcf0f7c83fff3cddfeeff3f8228c58411fbdb6b76f5.jpg)

![](images/29ecff7a1c0a15beefd81247f71fcbca5a5808596abdd5321e73e28c21db3c72.jpg)

![](images/0f6e1779ca2f8c30ed43804c7fb5f9f204d254a04b9a037356a9d84a23abcf70.jpg)

![](images/e349901a5557ccb956578c6a92895d4c56bdccbfce90f8da1c71887e618caaab.jpg)

## Highlights

Degradation-Aligned Self-Supervised Learning for State of Health Estimation of Lithium-Ion Batteries under Label Sparsity

Jiaqi Yao, Julia Kowal

• A ranking-based SSL framework is proposed for SOH estimation under label sparsity.

• A CNN-GRU model is developed for SOH estimation with local feature fusion.

• Insights are presented on the influences of label distribution of degradation data.

# Degradation-Aligned Self-Supervised Learning for State of Health Estimation of Lithium-Ion Batteries under Label Sparsity

Jiaqi Yao<sup>a,∗</sup>, Julia Kowal<sup>a</sup>

<sup>a</sup>Department of Electrical Energy Storage Technology (EET), Technische Universität Berlin, Einsteinufer 11, 10587, Berlin, Germany

## Abstract

An accurate estimation of the state of health (SOH) underpins a safe and optimized use of the battery system. Although compelling, data-driven SOH estimation models typically require large amounts of high-quality labeled cycling data, while in practice such labels are often sparse in both quantity and coverage. Therefore, in this work, we propose a degradation-aligned self-supervised learning (SSL) framework based on a convolutional neural network-gated recurrent unit (CNN-GRU) model, which learns aging-consistent representations from unlabeled data through a cycle-order ranking objective as the pretext task for pretraining, thereby enabling robust SOH estimation after fine-tuning on sparsely labeled data. Test results showcase that the proposed ranking-based SSL approach proves to endow the pretrained model with degradationaligned information from unlabeled data, and after fine-tuning the model can carry out accurate, robust SOH estimation, even when only an extremely limited amount of 1% of unevenly distributed labeled training data is available, where the MAE of 1.718% and RMSE of 2.329% can be achieved on the test cell. In addition, indepth analyses are presented regarding the influences of label distribution of battery degradation data. We believe this work could shed new light on SOH estimation of lithium-ion batteries under label sparsity in real-world applications.

Keywords: Lithium-ion batteries, Battery management systems, State of health estimation, Deep learning, Self-supervised learning

## 1. Introduction

The growing urgency of global climate change has significantly accelerated the transition to clean, sustainable energy systems [1, 2, 3, 4]. Renewable energy sources such as wind [5] and solar power [6, 7] have been rapidly developed and deployed worldwide. However, their inherent intermittency and volatility pose significant challenges to the stability and reliability as an energy supply, thereby creating a strong demand for dependable, eficient energy storage systems [8]. Benefiting from many advantages, including high energy and power densities [9, 10], low self-discharge [11], and longevity [12], lithium-ion batteries have been widely utilized across numerous fields, from portable consumer electronics [13, 14] to electric transportation [15, 16]. Regardless of the specific application, an accurate determination of the state of health (SOH) is indispensable for lithium-ion batteries and is a core functionality of battery management systems (BMSs). At some time $t ,$ the SOH of a battery can be defined either based on the capacity or the internal resistance [17, 18]:

$$
\large S O H _ { C } ( t ) = \frac { C _ { a c t u a l } ( t ) } { C _ { r a t e d } }\tag{1}
$$

$$
\ S O H _ { R } ( t ) = \frac { R _ { a c t u a l } ( t ) - R _ { E O L } } { R _ { r a t e d } - R _ { E O L } }\tag{2}
$$

where $C _ { a c t u a l } ( t )$ and $C _ { r a t e d }$ respectively denote the actual battery capacity at time t and the rated capacity, while $R _ { a c t u a l } ( t ) , R _ { E O L } ,$ and $R _ { r a t e d }$ respectively denote the actual battery internal resistance at time t, the predefined end-of-life (EOL) internal resistance, and the rated internal resistance. Usually, the capacity-based definition of $S O H _ { C }$ is utilized without further clarification, as in this work. SOH is a defined metric for quantifying battery degradation. The determination of SOH plays a vital role in battery systems [19, 20, 21], as it not only provides essential information for the assessment of remaining useful life (RUL) and the early scheduling of necessary maintenance, but also underpins a safe and optimized use of the battery by supporting the prevention of unexpected failures and improving the overall lifespan utilization of the battery systems.

However, SOH is closely related to the complex underlying electrochemical processes and can hardly be directly measured; thus, it has to be estimated from the measurable information during operation [22]. In general, SOH estimation approaches can be categorized into three types [23, 24]: direct measurement approaches, modelbased approaches, and data-driven approaches. Direct measurement approaches estimate the SOH based on the knowledge acquired from measurements, such as deriving the actual usable capacity by applying coulomb counting on a standardized discharge capacity test [25]. Although straightforward and easy to implement, such approaches can hardly be applied in real applications, as the fully controlled operating conditions and complete discharge cycles required are generally unavailable in practice. Model-based SOH estimation approaches can be further categorized into two groups, namely degradation modeling approaches and parameter identification approaches [24]. Degradation modeling approaches exploit various models that describe battery degradation, such as empirical models [26, 27] and electrochemical models [28, 29], to infer the current usable capacity, while parameter identification approaches cast the problem of SOH estimation as a parameter identification problem utilizing battery models like equivalent circuit models (ECMs) [30] and electrochemical models [31], and nonlinear state observers like extended Kalman filters (EKFs) [32], unscented Kalman filters (UKFs) [33], and particle filters [34]. Despite the fact that modelbased approaches are generally considered accurate and robust, in-depth domain knowledge is required for the modeling of the electrical, aging, and even thermal behavior of the battery, and specifically designed characterization tests are compulsory to parameterize these models.

In recent years, data-driven solutions have come into the spotlight for various applications. Especially in the field of deep learning, prosperous progresses are being made. In the context of battery SOH estimation, data-driven approaches aim to capture the underlying degradation patterns directly from operational data without requiring prior knowledge of battery electrochemical dynamics. However, most existing works choose to manually extract input features for data-driven models due to the belief that handcrafted health indicators can more explicitly characterize battery degradation and thus reduce the dificulty of model learning with increased interpretability [35]. In Ref. [36], the authors proposed a lightweight local health indicator extraction approach for multi-stage fast charging protocols, where the segmented charging data of certain state of charge (SOC) windows was used as the input features, which were then fed into a hybrid deep learning model for the mapping of SOH estimates. The proposed approach achieved precise SOH estimation, with mean absolute errors (MAEs) and root mean square errors (RMSEs) below 1%. In Ref. [37], the authors proposed an improved gated recurrent unit (GRU) network combined with the whale optimization algorithm for SOH estimation, where the time duration in the voltage ranges from 3.55 V to 3.75 V and from 3.80 V to 4.15 V during constant current (CC) charging was extracted as the two input features. Test results showed that the proposed method is able to achieve an average error of less than 1% and presents good generalization capability. In Ref. [38], the authors conducted a comprehensive characterization of the aging behavior patterns of lithium-ion batteries, where five aging patterns were identified, covering the rate of voltage change during discharging and CC charging, the duration of CC charging, the rate of current change during constant voltage (CV) charging, and the rate of temperature change during CV charging. These aging patterns were then used as health indicators for the input of deep neural networks (DNNs). Test results showed that the proposed health indicators consistently improved estimation performance across diferent DNN-based SOH estimators. In Ref. [39], incremental capacity analysis (ICA) was utilized for feature extraction, where the incremental capacity diferences of five voltage intervals were selected as the input feature candidates of a multi-layer perceptron (MLP) for SOH and RUL estimation. Test results showed that the proposed approach could achieve relative error rates below 3% for SOH estimation. In Ref. [40], the authors focused on real-life electric vehicle (EV) driving scenarios and proposed a snapshotbased approach with a long short-term memory (LSTM) network for flexible SOH estimation, where the voltage and current sequences of partial charging segments with a fixed window size were used as the model input. In addition, the authors fused the snapshot-based and conventional history-based approaches to develop a noise-robust approach. Test results showed that the proposed approaches were able to achieve an average error of less than 2.46% over all presented experiments. In Ref. [41], on the other hand, the authors directly took the full charging curves of voltage, current, and temperature as the input features of the proposed GRU-convolutional neural network (CNN) for SOH estimation. Test results on the National Aeronautics and Space Administration (NASA) dataset and the Oxford battery degradation dataset showcased that the proposed approach was able to achieve accurate SOH estimation with the maximum estimation error of less than 4.3%, demonstrating the fact that deep learning models are capable of automatically extracting useful features from the measured cycling data and mapping them into accurate SOH estimates.

However, data-driven SOH estimation approaches rely on large amounts of highquality labeled battery cycling data, where the SOH labels are typically obtained through standardized checkup tests that cover the entire lifespan of the batteries under controlled conditions. Such calibration tests are time-consuming and expensive for laboratories, and can seldom be conducted in real-world deployment. Furthermore, late-life SOH labels are inherently more dificult to collect, as reaching the deep-degradation regime requires extensive cycling over a long period of time with considerable experimental costs, resulting in a pronounced imbalance in label distribution across the battery lifespan. As a result, in practical application scenarios, battery aging data often face the problem of label sparsity, both in quantity and coverage, while large volumes of unlabeled operational cycling data are available. Self-supervised learning (SSL), a branch of machine learning, is specifically aimed at such challenges. SSL methods enforce the model to learn meaningful data representations without relying on manual labels by exploiting underlying data structures or automatically constructed pretext tasks derived from the unlabeled data itself [42, 43, 44]. In general, SSL techniques can be categorized into four types [44]: generative approaches, which learn representations by reconstructing the input data or predicting the missing parts of it [45, 46]; contrastive approaches, which learn representations by forcing similar samples to be closer and dissimilar samples to be more distant in the latent space [47, 48]; contrastive generative approaches, which combine both generative and contrastive objectives and exploit the advantages of both [49, 50]; context-based approaches, which learn representations by leveraging contextual information of the samples, such as temporal order and spatial structures [51, 52]. In fact, SSL techniques have been sparsely applied in some previous works in the field of battery SOH estimation to address the problem of label sparsity. In Ref. [53], the authors proposed an SSL framework utilizing a reconstruction-based generative approach with an auto-encoder-decoder, aiming to address the problems of limited labeled data and underutilized measurements during degradation. In their work, an auto-encoder-decoder was trained to reconstruct the partial capacity-voltage curve as the pretext task. After the SSL pretraining, the trained encoder was transferred to the downstream network for SOH estimation, which was then fine-tuned using the sparse labeled data. Test results showed that the proposed framework was able to achieve a robust, accurate SOH estimation using a very limited amount of labeled data. In Ref. [54], the authors proposed a self-supervised framework incorporating weak labels to reduce the demand for large amounts of annotated battery aging data for deep learning-based SOH estimation approaches, where the raw data was first processed into three-dimensional feature maps with enriched information. Afterwards, the authors exploited the generative pretext tasks of masked image reconstruction and charging capacity estimation for pretraining. Similar to the aforementioned work, a small amount of labeled data was then used for fine-tuning on the downstream task of SOH estimation. Test results showed that the proposed model demonstrated strong generalization even with only a few labeled data points. On the other hand, in Ref. [55], the authors proposed a multi-level contrastive SSL approach with dynamic embedding, aiming to leverage the multi-level physical temporal dependence of the data to solve the problem of label scarcity. Frequency domain information was extracted through the discrete Fourier transform, after which the embedding was obtained using their proposed multi-scale dynamic embedding method. Contrastive learning combined with mask reconstruction was utilized for SSL pretraining. Finally, the pretrained model was fine-tuned using limited labeled data for SOH estimation.

As a matter of fact, current SSL approaches in the field of SOH estimation tend to focus on reconstruction-based pretext tasks, where the model is trained to recover the input or its masked parts, as such strategies help the model capture general underlying patterns of the data. However, the learned representations are not necessarily aligned with the battery degradation process that is most relevant to the task of SOH estimation. In contrast, the relative aging order between cycles provides a much more direct and task-relevant self-supervised signal, since battery aging is inherently an ordered process. Therefore, in this work, we propose a degradationaligned SSL framework that learns aging-consistent representations from unlabeled CC charging curves through a cycle-order ranking objective as the pretext task for pretraining, thereby enabling robust SOH estimation after fine-tuning on only a limited amount of labeled data for practical scenarios where data labels are sparse in both quantity and coverage. In addition, we develop a CNN-GRU model that extracts local patterns from charging curves via convolutional layers and then integrates these features sequentially through recurrent units for accurate SOH estimation as well as self-supervised pretraining. To the best of our knowledge, this work is the first to exploit the intrinsic information behind the cycle order on battery aging for self-supervised SOH estimation. The main contribution of this work is as follows:

1. A ranking-based SSL framework is proposed to learn degradation-consistent representations from unlabeled CC charging curves via cycle-order ranking as pretraining for robust SOH estimation in scenarios where only sparsely labeled data are available for fine-tuning.

2. A CNN-GRU model is developed to extract and sequentially integrate the local patterns from charging curves for accurate SOH estimation as well as self-supervised pretraining.

3. In-depth analyses are presented regarding the influences of label distribution of battery degradation data.

The rest of this paper is structured as follows: Section 2 introduces the proposed SSL framework for SOH estimation of lithium-ion batteries under label sparsity, with a detailed explanation of the developed model, the ranking-based SSL algorithm, and the overall workflow. Section 3 describes the experimental setups, including data preparation, settings of the presented models and experiments, and the utilized evaluation metrics. In Section 4, comprehensive results of a variety of experiments are presented, together with in-depth analyses. The key takeaways of this work are summarized in Section 5.

## 2. Methodology

2.1. CNN-GRU Model for Self-Supervised Pretraining and SOH Estimation   
2.1.1. Convolutional Neural Networks

![](images/cdbd99ccb8f6f98d8d29caf804138e89004988ad35b7706a6d9b5a6cab4e7700.jpg)  
Figure 1: Working scheme of an example two-layer 1D CNN.

CNNs [56] are a class of deep learning models that apply trainable convolution kernels with local receptive fields and shared weight parameters to extract local patterns from structured data. Fig. 1 shows the working scheme of an example twolayer 1D CNN with a kernel size of $k = 7$ . The dilation factor d and the stride s are assumed to be 1. In order to preserve the input length after each convolution operation, the same padding technique is adopted in the network. Specifically, when the kernel size k is an odd number, the padding size on both sides of the 1D sequence is given by:

$$
p = { \frac { k - 1 } { 2 } }\tag{3}
$$

With the same padding technique, the output sequence will have the same length as the input sequence without being shortened. In this case, the output vector at time step $t ,$ namely $_ { \mathbf { \nabla } \mathbf { y } _ { t } , }$ is dependent on the inputs $x _ { t - p } , x _ { t - p + 1 } , . . . , x _ { t } , . . . , x _ { t + p - 1 } , x _ { t + p } ,$ where the inputs with indices outside the original index range of the input sequence are the appended paddings. More generally, at some time step $t ,$ the 1D convolution operation under same padding $G ( t )$ can be formulated as:

$$
G ( t ) = ( X * F ) ( t ) = \sum _ { i = 0 } ^ { k - 1 } F ( i ) \cdot x _ { t + i - p }\tag{4}
$$

where X is the input sequence and F denotes the convolution kernel with kernel size k. CNNs efectively capture local patterns in input data with relatively few parameters, which is why they are widely used across tasks such as image recognition, object detection, and signal processing. In our context of battery SOH estimation, the cycling curves of batteries can also be viewed as structured one-dimensional signals, where adjacent measurements often exhibit strong local correlations. Since battery degradation gradually alters the shape and patterns of the cycling measurement trajectories, CNNs are a natural choice for the extraction of local patterns correlated with aging from such data, which is why they are utilized as part of the network backbone in this work for SOH estimation as well as the self-supervised pretraining.

## 2.1.2. Gated Recurrent Units

![](images/516a12c237c49b16b8106a609984691d4e27f3cde834dd17306c0f26cd486480.jpg)  
Figure 2: Internal architecture of GRU.

GRUs [57] are an advanced variant of recurrent neural networks (RNNs) that are designed for sequential dependency modeling while mitigating the problem of vanishing gradient in vanilla RNNs. With the introduction of gating mechanisms, GRUs can regulate the information flow actively and update the hidden state selectively based on the importance of historical and current information. In addition, the lean internal architecture of GRUs facilitates high computational eficiency and lower dependencies on the amount of required training data, especially when compared with the other frequently applied variant of RNN with gating mechanisms, namely LSTM [58], making them a perfect choice for our case of SOH estimation under label sparsity. The internal architecture of the GRU is demonstrated in Fig. 2. At time step t, given the input vector $\mathbf { \Delta } \mathbf { x } _ { t }$ , the update gate $z _ { t }$ and reset gate $\mathbf { \nabla } _ { r _ { t } }$ can be calculated as:

$$
z _ { t } = \sigma ( W _ { x z } x _ { t } + W _ { h z } h _ { t - 1 } + b _ { z } )\tag{5}
$$

$$
r _ { t } = \sigma ( W _ { x r } x _ { t } + W _ { h r } h _ { t - 1 } + b _ { r } )\tag{6}
$$

where $\sigma ( \cdot )$ denotes the sigmoid activation function, $\scriptstyle h _ { t - 1 }$ denotes the hidden state of the previous time step, and W and b denote the respective weight matrix and bias vector. The candidate hidden state $\tilde { h } _ { t }$ can thereby be computed by:

$$
\tilde { h } _ { t } = t a n h ( W _ { x h } x _ { t } + W _ { h h } ( r _ { t } \odot h _ { t - 1 } ) + b _ { h } )\tag{7}
$$

where $\odot$ denotes the Hadamard product, namely element-wise multiplication. In the end, the final output of the hidden state $h _ { t }$ is updated as:

$$
h _ { t } = ( 1 - z _ { t } ) \odot h _ { t - 1 } + z _ { t } \odot \tilde { h } _ { t }\tag{8}
$$

The reset gate controls the forgetting and retention of the historical information for the calculation of the candidate hidden state. The candidate hidden state combines the current input and the selected historical information. The new hidden state is the final output at the current time step from the fusion of the previous hidden state and candidate hidden state controlled by the update gate. In this work, the GRU is utilized to sequentially integrate the local patterns extracted by the previous CNN from CC charging curves, thereby comprehensively capturing the underlying patterns correlated with battery degradation for self-supervised representation learning and downstream SOH estimation.

## 2.1.3. CNN-GRU Model

Fig. 3 shows the architecture of the proposed CNN-GRU model. The model consists of two parts, namely the encoder and head. The encoder aims to learn the underlying patterns from the input charging curves and is composed of two stacked CNN blocks and one GRU layer, where each CNN block further consists of one 1D convolutional layer and one rectified linear unit (ReLU) as the activation function. The CNN is responsible for extracting local patterns from the input charging curves, and the GRU is used to sequentially integrate the extracted features so as to capture their temporal dependencies. Based on the learned latent representation of the encoder, the head is further used to accomplish the target task, either self-supervised pretraining or SOH regression, which is composed of two fully connected (FC) layers and ReLU in between as the activation function. The same head architecture is used for both self-supervised pretraining and SOH estimation, but of course with diferent loss functions and sizes for the outputs. For the proposed ranking-based self-supervised pretraining, the head is used to generate an aging score $s _ { a g e }$ based on the learned representations. For reconstruction-based self-supervised pretraining, the head is used to reconstruct the respective input charging curve $\hat { U } _ { c c }$ based on the embedded vector generated by the encoder. For the downstream SOH estimation task, the head performs regression for SOH<sup>ˆ</sup> . We use the CC segment of charging as input, since, in contrast to the highly dynamic discharge process with substantial fluctuations in the operating conditions, the charging process generally follows a clearer protocol and is acquired under more stable conditions for EVs and electronics [36, 40]. As a result, it is more feasible to charge data to build high-quality, large-scale datasets that are continuously accumulated across the entire usage history, with strong comparability across diferent cycles. In addition, during the CC charging stage, the curve shape often exhibits evident pattern shifts with battery aging, making it a strong indicator of battery health. Since previous works have showcased that deep learning models are able to extract the aging-related features from the charging curve automatically [41], we use the voltage measurement of the CC charging segment $U _ { c c }$ directly as the input of the model.

![](images/381793a48b538d88d0eddfcf6aefac9a13ac286cb7b8431a0a3105d6bd6fa5c2.jpg)  
Figure 3: The proposed CNN-GRU model for self-supervised pretraining and SOH estimation.

## 2.2. Cycle-Order Ranking for Degradation-Aligned SSL

Battery aging is an inherently progressive process, in which later cycles generally correspond to deeper degradation states than earlier ones. Such an ordinal relationship provides a natural, intuitive source of self-supervision even in the absence of explicit SOH labels. Motivated by this observation, we introduce an intra-cell cycle-order ranking objective for self-supervised pretraining to encourage the model to learn degradation-aligned representations from unlabeled charging data. Specifically, for each sampled mini-batch during training, the samples are first grouped according to their cell IDs, and ranking is only performed within each cell group. This design avoids unreliable comparisons across diferent cells, whose degradation trajectories may difer due to inter-cell variability. Within each cell group, random sample pairs are constructed for eficient optimization. In order to improve the reliability of the ranking signal, pairs with identical cycle numbers or with cycle-number diferences smaller than a predefined threshold $d _ { \mathrm { m i n } }$ are excluded, since their degradation order is either undefined or too weak to provide clear supervision due to possible capacity recovery phenomena. For a valid pair of samples (i, j) from the same cell, the ranking label is defined as:

Algorithm 1 Computation of intra-cell cycle-order ranking loss for a sampled mini  
batch.   
Require: Mini-batch $\{ ( c _ { i } , n _ { i } , s _ { i } ) \} _ { i = 1 } ^ { B }$ , where $c _ { i }$ is the cell $\operatorname { I D } , n _ { i }$ is the cycle number,   
and $s _ { i }$ is the predicted aging score of sample $i ;$ minimum cycle gap $d _ { \mathrm { m i n } }$   
1: $\mathcal { G } \gets \mathrm { G r o u p B y C e l l } ( \{ 1 , \dots , B \} ; \{ c _ { i } \} _ { i = 1 } ^ { B } )$ ▷ group sample indices by cell ID   
2: $s  \emptyset$ ▷ store group-wise ranking losses   
3: for all $G _ { m } \in \mathcal { G }$ do   
4: if $| G _ { m } | < 2$ then   
5: continue ▷ at least two samples are needed   
6: end if   
7: $\tilde { G } _ { m } \gets \mathrm { P e r m u t e } ( G _ { m } )$ ▷ generate a shufled index order for subsequent paring   
8: $\mathcal { P } _ { m }  \{ ( i , j ) \mid i \in G _ { m } , \ j \in { \tilde { G } } _ { m } , \ n _ { i } \neq n _ { j } , \ | n _ { i } - n _ { j } | \geq d _ { \operatorname* { m i n } } \}$ ▷ construct   
sample pairs with ambiguous pairs removed   
9: if $| { \mathcal { P } } _ { m } | = 0$ then   
10: continue ▷ skip if no valid pair remains   
11: end if   
12: $y _ { i j }  \{ { + 1 , \quad n _ { i } > n _ { j } }  , \quad \forall ( i , j ) \in \mathcal { P } _ { m }$ ▷ later cycles should receive larger   
aging scores   
13: $\begin{array} { r } { \breve { \mathcal { L } } _ { m } \gets \frac { 1 } { | \mathcal { P } _ { m } | } \sum _ { \tiny \begin{array} { c } { , } \end{array} } \ln ( 1 + \exp ( - y _ { i j } ( s _ { i } - s _ { j } ) ) ) } \end{array}$ ▷ group-wise logistic ranking   
$( i , j ) \in \mathcal { P } _ { m }$   
loss   
14: $S \gets S \cup \{ \mathcal { L } _ { m } \}$ ▷ collect valid group losses   
15: end for   
16: if $| { \cal S } | = 0$ then   
17: return None ▷ no valid ranking supervision in this mini-batch   
18: else   
19: $\mathcal { L } _ { \mathrm { { r a n k } } } \gets \frac { 1 } { | \mathcal { S } | } \sum _ { \substack { \int _ { - \infty } ^ { } \mathcal { L } _ { \mathcal { S } } } } \mathcal { L } _ { m }$ ▷ average over valid cell groups   
$\boldsymbol { \mathcal { L } } _ { m } ^ { \overline { { \mathbf { \Lambda } } } } \in \boldsymbol { S }$   
20: return ${ \mathcal { L } } _ { \mathrm { r a n k } }$   
21: end if

$$
y _ { i j } = { \left\{ \begin{array} { l l } { + 1 , } & { { \mathrm { i f ~ } } n _ { i } > n _ { j } } \\ { - 1 , } & { { \mathrm { i f ~ } } n _ { i } < n _ { j } } \end{array} \right. }\tag{9}
$$

where $n _ { i }$ and $n _ { j }$ denote the corresponding cycle numbers. Here, $y _ { i j } = + 1$ indicates that sample i comes from a later cycle and is expected to have a larger aging score than sample $j ,$ , as the pairwise logistic ranking loss is designed to be:

$$
\ell _ { i j } = \ln ( 1 + \exp ( - y _ { i j } ( s _ { i } - s _ { j } ) ) )\tag{10}
$$

where $s _ { i }$ and $s _ { j }$ are the aging scores predicted by the model under training. It is referred to as a logistic ranking loss because it is derived from the negative loglikelihood of a logistic model, in which the probability of a correct pairwise order $p _ { i j }$ is modeled as a sigmoid function of the margin $y _ { i j } ( s _ { i } - s _ { j } )$ , namely:

$$
p _ { i j } = \frac { 1 } { 1 + \exp ( - y _ { i j } ( s _ { i } - s _ { j } ) ) }\tag{11}
$$

where the margin indicates both whether the predicted order is correct and how confidently the model makes this prediction. Minimizing the negative log-likelihood of this probability $p _ { i j }$ yields the loss $\ell _ { i j }$ , which encourages the score diference $s _ { i } - s _ { j }$ to be consistent with the relative cycle order. For the m-th valid cell group, with $\mathcal { P } _ { m }$ denoting the set of valid sample pairs constructed within that group, the group-wise ranking loss is then computed as:

$$
\mathcal { L } _ { m } = \frac { 1 } { | \mathcal { P } _ { m } | } \sum _ { ( i , j ) \in \mathcal { P } _ { m } } \ell _ { i j }\tag{12}
$$

Finally, the overall ranking loss for the mini-batch is obtained by averaging over all valid cell groups:

$$
\mathcal { L } _ { \mathrm { r a n k } } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathcal { L } _ { m }\tag{13}
$$

where M is the number of valid cell groups in the mini-batch. Algorithm 1 summarizes the intra-cell cycle-order ranking objective for a sampled mini-batch.

During ranking-based self-supervised pretraining, the model, consisting of the encoder and the head, is expected to predict a dimensionless aging score from the CC charging curve input. By explicitly enforcing that the predicted aging scores follow the relative degradation order across cycles, the proposed ranking objective encourages the encoder to capture the underlying patterns correlated to the degradation process from unlabeled charging curves. In this way, the learned representations become better aligned with the evolution of battery degradation, which is closely related to the downstream SOH estimation task, thereby benefiting cases with label sparsity in both quantity and coverage.

![](images/597fb0074c681ea29086f856a28beeb095357402d486a8e0f3b083ca02392179.jpg)  
Figure 4: Workflow of the proposed SSL framework for SOH estimation under label sparsity.

## 2.3. Degradation-Aligned SSL Framework

Fig. 4 shows the overall workflow of the proposed SSL framework for SOH estimation under label sparsity. First, as data preparation, the collected battery cycling aging data undergo a series of preprocessing procedures, including cleaning of dirty samples and outliers, formatting for a unified structure, and labeling the data partially based on the experiment settings to simulate practical scenarios where the battery aging data are under label sparsity, both in quantity and coverage. The preprocessed data are then split into training, validation, and test sets for parameter optimization, hyperparameter fine-tuning, and objective evaluation, respectively. The unlabeled data will be used for self-supervised pretraining, while the partially labeled subset will be used for fine-tuning the pretrained model during downstream adaptation. During self-supervised pretraining, the CNN-GRU model introduced in Section 2.1, composed of an encoder and a head, is first initialized. Afterwards, the model is trained on the pretext task with unlabeled charging data using the rankingbased SSL algorithm proposed in Section 2.2. The outcome of the pretraining can be validated by inspecting the correlation between the pretrained model’s predicted aging score and the ground-truth SOH values, if available. The pretrained model will be saved and transferred to downstream adaptation on the task of SOH estimation, where only the pretrained encoder will be loaded, and a new head will be instantiated for SOH regression. Subsequently, the labeled subset of data will be used for fine-tuning the model, during which only the new head is updated, with the parameters of the pretrained encoder frozen. The fine-tuned model is ready for accurate, robust SOH estimation on new cells despite label sparsity in the training data.

## 3. Experimental Setup

## 3.1. Data Preparation

Table 1: Specifications of the utlized CX2 battery.
<table><tr><td>Parameter</td><td>Data</td></tr><tr><td>Nominal Capacity</td><td>1.35 Ah</td></tr><tr><td>Cell Chemistry</td><td>LCO</td></tr><tr><td>Cell Format</td><td>Prismatic</td></tr><tr><td>End-of-Charge Voltage</td><td>4.20 V</td></tr><tr><td>End-of-Discharge Voltage</td><td>2.70 V</td></tr></table>

In this work, the experiments and analyses are based on the public battery aging dataset, which is widely used in the research community, from the Center for Advanced Life Cycle Engineering (CALCE) at the University of Maryland [59]. Specifically, the cells CX2-34, CX2-36, CX2-37, and CX2-38 are utilized in this work, of which the specifications are shown in Table 1. These prismatic cells have lithium cobalt oxide (LCO) cathodes and a rated capacity of 1.35 Ah. The four cells underwent cyclic aging from the beginning of life until diferent depths of degradation. For charging, they went through a CC-CV protocol, where the cells were charged under the constant current rate of 0.5 C until the end-of-charge voltage of 4.2 V was reached, after which CV charging carried on until the charging current dropped below 0.05 A. For discharging, the cells went through a CC protocol with a current rate of 1 C [18] until the end-of-discharge voltage of 2.7 V was reached.

![](images/20abc51b3b80166e5f219e8a356af56742460b649c5da35e8946a56adfac5edd.jpg)

![](images/c92a43a68ec7282a388c26369c53990cb5ab2b91ee33c40ec64c22507a750190.jpg)

![](images/4fb8c988784155e141b41897312f1210e7d02d236c62253551bea2ba542b416b.jpg)  
(a)

![](images/6599a681fb7463da91b6593b09bb951e919103469ecb736559281982c1362ec2.jpg)  
(b)

![](images/7c62ad49632f2bf65f515f704604d635c63c3dc9cff395890a2cc8d611d01116.jpg)  
(c)

![](images/31cef7b7829030c5e8c9641a975e856df82a59954b7bcfec45ef503693a1291c.jpg)  
(d)  
Figure 5: Aging profiles of the cells, including the current and voltage measurement during CC charging and the evolution of SOH. (a) Cell CX2-34. (b) Cell CX2-36. (c) Cell CX2-37. (d) Cell CX2-38.

BatteryML [60], an open-source platform for machine learning-based battery degradation diagnostics, is used for preliminary preprocessing of the raw cycling data, including the initial integration and ordering of the measurement files, compilation of basic information, and rough cleaning of dirty entries. Afterward, the voltage measurements of the CC charging fragment are extracted for each cycling curve and resampled to a fixed length of 300 points to facilitate later training and to enforce the network to learn the underlying patterns from the partial charging curve rather than simply using the sequence length as the indicator. The corresponding SOH labels are calculated based on Equation 1 using the full discharge capacity. Consequently, more thorough data cleaning is performed, removing outliers based on anomalies in current and voltage readings, as well as on the rolling median absolute deviation (MAD) of the lengths of CC charging segments at the cell level. The preprocessed aging profiles of the cells are shown in Fig. 5. As can be observed, the overall shape and pattern of the voltage measurement during CC charging demonstrate an obvious shift through the aging process: the initial voltage reading becomes higher due to impedance rise, and the duration of the CC fraction becomes shorter due to the combination of loss of lithium inventory (LLI), loss of active material (LAM), and impedance rise, which intuitively underpins the reliability of using voltage measurement during CC charging as the input feature for SOH estimation. The capacity degradation trends of the cells appear similar, where the four cells are aged to diferent depths of degradation. Based on this observation, CX2-37 and CX2-38 are selected for training because together they cover relatively early and late degradation stages, enabling the model to learn from a wider range of aging conditions. CX2-36 and CX2-34, which fall between these two extremes in terms of degradation depth, are used for validation and testing, respectively. In this way, the validation and test cells remain unseen during training, while their degradation states are still enclosed by the training distribution rather than lying outside it. This split is therefore intended to enable a more stable and meaningful evaluation of cross-cell generalization.

## 3.2. Experiment Settings

Z-score normalization is utilized on the training set to accelerate convergence in this work using the following formula:

$$
z = { \frac { x - \mu } { \sigma } }\tag{14}
$$

where z is the normalized value, x is the raw value, $\mu$ is the mean, and $\sigma$ is the standard deviation. The utilized CNN-GRU models all have the same architectural hyperparameters regardless of whether they are for self-supervised pretraining or SOH estimation. For the encoder part, the two-layer CNN uses a slightly larger kernel size of 7 to capture local patterns across more adjacent points and increases the number of channels from 1 to 32 and then to 64. Replicate values are used for the same padding on both sides of the input sequence. After the CNN, a one-layer GRU with an input size of 64 and an output size of 128 is used to fuse the extracted local patterns. For the head part, the fully connected layers have diferent numbers of neurons for diferent tasks. For ranking-based self-supervised pretraining and for SOH regression, the two fully connected layers decrease the size of the latent vector gradually from 128 to 64, and then to 1, yielding a scaler of aging score or SOH estimate. For the reconstruction pretext task, which is used as a comparison in the experiments, the two fully connected layers gradually increase the size of the latent vector from 128 to 256, and then to 300. The batch size of 256 is used in this work. Automatic hyperparameter fine-tuning of the learning rate is conducted for all models, including both self-supervised pretraining and fine-tuning, using the hyperparameter optimization framework Optuna [61], utilizing the tree-structured Parzen estimator (TPE) algorithm, based on their performance on the validation set for a fair comparison.

The ranking-based self-supervised pretraining uses a minimum cycle gap of $d _ { m i n } =$ 50 to mitigate disturbances to the supervision signal caused by the capacity recovery efect during the aging test. Besides the proposed ranking-based SSL (SSL-Rank) approach, we further apply two other SSL approaches for pretraining as comparison, including the most frequently used reconstruction-based SSL (SSL-Recon) approach and a multi-objective SSL (SSL-MO) approach. As the pretext task, SSL-Recon aims to reconstruct the voltage curves during CC charging through the encoder and the head, thereby encouraging the model to learn informative representations that preserve structural characteristics of the input charging signals. And as the name indicates, SSL-MO combines reconstruction and ranking as its pretext task in order to test the possibility that these two self-supervised objectives can complement each other and thereby yield more informative representations for downstream SOH estimation. The SSL models are first pretrained using the respective SSL strategy on the unlabeled data, after which a new head will be instantiated on top of the pretrained encoder for adaptation using the limited labeled data. During the adaptation, the pretrained encoder will be frozen, meaning only the freshly instantiated head will be updated. Certainly, at the same time, we train ordinary supervised learning (SL) models using the limited labeled data for SOH estimation as a baseline.

## 3.3. Evaluation Metrics

The regression performance of the SOH estimation task is evaluated using the following four evaluation metrics in this work, namely MAE, RMSE, maximum absolute error (MAX), and the coeficient of determination $R ^ { 2 }$

$$
\ M A E ( { \pmb y } , { \hat { \pmb y } } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | y _ { i } - { \hat { y } } _ { i } |\tag{15}
$$

$$
R M S E ( { \pmb y } , { \hat { \pmb y } } ) = { \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } - { \hat { y } } _ { i } ) ^ { 2 } } }\tag{16}
$$

$$
R ^ { 2 } ( { \pmb y } , \hat { \pmb y } ) = 1 - \frac { \displaystyle \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } { \displaystyle \sum _ { i = 1 } ^ { n } ( y _ { i } - \bar { y } ) ^ { 2 } }\tag{17}
$$

$$
M A X ( \pmb { \hat { y } } ) = \operatorname* { m a x } \{ | y _ { i } - \hat { y } _ { i } | \} _ { i = 1 } ^ { n }\tag{18}
$$

where $\textbf {  { y } }$ and $\hat { y }$ denote the labels and the estimates, respectively. n is the number of samples, while $y _ { i }$ and $\hat { y } _ { i }$ denote the label and the estimate of the i-th sample, respectively. $\bar { y }$ is the mean of the labels $\mathbf { \pmb { y } }$ . The four evaluation metrics for regression have diferent emphases. MAE is an intuitive measure for the average magnitude of estimation errors, while RMSE gives greater weight to large errors due to the squaring operation. $R ^ { 2 }$ is used to evaluate how well the estimations capture the overall variation trend of the labels during fitting, and MAX reflects the worst-case estimation deviation across all samples.

In addition, Spearman’s rank correlation coeficient $\rho$ is employed to evaluate the monotonic relationship between the predicted aging scores and the ground-truth SOH values after the self-supervised pretraining, thereby assessing whether the learned representations can consistently reflect the real degradation process. It is defined as:

$$
\rho = \frac { \mathrm { c o v } ( R _ { s } , R _ { y } ) } { \sigma _ { R _ { s } } \sigma _ { R _ { y } } }\tag{19}
$$

where $\mathbf { \delta } _ { R _ { s } }$ and $R _ { y }$ denote the ranks of the predicted aging scores and the groundtruth SOH labels, respectively. cov denotes covariance, and $\sigma _ { R _ { s } }$ and $\sigma _ { R _ { y } }$ denote the standard deviations of the corresponding rank variables. $\rho$ is defined in the range from -1 to 1, where $\rho = 1$ indicates a perfect positive monotonic relationship, and $\rho = - 1$ indicates a perfect negative monotonic relationship.

## 4. Results and Discussion

In this section, we present the results and analyses of a variety of experiments conducted to evaluate the pretrained model via degradation alignment, compare diferent SSL and SL approaches on sparsely labeled data, and examine the influence of label distribution.

## 4.1. Evaluation of the Pretrained Model via Degradation Alignment

In this experiment, we aim to evaluate the pretrained model using the proposed ranking-based SSL approach before downstream fine-tuning. To this end, the pretrained model is used to generate an aging score for each input charging curve, and the resulting scores are compared with the corresponding ground-truth degradation process in order to verify their alignment. Since the proposed ranking objective is designed to encode the relative degradation order of battery cycles into the learned representations and then map it into an unscaled aging score prediction, efective pretraining is expected to produce aging scores that vary monotonically with SOH. Therefore, both the overall relationship between the predicted aging scores and the degradation process, and their Spearman’s rank correlation coeficient are analyzed to assess whether the pretrained model has successfully captured degradation-aligned information from unlabeled charging data.

Fig. 6 shows the correlation between the aging score predicted by the pretrained model under the proposed ranking-based SSL and the ground-truth degradation processes of the four cells, namely CX2-37, CX2-38, CX2-36, and CX2-34, across the training, validation, and testing sets. Since the learned aging score of the pretrained model is an unscaled value without a unit, we first normalize it into the range from 0 to 1 for better visualization, as well as the ground-truth capacity fade, using the following formula of min-max normalization:

$$
x ^ { \prime } = \frac { x - x _ { m i n } } { x _ { m a x } - x _ { m i n } }\tag{20}
$$

where $x ^ { \prime }$ is the normalized value, x is the original value, and $x _ { m a x }$ and $x _ { m i n }$ denote the maximum and minimum of the variable, respectively. In the left column of Fig. 6, namely Fig. 6a, Fig. 6c, Fig. 6e, and Fig. 6g, we compare the aforementioned normalized aging scores with the ground-truth capacity fade over the entire lifespan of each cell. As can be observed, despite the minor local fluctuations, the aging scores predicted by the pretrained model generally demonstrate a clear monotonic overall trend that is well aligned with the capacity degradation trajectory of each cell. In particular, as the battery ages and the capacity gradually fades, the predicted aging scores consistently evolve in the corresponding direction, indicating that the pretrained model has successfully captured degradation-relevant information from unlabeled charging data. This is particularly evident for cells CX2-37 and CX2-38, namely the two training cells, as the ranking objective was directly imposed on them during the training process. Especially for the training cell CX2-38, the aging scores predicted by the pretrained model can even reconstruct the diferent degradation paces of the ground-truth degradation process, showing clearly diferent slopes before and after the 1350th cycle. For the validation cell CX2-36, from around the 600th cycle, the predicted aging score begins to show slight divergence from the groundtruth capacity fade, although the monotonicity and alignment are not compromised, and the dip indicating capacity recovery at around the 1300th cycle is well captured by the pretrained model. Similarly, on the test cell CX2-34, the predicted aging score generally aligns quite well with the ground-truth capacity fade, despite local fluctuations at the beginning of life (BOL) and a slight divergence in the middle of life. The capacity recovery phenomenon at around the 1150th cycle is well captured by the pretrained model based on the unlabeled CC charging data as well.

![](images/1eeda63eb557556f6174e65f8ec6c5ae6fa5de401c2c67e86c630d3c5b51b2dd.jpg)  
(a)

![](images/518e44a82a6c5274a6d0ae9dbb87da316b7af4c3f49d9cdcc284768c941388df.jpg)  
(b)

![](images/b3059d8c42512cd0c5af8ea54714fbc36c35cd6f81bcd949f33837ff65c0b972.jpg)  
(c)

![](images/54133b18a5f9a79ebacc6b7122287bf16a6c9c2c35bf6420f6d85cb0c1b557fb.jpg)  
(d)

![](images/f116dbc25c6815280afae9c6a65369e6e06591e36418b8e2d348cf6dc54cbe2d.jpg)  
(e)

![](images/a6991bf1777d5349491feed577c966df81931edf3caff6de63547c97961a9284.jpg)  
(f)

![](images/4cebefc4d8d049e52c9b1ae6744c77b7061cd60e88d6a78aecbc6f20e0ce51fc.jpg)  
(g)

![](images/9e3eac7e98a028568fad4f615755ff24e926c625ebfee535f5dc489c026e0213.jpg)  
(h)  
Figure 6: Correlation between the aging score predicted by the pretrained model under rankingbased SSL and the respective ground-truth degradation process. (a) Cell CX2-37, normalized aging score and capacity fade. (b) Cell CX2-37, normalized aging score and SOH. (c) Cell CX2-38, normalized aging score and capacity fade. (d) Cell CX2-38, normalized aging score and SOH. (e) Cell CX2-36, normalized aging score and capacity fade. (f) Cell CX2-36, normalized aging score and SOH. (g) Cell CX2-34, normalized aging score and capacity fade. (h) Cell CX2-34, normalized aging score and SOH.

In the right column of Fig. 6, namely Fig. 6b, Fig. 6d, Fig. 6f, and Fig. 6h, the correlation between the normalized aging score and the ground-truth SOH is showcased for each cell, with Spearman’s rank correlation coeficient calculated and marked in each case. The normalized aging scores generally exhibit a clear monotonic decreasing trend as the ground-truth SOH increases, or equivalently, a monotonic increasing trend as the battery degrades. This indicates that the predicted aging scores are strongly consistent with the underlying aging progression of each cell. Although a few outliers can still be observed locally, the overall relationship remains highly correlated. The calculated Spearman’s rank correlation coeficients ρ, namely - 0.997 for cell CX2-37, -0.998 for cell CX2-38, -0.996 for cell CX2-36, and -0.997 for cell CX2-34, are all extremely close to -1, indicating an almost perfect monotonic inverse relationship between the predicted aging score and the ground-truth SOH for all cells, which further supports the fact that our proposed ranking-based SSL approach is able to enforce the pretrained model to extract degradation-aligned information from unlabeled charging data, such that the learned aging scores consistently reflect the underlying battery aging progression across the entire lifespan.

## 4.2. Model Performance Comparison under Label Sparsity

In this experiment, we aim to compare the performance of the models pretrained using diferent SSL strategies after fine-tuning, as well as under complete SL with no pretraining on the unlabeled data. More specifically, we consider three diferent SSL approaches: SSL-Rank, our proposed SSL approach using the cycle-order objective as the pretext task; SSL-Recon, the most frequently used SSL approach using reconstruction of the input CC charging curves as the pretext task; SSL-MO, which combines reconstruction and ranking as its pretext task in order to test the possibility that these two self-supervised objectives can complement each other and thereby yield more informative representations for downstream SOH estimation. All models are first pretrained on the unlabeled CC charging data using their respective SSL strategies and then fine-tuned using sparsely labeled CC charging data. In addition, we train a pure SL model from scratch without pretraining on the same sparsely labeled CC charging data. Here, we consider a sparse-label setting in which only data above the 80% SOH threshold are labeled, and the proportion of labeled samples varies from 1% to 100%, where 100% of labeled data corresponds to approximately 1200 training samples. This setup is consistent with practical scenarios, since 80% SOH is commonly adopted as the EOL criterion for lithium-ion batteries, and thus data from this earlier degradation region are more likely to be available than labeled samples from deep-degradation stages, whose acquisition requires much longer cycling time and higher experimental cost. The labeled data points for fine-tuning or complete SL are sampled uniformly from all data points within the defined SOH range for each training cell.

Fig. 7 shows the SOH estimation results on the test cell CX2-34 of diferent models after fine-tuning or complete training on diferent ratios of labeled data. At first glance, we can see that all models pretrained using SSL strategies are able to carry out satisfactory SOH estimation under diferent ratios of labeled data from 1% to 100% on the entire lifespan of the test cell, while the pure SL model without pretraining shows great fluctuations in its estimation under ratios of labeled data above 50% and fails the task of SOH estimation under ratios of labeled data below 40%, because the number of samples available for SL from scratch becomes too small for the model to capture the underlying battery degradation dynamics. Compared with models pretrained with the other two SSL approaches, the proposed SSL-Rank model consistently outperforms the rest of the models and is able to track the groundtruth degradation progression accurately under all scenarios with diferent ratios of labeled data, even when the labeled data is extremely sparse, as low as 1%. In general, since the labeled data are considered to be in the earlier degradation region, the estimations of the models fit better to the early-life ground-truth SOHs, while slight divergences can be observed towards the EOL. Nevertheless, the proposed SSL-Rank model is able to carry out an SOH estimation that not only is steady and smooth in early life, but also converges well to the ground-truth in late life, as shown in the zoomed-in sections. In comparison, the SSL-Recon model exhibits significantly higher oscillations in its estimation, and the estimation errors of the SSL-Recon and SSL-MO models become more pronounced in the deep-degradation regime.

Fig. 8 is the box plot of the absolute error of diferent models for SOH estimation under diferent ratios of labeled data. The aforementioned observations can be better confirmed in this visualization. The three SSL models achieve consistently good performance across all testing scenarios with varying proportions of labeled data. The SSL-Recon model and the SSL-MO model achieve comparable overall performance in most cases, though the latter typically produces more outliers. The proposed SSL-Rank model is able to outperform the rest of the models in every testing scenario, regardless of whether the labeled proportion is abundant or scarce, with fewer outliers in addition. The baseline SL model can still carry out relatively good SOH estimation when the amount of labeled data is suficient, namely 50% or more, but is not able to learn the underlying degradation dynamics and thus fails to carry out SOH estimation when fewer labeled data are available. It is intriguing that even in cases where 100% or 50% labeled data is available, the SL model is still outperformed by all the SSL models, indicating that leveraging unlabeled charging data through SSL pretext tasks not only compensates for label scarcity, but also improves the model initialization and guides it toward a more favorable parameter space for downstream optimization, so that the model can make better use of the labeled data during fine-tuning and achieve superior SOH estimation performance even when labels are relatively abundant.

![](images/1373914c8c0496984b3fdbc6146a682d944ad11c493dd3b4d5237eb1d9f309a8.jpg)  
(a)

![](images/5fc6548d9920c88719e1f2de86525dc5e0fbd01481d1769b64b576598b122d60.jpg)  
(b)

![](images/712db89e12df582ce078ab30c8633f311d1a915796c544d4ef93182e3c8e3398.jpg)  
(c)

![](images/a45f8e38cef8d009d98bd8fa7bce09fd6710ff95a915a68b62a4ce27e4edbe49.jpg)  
(d)

![](images/b3740564652b66fbe0021c19531d9ff0c38e3b153188f2031d519170655d70ca.jpg)  
(e)

![](images/4d4a2689953de47727cd352009b63e0ff13afc32e40ac2c000606f8ca884ceef.jpg)  
(f)

![](images/6bdf7331326da9b85a03e1b7dc8f43eac84dfb5b13935489f433decf1858645b.jpg)  
(g)

![](images/21c0fc0e6099a6f650891a48d187b2b019dc66fbf098d2c7699628f5cd965322.jpg)  
(h)  
Figure 7: SOH estimation results on the test cell CX2-34 of diferent models after fine-tuning or complete training on diferent ratios of labeled data. (a) 100% of labeled data available. (b) 50% of labeled data available. (c) 40% of labeled data available. (d) 30% of labeled data available. (e) 20% of labeled data available. (f) 10% of labeled data available. (g) 5% of labeled data available. (h)1% of labeled data available.

![](images/730add2b80f394af637cb1aee17422a6759b760a20012b45b35c1b19e7a2a67f.jpg)  
Figure 8: Box plot of the absolute error for SOH estimation under diferent ratios of labeled data.

Table 2 summarizes the experiment results on diferent models’ performance under varying ratios of labeled data above 80% SOH. The proposed SSL-Rank model leads across all four evaluation metrics in all testing scenarios with diferent labeled data ratios, achieving MAEs of 1.715%, 1.722%, 1.692%, and 1.718% for labeled data ratios of 100%, 50%, 10%, and 1%, respectively. The single-objective SSL-Recon model is outperformed by the dual-objective SSL-MO model in most test cases, except at 20% and 10% labeled data ratios, in terms of MAE. In the case when 30% of labeled data is available, the SSL-MO model achieves an MAE of 2.283%, which is lower than the MAE of 2.449% of the SSL-Recon model, while the RMSE of 3.311% of the SSL-MO model is higher than the RMSE of 3.169% of the SSL-Recon model, indicating that the estimation of the SSL-MO model contains more outliers. The SL model achieves MAEs of 4.017% and 4.031% in test scenarios with labeled data ratios to maintain an $R ^ { 2 }$ of over 0.970, indicating its strong performance in SOH regression. The MAXs of around 10% typically characterize the final estimation error in the deepest-degradation regime due to significantly accelerated aging in this stage.

Table 2: Comparison of diferent models’ performance under varying ratios of labeled data above 80% SOH.
<table><tr><td rowspan="2">Labeled Ratio [%]</td><td rowspan="2">Model</td><td colspan="4">Evaluation Metrics</td></tr><tr><td>MAE [%]</td><td>RMSE [%]</td><td> $R ^ { 2 }$ </td><td>MAX [%]</td></tr><tr><td rowspan="4">100</td><td>SSL-Rank</td><td>1.715</td><td>2.255</td><td>0.972</td><td>10.458</td></tr><tr><td>SSL-Recon</td><td>2.768</td><td>3.789</td><td>0.921</td><td>12.874</td></tr><tr><td>SSL-MO</td><td>2.409</td><td>3.407</td><td>0.936</td><td>14.556</td></tr><tr><td>SL</td><td>4.017</td><td>5.384</td><td>0.841</td><td>19.625</td></tr><tr><td rowspan="4">50</td><td>SSL-Rank</td><td>1.722</td><td>2.302</td><td>0.971</td><td>10.937</td></tr><tr><td>SSL-Recon</td><td>2.557</td><td>3.366</td><td>0.938</td><td>11.380</td></tr><tr><td>SSL-MO</td><td>2.370</td><td>3.303</td><td>0.940</td><td>13.889</td></tr><tr><td>SL</td><td>4.301</td><td>5.381</td><td>0.841</td><td>18.038</td></tr><tr><td rowspan="4">40</td><td>SSL-Rank</td><td>1.699</td><td>2.259</td><td>0.972</td><td>10.724</td></tr><tr><td>SSL-Recon</td><td>2.590</td><td>3.401</td><td>0.936</td><td>11.168</td></tr><tr><td>SSL-MO</td><td>2.362</td><td>3.370</td><td>0.938</td><td>14.576</td></tr><tr><td>SL</td><td>12.203</td><td>14.317</td><td>-0.127</td><td>28.848</td></tr><tr><td rowspan="4">30</td><td>SSL-Rank</td><td>1.707</td><td>2.279</td><td>0.971</td><td>10.815</td></tr><tr><td>SSL-Recon</td><td>2.449</td><td>3.169</td><td>0.945</td><td>11.838</td></tr><tr><td>SSL-MO</td><td>2.283</td><td>3.311</td><td>0.940</td><td>14.869</td></tr><tr><td>SL</td><td>12.442</td><td>14.593</td><td>-0.171</td><td>29.566</td></tr><tr><td rowspan="4">20</td><td>SSL-Rank</td><td>1.692</td><td>2.241</td><td>0.972</td><td>10.624</td></tr><tr><td>SSL-Recon</td><td>2.564</td><td>3.350</td><td>0.938</td><td>11.878</td></tr><tr><td>SSL-MO</td><td>2.597</td><td>3.588</td><td>0.929</td><td>15.300</td></tr><tr><td>SL</td><td>12.184</td><td>14.290</td><td>-0.123</td><td>28.659</td></tr><tr><td rowspan="4">10</td><td>SSL-Rank</td><td>1.692</td><td>2.249</td><td>0.972</td><td>10.686</td></tr><tr><td>SSL-Recon</td><td>2.655</td><td>3.528</td><td>0.932</td><td>11.839</td></tr><tr><td>SSL-MO</td><td>2.803</td><td>3.873</td><td>0.918</td><td>16.210</td></tr><tr><td>SL</td><td>11.966</td><td>14.039</td><td>-0.084</td><td>27.791</td></tr><tr><td rowspan="4">5</td><td>SSL-Rank</td><td>1.693</td><td>2.258</td><td>0.972</td><td>10.749</td></tr><tr><td>SSL-Recon</td><td>2.806</td><td>3.789</td><td>0.921</td><td>12.592</td></tr><tr><td>SSL-MO</td><td>2.721</td><td>3.756</td><td>0.922</td><td>15.753</td></tr><tr><td>SL</td><td>11.969</td><td>14.043</td><td>-0.085</td><td>27.803</td></tr><tr><td rowspan="4">1</td><td>SSL-Rank</td><td>1.718</td><td>2.329</td><td>0.970</td><td>11.157</td></tr><tr><td>SSL-Recon</td><td>3.075</td><td>4.293</td><td>0.899</td><td>14.228</td></tr><tr><td>SSL-MO</td><td>2.689</td><td>3.707</td><td>0.924</td><td>15.689</td></tr><tr><td>SL</td><td>11.957</td><td>14.029</td><td>-0.082</td><td>27.749</td></tr></table>

of 100% and 50%, respectively, but collapses when fewer labeled data are available, with MAEs around 12%. In all test scenarios, the proposed SSL-Rank model is able

The results of this experiment showcase the efectiveness and robustness of the proposed SSL framework across diferent labeled data proportions under the practical earlier-life labeling setting. Even with an extremely limited amount of labeled data, the proposed method is still able to maintain a high estimation accuracy, further demonstrating that the self-supervised pretraining stage has successfully endowed the model with informative and degradation-aligned prior knowledge that can be efectively transferred to downstream SOH estimation under label sparsity in both quantity and coverage.

## 4.3. Analysis on the Influence of Label Distribution

In this experiment, we aim to evaluate the SSL-Rank model’s performance under diferent distributions of sparsely labeled data and thereby analyze the influence of label distribution on the task of SOH estimation. Since the test results in Section 4.2 already show that the model pretrained utilizing our proposed ranking-based SSL approach is able to carry out accurate and robust SOH estimation after fine-tuning, even only on an extremely small number of labeled data, we explore the diferent distributions on rather small labeled ratios here, namely 10%, 5%, 2%, and 1% of the whole training set, where 10% corresponds to approximately 260 samples. The four studied types of distribution are uniform, random, early-only, and late-only. For the uniform setting, the labeled samples are selected to cover the entire available SOH range as evenly as possible. For the random setting, the labeled samples are randomly drawn from all available data. In the early-only setting, only samples from the extreme early degradation stage, namely above 85% SOH, are retained as labeled data, whereas in the late-only setting, only samples from the late degradation stage, namely below 60% SOH, are labeled. This experiment is designed as a supplementary analysis to the main experiment in Section 4.2, with the focus shifted from the overall efectiveness of the proposed SSL framework to the specific influence of label distribution of the training set under very limited labeling budgets.

Fig. 9 shows the regression performance of the proposed SSL-Rank model finetuned with 10% or 5% of labeled data sampled from diferent distributions. At first glance, we can see that the type of distribution of the labeled data has some minor influence on the overall regression performance. The $R ^ { 2 }$ decreases from 0.979 to 0.977 when 10% of the labeled data are sampled randomly from the whole lifespan instead of uniformly, and decreases from 0.981 to 0.977 when the labeled ratio is 5%, indicating that a more even coverage of the degradation trajectory is still beneficial.

![](images/376bc1e377aa3c1712c04bb3d6738e52618e0087bdba0a64a97fc4f2e1dcf8db.jpg)  
(a)

![](images/ec157ca3093cc2183a22e17f37e2ab469172b44f318dc0a66513069769066075.jpg)  
(b)

![](images/e707a0c11f66e552a5dfe1e9859708b8ed94798acb0449f62e80484c0c9534ec.jpg)  
(c)

![](images/399a6b83e1792674118ea563f8e2dd7b018addfc6e4744eecf1b1f57647c55e8.jpg)  
(d)

![](images/34ef6341b43e27c81e046e0aa7b73b92dda2074716028d0b3190e6f3be0984d8.jpg)  
(e)

![](images/4d76d5821cfe48005bc610bfd44fcaf7db535f9d17dc3eeb1a08c2b2cce8f282.jpg)  
(f)

![](images/cd04beed26ffb07d29d27c6adbdf50eb7e87c626d9e5eb9008e820833e485bf1.jpg)  
(g)

![](images/b7c711074db779b10fc64204a8f53e0275b12d822b2269d586eb8fd0905144ae.jpg)  
(h)  
Figure 9: Regression performance of the proposed SSL-Rank model fine-tuned with diferent ratios and distributions of labeled data. (a) 10% labeled data, uniform distribution. (b) 10% labeled data, random distribution. (c) 10% labeled data, early-only distribution. (d) 10% labeled data, late-only distribution. (f) 5% labeled data, uniform distribution. (f) 5% labeled data, random distribution. (g) 5% labeled data, early-only distribution. (h) 5% labeled data, late-only distribution.

The early-only distribution, on the other hand, seems to yield worse training outcomes: the $R ^ { 2 }$ is 0.974 when the labeled ratio is 10%, whereas the model fails the SOH estimation task when the ratio decreases to 5%. In fact, we have also tried to sample 10% of the labeled data only from the range above 90%, where the model fails as well. The deterioration in model performance when labeled data are only available for the extremely early life stage is due to the limited diversity of charging curves in this stage. As shown in Fig. 5, the pattern shift in the voltage curve during CC charging is minor in the early-degradation stage, but it accelerates significantly toward the late-degradation stage, meaning that labeled samples drawn from the early-only distribution are not diverse enough and thus the model can hardly generalize on such data. At the same time, we observe that the lack of data diversity can be compensated to some degree by larger data quantity, as showcased by the regression performance under 10% and 5% of early-only labeled data. On the contrary, the late-only distribution leads to the overall performance even better than the uniform distribution. A plausible reason is that the voltage curves in this stage exhibit much more pronounced aging-related variations and pattern shifts, so that the labeled samples provide stronger supervisory signals for calibrating the pretrained model to the SOH regression task. In addition, a major source of error for uniform and random distributions is the divergence between the estimation and the ground truth toward the EOL, where significantly accelerated aging happens. This is also compensated for when more labeled samples in the late-degradation region are available for model fine-tuning.

Fig. 10 visualizes the MAEs of the SSL-Rank model’s estimation with diferent ratios and distributions of labeled data for fine-tuning across diferent SOH windows. In general, the estimation error in the SOH range below 60% is significantly higher than in other SOH windows for each case due to the accelerated late-life degradation. The model fails the SOH estimation task when fine-tuned on less than 5% of labeled data under early-only distribution, causing the diferent scales of the bar graphs. In Fig. 10a, we can clearly observe an increasing trend in the SOH estimation error in the late-life SOH window below 60% from uniform distribution to random distribution and then to early-only distribution. Sampling labeled data from the late-only distribution increases the estimation error in the early-life SOH window above 90%, but at the same time significantly decreases the estimation error in the late-life SOH window below 60% as well as in the middle-life SOH windows from 60% to 90%, resulting in an improvement in the overall performance.

Table 3 summarizes the estimation results of the proposed SSL-Rank model finetuned under diferent distributions of sparsely labeled data. Many of the aforementioned observations can be confirmed. The model fine-tuned with late-only labeled data outperforms the rest of the label distributions consistently in terms of MAE, with the MAEs of 1.320%, 1.369%, 1.369%, and 1.350% for cases where 10%, 5%, 2%, and 1% of the data are labeled, respectively. This suggests that, under the same labeling budget, label coverage over the more informative late-degradation region is particularly beneficial for downstream SOH estimation, which indicates that, when only a very limited number of labels can be acquired, prioritizing samples from the later degradation stage may be more advantageous than distributing labels evenly or concentrating them in the early-life stage. The failure of the model fine-tuned with early-only labeled data in cases of fewer labeled data shows that the lack of data diversity and data quantity jointly afect the efectiveness of downstream fine-tuning. However, to some extent, these two factors can compensate for each other: limited diversity may be partially alleviated by a larger number of labeled samples, while limited quantity may be mitigated if the labeled data cover more informative and diverse degradation stages. These findings further confirm the efectiveness of the proposed ranking-based SSL strategy, which enables the model to capture degradation-relevant information from unlabeled data and thus alleviates the dependence of downstream fine-tuning on both label quantity and label diversity.

![](images/cd0ee20d6e79217b2cd8dc3f77563ba6cc073fde548e1041fb6c87ce8a0d849c.jpg)  
(a)

![](images/9b6a27c4fa429eaaffda51ef70bab21b7482a7679b265233028caff7e32c26b2.jpg)  
(b)

![](images/654b3ecc617171a917847beafd592af4a818c84c25e84f9def49d8b548bd52f7.jpg)  
(c)

![](images/9c9c1e0de3e59fdf8a3f34407c98fde955dd119d38ae358e05689f1f6115618a.jpg)  
(d)  
Figure 10: Bar graph of the MAEs of the SSL-Rank model’s estimation with diferent ratios and distributions of labeled data for fine-tuning across SOH windows. (a) 10% of labeled data available. (b) 5% of labeled data available. (c) 2% of labeled data available. (d) 1% of labeled data available.

Table 3: Evaluation of the SSL-Rank model’s performance under diferent distributions of sparsely labeled data.
<table><tr><td rowspan="2">Labeled Ratio [%]</td><td rowspan="2">Distribution</td><td colspan="4">Evaluation Metrics</td></tr><tr><td>MAE [%]</td><td>RMSE [%]</td><td> $R ^ { 2 }$ </td><td>MAX [%]</td></tr><tr><td rowspan="4">10</td><td>Uniform</td><td>1.569</td><td>1.946</td><td>0.979</td><td>9.032</td></tr><tr><td>Random</td><td>1.566</td><td>2.037</td><td>0.977</td><td>9.515</td></tr><tr><td>Early-Only</td><td>1.615</td><td>2.160</td><td>0.974</td><td>10.058</td></tr><tr><td>Late-Only</td><td>1.320</td><td>1.744</td><td>0.983</td><td>7.605</td></tr><tr><td rowspan="4">5</td><td>Uniform</td><td>1.519</td><td>1.858</td><td>0.981</td><td>7.724</td></tr><tr><td>Random</td><td>1.586</td><td>2.032</td><td>0.977</td><td>9.199</td></tr><tr><td>Early-Only</td><td>11.682</td><td>13.728</td><td>-0.037</td><td>27.632</td></tr><tr><td>Late-Only</td><td>1.369</td><td>1.862</td><td>0.981</td><td>7.565</td></tr><tr><td rowspan="4">2</td><td>Uniform</td><td>1.508</td><td>1.804</td><td>0.982</td><td>7.480</td></tr><tr><td>Random</td><td>1.516</td><td>1.935</td><td>0.979</td><td>8.791</td></tr><tr><td>Early-Only</td><td>11.654</td><td>13.694</td><td>-0.031</td><td>27.572</td></tr><tr><td>Late-Only</td><td>1.369</td><td>1.863</td><td>0.981</td><td>7.659</td></tr><tr><td rowspan="4">1</td><td>Uniform</td><td>1.589</td><td>1.904</td><td>0.980</td><td>8.054</td></tr><tr><td>Random</td><td>1.635</td><td>2.013</td><td>0.978</td><td>8.753</td></tr><tr><td>Early-Only</td><td>11.266</td><td>13.240</td><td>0.036</td><td>27.212</td></tr><tr><td>Late-Only</td><td>1.350</td><td>1.839</td><td>0.981</td><td>7.420</td></tr></table>

## 5. Conclusion

An accurate estimation of SOH underpins a safe and optimized use of the battery system. Although compelling, data-driven SOH estimation models rely on large amounts of high-quality labeled cycling data, where the SOH labels are typically obtained through standardized checkup tests that cover the entire lifespan of the batteries under controlled conditions. However, in practical application scenarios, battery aging data often face the problem of label sparsity in both quantity and coverage due to the time-consuming and costly nature of such lifespan-wide calibration tests, while large volumes of unlabeled operational cycling data are available. Therefore, in this work, we propose a degradation-aligned SSL framework that learns aging-consistent representations from unlabeled CC charging curves through a cycleorder ranking objective as the pretext task for pretraining, thereby enabling robust SOH estimation after fine-tuning on only a limited amount of labeled data for practical scenarios where data labels are sparse in both quantity and coverage. In addition, we develop a CNN-GRU model that extracts local patterns from charging curves via convolutional layers and then integrates these features sequentially through recurrent units for accurate SOH estimation as well as self-supervised pretraining. Comprehensive experiments are conducted on the CALCE battery aging dataset to evaluate the model performance after self-supervised pretraining and after fine-tuning on sparsely labeled data. Test results not only demonstrate that the proposed ranking-based SSL approach proves to endow the pretrained model with degradation-aligned information from unlabeled data, where Spearman’s correlation coeficients between the learned aging scores and the ground-truth SOHs are extremely close to -1, but also showcase that the proposed SSL-Rank model can carry out accurate, robust SOH estimation after fine-tuning, even when only an extremely limited amount of 1% of unevenly distributed labeled training data is available, where an MAE of 1.718% and an RMSE of 2.329% can be achieved on the test cell. In addition, in-depth analyses are presented regarding the influences of label distribution of battery degradation data. So far, only one cell model under one cycling condition has been studied, so it would be meaningful for future work to extend the proposed SSL framework to cross-cell and cross-condition settings that better reflect the variability encountered in real-world applications. Considering the practical motivation and the encouraging experimental results, we believe this work could shed new light on SOH estimation of lithium-ion batteries under label sparsity in real-world applications.

## Data Availability

The public dataset utilized in this work can be accessed through Ref. [59].

## CRediT Authorship Contribution Statement

Jiaqi Yao: Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Resources, Data Curation, Writing - Original Draft, Writing - Review & Editing, Visualization. Julia Kowal: Writing - Review & Editing, Supervision.

## Acknowledgement

We thank the High-Performance Computing Cluster of TU Berlin ZECM for the GPU resources and the Open Access Publication Fund of TU Berlin for the support.

## Abbreviations

The following abbreviations are used in this manuscript: BMS Battery Management System BOL Beginning of Life CALCE Center for Advanced Life Cycle Engineering CC Constant Current CV Constant Voltage CNN Convolutional Neural Network DNN Deep Neural Network ECM Equivalent Circuit Model EKF Extended Kalman Filter EOL End of Life EV Electric Vehicle FC Fully Connected GRU Gated Recurrent Unit ICA Incremental Capacity Analysis LAM Loss of Active Material LCO Lithium Cobalt Oxide LLI Loss of Lithium Inventory LSTM Long Short-Term Memory MAD Median Absolute Deviation MAE Mean Absolute Error MAX Maximum Absolute Error MLP Multi-Layer Perceptron NASA National Aeronautics and Space Administration ReLU Rectified Linear Unit RMSE Root Mean Square Error RNN Recurrent Neural Network RUL Remaining Useful Life SL Supervised Learning SOC State of Charge SOH State of Health SSL Self-Supervised Learning SSL-MO Multi-Objective SSL SSL-Rank Ranking-Based SSL SSL-Recon Reconstruction-Based SSL TPE Tree-Structured Parzen Estimator UKF Unscented Kalman Filter

## References

[1] Q. Hassan, P. Viktor, T. J. Al-Musawi, B. Mahmood Ali, S. Algburi, H. M. Alzoubi, A. Khudhair Al-Jiboory, A. Zuhair Sameen, H. M. Salman, M. Jaszczur, The renewable energy role in the global energy Transformations, Renewable Energy Focus 48 (2024) 100545. doi:10.1016/j.ref.2024.100545.

[2] B. Lin, Z. Li, Towards world’s low carbon development: The role of clean energy, Applied Energy 307 (2022) 118160. doi:10.1016/j.apenergy.2021.118160.

[3] A. Olabi, M. A. Abdelkareem, Renewable energy and climate change, Renewable and Sustainable Energy Reviews 158 (2022) 112111. doi:10.1016/j.rser.2022.112111.

[4] Z. Feng, B. Yan, X. Shen, F. Zhang, Z. Tariq, W. Ouyang, Z. Han, A hybrid CNN-transformer surrogate model for the multi-objective robust optimization of geological carbon sequestration, Advances in Water Resources 196 (2025) 104897. doi:10.1016/j.advwatres.2025.104897.

[5] S. Roga, S. Bardhan, Y. Kumar, S. K. Dubey, Recent technology and challenges of wind energy generation: A review, Sustainable Energy Technologies and Assessments 52 (2022) 102239. doi:10.1016/j.seta.2022.102239.

[6] H. H. Pourasl, R. V. Barenji, V. M. Khojastehnezhad, Solar energy status in the world: A comprehensive review, Energy Reports 10 (2023) 3474–3493. doi:10.1016/j.egyr.2023.10.022.

[7] J. Xie, Y.-Z. Li, Telemetry-driven physics-informed prediction for on-orbit thermal–electrical performance of solar panels in satellite power system, Aerospace Science and Technology 177 (2026) 112268. doi:10.1016/j.ast.2026.112268.

[8] J. Yao, D. Droese, J. Kowal, Joint online estimation of state of charge and internal temperature of lithium-ion batteries with multi-task learning, Journal of Energy Storage 155 (2026) 121468. doi:10.1016/j.est.2026.121468.

[9] H. Niu, N. Zhang, Y. Lu, Z. Zhang, M. Li, J. Liu, N. Zhang, W. Song, Y. Zhao, Z. Miao, Strategies toward the development of high-energydensity lithium batteries, Journal of Energy Storage 88 (2024) 111666. doi:10.1016/j.est.2024.111666.

[10] T. Horiba, T. Maeshima, T. Matsumura, M. Koseki, J. Arai, Y. Muranaka, Applications of high power density lithium ion batteries, Journal of Power Sources 146 (1-2) (2005) 107–110. doi:10.1016/j.jpowsour.2005.03.205.

[11] N. Nasajpour-Esfahani, H. Garmestani, M. Bagheritabar, D. J. Jasim, D. Toghraie, S. Dadkhah, H. Firoozeh, Comprehensive review of lithium-ion battery materials and development challenges, Renewable and Sustainable Energy Reviews 203 (2024) 114783. doi:10.1016/j.rser.2024.114783.

[12] J. Zhang, H. Huang, G. Zhang, Z. Dai, Y. Wen, L. Jiang, Cycle life studies of lithium-ion power batteries for electric vehicles: A review, Journal of Energy Storage 93 (2024) 112231. doi:10.1016/j.est.2024.112231.

[13] Y. Liang, C.-Z. Zhao, H. Yuan, Y. Chen, W. Zhang, J.-Q. Huang, D. Yu, Y. Liu, M.-M. Titirici, Y.-L. Chueh, H. Yu, Q. Zhang, A review of rechargeable batteries for portable electronic devices, InfoMat 1 (1) (2019) 6–32. doi:10.1002/inf2.12000.

[14] G. Zubi, R. Dufo-López, M. Carvalho, G. Pasaoglu, The lithium-ion battery: State of the art and future perspectives, Renewable and Sustainable Energy Reviews 89 (2018) 292–308. doi:10.1016/j.rser.2018.03.002.

[15] P. H. Camargos, P. H. J. Dos Santos, I. R. Dos Santos, G. S. Ribeiro, R. E. Caetano, Perspectives on Li-ion battery categories for electric vehicle applications: A review of state of the art, International Journal of Energy Research 46 (13) (2022) 19258–19268. doi:10.1002/er.7993.

[16] S. S. Rangarajan, S. P. Sunddararaj, A. Sudhakar, C. K. Shiva, U. Subramaniam, E. R. Collins, T. Senjyu, Lithium-Ion Batteries—The Crux of Electric Vehicles with Opportunities and Challenges, Clean Technologies 4 (4) (2022) 908–930. doi:10.3390/cleantechnol4040056.

[17] W. Diao, J. Jiang, C. Zhang, H. Liang, M. Pecht, Energy state of health estimation for battery packs based on the degradation and inconsistency, Energy Procedia 142 (2017) 3578–3583. doi:10.1016/j.egypro.2017.12.248.

[18] J. Yao, H. Zhao, J. Kowal, Fast-adaptive early-stage remaining useful life prediction of lithium-ion batteries with meta-learning, Journal of Power Sources 660 (2025) 238569. doi:10.1016/j.jpowsour.2025.238569.

[19] S. Jin, X. Sui, X. Huang, S. Wang, R. Teodorescu, D.-I. Stroe, Overview of Machine Learning Methods for Lithium-Ion Battery Remaining Useful Lifetime Prediction, Electronics 10 (24) (2021) 3126. doi:10.3390/electronics10243126.

[20] O. Demirci, S. Taskin, E. Schaltz, B. Acar Demirci, Review of battery state estimation methods for electric vehicles-Part II: SOH estimation, Journal of Energy Storage 96 (2024) 112703. doi:10.1016/j.est.2024.112703.

[21] R. Xiong, L. Li, J. Tian, Towards a smarter battery management system: A critical review on battery state of health monitoring methods, Journal of Power Sources 405 (2018) 18–29. doi:10.1016/j.jpowsour.2018.10.019.

[22] S. M. Rezvanizaniani, Z. Liu, Y. Chen, J. Lee, Review and recent advances in battery health monitoring and prognostics technologies for electric vehicle (EV) safety and mobility, Journal of Power Sources 256 (2014) 110–124. doi:10.1016/j.jpowsour.2014.01.085.

[23] L. Chen, Z. Lü, W. Lin, J. Li, H. Pan, A new state-of-health estimation method for lithium-ion batteries through the intrinsic relationship between ohmic internal resistance and capacity, Measurement 116 (2018) 586–595. doi:10.1016/j.measurement.2017.11.016.

[24] X. Hu, J. Jiang, D. Cao, B. Egardt, Battery Health Prognosis for Electric Vehicles Using Sample Entropy and Sparse Bayesian Predictive Modeling, IEEE Transactions on Industrial Electronics (2015) 1–1doi:10.1109/TIE.2015.2461523.

[25] K. S. Ng, C.-S. Moo, Y.-P. Chen, Y.-C. Hsieh, Enhanced coulomb counting method for estimating state-of-charge and state-of-health of lithium-ion batteries, Applied Energy 86 (9) (2009) 1506–1511. doi:10.1016/j.apenergy.2008.11.021.

[26] X. Li, J. Jiang, L. Y. Wang, D. Chen, Y. Zhang, C. Zhang, A capacity model based on charging process for state of health estimation of lithium ion batteries, Applied Energy 177 (2016) 537–543. doi:10.1016/j.apenergy.2016.05.109.

[27] J. Wang, P. Liu, J. Hicks-Garner, E. Sherman, S. Soukiazian, M. Verbrugge, H. Tataria, J. Musser, P. Finamore, Cycle-life model for graphite-LiFePO4 cells, Journal of Power Sources 196 (8) (2011) 3942–3948. doi:10.1016/j.jpowsour.2010.11.134.

[28] M. Safari, C. Delacourt, Simulation-Based Analysis of Aging Phenomena in a Commercial Graphite/LiFePO4 Cell, Journal of The Electrochemical Society 158 (12) (2011) A1436. doi:10.1149/2.103112jes.

[29] J. Christensen, J. Newman, A Mathematical Model for the Lithium-Ion Negative Electrode Solid Electrolyte Interphase, Journal of The Electrochemical Society 151 (11) (2004) A1977. doi:10.1149/1.1804812.

[30] IL-Song Kim, A Technique for Estimating the State of Health of Lithium Batteries Through a Dual-Sliding-Mode Observer, IEEE Transactions on Power Electronics 25 (4) (2010) 1013–1022. doi:10.1109/TPEL.2009.2034966.

[31] S. Hosseininasab, C. Lin, S. Pischinger, M. Stapelbroek, G. Vagnoni, State-ofhealth estimation of lithium-ion batteries for electrified vehicles using a reducedorder electrochemical model, Journal of Energy Storage 52 (2022) 104684. doi:10.1016/j.est.2022.104684.

[32] G. L. Plett, Extended Kalman filtering for battery management systems of LiPB-based HEV battery packs, Journal of Power Sources 134 (2) (2004) 277– 292. doi:10.1016/j.jpowsour.2004.02.033.

[33] S. Liu, X. Dong, X. Yu, X. Ren, J. Zhang, R. Zhu, A method for state of charge and state of health estimation of lithium-ion battery based on adaptive unscented Kalman filter, Energy Reports 8 (2022) 426–436. doi:10.1016/j.egyr.2022.09.093.

[34] S. Schwunk, N. Armbruster, S. Straub, J. Kehl, M. Vetter, Particle filter for state of charge and state of health estimation for lithium– iron phosphate batteries, Journal of Power Sources 239 (2013) 705–710. doi:10.1016/j.jpowsour.2012.10.058.

[35] Z. Ren, C. Du, A review of machine learning state-of-charge and state-of-health estimation algorithms for lithium-ion batteries, Energy Reports 9 (2023) 2993– 3021. doi:10.1016/j.egyr.2023.01.108.

[36] B. Xia, M. Ye, M. Wei, Q. Wang, G. Lian, Y. Li, SOH estimation of lithiumion batteries with local health indicators in multi-stage fast charging protocols, Energy 334 (2025) 137617. doi:10.1016/j.energy.2025.137617.

[37] Z. Chen, Y. Peng, J. Shen, Q. Zhang, Y. Liu, Y. Zhang, X. Xia, Y. Liu, State of health estimation for lithium-ion batteries based on fragmented charging data

and improved gated recurrent unit neural network, Journal of Energy Storage 115 (2025) 115952. doi:10.1016/j.est.2025.115952.

[38] Z. Xia, J. A. A. Qahouq, Lithium-Ion Battery Ageing Behavior Pattern Characterization and State-of-Health Estimation Using Data-Driven Method, IEEE Access 9 (2021) 98287–98304. doi:10.1109/ACCESS.2021.3092743.

[39] S. Zhang, B. Zhai, X. Guo, K. Wang, N. Peng, X. Zhang, Synchronous estimation of state of health and remaining useful lifetime for lithium-ion battery using the incremental capacity and artificial neural networks, Journal of Energy Storage 26 (2019) 100951. doi:10.1016/j.est.2019.100951.

[40] G.-W. You, S. Park, D. Oh, Diagnosis of Electric Vehicle Batteries Using Recurrent Neural Networks, IEEE Transactions on Industrial Electronics 64 (6) (2017) 4885–4893. doi:10.1109/TIE.2017.2674593.

[41] Y. Fan, F. Xiao, C. Li, G. Yang, X. Tang, A novel deep learning framework for state of health estimation of lithium-ion battery, Journal of Energy Storage 32 (2020) 101741. doi:10.1016/j.est.2020.101741.

[42] A. Jaiswal, A. R. Babu, M. Z. Zadeh, D. Banerjee, F. Makedon, A Survey on Contrastive Self-Supervised Learning, Technologies 9 (1) (2020) 2. doi:10.3390/technologies9010002.

[43] X. Liu, F. Zhang, Z. Hou, L. Mian, Z. Wang, J. Zhang, J. Tang, Self-supervised Learning: Generative or Contrastive, IEEE Transactions on Knowledge and Data Engineering (2021) 1–1doi:10.1109/TKDE.2021.3090866.

[44] J. Gui, T. Chen, J. Zhang, Q. Cao, Z. Sun, H. Luo, D. Tao, A Survey on Self-Supervised Learning: Algorithms, Applications, and Future Trends, IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (12) (2024) 9052– 9071. doi:10.1109/TPAMI.2024.3415112.

[45] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2019). arXiv:1810.04805, doi:10.48550/arXiv.1810.04805.

[46] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner,

S. McCandlish, A. Radford, I. Sutskever, D. Amodei, Language Models are Few-Shot Learners (2020). arXiv:2005.14165, doi:10.48550/arXiv.2005.14165.

[47] Z. Wu, Y. Xiong, S. X. Yu, D. Lin, Unsupervised Feature Learning via Nonparametric Instance Discrimination, in: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, IEEE, Salt Lake City, UT, 2018, pp. 3733–3742. doi:10.1109/CVPR.2018.00393.

[48] T. Chen, S. Kornblith, M. Norouzi, G. Hinton, A Simple Framework for Contrastive Learning of Visual Representations (2020). arXiv:2002.05709, doi:10.48550/arXiv.2002.05709.

[49] H. Chen, Y. Wang, B. Lagadec, A. Dantcheva, F. Bremond, Joint Generative and Contrastive Learning for Unsupervised Person Re-identification, in: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, Nashville, TN, USA, 2021, pp. 2004–2013. doi:10.1109/CVPR46437.2021.00204.

[50] Z. Qi, R. Dong, G. Fan, Z. Ge, X. Zhang, K. Ma, L. Yi, Contrast with Reconstruct: Contrastive 3D Representation Learning Guided by Generative Pretraining (2023). arXiv:2302.02318, doi:10.48550/arXiv.2302.02318.

[51] S. Gidaris, P. Singh, N. Komodakis, Unsupervised Representation Learning by Predicting Image Rotations (2018). arXiv:1803.07728, doi:10.48550/arXiv.1803.07728.

[52] P. Agrawal, J. Carreira, J. Malik, Learning to See by Moving (2015). arXiv:1505.01596, doi:10.48550/arXiv.1505.01596.

[53] Y. Che, Y. Zheng, X. Sui, R. Teodorescu, Boosting battery state of health estimation based on self-supervised learning, Journal of Energy Chemistry 84 (2023) 335–346. doi:10.1016/j.jechem.2023.05.034.

[54] T. Wang, Z. Ma, S. Zou, Z. Chen, P. Wang, Lithium-ion battery state-of-health estimation: A self-supervised framework incorporating weak labels, Applied Energy 355 (2024) 122332. doi:10.1016/j.apenergy.2023.122332.

[55] Q. Liang, M. Zhang, Y. Jin, L. Xia, C. Liu, T. Zhang, Multi-level contrastive self-supervised learning with dynamic spectral-temporal embedding for state-of-health estimation of lithium-ion battery with limited labeled data, IEEE Transactions on Instrumentation and Measurement (2025) 1– 1doi:10.1109/TIM.2025.3614866.

[56] Y. Lecun, L. Bottou, Y. Bengio, P. Hafner, Gradient-based learning applied to document recognition, Proceedings of the IEEE 86 (11) (1998) 2278–2324. doi:10.1109/5.726791.

[57] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, Y. Bengio, Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (2014). doi:10.48550/ARXIV.1406.1078.

[58] S. Hochreiter, J. Schmidhuber, Long Short-Term Memory, Neural Computation 9 (8) (1997) 1735–1780. doi:10.1162/neco.1997.9.8.1735.

[59] Battery Data | Center for Advanced Life Cycle Engineering, https://calce.umd.edu/battery-data.

[60] H. Zhang, X. Gui, S. Zheng, Z. Lu, Y. Li, J. Bian, BatteryML:An Open-source platform for Machine Learning on Battery Degradation (2023). doi:10.48550/ARXIV.2310.14714.

[61] T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama, Optuna: A Nextgeneration Hyperparameter Optimization Framework (2019). arXiv:1907.10902, doi:10.48550/arXiv.1907.10902.
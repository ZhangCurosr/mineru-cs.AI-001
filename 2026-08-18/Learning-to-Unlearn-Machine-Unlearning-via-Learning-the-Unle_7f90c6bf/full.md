# Learning to Unlearn: Machine Unlearning via Learning the Unlearning Behaviors

Hang Zhang   
State Key Laboratory for Novel   
Software Technology   
Nanjing University   
Nanjing, China   
zhanghang@lamda.nju.edu.cn   
Weijie Xu   
State Key Laboratory for Novel   
Software Technology   
Nanjing University   
Nanjing, China   
xuwj@lamda.nju.edu.cn   
Kaifeng Zhang   
State Key Laboratory for Novel   
Software Technology   
Nanjing University   
Nanjing, China   
zhangkf2022@lamda.nju.edu.cn

Ye Zhu Centre for Cyber Resilience and Trust Deakin University Burwood, VIC, Australia ye.zhu@ieee.org

Yixiao Ma   
State Key Laboratory for Novel   
Software Technology   
Nanjing University   
Nanjing, China   
mayx@lamda.nju.edu.cn   
Kai Ming Ting<sup>✉</sup>   
State Key Laboratory for Novel   
Software Technology   
Nanjing University   
Nanjing, China   
tingkm@nju.edu.cn

## Abstract

Various machine unlearning techniques have been developed in response to privacy legislation requirements, enabling individuals to exercise their legal right to have their data $D _ { f }$ removed from a machine learning model. This process is typically accomplished via the use of an unlearning function denoted as �. Existing methods focus on designing an intricate � to unlearn $D _ { f } \subset D$ from a previous model �(�), so that the unlearned model performs as closely as possible to the retrained model $A ( D \setminus D _ { f } )$ . However, these meth ods often suffer from high computational costs when dealing with massive training data, as the complex structures of � become a bottleneck even for models with fewer parameters. Inspired by Learning to Optimize, we introduce the first learning-based model-agnostic approach, Learning-to-UnLearn (L2UL). Our core insight is to shift from manually designing � to learning the unlearning behaviors from a distribution perspective, thereby acquiring a simple and efficient � via learning. Our experimental results demonstrate that the accuracy achieved by L2UL is comparable to that of retraining while exhibiting impressive efficiency, particularly in data-intensive scenarios. Furthermore, we validate the performance and scalability of our method on larger models ResNet.

## CCS Concepts

• Security and privacy; • Computing methodologies → Machine learning; Artificial intelligence;

Keywords Machine unlearning, Data privacy, Privacy-preserving learning.

ACM Reference Format:   
Hang Zhang, Kaifeng Zhang, Yixiao Ma, Weijie Xu, Ye Zhu, and Kai Ming Ting. 2018. Learning to Unlearn: Machine Unlearning via Learning the Unlearning Behaviors . In Proceedings of Make sure to enter the correct conference titlefrom your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 12 pages. https://doi.org/XXXXXXX. XXXXXXX

## 1 Introduction

With the progress of society, there has been a significant increase in the amount of available data. This growth has enriched our lives by allowing us to access a wide range of data shared by numerous individuals online. However, it also presents risks to the privacy of users. In response to this, legislative measures like CCPA (California Consumer Privacy Act), PIPEDA (Personal Information Protection and Electronic Documents Act), and GDPR (General Data Protection Regulation) have been introduced to establish rules that protect users privacy by the right to delete their data. The task of erasing data from a machine learning model is named machine unlearning [1, 4, 22].

A naive approach is to retrain the model on the remaining data, but this is expensive as models are often trained on very large datasets. Proposing efficient and effective machine unlearning methods has attracted the attention of researchers in recent years.

Current approaches focus on designing a complex unlearning function � so that the unlearned model performs close to the retrained model. For example, some methods design � as a retraining on all or part of the data, which is very time-consuming and requires retraining every time on an unlearning request. Some methods necessitate the calculation of the Hessian matrix for the designed �, which further increases the computational cost. Consequently, even for relatively simple models, the unlearning efficiency remains bottlenecked by the complexity of the designed �, especially when navigating massive datasets.

The goal of an unlearning function � is to produce a model that has unlearned the user data and performs close to the model retrained on the remaining data. Why don’t we learn to unlearn directly?

Inspired by Learning to Optimize (L2O) [6, 27], traditional optimization algorithms like Adam are manually designed by human experts through theoretical derivation. In contrast, the L2O approach aims to let machines automatically learn how to optimize. Specifically, the L2O model takes the current state of the optimization iteration as input (such as the current solution and its gradient) and learns to predict the update amount for the next iteration. Instead of designing a complex and highly time-consuming $U ,$ we use the retrained model as ground truth to learn a � whose structure is simple but enough to unlearn $D _ { f }$ effectively. In this way, we can achieve the goal of unlearning much more efficiently. Although many methods claim learning to unlearn [5, 17], they usually refer to learning a new decision boundary to achieve forgetting rather than learning the behavior of unlearning.

In this paper, we introduce a model-agnostic framework called Learning-to-UnLearn (L2UL), which employs a machine-learning approach to learn a � .

The main contributions of this work are:

(1) Pioneering a new framework of machine unlearning via learn ing, named Learning-to-UnLearn (L2UL).

(2) Providing the generalization bound of a model unlearned by L2UL in the context of Logistic Regression and Multi-Layer Perceptron (MLP).

(3) Conducting an empirical validation of the effectiveness and efficiency of our proposed method.

The advantages of the proposed L2UL over current methods are two-fold: (i) Unlike existing methods, in L2UL, the acquisition of � is achieved through learning rather than designing. (ii) Due to the simple structure of �, the execution time of unlearning via the learned � is very short.

## 2 Problem Formulation

Notations used in this paper are summarized in Table 1.

Table 1: Key notations used in the paper.
<table><tr><td>Symbols</td><td>Definition</td></tr><tr><td> $\chi$ </td><td>Sample space</td></tr><tr><td> $_ y$ </td><td>Label space</td></tr><tr><td> $D$ </td><td>Training set</td></tr><tr><td> $D _ { f }$ </td><td>Forget set</td></tr><tr><td> $A ( \cdot )$ </td><td>Learning algorithm</td></tr><tr><td> $U ( \cdot )$ </td><td>Unlearning algorithm</td></tr><tr><td> $\theta$ </td><td>Parameters of the model trained on D</td></tr><tr><td> $\theta ^ { \prime }$ </td><td>Parameters of the model trained on  $D \backslash D _ { f }$ </td></tr><tr><td> $\mathcal { P }$ </td><td>Probability distribution</td></tr><tr><td> $x$ </td><td>A vector in  $\mathbb { R } ^ { d }$ </td></tr><tr><td> $\kappa$ </td><td>Kernel Function</td></tr><tr><td> $\kappa _ { I }$ </td><td>Isolation Kernel</td></tr><tr><td> $t , \psi$ </td><td>Parameter of  $\kappa _ { I }$ </td></tr></table>

Let X denote the sample space, and Y denote the label space. A data point is an element in ${ \mathcal { Z } } = { \boldsymbol { \chi } } \times { \boldsymbol { y } } .$ A hypothesis function $h : X \to y$ is learned on training set $D \subset { \mathcal { Z } }$ by algorithm � to assign $y \in y \mathrm { ~ t o ~ } x \in X$

Definition 2.1. A machine learning algorithm is a map � from a subset of $z$ to a hypothesis function $h \in { \mathcal { H } } .$ , where H is the space of all hypothesis functions of the algorithm.

After � learned a model from a dataset $D , A ( D )$ can be applied to a test set. When some users request to have their data $D _ { f } \subset D$ deleted, model �(�) is obliged to have unlearned their data. The task of unlearning data from a learned model is machine unlearning.

Figure 1 shows the workflow of machine unlearning. To fulfill a request to unlearn $D _ { f }$ from $A ( D )$ , an unlearning algorithm ${ \cal U } ( D , D _ { f } , A ( D ) )$ produces a model which is expected to be the same or close to the model $A ( D \setminus D _ { f } )$ retrained on the dataset $D \backslash D _ { f }$

Definition 2.2. A machine unlearning algorithm is a map $U :$ $2 ^ { \mathcal { Z } } \times 2 ^ { \mathcal { Z } } \times \mathcal { H }  \mathcal { H } ,$ where $_ { 2 ^ { \mathcal { Z } } }$ is the power set of $Z ,$ which is set containing all of $z _ { \mathrm { ~ s ~ } }$ subset. � takes $D , D _ { f } , A ( D )$ as input, and outputs the unlearned model, which is expected to be as close to the retrained model $A ( D \setminus D _ { f } )$ as possible.

Definition 2.3. Exact Unlearning: given a randomized learning algorithm �, a dataset �, and a data subset $D _ { f } \subset D$ to be forgotten, an unlearning algorithm � is an exact unlearning process if and only if $\mathcal { P } ( U ( D , D _ { f } , A ( D ) ) ) = \mathcal { P } ( A ( D \setminus D _ { f } ) )$ , where $\mathcal { P } ( A ( D ) )$ defines the distribution of all models trained on a dataset � by a learning algorithm �.

The naive unlearning algorithm simply retrains a new model on $D \setminus D _ { f }$ to achieve exact unlearning. However, it is expensive to retrain when $| D \setminus D _ { f } |$ is large.

## 3 Related Work

Many machine unlearning algorithms that remove some user data from a previously trained model without retraining have been proposed recently. According to the workflow shown in Figure 1, we divide the related work into three categories: �-oriented, $D _ { f }$ -oriented, and $A ( D )$ -oriented unlearning.

�-oriented unlearning operates on subsets of � such that only retaining on some subsets is required upon a request to unlearn $D _ { f } .$ For example, SISA [1] divides the training data � into shards and divides each shard into slices. Each shard is used to train a constituent model, incrementally incorporating slices and preserving its parameters until the training set is extended with a new slice. When $D _ { f }$ is required to be forgotten, retraining is limited to the specific constituent model whose shards encompass $D _ { f }$ . Many methods share a similar intuition, such as [2, 4, 10, 13, 15, 38].

$D _ { f }$ -oriented unlearning considers the impact of forgotten data on the model, through influence functions [12, 34, 36] and Markov Chain Monte Carlo-based sampling [9, 21]. When $D _ { f }$ needs to be forgotten, the impact of $D _ { f }$ on the model can be erased accordingly.

�(�)-oriented unlearning considers how the parameters of a learned model should be updated after $D _ { f }$ is forgotten from the training set [18, 20, 25, 28, 37]. [29] updates the parameters by adding noise to the parameters to forget and then tuning the model on the remaining data $D \backslash D _ { f } .$ . Few-shot learning [8] is used in the case where � is unavailable, and only model $A ( D )$ is known.

Although existing works can produce an unlearned model close to the retrained model, none of them use a learning technique to obtain � . Our proposed L2UL, an �(�)-oriented unlearning algorithm, tries to learn the unlearning function � , instead of designing one like current works.

![](images/4b40e08f3ef99bec4e1d49ae4eb93fdbb8a8dcc495429a47b8217e604cbf5521.jpg)  
Figure 1: The Machine unlearning workflow and the framework of our proposed method L2UL. The model �(�) is learned on dataset � (which includes $D _ { f } )$ by �. An unlearning algorithm � unlearns $D _ { f }$ from �(�) to obtain a revised model, which is expected to be equal to the retrained model $A ( D \setminus D _ { f } )$

## 4 Proposed Method

## 4.1 The Framework of L2UL

Recall Definition 2.2 and Figure 1, � has three inputs, � (represented as ${ \bf { a } } \left| D \right| \times d$ matrix), $D _ { f }$ (represented as a $| D _ { f } | \times d$ matrix) and the parameter vector � of model $A ( D )$ . A machine unlearning algorithm � must satisfy the following properties:

Proposition 4.1. The output of � should remain unchanged after any two rows in the data matrix of � or $D _ { f }$ are exchanged.

This is easy to understand because exchanging any two rows of � and $D _ { f }$ does not change the dataset.

An effective method that satisfies the Proposition 4.1 is to represent � and $D _ { f }$ as distribution $\mathcal { P } _ { D }$ and $\mathcal { P } _ { D _ { f } }$ . No matter how the order of points of the dataset changes, its distribution remains the same. Therefore, we redefine Definition 2.2 as:

Definition 4.2. A machine unlearning algorithm is a function �: $\mathcal { P } \times \mathcal { P } \times \mathcal { H }  \mathcal { H }$ , where $\mathcal { P }$ is the space containing all the probability distributions.

Different from Definition 2.2, Definition 4.2 defines machine unlearning from the perspective of distribution. Based on Definition 4.2 and the unlearning workflow shown in Figure 1, L2UL is naturally proposed by using three components of the unlearning workflow: $D , D _ { f }$ , and $A ( D )$ . Figure 1 shows the framework of L2UL, which uses a neural network $N _ { U } { } ^ { 1 }$ to produce the unlearned model $A _ { \theta ^ { \prime } }$

In order to vectorize distribution $\mathcal { P } _ { D }$ and $\mathcal { P } _ { D _ { f } }$ as input to $N _ { U }$ , we employ Kernel Mean Embedding (KME) [26] to transform distribution $\mathcal { P } _ { D }$ and $\mathcal { P } _ { D _ { f } }$ to vector $\mu _ { D }$ and $\mu _ { f }$ in Reproducing Kernel Hilbert Space (RKHS).

Kernel mean embedding is a nonparametric method that uses an element of an RKHS as the representation of a probability distribution. The mean feature map $\mu _ { \mathbb { P } } = \mathbb { E } [ \kappa ( \cdot , D ) ]$ of points � based on kernel � embeds distribution P into RKHS, and preserves all of the statistical features of P.

In our framework, when using some common kernels such as the Gaussian kernel, the Laplacian kernel, etc., $\mu _ { \mathbb { P } }$ will map P into an infinite-dimensional space, which cannot be used as input to the neural network. In other words, we need a kernel whose RKHS dimension is finite. One approach is to use low-rank representations of the kernel matrix. The most popular examples are Nyström method [14, 19, 35] and the Random Fourier Features [24, 39]. However, these are all approximate methods and will reduce effectiveness.

Another approach is to use a kernel with an exact and finitedimensional feature map. Isolation Kernel [33] is such a kernel that directly obtains the feature map without calculating the kernel matrix, which has been widely used and performs well [3, 31, 40]. Let $D \subset \mathbb { R } ^ { d }$ be a dataset sampled from an unknown distribution P. Isolation Kernel uses a partition mechanism such as Voronoi Diagram [23, 41] or hypersphere [32] to partition the data into |D| cells, where $| { \mathcal { D } } | = \psi ,$ and $\mathcal { D } \subset D$ is sampled from � as the seeds of partition mechanism. Let $\mathbb { H } _ { \psi } ( D )$ denotes the set of all partitions �, each cell $\mathcal { I } [ z ]$ of partition � isolates $z \in \mathcal { D }$ from the rest of the points in D, the definition of Isolation Kernel is shown as follows:

Definition 4.3. Isolation Kernel: $\forall x , y \in \mathbb { R } ^ { d }$ , Isolation Kernel of � and � is defined to be the probability that � and � fall into the same cell $\mathcal { I } [ z ]$ of partition � over all the partitions $H \in \mathbb { H } _ { \psi } ( D )$ $\kappa _ { I } ( x , y | D ) = \mathbb { E } _ { \mathbb { H } _ { \psi } ( D ) } \left[ \mathbf { 1 } ( x , y \in \mathcal { I } [ z ] | \mathcal { I } [ z ] \in H ) \right]$

Given the � partitions $H _ { 1 } , . . . , H _ { t }$ , the feature map $\Phi ( x )$ of $\kappa _ { I }$ is a � × �-dimensional binary column vector.

Definition 4.4. Isolation Kernel’s feature map, denoted as $\Phi ( x )$ , is a vector that represents the cell in which � falls into over partition of � times. For � $\epsilon \mathbb { R } ^ { d } .$ , the dimension of Φ(�) is $\psi \times t ,$ where � is the number of cells in one partition �. Each element of Φ(�) is either 0 or 1, indicating the cell in which � falls.

Since the dimension of $\Phi ( x )$ is finite, the KME of $\kappa _ { I }$ can be obtained by simply computing the mean of all feature maps.

Then $\mu _ { D } , \mu _ { f } .$ , and the parameters � of model � are concatenated as the input x of the neural network, which contains two hidden layers. The KME of distribution $\mathbb { P }$ on the given data � is:

$$
\mu _ { \mathbb { P } } = \mathbb { E } [ \kappa _ { I } ( \cdot , D ) ] = \frac { \sum _ { x \in D } \Phi ( x ) } { | D | } ,\tag{1}
$$

where $\Phi ( X )$ is IK’s finite-dimensional feature map determined by two hyperparameters $t , \psi .$ This is also known as the isolation distribution kernel (IDK)[32]. According to Equation 1, we have: $\mu _ { D } =$ $\frac { \sum _ { x \in D } \Phi ( x ) } { | D | }$ and $\begin{array} { r } { \mu _ { f } = \frac { \sum _ { x \in D _ { f } } \Phi ( x ) } { | D _ { f } | } } \end{array}$ . �<sub>�</sub> and $\mu _ { f }$ are later concatenated with the parameters of the original model as input x.

The output $\theta ^ { \prime }$ is the parameter of the unlearned model ${ \cal U } ( D , D _ { f } , A ( D ) )$ And Mean Squared Error (MSE) loss is used as the objective function for training �:

$$
\mathcal { L } = \frac { 1 } { | \theta ^ { \prime } | } \sum _ { i = 1 } ^ { | \theta ^ { \prime } | } ( \theta _ { i } ^ { \prime } - \theta _ { i } ^ { r } ) ^ { 2 } ,\tag{2}
$$

where $\theta ^ { r }$ is the parameters of retrained model $A ( D \setminus D _ { f } )$

Intuitively, we are focusing on models with relatively few parameters but very large training datasets. Because the model has few parameters, � can be learned efficiently, and the large amount of training data can be represented via kernel mean embedding. Therefore, L2UL is effective and highly efficient.

## 4.2 How to Learn �

In order to train �, first we need a training set $\mathcal { D } = \mathcal { X } \times \mathcal { X }$ . We produce D by the following three steps. First, we generate � by randomly sampling � points from $D \left( s \ll \vert D \vert \right)$ , and train a model $A ( { \mathfrak { D } } )$ on �. Second, we randomly select some samples as a forget dataset $\mathfrak { D } _ { f } \subset \mathfrak { D } . \mu _ { \mathfrak { D } } , \mu _ { \mathfrak { D } _ { f } }$ of � and ${ \mathfrak { D } } _ { f }$ are obtained using Equation 1. The parameter � of model �(�) is concatenated with $\mu _ { \mathfrak { D } }$ and $\mu _ { \mathfrak { D } _ { f } }$ as x. Finally, we obtain $\theta ^ { r } .$ , the parameter of retrained model $A ( \mathfrak { D } \backslash \mathfrak { D } _ { f } )$ . Now we have one instance $( \mathbf { x } , \theta ^ { r } ) \in \mathcal { X } \times \mathcal { Y }$

The above steps are repeated � times to obtain $\mathcal { D }$ containing � samples. $N _ { U }$ is later trained on $\mathcal { D }$ by minimizing the loss function $\mathcal { L }$ shown in Equation 2. Since the parameters in KME do not require learning, learning $N _ { U }$ is equivalent to learning �. When there is a $D _ { f }$ that needs to be unlearned from model $A ( D )$ , we can obtain the unlearned model $A _ { \theta ^ { \prime } }$ via the learned $N _ { U }$

An important point to highlight is that once we have obtained $N _ { U }$ if we need to unlearn any user data $D _ { f } ,$ we can obtain the unlearned model ${ \cal U } ( D , D _ { f } , A ( D ) )$ via the learned $N _ { u }$ directly without having to repeat the preprocessing and training phase.

The pseudo code of preprocessing for obtaining D and Φ(·) for L2UL, and unlearning $D _ { f }$ are shown in Algorithm 1 and Algorithm 2, respectively.

## 4.3 Generalization Bound Analysis

Here, we provided the generalization bound of the classifier (Logistic Regression, Multilayer Perceptron) unlearned by L2UL.

Denote the expected loss of Logistic Regression model $A ( x | \theta ) =$ $\begin{array} { r } { \frac { e ^ { \theta x + b } } { 1 + e ^ { \theta x + b } } \operatorname { a s : } L ( A ) = E _ { x , y } [ y \log ( A ( x ) ) + ( 1 - y ) \log ( 1 - A ( x ) ] . } \end{array}$

L2UL learns the unlearning function � . The expected loss of the model unlearned by $U$ is upper-bounded, as shown in the following Theorem 4.5.

## Theorem 4.5.

$$
L ( U ( A ( D ) , D , D _ { f } ) ) \le L ( A ( D \setminus D _ { f } ) ) + 2 R \sqrt \epsilon ,
$$

Algorithm 1: Preprocessing   
Input :�: Dataset, �: Number of Samples in ${ \mathcal { D } } ,$ �: size of   
${ \mathfrak { D } } , \psi ,$ �: Parameters of $\kappa _ { I }$   
Output $: \mathcal { D } , \Phi ( \cdot )$   
1 $\mathcal { D } = \{ \} ;$   
2 Get the map function $\Phi ( \cdot )$ using $\kappa _ { I } . ;$   
3 $\forall x \in D ,$ get the feature map Φ(�);   
for � ← 1 to � do   
5 � ← randomly sample � points from $D ;$   
6 ${ \mathfrak { D } } _ { f }$ ← randomly sample from � ;   
7 $A ( \mathfrak { D } ) , A ( \mathfrak { D } \backslash \mathfrak { D } _ { f } ) $ train A on ${ \mathfrak { D } } , { \mathfrak { D } } \backslash { \mathfrak { D } } _ { f } ;$   
8 $\theta , \theta ^ { r } \gets$ Extract parameters of $A ( { \mathfrak { D } } ) , A ( { \mathfrak { D } } \setminus { \mathfrak { D } } _ { f } ) ;$   
9 $\mu _ { \mathfrak { D } } , \mu _ { f }$ ← obtain KME using Equation 1 ;   
10 x ← concatenation of �<sub>�</sub>, �<sub>�</sub> and � ;   
11 $\mathcal { D }  \mathcal { D } \cup \{ ( \mathbf { x } , \theta ^ { r } ) \}$ ;   
12 end   
13 Return $\mathcal { D } , \Phi ( \cdot ) .$

```perl
Algorithm 2: Unlearn $D _ { f }$
Input $: N _ { U } , \mu _ { D } , \Phi ( \cdot ) , A ( D )$
Output :�<sub>�</sub>′
1 $\forall x \in D _ { f }$ , get the feature map $\Phi ( x ) ;$
2 � ← obtain KME of $D _ { f }$ through Equation 1;
3 x ← concatenation of $\mu _ { D } , \mu _ { f }$ and $A ( D ) { \mathrm { \bar { s } } }$ parameter $\theta$ ;
4 $\theta ^ { \prime } \gets N _ { U } ( \mathbf { x } ) ;$
5 Return unlearned model $A _ { \theta ^ { \prime } }$
```

where � is the radius ofthe dataset � $( \forall x \in X , | | x | | \leq R ) ,$ , and � is the generalization error bound (MSE) ofthe unlearning model � in L2UL.

Similarly, the expected loss of Multilayer Perceptron (MLP) unlearned by � is also upper-bounded, as shown in Theorem 4.6.

Theorem 4.6.

$$
L ( U ( A ( D ) , D , D _ { f } ) ) \le L ( A ( D \setminus D _ { f } ) ) + C { \sqrt { \epsilon } } ,
$$

where � is cross entropy loss, � is a finite scalar determined by the MLP structure and � is the generalization bound (MSE) ofthe unlearning model � in L2UL.

Both Theorem 4.5 and 4.6 indicate that the expected loss of the classifier unlearned by L2UL is close to that of retrained classifier.

## 5 Experiments

System: The experiments are executed on a Linux machine with 1T GB RAM and an AMD 128-core CPU, with each core running at 2 GHz.

Data: We use seven public datasets in our experiments<sup>2</sup>. The specifications of the datasets are summarised in Table 2.

Table 2: Dataset Summary. n=no.instances, d=no.attributes.
<table><tr><td></td><td></td><td></td><td>|Magic Adult Sepsus</td><td>Skin</td><td>Covetype</td><td>SUSY</td><td>HIGGS</td></tr><tr><td>n</td><td></td><td>1902032562</td><td>40328</td><td>245057</td><td>581012</td><td></td><td>5000000 11000000</td></tr><tr><td>d</td><td>10</td><td>14</td><td>89</td><td>3</td><td>54</td><td>18</td><td>28</td></tr></table>

Comparison algorithms: We compare $\mathrm { L } 2 \mathrm { U L } ^ { 3 }$ with Retrain, one �-oriented unlearning algorithm: SISA and two $A ( D )$ -oriented unlearning algorithms: DeltaGrad, FYEMU.

(1) Retrain: As the most naive approach, it just retrains the model on the remaining dataset $D \backslash D _ { f }$

(2) SISA: SISA [1] is a well-known framework that partitions the data to reduce the retraining time by just retraining a single model on the shard that needs to be forgotten.

(3) DeltaGrad: DeltaGrad [37] unlearns the data by differentiating the optimization path with the Quasi-Newton method based on information cached during the training phase.

(4) FYEMU: FYEMU [29] is a unlearning algorithm for Neural Networks, which first unlearns data by adding noise to the model parameters, and then obtains a new model through fine-tuning.

Evaluation: We evaluate the efficiency and effectiveness of unlearning algorithms using unlearning time and the accuracy of the unlearned classifier on test data (20%), which is averaged over 10 runs. In each run, we unlearn one randomly selected instance<sup>4</sup>.

## 5.1 Unlearn linear classifier: Logistic Regression

To begin, we assess the performance of our L2UL approach using a linear classifier known as Logistic Regression (LR). We have demonstrated that the generalization error of the unlearned classifier, denoted as ${ \cal U } ( D , D _ { f } , A ( D ) )$ , is bounded. Additionally, here we find that L2UL is both efficient and effective when applied to LR.

The results of unlearning one instance in terms of accuracy and unlearning time are shown in Figure 2. L2UL achieves comparable accuracy to Retrain and SISA and outperforms DeltaGrad over all seven datasets. Compared with Retain, SISA, and DeltaGrad, L2UL speeds up 630, 149, and 282 times on the Magic dataset, and 422500, 49715, and 51500 times on the HIGGS dataset, respectively.

## 5.2 Unlearn non-linear classifier: Multilayer Perceptron

Nonlinear classifiers can learn complex nonlinear boundaries, which require a more complex process to learn. Besides, a nonlinear classifier usually has more parameters, which makes its parameter space very large. Implementing unlearning in this large space is a more challenging problem.

In this subsection, we test the performance of L2UL on Multilayer Perceptron (MLP) [30]. We use cross-entropy as the loss of the MLP, which has 1 hidden layer with 10 neurons.

The results of unlearning one instance are shown in Figure 3. L2UL achieves comparable accuracy to Retrain and SISA except HIGGS (we will discuss this in Section 5.4). L2UL achieves close accuracy to Deltagrad on Sepsis and Skin and outperforms Delta-Grad on five other datasets. L2UL outperforms FYEMU on the two smallest datasets (Magic and Adult) and achieves close accuracy to FYEMU on five other datasets.

Compared with Retain, SISA, DeltaGrad, and FYEMU, L2UL speeds up 15087, 1203, 730, and 43 times, respectively, on the Magic dataset, and 398000, 105250, 294750, and 32750 times on the HIGGS dataset. The efficiency of L2UL comes from the simple structure of $U ,$ which ensures that a single run of � is very quick. Moreover, once � is learned, the parameters of � do not need to change with high probability when a new $D _ { f }$ comes. We will discuss this in Section 8. L2UL just directly uses the learned � to unlearn the new $D _ { f } .$ , while the other algorithms require the costly computation again to unlearn $D _ { f }$ . This difference will be more significant when multiple unlearning requests come in sequence.

The results of unlearning 100 and 1000 instances in terms of accuracy and unlearning time are shown in Figure 4 and Figure 5.

## 5.3 Unlearning Efficacy

In addition to effectiveness and efficiency, the machine unlearning algorithm must also be evaluated to determine whether it has forgotten users’ information. We first show the two very commonly used evaluation metrics. Membership Inference Attack [7, 16] (as used in Table 5) and accuracy on unlearned data. The results in terms of Membership Inference Attack and accuracy on $D _ { f } ( \vert D _ { f } \vert = 1 0 0 0$ for MLP) in Table 3 show that L2UL has forgotten the information of $D _ { f }$ from $A ( D ) ^ { 5 }$ . The � has learned the unlearning behaviors.

However, it is not the case that the lower these two scores are, the better the model performance is, because this may give rise to other issues, such as information exposure [11] <sup>6</sup>. Therefore, we use an example to demonstrate the ability of the proposed algorithm L2UL for calibrating contaminated models, which also shows that L2UL has forgotten $D _ { f }$

As previously stated, training data can sometimes be contaminated, which negatively impacts the model’s performance. Machine Unlearning is an effective method for cleaning the model when dirty data is detected. A simple example is shown in Figure 6(a). The artificial dataset has two classes, which can be classified by a linear model. But when the training set is contaminated (Figure 6(b) right shows an example with 50 contaminated points), the accuracy of the model will drop.

Figure 6(c) shows the results of LR in terms of Accuracy. When data is not contaminated, the accuracy is 1, as expected. As the number of dirty points grows, the accuracy declines to approximately 0.5. When we use L2UL to clean the contaminated model by treating the dirty points as $D _ { f } ,$ the accuracy of the unlearning model remains 1.00 as the number of dirty points grows. Similar results are shown in Figure 6(d) when we replace LR with MLP. The difference is that the accuracy and F1 of MLP decrease more slowly than LR. The accuracy and F1 of the unlearned model are always 1.00. This shows that L2UL has indeed achieved the unlearning of dirty data.

![](images/e1d447a6baaa2a384b7cf8233480f4d366ffb7f59511942a2c027deb0a77ee65.jpg)

![](images/e19045b54a6f3a27a655f5856be0784233b4b377d59b514290b009c9e69c39e6.jpg)  
Figure 2: Results of unlearning Logistic Regression.

![](images/c52d07fbfa3456ccc741c698e00f217848847b19ab968f6f1e84a792e6dc3a43.jpg)

![](images/20d9bb8502039d6e2181cabeefd4b5af45f5b36412ad3df84b2051c4044aa3ec.jpg)  
Figure 3: Results of unlearning MLP $( | D _ { f } | = 1 )$

![](images/ba82b2622efaa65f05040d2759b06c9115971fc9b03da0392b94b15396b49d46.jpg)

![](images/83e738a6622db325876662929e296ea443f1eb7f3f42a13881d5a741afbae5e2.jpg)  
Figure 4: Results of unlearning MLP $( | D _ { f } | = 1 0 0 )$

Table 3: Results in terms of Membership Inference Attack and accuracy on unlearn data.
<table><tr><td></td><td colspan="3">MIA</td><td colspan="4">Accuracy on  $D _ { f }$ </td></tr><tr><td>Datasets</td><td>Retrain</td><td>FYEMU</td><td>L2UL</td><td>Retrain</td><td>SISA</td><td>FYEMU</td><td>L2UL</td></tr><tr><td>Magic</td><td>0.79</td><td>0.65</td><td>0.75</td><td>0.85</td><td>0.79</td><td>0.71</td><td>0.78</td></tr><tr><td>Adult</td><td>0.80</td><td>0.68</td><td>0.74</td><td>0.81</td><td>0.78</td><td>0.75</td><td>0.76</td></tr><tr><td>Skin</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>Covetype</td><td>0.73</td><td>0.68</td><td>0.67</td><td>0.82</td><td>0.76</td><td>0.76</td><td>0.71</td></tr><tr><td>SUSY</td><td>0.72</td><td>0.69</td><td>0.69</td><td>0.73</td><td>0.71</td><td>0.70</td><td>0.71</td></tr><tr><td>HIGGS</td><td>0.65</td><td>0.68*</td><td>0.50</td><td>0.74</td><td>0.77*</td><td>0.69</td><td>0.63</td></tr></table>

∗ indicates that the results show that complete forgetting of the data was not achieved.

![](images/43293d69ca2e56782bd6819faa927feb55ec3f64f16dd72fe9c39cc353e49305.jpg)

![](images/b6024c60a9a6260960bbb580e10bde3ad4e8f241651d1720240eeab30b3eeb85.jpg)  
Figure 5: Results of unlearning MLP $( | D _ { f } | = 1 0 0 0 )$

![](images/0989c2f735f882568d16b90bd12679b861aded8b8fc0023cdbf135d65e6453ca.jpg)  
(a) Clean data

![](images/85b47dde0193706b4fd16a1daf45d33759ab99d55b6dc402184018dccf6a1dee.jpg)  
(b) Contaminated data

![](images/1b4f8df37e0c87172a5cc1b584a50bef0644c89248c5b3cdc108f0c627e6f555.jpg)  
(c) Accuracy of LR

![](images/aad5e19c73b5ae592920c665ab9563bb6103b436d888775871a25058cb8918fe.jpg)  
(d) Accuracy of MLP  
Figure 6: Artificial dataset with clean data and contaminated data, and the results of LR and MLP.

## 5.4 Parameter Sensitivity Analysis

There are four hyperparameters in the Preprocessing and Unlearning phase of $\mathbf { L } 2 \mathbf { U L } \colon \psi , t$ for Isolation Kernel and �, � for generating $\mathcal { X } \times \mathcal { X }$ . In our experiment, the default setting of hyperparameters is: $\psi = 4 , t = 1 0 0 , m = 1 0 0 0 , s = 1 0 0 0 \mathrm { f o r L R , a n d } s = 3 0 0 0$ for MLP.

We report the accuracy of unlearning 1, 100, and 1,000 instances from LR and MLP on the largest datasets SUSY and HIGGS, with � ∈ [2, 4, 8, 16, 32, 64, 128, 256][32]. [32] shows that the Isola tion Distritbuion Kernel is not sensitive to � , hence, the sensitivity analysis of � is omitted here.

The results of $\psi$ are shown in the first row of Figure 7 and Figure 8, L2UL is robust to � when $\psi \geq 4 .$ . The results of � are shown in the second row of Figure 7 and Figure 8. For LR on SUSY, HIGGS, and MLP on SUSY, L2UL is robust to �, and the accuracy of L2UL is close to that of retrain. For MLP on HIGGS, as � increases, the accuracy of L2UL increases from 0.60 to 0.65. This is because when � (sample size) is small, we cannot get a good model �(�) and $A ( \mathfrak { D } \backslash \mathfrak { D } _ { f } )$ as input into L2UL. When sample size � increases, we can have a better model $A ( { \mathfrak { D } } )$ and $A ( \mathfrak { D } \backslash \mathfrak { D } _ { f } )$ , so that a good � and unlearned model ${ \cal U } ( D , D _ { f } , A ( D ) )$ with higher accuracy can be obtained on HIGGS.

The results of � are shown in the third row of Figure 7 and Figure 8. L2UL is robust to �, as $N _ { U }$ is a simple model that can achieve good performance with a small dataset ${ \mathcal { D } } .$

In a nutshell, L2UL is robust to the parameters $\psi$ and �. Compared to the parameters of the L2UL model and the parameters of

Table 4: Factor of time complexity (�-or: �-oriented, $D _ { f } { \bf - 0 r } \colon$ $D _ { f }$ -oriented, �(�)-or: �(�)-oriented).
<table><tr><td>U</td><td> $D { \mathrm { - } } \mathrm { o r }$ </td><td> $D _ { f ^ { - } } \mathrm { o r }$ </td><td> $A ( D ) { \mathsf { - o r } }$ </td><td>L2UL</td></tr><tr><td>Factor</td><td>A&amp;D</td><td> $D _ { f }$ </td><td> $D _ { f } \& \theta$ </td><td> $D _ { f }$ </td></tr></table>

IDK, L2UL requires stronger data as input. L2UL learns an unlearning function �, only if the input data $\mathcal { D }$ contains sufficient unlearning information <sup>7</sup>, can a correct and good unlearning function be learned.

## 6 Complexity Analysis

We show the factor that affects the time complexity of unlearning algorithms as follows (time complexity of the machine learning model � is denoted as T(�)) :

(1) �-oriented unlearning algorithm, such as SISA [1], has a time complexity of $\mathcal { T } ( A ( | D | / k ) )$ , where � is the number of shards. Because SISA retrains � on some shards, the time complexity is limited by � and |�|.

(2) $D _ { f }$ -oriented unlearning algorithm, such as [12], considers the influence of $D _ { f }$ , whose time complexity is ${ \cal O } ( | D _ { f } | d ^ { 3 } )$ , which is limited by $D _ { f }$

(3) $A ( D )$ -oriented unlearning algorithm, such as [25], considers how to update parameters. Its time complexity is $O ( | D _ { f } | d ^ { 2 } +$ $d ^ { \vert \theta \vert } )$ ) and is limited by $D _ { f }$ and |�|.

![](images/733a6450e9fc1f5b8e3862d5bd4d10154f364b20c03b9cae691e40bba27510ed.jpg)

![](images/8329f1689582c9443abbe12f546e734fd114d99549e014d8cabb9fd41105a5b1.jpg)

![](images/44d87c7da957c1e750f7e1af9e983b41304067f5161ddcc02ac1090037ab3e13.jpg)

![](images/b45ad5156fabffe72f9fc45599a0d4ea9c81f53add8fc0f2f6cfe822e4034afc.jpg)

![](images/f3f6fd93ec4bcfbcc2bcec483fc09dd8fb60007bba1d5e1813f4d8d678c55e6a.jpg)

![](images/0cc48b93d872daf5ec564d8f3b024831d75e5e8608903fd43f7f028f1a6cb4c7.jpg)  
Figure 7: Parameter sensitivity analysis of � (top), � (middle), � (bottom) on SUSY (left) and HIGGS (right) for LR.

![](images/c1b9b69a9d8d854087ef4501be9c7d4f25b9c939e1885d3368661ccecb387bd7.jpg)

![](images/701f2ed48afec695c7048f4b1e5a4477af6cf6b025a0a019df02c7c9201fcd16.jpg)

![](images/5bb85c110b6adac72caf58188e075e897e814cba15a09ffb703e742ba107aeea.jpg)

![](images/a068bcd8ebb4185ea08eb53ad95cfbe0ac0f8e074e5d1187f51b60f96f14d780.jpg)

![](images/4f85a2f26b015df16986f3c6103be1312e7bab11d4806227ca25ab7698a54566.jpg)

![](images/79bc526c54815b827e61ddd13357d297abfe0a18a5ca72ceb6b7cda95247e0d2.jpg)  
Figure 8: Parameter sensitivity analysis of �(top), �(middle), �(bottom) on SUSY (left) and HIGGS (right) for MLP.

The factors of time complexity are summarized in Table 4. Intuitively, $D _ { f }$ is unavoidable because we at least need to know the samples we need to forget. L2UL is also subject to this restriction. Given dataset �, we give the time complexity and space complexity of L2UL as follows:

\* Time complexity of training � is $\mathcal { T } ( U ( m ) )$ .

\* Time complexity of preprocessing is $O ( \psi t | D | d + m \mathcal { T } ( A ( s ) ) )$

\* Time complexity of unlearning $D _ { f }$ from $U \operatorname { i s } O ( \psi t | D _ { f } | d )$

\* Space complexity of � is $O ( \psi t d )$

When unlearning $D _ { f } , { \mathrm { L 2 U I } }$ needs to calculate the feature map of $D _ { f }$ first. But we can store the feature map of every point from � in advance. In this way, when unlearning $D _ { f } ,$ we only need to select the corresponding feature maps and take the mean. Although the time complexity still depends on $D _ { f } ,$ , the operation of just taking mean is very fast. As a trade-off, space complexity goes up to $O ( \psi t | D | )$ .

Table 5: Results of unlearning ResNet-18 on CIFAR-10 dataset.
<table><tr><td></td><td>10</td><td>100</td><td>1000</td><td>3000</td></tr><tr><td rowspan="3">ToW(↑)</td><td>Retrain 1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>ADV+IMP</td><td>0.32 0.13</td><td>0.01</td><td>0.01</td></tr><tr><td>RUM</td><td>0.60 0.80</td><td>0.83</td><td>0.83</td></tr><tr><td></td><td>L2UL(Ours) Retrain</td><td>0.80 1.00</td><td>0.88</td><td>0.99</td></tr><tr><td rowspan="2">MIA gap(↓)</td><td>ADV+IMP</td><td>0.00 0.30</td><td>0.00 0.00 0.17</td><td>0.00 0.14</td></tr><tr><td>RUM</td><td>0.13 0.30 0.13</td><td>0.15</td><td>0.12</td></tr><tr><td></td><td>L2UL(Ours)</td><td>0.30 0.13</td><td>0.17</td><td>0.14</td></tr><tr><td rowspan="3">Time(↓)</td><td>Origin (A(D) Retrain</td><td>1044.9 1020.5</td><td>874.4</td><td></td></tr><tr><td>ADV+IMP</td><td>28.4 78.7</td><td>783.2 1665.5</td><td>702.7 12088.2</td></tr><tr><td>RUM</td><td>13.6 13.6</td><td>13.2</td><td>12.8</td></tr><tr><td></td><td>L2UL(Ours)</td><td>0.5</td><td>0.5 0.5</td><td>0.6</td></tr></table>

## 7 Unlearn large-scale parameters: ResNet

This paper and the method L2UL we propose primarily focus on the unlearning of simple models (with fewer parameters) trained on large amounts of training data. As demonstrated in the previous sections, the experimental results confirm that L2UL not only maintains high fidelity to the retrained model but also achieves significant gains in computational efficiency. Furthermore, in this section, we verify the effectiveness of our proposed L2UL on a model with a larger parameter size. We tested the performance of the unlearning ResNet-18 model on the CIFAR-10 dataset and compared it with:

ADV+IMP: [5] Misclassify each instance outside of its original prediction, or relabel the instance to a different label. At the same time, use adversarial examples to overcome forgetting at the representation level and use weight importance indicators to accurately locate network parameters that propagate unnecessary information to reduce the time required for forgetting.

RUM: [42] The forgotten set is refined into homogenized subsets based on different features. The meta-algorithm employs the existing algorithms to forget each subset, ultimately providing a model that has forgotten the entire forgotten set.

We train ResNet-18 for 50 epochs on CIFAR-10 with a learning rate of 0.0001. And we report two evaluation metrics, ToW and MIA (Membership Inference Attack) gap, that are used in RUM [42].

The results are shown in Table 5. We highlight the following three key observations: (a) L2UL achieves the highest ToW scores when forgetting 10, 100, 1000, and 3000 samples. (b) L2UL and the comparison algorithms have a closed low MIA gap. (c) L2UL only takes a very short time to complete the unlearning.

In summary, our experimental results demonstrate not only the scalability of L2UL on large-scale data but also its scalability in terms of model parameters. L2UL maintains the performance of the model after forgetting data, and, more importantly, its unlearning efficiency is far superior to existing algorithms.

## 8 Disscussion

## 8.1 How to learn � when the model cannot be retrained?

It works well for L2UL to use Equation 2 as its loss function only when the parameter of the retrained model $A ( { \mathfrak { D } } \backslash { \mathfrak { D } } _ { f } )$ can be obtained. However, the inability to retrain the model is a common challenge faced by machine unlearning. For example, we can’t retrain the model on streaming data. In this case, L2UL can no longer be used to generate training data D, and the loss function 2 cannot be used to train $N _ { U }$ . How to find a new loss function when the model cannot be retrained is a challenge.

## 8.2 Retrain � after unlearning �

It is worth emphasizing that when we employ � to unlearn model, user’s information $D _ { f }$ will be included in �. Therefore, we must unlearn � after we use � to unlearn �.

Theorem 8.1. When sampling � subsets �from �. The probability that � is sampled at least � times is

$$
\mathcal { P } ( k \geq k _ { 0 } ) = \sum _ { i = k _ { 0 } } ^ { m } C _ { m } ^ { k } ( \frac { s } { | D | } ) ^ { i } ( 1 - \frac { s } { | D | } ) ^ { m - i } .
$$

In our experiments, $\mathcal { P } ( k \geq 2 ) < 0$ .004 on HIGGS, which means the probability that the data $\{ x , y \} \in D$ is used to train � is very low. So we barely need to retrain �. Even if we need, the time cost is affordable, as it only takes 5.2 seconds to retrain � on HIGGS.

## 8.3 Subdivide the machine unlearning task

Current machine unlearning methods often treat forgetting tasks as a monolithic problem, neglecting the inherent diversity in model complexity and data scale. We argue for a more nuanced, taskspecific approach. Since a forgetting task is defined by both the model and the training data, we propose a taxonomy that categorizes machine unlearning into four distinct scenarios based on model size and dataset size:

1) Simple models on small datasets. This represents the most straightforward scenario, where retraining from scratch remains computationally feasible and is often the preferred solution.

2) Complex models on small datasets. Although the training data is limited, the increased model complexity significantly raises the cost of retraining, making it an impractical choice.

3) Simple models on large datasets. This is the focal scenario of our work. Because the model itself is lightweight, applying sophisticated unlearning techniques designed for complex architectures would be unnecessarily resource-intensive—essentially overkill. Instead, a lightweight method, such as the one we propose, is sufficient to achieve effective forgetting.

4) Complex models on large datasets (e.g., large language models). This constitutes the most challenging scenario, demanding advanced and scalable unlearning strategies.

The key advantage of this taxonomy lies in its strategic flexibility: by matching the unlearning method to the specific complexity of the task, we can achieve efficient data removal without incurring prohibitive computational costs, ensuring we neither "use a sledgehammer to crack a nut" nor deploy inadequate solutions for demanding scenarios.

## 9 Conclusion

In this paper, we propose Learning-to-UnLearn (L2UL), to the best of our knowledge, the first framework to address machine unlearning by learning the unlearning behaviors rather than manually designing complex functions. By prioritizing a distribution-oriented perspective, L2UL achieves a remarkably streamlined yet potent unlearning capability. Focusing on scenarios where the model is simple but the data training dataset is large, we utilized Logistic Regression and Multilayer Perceptrons to demonstrate the ability of L2UL to handle massive datasets with exceptional speed and no loss in accuracy. Both theoretical and empirical results prove that L2UL significantly outperforms existing methods in efficiency. Furthermore, tests on contaminated models and sensitivity analyses confirm the effectiveness and stability of L2UL across various configurations. Finally, successful validation on ResNet underscores the scalability of L2UL to more complex models, offering a robust and efficient solution for modern privacy requirements.

## A Proofs

## A.1 Proof of Theorem 4.5

Theorem 4.5

$$
\begin{array} { r } { L ( f _ { U ( A ( D ) , D , D _ { f } ) } ) \le L ( f _ { A ( D \backslash D _ { f } ) } ) + 2 R \sqrt \epsilon , } \end{array}
$$

where � is the radius of the dataset $X ( \forall x \in X , | | x | | \leq R )$ , and � is the generalization bound (MSE) of the unlearning model �.

PROOF. Denote $U ( A ( D ) , D , D _ { f } ) \mathop { - A } ( D \backslash D _ { f } )$ as �, then $| | e | | \leq { \sqrt { \epsilon } }$ By Taylor Expansion, we approximately have

$$
\begin{array} { r l } & { \quad L ( f _ { U ( A ( D ) , D , D _ { f } ) } ) - L ( f _ { A ( D \backslash D _ { f } ) } ) } \\ & { \quad = \frac { \partial L ( f _ { A ( D \backslash D _ { f } ) } ) } { \partial A ( D \backslash D _ { f } ) } \sp \top e + o ( \| e \| ) , } \end{array}
$$

where $\begin{array} { r } { \operatorname* { l i m } _ { \| e \|  0 } \frac { o ( \| e \| ) } { \| e \| } = 0 } \end{array}$ . Then by Cauchy–Schwarz inequality and convexity of norm function, we have

$$
\begin{array} { r l } & { \quad L \big ( f _ { U ( A ( D ) , D , D _ { f } ) } \big ) - L \big ( f _ { A ( D \backslash D _ { f } ) } \big ) } \\ & { = E _ { x , y } \big [ \big ( f _ { A ( D \backslash D _ { f } ) } ( x ) - y \big ) x ^ { \top } \big ] e + o ( \| e \| ) } \\ & { { \le } | | E _ { x , y } \big [ \big ( f _ { A ( D \backslash D _ { f } ) } ( x ) - y \big ) x \big ] | | \cdot | | e | | + o ( \| e \| ) } \\ & { { \le } E _ { x , y } \big [ | | \big ( f _ { A ( D \backslash D _ { f } ) } ( x ) - y \big ) x | | \big ] \cdot | | e | | + o ( \| e \| ) } \\ & { { \le } 2 R \sqrt { \epsilon } + o ( \sqrt { \epsilon } ) . } \end{array}
$$

Here we refer $U ( A ( D ) , D , D _ { f } )$ and $A ( D \setminus D _ { f } )$ as the parameters of the unlearned model and retrained model, respectively.

## A.2 Proof of Theorem 4.6

Theorem 4.6

$$
\begin{array} { r } { L ( f _ { U ( A ( D ) , D , D _ { f } ) } ) \le L ( f _ { A ( D \backslash D _ { f } ) } ) + A \sqrt { \epsilon } , } \end{array}
$$

where � is a finite scalar determined by the MLP structure and � is the generalization bound (MSE) of the unlearning model �.

PROOF. For the (� + 1)-layer MLP model (the 0-th layer is the input layer, the (� + 1)-th layer is the output layer), we use softmax activation for the output layer, and sigmoid activation function for the rest of the layers. We denote the number of neurons in the i-th layer as $n _ { i } ,$ then the weight between the $( i - 1 )$ -th layer and �-th layer are $W _ { i }$ of shape $\left( n _ { k } , n _ { k - 1 } \right)$ and $B _ { i }$ of shape $( n _ { i } , 1 )$ . For a finite test set $X , \left| X \right| = m$ , we denote it as $A _ { 0 }$ . Then, the forward propagation scheme is described as follows:

$$
Z _ { k } = W _ { k } A _ { k } + B _ { k } , A _ { k } = f _ { k } ( Z _ { k } )
$$

for $k \in \{ 1 , 2 , . . . , N + 1 \}$ , where $f _ { k } ( \cdot )$ is the activation function we use in the k-th layer (softmax for the output layer, sigmoid for the rest layers).

For the back propagation, when using Cross-Entropy Loss, we have

$$
d Z _ { N + 1 } = A _ { N + 1 } - Y , d W _ { N + 1 } = \frac { 1 } { m } d Z _ { N + 1 } A _ { N } ^ { \top } ,
$$

$$
d B _ { N + 1 } = \frac { 1 } { m } d Z _ { N + 1 } 1 _ { ( m , 1 ) } ,
$$

and

$$
d Z _ { k } = W _ { k + 1 } ^ { T } d Z _ { k + 1 } \odot f _ { k } ^ { \prime } ( Z _ { k } ) ,\tag{3}
$$

$$
d ~ W _ { k } = { \frac { 1 } { m } } d ~ Z _ { k } A _ { k - 1 } ^ { \top } , ~ d ~ B _ { k } = { \frac { 1 } { m } } d ~ Z _ { k } 1 _ { ( m , 1 ) }
$$

for $k \in \{ 1 , 2 , . . . , N , N + 1 \}$ , where � means the derivative of the Cross Entropy Loss, $1 _ { ( m , 1 ) }$ is a column vector of size m, with each entry $= 1 .$ , and ⊙ is the element-wise multiplication.

Denote $A ( D \setminus D _ { f } )$ as Θ and $U ( A ( D ) , D , D _ { f } )$ as ${ \hat { \Theta } } .$ . We have

$$
\Theta = { \binom { W } { B } } ,\tag{4}
$$

where �, � are the concatenation of each layer’s row-flatten parameter, i.e. $W = [ \mathrm { r f t } ( W _ { 1 } ) , . . . , \mathrm { r f t } ( W _ { N + 1 } ) ] ^ { \top } , W = [ \mathrm { r f t } ( B _ { 1 } ) , . . . , \mathrm { r f t } ( B _ { N + 1 } ) ] ^ { \top }$ where rtf(·) is the operator that flatten a matrix into a row vector. The same notation goes for Θˆ . As in 4.5, we only need to show that � Θ is bounded, which can be achieved from the bound for � � and $d ; W ,$ , since $\| d \Theta \| ^ { 2 } = \| d W \| ^ { 2 } + \| d B \| ^ { 2 }$

For $W _ { N + 1 } ,$ we have

$$
\Vert d \operatorname { r f t } ( W _ { N + 1 } ) \Vert ^ { 2 } = \Vert \frac { ( A _ { N + 1 } - Y ) A _ { N } ^ { \top } } { m } \Vert _ { F } ^ { 2 } \leq 4 n _ { N } n _ { N + 1 } .
$$

since the absolute value of each entry in $A _ { N + 1 } , A _ { N } , Y$ is smaller than 1. For the bound on � rf $\mathrm { t } W _ { k } ( k \neq N + 1 )$ , we first need to get the bound $R _ { k }$ for � rf $Z _ { k }$ . We know that

$$
\| d \mathrm { \ r f t } ( Z _ { N + 1 } ) \| _ { \infty } \leq R _ { N + 1 } = 2 .
$$

Then according to Equation 3, we have

$$
\| d \mathrm { \ r f t { } Z _ { \boldsymbol { k } } \| _ { \infty } } \le \frac { R } { 4 } n _ { k + 1 } R _ { k + 1 } \triangleq R _ { k } ,
$$

under the assumption $\| W \| _ { \infty } \leq R .$ . The iterative form of $R _ { k }$ can be expressed equivalently as

$$
R _ { k } = ( \frac { R } { 4 } ) ^ { N + 1 - k } \prod _ { j = k + 1 } ^ { N + 1 } n _ { j } .
$$

Coming back to $W _ { k }$ , for $k \leq N$ , we have

$$
\| d \operatorname { r f t } ( W _ { k } ) \| ^ { 2 } \leq R _ { k } ^ { 2 } n _ { k } n _ { k - 1 } .
$$

Learning to Unlearn: Machine Unlearning via Learning the Unlearning Behaviors

Finally, we have

$$
\begin{array} { l } { \displaystyle \| d \mathrm { \ r f t { \boldsymbol W } } \| ^ { 2 } = \sum _ { k = 1 } ^ { N + 1 } \| d \mathrm { \ r f t { \boldsymbol W } } _ { k } \| ^ { 2 } } \\ { \le \underbrace { 4 n _ { N } n _ { N + 1 } + \sum _ { k = 1 } ^ { N } ( \frac { R ^ { 2 } } { 1 6 } ) ^ { N + 1 - k } } _ { \triangleq A _ { W } } \frac { \prod _ { j = k - 1 } ^ { N + 1 } n _ { j } ^ { 2 } } { n _ { k } n _ { k + 1 } } . } \end{array}
$$

Similarly, we have

$$
\| d \operatorname { r f t } ( W _ { k } ) \| ^ { 2 } \leq R _ { k } ^ { 2 } n _ { k } n _ { k - 1 } .\tag{5}
$$

Finally, we have

$$
\begin{array} { l } { \displaystyle \| d \mathrm { \ r f t } { B } \| ^ { 2 } = \sum _ { k = 1 } ^ { N + 1 } \| d \mathrm { \ r f t } { B } _ { k } \| ^ { 2 } } \\ { \displaystyle \le \underbrace { 4 n _ { N + 1 } + \sum _ { k = 1 } ^ { N } ( \frac { R ^ { 2 } } { 1 6 } ) ^ { N + 1 - k } \frac { \prod _ { j = k } ^ { N + 1 } n _ { j } ^ { 2 } } { n _ { k } } } _ { \displaystyle \triangleq A _ { B } } . } \end{array}\tag{6}
$$

Then ∥� $\begin{array} { r } { \Theta \| \le \sqrt { A _ { w } ^ { 2 } + A _ { B } ^ { 2 } } \triangleq A . } \end{array}$

□

## A.3 Proof of Complexity

\* Time complexity of preprocessing is $O ( \psi t | D | d + m \mathcal { T } ( A ( s ) ) )$ PROOF. Preprocessing first requires a total time of $O ( \psi t | D | d$ to produce the feature map (line 2 in Algorithm 1), and requires $O ( \psi t s + \mathcal { T } ( A ( s ) )$ for each loop, So the time complexity of Algorithm 1 is $O ( \psi t | D | d + m \mathcal { T } ( A ( s ) ) )$ ). □

\* Time complexity of training � is $\mathcal { T } ( U ( m ) )$ .

PROOF. The time complexity of Train U is using the L2UL to train $N _ { U }$ on the � data points, so the time complexity is $\mathcal { T } ( U ( m ) )$ . □

\* Time complexity of unlearning $D _ { f }$ from � is $O ( \psi t | D _ { f } | d )$ PROOF. Algorithm $^ 2$ requires time of $O ( \psi t | D _ { f } | d )$ to get the feature map (line 1). and $O ( \psi t | D _ { f } | )$ to get $\mu _ { f }$ (line $^ { 2 ) }$ Therefore, the time complexity to unlearn $D _ { f }$ from the � is $O ( \psi t | D _ { f } | d )$ □

\* Space complexity of � is $O ( \psi t d )$

PROOF. we need $O ( \psi t )$ to store the $\mu _ { D }$ , and $O ( \psi t d )$ to store the seeds for partition in isolation kernel. Therefore, the space complexity of � is $O ( \psi t d )$ . □

## A.4 Proof of Theorem 7.1

Theorem 7.1

We sample � subsets � from �. The probability that sample � is sampled at least $k _ { 0 }$ times is:

$$
\mathcal { P } ( k \geq k _ { 0 } ) = \sum _ { i = k _ { 0 } } ^ { m } C _ { m } ^ { k } ( \frac { s } { | D | } ) ^ { i } ( 1 - \frac { s } { | D | } ) ^ { m - i } .
$$

PROOF. Sampling � points from |�| points, The probability that point � is sampled is $\begin{array} { r } { \mathcal { P } ( x ) = \frac { s } { | D | } } \end{array}$

Do this sampling � times, the probability that point � is sampled � times is:

$$
P ( k ) = C _ { m } ^ { k } ( \frac { s } { | D | } ) ^ { k } ( 1 - \frac { s } { | D | } ) ^ { m - k } .
$$

So the probability that sample � is sampled at least $k _ { 0 }$ times is:

$$
\mathcal { P } ( k \geq k _ { 0 } ) = \sum _ { i = k _ { 0 } } ^ { m } C _ { m } ^ { k } ( \frac { s } { | D | } ) ^ { i } ( 1 - \frac { s } { | D | } ) ^ { m - i } .
$$

□

## References

[1] Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. 2021. Machine unlearning. In 2021 IEEE Symposium on Security and Privacy (SP). IEEE, 141–159.

[2] Jonathan Brophy and Daniel Lowd. 2021. Machine unlearning for random forests. In International Conference on Machine Learning. PMLR, 1092–1104.

[3] Yang Cao, Haolong Xiang, Hang Zhang, Ye Zhu, and Kai Ming Ting. 2025. Anomaly detection based on isolation mechanisms: A survey. Machine Intelligence Research 22, 5 (2025), 849–865.

[4] Yinzhi Cao and Junfeng Yang. 2015. Towards making systems forget with machine unlearning. In 2015 IEEE symposium on security and privacy. IEEE, 463–480.

[5] Sungmin Cha, Sungjun Cho, Dasol Hwang, Honglak Lee, Taesup Moon, and Moontae Lee. 2024. Learning to unlearn: Instance-wise unlearning for pre-trained classifiers. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 11186–11194.

[6] Tianlong Chen, Xiaohan Chen, Wuyang Chen, Howard Heaton, Jialin Liu, Zhangyang Wang, and Wotao Yin. 2022. Learning to optimize: A primer and a benchmark. Journal ofMachine Learning Research 23, 189 (2022), 1–59.

[7] Vikram S Chundawat, Ayush K Tarun, Murari Mandal, and Mohan Kankanhalli. 2023. Can bad teaching induce forgetting? Unlearning in deep networks using an incompetent teacher. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37. 7210–7217.

[8] Vikram S Chundawat, Ayush K Tarun, Murari Mandal, and Mohan Kankanhalli. 2023. Zero-shot machine unlearning. IEEE Transactions on Information Forensics and Security (2023).

[9] Shaopeng Fu, Fengxiang He, and Dacheng Tao. 2021. Knowledge Removal in Sampling-based Bayesian Inference. In International Conference on Learning Representations.

[10] Antonio Ginart, Melody Guan, Gregory Valiant, and James Y Zou. 2019. Making ai forget you: Data deletion in machine learning. Advances in neural information processing systems 32 (2019).

[11] Aditya Golatkar, Alessandro Achille, and Stefano Soatto. 2020. Eternal sunshine of the spotless net: Selective forgetting in deep networks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9304–9312.

[12] Chuan Guo, Tom Goldstein, Awni Hannun, and Laurens Van Der Maaten. 2020. Certified Data Removal from Machine Learning Models. In International Confer ence on Machine Learning. PMLR, 3832–3842.

[13] Varun Gupta, Christopher Jung, Seth Neel, Aaron Roth, Saeed Sharifi-Malvajerdi, and Chris Waites. 2021. Adaptive machine unlearning. Advances in Neural Information Processing Systems 34 (2021), 16319–16330.

[14] Sanjiv Kumar, Mehryar Mohri, and Ameet Talwalkar. 2012. Sampling methods for the Nyström method. The Journal ofMachine Learning Research 13, 1 (2012), 981–1006.

[15] Huawei Lin, Jun Woo Chung, Yingjie Lao, and Weijie Zhao. 2023. Machine Unlearning in Gradient Boosting Decision Trees. In Proceedings ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 1374–1383.

[16] Jiancheng Liu, Parikshit Ram, Yuguang Yao, Gaowen Liu, Yang Liu, PRANAY SHARMA, Sijia Liu, et al. 2024. Model sparsity can simplify machine unlearning. Advances in Neural Information Processing Systems 36 (2024).

[17] Zhuo Ma, Yang Liu, Ximeng Liu, Jian Liu, Jianfeng Ma, and Kui Ren. 2022. Learn to forget: Machine unlearning via neuron masking. IEEE Transactions on Dependable and Secure Computing 20, 4 (2022), 3194–3207.

[18] Zhuo Ma, Yang Liu, Ximeng Liu, Jian Liu, Jianfeng Ma, and Kui Ren. 2022. Learn to forget: Machine unlearning via neuron masking. IEEE Transactions on Dependable and Secure Computing (2022).

[19] Cameron Musco and Christopher Musco. 2017. Recursive sampling for the nystrom method. Advances in neural information processing systems 30 (2017).

[20] Seth Neel, Aaron Roth, and Saeed Sharifi-Malvajerdi. 2021. Descent-to-delete: Gradient-based methods for machine unlearning. In Algorithmic Learning Theory. PMLR, 931–962.

[21] Quoc Phong Nguyen, Ryutaro Oikawa, Dinil Mon Divakaran, Mun Choon Chan, and Bryan Kian Hsiang Low. 2022. Markov chain monte carlo-based machine

unlearning: Unlearning what needs to be forgotten. In Proceedings ofthe 2022 ACM on Asia Conference on Computer and Communications Security. 351–363.

[22] Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, and Quoc Viet Hung Nguyen. 2022. A survey of machine unlearning. arXiv preprint arXiv:2209.02299 (2022).

[23] Xiaoyu Qin, Kai Ming Ting, Ye Zhu, and Vincent Lee. 2019. Nearest-neighbourinduced isolation similarity and its impact on density-based clustering. In Proceedings ofthe 33rd AAAI Conference on AI (AAAI 2019), AAAI Press.

[24] Ali Rahimi and Benjamin Recht. 2007. Random features for large-scale kernel machines. Advances in neural information processing systems 20 (2007).

[25] Ayush Sekhari, Jayadev Acharya, Gautam Kamath, and Ananda Theertha Suresh. 2021. Remember what you want to forget: Algorithms for machine unlearning. Advances in Neural Information Processing Systems 34 (2021), 18075–18086.

[26] Alex Smola, Arthur Gretton, Le Song, and Bernhard Schölkopf. 2007. A Hilbert space embedding for distributions. In International conference on algorithmic learning theory. Springer, 13–31.

[27] Ke Tang and Xin Yao. 2024. Learn to optimize—a brief overview. National Science Review 11, 8 (2024), nwae132.

[28] Ayush Kumar Tarun, Vikram Singh Chundawat, Murari Mandal, and Mohan Kankanhalli. 2023. Deep regression unlearning. In International Conference on Machine Learning. PMLR, 33921–33939.

[29] Ayush K Tarun, Vikram S Chundawat, Murari Mandal, and Mohan Kankanhalli. 2023. Fast yet effective machine unlearning. IEEE Transactions on Neural Networks and Learning Systems (2023).

[30] Hind Taud and JF Mas. 2018. Multilayer perceptron (MLP). Geomatic approaches for modeling land change scenarios (2018), 451–455.

[31] Kai Ming Ting, Zongyou Liu, Hang Zhang, and Ye Zhu. 2022. A new distributional treatment for time series and an anomaly detection investigation. Proceedings of the VLDB Endowment 15, 11 (2022), 2321–2333.

[32] Kai Ming Ting, Bi-Cun Xu, Takashi Washio, and Zhi-Hua Zhou. 2020. Isolation Distributional Kernel: A New Tool for Kernel based Anomaly Detection. In Proceedings ofthe 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 198–206.

[33] Kai Ming Ting, Yue Zhu, and Zhi-Hua Zhou. 2018. Isolation kernel and its effect on SVM. In Proceedings ofthe 24th ACM SIGKDD International Conference on

Knowledge Discovery & Data Mining. 2329–2337.

[34] Alexander Warnecke, Lukas Pirch, Christian Wressnegger, and Konrad Rieck. 2023. Machine unlearning of features and labels. Network and Distributed System Security (NDSS) Symposium (2023).

[35] Christopher Williams and Matthias Seeger. 2000. Using the Nyström method to speed up kernel machines. Advances in neural information processing systems 13 (2000).

[36] Ga Wu, Masoud Hashemi, and Christopher Srinivasa. 2022. Puma: Performance unchanged model augmentation for training data removal. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 36. 8675–8682.

[37] Yinjun Wu, Edgar Dobriban, and Susan Davidson. 2020. Deltagrad: Rapid retraining of machine learning models. In International Conference on Machine Learning. PMLR, 10355–10366.

[38] Zhaomin Wu, Junhui Zhu, Qinbin Li, and Bingsheng He. 2023. DeltaBoost: Gradient Boosting Decision Trees with Efficient Machine Unlearning. Proceedings ofthe ACM on Management ofData 1, 2 (2023), 1–26.

[39] Tianbao Yang, Yu-Feng Li, Mehrdad Mahdavi, Rong Jin, and Zhi-Hua Zhou. 2012. Nyström method vs random fourier features: A theoretical and empirical comparison. Advances in neural information processing systems 25 (2012).

[40] Hang Zhang, Kai Ming Ting, and Ye Zhu. 2025. Kernel-bounded clustering: Achieving the objective of spectral clustering without eigendecomposition. Artificial Intelligence (2025), 104440.

[41] Hang Zhang, Kaifeng Zhang, Kai Ming Ting, and Ye Zhu. 2023. Towards a Persistence Diagram that is Robust to Noise and Varied Densities. In Proceedings of the 40th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 202), Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (Eds.). PMLR, 41952–41972.

[42] Kairan Zhao, Meghdad Kurmanji, George-Octavian Barbulescu, Eleni Triantafil-˘ lou, and Peter Triantafillou. 2024. What makes unlearning hard and what to do about it. Advances in Neural Information Processing Systems 37 (2024), 12293–12333.

Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009
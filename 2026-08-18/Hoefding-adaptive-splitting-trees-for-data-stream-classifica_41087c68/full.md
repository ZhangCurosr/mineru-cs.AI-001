# Hoefding adaptive splitting trees for data stream classification with concept drift and ensemble learning

Daniel Nowak Assis<sup>1,2</sup>, Jean Paul Barddal<sup>1</sup>, Fabr´ıcio Enembreck<sup>1</sup>

<sup>1</sup>Programa de P´os-Gradua¸c˜ao em Inform´atica (PPGIa), Pontif´ıcia Universidade Cat´olica do Paran´a (PUCPR), Rua Imaculada Concei¸c˜ao 1155, 80215-901, Curitiba, Brazil . <sup>2</sup>Sorbonne Universit´e, CNRS, LIP6, Paris, France.

Contributing authors: daniel.nassis@ppgia.pucpr.br;   
jean.barddal@ppgia.pucpr.br; fabricio@ppgia.pucpr.br;

## Abstract

Ensembles of decision trees are well-established methods for data stream classification. In ensemble learning, Hoefding Trees are widely adopted as base learners, performing periodic split attempts according to the Hoefding bound. Recent studies, however, indicate that this standard splitting mechanism lacks adaptability, while adaptive trees that trigger splits in response to performance degradation have achieved superior results. In this paper, we identify limitations in the use of adaptive-splitting decision trees as ensemble base learners, showing that change detectors often fail to promote suficient diversity within ensembles. To address this issue, we propose two novel decision tree models, termed Hoefding Adaptive Splitting Trees. These models combine the periodic splitting strategy of Hoefding Trees, which fosters ensemble diversity, with adaptive splitting mechanisms that employ change detection algorithms to identify performance decay and determine split points. Experimental results demonstrate that Hoefding Adaptive Splitting Trees enhance ensemble performance and achieve state-of-the-art results across a comprehensive evaluation, including benchmark comparisons, computational cost analysis, and concept drift adaptation.

Keywords: Data Stream Classification, Ensemble, Decision Tree, Concept Drift

## 1 Introduction

Learning systems designed for online scenarios are well-known and an established area in Machine Learning. Data stream mining contemplates knowledge discovery in highspeed arriving data. In practice, it requires online deployment due to the high amount of data generated in real time. For an eficient learning system to work on streaming data, the algorithms deployed must be aware of the hasty and potentially infinite nature of streaming data. As pointed out in [1], ideally, a system designed for mining streaming data must i) process an instance of data, analyze it one time, and discard it, ii) use a limited amount of memory, iii) process an instance as fast as possible, and iv) be able to make predictions at any time.

Another challenge in streaming data is concept drift [2]. A concept drift occurs when the probabilistic properties of the data change over time. In a classification problem, a concept drift occurs between the times t and $t + \Delta$ , if $P _ { t } ( X , Y ) \neq P _ { t + \Delta } ( X , Y )$ where $P _ { t }$ refers to the joint distribution at time t given a set of examples X and their class labels Y. Concept drifts may decrease the accuracy of machine learning models, which must swiftly detect and adapt to those situations to avoid compromising the entire learning process.

Ensemble-based methods are considered state-of-the-art algorithms for the data stream classification problem. In the construction and evaluation of ensembles in the literature, the first choice as a starting point was, in general, the Hoefding tree [3]. The Hoefding Tree algorithm is an incremental method that gradually expands the tree by making periodic split attempts as the data streams evolve, relying on the principles outlined in the Hoefding Theorem [4]. It has long been considered a state-of-theart approach for constructing decision trees in the context of mining streaming data. However, recent insights from [5] highlight a limitation: the periodic split attempts may not efectively adapt to the evolving nature of the data stream. Since these split attempts occur at regular intervals, they fail to capture the precise moments when changes in accuracy or data distribution occur within the leaves. Furthermore, the algorithm continues to search for the best split even during periods of stability, causing increased processing time without significant improvements to the tree structure.

Local Adaptive Streaming Trees (LAST), introduced in [5], overcome this limitation by continuously monitoring tree performance or class-distribution purity and splitting whenever change detectors flag a change, thereby outperforming Hoefdingbased trees <sup>1</sup>. Despite these promising results, LAST lacks the strategies needed to serve as a base model for ensemble learners. In practice, it has been observed that i) change detectors operate per base model, so instances afect individual learners rather than the ensemble as a whole; ii) the trees are learned without any diversity induction scheme; and iii) updating change detectors repeatedly incurs high computational overhead.

Overall, it has been observed that i) the high correlation among decision tree outputs in the initial moments of the stream afects change detection across base learners, inducing similar splitting times and low diversity; and ii) splitting time can be afected by random sampling, and LAST’s soft split constraint can produce poor splits, unlike the monolithic version, which reacts to changes in the model’s true performance or distribution. To address these challenges, we propose two new decision trees, namely Hoefding Adaptive Splitting Trees (HASTs). These models combine a periodic splitting mechanism, which induces diversity among the decision tree learners by producing diferent trees, with an adaptive splitting mechanism that tracks either leaf-node performance or class-distribution purity and can react to changes at the leaf-node level. The two approaches difer in their combination strategy, drawing, respectively, on Hoefding Trees [3] and Extremely Fast Decision Trees [7].

The contributions of this paper are summarized as follows:

1. Two novel decision tree architectures that demonstrate state-of-the-art results in ensemble setups.

2. Comprehensive experimental analysis with state-of-the-art ensembles and insights into their performance, computational cost, and how they react to concept drifts.

3. Indication of future directions for the proposal of new ensembles.

The remainder of this paper is organized as follows. Section 2 details the fundamental concepts of data stream mining and concept drift. Section 3 discusses the algorithms used in this work, such as online decision trees and ensembles. Section 4 revisits the limitations of LAST as base learners of ensembles and introduces the proposed decision trees. Section 5 outlines the experimental protocol used in this work. Section 6 discusses the results obtained. In Section 7, we conclude the paper by presenting the main findings of this study and identifying future research directions.

## 2 Background and Definitions

Data streams are continuous and sequential data flows that arrive over time, with no predefined stopping point, often treated as potentially infinite. In a structured data stream, each arriving datum is a feature vector $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { d }$ . Classification (a prediction of a class $y _ { n }$ given an input ${ \vec { x } } _ { n } )$ in this framework fundamentally difers from batch classification because in streaming data, storing all data is unfeasible, and thus, eficient stream mining requires classifiers that can ideally process data instances as rapidly as they arrive. If processing lags, the only options are to ignore new data (rendering the classifier less current) or attempt to store it, risking memory exhaustion and system collapse.

Another problem in data streams is concept drift, characterized by temporal variations in the data’s probability distribution. Specifically, drift occurs between time t and $t + \Delta$ if the joint distribution $P ( \vec { x } , y )$ changes $( P _ { t } ( \vec { x } , y ) \neq P _ { t + \Delta } ( \vec { x } , y ) )$ . Based on the relationship $P ( \vec { x } , y ) = P ( \vec { x } ) P ( y | \vec { x } )$ , three types of drift are distinguished [8]:

1. Virtual drift: Changes solely in the marginal distribution of features, $P ( \vec { x } )$

2. Real drift: Changes solely in the posterior probability distribution (the classification boundary), $P ( y | \vec { x } )$

3. Changes afecting both $P ( \vec { x } )$ and $P ( y | \vec { x } )$ simultaneously [2].

In addition to the categories above, concept drifts can be categorized according to the temporal patterns exhibited. An abrupt shift is a sudden change from concept $C _ { 1 }$ to $C _ { 2 }$ . A gradual shift is a slower transition between concepts. An incremental shift occurs through a series of intermediate concepts. Finally, a recurring concept scenario allows a previously observed concept to reappear.

For the interested reader in the concept drift problem, we recommend the following recent studies that broaden the topic [2, 9].

## 3 Online Decision Trees and Ensembles

In this section, we introduce existing works on online decision trees and ensembles that use tree variants for data stream classification.

## 3.1 Hoefding Trees

Hoefding trees [3] are incremental and online decision trees that adopt a representation similar to their batch counterparts, such as C4.5 [10] and CART [11] algorithms. In contrast to standard decision tree learning, which performs a greedy evaluation of linear splits in a data batch, Hoefding Trees continuously increment the statistics required to check whether a split should be performed, and this analysis is carried out periodically.

Targeting eficiency, the two main components of Hoefding Trees are the Hoefding bound constraint to prevent overgrowth and the periodic evaluation of splits in leaf nodes. Figure 1 presents the incremental training and splitting mechanism of Hoefding Trees, where the Grace Period (GP) is a user-given parameter that sets the frequency at which a leaf node will attempt to split. Denoting n as the number of samples seen at leaf l, if n mod $G P = 0 { \mathrm { : } }$ , a split attempt will ensue. Then it must pass the Hoefding Test to result in a split.

![](images/9977222e9145f988da380b2ca853ddb869136ada14443bb37823095004b2a332.jpg)  
Fig. 1: Training process of Hoefding Trees

The Hoefding bound [4] is a theorem that presents a probability inequality for the diference between the mean (X) and expected value $( \mathbb { E } [ X ] )$ of a set of random variables. With a level of confidence $\delta ,$ one can derive that $\dot { \overline { { X } } } - \mathbb { E } [ X ] \geq \epsilon$ following Equation 1, where $R$ is the range of the random variable, and n is the number of observations.

$$
\epsilon = \sqrt { \frac { R ^ { 2 } \log ( \frac { 1 } { \delta } ) } { 2 n } }\tag{1}
$$

The Hoefding Tree applies this bound for impurity measures to determine if $X _ { a } ,$ the attribute with the highest impurity measure G (such as information gain [10] or the Gini index [11]), is the ideal attribute to split on a split attempt. Given $X _ { b }$ , the attribute with the second-highest impurity measure, if $G ( X _ { a } ) - G ( X _ { b } ) \geq \epsilon ,$ a split will occur on $X _ { a }$

Note that lim $\mathfrak { l } _ { n \to + \infty } \epsilon = 0$ , and if $G ( X _ { a } )$ and $G ( X _ { b } )$ have similar values in a node, a split will require many observations to occur. To relax the Hoefding constraint in these situations, the Hoefding Tree has a tie threshold. Given τ (user-given threshold), a split will ensue when $\tau > \epsilon$

Assuming d to be the number of attributes, v the maximum number of values per attribute, c the number of classes, and l the number of leaf nodes, the Hoefding tree algorithm requires $\mathcal { O } ( l d v c )$ memory to store the necessary counts.

An aspect that established Hoefding Trees as accurate decision trees was the addition of Naive Bayes at leaf nodes [12]. It was observed that Naive Bayes could outperform decision trees in initial streaming scenarios where the decision tree has trained with little data. In [13], the authors propose the selection of Naive Bayes or majority class strategy prediction according to the method with the highest accuracy at the leaf.

Another critical aspect of Hoefding Trees is memory management since storing all instances from a data stream is unfeasible. Leaf nodes maintain statistics from data, such as histograms for nominal data, and rely on forgetting mechanisms. We refer the interested reader to [14] for more details.

## 3.2 Extremely Fast Decision Trees

In [7], the authors propose EFDT (Extremely Fast Decision Trees), an extension of Hoefding Trees. The first change to Hoefding Trees refers to the Hoefding Bound. Instead of comparing the split evaluation function of the two features that maximize this function, the comparison is made between the feature $X _ { a }$ that maximizes the split evaluation function and the case where no split happens $( X _ { \varnothing } )$ . Equation 2 is used as a condition for performing a split in $X _ { a }$

$$
G ( X _ { a } ) - G ( X _ { \varnothing } ) > { \sqrt { \frac { R ^ { 2 } \log ( { \frac { 1 } { \delta } } ) } { 2 n } } }\tag{2}
$$

Another extension proposed in EFDT is a split reevaluation mechanism [7]. As more instances arrive, the probability of a better split appearing is higher than that of a split that ensued earlier. The reevaluation mechanism is similar to the splitting mechanism. The Hoefding bound is used to compare the value of the split evaluation function of $X _ { a }$ in case a new split replaces the tree branch by a new split and the current feature $( X _ { c u r r e n t } )$ split evaluation function at the time the split occurred. A branch of the tree is replaced by a new split if Equation 3 holds.

$$
G ( X _ { a } ) - G ( X _ { \mathrm { c u r r e n t } } ) > \sqrt { \frac { R ^ { 2 } \log ( \frac { 1 } { \delta } ) } { 2 n } }\tag{3}
$$

Since EFDT must store data at intermediate nodes to re-evaluate splits, it requires O(ndvc) memory, where n is the number of nodes. The default value for periodic reevaluation was defined as 2000 by the authors.

In [15], the authors compare EFDT and Hoefding Trees as base learners for ensembles, a similar aim to this work [5]. Therefore, we also perform experiments with EFDT as a base learner of ensembles since the results in [15] suggest that EFDT is superior to Hoefding Trees at an ensemble level. We also improve the evaluation of base learners in this work compared to [15] and explain the improvements in Section 5.

## 3.3 Adaptive Splitting Trees

Hoefding-based Trees follow a fixed periodic splitting strategy, activated at intervals of GP instances and controlled by the Hoefding test. This policy is invariant to stream dynamics and tree performance, leading to blind recurrent splitting attempts and greedy split searches when little change occurs in the leaf, which results in little change in the tree structure.

In [5], the authors overcame this limitation by proposing the Local Adaptive Streaming Tree (LAST). The rationale behind LAST is that change detectors can provide adaptability to determine split points at leaf nodes. Figure 2 presents LAST’s incremental training and splitting mechanism. In a leaf node, change detectors can monitor either the impurity or error rate. If a change detector at the leaf triggers a change, then a split will ensue if $G ( X _ { a } ) > 0$

![](images/f1ff17ce15ab15d6416df9c5c92d81ab86c956a42f6f7c82bd61c9b743348fde.jpg)  
Fig. 2: LAST’s training process

Change detectors are constantly updated with arriving instances, meaning that these algorithms track how the stream evolves. Since LAST applies the softest split constraint $( G ( X _ { a } ) > 0 )$ , the change detectors efectively control how the tree grows.

LAST requires O(ψldvc) memory, where ψ is the memory complexity of the change detection algorithm applied.

In [6], the authors presented a more thorough evaluation of LAST as a single monolithic decision tree, comparing LAST with diferent change detectors from the literature and performing an ablation analysis. This difers from the present paper, whose scope is ensemble learning.

## 3.4 Online Ensembles

Ensembles are state-of-the-art algorithms for data stream classification, often adapting a batch version counterpart.

In [16], the authors proposed an online version of Bagging [17]. For each incoming instance, a base learner trains with k copies of the instance, where k is a random variable that follows a Poisson(λ = 1) distribution, simulating random sampling with replacement in online scenarios. This process induces diversity amongst its members, which have their votes cast and combined using simple majority voting. In the same work, the authors proposed an online version of AdaBoost.M1 [18]. For each classifier, correct and wrong classification weights are maintained. When a new incoming instance arrives, the λ parameter starts at one, and the classifier is trained with k = Poisson(λ) copies of the instance. If a learner makes a wrong classification, the value λ increases based on the learner’s wrong weights, and in case the classifier makes a correct prediction, the value of λ decreases for the training of the next classifier. In contrast to Online Bagging, Online Boosting assigns each ensemble member a weight, which is used during prediction in a weighted majority voting scheme.

To deal with concept drifts, the authors in [19] propose ADWIN-Bagging and ADWIN-Boosting. For each ensemble classifier, an ADaptive WINdowing (ADWIN) [20] drift detector monitors the error rates of the classifier. If the detector coupled with a classifier detects a concept drift, the least accurate member of the ensemble (in terms of the estimated ADWIN error) is reset.

Extending the ADWIN-Bagging algorithm, authors in [21] propose the Leveraging Bagging algorithm. Instead of sampling being conducted by Poisson(λ = 1), sampling follows Poisson(λ = 6), making classifiers more specialized. In [21], the authors also propose using error output codes, as in [22], where binary classifiers are created per class, and the final ensemble prediction is a sum of the results of classifiers per class. This approach increases computational cost because the number of base learners scales with the number of classes. However, it can sometimes outperform standard Leveraging Bagging, despite the latter achieving the best average ranking in the original study [21].

Extending the Online Boosting classifier, the authors proposed in [23] the Boostinglike Online Learning Ensemble (BOLE) algorithm. Training starts with λ = 1, and classifiers are sorted by correct prediction rate (regarding correct and wrong classification weights [16]). The classifier with the worst correct prediction rate is trained first. If this classifier gets a correct classification, the best classifier not yet trained (in terms of correct prediction rate) receives an altered λ for training. Otherwise, the worst classifier not yet trained receives it. BOLE significantly increases λ when many base learners misclassify instances, since the worst base learners influence each other until a base learner makes the right prediction. The performance of the ensemble overall influences the values of λ.

In [24], the authors propose the Adaptive Random Forest (ARF) algorithm, a version of the Random Forest algorithm [25], where, in each split attempt, a random subset of features is considered for selecting the split with the best quality. Sampling is conducted by Poisson(λ = 6). To deal with drifting scenarios, each classifier has two drift detectors, one for warnings and one for concept drift detection. If the warning detector triggers a change, a new classifier is trained in the background. Moreover, if the concept drift detector triggers a change, the previously trained background classifier replaces the original one. In addition to that, the classifier’s votes are weighted by accuracy.

In [26], the authors propose the Streaming Random Patches (SRP) algorithm, a version of the Random Patches [27], using the same mechanisms as in ARF, but instead of considering a random subset of features per node, all the split attempts are evaluated with a random subset of features defined at the creation of the base learner. In SRP, sampling is conducted by $\mathrm { P o i s s o n } ( \lambda = 6 )$

In [28], the authors propose an extension to the Adaptive Random Forest (ARF) algorithm, namely Adaptive Regularized Ensemble (ARE). The authors integrate an instance selection technique for the base learners of ARF, where base learners train mostly with instances that they incorrectly classify. To avoid bias due to noisy instances, the authors propose a rejection strategy to train with instances that the base learner correctly classified. The base classifier will train with an instance that it correctly classified after rejecting ζ (user-given) instances of the same class. The authors also integrate a classifier selection technique, in which only the classifiers that perform above the average accuracy of the ensemble vote are selected.

Another recent approach is Adaptive Random Tree Ensemble (ARTE), proposed in [29]. ARTE base learners have random feature selection per node level, like ARF; however, the percentage selected is not fixed among base learners. In particular, the percentage selected per base learner is drawn from a uniform distribution. ARTE base learners also select the best split with random cut points, as in [30].

## 4 Hoefding Adaptive Splitting Trees Base Learners

As highlighted in [31], one of the key components of state-of-the-art ensembles is diversity, often induced via the distribution of a diferent number of copies of an instance to its base learners via random sampling with replacement, e.g., Bagging [17], or instance weighting, e.g., Boosting [18]. However, under adaptive splitting, ensemble members tend to generate highly correlated predictions in the early phases of the data stream, potentially undermining ensemble diversity [15].

Incremental decision trees begin learning from a single root node; consequently, during the early stages of the data stream, their predictions are issued by either a majority-class classifier or a Naive Bayes model, whichever attains higher accuracy [13]. Because every ensemble member behaves this way, the members produce nearly identical outputs, and the change detectors attached to them are therefore updated with highly similar inputs. Since instances are weighted through Poisson(λ) sampling with expected value λ, the detectors exhibit similar behavior, trigger splits at closely aligned times, and ultimately yield reduced ensemble diversity. When the only source of diferentiation among detectors is the variance introduced by Poisson weighting, split decisions are governed more by random sampling efects than by genuine performance diferences or the underlying class distribution. Under these conditions, the LAST soft split criterion $\left( G ( X _ { a } ) > 0 \right)$ becomes liable to trigger suboptimal splits.

To overcome this limitation, we propose two decision trees that enhance the splitting mechanism to be simultaneously adaptive, reacting to changes in the model’s performance or distribution, and induce diversity among base learners of the ensemble. Hoefding-based Trees, such as Hoefding Tree and EFDT, are sensitive to the number of copies they receive for training regarding splitting moment, since splitting time depends on the number of observations present in the leaf node, and present a hard constraint based on the Hoefding bound. Therefore, we propose a combination of the adaptive and periodic splitting mechanism, as illustrated in Figure 3.

![](images/d616d7526d7ee16ae3fff1e9b481d644b33747d25915456dc2cf74c25d9e0707.jpg)  
Fig. 3: Training process of combining adaptive and periodic splitting mechanism

Algorithm 1 presents the proposed methods. First, we propose the Hoefding Local Adaptive Splitting Tree (HLAST). HLAST maintains change detectors at leaf nodes that monitor either the predictive performance or the class distribution purity, as LAST does, and periodically performs split attempts with the Hoefding bound, as Hoefding Trees do. If the change detector flags a change, and $G ( X _ { a } ) > 0$ (Algorithm 1, line 18), or $\begin{array} { r } { G ( X _ { a } ) - G ( X _ { b } ) > \sqrt { \frac { R ^ { 2 } \log ( \frac { 1 } { \delta } ) } { 2 n } } } \end{array}$ (Algorithm 1, line 14) in a periodic split attempt, the leaf is split on $X _ { a } .$

We also propose the Extremely Fast Local Adaptive Streaming Tree (EFLAST). EFLAST performs splits periodically, if $G ( X _ { a } ) > \sqrt { \frac { R ^ { 2 } \log ( \frac { 1 } { \delta } ) } { 2 n } }$ (Algorithm 1, line 16) or adaptively, if the change detector flags a change and $G ( X _ { a } ) > 0$ . We also maintain the splitting reevaluation mechanism (Algorithm 1, line 16), where non-leaf nodes replace their branch by a new split if $\begin{array} { r } { G ( X _ { a } ) - G ( X _ { c u r r e n t } ) > \sqrt { \frac { R ^ { 2 } \log ( \frac { 1 } { \delta } ) } { 2 n } } } \end{array}$

Overall, the proposed trees still periodically search for the best splits and incur a computational overhead due to the use of change detectors; however, we show experimentally that such overhead is not prohibitive.

Algorithm 1 HLAST and EFLAST: splitting strategy by combination of adaptive   
decision trees and HT or EFDT   
Input: S: a data stream,   
X : a feature,   
G(·) : a purity measure (such as Gini or Entropy),   
ψ : a Change Detection algorithm   
GP: Grace Period (sample frequency that a leaf node will attempt to split)   
DT: Decision Tree (The constructed decision tree)   
splitting type : splitting mechanism selection between HT or EFDT (instantiating   
HLAST or EFLAST, respectively)   
1: for $( { \vec { x } } _ { n } , y _ { n } ) \in S$ do   
2: if splitting type = EFDT then   
3: Traverse tree and ascertain that non-leaf nodes pass the reevaluation of   
split attempts   
4: end if   
5: Let $l \gets D T ( \vec { x } _ { n } )$ ▷ traverse tree until leaf node   
6: $n _ { l } \gets n _ { l } + 1$ ▷ increment number of samples in the leaf   
7: if $\neg ( l$ has samples from only one class) then   
8: if $n _ { l }$ mod $G P = 0$ then   
9: Compute $G ( X _ { i } )$ for each $X _ { i } \in X _ { l }$ stored in l   
10: Let $X _ { a }$ be the feature with highest G   
11: Let $A _ { b }$ be the feature with second highest $G$   
12: Let $\epsilon  \sqrt { \frac { R ^ { 2 } \log ( \frac { 1 } { \delta } ) } { 2 n _ { l } } }$ ▷ Eq. (1)   
13: if splitting type = HT then   
14: Let cond $ G ( A _ { a } ) - G ( A _ { b } ) > \epsilon$ ▷ HLAST tree type   
15: else   
16: Let cond $ G ( A _ { a } ) > \epsilon$ ▷ EFLAST tree type   
17: end if   
18: if ((cond $\sqrt { \tau > \epsilon ) \lor ( l _ { \psi } }$ detected a change $\land G ( A _ { a } ) > 0 ) \ ) \land A _ { a } \neq \emptyset$ then   
19: Replace l by leaf nodes that split on $X _ { a }$   
20: for each leaf node $l _ { i }$ from splitting on $X _ { a }$ do   
21: $n _ { l _ { i } } \gets 0$   
22: $l _ { i \psi }  \psi$ ▷ creates a new change detector at each leaf node   
23: end for   
24: end if   
25: end if   
26: end if   
27: end for

## 5 Methodology

This section assesses the impact of the proposed tree models when integrated into ensembles constructed with various algorithms. The algorithms evaluated in this work were implemented in Java by extending the Massive Online Analysis (MOA) software [1]. All of our experiments are reproducible, and the source code and additional results (such as raw accuracy and F1-Score results) are publicly available on the paper’s support repository<sup>2</sup>. We performed all the experiments on an Intel(R) Core(TM) i7-12700H @ 2.30 GHz with 16 GB of RAM. We performed experiments with 13 realworld datasets made available in [32] and 24 synthetic datasets retrieved from [33]. Their main idea is to test how methods react to concept drift [19]. Table 1 describes the datasets used in this work. The datasets at the top are synthetic, while the ones at the bottom are datasets representing real-world data.

Table 1: Description of the evaluated datasets.
<table><tr><td>Dataset</td><td># Samples</td><td># Features</td><td># Classes</td><td>Majority Class (%)</td><td>Concept Drift</td></tr><tr><td>AGR-f(7,8,9,10,9,8,7)ra AGR-f(7,8,9,10,9,8,7)rg</td><td>106</td><td>9</td><td>2</td><td>52.83</td><td>Recurrent-Abrupt</td></tr><tr><td></td><td>10⁶</td><td>9</td><td>2</td><td>52.83</td><td>Recurrent-Gradual</td></tr><tr><td>AGR-f(1,2,3,4,5,6,7,8,9,10)a</td><td>10⁶</td><td>9</td><td>2</td><td>52.83</td><td>Abrupt</td></tr><tr><td>AGR-f(10,9,8,7,6,5,4,3,2,1)a</td><td>10⁶</td><td>9</td><td>2</td><td>52.83</td><td>Abrupt</td></tr><tr><td>AGR-f(1,2,3,4,5,6,7,8,9,10)g</td><td>106</td><td>9</td><td>2</td><td>52.83</td><td>Gradual</td></tr><tr><td>AGR-f(10,9,8,7,6,5,4,3,2,1)g</td><td>10⁶</td><td>9</td><td>2</td><td>52.83</td><td>Gradual</td></tr><tr><td>SEA-f(1,2,3,4,3,2,1)ra</td><td>106</td><td>3</td><td>2</td><td>59.91</td><td>Recurrent-Abrupt</td></tr><tr><td>SEA-f(1,2,3,4,3,2,1)rg</td><td>10⁶</td><td>3</td><td>2</td><td>59.91</td><td>Recurrent-Gradual</td></tr><tr><td>SEA-f(1,2,3,4)a</td><td>10⁶</td><td>3</td><td>2</td><td>59.91</td><td>Abrupt</td></tr><tr><td>SEA-f(1,2,3,4)g</td><td>106</td><td>3</td><td>2</td><td>59.91</td><td>Gradual</td></tr><tr><td>SEA-f(4,3,2,1)a</td><td>10⁶</td><td>3</td><td>2</td><td>59.91</td><td>Abrupt</td></tr><tr><td>SEA-f(4,3,2,1)g</td><td>106</td><td>3</td><td>2</td><td>59.91</td><td>Gradual</td></tr><tr><td> $_ { \mathrm { L E D - f } ( 7 , 5 , 3 , 1 , 3 , 5 , 7 ) _ { r a } }$ </td><td>10⁶</td><td>24</td><td>10</td><td>10.28</td><td>Recurrent-Abrupt</td></tr><tr><td>LED-f(7,5,3,1,3,5,7)rg</td><td>10⁶</td><td>24</td><td>10</td><td>10.28</td><td>Recurrent-Gradual</td></tr><tr><td>LED-f(1,3,5,7)a</td><td>106</td><td>24</td><td>10</td><td>10.28</td><td>Abrupt</td></tr><tr><td>LED-f(1,3,5,7)g</td><td>10⁶</td><td>24</td><td>10</td><td>10.28</td><td>Gradual</td></tr><tr><td>LED-f(7,5,3,1)a</td><td>10⁶</td><td>24</td><td>10</td><td>10.28</td><td>Abrupt</td></tr><tr><td>LED-f(7,5,3,1)g</td><td>106</td><td>24</td><td>10</td><td>10.28</td><td>Gradual</td></tr><tr><td>RBFs</td><td>106</td><td>10</td><td>5</td><td>30.01</td><td>Incremental</td></tr><tr><td>RBFm RBFf</td><td>10⁶</td><td>10</td><td>5</td><td>30.01</td><td>Incremental</td></tr><tr><td>HYPERs</td><td>10⁶</td><td>10</td><td>5</td><td>30.01</td><td>Incremental</td></tr><tr><td>HYPERm</td><td>10⁶</td><td>10</td><td>2</td><td>50</td><td>Incremental</td></tr><tr><td></td><td>10⁶</td><td>10</td><td>2</td><td>50</td><td>Incremental</td></tr><tr><td>HYPERf Outdoor</td><td>10⁶</td><td>10</td><td>2</td><td>50</td><td>Incremental</td></tr><tr><td>Elec</td><td>4,000</td><td>21</td><td>40</td><td>4.11</td><td>Unknown</td></tr><tr><td>Rialto</td><td>45,312</td><td>8</td><td>2</td><td>57.41</td><td>Unknown</td></tr><tr><td>Airlines</td><td>82,250</td><td></td><td>10</td><td>10</td><td>Unknown</td></tr><tr><td>CoverType</td><td>539,383</td><td>27</td><td></td><td>55.47</td><td>Unknown</td></tr><tr><td></td><td>581,012</td><td>54</td><td>272</td><td>48.75</td><td>Unknown</td></tr><tr><td>Nomao</td><td>34,465</td><td>119</td><td></td><td>71.44</td><td>Unknown</td></tr><tr><td>Poker</td><td>829,201</td><td>10</td><td>10</td><td>47.78</td><td>Unknown</td></tr><tr><td>NOAA</td><td>18,158</td><td>8</td><td></td><td>69.74</td><td>Unknown</td></tr><tr><td>INSECTSa</td><td>52,848</td><td>33</td><td>26</td><td>16.07</td><td>Abrupt</td></tr><tr><td>INSECTSi</td><td>57,018</td><td>33</td><td>6</td><td>11.56</td><td>Incremental</td></tr><tr><td>INSECTS9</td><td>24,150</td><td>33</td><td>6</td><td>15.76</td><td>Gradual</td></tr><tr><td>Asfault LADPU</td><td>8,066 22,950</td><td>62 96</td><td>5 10</td><td>55.59 10</td><td>Unknown Unknown</td></tr></table>

The synthetic dataset generators are described as follows.

## AGRAWAL [34]

The AGRAWAL generator simulates a loan-approval scenario with nine features that represent applicant attributes: salary (numerical, 0–150K), commission (numerical, 0– 75K), age (numerical, 20–80), eloan (boolean), car (nominal, 1–20), zipcode (nominal, 0–8), hvalue (numerical, 50K–600K), hyears (numerical, 1–30), and loan (numerical, 0–500K). Ten classification functions encode distinct loan-approval rules over these attributes, mapping feature combinations to a binary label (approved or denied). These functions are progressively more complex from $f _ { 1 }$ to $f _ { \mathrm { 1 0 } } \mathrm { : }$ the earlier ones depend on a single attribute with simple cut points, while the later ones combine several attributes through compound conditions and derived quantities. For instance, $f _ { 1 }$ assigns the label using only $a g e { , }$ approving applicants younger than 40 or at least 60 years old and denying the remaining middle-age range. In contrast, $f _ { 1 0 }$ is the most elaborate rule, computing a derived disposable income from salary, commission, and loan together with an equity term that depends on hvalue and hyears, and approving the loan only when this disposable income is positive. Concept drift is simulated by switching between these functions. Following the table, we use a recurrent variant cycling through $\{ f _ { 7 } , f _ { 8 } , f _ { 9 } , f _ { 1 0 } , f _ { 9 } , f _ { 8 } , f _ { 7 } \}$ (6 drifts) and monotone ascending $\{ f _ { 1 } , \ldots , f _ { 1 0 } \}$ and descending $\{ f _ { 1 0 } , \ldots , f _ { 1 } \}$ variants (9 drifts each).

## SEA [35]

The SEA generator produces three numerical features $f _ { 1 } , f _ { 2 } , f _ { 3 }$ uniformly sampled from [0, 10), where $f _ { 3 }$ is entirely irrelevant to the class label, making this a threefeature problem with an embedded irrelevant dimension. The class label is binary and is determined by a threshold applied to the sum of the two relevant features. Four concepts are defined as $c _ { 1 } \colon ( f _ { 1 } + f _ { 2 } \leq 8 ) , c _ { 2 } \colon ( f _ { 1 } + f _ { 2 } \leq 9 ) , c _ { 3 } \colon ( f _ { 1 } + f _ { 2 } \leq 7 )$ and $c _ { 4 } \colon ( f _ { 1 } + f _ { 2 } \leq 9 . 5 )$ : an instance receives class 1 if the condition for the active concept holds, and class 0 otherwise. The recurrent variant cycles through concepts in the order $\{ c _ { 1 } , c _ { 2 } , c _ { 3 } , c _ { 4 } , c _ { 3 } , c _ { 2 } , c _ { 1 } \}$ (6 drifts), while the monotone abrupt and gradual variants follow $\{ c _ { 1 } , c _ { 2 } , c _ { 3 } , c _ { 4 } \} ~ { \mathrm { o r } } ~ \{ c _ { 4 } , c _ { 3 } , c _ { 2 } , c _ { 1 } \}$ (3 drifts each). Gradual transitions are controlled by a window in which instances are drawn from both the outgoing and incoming concept with a probability that changes linearly.

## LED [11]

The LED generator simulates reading a seven-segment LED display showing digits 0–9, where each digit is encoded by 7 boolean features corresponding to the active segments. An additional 17 boolean features are appended and are entirely irrelevant to the classification task, making feature selection non-trivial. Furthermore, each boolean feature is independently flipped with a 10% probability, introducing label noise. Concept drift is simulated by progressively masking or unmasking the originally relevant features: as the number of relevant features decreases, fewer segments are correctly observed and the classification boundary becomes harder to learn. In our experimental setup the order of relevant feature counts across concepts is {7, 5, 3, 1, 3, 5, 7} for the recurrent variant $\left( \mathrm { L E D } _ { r a } \right.$ and $\mathrm { L E D } _ { r g } .$ 6 drifts) and {1, 3, 5, 7} or {7, 5, 3, 1} for the monotone variants $\mathrm { ( L E D } _ { a }$ and $\operatorname { L E D } _ { g } , 3$ drifts each). Abrupt variants transition instantaneously between concepts, whereas gradual variants mix instances from adjacent concepts over a transition window.

## RBF [1]

The Radial Basis Function (RBF) generator creates a Gaussian mixture model over 10 features and 5 classes, in which each of the 50 randomly initialised centroids is associated with a standard deviation, a weight, and a class label; instances are drawn by selecting a centroid with probability proportional to its weight and sampling from the corresponding Gaussian. Incremental concept drift is introduced by continuously moving the centroids through the feature space at a fixed speed, so that the decision boundaries shift gradually over time rather than switching abruptly. Three speed variants are evaluated: RBF (slow, speed $= 1 0 ^ { - 5 } )$ , $\mathrm { R B F } _ { m }$ (moderate, speed $= 1 0 ^ { - 4 } )$ , and $\mathrm { R B F } _ { f }$ (fast, speed $= 1 0 ^ { - 3 } )$ . The increasing speed makes it progressively harder for learners to track the drifting class regions.

## HYPER [36]

A hyperplane in $\mathbb { R } ^ { d }$ is a $( d \mathrm { - } 1 )$ -dimensional afine subspace defined by $\textstyle \sum _ { i = 1 } ^ { d } w _ { i } x _ { i } = w _ { 0 }$ which partitions the space into two half-spaces and thus provides a natural binary decision boundary. Instances are generated uniformly at random in $[ 0 , 1 ] ^ { d }$ , and their class is determined by which side of the hyperplane they fall on. Incremental concept drift is simulated by continuously rotating the hyperplane: at each step, a random weight $w _ { i }$ is perturbed by an amount proportional to the magnitude parameter $\sigma ,$ and all weights are subsequently re-normalised. Three speed variants are evaluated: HYPER (slow, $\sigma = 1 0 ^ { - 4 } )$ , $\mathrm { I Y P E R } _ { m }$ (moderate, $\sigma = 1 0 ^ { - 3 } )$ , and $\mathrm { H Y P E R } _ { f }$ (fast, $\sigma = 1 0 ^ { - 2 } )$ . All variants were configured with $d = 1 0$ features.

In contrast to the experiments conducted in [15], all ensembles were set with 100 base learners, as used in recent state-of-the-art ensembles [24, 26]. All parameters from ensembles were set to default as implemented in the MOA framework. In addition to the ensembles evaluated in [15], we also experimented with SRP [26], ARE [28], and ARTE [29]. We acknowledge SGBT [37], a method that adapts gradient boosting trees for streaming scenarios; however, many datasets ran for a week constantly demanding memory swap and presented a high computational cost, and were therefore excluded from the experiments. We also chose not to include in our experiments ensemble algorithms specifically designed for imbalanced contexts, such as ROSE [38], KUE [33], and others cited in [39], opting instead to focus on general-purpose ensemble methods. Only the best state-of-the-art methods are evaluated for visualization convenience, including Leveraging Bagging, ARF, SRP, ARE, and ARTE, based on the results reported in [24, 26, 29].

As in [7], we assess the predictive performance obtained with a test-then-train validation strategy, where every instance is used first for testing and then for training, known as Prequential evaluation [40]. All metrics are averaged over 20 evenly spaced points of the stream, as more points did not change results drastically.

To statistically compare the results obtained by the ensembles of base learners, we performed a Friedman test with a significance level of 1% $\mathrm { ( p \mathrm { - v a l u e < 0 . 0 1 } ) }$ and a pairwise one-sided Wilcoxon signed-rank post-hoc test with a Holm correction [41, 42]. We visually represent the results of this test using a diagram as in [43]. In this diagram, the methods are sorted according to their average ranking output by the statistical test. The methods connected by a line do not present statistically significant diferences.

Time processing and memory eficiency are key components of data stream mining algorithms, as cited earlier and in [1]. Although not covered in the experimentation presented in [15], in this work we evaluate CPU-Time, memory usage (peak RAM, in

MB, throughout time), and the mean tree size (number of nodes) in the ensemble. For these metrics, we provide violin plots sorted by median.

LAST variants that incorporate data distribution monitoring at the leaf nodes are indicated by the sufix $^ { 6 6 } D ^ { 9 }$ , as in $\mathrm { L A S T } _ { D } , \mathrm { H L A S T } _ { D }$ and $\mathrm { E F L A S T } _ { D }$ . The detector used in all $\mathrm { L A S T }$ versions was $\mathrm { H D D M } _ { A } \ [ 4 4 ]$ , given the analysis done in [6]. It is important to clarify the relationship between this work and [6], which evaluates LAST exclusively as a monolithic decision tree and reports no ensemble experiments. Two of its contributions are directly relevant here. First, Nowak Assis et al. [6] conjecture that $\mathrm { L A S T } _ { D }$ , which monitors class-distribution purity rather than error rate, may suit ensemble use better than LAST: because of the instance sampling process, leaf nodes are expected to undergo more frequent distributional changes, which could lead $\mathrm { L A S T } _ { D }$ to induce greater diversity among base learners. We investigate this hypothesis in the present work. Second, among the change detectors evaluated, HDDM<sub>A</sub>, $\mathrm { M D D M } _ { A }$ , and ADWIN stand out: they produce trees as accurate as the DDM-type detectors (DDM, EDDM, RDDM) while remaining more computationally eficient, owing to stricter detection bounds and fewer false positives. We adopt HDDM<sub>A</sub>, which achieved the best overall ranking, though $\mathrm { M D D M } _ { A }$ and ADWIN performed comparably.

## 6 Results

This section presents results for experimentation conducted to assess the proposed methods. First, Section 6.1 presents a benchmark for ensembles and base learners. Next, Section 6.2 presents a comparison of tree sizes, while Section 6.3 discusses the computational cost of the methods. Finally, Section 6.4 analyzes the F1-Score of methods in concept drifting scenarios.

## 6.1 Benchmark

This section benchmarks the proposed trees against Hoefding Trees (HT), EFDT, and the original adaptive trees LAST and $\mathrm { L A S T } _ { D }$ as base learners of the five ensembles evaluated in this work, where we report F1-Score. We discuss the behaviour of each base tree per dataset, so that the properties of the data that favour the proposed trees become explicit. The raw F1-Score and accuracy values for every combination of ensemble and base learner are available in the paper’s repository.

## 6.1.1 Best-performing base tree

Figure 4 summarizes how each base tree compares against a standard Hoefding Tree across all ensembles. The horizontal axis reports the percentage of cases in which the base learner wins over HT, while the vertical axis reports the F1-Score diference to HT in percentage points, where the marker is the median diference across all datasets and ensembles and the vertical bars span the lower and upper quantiles of that diference. In real-world data, $\mathrm { H L A S T } _ { D }$ and HLAST are among the best-performing trees, winning over HT in 75% and 63% of the cases, respectively, ahead of EFDT and EFLAST, and largely ahead of the original adaptive trees LAST and $\mathrm { L A S T } _ { D }$ , which win in only

38% and 42% of the cases. Considering the quantiles, HLAST and HLAST keep a median diference at or above zero, and their upper quantile reaches above two and one percentage points, respectively, meaning that on the favorable datasets the gains are substantial, while their lower quantile stays close to zero, meaning they rarely lose much when they do not win. The original adaptive trees show the opposite behaviour, with a median diference below zero, which confirms that LAST and LAST are weak ensemble base learners, while combining the adaptive mechanism with the periodic Hoefding split recovers and surpasses the performance of HT.

Figure 5 details this diference per dataset, with each point colored by the ensemble that produced it. The gains of the proposed trees concentrate on datasets with a larger number of classes, such as Outdoor (40 classes), Rialto, Poker and LADPU (10 classes), CoverType (7 classes) and the INSECTS variants (6 classes), reaching up to 16 percentage points in LADPU and around 10 points in Outdoor and Rialto. The highest gains come from Leveraging Bagging and ARE, the ensembles that grow the largest trees (Table 2), so the adaptive splitting strategy has the most pronounced efect on them, but the remaining ensembles also present positive gains in general, and the proposed trees produce higher predictive quality across the board. On binary datasets such as Electricity, Airlines, NOAA and Nomao, the diference stays close to zero. This behaviour follows directly from the description of the proposed trees. With more classes, a leaf node takes longer to become pure and ofers more opportunities to split, so letting the change detector decide when to split, on top of the periodic Hoefding attempts, produces larger and more diverse trees (Table 2). On binary problems the concept is learned quickly and the additional adaptive splitting remains mostly idle, leaving little room for improvement.

The datasets on which the proposed trees perform best are also the most challenging, according to the streaming benchmark in [32]. That work characterizes real-world streams by their temporal dependence and shows that, due to autocorrelation, a naive No-Change baseline—one that simply predicts the label of the most recent instance— can already achieve strong performance, rendering several commonly used benchmark datasets deceptively easy. This is not the case for Rialto, LADPU, and Asfault, where the proposed trees achieve their highest gains: these datasets lack such autocorrelation and thus require the model to genuinely learn the underlying concept. Even on datasets with autocorrelation, such as Electricity and CoverType, the ensembles outperform the No-Change baseline by a wide margin. In terms of F1-Score, ARTE and HLAST reach 91.33% against 84.43% on Electricity and 86.60% against 76.41% on CoverType, confirming that the reported gains reflect genuine concept learning rather than artifacts of easy data.

Figure 5 further shows that the two proposed trees do not behave identically across ensembles. Under SRP, HLAST is the weakest of the three base learners, with a standard HT outperforming it in roughly 70% of the cases. The reason lies in how SRP operates: it assigns each base learner a fixed random subset of features at creation, so some learners are inevitably built on weak or uninformative subsets. Because HLAST relies on a soft adaptive split driven by the error rate, these learners continue splitting even when their features are poor, and since SRP weights votes by accuracy, the resulting large but weakly featured trees bias the weighted voting and degrade the ensemble. $\mathrm { H L A S T } _ { D }$ avoids this pitfall: by monitoring class-distribution purity rather than the error rate, its growth is not driven by fluctuations of the error detector. $\mathrm { H L A S T } _ { D }$ is therefore the natural choice to pair with SRP, whereas HLAST is better suited to ensembles such as ARTE.

![](images/509c48f9dd2958ea516b22e2cf682b037f32fc642d2f04bc7b1b3d9e61dff827.jpg)  
Fig. 4: F1-Score of each base learner against HT, reported as the percentage of cases in which the base learner wins over HT (win rate) and the F1-Score diference quantiles in percentage points, for synthetic (left) and real-world (right) data

F1-Score: HLAST / HLAST vs HT per real-world dataset  
![](images/757dcc0e87677e18d8e9d06d21c15a8364f6bcd0357a716090d988d71bc73e83.jpg)  
Fig. 5: F1-Score diference between HLAST/HLAST<sub>D</sub> and HT in real-world data for all ensembles evaluated

Figure 6 shows that on synthetic data the proposed trees have little efect, with the F1-Score diference to HT staying close to zero for all ensembles. The synthetic generators used in the benchmark configure simple linear concepts, as in AGRAWAL, SEA and HYPER, or binary patterns, as in LED, and their main purpose is to test how methods react to concept drift. As further discussed in Section 6.4, the ensembles react to drift in a very similar way, and the simple concepts are learned fast by the trees, so growing them further with adaptive splitting does not change the results in a meaningful way. The impact of the proposed trees is therefore more evident on real-world data, where the concepts are more complex.

F1-Score: HLAST / HLAST vs HT per synthetic dataset  
![](images/ed5afc18fcc478a082fa8ff89f7005212cbbe78a82da9f84ba1aabf92141748a.jpg)  
Fig. 6: F1-Score diference between HLAST/HLAST and HT in synthetic data for all ensembles evaluated

## 6.1.2 Best-performing ensemble

Figure 7 reports the raw F1-Score of HT, HLAST, and $\mathrm { H L A S T } _ { D }$ for every ensemble, marking the best base learner within each ensemble and the overall best result per dataset. The proposed trees provide the overall best result on most real-world datasets, and almost all of these wins occur on the multi-class datasets, in line with the pertree analysis. Specifically, ARTE with HL ${ \bf A S T } _ { D }$ delivers the best result on Rialto, the INSECTS variants, LADPU, and Asfault; ARTE with HLAST leads on Nomao; and SRP with HLAST or $\mathrm { H L A S T } _ { D }$ leads on Outdoor, Airlines, CoverType, and Poker.

Considering the results as a whole, the proposed trees achieve the strongest performance, and we recommend ARTE as the ensemble to pair with them. ARTE combines a per-learner random feature percentage with random cut points, which already induces high diversity; the proposed trees build on this by splitting at more informed moments rather than blindly at fixed intervals. The combination is particularly efective on real-world data, where ARTE with $\mathrm { H L A S T } _ { D }$ outperforms ARTE with HT on almost every dataset and yields the strongest and most consistent results across the entire benchmark.

Figure 8 shows that, on synthetic data, the diferences between base learners are small and the best result alternates among ensembles and trees, with Leveraging Bagging leading on LED, SRP on HYPER, and ARTE on SEA. This reinforces the point that synthetic concepts do not clearly separate the methods, and that the advantage of the proposed trees (and of ARTE in particular) stems mainly from the more complex real-world data.

F1-Score: HT / HLAST / HLASTp per ensemble, per real-world dataset ( = best per ensemble, ★ = overall best)  
![](images/3284938d46d70dd79144b3dc43b9b89def20d413bec47975e70e2c07d37e09cd.jpg)  
Fig. 7: Raw F1-Score for $\mathrm { H T / H L A S T / H L A S T } _ { D }$ in real-world data for all ensembles evaluated

F1-Score: HT / HLAST / HLASTp per ensemble, per synthetic dataset ( = best per ensemble, ★ = overall best)  
![](images/88b7b1c2bbd3b132a17cf9ac143e535e78c3cb19814263312d94d24d7fbe76b5.jpg)  
Fig. 8: Raw F1-Score for $\mathrm { H T / H L A S T / H L A S T } _ { D }$ in synthetic data for all ensembles evaluated

F1-Score: HT / HLAST / HLAST per ensemble, per synthetic dataset ( = best per ensemble, ★ = overall best)  
![](images/d54744ad5b4ab0aa8fe64f8c46ff5dc0394610df0b16dad523087d5cef1b9a8e.jpg)  
Fig. 8: (Continued) Raw F1-Score for $\mathrm { H T / H L A S T / H L A S T } _ { D }$ in synthetic data for all ensembles evaluated

The main points observed in this section were:

1. HLAST and $\mathrm { H L A S T } _ { D }$ are among the best-performing base trees on real-world data, surpassing HT and EFDT, while the original adaptive trees LAST and $\mathrm { L A S T } _ { D }$ are the weakest as ensemble base learners.

2. The proposed trees yield larger gains on datasets with more classes, where leaves stay impure longer and the adaptive splitting has more room to act, and have little efect on binary and synthetic data, where the simpler concepts are learned fast.

3. HL ${ \bf A S T } _ { D }$ is the most suitable tree for ${ \mathrm { S R P } } ,$ as HLAST lets learners built on weak feature subsets keep growing and biases $\mathrm { S R P } \mathrm { s }$ accuracy-weighted voting.

4. The proposed trees achieve the best performance overall, and ARTE with $\mathrm { H L A S T } _ { D }$ is the recommended combination, providing the strongest and most consistent results on real-world data.

## 6.2 Tree Size

Table 2 shows the mean tree size and standard deviation of ARF decision trees, which had similar behavior in other ensembles as well. As expected, LAST presented lower standard deviation compared to other base learners, showing that LAST presents low diversity as a base learner of the ensemble. $\mathrm { E F D T }$ presented lower tree size compared to Hoefding-based Trees due to the reevaluation process. However, lower tree size does not imply lower memory cost, as EFDT needs to store data at non-leaf nodes.

Table 2: Mean and standard deviation of tree size of ARF base learners
<table><tr><td>HT</td><td>EFDT</td><td>LAST</td><td> $\bf { L A S T _ { D } }$ </td></tr><tr><td> $6 8 3 , 9 1 \pm 3 8 3 6 . 1 4$ </td><td> $3 4 4 . 2 0 \pm 7 1 7 . 4 4$ </td><td> $4 0 . 7 0 \pm 2 2$ </td><td> $\overline { { 1 0 . 2 7 \ : \pm \ : 9 . 2 2 } }$ </td></tr><tr><td> $\mathbf { \overline { { H L A S T } } }$ </td><td> $\mathbf { H L A S T _ { D } }$ </td><td> $\mathbf { \overline { { E F L A S T } } }$ </td><td> $\mathbf { E F L A S T _ { D } }$ </td></tr><tr><td> $1 0 6 2 . 4 2 \pm 8 0 6 2 . 9 4$ </td><td> $\overline { { 1 1 0 6 . 7 7 \ : \pm \ : 8 1 0 9 . 6 1 } }$ </td><td> $3 4 7 . 1 5 \pm 8 1 2 . 3 7$ </td><td> $\overline { { 3 5 9 . 3 3 \pm 7 8 2 . 6 8 } }$ </td></tr></table>

Fig. 9 shows the distribution of average tree size in real-world and synthetic datasets. In real-world datasets, ARTE with $\mathrm { H L A S T } _ { D }$ presented higher median and third quartile compared to ARTE with HT, but still presented lower third quartile, median and first quartile compared to SRP with HT.

ARE and LevBag presented the largest tree sizes among all ensembles and their respective base learners. The high tree size of ARE could be explained by the input of the change detectors, which depend mostly on instances that were misclassified and used for training by the base learner. An update of the change detector with all the instances, independently of rejection, could mitigate this. In synthetic data, ARTE with $\mathrm { H L A S T } _ { D }$ also presented a higher median and third quartile than ARTE with HT and lower third quartile, median and first quartile compared to SRP with HT.

The main points observed in this section were:

1. LAST presents a low standard deviation in tree size, showing lower diversity of the base learner as a member of the ensemble.

2. ARE base learners’ change detectors receive mostly instances incorrectly classified by the base learner, and updating the change detector with all instances could help the change detector to track the performance of the base learner, independently of training or not with the instance.

3. ARTE with $\mathrm { H L A S T } _ { D }$ presents a higher tree size compared to ARTE with HT, but still a lower tree size compared to SRP with HT.

![](images/aafecb871178e5da0f7127317271c11d2b354815fa33b3e0b0f60d409dc50afc.jpg)  
Fig. 9: Distribution of average tree size of the ensembles with HT, EFDT and bestranking reported proposed method per ensemble as base learners for real-world (top) and synthetic data (down)

## 6.3 Computational Cost

Fig. 10 shows the distribution of CPU-Time in real-world and synthetic datasets. In real-world datasets, ARTE with HLAST<sub>D</sub> and ARTE with HT presented similar quantiles, while also presenting lower quantiles compared to ARF with HT and SRP with HT. The random splitting mechanism of ARTE reduces the cost of checking all possible splits, and, by simply resetting the classifier in case of concept drift detection, also lowers the cost of maintaining a background classifier. ARE with HT and $\mathrm { H L A S T } _ { D }$ presented the lowest quantiles, showing the eficiency of the instance selection approach. In synthetic datasets, the diference between ARTE with HT and $\mathrm { H L A S T } _ { D }$ is even more evident, as the number of instances of the synthetic datasets was $1 0 ^ { 6 }$ . SRP with HT presented a greater first quartile compared to ARTE with HT and HL ${ \bf \Omega } _ { \Lambda } \mathrm { S T } _ { D }$ third quartile. ARF with HT also presented higher first quartile compared to ARTE with HT and $\mathrm { H L A S T } _ { D }$ median.

![](images/fbee0bb118a161bf57c7b9dd26a3be1b7a046852fab3ba80e4e18f61f970b977.jpg)  
Fig. 10: Distribution of CPU-Time (in seconds) of the ensembles with HT, EFDT and best-ranking reported proposed method per ensemble as base learners for realworld (top) and synthetic data (down)

Fig. 11 shows the distribution of peak RAM (MB) in real-world and synthetic datasets. The same pattern as for CPU-Time follows for both real and synthetic data, where ARTE with HL $\mathrm { A S T } _ { D }$ and ARTE with HT present lower quantiles compared to ARF with HT and SRP with HT.

The main points outlined in this section were:

1. ARTE with HL $\mathrm { A S T } _ { D }$ and ARTE with HT present lower quantiles compared to ARF with HT and SRP with HT for CPU-Time and RAM-Hours in real-world and synthetic data.

2. The random splitting mechanism of ARTE reduces the cost of checking all possible splits and, by simply resetting the classifier in case of concept drift detection, also lowers the cost of maintaining a background classifier.

3. ARE presented the lowest quantiles, showing the eficiency of the regularization approach.

![](images/67ef8d8deb520812d03e34be870a20521ae08b542aae1e63ba1bf2fa321abdf4.jpg)  
Fig. 11: Distribution of Peak RAM (MB) usage throughout time of the ensembles with HT, EFDT and best-ranking reported proposed method per ensemble as base learners for real-world (top) and synthetic data (down)

## 6.4 Concept Drift

Across the synthetic streams the ensembles describe very similar F1-Score curves over time, and after each drift, marked by the vertical lines in Figures 12–15, the curves recover to a comparable level within a short window, which indicates that on these simple concepts the dominant factor is how each ensemble reacts to drift rather than which base tree it carries, in line with the small efect of the proposed trees on synthetic data reported in Section 6.1.1. What separates the ensembles is how well their voting mechanism matches the structure of the current concept, and a few consistent patterns follow from the characteristics of each generator. In AGRAWAL (Fig. 12–15) ARTE attains the lowest F1-Score, because the concepts of AGRAWAL are defined by sharp cut points on the numerical attributes and the random split-point mechanism of ARTE seldom places a split exactly on these boundaries, which is most visible in the monotone variants (Fig. 14 and 15) where the ARTE curves settle a few points below the rest.

SRP behaves in the opposite way and reaches the highest F1-Score in this dataset, since the trees that were assigned the features active in the current concept fit the cut points well and, being weighted by accuracy, take over the vote.

In SEA the ordering is reversed and SRP, together with Leveraging Bagging, produces the lowest F1-Score among the ensembles right after each drift (Fig. 12, 13 and 14), because the concept is defined by the sum of two relevant features and, when the drift changes which features are decisive, many trees built on the features that became irrelevant still carry a large vote weight inherited from their high accuracy under the previous concept, which delays the ensemble in tracking the new concept. In LED the curves almost overlap, with the exception of ARE, whose F1-Score stabilises a few points below the rest (the lower curve in Fig. 12), because the seventeen irrelevant features and the 10% label noise of LED penalise the larger trees that ARE grows, which end up splitting on noisy attributes without a matching gain in leaf purity.

![](images/ba5e215b0212b4b5f6d23563c17d863e2838466a2dc3074c38e63cd7ec64d966.jpg)  
Fig. 12: F1-Score (in %) of ensembles in datasets that simulate recurrent and abrupt changes

![](images/9bb7f7e2b6e22067bc18c11a55f5ebb7ed00e5a2b74d2d2011b5dc4db7b5cdca.jpg)  
Fig. 13: F1-Score (in %) of ensembles in datasets that simulate recurrent and gradual changes

![](images/41088a9056b2ea94955f6ad5c2d2a5fa271f45129a4c17c2cbfd0d9a5922441b.jpg)  
Fig. 14: F1-Score (in %) of ensembles in datasets that simulate abrupt changes

![](images/45638f2142ddbbcbaafc3e4a00fe843c143eaf771180be15d73fb1a888173434.jpg)  
Fig. 15: F1-Score (in %) of ensembles in datasets that simulate gradual changes

On the streams with incremental drift the F1-Score curves spread more than on the abrupt ones, since the boundary moves continuously and the ensembles never settle on a fixed concept. In RBF (Fig. 16) all ensembles stay tightly grouped at a high F1-Score except Leveraging Bagging with a Hoefding Tree base, whose curve falls further below the group as the drift speed grows, from a small gap in $\mathrm { R B F } _ { s }$ to a pronounced drop in $\mathrm { R B F } _ { f } ,$ where it stabilises around twenty points under the rest, while the remaining ensembles, including the proposed HLAST variants, track the moving centroids without a comparable loss. In HYPER (Fig. 17) the spread is wider and no single ensemble dominates, but ARE is consistently among the best, because the instances misclassified during the intermediate concepts between two drifts feed additional training to its trees and the larger trees that ARE grows follow the rotating hyperplane more smoothly, whereas several ARF and ARTE configurations lag as the speed increases. In the INSECTS streams (Fig. 18) the ensembles produce very close F1-Score curves and all of them climb steadily as more instances arrive, with ARTE slightly ahead.

![](images/e3b55cf4ed40baccd2d3e34ab6bf80a81350dc52ff13482faf34fe29ec20aa64.jpg)  
Fig. 16: F1-Score (in %) of ensembles in the RBF dataset, which simulates incremental changes

![](images/46007f4f37c16201c981169bf45bcfc881a30a98ef664294373d2f649957ca5c.jpg)  
Fig. 17: F1-Score (in %) of ensembles in the HYPER dataset, which simulates incremental changes

![](images/fc15641c92547ddb308bf4216ccc85cfc7432d6a2b711d14bb2723cc041c8189.jpg)  
Fig. 18: F1-Score (in %) of ensembles in the INSECTS datasets

The main findings outlined in this section were:

1. On the synthetic streams the ensembles describe very similar F1-Score curves that recover quickly after each drift, so the drift-reaction mechanism of the ensemble, rather than the base tree, drives the diferences, and specific properties of each generator explain the gaps that do appear.

2. ARTE struggles on datasets with sharp decision boundaries such as AGRAWAL, SRP and Leveraging Bagging lose F1-Score on the feature-dependent concepts of SEA, ARE trails on the noisy LED streams, and on the incremental drifts ARE adapts best while Leveraging Bagging with a plain Hoefding Tree degrades sharply as the speed of the RBF drift increases.

## 7 Conclusion

This paper presented the Hoefding Adaptive Splitting Trees (HLAST and EFLAST) to overcome the limitations that both Hoefding-based trees and the adaptive LAST face as base learners of data stream ensembles. By combining the periodic split attempts of a Hoefding Tree with an adaptive split driven by change detectors, the proposed trees keep growing at more informed moments, which promotes ensemble diversity while preserving responsiveness to concept drift.

Reporting F1-Score over a benchmark of real-world and synthetic streams across five ensembles, the experiments showed that HLAST and $\mathrm { H L A S T } _ { D }$ are among the best-performing base trees on real-world data, surpassing HT and EFDT, while the original adaptive trees LAST and $\mathrm { L A S T } _ { D }$ are the weakest, which confirms that pairing the adaptive mechanism with the periodic Hoefding split is what recovers and then exceeds the performance of HT. The gains concentrate on datasets with more classes, where leaves stay impure for longer and the adaptive split has more room to act, reaching up to sixteen percentage points over HT, and on the most challenging real-world streams without temporal autocorrelation, such as Rialto, LADPU and Asfault, which shows that the improvement reflects genuine concept learning rather than artifacts of easy data. On binary and synthetic streams, whose simpler concepts are learned quickly, the proposed trees leave the results essentially unchanged.

Taken together, the proposed trees achieve the best F1-Score overall, and ARTE with $\mathrm { H L A S T } _ { D }$ is the recommended combination, improving over ARTE with HT on almost every real-world dataset and yielding the strongest and most consistent results of the benchmark. The pairing between tree and ensemble, however, matters: $\mathrm { H L A S T } _ { D }$ , which monitors the class distribution purity, is the tree to couple with SRP, since the error-driven HLAST lets learners built on weak random feature subsets keep growing and biases the accuracy-weighted voting of SRP. This predictive quality comes at a competitive cost, as the analysis of tree size, CPU time and memory showed that ARTE with $\mathrm { H L A S T } _ { D }$ grows larger trees than ARTE with HT but stays smaller and cheaper than SRP with HT in both CPU time and peak RAM, since its random split points avoid evaluating every candidate cut and its reset-on-drift policy spares the cost of maintaining background learners, while ARE attains the lowest cost of all through its instance-selection mechanism.

The study also provided insights into the behavior of diferent ensembles under various drift scenarios through their F1-Score curves over time, where the ensembles recover to comparable levels shortly after each drift and the diferences that remain follow from how each approach deals with feature selection and sampling rather than from the base tree. ARTE loses F1-Score on datasets with sharp decision boundaries such as AGRAWAL, since its random split points seldom land on the well-defined cut points, while SRP and Leveraging Bagging struggle on the feature-dependent concepts of SEA, where trees built on features that became irrelevant retain a large vote weight inherited from a past concept. ARE, in turn, trails on the noisy LED streams because its larger trees split on irrelevant attributes, yet this same aggressive growth makes it the best at tracking the incremental drifts of HYPER, whereas Leveraging Bagging with a plain Hoefding Tree degrades sharply as the speed of the RBF drift increases.

The results obtained provided insights for future work. First, ARE has set a good direction for the proposal of new ensembles, aiming to obtain the best predictive quality with the lowest computational cost possible. The design of extremely eficient ensembles can be further extended to:

1. Instead of rejecting instances, apply lower sampling. For example, misclassified instances are trained with Poisson $( \lambda = 6 )$ copies, while correctly classified instances are trained with $\mathrm { P o i s s o n } ( \lambda = 1 )$ copies. The choice of varying $\lambda = 6$ and $\lambda = 1$ comes from the proposal in [21], which observed that $\lambda = 6$ provides deeper trees and higher classification accuracy, opposed to the traditional $\lambda = 1$ proposed in [16].

2. Adapt ARTE with regularization techniques.

3. Apply pre-pruning techniques to the decision tree base learners [45].

Future research directions also include adapting these methods to regression problems, tuning the hyperparameters of the base learners, and increasing ensemble diversity by selecting a diferent percentage of features at each node.

## 8 Acknowledgments

This work was financed by the Pontif´ıcia Universidade Cat´olica do Paran´a (PUCPR) through the PIBIC Master – Combined Degree program. Finally, we sincerely

thank the reviewers for their constructive feedback, which helped refine our original manuscript and strengthen the assessment of our methods.

## References

[1] Bifet, A., Holmes, G., Pfahringer, B., Kranen, P., Kremer, H., Jansen, T., Seidl, T.: Moa: Massive online analysis, a framework for stream classification and clustering. In: Proceedings of the First Workshop on Applications of Pattern Analysis, pp. 44–50 (2010). PMLR

[2] Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., Zhang, G.: Learning under concept drift: A review. IEEE Transactions on Knowledge and Data Engineering 31(12), 2346–2363 (2019) https://doi.org/10.1109/TKDE.2018.2876857

[3] Domingos, P., Hulten, G.: Mining high-speed data streams. In: Proceedings of the Sixth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. KDD ’00, pp. 71–80. Association for Computing Machinery, New York, NY, USA (2000). https://doi.org/10.1145/347090.347107

[4] Hoefding, W.: Probability inequalities for sums of bounded random variables. The collected works of Wassily Hoefding, 409–426 (1963)

[5] Assis, D.N., Barddal, J.P., Enembreck, F.: Just change on change: Adaptive splitting time for decision trees in data stream classification. In: Proceedings of ACM SAC Conference (SAC’24). ACM SAC’ 24. Association for Computing Machinery, New York, NY, USA (2024)

[6] Nowak Assis, D., Barddal, J.P., Enembreck, F.: Behavioral insights of adaptive splitting decision trees in evolving data stream classification. Knowl Inf Syst (2025)

[7] Manapragada, C., Webb, G.I., Salehi, M.: Extremely fast decision tree. In: Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. KDD ’18, pp. 1953–1962. Association for Computing Machinery, New York, NY, USA (2018). https://doi.org/10.1145/3219819. 3220005

[8] Gama, J., Zliobaite, I., Bifet, A., Pechenizkiy, M., Bouchachia, A.: A survey on <sup>ˇ</sup> concept drift adaptation. ACM Comput. Surv. 46(4) (2014) https://doi.org/10. 1145/2523813

[9] Aguiar, G.J., Cano, A.: A comprehensive analysis of concept drift locality in data streams. Knowledge-Based Systems 289, 111535 (2024) https://doi.org/10.1016/ j.knosys.2024.111535

[10] Quinlan, J.R.: C4.5: Programs for machine. Morgan Kaufmann Publishers, 340 Pine Street, 6th Floor San Francisco, CA 94104 USA (1992)

[11] Breiman, L.: Classification and regression trees. Wadsworth Statistics, Wadsworth, Belmont, CA, (1984)

[12] Gama, J., Rocha, R., Medas, P.: Accurate decision trees for mining high-speed data streams. KDD ’03, pp. 523–528. Association for Computing Machinery, New York, NY, USA (2003). https://doi.org/10.1145/956750.956813

[13] Holmes, G., Kirkby, R., Pfahringer, B.: Stress-testing hoefding trees. In: Knowledge Discovery in Databases: PKDD 2005, pp. 495–502. Springer, Berlin, Heidelberg (2005)

[14] Holmes, G., Kirkby, R., Pfahringer, B.: Stress-testing hoefding trees. In: Jorge, A.M., Torgo, L., Brazdil, P., Camacho, R., Gama, J. (eds.) Knowledge Discovery in Databases: PKDD 2005, pp. 495–502. Springer, Berlin, Heidelberg (2005)

[15] Manapragada, C., Gomes, H.M., Salehi, M., Bifet, A., Webb, G.I.: An eager splitting strategy for online decision trees in ensembles. In: Data Min Knowl Disc, vol. 36, pp. 566–619 (2022)

[16] Oza, N.C., Russell, S.J.: Online bagging and boosting. In: Proceedings of the Eighth International Workshop on Artificial Intelligence and Statistics. Proceedings of Machine Learning Research, vol. R3, pp. 229–236. PMLR, ??? (2001)

[17] Breiman, L.: Bagging predictors. In: Machine Learning, vol. 24, pp. 123–140 (1996). Springer

[18] Freund, Y., Schapire, R.E.: Experiments with a new boosting algorithm. In: Proceedings of the Thirteenth International Conference on International Conference on Machine Learning. ICML’96, pp. 148–156. Morgan Kaufmann Publishers Inc., San Francisco, CA, USA (1996)

[19] Bifet, A., Holmes, G., Pfahringer, B., Kirkby, R., Gavald\`a, R.: New ensemble methods for evolving data streams. KDD ’09, pp. 139–148. Association for Computing Machinery, New York, NY, USA (2009). https://doi.org/10.1145/1557019. 1557041

[20] Bifet, A., Gavald\`a, R.: Learning from time-changing data with adaptive windowing, vol. 7 (2007). https://doi.org/10.1137/1.9781611972771.42

[21] Bifet, A., Holmes, G., Pfahringer, B.: Leveraging bagging for evolving data streams. In: Machine Learning and Knowledge Discovery in Databases, pp. 135–150. Springer, Berlin, Heidelberg (2010)

[22] Dietterich, T.G., Bakiri, G.: Solving multiclass learning problems via errorcorrecting output codes. J. Artif. Int. Res. 2(1), 263–286 (1995)

[23] Barros, R.S.M.d., Carvalho Santos, S., Gon¸calves J´unior, P.M.: A boosting-like online learning ensemble. In: 2016 International Joint Conference on Neural Networks (IJCNN), pp. 1871–1878 (2016). https://doi.org/10.1109/IJCNN.2016. 7727427

[24] Gomes, H.M., Bifet, A., Read, J., Barddal, J.P., Enembreck, F., Pfharinger, B., Holmes, G., Abdessalem, T.: Adaptive random forests for evolving data stream classification. Machine Learning 106, 1469–1495 (2017)

[25] Breiman, L.: Random forests. Mach. Learn. 45(1), 5–32 (2001) https://doi.org/ 10.1023/A:1010933404324

[26] Gomes, H.M., Read, J., Bifet, A.: Streaming random patches for evolving data stream classification. In: 2019 IEEE International Conference on Data Mining (ICDM), pp. 240–249 (2019). https://doi.org/10.1109/ICDM.2019.00034

[27] Louppe, G., Geurts, P.: Ensembles on random patches. In: Flach, P.A., De Bie, T., Cristianini, N. (eds.) Machine Learning and Knowledge Discovery in Databases, pp. 346–361. Springer, Berlin, Heidelberg (2012)

[28] Paim, A.M., Enembreck, F.: Adaptive regularized ensemble for evolving data stream classification. Pattern Recognition Letters 180, 55–61 (2024) https://doi. org/10.1016/j.patrec.2024.02.026

[29] Paim, A.M., Enembreck, F.: Adaptive random tree ensemble for evolving data stream classification. Knowledge-Based Systems 309, 112830 (2024) https://doi. org/10.1016/j.knosys.2024.112830

[30] Geurts, P., Ernst, D., Wehenkel, L.: Extremely randomized trees. Machine Learning 63, 3–42 (2005)

[31] Kuncheva, L.I., Whitaker, C.J.: Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy. In: Machine Learning, vol. 51, pp. 181–207 (2003). Springer

[32] Souza, V.M., Reis, D.M., Maletzke, A.G., Batista, G.E.: Challenges in benchmarking stream learning algorithms with real-world data. Data Mining and K1nowledge Discovery 34, 1805–1858 (2020)

[33] Cano, A., Krawczyk, B.: Kappa updated ensemble for drifting data stream mining. Machine Learning 109, 175–218 (2019)

[34] Agrawal, R., Imielinski, T., Swami, A.: Database mining: A performance perspective. IEEE Trans. on Knowl. and Data Eng. 5(6), 914–925 (1993) https: //doi.org/10.1109/69.250074

[35] Street, W.N., Kim, Y.: A streaming ensemble algorithm (sea) for large-scale classification. In: Proceedings of the Seventh ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. KDD ’01, pp. 377–382. Association for Computing Machinery, New York, NY, USA (2001)

[36] Hulten, G., Spencer, L., Domingos, P.: Mining time-changing data streams. In: Proceedings of the Seventh ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. KDD ’01, pp. 97–106. Association for Computing Machinery, New York, NY, USA (2001)

[37] Gunasekara, N., Pfahringer, B., Gomes, H., Bifet, A.: Gradient boosted trees for evolving data streams. Machine Learning 113, 3325–3352 (2024)

[38] Cano, A., Krawczyk, B.: Rose: robust online self-adjusting ensemble for continual learning on imbalanced drifting data streams. Machine Learning 111, 2561–2599 (2022)

[39] Aguiar, G., Cano, A., Krawczyk, B.: A survey on learning from imbalanced data streams: taxonomy, challenges, empirical study, and reproducible experimental framework. Machine Learning 113, 4165–4243 (2024)

[40] Gama, J., Sebasti˜ao, R., Rodrigues, P.P.: On evaluating stream learning algorithms. Machine Learning 90, 317–346 (2013)

[41] Garc´ıa, S., Herrera, F.: An extension on “statistical comparisons of classifiers over multiple data sets” for all pairwise comparisons. Journal of Machine Learning Research 9(89), 2677–2694 (2008)

[42] Benavoli, A., Corani, G., Mangili, F.: Should we really use post-hoc tests based on mean-ranks? Journal of Machine Learning Research 17(5), 1–10 (2016)

[43] Demˇsar, J.: Statistical comparisons of classifiers over multiple data sets. The Journal of Machine learning research 7, 1–30 (2006)

[44] Fr´ıas-Blanco, I., Campo-Avila, J.d., Ramos-Jim´enez, G., Morales-Bueno, R.,<sup>´</sup> Ortiz-D´ıaz, A., Caballero-Mota, Y.: Online and non-parametric drift detection methods based on hoefding’s bounds. IEEE Transactions on Knowledge and Data Engineering 27(3), 810–823 (2015) https://doi.org/10.1109/TKDE.2014.2345382

[45] Barddal, J.P., Enembreck, F.: Learning regularized hoefding trees from data streams. SAC ’19, pp. 574–581. Association for Computing Machinery, New York, NY, USA (2019). https://doi.org/10.1145/3297280.3297334
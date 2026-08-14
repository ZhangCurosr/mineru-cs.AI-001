# CABS+: Efficient and Scalable Model Merging via Conflict-Aware Sparsification and Adaptive Weight Allocation

Yuchen Liu, Zongzhen Yang, Binhang Qi, Hailong Sun\*, Xiang Gao

Abstract—Model merging has recently attracted significant attention as a promising paradigm for constructing unified multitask models without requiring additional retraining. However, due to the widespread presence of parameter conflicts and knowledge interference across tasks, the performance of merged models is often unsatisfactory. To address these challenges, prior work introduced the Conflict-Aware and Balanced Sparsification (CABS) method, which reduces parameter interference through structured pruning and sequential masking. However, CABS, like the mainstream model merging methods, relies on grid search to determine scaling coefficients, leading to exponential time complexity, limiting its practical applicability. Meanwhile, its optimization objective is prone to being dominated by high performance tasks, resulting in suboptimal overall performance.

To overcome these limitations, we extend CABS and propose the enhanced method, termed CABS+. Specifically, the Adaptive Weight Allocation (AWA) strategy optimizes merging coefficients via a gradient-free search scheme to reduce time complexity, while the asymmetric fitness function promotes more comprehensive performance gains across tasks. Moreover, to better understand the key factors influencing model merging performance, we conduct the systematic empirical study and propose a new metric, Relative Synergy Score (RSS), to quantify model mergeability, providing practical guidance for model selection in real-world applications. We compare CABS+ with state-ofthe-art model merging methods, including CABS, AdaMerging, and WUDIMerging, across 27 datasets and 5 models covering large language models, small-scale language models, and vision models. Extensive experiments across diverse tasks and model scales verify the effectiveness and efficiency of CABS+. Compared with AdaMerging and WUDIMerging, CABS+ achieves overall performance improvements of 16.97% and 12.93%, exhibits stronger stability and robustness under varying task numbers and model architectures, while requiring less than 25% of the GPU memory used by AdaMerging and achieving nearly a 4× speedup in merging time compared with WUDIMerging.

Index Terms—Model Merging, Task Vectors, Structured Pruning, Model Mergeability.

## I. INTRODUCTION

W <sup>ITH</sup> <sup>the</sup> <sup>widespread</sup> <sup>adoption</sup> <sup>of</sup> <sup>the</sup> <sup>pretraining</sup> <sup>fine-</sup> tuning paradigm, a large number of task-specific model checkpoints have been accumulated in the open-source community. However, directly deploying multiple specialized models for different tasks incurs substantial storage and maintenance overhead, which is particularly undesirable in resourceconstrained or real-world application scenarios [1]. Although multi-task learning can partially alleviate this issue, it typically relies on joint training over multiple tasks, resulting in considerable computational cost. Moreover, it often requires access to aggregated multi-task data, which may raise concerns regarding data sharing and privacy [2]. To address these limitations, model merging has emerged as an efficient alternative paradigm. The key idea is to combine multiple expert models in the parameter space to construct a unified multi-task model, without requiring access to multi-task training data or additional retraining.

In recent years, model merging has been extensively stud ied in the deep learning community and has demonstrated promising practical utility across various domains [3], [4]. In the context of large language models (LLMs), task vector approaches have emerged as a dominant paradigm [5]– [9]. However, due to the widespread presence of parameter conflicts and knowledge interference among expert models, the merged model often exhibits inferior performance on individual tasks compared to its corresponding single-task counterparts [10]. To address this issue, recent studies have explored model merging from the perspective of parameter space geometry. In particular, sparsification of task vectors has been shown to effectively reduce parameter conflicts and has emerged as a strong paradigm for improving merging performance [11]–[15]. Such sparsification-based methods are closely related to model pruning techniques, where magnitudebased pruning [16] is commonly used to retain important parameters while removing redundant updates. However, prior studies [12] have observed that, in the context of model merging, magnitude-based pruning can be less effective than random sparsification. This is primarily because it tends to preserve highly overlapping and unevenly distributed parameters, which may exacerbate inter-task interference and limit the final performance.

To address the limitations of existing sparsification-based model merging methods, we previously proposed the Conflict-Aware and Balanced Sparsification method (CABS, accepted by ICML 2025). CABS performs cross-layer uniform pruning to maintain the balanced distribution of information within each task vector, and further incorporates a sequential masking mechanism to encourage separation among task vectors in the parameter space, thereby reducing potential conflicts. However, CABS relies on grid search to determine task scaling coefficients, in which time complexity grows exponentially with the number of tasks to be merged, leading to high prac tical overhead. In addition, its optimization objective focuses on the aggregated performance across all tasks, which tends to bias the resulting scaling coefficients toward a subset of high-performing tasks, thereby weakening performance on the remaining ones. Existing approaches for optimizing merging coefficients can be broadly categorized into two groups. The first introduces additional modules to model data features or meta-information, typically incurring extra training cost and computational overhead [17], [18]. The second category determines scaling coefficients using test data without requiring additional training data, as exemplified by AdaMerging [8]. However, such methods often demand substantial GPU memory when applied to large-scale language models with billions of parameters and long-context inputs, with memory consumption growing linearly with the number of models to be merged, which makes them less feasible on widely accessible hardware platforms (e.g., V100 GPUs) and limits the scalability of model merging in practical deployments.

To address the aforementioned limitations, this work extends CABS and proposes an enhanced framework, termed CABS+. Specifically, CABS+ introduces the Adaptive Weight Allocation (AWA) strategy, which performs efficient optimization via the gradient-free search mechanism. By eliminating the need for gradient computation and backpropagation, AWA reduces the memory overhead of the merging process to the level of standard inference, making it feasible to merge billion parameter models on a single GPU with limited memory. Meanwhile, AWA incorporates the boundary-constrained search space along with the asymmetric fitness evaluation strategy to account for scale differences across task-specific losses. Compared to conventional black-box optimization methods, this design facilitates faster convergence and mitigates the tendency of the optimization process to be dominated by highloss tasks. In addition, AWA is naturally compatible with the CABS pruning strategy. Since task vectors have already undergone conflict reduction during the CABS pruning stage, the resulting optimization landscape for merging coefficients becomes smoother with reduced non-convexity. This property provides favorable conditions for efficient search and stable convergence of AWA, thereby improving the efficiency and reliability of the optimization process. Furthermore, to fill the gap in the community regarding the lack of systematic exploration into key factors affecting model merging, we conduct the multi-dimensional empirical study on model mergeability and propose Relative Synergy Score (RSS) to quantify it. Through extensive analysis, multiple key factors influencing mergeability are identified and validated, including task heterogeneity, data distribution heterogeneity, training configurations, model architecture, and model scale, thereby providing practical guidelines for model selection prior to merging.

Extensive experimental results show that CABS+ achieves superior performance across various settings while also offering notable efficiency advantages. In terms of performance, CABS+ demonstrates stronger robustness to variations in model architectures and the number of tasks, and achieves higher average performance than state-of-the-art methods, outperforming AdaMerging and WUDIMerging by 16.97% and 12.93%, respectively. In representative large-model efficiency experiments on Mistral, CABS+ requires less than 25% of the GPU memory used by AdaMerging and achieves a nearly 4× speedup in merging time compared with WUDIMerging.

The main contributions of this work are summarized as follows:

• We propose CABS+, which extends CABS to address its high time complexity, reduces excessive GPU memory consumption of comparable methods, and mitigates the tendency of optimization to be dominated by tasks with larger loss scales, thereby achieving more comprehensive performance improvements across tasks.

• We conduct the systematic investigation of model mergeability and identify six critical factors that affect model merging performance. To support this analysis, we propose the new metric, termed Relative Synergy Score, to quantify model mergeability, thereby providing practical guidelines for model selection before merging.

• Extensive experiments across 27 tasks and 5 models of varying scales demonstrate that CABS+ outperforms AdaMerging and WUDIMerging by 16.97% and 12.93% in overall performance, respectively, and exhibits advantages in both memory usage and merging efficiency.

Our source code is available at https://anonymous.4open.   
science/r/CABS Plus-70C1.

## II. RELATED WORK

## A. Data-Free Model Merging

The most basic merging strategy is simple averaging [19], [20], which directly computes the mean of corresponding parameters across models. However, this approach often fails to account for task-specific parameter variations, leading to suboptimal performance. To achieve more refined weight allocation, Fisher Merging [21] introduces the Fisher information matrix to assess the importance of each expert model’s parameters and assign corresponding merging weights accordingly. Similarly, RegMean [5] adopts an alternative approach by minimizing the prediction discrepancy between the merged model and individual task models to merge. Subsequently, Task Arithmetic [7] proposes an innovative method based on task vectors, defined as the difference between the finetuned model and the pretrained model parameters, and employs scaling coefficients to flexibly control the contribution of each task vector during the merging process. WUDI-Merging [9] considers the geometric relationships of task vectors in parameter space, characterizing the linear subspace structure and suppressing inter-task interference through the optimization process. Additional approaches focus on aligning the losses between the merged model and the task-specific models [22], [23]. Although such methods can effectively merge models, significant parameter redundancy and sign conflicts often exist between expert models, which can severely degrade the performance of the merged model.

## B. Test-Time Adaptive Model Merging

Test-Time adaptive methods employs certain test data to resolve conflicts between tasks. For instance, AdaMerging [8] minimizes the unsupervised entropy on test samples as an objective and dynamically learns the merging coefficients via gradient descent. Representation Surgery [24] seeks to minimize the discrepancy between the merged model and individual expert models in feature space, thereby effectively alleviating representation bias during feature extraction. However, such methods frequently incur substantial GPU memory usage when applied to large scale models, thereby limiting their feasibility in resource-constrained environments.

## C. Sparsification-Based Model Merging

Sparsification-based methods primarily target the removal of redundant or conflict parameters through sparsification. TIES-Merging [13] preserves critical knowledge by pruning low-magnitude redundant parameters and resolving sign conflicts. Consensus Merging [25] further enhances performance by eliminating weights that negatively impact the merging process, often referred to as selfish or catastrophic weights. DARE [12], inspired by the Dropout [26] mechanism, randomly discards a large portion of parameters and rescales the remaining weights, demonstrating the potential of sparsity to alleviate conflicts. These sparsification strategies, which have shown remarkable effectiveness in model merging, are closely related to conventional model compression and pruning techniques used to reduce computational cost while retaining core performance. Notably, magnitude-based pruning assumes that parameters with larger absolute values carry more critical information [27]–[29]. However, in the context of model merging, directly applying magnitude pruning can lead to highly uneven weight distributions, exacerbating inter-task conflicts rather than mitigating them.

To address this imbalance in weight distribution, structured pruning techniques were incorporated into the model merging process. The Conflict-Aware and Balanced Sparsification method (CABS, accepted by ICML 2025), employs sequential pruning to generate masks that eliminate parameter overlap among task vectors, while integrating $n : m$ pruning [30], [31] to ensure globally balanced weight distributions. This approach significantly improves both the stability and performance of multi-task model merging. However, CABS relies on the grid search strategy to determine the scaling coefficients, resulting in prohibitively high time complexity and limiting its practical applicability. Moreover, its optimization objective typically aggregates the performance of all tasks in a simple manner, which can lead to uneven performance improvements across tasks and suboptimal overall outcomes.

## III. METHODOLOGY

To address the above issues, we propose CABS+, whose overall pipeline is illustrated in Fig. 1. Compared with CABS, CABS+ mainly differs in the introduction of a gradient-free adaptive weight allocation (AWA) strategy, which reduces the time complexity of the previous grid-search approach while improving model merging performance. The detailed procedure is given in Algorithm 1.

## A. Conflict-Aware Sparsification (CA)

Sequential pruning and masking. CA mainly adopts a sequential pruning strategy to avoid parameter overlap between task vectors, thereby effectively eliminating parameter conflicts during merging. Specifically, task vector $\tau _ { A }$ is first pruned, and the positions of the parameters to be retained are marked to generate a mask, denoted as $m a s k _ { A }$ . Subsequently, this mask is used to guide the pruning of the subsequent task vector $\tau _ { B }$ , so as to prevent parameter overlap at the same positions. More specifically, in order to eliminate the overlap with $\tau _ { A }$ , the parameters retained in task vector $\tau _ { B }$ are computed as follows before the subsequent pruning:

$$
\tau _ { \mathrm { B \ r e m a i n i n g } } = \tau _ { B } \odot ( 1 - \mathrm { m a s k } _ { A } ) .\tag{1}
$$

Afterward, the task vector τ<sub>B remaining</sub> is pruned to generate the corresponding new $m a s k _ { B }$ . Then, the pruned task vector $\tilde { \tau } _ { B }$ is merged with the previously pruned task vector $\tilde { \tau } _ { A }$ to obtain a model parameter matrix without overlap. If multiple tasks exist, the same procedure is applied.

Minimizing overlap under low sparsity. When the sum of the retained parameter ratios of all task vectors exceeds 1, overlap cannot be completely avoided. For example, if both task A and task B retain 60% of the parameters after pruning, only 40% of the parameter space in task vector $\tau _ { B }$ can remain fully non-overlapping with $\tau _ { A }$ . The remaining 20% parameters of $\tau _ { B }$ are therefore selected from the overlapping region. For this unavoidable overlap, sign selection and parameter averaging strategies similar to TIES-Merging [13] are adopted to preserve the performance of both tasks as much as possible. This overlap-restricted strategy consistently provides stable performance improvements across different tasks and model settings. A detailed derivation showing that the nonoverlapping masks generated by CA make the pruned task vectors orthogonal in the Frobenius inner product, eliminate the cross term in the merged update norm, and enable independent scaling of task-vector contributions is provided in the supplementary material.

## B. Balanced Sparsification (BS)

The weight matrix of the model is first divided into m non-overlapping blocks of consecutive weights. Within each block, only the n parameters with the largest absolute values are retained, and the rest are pruned. This block-wise local pruning strategy is applied uniformly across all layers of the model, so that the retained weights are distributed more evenly throughout the network rather than concentrated in one or two layers, thereby preventing severe parameter conflicts when multiple task models are merged.

It is worth noting that the proposed BS strategy differs fundamentally from traditional n:m pruning in terms of its objective. Conventional n:m pruning primarily aims to reduce computational and memory costs through structured sparsity for model compression and inference acceleration. In contrast, BS is designed to mitigate parameter conflicts during model merging by enforcing more balanced feature distributions across different task vectors with higher sparsity ratios. Consequently, unlike conventional sparse models, the final merged model produced by BS remains dense. This strategy prioritizes merging performance and task compatibility rather than inference efficiency.

![](images/d70dc778426cabd2c9b2c253669d0855e8c19a6f53fdd8361ea0d67470a631f2.jpg)

![](images/2b9397890932992258e28800d6d94e9608467c03948bc34ee68877f0df40d593.jpg)  
(a)Pruning task vectors with CABS

![](images/47c52a5d52da30668f30b1dc27e1ebe6f3efaa35af6972cd669b03eaa23e7b11.jpg)

(b)Adaptive weight allocation strategy (AWA)  
![](images/bd60788a29d5b123ff9818cb82d7021eb06890f94526ed1b6c6c89f1ad134ed5.jpg)  
(c)Merging parameters with CABS+  
Fig. 1. Illustration of the overall framework of CABS+. (a) Conflict-Aware and Balanced Sparsification; (b) Adaptive Weight Allocation strategy; (c) Overal pipeline of CABS+.

## C. Adaptive Weight Allocation (AWA)

CA effectively ensures orthogonality among task vectors in the parameter space to minimize conflicts, while BS guarantees the uniform retention of each task vector’s information to enhance stability. However, achieving better model merging performance still faces a key challenge: How to determine the scaling coefficient for each task vector? Even if the task vectors are already orthogonal and evenly distributed in the parameter space, the scaling coefficients λ that determine their contributions must still be set when merging them into the base model. Improper allocation of these coefficients can significantly degrade the performance of the merged model.

In the previous conference version, we used grid search to determine these coefficients. However, this approach has two critical limitations. First, its computational time complexity grows exponentially with the number of tasks being merged. Second, the objective focuses solely on the sum of performance across all tasks. This often causes the selected coefficients to be dominated by a few high-performing tasks and prevents achieving optimal merging performance. Existing methods for optimizing merging coefficients, such as AdaMerging, mostly rely on gradient-based optimization. This requires constructing the full computation graph during the forward pass to enable backpropagation through the chain rule. In the context of model merging, multiple task vectors must be simultaneously incorporated into the computation graph, as they jointly participate in the optimization of merging coefficients. This requires the GPU to maintain not only the base model parameters but also all task-specific parameter updates at the same time. To compute loss.backward(), the intermediate results of each layer must be stored. For large language models with billions of parameters and contexts of several thousand tokens, storing the intermediate activations needed for gradient computation consumes enormous GPU memory. When multiple task vectors are involved, this memory overhead is further amplified, as each task contributes additional parameters and corresponding computation paths in the graph. Specifically, memory complexity grows linearly with the number of task vectors, as well as model depth L and sequence length T, leading to substantial memory overhead in model merging scenarios. Even with a batch size of 1, this often exceeds the memory capacity of a single highperformance GPU such as the A100. This memory bottleneck severely limits the scalability of model merging techniques.

To address these issues, we propose a new computationally efficient adaptive weight allocation strategy, which is a gradient-free automatic optimization mechanism improved based on the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) [32]. Since this method does not require gradient computation or backpropagation, the GPU memory usage is strictly limited to the level of the inference stage, making it possible to merge billion parameter scale models on consumerlevel GPUs such as the V100. At the same time, this strategy can be well combined with the CA and BS methods described above. In general, evolutionary algorithms tend to suffer from slow convergence. However, because the task vectors can minimize task conflicts after the CABS pruning process, the optimization landscape of the merging coefficients becomes smoother and the non-convexity is significantly reduced. This allows the proposed AWA optimization strategy to perform the search more efficiently and achieve faster convergence, thereby improving the stability and reliability of the overall optimization process.

Algorithm 1 CABS+   
Input: Task vectors $\tau _ { A } , \tau _ { B } ,$ , base model $W _ { \mathrm { b a s e } } ,$ sparsity level   
$n , m ,$ feasible bounds $[ l , u ]$ , population size $\dot { K } .$ , max   
generations G   
Output: Merged model parameters $W _ { \mathrm { f i n a l } }$   
1: Phase 1: Conflict-Aware and Balanced Sparsification (CABS)   
2: Apply n : m pruning to $\tau _ { A }$ and compute mask<sub>A</sub>   
3: τ<sub>B</sub> <sub>remaining</sub> $= \bar { \tau } _ { B } \odot ( \mathrm { 1 - m a s k } _ { A } )$ to eliminate overlap with $\tau _ { A }$   
4: Apply n : m pruning to τ<sub>B remaining</sub> to compute mask<sub>B</sub>   
5: Define pruned vectors: $\tilde { \tau } _ { A } = \mathrm { m a s } \dot { \mathbf { k } } _ { A } \odot \tau _ { A } \mathrm { , ~ } \tilde { \tau } _ { B } = \mathrm { m a s } \mathbf { k } _ { B } \odot \tau _ { B }$   
6: Phase 2: Adaptive Weight Allocation (AWA)   
7: Initialize: covariance $C ^ { ( 0 ) } = I ,$ step size $\sigma ^ { ( 0 ) } = 0 . 0 5$   
8: Compute base loss $L _ { \mathrm { b a s e } , t }$ for each task t at initial state $\lambda _ { 0 } = \mathbf { 1 }$   
9: for $\overset { \vartriangle } { \boldsymbol { g } } = \boldsymbol { 0 }$ to $G - 1$ do   
10: for $k = 1$ to $K$ do   
11: Sample coefficients: $\lambda _ { k } ^ { ( g ) } \sim m ^ { ( g ) } + \sigma ^ { ( g ) } \mathcal { N } ( 0 , C ^ { ( g ) } )$   
12: Bound constraint P : $\ddot { \lambda } _ { k , i } ^ { ( g ) } = \operatorname* { m i n } ( u , \operatorname* { m a x } ( \dot { l } , \lambda _ { k , i } ^ { ( g ) } ) )$   
13: Construct candidate model:   
$W _ { k } = W _ { \mathrm { b a s e } } + \lambda _ { k , A } ^ { ( g ) } \tilde { \tau } _ { A } + \lambda _ { k , B } ^ { ( g ) } \tilde { \tau } _ { B }$   
14: Evaluate asymmetric fitness: $\begin{array} { r } { \bar { F } ( \lambda _ { k } ^ { ( g ) } ) = \sum _ { t } f _ { t } ( \lambda _ { k } ^ { ( g ) } ) } \end{array}$   
15: end for   
16: Sort candidates by $F ( \lambda _ { k } ^ { ( g ) } )$ and select top $\mu = K / 2$ optimal   
samples   
17: Update $m ^ { ( g + 1 ) }$ using logarithmically weighted average of top   
µ samples   
18: Update $\sigma ^ { ( g + 1 ) }$ and $C ^ { ( g + 1 ) }$ using evolution paths $p _ { \sigma }$ and $p _ { c }$   
19: end for   
20: Extract optimal coefficients: $\lambda ^ { * } = \arg$ min $F ( \lambda )$   
21: Merge the pruned vectors with the base model using optimal   
coefficients:   
$W _ { \mathrm { f i n a l } } = W _ { \mathrm { b a s e } } + \lambda _ { A } ^ { * } \times \tilde { \tau } _ { A } + \lambda _ { B } ^ { * } \times \tilde { \tau } _ { B }$   
22: Return $W _ { \mathrm { f i n a l } }$

Gradient-free sampling and boundary constraints. AWA formulates the determination of the scaling coefficients as an optimization problem in a continuous space. Specifically, K candidate solutions are first sampled from a multivariate normal distribution:

$$
\lambda _ { k } ^ { ( g ) } \sim { \bf m } ^ { ( g ) } + \sigma ^ { ( g ) } \mathcal { N } ( { \bf 0 } , { \bf C } ^ { ( g ) } ) , \quad k = 1 , . . . , K .\tag{2}
$$

Here, $g$ denotes the iteration index, $m ^ { ( g ) }$ is the mean vector of the current distribution, and $C$ is the covariance matrix of the algorithm with the initial value set to the identity matrix

I. The step size is denoted by σ, whose initial value is set to 0.05 by default. The coefficient vector is denoted by $\lambda ,$ whose dimension D equals the number of tasks to be merged.

To ensure that the scaling coefficients remain within a reasonable range rather than exploring an unbounded space, we introduce boundary constraints based on the CMA-ES algorithm. The feasible region $\Omega = \lbrace \mathbf { v } \in \mathbb { R } ^ { D } \ \vert \ l \leq v _ { i } \leq$ u, $\forall i = 1 , \ldots , D \}$ is defined in a D dimensional space, where the lower bound is set to $\scriptstyle { l = 0 . 1 }$ and the upper bound is set to $u { = } 2$ . At the g-th iteration, for each $\lambda _ { k } ^ { ( g ) }$ , the projection function $P _ { \Omega }$ is applied to obtain the constrained coefficient $\lambda _ { k } ^ { ( g ) }$

$$
\lambda _ { k } ^ { ( g ) } = \mathcal { P } _ { \Omega } \left( \mathbf { m } ^ { ( g ) } + \sigma ^ { ( g ) } \mathcal { N } ( \mathbf { 0 } , \mathbf { C } ^ { ( g ) } ) \right) .\tag{3}
$$

The explicit expression in component-wise form is given as:

$$
\left[ \lambda _ { k } ^ { \left( g \right) } \right] _ { i } = \operatorname* { m i n } \left( u , \operatorname* { m a x } \left( l , \left[ \lambda _ { k } ^ { \left( g \right) } \right] _ { i } \right) \right) , \quad i = 1 , . . . , D .\tag{4}
$$

Asymmetric fitness evaluation for handling scale differences. In the standard CMA-ES strategy, the optimization objective is to minimize the sum of the absolute loss values over all tasks:

$$
J _ { \mathrm { t r a d } } ( \lambda ) = \sum _ { t = 1 } ^ { T } L _ { t } \left( \theta ( \lambda ) \right) .\tag{5}
$$

where θ denotes the merged model. However, since different tasks have different loss scales, the optimizer tends to be dominated by tasks with larger losses while ignoring tasks with smaller losses. According to our experimental observations, this typically manifests as improving the performance of some tasks at the cost of degrading the performance of others during the optimization process.

To address this question, we introduce the asymmetric penalty mechanism in AWA. For each individual $\dot { \lambda } _ { k } ^ { ( g ) }$ , the fitness function $F \left( \lambda _ { k } ^ { \left( g \right) } \right)$ is computed. Specifically, before the search begins, the baseline loss of each task under the initial state $\lambda _ { k } ^ { ( g ) }$ is first computed:

$$
L _ { \mathrm { b a s e } } ^ { ( t ) } = \mathcal { L } _ { t } \left( \theta ( \lambda ^ { ( 0 ) } ) \right) .\tag{6}
$$

For any candidate solution λ, the normalized relative change with respect to the baseline is computed to eliminate scale differences across tasks:

$$
\Delta _ { t } ( \lambda ) = \frac { L _ { t } ( \theta ( \lambda ) ) - L _ { \mathrm { b a s e } } ^ { ( t ) } } { L _ { \mathrm { b a s e } } ^ { ( t ) } } .\tag{7}
$$

Subsequently, the asymmetric penalty function $f _ { t }$ is constructed, which applies a large penalty to tasks whose loss has increased, meaning that they have experienced performance drop, for example with $\alpha { = } 1 0 0$ and $\beta { = } 1$

$$
f _ { t } ( \lambda ) = \left\{ { \begin{array} { l l } { \alpha \cdot \Delta _ { t } ( \lambda ) } & { { \mathrm { i f } } \Delta _ { t } ( \lambda ) > 0 . } \\ { \beta \cdot \Delta _ { t } ( \lambda ) } & { { \mathrm { i f } } \Delta _ { t } ( \lambda ) \leq 0 . } \end{array} } \right.\tag{8}
$$

The final fitness function is obtained as the sum of the asymmetric scores across all tasks:

$$
F ( \lambda ) = \sum _ { t = 1 } ^ { T } f _ { t } ( \lambda )\tag{9}
$$

Population selection and mean update. The samples are ranked according to $F ( \lambda )$ , and the top $\mu$ samples $( \mu = K / 2 )$ are selected to update the distribution parameters $m ^ { ( g + 1 ) }$ $\sigma ^ { ( g + 1 ) } , \ C ^ { ( g + 1 ) }$ in order to maximize the likelihood estimate. The new distribution mean $m ^ { ( g + 1 ) }$ is computed as the weighted average of the top $\mu$ individuals. Let $\lambda _ { 1 : K } ^ { ( g ) } , . . . , \lambda _ { \mu : K } ^ { ( g ) }$ denote the top $\mu$ individuals in the current iteration, then:

$$
{ { \bf m } ^ { ( g + 1 ) } } = \sum _ { i = 1 } ^ { \mu } w _ { i } \lambda _ { i : K } ^ { ( g ) } .\tag{10}
$$

where $w _ { i }$ are the normalized weights $( w _ { 1 } \ > \ \cdots > w _ { \mu } \ >$ $0 , \Sigma w _ { i } = 1 )$ , representing the shift of the search center. To ensure that higher-ranked individuals contribute more to the mean update, the weights are typically defined as:

$$
w _ { i } ^ { \prime } = \ln ( \mu + 0 . 5 ) - \ln ( i ) , \quad \mathrm { f o r \ } i = 1 , . . . , \mu .\tag{11}
$$

They are then normalized so that their sum equals 1:

$$
w _ { i } = \frac { w _ { i } ^ { ' } } { \sum _ { j = 1 } ^ { \mu } w _ { j } ^ { ' } } .\tag{12}
$$

where $w _ { 1 }$ is the largest and exerts the strongest influence in pulling the mean $m$ toward its position, while $w _ { \mu }$ is the smallest, and all weights less than $w _ { \mu }$ are set to zero, contributing no further updates.

Step size control and covariance matrix adaptation. AWA uses the conjugate evolution path to control the step size σ. The principle is as follows: if the evolution path remains approximately orthogonal over multiple iterations, indicating a random walk, the step size is too small; if the path direction remains consistent over iterations, indicating straight-line movement, the step size is too large.

First, the vector $p _ { \sigma }$ that records the step size path is updated:

$$
\begin{array} { l } { { { \bf p } _ { \sigma } ^ { ( g + 1 ) } = ( 1 - c _ { \sigma } ) { \bf p } _ { \sigma } ^ { ( g ) } } } \\ { { \qquad + \sqrt { c _ { \sigma } ( 2 - c _ { \sigma } ) \mu _ { \mathrm { e f f } } } \cdot \left( { \bf C } ^ { ( g ) } \right) ^ { - \frac { 1 } { 2 } } \frac { { { \bf m } ^ { ( g + 1 ) } } - { { \bf m } ^ { ( g ) } } } { { \sigma ^ { ( g ) } } } . } } \end{array}\tag{13}
$$

where $c _ { \sigma }$ is the decay time constant, $C ^ { - \frac { 1 } { 2 } }$ denotes the isotropic transformation of the covariance matrix, and $\mu _ { \mathrm { e f f } }$ is the variance-effective selection mass:

$$
\mu _ { \mathrm { e f f } } = \frac { 1 } { \sum _ { i = 1 } ^ { \mu } w _ { i } ^ { 2 } } = \frac { 1 } { \| \mathbf { w } \| ^ { 2 } } .\tag{14}
$$

The step size $\sigma ^ { ( g + 1 ) }$ is then updated based on $p _ { \sigma } ^ { ( g + 1 ) }$

$$
\sigma ^ { ( g + 1 ) } = \sigma ^ { ( g ) } \exp \left( \frac { c _ { \sigma } } { d _ { \sigma } } \left( \frac { \parallel \mathbf { p } _ { \sigma } ^ { ( g + 1 ) } \parallel } { E \parallel \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) \parallel } - 1 \right) \right) .\tag{15}
$$

At this stage, if the path length ∥ $p _ { \sigma }$ ∥ exceeds the expected length for a random walk, $\sigma$ is increased; otherwise, it is decreased. Simultaneously, the evolution path $p _ { c }$ , which records the anisotropy of the search trajectory, is updated:

$$
\mathbf { p } _ { c } ^ { ( g + 1 ) } = ( 1 - c _ { c } ) \mathbf { p } _ { c } ^ { ( g ) } + \sqrt { c _ { c } ( 2 - c _ { c } ) \mu _ { \mathrm { e f f } } } \frac { \mathbf { m } ^ { ( g + 1 ) } - \mathbf { m } ^ { ( g ) } } { \sigma ^ { ( g ) } } ,\tag{16}
$$

Finally, the covariance matrix is updated using the decayed previous matrix $c ^ { ( g ) }$ , the historical evolution path $p _ { c } ^ { ( g + 1 ) }$ , and the information from the current population:

$$
\begin{array} { l } { { \displaystyle { \bf C } ^ { ( g + 1 ) } = } \ ~ } \\ { { \displaystyle ~ + c _ { 1 } \left( { \bf p } _ { c } ^ { ( g + 1 ) } \left( { \bf p } _ { c } ^ { ( g + 1 ) } \right) ^ { T } \right) } } \\ { { \displaystyle ~ + c _ { \mu } \sum _ { i = 1 } ^ { \mu } w _ { i } { \bf y } _ { i } { \bf y } _ { i } ^ { T } } . } \end{array}\tag{17}
$$

where $\mathbf { y } _ { i } = ( \boldsymbol { \lambda } _ { i : K } ^ { ( g ) } - \mathbf { m } ^ { ( g ) } ) / \sigma ^ { ( g ) }$ denotes the deviation vectors after subtracting the mean.

The adaptive update of the covariance matrix $C$ endows AWA with a fundamental advantage beyond simple heuristic search. It can not only dynamically adjust the scale of the search space, but also implicitly learn the relationships between different task coefficients. For example, if the optimization landscape exhibits off-diagonal characteristics, such as an increase in $\lambda _ { 1 }$ requiring a specific proportional decrease in $\lambda _ { 2 }$ to maintain loss reduction, AWA automatically adapts the search distribution to align with the objective function’s contours, forming a hyperellipsoid that allows efficient descent along correlated directions. This optimization is unattainable with grid search, which evaluates each task coefficient independently.

Final output and model merging. After the search finishes, the optimal coefficient vector is obtained:

$$
\lambda ^ { * } = \mathrm { a r g m i n } F ( \lambda ) .\tag{18}
$$

The final model merging is then performed:

$$
\theta _ { m e r g e d } = \theta _ { b a s e } + \sum _ { t = 1 } ^ { T } \lambda _ { t } ^ { * } \cdot \tilde { \tau } _ { t } .\tag{19}
$$

where $T$ denotes the number of models to be merged, and $\tilde { \tau } _ { t }$ represents the task vectors processed by CABS pruning.

Time complexity comparison. Compared with grid search, AWA has the significant advantage in time complexity. Assume there are $T$ tasks, and $S$ search steps are sampled in each dimension. Grid search requires $\mathcal { O } ( \bar { S } ^ { T } )$ evaluations, which makes it prone to the curse of dimensionality when merging multiple tasks, resulting in a substantial increase in time consumption. In contrast, the complexity of AWA mainly depends on the population size K and the number of iterations G. The upper bound of the total number of evaluations is ${ \mathcal { O } } ( K \times G )$ , which avoids the exponential dependence on the number of tasks. This not only significantly reduces the time required for merging, but also enables the search for better solutions in the wider continuous space.

## IV. EXPERIMENTS

## A. Experimental Setup

Benchmarks and Checkpoints. We conduct large language model evaluations on the LLM Leaderboard benchmark [33] using the Mistral-7B-v0.1 [34] backbone and its finetuned variants WildMarcoroni-Variant1-7B and WestSeverus-7B-DPO-v2. This benchmark includes six tasks: AI2 Reasoning Challenge [35], HellaSwag [36], MMLU [37], TruthfulQA [38], Winogrande [39], and GSM8K [40]. In addition, we perform experiments on the Open LLM Leaderboard 2 [41] benchmark using Qwen-2.5-7B-Instruct [42] and its finetuned variants fq2.5-7B and Tsunami-0.5-7B. This benchmark consists of six more challenging tasks, including IFEval [43], BBH [44], MATH [45], GPQA [46], MUSR [47], and MMLU-Pro [37]. All corresponding checkpoints are publicly available on Hugging Face. The evaluations on both benchmarks are conducted using the EleutherAI Language Model Evaluation Harness [48], which is a widely recognized standard framework for evaluating LLM capabilities.

For smaller models, we evaluate RoBERTa [49] and GPT-2 [50] on the GLUE benchmark [51]. Specifically, we consider the CoLA [52], MNLI [53], MRPC [54], QNLI, QQP, RTE [55]–[58], and SST-2 [59] datasets. To further increase task difficulty and diversity, we additionally include the multiplechoice reading comprehension task RACE [60] and the question answering task SQuAD [61] in the RoBERTa experiments. These datasets and their corresponding finetuned checkpoints are obtained from the FusionBench [62] repository, which is a comprehensive benchmark and unified library for model merging. Detailed descriptions of all datasets and links to the corresponding checkpoints can be found in Sections B.7 and B.8 of the appendix in our conference version.

Evaluation Metrics. For tasks from the GLUE benchmark, we uniformly use accuracy as the evaluation metric. For tasks from the LLM Leaderboard benchmark and the Open LLM Leaderboard 2 benchmark, we adopt the default metrics specified in the corresponding leaderboard descriptions for each task, such as success rate and accuracy. Detailed descriptions of the metrics used can be found in Section B.9 of the appendix in our conference version.

Baselines. We compare our CABS+ method with several representative and widely used model merging approaches, including Task Arithmetic and TIES-Merging [13], as well as AdaMerging [8], which similar to our method, determines task scaling coefficients without requiring additional training data or architectural modifications. We also include our previous CABS method, which has been accepted by ICML 2025, as well as the recent state-of-the-art method WUDI-Merging [9]. In addition, since CABS+ involves a prepruning stage, we further combine the above methods with magnitude pruning and DARE pruning to evaluate their performance after pruning, thereby enabling a fairer comparison.

To assess how far current model merging methods are from the expected ideal performance in large models, a new concept termed the “ideal model” is proposed. In our experiments, it is defined as the best performance achieved for each task among multiple checkpoints derived from the same base model under different fine-tuning configurations, serving as an upper bound for model merging performance.

Implementation Details. All model merging experiments are conducted on V100 GPUs with 32 GB memory on the Crater server platform [63]. To ensure the consistency and stability of the results, each experimental configuration is evaluated three times, and the average results are reported. For large models, inference is performed using the lm-evaluationharness version 0.4.0 library, with the batch size set to auto and all other parameters kept at their default values.

For the AWA strategy in CABS+, the population size is set to K=6, the initial step size is set to σ=0.05, and the number of iterations is set to G=50. An early stopping mechanism is applied such that the search terminates when the loss shows no significant change for 6 consecutive iterations. The remaining parameters follow the standard settings of the Covariance Matrix Adaptation Evolution Strategy. For the BS pruning strategy, the hyperparameters are consistent with those in the conference version. Specifically, the sparsity level is set to 0.90 for small models and 0.75 for large models.

## B. Results on Large Language Models

Tables I and II present the comparison results on large language models. In these tables, the last column, “AVG”, denotes the average performance of the merged model, while the values in parentheses indicate the performance improvement relative to the conference version of CABS. Bold values in each column represent the best performance, whereas underlined values indicate the second-best results. It is evident that CABS+ achieves a substantial performance improvement over the CABS in the merging experiments on fq2.5-7B and Tsunami-0.5-7B. Its overall performance is highly competitive with the ideal model baseline while further surpassing the recent state-of-the-art method WUDIMerging. In addition, we also conduct experiments under different merging orders, and the results demonstrate that CABS+ exhibits strong robustness to the order of model merging. In the model merging experiments on WildMarcoroni-Variant1-7B and WestSeverus-7B-DPO-v2, the performance of CABS+ is only marginally lower than that of WUDIMerging and remains highly comparable overall. Notably, although the CABS has already surpassed the ideal model baseline, we are still able to achieve further performance improvements on top of it, demonstrating the continued optimization capability of the proposed method.

Beyond the overall comparison, Tables I and II show that pruning strategies generally improve most merging methods on the Qwen2.5 series, but often degrade performance on the Mistral series. In contrast, CABS+ maintains stable performance across both model families, suggesting that its conflict-aware masking mechanism can reduce interference in heterogeneous task settings while better preserving shared representations in homogeneous settings. A detailed analysis is provided in the supplementary material.

## C. Results on Small-Scale Language Models

For the small scale model experiments, we mainly conduct evaluations on GPT-2 and RoBERTa, respectively assessing the performance under different numbers of merged tasks. Specifically, Tables IV, III, and V present the results on RoBERTa under 4 tasks, 6 tasks, and 2 tasks merging settings, respectively. Tables VI and Table VII correspond to the results on GPT-2 under different task number merging scenarios.

From Table IV, it can be observed that when performing 4 tasks merging on RoBERTa, the current CABS+ consistently achieves significant performance improvements over the original CABS across different task merging orders, while maintaining strong stability under varying merging sequences. At the same time, it clearly outperforms both WUDIMerging and AdaMerging. For the 2 tasks and 6 tasks merging settings reported in Tables V and III, respectively, the original CABS itself already surpasses the other compared methods, and CABS+ further improves the performance on top of this strong baseline. To provide a more comprehensive comparison of different methods under varying numbers of merged tasks, we further summarize their average performance on RoBERTa across different task number settings, as shown in Figure 2. As the number of tasks increases, the overall merging performance declines due to the growing task heterogeneity. This effect becomes particularly pronounced when transitioning from 4 tasks merging to 6 tasks merging, where the inclusion of question answering and multiple choice reading comprehension tasks, specifically SQuAD and RACE, introduces additional complexity. Despite these challenges, CABS+ consistently demonstrates superior performance over the compared methods under different task number configurations, while also exhibiting stronger robustness to variations in the number of merged tasks.

RESULTS OF 7B LLMS ON THE OPEN LLM LEADERBOARD 2.  
TABLE I
<table><tr><td>Method</td><td>MMLU</td><td>IFEval</td><td>BBH</td><td>MATH</td><td>GPQA</td><td>MUSR</td><td>AVG</td></tr><tr><td>Tsunami-0.5-7b</td><td>45.08</td><td>55.85</td><td>55.55</td><td>33.16</td><td>31.84</td><td>44.35</td><td>44.31</td></tr><tr><td>fq2.5-7b</td><td>44.83</td><td>44.97</td><td>56.23</td><td>34.36</td><td>32.13</td><td>46.57</td><td>43.18</td></tr><tr><td>Ideal Model</td><td>45.08</td><td>55.85</td><td>56.23</td><td>34.36</td><td>32.13</td><td>46.57</td><td>45.04</td></tr><tr><td>Task-Arithmetic</td><td>44.83</td><td>43.75</td><td>56.17</td><td>28.36</td><td>31.31</td><td>46.72</td><td>41.86</td></tr><tr><td>+ Magnitude</td><td>44.91</td><td>55.14</td><td>55.36</td><td>35.01</td><td>31.65</td><td>43.28</td><td>44.23</td></tr><tr><td>+ DARE</td><td>44.71</td><td>45.49</td><td>55.71</td><td>35.53</td><td>32.56</td><td>42.76</td><td>42.79</td></tr><tr><td>TIES-Merging</td><td>45.10</td><td>53.72</td><td>55.77</td><td>34.67</td><td>31.94</td><td>45.01</td><td>44.37</td></tr><tr><td>+ DARE</td><td>44.88</td><td>55.91</td><td>55.58</td><td>35.43</td><td>31.77</td><td>44.21</td><td>44.63</td></tr><tr><td>AdaMerging</td><td>44.74</td><td>47.77</td><td>55.99</td><td>31.37</td><td>31.46</td><td>46.86</td><td>43.03</td></tr><tr><td>+ Magnitude</td><td>44.95</td><td>48.83</td><td>56.09</td><td>32.36</td><td>31.86</td><td>46.19</td><td>43.38</td></tr><tr><td>+ DARE</td><td>44.93</td><td>54.41</td><td>55.95</td><td>33.20</td><td>32.47</td><td>44.21</td><td>44.20</td></tr><tr><td>WUDIMerging</td><td>45.05</td><td>53.54</td><td>55.96</td><td>33.18</td><td>31.79</td><td>45.00</td><td>44.09</td></tr><tr><td>+ Magnitude</td><td>44.99</td><td>56.52</td><td>56.11</td><td>33.15</td><td>31.96</td><td>45.53</td><td>44.71</td></tr><tr><td>+ DARE</td><td>44.91</td><td>54.65</td><td>55.73</td><td>34.67</td><td>32.72</td><td>43.16</td><td>44.31</td></tr><tr><td>CABS-fqFirst</td><td>44.81</td><td>44.97</td><td>56.03</td><td>31.66</td><td>31.58</td><td>47.25</td><td>42.72</td></tr><tr><td>CABS+-fqFirst (ours)</td><td>44.63</td><td>55.82</td><td>56.03</td><td>35.73</td><td>31.99</td><td>46.31</td><td>45.09 (+2.37)</td></tr><tr><td>CABS-TFirst</td><td>44.86</td><td>44.45</td><td>55.92</td><td>30.37</td><td>31.66</td><td>46.19</td><td>42.24</td></tr><tr><td>CABS+-TFirst (ours)</td><td>44.55</td><td>57.32</td><td>56.15</td><td>34.69</td><td>32.92</td><td>44.99</td><td>45.10 (+2.86)</td></tr></table>

TABLE II

PERFORMANCE COMPARISON OF 7B LLMS ON LLM LEADERBOARD USING DIFFERENT MERGING METHODS.
<table><tr><td>Method</td><td>ARC</td><td>Hella.</td><td>MMLU</td><td>TQA</td><td>Wino.</td><td>GSM8K</td><td>AVG</td></tr><tr><td>WestSeverus</td><td>71.30</td><td>88.26</td><td>63.92</td><td>72.72</td><td>83.69</td><td>74.27</td><td>75.69</td></tr><tr><td>WildMarcoroni Ideal Model</td><td>73.63</td><td>88.67</td><td>63.96</td><td>70.07</td><td>84.34</td><td>74.48</td><td>75.86</td></tr><tr><td>Task Arithmetic</td><td>73.63</td><td>88.67</td><td>63.96</td><td>72.72</td><td>84.34</td><td>74.48</td><td>76.30</td></tr><tr><td>+ Magnitude + DARE</td><td>72.52 71.93</td><td>89.25 89.32</td><td>63.39 63.18</td><td>74.00 73.85</td><td>83.46 84.12</td><td>73.38 72.22</td><td>76.02 75.77</td></tr><tr><td></td><td>72.64</td><td>88.86</td><td>63.54</td><td>72.82</td><td>84.03</td><td>73.44</td><td>75.89</td></tr><tr><td>TIES-Merging</td><td>71.42</td><td>89.17</td><td>63.16</td><td>73.82</td><td>84.74</td><td>73.01</td><td>75.89</td></tr><tr><td>+ DARE</td><td>71.87</td><td>88.95</td><td>63.56</td><td>72.87</td><td>84.61</td><td>73.21</td><td>75.85</td></tr><tr><td>AdaMerging</td><td>72.84</td><td>88.65</td><td>63.35</td><td>72.62</td><td>84.14</td><td>73.81</td><td>75.90</td></tr><tr><td>+ Magnitude</td><td>72.92</td><td>87.77</td><td>63.42</td><td>73.01</td><td>83.90</td><td>75.24</td><td></td></tr><tr><td>+ DARE</td><td>72.05</td><td>87.26</td><td>63.79</td><td>72.64</td><td>83.71</td><td></td><td>76.04</td></tr><tr><td>WUDIMerging</td><td>75.33</td><td></td><td></td><td>73.96</td><td></td><td>75.73</td><td>75.86</td></tr><tr><td>+ Magnitude</td><td>74.05</td><td>88.43 87.04</td><td>63.84 63.97</td><td>64.68</td><td>83.76 80.13</td><td>75.62</td><td>76.82</td></tr><tr><td>+ DARE</td><td>73.70</td><td>87.97</td><td>63.93</td><td>69.16</td><td>79.29</td><td>75.55 71.54</td><td>74.24</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>74.27</td></tr><tr><td>CABS</td><td>72.92</td><td>88.89</td><td>63.50</td><td>74.41</td><td>84.63</td><td>74.65</td><td>76.50</td></tr><tr><td>CABS+ (ours)</td><td>74.20</td><td>89.15</td><td>63.67</td><td>73.76</td><td>84.52</td><td>74.93</td><td>76.71</td></tr></table>

TABLE III  
PERFORMANCE OF MERGING SIX TASK VECTORS ON ROBERTA MODELS.
<table><tr><td rowspan=1 colspan=6>METHOD        RTE  MRPC</td><td rowspan=1 colspan=4>CoLA  SST-2 RACE  SQuAD</td><td rowspan=1 colspan=1>AVG</td></tr><tr><td rowspan=1 colspan=2>Finetuned Model</td><td rowspan=1 colspan=4>79.42   91.18</td><td rowspan=1 colspan=2>85.04  94.04</td><td rowspan=1 colspan=1>71.71</td><td rowspan=1 colspan=1>79.82</td><td rowspan=1 colspan=1>83.54</td></tr><tr><td rowspan=11 colspan=2>Task Arithmetic+ Magnitude+ DARETIES-Merging+ DAREAdaMerging+ Magnitude+ DAREWUDIMerging+ Magnitude+ DARE</td><td rowspan=1 colspan=3>67.15</td><td rowspan=1 colspan=1>79.41</td><td rowspan=1 colspan=1>72.00</td><td rowspan=1 colspan=1>85.78</td><td rowspan=1 colspan=1>56.21</td><td rowspan=1 colspan=1>38.82</td><td rowspan=1 colspan=1>66.56</td></tr><tr><td rowspan=1 colspan=3>72.56</td><td rowspan=1 colspan=1>81.13</td><td rowspan=1 colspan=1>75.26</td><td rowspan=1 colspan=1>87.50</td><td rowspan=1 colspan=1>56.99</td><td rowspan=1 colspan=1>36.23</td><td rowspan=1 colspan=1>68.28</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>71.12</td><td rowspan=1 colspan=1>71.</td><td rowspan=1 colspan=1>65.44</td><td rowspan=1 colspan=1>72.48</td><td rowspan=1 colspan=1>83.37</td><td rowspan=1 colspan=1>59.57</td><td rowspan=1 colspan=1>51.39</td><td rowspan=1 colspan=1>67.23</td></tr><tr><td rowspan=1 colspan=3>68.94</td><td rowspan=1 colspan=1>86.01</td><td rowspan=1 colspan=1>66.43</td><td rowspan=1 colspan=1>83.33</td><td rowspan=1 colspan=1>40.11</td><td rowspan=1 colspan=1>47.94</td><td rowspan=1 colspan=1>65.46</td></tr><tr><td rowspan=1 colspan=3>74.40</td><td rowspan=1 colspan=1>83.83</td><td rowspan=1 colspan=1>72.92</td><td rowspan=1 colspan=1>56.37</td><td rowspan=1 colspan=1>60.38</td><td rowspan=1 colspan=1>53.80</td><td rowspan=1 colspan=1>66.95</td></tr><tr><td rowspan=2 colspan=3>49.8257.40</td><td rowspan=1 colspan=1>73.53</td><td rowspan=1 colspan=1>70.95</td><td rowspan=1 colspan=1>91.06</td><td rowspan=1 colspan=1>50.75</td><td rowspan=1 colspan=1>54.20</td><td rowspan=1 colspan=1>65.05</td></tr><tr><td rowspan=1 colspan=1>69.85</td><td rowspan=1 colspan=1>70.18</td><td rowspan=1 colspan=1>87.39</td><td rowspan=1 colspan=1>59.63</td><td rowspan=1 colspan=1>47.66</td><td rowspan=1 colspan=1>65.35</td></tr><tr><td rowspan=2 colspan=3>46.2147.29</td><td rowspan=2 colspan=1>60.7868.34</td><td rowspan=2 colspan=1>63.9569.13</td><td rowspan=1 colspan=1>70.30</td><td rowspan=1 colspan=1>46.05</td><td rowspan=1 colspan=1>38.73</td><td rowspan=1 colspan=1>54.34</td></tr><tr><td rowspan=1 colspan=1>51.15</td><td rowspan=1 colspan=1>49.81</td><td rowspan=1 colspan=1>35.33</td><td rowspan=1 colspan=1>53.51</td></tr><tr><td rowspan=1 colspan=1>de</td><td rowspan=1 colspan=3>47.21</td><td rowspan=1 colspan=1>69.37</td><td rowspan=1 colspan=1>69.11</td><td rowspan=1 colspan=1>63.42</td><td rowspan=1 colspan=1>61.29</td><td rowspan=1 colspan=1>39.71</td><td rowspan=1 colspan=1>58.35</td></tr><tr><td rowspan=1 colspan=3>53.65</td><td rowspan=1 colspan=1>66.47</td><td rowspan=1 colspan=1>65.36</td><td rowspan=1 colspan=1>89.45</td><td rowspan=1 colspan=1>57.71</td><td rowspan=1 colspan=1>43.04</td><td rowspan=1 colspan=1>62.61</td></tr><tr><td rowspan=1 colspan=2>CABSCABS+ (ours)</td><td rowspan=1 colspan=3>68.9571.48</td><td rowspan=1 colspan=1>82.1184.31</td><td rowspan=1 colspan=1>73.9270.18</td><td rowspan=1 colspan=1>90.8390.71</td><td rowspan=1 colspan=1>58.9760.67</td><td rowspan=1 colspan=1>42.9645.60</td><td rowspan=1 colspan=1>69.6270.49 (+0.87)</td></tr></table>

TABLE IV  
PERFORMANCE OF MERGING FOUR TASK VECTORS ON ROBERTA MODELS.
<table><tr><td rowspan=1 colspan=3>Method</td><td rowspan=1 colspan=3>CoLA</td><td rowspan=1 colspan=2>SST-2  RTE</td><td rowspan=1 colspan=1>MRPC</td><td rowspan=1 colspan=1>Avg</td></tr><tr><td rowspan=1 colspan=3>Finetuned Model</td><td rowspan=1 colspan=3>85.04</td><td rowspan=1 colspan=1>94.04</td><td rowspan=1 colspan=1>79.42</td><td rowspan=1 colspan=1>91.18</td><td rowspan=1 colspan=1>87.42</td></tr><tr><td rowspan=12 colspan=3>Task Arithmetic+ Magnitude+ DARETIES-Merging+ DAREAdaMerging+ Magnitude+ DAREWUDIMerging+ Magnitude+ DARE</td><td rowspan=1 colspan=3>76.32</td><td rowspan=1 colspan=1>90.83</td><td rowspan=1 colspan=1>69.68</td><td rowspan=1 colspan=1>81.37</td><td rowspan=1 colspan=1>79.55</td></tr><tr><td rowspan=1 colspan=3>82.07</td><td rowspan=1 colspan=1>87.04</td><td rowspan=1 colspan=1>65.34</td><td rowspan=1 colspan=1>79.66</td><td rowspan=1 colspan=1>78.53</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2>7</td><td rowspan=3 colspan=1>76.9982.36</td><td rowspan=2 colspan=1>90.14</td><td rowspan=2 colspan=1>70.76</td><td rowspan=2 colspan=1>81.13</td><td rowspan=2 colspan=1>79.76</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2>8</td><td rowspan=1 colspan=1>86.93</td><td rowspan=1 colspan=1>61.01</td><td rowspan=1 colspan=1>79.41</td><td rowspan=1 colspan=1>77.43</td></tr><tr><td rowspan=1 colspan=3>77.66</td><td rowspan=1 colspan=1>90.94</td><td rowspan=1 colspan=1>69.31</td><td rowspan=1 colspan=1>81.62</td><td rowspan=1 colspan=1>79.88</td></tr><tr><td rowspan=2 colspan=3>79.5169.76</td><td rowspan=1 colspan=1>91.97</td><td rowspan=1 colspan=1>71.29</td><td rowspan=1 colspan=1>79.31</td><td rowspan=2 colspan=1>80.5275.76</td></tr><tr><td rowspan=1 colspan=1>92.55</td><td rowspan=1 colspan=1>59.99</td><td rowspan=1 colspan=1>80.73</td></tr><tr><td rowspan=1 colspan=3>40.84</td><td rowspan=1 colspan=1>85.21</td><td rowspan=1 colspan=1>52.71</td><td rowspan=1 colspan=1>84.56</td><td rowspan=1 colspan=1>65.83</td></tr><tr><td rowspan=1 colspan=3>78.24</td><td rowspan=1 colspan=1>92.09</td><td rowspan=1 colspan=1>74.73</td><td rowspan=1 colspan=1>82.11</td><td rowspan=1 colspan=1>81.79</td></tr><tr><td rowspan=2 colspan=3>69.5179.77</td><td rowspan=2 colspan=1>71.3391.97</td><td rowspan=1 colspan=1>47.29</td><td rowspan=1 colspan=1>69.85</td><td rowspan=1 colspan=1>64.50</td></tr><tr><td rowspan=1 colspan=1>72.92</td><td rowspan=1 colspan=1>80.15</td><td rowspan=1 colspan=1>81.20</td></tr><tr><td rowspan=6 colspan=3>CABS (CSRM)CABS+ (ours)CABS (MRSC)CABS+ (ours)CABS (RCMS)CABS+ (ours)CABS (SCMR)</td><td rowspan=2 colspan=3>78.2480.88</td><td rowspan=2 colspan=1>92.3291.55</td><td rowspan=1 colspan=1>74.37</td><td rowspan=2 colspan=1>81.6280.64</td><td rowspan=4 colspan=1>81.6482.42 (+0.78)81.7082.47 (+0.77)81.64</td></tr><tr><td rowspan=1 colspan=1>76.62</td></tr><tr><td rowspan=1 colspan=3>76.89</td><td rowspan=1 colspan=1>92.09</td><td rowspan=2 colspan=1>74.7376.9975.09</td><td rowspan=2 colspan=1>83.0982.0781.62</td></tr><tr><td rowspan=1 colspan=3>79.4877.76</td><td rowspan=1 colspan=1>91.3292.09</td></tr><tr><td rowspan=1 colspan=3>80.38</td><td rowspan=1 colspan=1>91.34</td><td rowspan=2 colspan=1>77.3173.65</td><td rowspan=2 colspan=1>80.7282.60</td><td rowspan=2 colspan=1>82.44 (+0.80)81.69</td></tr><tr><td rowspan=1 colspan=3>78.52</td><td rowspan=1 colspan=1>91.97</td></tr><tr><td rowspan=1 colspan=3>CABS+ (ours)</td><td rowspan=1 colspan=3>81.17</td><td rowspan=1 colspan=1>91.26</td><td rowspan=1 colspan=1>75.83</td><td rowspan=1 colspan=1>81.34</td><td rowspan=1 colspan=1>82.40 (+0.71)</td></tr></table>

![](images/92fa2788065fe07c36a104caaad7f37a8d9893e1a34054716aecdf86a5da845f.jpg)  
Fig. 2. Performance comparison of different merging methods on RoBERTa models across varying numbers of tasks.

When performing task merging on GPT-2, we obtain observations similar to those on RoBERTa. As shown in Tables VII and VI, under the 6 tasks merging setting, CABS+ achieves a clear improvement over CABS, with the average performance increasing by 1.56%, while also surpassing WUDIMerging. Under the 2 tasks merging setting, even though the original CABS already significantly outperforms the other compared methods and achieves results comparable to the finetuned model baseline, incorporating the AWA strategy into CABS+ still brings performance gain. This further demonstrates the effectiveness of the proposed AWA strategy.

TABLE V  
PERFORMANCE OF MERGING RTE-MRPC TASK PAIR ON ROBERTA MODELS.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=3>RTE</td><td rowspan=1 colspan=1>MRPC</td><td rowspan=1 colspan=1>AVG</td></tr><tr><td rowspan=1 colspan=1>Finetuned Model</td><td rowspan=1 colspan=3>79.42</td><td rowspan=1 colspan=1>91.18</td><td rowspan=1 colspan=1>85.30</td></tr><tr><td rowspan=11 colspan=1>Task Arithmetic+ Magnitude+ DARETIES-Merging+ DAREAdaMerging+ Magnitude+ DAREWUDIMerging+ Magnitude+ DARE</td><td rowspan=1 colspan=3>73.29</td><td rowspan=1 colspan=1>87.01</td><td rowspan=1 colspan=1>80.15</td></tr><tr><td rowspan=1 colspan=3>74.73</td><td rowspan=1 colspan=1>86.03</td><td rowspan=1 colspan=1>80.38</td></tr><tr><td rowspan=1 colspan=3>72.92</td><td rowspan=1 colspan=1>88.24</td><td rowspan=1 colspan=1>80.58</td></tr><tr><td rowspan=1 colspan=3>74.37</td><td rowspan=1 colspan=1>86.03</td><td rowspan=1 colspan=1>80.20</td></tr><tr><td rowspan=1 colspan=3>72.56</td><td rowspan=1 colspan=1>88.73</td><td rowspan=1 colspan=1>80.65</td></tr><tr><td rowspan=1 colspan=3>53.07</td><td rowspan=1 colspan=1>88.97</td><td rowspan=1 colspan=1>71.02</td></tr><tr><td rowspan=1 colspan=3>60.65</td><td rowspan=1 colspan=1>86.09</td><td rowspan=1 colspan=1>73.37</td></tr><tr><td rowspan=1 colspan=3>52.71</td><td rowspan=1 colspan=1>78.68</td><td rowspan=1 colspan=1>65.70</td></tr><tr><td rowspan=1 colspan=3>47.26</td><td rowspan=1 colspan=1>71.63</td><td rowspan=1 colspan=1>59.45</td></tr><tr><td rowspan=2 colspan=3>47.2972.92</td><td rowspan=1 colspan=1>74.81</td><td rowspan=1 colspan=1>61.05</td></tr><tr><td rowspan=1 colspan=1>92</td><td rowspan=1 colspan=1>80.64</td><td rowspan=1 colspan=1>76.78</td></tr><tr><td rowspan=2 colspan=1>CABSCABS+ (ours)</td><td rowspan=1 colspan=2>7</td><td rowspan=1 colspan=1>74.01</td><td rowspan=1 colspan=1>88.97</td><td rowspan=2 colspan=1>81.4982.06</td></tr><tr><td rowspan=1 colspan=3>73.20</td><td rowspan=1 colspan=1>90.91</td></tr></table>

TABLE VI

PERFORMANCE OF MERGING COLA-MRPC TASK PAIR ON GPT-2 MODELS.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>CoLA</td><td rowspan=1 colspan=1>MRPC</td><td rowspan=1 colspan=1>AVG</td></tr><tr><td rowspan=1 colspan=1>Finetuned Model</td><td rowspan=1 colspan=1>76.80</td><td rowspan=1 colspan=1>80.39</td><td rowspan=1 colspan=1>78.60</td></tr><tr><td rowspan=11 colspan=1>Task Arithmetic+ Magnitude+ DARETIES-Merging+ DAREAdaMerging+ Magnitude+ DAREWUDIMerging+ Magnitude+ DARE</td><td rowspan=1 colspan=1>75.55</td><td rowspan=1 colspan=1>77.45</td><td rowspan=1 colspan=1>76.50</td></tr><tr><td rowspan=1 colspan=1>76.61</td><td rowspan=1 colspan=1>79.66</td><td rowspan=1 colspan=1>78.13</td></tr><tr><td rowspan=1 colspan=1>76.70</td><td rowspan=1 colspan=1>77.21</td><td rowspan=1 colspan=1>76.95</td></tr><tr><td rowspan=1 colspan=1>76.89</td><td rowspan=1 colspan=1>77.94</td><td rowspan=1 colspan=1>77.42</td></tr><tr><td rowspan=1 colspan=1>77.09</td><td rowspan=1 colspan=1>76.72</td><td rowspan=1 colspan=1>76.91</td></tr><tr><td rowspan=1 colspan=1>69.13</td><td rowspan=1 colspan=1>72.32</td><td rowspan=1 colspan=1>70.73</td></tr><tr><td rowspan=1 colspan=1>74.50</td><td rowspan=1 colspan=1>81.10</td><td rowspan=1 colspan=1>77.80</td></tr><tr><td rowspan=1 colspan=1>61.35</td><td rowspan=1 colspan=1>76.78</td><td rowspan=1 colspan=1>69.07</td></tr><tr><td rowspan=1 colspan=1>47.77</td><td rowspan=1 colspan=1>73.79</td><td rowspan=1 colspan=1>60.78</td></tr><tr><td rowspan=1 colspan=1>47.79</td><td rowspan=1 colspan=1>76.97</td><td rowspan=1 colspan=1>62.38</td></tr><tr><td rowspan=1 colspan=1>47.71</td><td rowspan=1 colspan=1>73.54</td><td rowspan=1 colspan=1>60.63</td></tr><tr><td rowspan=2 colspan=1>CABSCABS+ (Ours)</td><td rowspan=2 colspan=1>76.4176.70</td><td rowspan=1 colspan=1>80.88</td><td rowspan=1 colspan=1>78.65</td></tr><tr><td rowspan=1 colspan=1>80.92</td><td rowspan=1 colspan=1>78.81</td></tr></table>

Beyond the aggregate performance trends, AdaMerging and WUDIMerging show noticeable instability across different backbone architectures and task-number settings. In comparison, CABS+ exhibits more stable performance under these varying conditions, which can be attributed to the conflict reduction introduced by CABS pruning and the balanced coefficient optimization enabled by AWA. Detailed analyses of these baseline behaviors and the stability mechanism of CABS+ are provided in the supplementary material.

## D. Efficiency Analysis

To further demonstrate the efficiency of CABS+, we compare CABS+ with the baseline methods CABS, AdaMerging, and WUDIMerging in terms of runtime and GPU memory consumption. Specifically, experiments are performed on RoBERTa under 4 task vectors merging setting, and on Mistral under 2 vectors merging setting. The detailed results are reported in Table VIII.

TABLE VII  
PERFORMANCE OF MERGING SIX TASK VECTORS ON GPT-2 MODELS.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=7>COLA  MNLI  MRPC  QNLI  QQP  RTE      AVG</td></tr><tr><td rowspan=1 colspan=1>Fintuned Model</td><td rowspan=1 colspan=1>76.80</td><td rowspan=1 colspan=1>82.08</td><td rowspan=1 colspan=1>80.39</td><td rowspan=1 colspan=4>88.27  89.64  65.34     80.42</td></tr><tr><td rowspan=1 colspan=1>Task-Arithmetic</td><td rowspan=1 colspan=1>68.94</td><td rowspan=1 colspan=1>65.30</td><td rowspan=1 colspan=1>70.59</td><td rowspan=1 colspan=1>65.28</td><td rowspan=1 colspan=1>81.35</td><td rowspan=1 colspan=1>46.21</td><td rowspan=1 colspan=1>66.28</td></tr><tr><td rowspan=1 colspan=1>+ Magnitude</td><td rowspan=1 colspan=1>51.68</td><td rowspan=1 colspan=1>54.33</td><td rowspan=1 colspan=1>36.52</td><td rowspan=1 colspan=1>56.69</td><td rowspan=1 colspan=1>76.56</td><td rowspan=1 colspan=1>50.18</td><td rowspan=1 colspan=1>54.33</td></tr><tr><td rowspan=1 colspan=1>+ DARE</td><td rowspan=1 colspan=1>68.74</td><td rowspan=1 colspan=1>64.13</td><td rowspan=1 colspan=1>70.83</td><td rowspan=1 colspan=1>65.13</td><td rowspan=1 colspan=1>80.48</td><td rowspan=1 colspan=1>45.85</td><td rowspan=1 colspan=1>65.86</td></tr><tr><td rowspan=1 colspan=1>TIES-Merging</td><td rowspan=1 colspan=1>68.84</td><td rowspan=1 colspan=1>68.61</td><td rowspan=1 colspan=1>69.36</td><td rowspan=1 colspan=1>66.98</td><td rowspan=1 colspan=1>82.37</td><td rowspan=1 colspan=1>44.77</td><td rowspan=1 colspan=1>66.82</td></tr><tr><td rowspan=2 colspan=1>+DAREAdaMerging</td><td rowspan=1 colspan=1>68.46</td><td rowspan=1 colspan=1>64.60</td><td rowspan=1 colspan=1>70.10</td><td rowspan=1 colspan=1>64.91</td><td rowspan=1 colspan=1>80.75</td><td rowspan=1 colspan=1>46.21</td><td rowspan=1 colspan=1>65.84</td></tr><tr><td rowspan=1 colspan=1>30.87</td><td rowspan=1 colspan=1>35.45</td><td rowspan=1 colspan=1>31.62</td><td rowspan=1 colspan=1>49.46</td><td rowspan=1 colspan=1>36.82</td><td rowspan=1 colspan=1>52.71</td><td rowspan=1 colspan=1>39.49</td></tr><tr><td rowspan=1 colspan=1>+ Magnitude</td><td rowspan=1 colspan=1>69.03</td><td rowspan=1 colspan=1>53.96</td><td rowspan=1 colspan=1>66.42</td><td rowspan=1 colspan=1>71.00</td><td rowspan=1 colspan=1>84.45</td><td rowspan=1 colspan=1>48.74</td><td rowspan=1 colspan=1>65.60</td></tr><tr><td rowspan=2 colspan=1>+DAREWUDIMerging</td><td rowspan=1 colspan=1>37.91</td><td rowspan=1 colspan=1>42.55</td><td rowspan=1 colspan=1>39.41</td><td rowspan=1 colspan=1>55.78</td><td rowspan=1 colspan=1>61.04</td><td rowspan=1 colspan=1>52.79</td><td rowspan=1 colspan=1>48.25</td></tr><tr><td rowspan=1 colspan=1>69.61</td><td rowspan=1 colspan=1>66.75</td><td rowspan=1 colspan=1>68.41</td><td rowspan=1 colspan=1>71.74</td><td rowspan=1 colspan=1>80.26</td><td rowspan=1 colspan=1>52.35</td><td rowspan=1 colspan=1>68.19</td></tr><tr><td rowspan=1 colspan=1>+ Magnitude</td><td rowspan=1 colspan=1>62.90</td><td rowspan=1 colspan=1>57.26</td><td rowspan=1 colspan=1>33.33</td><td rowspan=1 colspan=1>61.34</td><td rowspan=1 colspan=1>78.39</td><td rowspan=1 colspan=1>52.35</td><td rowspan=1 colspan=1>57.60</td></tr><tr><td rowspan=1 colspan=1>+DARE</td><td rowspan=1 colspan=1>69.22</td><td rowspan=1 colspan=1>61.53</td><td rowspan=1 colspan=1>73.04</td><td rowspan=1 colspan=1>68.20</td><td rowspan=1 colspan=1>78.21</td><td rowspan=1 colspan=1>47.65</td><td rowspan=1 colspan=1>66.31</td></tr><tr><td rowspan=1 colspan=1>CABSCABS+ (ours)</td><td rowspan=1 colspan=1>69.1369.24</td><td rowspan=1 colspan=1>67.4768.55</td><td rowspan=1 colspan=1>69.6171.36</td><td rowspan=1 colspan=1>66.3469.65</td><td rowspan=1 colspan=1>79.0580.80</td><td rowspan=1 colspan=1>51.2952.65</td><td rowspan=1 colspan=1>67.1568.71 (+1.56)</td></tr></table>

TABLE VIII

EFFICIENCY COMPARISON OF DIFFERENT MERGING METHODS ON ROBERTA AND MISTRAL MODELS IN TERMS OF RUNTIME AND GPU MEMORY USAGE.
<table><tr><td>Model</td><td>Method</td><td>AVG</td><td>Runtime</td><td>GPU Memory</td></tr><tr><td rowspan="3">Roberta</td><td>AdaMerging</td><td>80.52</td><td>12min</td><td>6.69GB</td></tr><tr><td>WUDIMerging</td><td>81.79</td><td>3min</td><td>5.10GB</td></tr><tr><td>CABS+ (ours)</td><td>82.42</td><td>3min</td><td>3.77GB</td></tr><tr><td rowspan="3">Mistral</td><td>AdaMerging</td><td>75.90</td><td>1h</td><td>66.70GB</td></tr><tr><td>WUDIMerging</td><td>76.82</td><td>4h</td><td>12.04GB</td></tr><tr><td>CABS+ (ours)</td><td>76.71</td><td>1h</td><td>15.95GB</td></tr></table>

For AdaMerging, the optimization process is based on gradient descent. According to the chain rule, all task vectors must be retained in GPU memory simultaneously, as they serve as essential nodes in the computational graph required for gradient computation. In addition, backpropagation requires the construction of the full computational graph and the storage of intermediate activations, leading to substantial GPU memory consumption, especially for large scale models. In contrast, CABS+ adopts a zeroth order optimization strategy, which eliminates the need for gradient computation and significantly reduces both computational overhead and GPU memory usage. The implementation is based on the cma library, where the optimization is executed using Numpy based matrix operations on the CPU. As a result, the GPU is only used for forward inference, serving as a loss evaluation module. After computing the loss, the results are transferred back to the CPU, where the internal distribution of the AWA strategy is updated and the next set of coefficients is generated. The merging of task vectors is then performed on the CPU, and the updated parameters are transferred to the GPU layer by layer. Consequently, at any given time, only the base model parameters and the merged parameters of a single layer are retained in GPU memory, which substantially reduces memory consumption. On the other hand, WUDIMerging adopts a layer wise optimization strategy. For each layer, the corresponding parameters of all task vectors are loaded into GPU memory, and gradient based optimization is performed. This process also requires additional computational graph construction and activation storage. Moreover, frequent data transfer between CPU and GPU is required to load and update layer specific parameters for each task vector. This results in significant overhead due to memory access and I/O communication. Such overhead becomes particularly pronounced on large models such as Mistral, where the total runtime can reach up to 4 hours. In comparison, CABS+ reduces the merging time to approximately one quarter of that required by WUDIMerging.

It is worth noting that different strategies are adopted in practice for models of different scales. When performing merging on RoBERTa, the model size is relatively small and GPU memory is sufficient. Therefore, following the common practice adopted by most existing merging methods, both the base model parameters and the task vectors are placed on the GPU for computation and optimization, which minimizes data transfer overhead and accelerates the merging process. In contrast, for large scale models such as Mistral, memory constraints necessitate the use of more memory efficient strategies. Under this setting, when using WUDIMerging, and following its original implementation, only the parameters of the current layer for all task vectors involved in optimization, together with the associated computational graph, are stored in GPU memory, while the remaining layers are kept in CPU memory and loaded on demand. For CABS+, only the base model parameters are maintained in GPU memory for forward inference, while all task vectors, the covariance matrix, and the weighted combination operations are stored in main memory or executed on the CPU. As a result, the GPU memory consumption of WUDIMerging scales linearly with the number of task vectors, corresponding to a space complexity of O(n), whereas the memory usage of CABS+ remains independent of the number of tasks, corresponding to O(1). In the case of two vectors merging, WUDIMerging requires storing only two layers parameters in GPU memory, which leads to relatively lower memory usage. However, due to the additional memory required for gradient computation and intermediate activations, its overall GPU memory consumption remains comparable to that of CABS+. As the number of vectors increases, the memory efficiency of CABS+ becomes more evident. For example, under the four task merging setting on RoBERTa, CABS+ exhibits noticeably lower GPU memory consumption than WUDIMerging.

![](images/1e7552b069a58a35859265fa54e5b1673e85ad3d78a6a7eee21d2569a7fdd804.jpg)

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Qwen2.5-leaderboardv2</td><td rowspan=1 colspan=1>Mistral-leaderboard</td><td rowspan=1 colspan=1>Roberta(4 tasks)</td><td rowspan=1 colspan=1>Roberta(2 tasks)</td><td rowspan=1 colspan=1>Roberta(6 tasks)</td><td rowspan=1 colspan=1>GPT-2(2 tasks)</td><td rowspan=1 colspan=1>GPT-2(6 tasks)</td><td rowspan=1 colspan=1>ViTB/32(6 tasks)</td><td rowspan=1 colspan=1>OverallImprovement</td></tr><tr><td rowspan=1 colspan=1>Imp. vs Ada</td><td rowspan=1 colspan=1>+4.81%</td><td rowspan=1 colspan=1>+1.07%</td><td rowspan=1 colspan=1>+2.42%</td><td rowspan=1 colspan=1>+15.54%</td><td rowspan=1 colspan=1>+8.36%</td><td rowspan=1 colspan=1>+11.42%</td><td rowspan=1 colspan=1>+74.00%</td><td rowspan=1 colspan=1>+18.16%</td><td rowspan=1 colspan=1>+16.97%</td></tr><tr><td rowspan=1 colspan=1>Imp. vs CABS</td><td rowspan=1 colspan=1>+6.77%</td><td rowspan=1 colspan=1>+0.27%</td><td rowspan=1 colspan=1>+0.94%</td><td rowspan=1 colspan=1>+0.70%</td><td rowspan=1 colspan=1>+1.25%</td><td rowspan=1 colspan=1>+0.20%</td><td rowspan=1 colspan=1>+2.32%</td><td rowspan=1 colspan=1>+1.75%</td><td rowspan=1 colspan=1>+1.78%</td></tr><tr><td rowspan=1 colspan=1>Imp. vs WUDI</td><td rowspan=1 colspan=1>+2.29%</td><td rowspan=1 colspan=1>-0.14%</td><td rowspan=1 colspan=1>+0.83%</td><td rowspan=1 colspan=1>+38.03%</td><td rowspan=1 colspan=1>+31.73%</td><td rowspan=1 colspan=1>+29.66%</td><td rowspan=1 colspan=1>+0.76%</td><td rowspan=1 colspan=1>+0.24%</td><td rowspan=1 colspan=1>+12.93%</td></tr></table>

Fig. 3. Comprehensive Performance and Improvement Comparison Across Methods.

Combined with the results reported in Table VIII, CABS+ not only achieves superior average performance on both models, but also effectively avoids the memory overhead introduced by gradient based optimization and the time cost caused by frequent I/O communication. Overall, CABS+ achieves a favorable balance between runtime efficiency and memory usage compared with other methods.

## E. Cross-modal Experiments and Ablation Studies

To further evaluate the generalization capability of CABS+, we conduct additional cross-modal experiments on ViT-B/32 across six vision tasks. Experimental results show that CABS+ achieves an average performance of 82.50, compared with 81.08 for CABS and 82.30 for WUDIMerging, yielding a 1.75% improvement over CABS while maintaining performance competitive with the state-of-the-art method WUDIMerging.

In addition, ablation studies verify the effectiveness of the asymmetric fitness function and boundary constraints in AWA. Experimental results demonstrate that these components improve optimization stability while substantially reducing the merging time compared with CABS. The complete results and corresponding experimental settings are provided in the supplementary material.

## F. Summary

In summary, CABS+ achieves superior performance across diverse merging configurations while maintaining low GPU memory consumption, allowing efficient execution on GPUs with limited memory capacity such as V100. A comprehensive comparison with AdaMerging, a method of the same category that determines task scaling coefficients without requiring additional training data, and the recent state of the art method WUDIMerging, is presented in Figure 3. Due to its robustness across different model architectures and varying numbers of tasks, CABS+ achieves significantly higher average performance, with improvements of 16.97% and 12.93% over these two methods, respectively. Furthermore, consistent performance gains are also observed compared with the strong baseline CABS.

## V. EMPIRICAL INVESTIGATION OF FACTORS AFFECTING MODEL MERGEABILITY

In the previous section, we concentrated on the core question of how to merge and introduced the CABS+ framework as an algorithmic advance designed to improve merging outcomes. However, a logically prior and practically important question that has been largely neglected by existing work is what to merge. This raises a fundamental question: do the intrinsic properties and extrinsic relationships of the source models fundamentally determine the difficulty of merging and the attainable performance ceiling?

Recent work [64] has begun to focus this question, highlighting that the geometry of task vectors and the degree of inter-model alignment are key determinants of successful model merging. In this section, we present the systematic empirical study of model mergeability. Rather than focusing solely on merging algorithms, we address the following core scientific questions:

TABLE IX  
EFFECT OF FINETUNING LEARNING RATE ON PARAMETER OVERLAP AND MERGE PERFORMANCE
<table><tr><td>Learning Rate|</td><td>Parameter Overlap (%) | Average Accuracy(%)</td><td></td></tr><tr><td> $1 \times 1 0 ^ { - 6 }$ </td><td>85.3</td><td>79.15</td></tr><tr><td> $5 \times 1 0 ^ { - 6 }$ </td><td>72.1</td><td>80.22</td></tr><tr><td> $1 \times 1 0 ^ { - 5 }$ </td><td>58.6</td><td>81.05</td></tr><tr><td> $2 \times 1 0 ^ { - 5 }$ </td><td>45.4</td><td>81.30</td></tr><tr><td> $5 \times 1 0 ^ { - 5 }$ </td><td>31.9</td><td>80.67</td></tr></table>

Q1. How do a model’s training configuration and optimization history(particularly its key hyperparameters) shape the geometric properties of its task vectors, and consequently influence its mergeability?

Q2. How do task heterogeneity and finetuning data distribution heterogeneity quantitatively affect the difficulty and effectiveness of model merging?

Q3. Do architectural differences in the base models and variations in model scale cause finetuned task vectors to exhibit systematically different mergeability characteristics?

By answering these questions, this section seeks not only to uncover the underlying regularities governing model merging, but also to establish a set of principled, scientifically grounded guidelines for model selection. These guidelines are intended to enable researchers and practitioners to make better a priori judgments about the suitability of candidate source models before attempting a merge, thereby shifting model merging from an art of post-hoc hyperparameter fiddling toward a principled practice of pre-merge screening.

To investigate these questions, we design a series of controlled experiments that isolate individual factors, with detailed experimental settings provided in the supplementary material. Importantly, to eliminate confounding effects introduced by the merging algorithm itself and to enable an objective assessment of source-model mergeability, all merging experiments in this chapter employ fixed, simple baseline merging methods (primarily non-sparsified Task Arithmetic, with DARE used as a comparison).

## A. Impact of Optimization History on Model Mergeability

To address Q1, we investigate how the optimization history of finetuned models influences their mergeability. Specifically, we focus on two critical optimization hyperparameters, namely the learning rate and the training epochs, and analyze how they shape the geometric properties of task vectors and affect the effectiveness of model merging.

1) Effect of Finetuning Learning Rate: First, we observe a negative correlation between the finetuning learning rate and the parameter overlap ratio. As shown in Table IX, the results reveal a clear and counterintuitive trend: lower learning rates lead to higher parameter overlap. This finding carries important theoretical implications. From the perspective of optimization trajectories, lower learning rates produce smoother and more conservative parameter updates, causing the optimization paths across different tasks to retain and leverage a larger portion of the shared features learned during pretraining. Consequently, task-specific parameter adjustments tend to concentrate within similar subspaces, resulting in higher overlap and greater representational similarity, as measured by metrics such as Centered Kernel Alignment (CKA). In contrast, higher learning rates induce more aggressive and task-specific parameter updates, leading to rapid divergence of optimization trajectories across tasks and a reduced overlap in the parameter space.

TABLE X  
EFFECT OF FINETUNING TRAINING EPOCH ON PARAMETER OVERLAP AND MERGE PERFORMANCE
<table><tr><td></td><td>Training Epochs | Parameter Overlap (%) | Average Accuracy(%)</td><td></td></tr><tr><td>1</td><td>42.5</td><td>80.11</td></tr><tr><td>3</td><td>46.8</td><td>81.30</td></tr><tr><td>5</td><td>44.1</td><td>81.12</td></tr><tr><td>10</td><td>38.2</td><td>80.45</td></tr></table>

Second, the relationship between learning rate and merge performance is non-monotonic. A closer inspection of the data in Table IX indicates that at extremely low learning rates $( 1 \times 1 0 ^ { - 6 } )$ , corresponding to very high parameter overlap (85.3%), the merged model achieves relatively low performance (79.15%), consistent with the notion that excessive overlap can induce destructive interference. As the learning rate increases moderately to $2 \times 1 0 ^ { - 5 }$ , the overlap decreases and merge performance rises to a peak (81.30%). However, when the learning rate becomes too high $( 5 \times 1 0 ^ { - 5 } )$ , despite continued reduction in overlap, merge performance begins to decline. This phenomenon likely arises because overly high learning rates lead to task-specific overfitting, producing task vectors that are excessively sharp and specialized. While conflicts are reduced, the amount of beneficial shared knowledge across models also diminishes, thereby hindering mergeability.

2) Effect of Finetuning Training Epoch: As shown in Table X, increasing the finetuning epoch produces a nonmonotonic effect on both parameter overlap and merge performance. The parameter overlap initially increases and reaches the peak at 3 epochs, suggesting that early-stage finetuning (epochs 1–3) primarily reinforces shared features across tasks. In contrast, later-stage finetuning (epochs 5–10) focuses more on learning task-specific features, causing optimization trajectories to diverge and reducing overlap. Merge performance exhibits a typical inverted-U pattern: models with insufficient finetuning (1 epoch) contain incomplete task knowledge, resulting in suboptimal merged accuracy (80.11%), whereas over-finetuned models (10 epochs) tend to overfit to their respective tasks, producing overly specialized task vectors that are less compatible with other tasks and yielding slightly lower merge performance (80.45%). In this experiment, the optimal merge performance (81.30%) occurs at 3 epochs, coinciding with the maximum parameter overlap. These results indicate that finetuning epoch affects mergeability through an optimal finetuning window in which models have sufficiently acquired task knowledge without overfitting, thereby retaining both generalization and compatibility across tasks.

TABLE XI  
IMPACT OF TASK HETEROGENEITY ON RELATIVE SYNERGY SCORE
<table><tr><td>Level</td><td>Task Pair</td><td>Scoreideal (%)</td><td>Scoremerged (%)</td><td>RSS (%)</td></tr><tr><td>Low</td><td> $\mathbf { M R P C + Q Q P }$ </td><td>89.50</td><td>90.15</td><td>+0.73</td></tr><tr><td>Medium</td><td>SST-2+RTE</td><td>86.21</td><td>85.95</td><td>-0.30</td></tr><tr><td>High</td><td>IFEval+GSM8K</td><td>60.39</td><td>51.90</td><td>-14.06</td></tr></table>

TABLE XII

IMPACT OF FINETUNING DATA DISTRIBUTION DIFFERENCES ON MERGE PERFORMANCE
<table><tr><td>Level</td><td>Task Pair</td><td> $\mathbf { S c o r e _ { i d e a l } }$  (%) |</td><td> $\mathbf { S c o r e } _ { \mathbf { m e r g e d } }$  (%)</td><td>RSS (%)</td></tr><tr><td>Low</td><td>IMDb+Yelp</td><td>92.50</td><td>92.85</td><td>+0.38</td></tr><tr><td>High</td><td>IMDb+Financial</td><td>90.80</td><td>84.15</td><td>-7.32</td></tr></table>

## B. Impact of Task and Data Distribution Heterogeneity

To address Q2, we investigate how heterogeneity arising from task semantics and finetuning data distributions influences model mergeability. Unlike optimization-related factors, which affect mergeability through differences in training dynamics, task and data heterogeneity introduce intrinsic sources of divergence that fundamentally constrain the compatibility of task vectors. When tasks differ substantially in their semantic objectives or data distributions, the corresponding parameter adaptations are more likely to occupy distinct regions of the parameter space, thereby increasing the likelihood of destructive interference during merging.

1) Effect of Task Heterogeneity: To quantify mergeability in a normalized and comparable manner, we adopt the Relative Synergy Score (RSS), defined as:

$$
\mathrm { R S S } = \frac { \mathrm { S c o r e } _ { \mathrm { m e r g e d } } - \mathrm { S c o r e } _ { \mathrm { i d e a l } } } { \mathrm { S c o r e } _ { \mathrm { i d e a l } } } \times 1 0 0 \%\tag{20}
$$

where $\mathrm { S c o r e } _ { \mathrm { m e r g e d } }$ denotes the average performance of the merged model across tasks, and $\mathrm { S c o r e } _ { \mathrm { i d e a l } }$ denotes the average performance of the individually finetuned task-specific models. Positive RSS values indicate synergistic merging, whereas negative RSS values indicate destructive interference.

As shown in Table XI, the experimental results clearly indicate a strong negative correlation between task heterogeneity and synergistic merge performance. In the low-heterogeneity setting, merging produces a positive synergistic gain, with an RSS of +0.73%. In the medium-heterogeneity setting, the merged model achieves performance close to the ideal baseline but with a slight degradation, yielding an RSS of −0.30%. In contrast, in the high-heterogeneity scenario, merging causes severe destructive interference, with performance falling well below the ideal model and an RSS of −14.06%. These results provide quantitative evidence supporting the intuition that merging similar tasks is easier and more effective.

2) Effect of Finetuning Data Distribution: As shown in Table XII, the results reveal a strong positive correlation between data distribution similarity and mergeability. Merging models trained on two highly similar review datasets (IMDb and Yelp) produces a small positive synergistic effect, with $\mathrm { R S S } ~ = ~ + 0 . 3 8 \%$ . In contrast, merging models trained on datasets with large domain differences (IMDb and Financial PhraseBank) leads to severe performance degradation, with $\mathrm { R S S } = - 7 . 3 2 \%$ . This outcome reflects the fact that differing data distributions guide models to learn distinct features, biases and shortcuts, which may conflict when merged. These findings indicate that data distribution mismatch can substantially reduce mergeability even when the task definitions are identical.

TABLE XIII  
COMPARISON OF AVERAGE PERFORMANCE SCORES ACROSS DIFFERENT MODEL ARCHITECTURES FOR VARIOUS SPARSIFICATION STRATEGIES
<table><tr><td>Method</td><td>RoBERTa(Encoder-only)</td><td>GPT-2(Decoder-only)</td></tr><tr><td rowspan="3">Task Arithmetic TA+DARE TA+Magnitude</td><td>79.55</td><td>76.50</td></tr><tr><td>79.76</td><td>76.95</td></tr><tr><td>78.53</td><td>78.13</td></tr><tr><td rowspan="3">TIES-Merging CABS</td><td>77.43</td><td>77.42</td></tr><tr><td>81.70</td><td>78.65</td></tr><tr><td>82.37</td><td>78.81</td></tr></table>

## C. Impact of Model Architecture and Scale

To address Q3, we investigate how differences in base model architecture and model scale influence mergeability. Unlike task and data heterogeneity, which affect mergeability through differences in learned functional representations, architectural and scale differences introduce structural and geometric variations in the parameter space itself. These variations can alter both the alignment and compatibility of task vectors, thereby affecting the feasibility and effectiveness of merging. Understanding these effects is essential for determining whether models derived from different pretrained backbones or with different parameter scales can be reliably merged.

1) Effect of Model Architecture: As shown in Table XIII, a notable contrast emerges between the two architectures. On RoBERTa (Encoder-only), random pruning (DARE) achieves better performance (79.76) than magnitude pruning (78.53), which is consistent with the previously observed anomalous behavior. However, on GPT-2 (Decoder-only), the trend reverses: magnitude pruning achieves significantly higher performance (78.13) than random pruning (76.95), and its performance approaches that of CABS (78.65).

This observation suggests that the effectiveness of magnitude pruning is not universal, but instead strongly dependent on model architecture. The plausible explanation lies in the differences in information processing mechanisms. Encoderonly models employ bidirectional attention, which may lead to more distributed and cooperative parameter updates, allowing randomly retained parameters to preserve sufficient functionality. In contrast, Decoder-only models rely on autoregressive causal attention, which may result in more localized and concentrated parameter updates, making the preservation of high-magnitude parameters more beneficial for maintaining model performance.

2) Effect of Model Scale: As shown in Table XIV, a clear trend emerges: mergeability improves consistently as model scale increases. On the 7B model, merging results in mild destructive interference, with an RSS of −1.06%. On the 13B model, this interference nearly disappears, yielding an RSS of −0.06%. On the 70B model, merging produces a clear synergistic effect, with an RSS of +1.28%.

TABLE XIV  
IMPACT OF MODEL SCALE ON MERGE SYNERGY
<table><tr><td>Model Scale</td><td> $\mathbf { S c o r e _ { i d e a l } }$  (%)|</td><td> $\mathbf { S c o r e _ { m e r g e d } }$  (%)|</td><td>RSS (%)</td></tr><tr><td>7B</td><td>85.10</td><td>84.20</td><td>-1.06</td></tr><tr><td>13B</td><td>87.50</td><td>87.45</td><td>-0.06</td></tr><tr><td>70B</td><td>90.20</td><td>91.35</td><td>+1.28</td></tr></table>

This trend can be attributed to two key factors: knowledge decoupling and parameter redundancy. Larger models possess substantially more parameters, enabling them to store taskspecific knowledge in more distinct and potentially nearorthogonal parameter subspaces. This implicit knowledge decoupling reduces the likelihood of destructive interference between task vectors during merging. Additionally, large models exhibit higher parameter redundancy. Even when some parameters are negatively affected by merging conflicts, other redundant parameters can compensate for the loss, resulting in improved robustness and overall merge performance. This finding provides a new perspective on the advantages of large-scale models: beyond superior single-task performance, they also offer intrinsic benefits for multi-task knowledge integration through model merging.

## D. Summary of Key Findings

Overall, the empirical results identify six factors that influence model mergeability, including learning rate, finetuning epoch, task heterogeneity, data distribution heterogeneity, model architecture, and model scale. These findings suggest that model merging performance is not determined solely by the merging algorithm, but also by the intrinsic properties and relationships of the source models. The key findings and contributions are summarized as follows:

• Training configurations, particularly learning rate and finetuning epoch, are critical intrinsic factors affecting model mergeability. These factors directly influence the geometric structure and compatibility of task vectors, thereby determining the effectiveness of model merging.

• Task heterogeneity constitutes a fundamental obstacle to successful merging. To enable quantitative analysis of this effect, a new metric, the Relative Synergy Score (RSS), is proposed to measure synergistic gain and destructive interference in a normalized and interpretable manner.

• Finetuning data distribution is another key factor affecting mergeability. Even when task definitions are identical, differences in training data distributions can significantly reduce model compatibility. This finding highlights the importance of carefully examining training data sources and distributions during the model selection stage prior to merging.

• Model architecture constitutes an important determinant of mergeability characteristics. Encoder-only and Decoder-only architectures exhibit systematically different responses to sparsification strategies, indicating that merge behavior is strongly architecture-dependent.

• Model scale positively correlates with mergeability. Larger models exhibit stronger knowledge decoupling and parameter redundancy, enabling more effective integration of task-specific knowledge.

These findings deepen the scientific understanding of model merging and provide practical, actionable guidelines for model selection prior to merging. They constitute an independent and complementary contribution alongside the CABS series methodology.

## VI. CONCLUSION

This paper proposed CABS+, an enhanced model merging framework that extends CABS with the Adaptive Weight Allocation strategy. By replacing grid search with gradientfree coefficient optimization, CABS+ improves merging efficiency, reduces GPU memory consumption, and promotes more balanced performance improvements across tasks. We further conducted the systematic empirical study of model mergeability and introduced the Relative Synergy Score as the quantitative measure. The analysis identifies key factors affecting mergeability and provides practical guidance for model selection before merging. Extensive experiments across diverse tasks, model architectures, and model scales demonstrate that CABS+ achieves robust performance with favorable efficiency. Future work will explore multimodal and heterogeneous model merging to further broaden its applicability.

## REFERENCES

[1] Y. Kim, S. Lee, A. Jung, B. Ryu, and S. Hong, “Task vector quantization for memory-efficient model merging,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 20 105– 20 115.

[2] G. Du et al., “Parameter competition balancing for model merging,” Advances in Neural Information Processing Systems (NeurIPS), vol. 37, pp. 84 746–84 776, 2024.

[3] H. Chen et al., “Toward Effective Model Merging in Semantic Segmentation,” IEEE Transactions on Neural Networks and Learning Systems (TNNLS), vol. 37, no. 4, pp. 1948–1962, Apr. 2026.

[4] W. Li, Y. Peng, M. Zhang, L. Ding, H. Hu, and L. Shen, “Deep Model Fusion: A Survey,” IEEE Transactions on Neural Networks and Learning Systems (TNNLS), pp. 1–17, 2025.

[5] X. Jin, X. Ren, D. Preotiuc-Pietro, and P. Cheng, “Dataless Knowledge Fusion by Merging Weights of Language Models,” in The Eleventh International Conference on Learning Representations (ICLR), Sep. 2022.

[6] T. Akiba, M. Shing, Y. Tang, Q. Sun, and D. Ha, “Evolutionary optimization of model merging recipes,” Nature Machine Intelligence (NMI), vol. 7, no. 2, pp. 195–204, Feb. 2025.

[7] G. Ilharco, M. T. Ribeiro, M. Wortsman, L. Schmidt, H. Hajishirzi, and A. Farhadi, “Editing models with task arithmetic,” in The Eleventh International Conference on Learning Representations (ICLR), Sep. 2023.

[8] E. Yang et al., “Adamerging: Adaptive model merging for multitask learning,” in The Twelfth International Conference on Learning Representations (ICLR), May 2024.

[9] R. Cheng, F. Xiong, Y. Wei, W. Zhu, and C. Yuan, “Whoever started the interference should end it: Guiding data-free model merging via task vectors,” in Forty-second International Conference on Machine Learning (ICML), 2025.

[10] M. Zhang, J. Liu, G. Ding, L. Ou, X. Yu, and B. Zhuang, “Channel merging: Preserving specialization for merged experts,” in Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), vol. 39, no. 21, 2025, pp. 22 479–22 487.

[11] Y. He, S. Zeng, Y. Hu, R. Yang, T. Zhang, and H. Zhao, “Mergebench: A benchmark for merging domain-specialized LLMs,” in The Thirtyninth Annual Conference on Neural Information Processing Systems (NeurIPS), 2026.

[12] L. Yu, B. Yu, H. Yu, F. Huang, and Y. Li, “Language models are super mario: Absorbing abilities from homologous models as a free lunch,” in Forty-first International Conference on Machine Learning (ICML), 2024.

[13] P. Yadav, D. Tam, L. Choshen, C. A. Raffel, and M. Bansal, “TIES-Merging: Resolving Interference When Merging Models,” Advances in Neural Information Processing Systems (NeurIPS), vol. 36, pp. 7093– 7115, Dec. 2023.

[14] M. Davari and E. Belilovsky, “Model breadcrumbs: Scaling multitask model merging with sparse masks,” in European Conference on Computer Vision (ECCV). Springer, 2024, pp. 270–287.

[15] Y. He, Y. Hu, Y. Lin, T. Zhang, and H. Zhao, “Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic,” Transactions on Machine Learning Research (TMLR), Oct. 2024.

[16] T. Liang, J. Glossner, L. Wang, S. Shi, and X. Zhang, “Pruning and quantization for deep neural network acceleration: A survey,” Neurocomputing, vol. 461, pp. 370–403, Oct. 2021.

[17] F. Z. Zhang, P. Albert, C. Rodriguez-Opazo, A. van den Hengel, and E. Abbasnejad, “Knowledge composition using task vectors with learned anisotropic scaling,” Advances in Neural Information Processing Systems (NeurIPS), vol. 37, pp. 67 319–67 354, 2024.

[18] J. Chen, Q. Zhang, W. Zhang, X. Luo, P. S. Yu, and Z. Qiao, “Learn to merge: Meta-learning for adaptive multi-task model merging,” 2026, arxiv.

[19] P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, and A. G. Wilson, “Averaging weights leads to wider optima and better generalization,” in 34th Conference on Uncertainty in Artificial Intelligence 2018, UAI 2018. Association For Uncertainty in Artificial Intelligence (AUAI), 2018, pp. 876–885.

[20] M. Wortsman et al., “Model soups: averaging weights of multiple finetuned models improves accuracy without increasing inference time,” in Proceedings of the 39th International Conference on Machine Learning (ICML), Jun. 2022, pp. 23 965–23 998.

[21] M. S. Matena and C. A. Raffel, “Merging Models with Fisher-Weighted Averaging,” Advances in Neural Information Processing Systems (NeurIPS), vol. 35, pp. 17 703–17 716, Dec. 2022.

[22] F. Xiong et al., “Multi-task model merging via adaptive weight disentanglement,” arXiv preprint arXiv:2411.18729, 2024.

[23] Y. Wei, A. Tang, L. Shen, Z. Hu, C. Yuan, and X. Cao, “Modeling multi-task model merging as adaptive projective gradient descent,” in Proceedings of the 42nd International Conference on Machine Learning (ICML), vol. 267, 13–19 Jul 2025, pp. 66 178–66 193.

[24] Q. Wei et al., “Representation surgery in model merging with probabilistic modeling,” in Forty-second International Conference on Machine Learning (ICML), Jun. 2025.

[25] K. Wang, N. Dimitriadis, G. Ortiz-Jimenez, F. Fleuret, and P. Frossard, “Localizing task information for improved model merging and compression,” in Forty-first International Conference on Machine Learning (ICML), 2024.

[26] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: a simple way to prevent neural networks from overfitting,” The journal of machine learning research (JMLR), vol. 15, no. 1, pp. 1929–1958, 2014.

[27] O. Kovaleva, S. Kulshreshtha, A. Rogers, and A. Rumshisky, “BERT busters: Outlier dimensions that disrupt transformers,” in Findings of the Association for Computational Linguistics: ACL-IJCNLP, 2021, pp. 3392–3405.

[28] G. Puccetti, A. Rogers, A. Drozd, and F. Dell’Orletta, “Outlier Dimensions that Disrupt Transformers are Driven by Frequency,” in Findings of the Association for Computational Linguistics: EMNLP 2022, Dec. 2022, pp. 1286–1304.

[29] L. Yin et al., “Outlier weighed layerwise sparsity (OWL): a missing secret sauce for pruning LLMs to high sparsity,” in Proceedings of the 41st International Conference on Machine Learning (ICML), ser. ICML’24, vol. 235, Jul. 2024, pp. 57 101–57 115.

[30] A. Zhou et al., “Learning n:m fine-grained structured sparse neural networks from scratch,” in International Conference on Learning Representations (ICLR), 2021.

[31] M. Xia, Z. Zhong, and D. Chen, “Structured Pruning Learns Compact and Accurate Models,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL), May 2022, pp. 1513–1528.

[32] N. Hansen and A. Ostermeier, “Completely derandomized selfadaptation in evolution strategies,” Evolutionary computation, vol. 9, no. 2, pp. 159–195, 2001.

[33] E. Beeching et al., “Open llm leaderboard,” 2023.

[34] A. Q. Jiang et al., “Mistral 7B,” arXiv preprint arXiv:2310.06825, 2023.

[35] P. Clark et al., “Think you have solved question answering? try arc, the ai2 reasoning challenge,” arXiv preprint arXiv:1803.05457, 2018.

[36] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi, “Hellaswag: Can a machine really finish your sentence?” in Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics (ACL), 2019, pp. 4791–4800.

[37] Y. Wang et al., “MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark,” 2024, https://arxiv.org/abs/2406.01574.

[38] S. Lin, J. Hilton, and O. Evans, “TruthfulQA: Measuring How Models Mimic Human Falsehoods,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL), May 2022, pp. 3214–3252.

[39] K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi, “Winogrande: An adversarial winograd schema challenge at scale,” Communications of the ACM, vol. 64, no. 9, pp. 99–106, 2021.

[40] K. Cobbe et al., “Training verifiers to solve math word problems,” arXiv preprint arXiv:2110.14168, 2021.

[41] C. Fourrier, N. Habib, A. Lozovskaya, K. Szafer, and T. Wolf, “Open llm leaderboard v2,” https://huggingface.co/spaces/open-llm-leaderboard/ open llm leaderboard, 2024.

[42] A. Yang et al., “Qwen2. 5 technical report,” arXiv preprint arXiv:2412.15115, 2024.

[43] J. Zhou et al., “Instruction-following evaluation for large language models,” 2023, https://arxiv.org/abs/2311.07911.

[44] M. Suzgun et al., “Challenging big-bench tasks and whether chain-ofthought can solve them,” 2022, https://arxiv.org/abs/2210.09261.

[45] D. Hendrycks et al., “Measuring mathematical problem solving with the math dataset,” 2021, https://arxiv.org/abs/2103.03874.

[46] D. Rein et al., “Gpqa: A graduate-level google-proof qa benchmark,” 2023, https://arxiv.org/abs/2311.12022.

[47] Z. Sprague, X. Ye, K. Bostrom, S. Chaudhuri, and G. Durrett, “Musr: Testing the limits of chain-of-thought with multistep soft reasoning,” 2024, https://arxiv.org/abs/2310.16049.

[48] L. Gao et al., “A framework for few-shot language model evaluation,” 2024.

[49] Y. Liu, “Roberta: A robustly optimized bert pretraining approach,” arXiv preprint arXiv:1907.11692, 2019.

[50] A. Radford et al., “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.

[51] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. Bowman, “Glue: A multi-task benchmark and analysis platform for natural language understanding,” in Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, 2018, pp. 353–355.

[52] A. Warstadt, A. Singh, and S. R. Bowman, “Neural network acceptability judgments,” Transactions of the Association for Computational Linguistics, vol. 7, pp. 625–641, 2019.

[53] N. Nangia, A. Williams, A. Lazaridou, and S. Bowman, “The repeval 2017 shared task: Multi-genre natural language inference with sentence representations,” in Proceedings of the 2nd workshop on evaluating vector space representations for NLP, 2017, pp. 1–10.

[54] B. Dolan and C. Brockett, “Automatically constructing a corpus of sentential paraphrases,” in Third international workshop on paraphrasing (IWP2005), 2005.

[55] I. Dagan, O. Glickman, and B. Magnini, “The pascal recognising textual entailment challenge,” in Machine learning challenges workshop. Springer, 2005, pp. 177–190.

[56] R. Bar-Haim et al., “The second pascal recognising textual entailment challenge,” in Proceedings of the second PASCAL challenges workshop on recognising textual entailment, vol. 1. Citeseer, 2006.

[57] D. Giampiccolo, B. Magnini, I. Dagan, and W. B. Dolan, “The third pascal recognizing textual entailment challenge,” in Proceedings of the ACL-PASCAL workshop on textual entailment and paraphrasing, 2007, pp. 1–9.

[58] L. Bentivogli, P. Clark, I. Dagan, and D. Giampiccolo, “The fifth pascal recognizing textual entailment challenge.” TAC, vol. 7, no. 8, p. 1, 2009.

[59] R. Socher et al., “Recursive deep models for semantic compositionality over a sentiment treebank,” in Proceedings of the 2013 conference on empirical methods in natural language processing (EMNLP), 2013, pp. 1631–1642.

[60] G. Lai, Q. Xie, H. Liu, Y. Yang, and E. Hovy, “Race: Large-scale reading comprehension dataset from examinations,” in Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2017, pp. 785–794.

[61] P. Rajpurkar, “Squad: 100,000+ questions for machine comprehension of text,” arXiv preprint arXiv:1606.05250, 2016.

[62] A. Tang et al., “FusionBench: A Unified Library and Comprehensive Benchmark for Deep Model Fusion,” Journal of Machine Learning Research (JMLR), vol. 26, no. 307, pp. 1–38, 2025.

[64] T. Bowen, L. Songning, W. Jiemin, S. Zhihao, G. Shiming, and Y. Yutao, “Beyond task vectors: Selective task arithmetic based on importance metrics,” arXiv preprint arXiv:2411.16139, 2024.

[63] “raids-lab/crater,” https://github.com/raids-lab/crater., 2026.
# Beyond MSE: Rethinking the Evaluation Metric and Benchmarking for Irregular Time Series Forecasting

Rongwen Li <sup>1</sup> Haixin Xie <sup>1</sup> Xiao Wang <sup>1</sup> Changjian Chen <sup>1</sup>

## Abstract

Existing research on irregular time-series forecasting has primarily focused on model design, while evaluation metrics remain insufficiently studied. Existing benchmarks typically use mean squared error (MSE) as the evaluation metric. We show that, in irregular forecasting, MSE is determined not only by the model prediction but also by the sample-specific timestamp sampling distributions, leading to a biased assessment of the models’ continuous-time predictive performance. To address this issue, we propose the Continuous-time Squared Error (CSE), which employs importance weighting to eliminate the influence of the timestamp sampling distributions. We further theoretically prove that CSE’s asymptotic estimation error with respect to continuous-time risk is no greater than that of MSE. Finally, we construct a systematic benchmark covering synthetic, semisynthetic, and eight real-world datasets to validate the effectiveness of CSE and systematically evaluate models’ continuous-time predictive performance. Experiments show that CSE can recover continuous-time risk more accurately than MSE, while relying solely on MSE may not fully reflect models’ continuous-time predictive performance in real-world scenarios. Our code can be obtained at https://github.com/hnu-vis/ITS-Bench.

## 1. Introduction

Irregular time series are ubiquitous in real-world applications such as healthcare, environmental monitoring, and human activity analysis (Ghassemi et al., 2015; Decorte et al., 2024; Afrifa-Yamoah et al., 2020). Unlike regularly sampled time series, their observations occur at non-uniform timestamps, and different samples may exhibit different sampling patterns. In recent years, considerable research effort has been devoted to modeling such non-uniform temporal processes, leading to the development of continuous-time (Bilos et al., 2021), attention-based (Shukla & Marlin, 2021), graph-based (Yalavarthi et al., 2024), and patch-based (Liu et al., 2026b) models. However, compared with the rapid progress in model design, evaluation metrics for irregular time-series forecasting remain insufficiently studied.

![](images/037fe76a3e8ef3e37ad8e212faf7b0333c0dfa5934f358dcdf23692849d4cf49.jpg)  
Figure 1. Illustration of evaluation bias under non-uniform sampling. Although Model B has lower continuous-time error, Model A obtains a lower MSE because of over-dense sampling within the interval [0.0, 0.4].

For time-series forecasting, an ideal evaluation metric is the continuous-time discrepancy between the predicted trajectory and the true trajectory X over the entire future interval. Given X over a time span T, let ℓ(t, X) denote the prediction error of a model at time t. This continuous-time discrepancy is defined as the continuous-time risk:

$$
R _ { \mathrm { c t } } = \mathbb { E } _ { X } \left[ { \frac { 1 } { | T | } } \int _ { T } \ell ( t , X ) d t \right] .\tag{1}
$$

This risk directly characterizes the model’s continuous-time predictive performance over the future interval.

Existing studies typically adopt mean squared error (MSE) and mean absolute error (MAE), which are widely used in regularly sampled forecasting. We argue that in irregular settings, MSE and MAE can be affected by the samplespecific timestamp sampling distributions and may therefore fail to accurately reflect a model’s performance. As illustrated in Fig. 1, although Model B has a lower continuous-time error than Model A (i.e., the orange line of Model B is closer to the black ground truth line over the whole time span), Model A obtains a lower MSE because of over-dense sampling within the interval $[ 0 . 0 , 0 . 4 ]$ . By theoretical analysis, we find that MSE provides a biased estimate of the continuous-time risk $R _ { c t } .$ , as it actually measures the observation-time risk $R _ { \mathrm { o b s } } = \mathbb { E } _ { X } [ \int _ { T } \ell ( t , X ) p ( t \mid X ) d t ]$ where $p ( t \mid X )$ is the conditional sampling distribution for the timestamps. This distribution is sample-specific and thus causes model performance evaluations to be confounded by sample-specific sampling patterns.

To eliminate the influence of the timestamp sampling distributions on evaluation results, we begin directly from $R _ { \mathrm { c t } }$ and construct its empirical estimator, termed the Continuoustime Squared Error (CSE), through importance sampling. CSE reweights errors according to the observation-time density $p ( t \mid X )$ , reducing the contribution of densely sampled regions while increasing that of sparsely sampled regions, thereby mitigating evaluation bias caused by timestamp sampling distributions. Returning to Fig. 1, CSE correctly identifies Model B as having the lower continuous-time error. We further prove that the asymptotic estimation error of CSE with respect to the continuous-time risk is no greater than that of MSE. Finally, we further decompose the discrepancy between MSE and CSE into a sampling-dependence gap and a temporal-distribution gap to investigate their respective effects in real-world settings. Similarly, we can derive the Continuous-time Absolute Error (CAE) as the debiased counterpart to MAE.

To evaluate the effectiveness of CSE, we require access to the true trajectory X of each dataset. To this end, we construct a systematic benchmark for irregular time-series forecasting covering synthetic, semi-synthetic, and real-world datasets. Both the synthetic and semi-synthetic datasets have the true continuous or regular trajectory. Based on a comprehensive evaluation of this benchmark, we have the following conclusions: (1) on synthetic and semi-synthetic datasets, CSE can recover continuous-time risk more accurately than MSE under non-uniform sampling; (2) on realworld datasets, relying solely on MSE may not fully reflect models’ continuous-time predictive performance; (3) the discrepancies between MSE and CSE are jointly influenced by the sampling-dependence and temporal-distribution gaps.

The main contributions are summarized as follows:

• Revealing the biased evaluation of MSE for model’s continuous-time predictive performance. We show that MSE estimates the observation-distribution risk rather than the continuous-time risk, thereby coupling model evaluation with sampling mechanisms.

• A continuous-time evaluation metric with theoretical guarantees. We propose Continuous-Time Squared Error (CSE), which estimates the continuoustime risk from irregular timestamps through importance weighting, and prove that its asymptotic estimation error is no greater than MSE.

• A systematic benchmark for continuous-time evaluation. We construct a benchmark covering synthetic, semi-synthetic, and real-world datasets to validate the effectiveness of CSE, systematically evaluate models continuous-time predictive performance, and analyze the sources of discrepancies between MSE and CSE.

## 2. Related Work

## 2.1. Irregular Time Series Forecasting

Irregular time series are typically characterized by sparse, asynchronous, and nonuniform sampling. Existing studies have primarily focused on modeling continuous-time dynamics from limited observations. One line of work explicitly describes the continuous evolution of latent states. For example, Latent ODE, Neural CDE, and Continuous Recurrent Unit handle arbitrary time intervals based on ordinary differential equations, controlled differential equations, and continuous state transitions, respectively (Rubanova et al., 2019; Kidger et al., 2020; Schirmer et al., 2022). Another line of work directly models irregular observations together with their temporal information. SeFT represents a time series as a set of observations, mTAN employs multi-time attention to aggregate asynchronous observations, and Raindrop uses graph structures to capture dynamic dependencies among variables (Horn et al., 2020; Shukla & Marlin, 2021; Zhang et al., 2022). More recent methods further integrate multiscale representations with graph structures. For instance, Warpformer models temporal patterns at different scales through time warping, whereas GraFITi formulates irregular forecasting as an information propagation process over a graph (Zhang et al., 2023; Yalavarthi et al., 2024).

Despite these advances, existing studies primarily focus on model design and typically evaluate predictions using MSE or MAE at observed future timestamps. In contrast, we study the evaluation protocol itself and examine how irregular sampling affects model comparison and selection.

## 2.2. Benchmarks and Evaluation for Irregular Time Series

With the growing interest in irregular time series, prior studies have developed related benchmarks from the perspectives of data resources, software frameworks, and standardized experimental settings. The MIMIC-IV benchmark organizes unified tasks for sparse and irregular clinical time series and compares a range of representative models (Bui et al., 2024). PYRREGULAR provides unified data structures and analytical tools, together with a standardized benchmark covering multiple datasets and classification methods (Spinnato & Landi, 2025). Time-IMM further introduces new datasets and evaluation benchmarks for irregular multimodal multivariate time series (Chang et al., 2025). For forecasting tasks, Physiome-ODE constructs controlled datasets from biological ODEs to provide more challenging and discriminative environments for comparing irregular forecasting models (Klotergens et al.¨ , 2025).

Existing benchmarks mainly improve datasets, tasks, and experimental standardization rather than the evaluation objective itself. Physiome-ODE is the most closely related work: it provides controlled ODE-generated trajectories to improve model discrimination, but still evaluates predictions using MSE at sampled query times (Klotergens et al.¨ , 2025). Consequently, it does not examine whether model conclusions depend on the query-time distribution or recover risk over the entire future interval. In contrast, we take continuous-time risk as the evaluation target, propose CSE to estimate it, and decompose the resulting evaluation discrepancy into sampling-dependence and temporaldistribution gaps.

## 3. Preliminaries

## 3.1. Irregular Time Series Forecasting

The dataset contains S irregular time series. For the sth sequence, its historical observations are denoted by $\mathcal { O } _ { s } = \dot { \{ ( q _ { s , i } , \mathbf { x } _ { s , i } ) \} } _ { i = 1 } ^ { H _ { s } }$ , where $q _ { s , i } \in \mathbb { R }$ and $\mathbf { x } _ { s , i } \in \mathbb { R } ^ { D }$ denote the irregular timestamp and the corresponding multivariate observation, respectively. Given the historical observations $\mathcal { O } _ { s } .$ , a forecasting model $f _ { \theta }$ aims to estimate the latent process over a future time interval T . For any query time $t \in \mathcal T$ , the model output is given by $\hat { \mathbf { X } } _ { s } ( t ) = f _ { \theta } ( \mathcal { O } _ { s } , t )$ Let ${ \bf X } _ { s } ( t )$ denote the ground-truth latent process, and define the prediction error at time t as $\ell ( t , X _ { s } ) = \mathcal { L } ( \mathbf { X } _ { s } ( t ) , \hat { \mathbf { X } } _ { s } ( t ) )$ . In practical test datasets, the ground-truth values are available only at a finite set of future timestamps $\{ t _ { s , j } \} _ { j = 1 } ^ { L _ { s } } ,$ which are generated from the conditional sampling distribution $p ( t \mid X _ { s } )$ .

## 3.2. MSE and MAE under Irregular Sampling

Existing studies on irregular time series forecasting typically compute MSE and MAE at the actually observed future timestamps. In the following, we use MSE as the example to illustrate the idea, i.e., $\ell ( t , X _ { s } ) = \left\| \mathbf { X } _ { s } ( t ) - \hat { \mathbf { X } } _ { s } ( t ) \right\| ^ { 2 }$ Accordingly, the MSE on the test set can be written as

$$
\mathrm { M S E } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \frac { 1 } { L _ { s } } \sum _ { j = 1 } ^ { L _ { s } } \ell ( t _ { s , j } , X _ { s } ) .\tag{2}
$$

This metric first averages the prediction errors over the future observations of each trajectory and then assigns equal weight to different trajectories. When the future timestamps

are regarded as conditional samples drawn from $p ( t \mid X )$ the above expression is essentially a Monte Carlo estimator of the following observation-distribution risk:

$$
R _ { \mathrm { o b s } } = \mathbb { E } _ { X } \left[ \int _ { \mathcal { T } } \ell ( t , X ) p ( t \mid X ) d t \right] = \mathbb { E } _ { X } \left[ \mathbb { E } _ { T \mid X } [ \ell ( T , X ) ] \right] .\tag{3}
$$

Therefore, MSE aggregates prediction errors according to the empirical frequency of future timestamps: densely observed temporal regions receive higher weights in the evaluation, whereas sparsely observed regions receive lower weights. In other words, the overall evaluation objective of MSE is determined not only by the model’s prediction errors but also by the conditional sampling distribution $p ( t \mid X )$ .

## 4. Continuous-Time Risk for Irregular Forecasting

In this section, we first define the continuous-time risk that directly corresponds to the continuous-time forecasting objective. We then construct its empirical estimator, CSE, using importance sampling and analyze the asymptotic properties of CSE relative to MSE. Finally, we introduce the globaltime risk and decompose the evaluation discrepancy into two components associated with sample-dependent sampling and population-level temporal nonuniformity.

## 4.1. Continuous-Time Risk

As discussed above, the observation-distribution risk corresponding to MSE is affected by the conditional sampling distribution $p ( t \mid X )$ , thereby coupling model evaluation with sample-specific sampling mechanisms. A natural way to reduce this dependence is to adopt a common target-time measure across all trajectories. For continuous-time forecasting, an ideal evaluation objective should also measure the discrepancy between the predicted trajectory and the ground-truth process over the entire future interval. We therefore adopt the uniform temporal measure and define

$$
R _ { \mathrm { c t } } = \mathbb { E } _ { X } \left[ { \frac { 1 } { | T | } } \int _ { T } \ell ( t , X ) d t \right] .\tag{4}
$$

$R _ { \mathrm { c t } }$ assigns equal weight to each unit of time within the future interval and is independent of the sampling distribution of the actual observation times. It therefore characterizes the forecasting capability of a model over the entire continuoustime process.

## 4.2. Estimating Continuous-Time Risk

The test set provides ground-truth observations only at a finite number of future timestamps, making it impossible to directly compute the continuous-time integral over the entire interval T . To estimate $R _ { \mathrm { c t } }$ from the available irregular observations, we use the actual conditional sampling distribution $p ( t \mid X )$ as the proposal distribution and rewrite the continuous-time risk through importance sampling as

$$
\begin{array} { l } { \displaystyle { R _ { \mathrm { c t } } = \mathbb { E } _ { X } \left[ \int _ { \mathcal { T } } \ell ( t , X ) \frac { 1 } { | { \mathcal T } | p ( t \mid X ) } p ( t \mid X ) d t \right] } } \\ { \displaystyle { ~ = \mathbb { E } _ { X } \left[ \mathbb { E } _ { T | X } \left[ \ell ( T , X ) \frac { 1 } { | { \mathcal T } | p ( T \mid X ) } \right] \right] . } } \end{array}\tag{5}
$$

Here, $1 / ( | \mathcal { T } | p ( t \mid X ) )$ corrects the actual conditional sampling distribution toward the uniform temporal distribution. It reduces the contribution of each observation in densely sampled regions while increasing the contribution of each observation in sparsely sampled regions, allowing each temporal location to contribute to the evaluation according to the length of the time interval it represents.

In practical datasets, the conditional sampling density p(t | $X _ { s } )$ is typically unknown. Let $\hat { p } ( t _ { s , i } \mid X _ { s } )$ denote the conditional temporal density estimated from the future timestamps of the s-th trajectory. Since standard importance sampling may exhibit high finite-sample variance due to extreme weights, we adopt self-normalized importance sampling and define the resulting empirical metric under squared error as the Continuous-Time Squared Error (CSE):

$$
\mathrm { C S E } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \frac { \sum _ { i = 1 } ^ { L _ { s } } \ell ( t _ { s , i } , X _ { s } ) / \hat { p } ( t _ { s , i } \mid X _ { s } ) } { \sum _ { i = 1 } ^ { L _ { s } } 1 / \hat { p } ( t _ { s , i } \mid X _ { s } ) } .\tag{6}
$$

The constant $1 / | \mathcal { T } |$ in the uniform temporal density cancels between the numerator and denominator. For each trajectory, a continuous-time error estimate is first obtained through self-normalized weighting, after which these estimates are averaged equally across all test trajectories. The same construction naturally yields the Continuous-time Absolute Error (CAE) under absolute error. In the following, we focus on CSE under squared error.

To reduce the influence of an evaluation timestamp on its own density estimate, we employ leave-one-out Gaussian kernel density estimation:

$$
\hat { p } ( t _ { s , i } \mid X _ { s } ) = \frac { 1 } { ( L _ { s } - 1 ) h _ { s } } \sum _ { j = 1 \atop j \neq i } ^ { L _ { s } } K \left( \frac { t _ { s , i } - t _ { s , j } } { h _ { s } } \right) ,\tag{7}
$$

where $K ( \cdot )$ is the Gaussian kernel function and $h _ { s }$ is the bandwidth associated with the s-th trajectory. Excluding the current evaluation point prevents it from introducing additional kernel mass at its own location, thereby yielding a more robust estimate of the local temporal density.

To further demonstrate the advantage of CSE over MSE, we theoretically compare their asymptotic estimation errors with respect to the continuous-time risk and introduce the following assumption and theorem.

Assumption 4.1. Let $\tau$ be a compact interval, and assume that different test trajectories are mutually independent. Conditional on $X _ { s } ,$ , the future timestamps are independently and identically distributed according to $p ( t \mid X _ { s } )$ . Suppose that the loss function is bounded, the conditional sampling density satisfies $p ( t \mid X ) \geq \rho > 0$ , and the density estimator ${ \hat { p } } ( t \mid X )$ is strongly consistent over $\tau$

Theorem 4.2 (Asymptotic Non-Inferiority of CSE). Let $\underline { { L } } = \operatorname* { m i n } _ { 1 \le s \le S } L _ { s }$ . Under Assumption 4.1, as both the number of test trajectories S and the minimum number of future observations per trajectory $\underline { { L } }$ tend to infinity, the following inequality holds almost surely:

$$
\operatorname* { l i m } _ { S , L \to \infty } \left| \mathrm { C S E } - R _ { \mathrm { c t } } \right| \le \operatorname* { l i m } _ { S , L \to \infty } \left| \mathrm { M S E } - R _ { \mathrm { c t } } \right| .\tag{8}
$$

In particular, equality holds when $p ( t \mid X ) = 1 / | T |$ almost everywhere.

The complete proof is provided in the Appendix A. Theorem 4.2 shows that, under the stated conditions, the asymptotic estimation error of CSE with respect to the continuoustime risk is no greater than that of MSE. When future observation times are uniformly distributed, the two metrics are asymptotically equivalent. When the observationdistribution risk differs from the continuous-time risk, CSE achieves a strictly smaller asymptotic error.

## 4.3. Sources of Evaluation Discrepancy

Because MSE and CSE converge to the observationdistribution risk and the continuous-time risk, respectively, they may yield different model evaluation results. To further analyze this discrepancy, we introduce the global-time risk and decompose the overall discrepancy into two components arising from sample-dependent sampling and populationlevel temporal nonuniformity.

Marginalizing the conditional sampling distribution over trajectories yields $p _ { T } ( t ) = \mathbb { E } _ { X } [ p ( t \mid X ) ]$ , which describes the distribution of timestamps at the dataset level. Based on this distribution, we define the global-time risk as

$$
R _ { \mathrm { g l o b a l } } = \mathbb { E } _ { X } \left[ \int _ { T } \ell ( t , X ) p _ { T } ( t ) d t \right] .\tag{9}
$$

$R _ { \mathrm { g l o b a l } }$ removes the influence of sample-specific sampling mechanisms while retaining the population-level nonuniform temporal coverage of the dataset. In this sense, it preserves the global temporal distribution induced by the data collection process, but eliminates the instance-level irregularity that may bias model evaluation. It therefore serves as an intermediate reference between the observed risk $R _ { \mathrm { o b s } }$ and the continuous-time risk $R _ { \mathrm { c t } }$ . Based on this risk, the overall evaluation discrepancy can be decomposed

as

wher

$$
\begin{array} { r l } & { R _ { \mathrm { o b s } } - R _ { \mathrm { c t } } = \underbrace { R _ { \mathrm { o b s } } - R _ { \mathrm { g l o b a l } } } _ { G _ { \mathrm { s a m p } } } + \underbrace { R _ { \mathrm { g l o b a l } } - R _ { \mathrm { c t } } } _ { G _ { \mathrm { t i m e } } } , } \\ { \mathrm { ~ } } \\ { \mathrm { ~ } } \\ { \mathrm { ~ } } & { \mathrm { ~ \it ~ \it ~ \mathscr ~ { ~ G ~ } ~ } _ { \mathrm { a m p } } = \int _ { \mathcal T } \mathrm { C o v } _ { X } \left( \ell ( t , X ) , p ( t | X ) \right) d t , } \\ { \mathrm { ~ } } & { \mathrm { ~ \it ~ \mathscr ~ { ~ G ~ } ~ } _ { \mathrm { i m e } } = \mathrm { C o v } _ { U \sim \mathrm { U n i f } ( \mathcal T ) } \left( \bar { \ell } ( U ) , | \mathcal { T } | p _ { T } ( U ) \right) } \end{array}\tag{10}
$$

where $\bar { \ell } ( t ) = \mathbb { E } _ { X } [ \ell ( t , X ) ]$ . The detailed derivation is provided in the Appendix B.

$G _ { \mathrm { s a m p } }$ is determined by the covariance, across trajectories, between the model error $\ell ( t , X )$ and the sample-specific sampling density $p ( t \mid X )$ at a fixed temporal location. If trajectories with larger errors are more likely to be observed at that time, this term is positive; otherwise, it is negative. When the sampling process is independent of the trajectory, $\mathsf { i . e . , } p ( t \mid X ) = p _ { T } ( t )$ , we have $G _ { \mathrm { s a m p } } = 0$ . Therefore, $G _ { \mathrm { s a m p } }$ characterizes the effect induced by sampledependent sampling.

$G _ { \mathrm { t i m e } }$ is determined by the covariance, across time, between the average model error $\bar { \ell } ( t )$ and the population-level relative sampling density $| \mathcal { T } | p _ { T } ( t )$ . If temporal regions with larger average errors have higher population-level sampling densities, this term is positive; otherwise, it is negative. When $p _ { T } ( t ) \ : = \ : 1 / | T |$ , we have $G _ { \mathrm { t i m e } } = 0$ Therefore, $G _ { \mathrm { t i m e } }$ characterizes the actual effect of population-level temporal nonuniformity on the evaluation of the current model.

To estimate the above discrepancies in practice, we first estimate the marginal temporal density using $\hat { p } _ { T } ( t ) ~ =$ $\textstyle { \frac { 1 } { S } } \sum _ { r = 1 } ^ { S } { \hat { p } } ( t \mid X _ { r } )$ . We then use ${ \hat { p } } _ { T } ( t )$ as the target distribution to construct a self-normalized importance-sampling estimator of $R _ { \mathrm { g l o b a l } } .$ , termed the Global-Time Squared Error (GSE). Its detailed form is provided in the Appendix C. The two discrepancy components can then be estimated as

$$
\widehat { G } _ { \mathrm { s a m p } } = \mathrm { M S E - G S E } , \qquad \widehat { G } _ { \mathrm { t i m e } } = \mathrm { G S E - C S E } .\tag{11}
$$

This decomposition enables us to separately quantify the effects of sample-dependent sampling and population-level temporal nonuniformity on model scores and rankings in practice, providing a basis for analyzing evaluation discrepancies in the subsequent experiments.

## 5. Benchmark Protocol

This section presents the benchmark setup, including the synthetic, semi-synthetic, and real-world datasets, representative baseline models, and unified training and evaluation protocols adopted to ensure fair and consistent comparisons.

## 5.1. Datasets

Synthetic and Semi-Synthetic Data. To obtain directly computable continuous-time risks, we construct two fully synthetic datasets and two semi-synthetic datasets. Synthetic-Regime gradually transitions from a smooth, lowamplitude regime to a high-frequency periodic regime, simulating a process whose forecasting difficulty varies substantially over time. Synthetic-Multiscale simultaneously contains trends, multiscale periodic patterns, and local transient variations, covering more complex temporal patterns. Both fully synthetic datasets provide analytical continuous ground truth that can be queried at arbitrary future times. The semi-synthetic datasets are constructed from the regularly sampled ETTm1 and Weather datasets (Wu et al., 2021). Specifically, we sample a finite number of historical and future timestamps from complete regular windows to form irregular samples, and use the equally weighted error over the complete future grid as the reference continuoustime risk. Training target times uniformly cover the future interval, whereas test timestamps are sampled from

$$
p _ { \alpha } ( u ) = \alpha \mathrm { B e t a } ( u ; 1 0 , 6 0 ) + ( 1 - \alpha ) \mathrm { U n i f o r m } ( 0 , 1 ) ,\tag{12}
$$

where $\alpha \in \{ 0 , 0 . 3 , 0 . 5 , 0 . 9 , 0 . 9 9 \}$ controls the degree of temporal nonuniformity. All sampling conditions share the same training data and model predictions, while only the test-time distribution is varied, thereby isolating the effect of the evaluation protocol itself. Detailed data-generation procedures and window configurations are provided in the Appendix D.1.

Real-World Datasets. We select eight real-world irregular time series datasets spanning different application domains and sampling patterns. USHCN contains longterm climate observations from weather stations across the United States (Menne et al., 2015). MIMIC-III consists of sparse clinical records from ICU patients (Johnson et al., 2016). HumanActivity contains irregularly collected location-sensor measurements recorded during human activities (Li et al., 2025). GDELT records multivariate event sequences triggered by international events (Leetaru & Schrodt, 2013). RepoHealth describes software repository development activities, whose sampling frequency varies with project activity (Chang et al., 2025). StudentLife contains mobile sensing data driven by user behavior and daily routines (Nepal et al., 2024). FNSPID consists of stock prices and trading information and exhibits structured temporal gaps induced by trading sessions (Chang et al., 2025). CESNET contains traffic records generated by network devices, with timestamps affected by system scheduling delays and logging jitter (Chang et al., 2025). For all datasets, we use only numerical time series and follow the corresponding public settings for preprocessing and data splitting. Detailed dataset statistics, preprocessing procedures, and window configurations are provided in the Appendix D.2.

## 5.2. Baselines

We select eleven representative irregular time series models covering different temporal modeling paradigms. These include recurrent or continuous-time models, namely GRU-D and NeuralFlow (Che et al., 2016; Bilos et al., 2021); set-, attention-, or Transformer-based models, including SeFT, mTAN, and Warpformer (Horn et al., 2020; Shukla & Marlin, 2021; Zhang et al., 2023); graph- or hypergraphbased models, including GraFITi, HyperIMTS, tPatchGNN, and ASTGI (Yalavarthi et al., 2024; Li et al., 2025; Zhang et al., 2024; Liu et al., 2026a); and pre-alignment and patchaggregation methods, including KAFNet and APN (Zhou et al., 2026; Liu et al., 2026b). Detailed descriptions and hyperparameter configurations for all models are provided in the Appendix D.3.

## 5.3. Training and Evaluation Protocol

All models use the same data splits and input information, and their training objective is uniformly set to the MSE over observed timestamps. The remaining training and hyperparameter settings follow the corresponding public implementations, with model selection performed according to validation MSE. In each independent run, every model produces only one set of test predictions, on which MSE, GSE, and CSE are computed. This ensures that the differences among the metrics arise solely from the temporal weighting applied during test-time evaluation. Both the conditional and marginal temporal densities are estimated exclusively from the test timestamps and are shared across all models on the same dataset. We further compute $\widehat { G } _ { \mathrm { s a m p } }$ and $\widehat { G } _ { \mathrm { t i m e } }$ to analyze the sources of evaluation discrepancies. All experiments are independently repeated using five random seeds. The main text reports the mean results, while complete results with standard deviations are provided in the Appendix. Detailed training configurations, density-estimation parameters, and other implementation details are also provided in the Appendix D.4.

## 6. Experiments

Using the proposed benchmark, we conduct experiments to investigate three questions: (1) Can CSE recover continuoustime risk and model rankings more accurately than MSE under non-uniform sampling? (2) Does observation-point MSE fully reflect models’ continuous-time predictive performance on real-world datasets? (3) How do the samplingdependence and temporal-distribution gaps influence MSE– CSE discrepancies?

## 6.1. Can CSE Better Recover Continuous-Time Risk and Rankings than MSE?

In this section, we evaluate the ability of CSE to recover continuous-time risks and model rankings in controlled environments with complete future ground truth. The experiments include two fully synthetic datasets and two semisynthetic datasets, and consider five representative models: SeFT, GRU-D, GraFITi, tPatchGNN, and KAFNet. For the fully synthetic datasets, we compute $R _ { \mathrm { c t } }$ through dense temporal integration. For the semi-synthetic datasets, we compute the reference risk over the complete regular future grid. The main text reports the results on Synthetic-Regime and Weather, while the remaining results are provided in the Appendix E.1.

Fig. 4(a) and Fig. 4(b) show the estimation errors of MSE and CSE with respect to $R _ { \mathrm { c t } }$ . Under uniform sampling, both metrics accurately recover the continuous-time risk. As α increases, the test-time distribution progressively deviates from the uniform distribution, causing the discrepancy between MSE and $R _ { \mathrm { c t } }$ to increase consistently, whereas CSE maintains a substantially lower estimation error. For example, when $\alpha = 0 . 9$ , the relative errors of MSE and CSE on Synthetic-Regime are 80.6% and 34.7%, respectively. Under the more extreme setting of $\alpha = 0 . 9 9$ , although the error of CSE increases, it remains substantially lower than that of MSE.

To further assess the recovery of model rankings, Table 2 compares the evaluation results of the five models under $\alpha = 0 . 9$ . On both Synthetic-Regime and Weather, CSE recovers exactly the same ranking of all five models as $R _ { \mathrm { c t } }$ whereas MSE produces a different ranking on both datasets.

Overall, CSE effectively mitigates the influence of nonuniform test sampling on model evaluation. Across both synthetic and semi-synthetic datasets, CSE recovers continuoustime risk and model rankings more accurately than MSE, validating its effectiveness as a continuous-time evaluation metric.

## 6.2. Does MSE Fully Reflect Continuous-Time Predictive Performance on Real-World Data?

The previous section shows that CSE more accurately recovers continuous-time risk and model rankings when the underlying future trajectory is available. We now examine whether observation-point MSE provides a sufficient evaluation of models’ continuous-time predictive performance on real-world datasets, where the complete continuous trajectory is unavailable. Table 1 reports the results of 11 models on eight datasets, while Fig. 3 summarizes pairwise rank inversions and changes in the selection of the best-performing model. Comparisons between MAE and CAE, together with complete results including standard deviations, are provided

Table 1. Forecasting performance under MSE and CSE on eight real-world datasets. Superscripts denote ranks. The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2"></td><td colspan="2">CESNET ×10−1</td><td colspan="2">FNSPID ×10−1</td><td colspan="2">GDELT ×100</td><td colspan="2">HumanActivity ×10−2</td><td colspan="2">MIMIC-III ×10-1</td><td colspan="2">RepoHealth ×10−1</td><td colspan="2">StudentLife ×10−1</td><td colspan="2">USHCN ×10−1</td><td rowspan="2">Average</td></tr><tr><td>MSE</td><td>CSE</td><td>MSE</td><td>CSE</td><td>MSE</td><td>CSE</td><td>MSE</td><td>CSE</td><td>MSE</td><td>CSE</td><td>MSE</td><td>CSE</td><td>MSE</td><td>CSE</td><td>MSE</td><td>CSE</td></tr><tr><td>APN</td><td>8.76(5)</td><td>8.58(5)</td><td>2.38(2)</td><td>2.38(2)</td><td>1.02(1)</td><td>0.99(1)</td><td>5.68(6)</td><td>5.65(6)</td><td>6.66(6)</td><td>6.72(6)</td><td>3.36(2)</td><td>3.57(1)</td><td>6.61(2)</td><td>6.68(3)</td><td>6.04(4)</td><td>6.71(5)</td><td>3.6</td></tr><tr><td>ASTGI</td><td>8.47(1)</td><td>8.26(1)</td><td>3.28(6)</td><td>3.25(6)</td><td>1.10(4)</td><td>1.04(4)</td><td>5.46(3)</td><td>5.41(3)</td><td>10.50(11)</td><td>10.62(11)</td><td>6.27(7)</td><td>6.53(7)</td><td>6.65(3)</td><td>6.65(2)</td><td>5.14(3)</td><td>5.66(3)</td><td>4.7</td></tr><tr><td>GRU-D</td><td>10.22(10)</td><td>9.91(10)</td><td>5.24(8)</td><td>5.20(8)</td><td>1.22(10)</td><td>1.15(10)</td><td>8.56(9)</td><td>8.56(9)</td><td>6.67(7)</td><td>6.75(7)</td><td>11.68(11)</td><td>11.95(11)</td><td>7.21(8)</td><td>7.23(8)</td><td>7.78(11)</td><td>8.59(11)</td><td>9.3</td></tr><tr><td>GraFITi</td><td>8.50(2)</td><td>8.28(2)</td><td>3.21(4)</td><td>3.22(4)</td><td>1.09(3)</td><td>1.01(2)</td><td>5.44(2)</td><td>5.40(2)</td><td>5.89(1)</td><td>6.02(2)</td><td>3.32(1)</td><td>3.61(2)</td><td>6.72(6)</td><td>6.70(4)</td><td>4.50(1)</td><td>4.89(1)</td><td>2.4</td></tr><tr><td>HyperIMTS</td><td>8.90(7)</td><td>8.65(7)</td><td>2.72(3)</td><td>2.71(3)</td><td>1.18(8)</td><td>1.09(8)</td><td>5.42(1)</td><td>5.38(1)</td><td>5.94(2)</td><td>5.92(1)</td><td>3.72(3)</td><td>3.97(3)</td><td>6.54(1)</td><td>6.56(1)</td><td>6.67(8)</td><td>7.30(8)</td><td>4.1</td></tr><tr><td>KAFNet</td><td>8.73(4)</td><td>8.54(4)</td><td>2.23(1)</td><td>2.22(1)</td><td>1.08(2)</td><td>1.02(3)</td><td>5.62(4)</td><td>5.58(4)</td><td>6.41(3)</td><td>6.47(3)</td><td>5.72(6)</td><td>5.98(6)</td><td>6.66(4)</td><td>6.72(5)</td><td>7.14(9)</td><td>7.82(9)</td><td>4.3</td></tr><tr><td>NeuralFlow</td><td>9.98(8)</td><td>9.69(8)</td><td>7.33(10) 7.29(10)</td><td></td><td>1.18(9)</td><td>1.09(9)</td><td>17.53(11)</td><td>17.51(11)</td><td>7.05(8)</td><td>7.19(9)</td><td>10.13(10)</td><td>10.44(10)</td><td></td><td>7.83(10) 7.86(10)</td><td>6.07(5)</td><td>6.74(6)</td><td>9.0</td></tr><tr><td>SeFT</td><td>10.11(9)</td><td>9.79(9)</td><td>8.70(11) 8.66(11)</td><td></td><td>1.13(6)</td><td>1.05(5)</td><td>9.38(10)</td><td>9.36(10)</td><td>7.10(9)</td><td>7.15(8)</td><td>9.13(8)</td><td>9.44(8)</td><td></td><td></td><td>8.94(11) 8.95(11) 7.64(10) 8.41(10)</td><td></td><td>9.1</td></tr><tr><td>Warpformer</td><td>8.58(3)</td><td>8.37(3)</td><td>3.66(7)</td><td>3.63(7)</td><td>1.30(11)</td><td>1.20(11)</td><td>5.65(5)</td><td>5.61(5)</td><td>6.42(4)</td><td>6.50(5)</td><td>5.04(5)</td><td>5.33(5)</td><td>6.78(7)</td><td>6.80(7)</td><td>6.13(6)</td><td>6.67(4)</td><td>5.9</td></tr><tr><td>mTAN</td><td>10.47(11)</td><td>10.14(11)</td><td>6.13(9)</td><td>6.09(9)</td><td>1.12(5)</td><td>1.05(6)</td><td>7.33(8)</td><td>7.30(8)</td><td>7.47(10)</td><td>7.54(10)</td><td>10.09(9)</td><td>10.39(9)</td><td>7.32(9)</td><td>7.34(9)</td><td>4.84(2)</td><td>5.23(2)</td><td>7.9</td></tr><tr><td>tPatchGNN</td><td>8.86(6)</td><td>8.61(6)</td><td>3.26(5)</td><td>3.24(5)</td><td>1.15(7)</td><td>1.08(7)</td><td>5.80(7)</td><td>5.76(7)</td><td>6.44(5)</td><td>6.48(4)</td><td>4.55(4)</td><td>4.82(4)</td><td>6.68(5)</td><td>6.73(6)</td><td>6.36(7)</td><td>7.05(7)</td><td>5.8</td></tr></table>

![](images/7f6eab3fffb9c04e3e136beac31a128ed4b54c23f623e8710cfaa18d1f66ce5b.jpg)  
(a) Sampling-dependence gap $\widehat { G } _ { \mathrm { s a m p } }$

![](images/c039729ebd4cb6ba15159171ae8470ec0d0c29222e71ef07ed4898a85fc8742b.jpg)  
(b) Temporal-distribution gap $\widehat { G } _ { \mathrm { t i m e } }$  
Figure 2. Normalized decomposition of the evaluation discrepancy into the sampling-dependence gap $\widehat { G } _ { \mathrm { s a m p } }$ and temporal-distribution gap $\widehat { G } _ { \mathrm { t i m e } }$ across models and datasets.

Table 2. Model evaluation under non-uniform test sampling with α = 0.9. Superscripts denote ranks; the best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Model</td><td colspan="3">Synthetic-Regime  $\times 1 0 ^ { - 2 }$ </td><td colspan="3">Weather  $\times 1 0 ^ { - 1 }$ </td></tr><tr><td>MSE</td><td>CSE</td><td> $R _ { \mathrm { c t } }$ </td><td>MSE</td><td>CSE</td><td> $R _ { \mathrm { c t } }$ </td></tr><tr><td>SeFT</td><td>1.09(3)</td><td>8.70(1)6.26(1)4.40(5)</td><td></td><td></td><td>4.96(5) 4.85(5)</td><td></td></tr><tr><td>GRU-D</td><td>0.65(1)</td><td></td><td>8.81(2) 6.48(2) 3.08(1) 4.61(3) 4.46(3)</td><td></td><td></td><td></td></tr><tr><td>GraFITi</td><td>2.57(5)</td><td>9.88(5) 7.66(5) 3.89(4) 4.92(4) 4.76(4)</td><td></td><td></td><td></td><td></td></tr><tr><td>tPatchGNN 0.91(2) 9.38(4) 6.91(4) 3.88(3) 3.69(1) 3.53(1)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>KAFNet</td><td></td><td></td><td></td><td></td><td>1.54(4) 9.04(3) 6.75(3) 3.48(2) 3.78(2) 3.65(2)</td><td></td></tr></table>

## in the Appendix E.2.

The comparison shows that CSE is not a fixed-direction or fixed-ratio correction of MSE. The direction and magnitude of score changes vary across both datasets and models, indicating that the two metrics cannot be converted through simple rescaling. Moreover, five of the eight datasets exhibit pairwise rank inversions when replacing MSE with CSE. MIMIC-III and StudentLife each contain three inversions, GDELT and USHCN each contain two, and RepoHealth contains one. The best-performing model also changes on MIMIC-III and RepoHealth.

The effects further differ across models. GraFITi ranks first on three datasets under MSE but on only one dataset under CSE. In contrast, the number of first-place rankings increases from two to three for HyperIMTS and from one to two for APN. Although GraFITi retains the best average rank, MSE and CSE do not provide fully consistent conclusions regarding the relative advantages of different models.

Overall, the substantial discrepancies in model scores, rankings, and model selection suggest that relying solely on observation-point MSE may not fully reflect models’ continuous-time predictive performance on real-world

![](images/4b203e96b5f2723124e87dcee5fef90ea7e23c5ba732c453f640b3cceccd80f0.jpg)  
(a) Pairwise rank inversions across datasets

![](images/4d889763195a83162363753a578fbaadd6304db6b4287d7ddc6b9d9943878bb1.jpg)  
(b) Changes in Top-1 model selection  
Figure 3. Model ranking changes when replacing MSE with CSE on eight real-world datasets. Panel (a) reports the number of pairwise rank inversions, while Panel (b) compares the frequency with which each model is ranked first.

datasets.

## 6.3. How Do the Sampling-Dependence and Temporal-Distribution Gaps Influence MSE–CSE Discrepancies?

The previous section demonstrates that MSE and CSE can produce different model scores, rankings, and model selections on real-world datasets. We now use the risk decomposition introduced above to analyze the sources of these discrepancies. Fig. 2 reports the normalized sampling-dependence gap, $\widehat { G } _ { \mathrm { s a m p } } = \bar { ( } \mathrm { M S E - G S E ) / M S E }$ , and the normalized temporal-distribution gap, $\widehat { G } _ { \mathrm { t i m e } } = ( \mathrm { G S E - C S E } ) / \mathrm { G S E } .$

The relative contributions of the two gaps vary substantially across datasets. GDELT and RepoHealth are primarily affected by the sampling-dependence gap, whose cross-model averages are approximately 5.9% and −5.8%, respectively. CESNET is mainly affected by the temporaldistribution gap, with an average value of approximately 3.3%. On USHCN, the sampling-dependence and temporaldistribution gaps are approximately −7.8% and −2.2%, respectively, indicating that both components contribute to the discrepancy between MSE and CSE. The two gaps are comparatively small on the remaining datasets.

Within the same dataset, models generally exhibit similar signs for each discrepancy component, suggesting that the overall direction is largely determined by the dataset’s sampling structure. However, the magnitude varies across models because of their different temporal error patterns. This model-dependent effect can alter model rankings when the original performance differences are small. For example, on MIMIC-III, the sampling-dependence gaps of GraFITi and HyperIMTS are −2.2% and +0.8%, respectively, changing the best-performing model from GraFITi under MSE to HyperIMTS under CSE. On RepoHealth, GraFITi exhibits a more negative sampling-dependence gap than APN, giving it a larger apparent advantage under MSE; this advantage disappears under CSE, making APN the best-performing model.

![](images/54eeb36a11ced80a4308e9805aa4e0457035bcfaab74c3a777c13b3cd3f68292.jpg)  
(a) Synthetic-Regime

![](images/a2f113a5292e64ef1dfbb83fd635e6f73f3cf15119b9f3406891d66a16c27d8f.jpg)  
(b) Weather  
Figure 4. Relative errors of MSE and CSE with respect to $R _ { \mathrm { c t } }$ under increasing sampling non-uniformity.

Overall, the discrepancies between MSE and CSE are jointly influenced by the sampling-dependence and temporaldistribution gaps, whose relative contributions vary across datasets and models.

## 7. Conclusion

This work revisits the evaluation of irregular time series forecasting from a continuous-time perspective. We show that observation-point MSE and MAE effectively estimate the observation-distribution risk induced by the sampling distribution, rather than the continuous-time risk over the entire future interval. To mitigate this objective mismatch, we propose the Continuous-Time Squared Error (CSE), which estimates the continuous-time risk through self-normalized importance weighting, and prove that its asymptotic estimation error is no greater than that of MSE. We further decompose the discrepancy between the two risks into a sampling-dependence gap and a temporal-distribution gap, thereby characterizing how different sampling structures affect model evaluation. Finally, we construct a benchmark covering synthetic, semi-synthetic, and real-world datasets to validate CSE and systematically evaluate models’ continuous-time predictive performance.

## References

Afrifa-Yamoah, E., Mueller, U. A., Taylor, S. M., and Fisher, A. J. Missing data imputation of high-resolution temporal climate time series data. Meteorological Applications, 27 (1):e1873, 2020.

Bilos, M., Sommer, J., Rangapuram, S. S., Januschowski, T., and Gunnemann, S. Neural flows: Efficient alternative¨ to neural ODEs. In Advances in Neural Information Processing Systems, pp. 21325–21337, 2021.

Bui, H., Warrier, H., and Gupta, Y. Benchmarking with MIMIC-IV, an irregular, spare clinical time series dataset. arXiv preprint arXiv:2401.15290, 2024. doi: 10.48550/ arXiv.2401.15290.

Chang, C., Hwang, J., Shi, Y., Wang, H., Peng, W.-C., Chen, T.-F., and Wang, W. Time-IMM: A dataset and benchmark

for irregular multimodal multivariate time series. CoRR, abs/2506.10412, 2025.

Che, Z., Purushotham, S., Cho, K., Sontag, D. A., and Liu, Y. Recurrent neural networks for multivariate time series with missing values. CoRR, abs/1606.01865, 2016.

Decorte, T., Mortier, S., Lembrechts, J. J., Meysman, F. J. R., Latre, S., Mannens, E., and Verdonck, T. Missing´ value imputation of wireless sensor data for environmental monitoring. Sensors, 24(8):2416, 2024.

Ghassemi, M., Pimentel, M. A. F., Naumann, T., Brennan, T., Clifton, D. A., Szolovits, P., and Feng, M. A multivariate timeseries modeling approach to severity of illness assessment and forecasting in ICU with sparse, heterogeneous clinical data. In Proceedings ofthe Twenty-Ninth AAAI Conference on Artificial Intelligence, January 25- 30, 2015, Austin, Texas, USA, pp. 446–453, 2015.

Horn, M., Moor, M., Bock, C., Rieck, B., and Borgwardt, K. M. Set functions for time series. In International Conference on Machine Learning, pp. 4353–4363, 2020.

Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L.-w. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., and Mark, R. G. MIMIC-III, a freely accessible critical care database. Scientific Data, 3:160035, 2016. doi: 10.1038/sdata.2016.35.

Kidger, P., Morrill, J., Foster, J., and Lyons, T. J. Neural controlled differential equations for irregular time series. In Advances in Neural Information Processing Systems, 2020.

Klotergens, C., Yalavarthi, V. K., Scholz, R., Stubbemann,¨ M., Born, S., and Schmidt-Thieme, L. Physiome-ODE: A benchmark for irregularly sampled multivariate timeseries forecasting based on biological ODEs. In International Conference on Learning Representations, 2025.

Leetaru, K. and Schrodt, P. A. Gdelt: Global data on events, location, and tone. ISA Annual Convention, 2013.

Li, B., Luo, Y., Liu, Z., Zheng, J., Lv, J., and Ma, Q. Hyper-IMTS: Hypergraph neural network for irregular multivariate time series forecasting. In International Conference on Machine Learning, pp. 35502–35518, 2025.

Liu, X., Qiu, X., Cheng, H., Wu, X., Guo, C., Yang, B., and Hu, J. ASTGI: Adaptive spatio-temporal graph interactions for irregular multivariate time series forecasting. In International Conference on Learning Representations, 2026a.

Liu, X., Qiu, X., Wu, X., Li, Z., Guo, C., Hu, J., and Yang, B. Rethinking irregular time series forecasting: A simple yet effective baseline. In AAAI Conference on Artificial Intelligence, pp. 23873–23881, 2026b.

Menne, M. J., Williams, C. N. J., and Vose, R. S. Long-term daily and monthly climate records from stations across the contiguous united states (u.s. historical climatology network). Environmental System Science Data Infrastructure for a Virtual Ecosystem (ESS-DIVE), 2015.

Nepal, S., Liu, W., Pillai, A., Wang, W., Vojdanovski, V., Huckins, J. F., Rogers, C., Meyer, M. L., and Campbell, A. T. Capturing the college experience: a four-year mobile sensing study of mental health, resilience and behavior of college students during the pandemic. Proceedings of the ACM on interactive, mobile, wearable and ubiquitous technologies, 8(1):1–37, 2024.

Rubanova, Y., Chen, T. Q., and Duvenaud, D. Latent ordinary differential equations for irregularly-sampled time series. In Advances in Neural Information Processing Systems, pp. 5321–5331, 2019.

Schirmer, M., Eltayeb, M., Lessmann, S., and Rudolph, M. Modeling irregular time series with continuous recurrent units. In International Conference on Machine Learning, pp. 19388–19405, 2022.

Shukla, S. N. and Marlin, B. M. Multi-time attention networks for irregularly sampled time series. In International Conference on Learning Representations, 2021.

Spinnato, F. and Landi, C. PYRREGULAR: A unified framework for irregular time series, with classification benchmarks. CoRR, abs/2505.06047, 2025.

Wu, H., Xu, J., Wang, J., and Long, M. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pp. 22419–22430, 2021.

Yalavarthi, V. K., Madhusudhanan, K., Scholz, R., Ahmed, N., Burchert, J., Jawed, S., Born, S., and Schmidt-Thieme, L. GraFITi: Graphs for forecasting irregularly sampled time series. In AAAI Conference on Artificial Intelligence, pp. 16255–16263, 2024.

Zhang, J., Zheng, S., Cao, W., Bian, J., and Li, J. Warpformer: A multi-scale modeling approach for irregular clinical time series. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 3273–3285, 2023.

Zhang, W., Yin, C., Liu, H., Zhou, X., and Xiong, H. Irregular multivariate time series forecasting: A transformable patching graph neural networks approach. In International Conference on Machine Learning, pp. 60179– 60196, 2024.

Zhang, X., Zeman, M., Tsiligkaridis, T., and Zitnik, M. Graph-guided network for irregularly sampled multivariate time series. In International Conference on Learning Representations, 2022.

Zhou, Z., Huang, Y., Wang, Y., Wu, Y., Kwok, J., and Liang, Y. Revitalizing canonical pre-alignment for irregular multivariate time series forecasting. In AAAI Conference on Artificial Intelligence, pp. 29115–29123, 2026.

## A. Proof of Theorem 1

For the s-th trajectory, denote

$$
p _ { s } ( t ) = p ( t \mid X _ { s } ) , \qquad \ell _ { s } ( t ) = \ell ( t , X _ { s } ) ,\tag{13}
$$

and define

$$
r _ { \mathrm { c t } } ( X _ { s } ) = \frac { 1 } { | T | } \int _ { T } \ell _ { s } ( t ) d t ,\tag{14}
$$

$$
r _ { \mathrm { o b s } } ( X _ { s } ) = \int _ { T } \ell _ { s } ( t ) p _ { s } ( t ) d t .\tag{15}
$$

Accordingly,

$$
R _ { \mathrm { c t } } = \mathbb { E } _ { X } [ r _ { \mathrm { c t } } ( X ) ] , \qquad R _ { \mathrm { o b s } } = \mathbb { E } _ { X } [ r _ { \mathrm { o b s } } ( X ) ] .\tag{16}
$$

We first introduce the oracle importance-sampling terms

$$
A _ { s } = \frac { 1 } { L _ { s } } \sum _ { i = 1 } ^ { L _ { s } } \frac { \ell _ { s } ( t _ { s , i } ) } { p _ { s } ( t _ { s , i } ) } , \qquad B _ { s } = \frac { 1 } { L _ { s } } \sum _ { i = 1 } ^ { L _ { s } } \frac { 1 } { p _ { s } ( t _ { s , i } ) } ,\tag{17}
$$

and their estimated counterparts

$$
\widehat { A } _ { s } = \frac { 1 } { L _ { s } } \sum _ { i = 1 } ^ { L _ { s } } \frac { \ell _ { s } ( t _ { s , i } ) } { \widehat { p } ( t _ { s , i } \mid X _ { s } ) } , \qquad \widehat { B } _ { s } = \frac { 1 } { L _ { s } } \sum _ { i = 1 } ^ { L _ { s } } \frac { 1 } { \widehat { p } ( t _ { s , i } \mid X _ { s } ) } .\tag{18}
$$

By Assumption $\mid , p _ { s } ( t ) \geq \rho > 0$ and $\widehat { p } ( t \mid X _ { s } )$ is uniformly consistent on $\tau$ . Hence, almost surely, for sufficiently large $L _ { s }$

$$
\operatorname* { i n f } _ { t \in \mathscr { T } } \widehat { p } ( t \mid X _ { s } ) \geq \frac { \rho } { 2 } ,\tag{19}
$$

and therefore

$$
\begin{array} { c } { \displaystyle \operatorname* { s u p } _ { t \in { \mathcal T } } \left| \frac { 1 } { \widehat { p } ( t \mid X _ { s } ) } - \frac { 1 } { p _ { s } ( t ) } \right| \leq \frac { 2 } { \rho ^ { 2 } } \displaystyle \operatorname* { s u p } _ { t \in { \mathcal T } } \left| \widehat { p } ( t \mid X _ { s } ) - p _ { s } ( t ) \right| } \\ { \displaystyle \xrightarrow [ ] { \mathrm { a . s . } } 0 . } \end{array}\tag{20}
$$

Let $M < \infty$ satisfy $0 \leq \ell ( t , X ) \leq M$ . It follows that

$$
\left| \widehat { A } _ { s } - A _ { s } \right| \leq M \operatorname* { s u p } _ { t \in \mathcal { T } } \left| \frac { 1 } { \widehat { p } ( t \mid X _ { s } ) } - \frac { 1 } { p _ { s } ( t ) } \right| \overset { \mathrm { a . s . } } { \longrightarrow } 0 ,\tag{21}
$$

$$
\left| { \widehat B } _ { s } - { B } _ { s } \right| \leq \operatorname* { s u p } _ { t \in { \mathcal { T } } } \left| \frac { 1 } { \widehat { p } ( t \mid X _ { s } ) } - \frac { 1 } { p _ { s } ( t ) } \right| \overset { \mathrm { a . s . } } { \longrightarrow } 0 .\tag{22}
$$

Conditional on $X _ { s }$ , the future timestamps are independently and identically distributed according to $p _ { s } ( t )$ . The strong law of large numbers therefore gives

$$
A _ { s } \xrightarrow { \mathrm { a . s . } } \mathbb { E } _ { T | X _ { s } } \left[ \frac { \ell _ { s } ( T ) } { p _ { s } ( T ) } \right] = \int _ { T } \ell _ { s } ( t ) d t ,\tag{23}
$$

$$
B _ { s } \xrightarrow { \mathrm { a . s . } } \mathbb { E } _ { T | X _ { s } } \left[ \frac { 1 } { p _ { s } ( T ) } \right] = | T | .\tag{24}
$$

Combining the above results yields

$$
\widehat { A } _ { s } \ \xrightarrow { \mathrm { a . s . } } \ \int _ { \mathcal { T } } \ell _ { s } ( t ) d t , \qquad \widehat { B } _ { s } \ \xrightarrow { \mathrm { a . s . } } \ | \mathcal { T } | .\tag{25}
$$

Define the trajectory-level CSE estimator as

$$
\widehat { r } _ { \mathrm { c t } , s } = \frac { \sum _ { i = 1 } ^ { L _ { s } } \ell _ { s } ( t _ { s , i } ) / \widehat { p } ( t _ { s , i } \mid X _ { s } ) } { \sum _ { i = 1 } ^ { L _ { s } } 1 / \widehat { p } ( t _ { s , i } \mid X _ { s } ) } = \frac { \widehat { A } _ { s } } { \widehat { B } _ { s } } .\tag{26}
$$

Since $| \tau | > 0$ , the continuous mapping theorem gives

$$
\widehat { r } _ { \mathrm { c t } , s } \xrightarrow { \mathrm { a . s . } } \frac { 1 } { | T | } \int _ { \tau } \ell _ { s } ( t ) d t = r _ { \mathrm { c t } } ( X _ { s } ) .\tag{27}
$$

For MSE, the conditional strong law directly gives

$$
\widehat { r } _ { \mathrm { o b s } , s } = \frac { 1 } { L _ { s } } \sum _ { i = 1 } ^ { L _ { s } } \ell _ { s } ( t _ { s , i } ) \ \xrightarrow { \mathrm { a . s . } } \ \mathbb { E } _ { T | X _ { s } } [ \ell _ { s } ( T ) ] = r _ { \mathrm { o b s } } ( X _ { s } ) .\tag{28}
$$

Let $\underline { { L } } = \operatorname* { m i n } _ { 1 \le s \le S } L _ { s }$ . For the joint limit below, Assumption 1 is understood to hold uniformly over the test trajectories. Consequently, Eqs. (27) and (28) imply

$$
\frac { 1 } { S } \sum _ { s = 1 } ^ { S } | \widehat { r } _ { \mathrm { c t } , s } - r _ { \mathrm { c t } } ( X _ { s } ) | \overset { \mathrm { a . s . } } { \longrightarrow } 0 ,\tag{29}
$$

$$
{ \frac { 1 } { S } } \sum _ { s = 1 } ^ { S } | { \widehat { r } } _ { \mathrm { o b s } , s } - r _ { \mathrm { o b s } } ( X _ { s } ) | \ { \xrightarrow { \mathrm { a . s . } } } \ 0\tag{30}
$$

as $\underline { { L } } \to \infty$

Since the test trajectories are mutually independent and the trajectory-level risks are bounded, another application of the strong law yields

$$
{ \frac { 1 } { S } } \sum _ { s = 1 } ^ { S } r _ { \mathrm { c t } } ( X _ { s } ) { \xrightarrow { \mathrm { a . s . } } } R _ { \mathrm { c t } } , \qquad { \frac { 1 } { S } } \sum _ { s = 1 } ^ { S } r _ { \mathrm { o b s } } ( X _ { s } ) { \xrightarrow { \mathrm { a . s . } } } R _ { \mathrm { o b s } } .\tag{31}
$$

Therefore, as $S , \underline { { L } }  \infty$

$$
\mathrm { C S E } \stackrel { \mathrm { a . s . } } { \longrightarrow } R _ { \mathrm { c t } } , \qquad \mathrm { M S E } \stackrel { \mathrm { a . s . } } { \longrightarrow } R _ { \mathrm { o b s } } .\tag{32}
$$

It follows that

$$
\operatorname* { l i m } _ { S , L \to \infty } | \mathrm { C S E } - R _ { \mathrm { c t } } | = 0 ,\tag{33}
$$

$$
\operatorname* { l i m } _ { S , L \to \infty } \left| \mathrm { M S E } - R _ { \mathrm { c t } } \right| = \left| R _ { \mathrm { o b s } } - R _ { \mathrm { c t } } \right|\tag{34}
$$

almost surely. Hence,

$$
\operatorname* { l i m } _ { S , \underline { { L } } \to \infty } \left| \mathrm { C S E } - R _ { \mathrm { c t } } \right| \le \operatorname* { l i m } _ { S , \underline { { L } } \to \infty } \left| \mathrm { M S E } - R _ { \mathrm { c t } } \right| .\tag{35}
$$

When $p ( t \mid X ) \ = \ 1 / | T |$ almost everywhere, we have $R _ { \mathrm { o b s } } ~ = ~ R _ { \mathrm { c t } }$ , and both asymptotic estimation errors are zero. More generally, equality holds whenever $R _ { \mathrm { { o b s } } } = R _ { \mathrm { { c t } } }$ whereas the inequality is strict when $R _ { \mathrm { { o b s } } } \neq R _ { \mathrm { { c t } } }$ . The selfnormalized estimator may be biased at a finite sample size; the result above concerns its consistency and asymptotic estimation error.

## B. Derivation of the Evaluation Discrepancy

Recall that

$$
p _ { T } ( t ) = \mathbb { E } _ { X } [ p ( t \mid X ) ] , \quad \quad { \bar { \ell } } ( t ) = \mathbb { E } _ { X } [ \ell ( t , X ) ] .\tag{36}
$$

For the sampling-dependence gap,

$$
\begin{array} { l } { G _ { \mathrm { s a m p } } = R _ { \mathrm { o b s } } - R _ { \mathrm { g l o b a l } } } \\ { \displaystyle = \mathbb { E } _ { X } \left[ \int _ { T } \ell ( t , X ) p ( t \mid X ) d t \right] - \mathbb { E } _ { X } \left[ \int _ { T } \ell ( t , X ) p _ { T } ( t ) d t \right] } \\ { \displaystyle = \int _ { T } \left\{ \mathbb { E } _ { X } [ \ell ( t , X ) p ( t \mid X ) ] - \mathbb { E } _ { X } [ \ell ( t , X ) ] \mathbb { E } _ { X } [ p ( t \mid X ) ] \right\} d t } \\ { \displaystyle = \int _ { T } \mathrm { C o v } _ { X } \left( \ell ( t , X ) , p ( t \mid X ) \right) d t . } \end{array}
$$

For the temporal-distribution gap, let $U \sim \operatorname { U n i f } ( \mathcal { T } )$ . Since $\mathbb { E } _ { U } [ | \mathcal { T } | _ { p _ { T } } ( U ) ] = 1$ , we have

$$
\begin{array} { r l r } {  { G _ { \mathrm { t i m e } } = R _ { \mathrm { g l o b a l } } - R _ { \mathrm { c t } } } } \\ & { } & { ~ = \int _ { \mathcal T } \bar { \ell } ( t ) p _ { T } ( t ) d t - \frac { 1 } { | \mathcal T | } \int _ { \mathcal T } \bar { \ell } ( t ) d t } \\ & { } & { ~ = \mathbb { E } _ { U } [ \bar { \ell } ( U ) | \mathcal T | p _ { T } ( U ) ] - \mathbb { E } _ { U } [ \bar { \ell } ( U ) ] \mathbb { E } _ { U } [ | \mathcal T | p _ { T } ( U ) ] } \\ & { } & { ~ = \mathrm { C o v } _ { U \sim \mathrm { U n i f } ( \mathcal T ) } ( \bar { \ell } ( U ) , | \mathcal T | p _ { T } ( U ) ) . \quad \quad \quad ( 3 8 ) } \end{array}
$$

Therefore,

$$
R _ { \mathrm { o b s } } - R _ { \mathrm { c t } } = G _ { \mathrm { s a m p } } + G _ { \mathrm { t i m e } } .\tag{39}
$$

Here, $G _ { \mathrm { s a m p } }$ captures the effect of sample-dependent sampling, whereas $G _ { \mathrm { t i m e } }$ captures the effect of population-level temporal non-uniformity.

## C. Global-Time Squared Error

The marginal timestamp density is estimated by averaging the trajectory-specific conditional densities:

$$
{ \widehat { p } } _ { T } ( t ) = { \frac { 1 } { S } } \sum _ { r = 1 } ^ { S } { \widehat { p } } ( t \mid X _ { r } ) .\tag{40}
$$

Using $\widehat { p } ( t \mid X _ { s } )$ as the proposal density and $\widehat { p } _ { T } ( t )$ as the

Table 3. Complete MSE and CSE results on CESNET, FNSPID, GDELT, and HumanActivity. Values are mean ± standard deviation over three random seeds; standard deviations are rounded to three decimal places.
<table><tr><td rowspan="2">Model</td><td colspan="2">CESNET  $\times 1 0 ^ { - 1 }$ </td><td colspan="2">FNSPID  $\times 1 0 ^ { - 1 }$ </td><td colspan="2">GDELT  $\times 1 0 ^ { 0 }$ </td><td colspan="2">HumanActivity  $\times 1 0 ^ { - 2 }$ </td></tr><tr><td>MSE</td><td>CSE</td><td>MSE</td><td>CSE</td><td>MSE</td><td>CSE</td><td>MSE</td><td>CSE</td></tr><tr><td>APN</td><td> $8 . 7 6 { \pm } 0 . 0 0 7 ^ { ( 5 ) }$ </td><td> $8 . 5 8 { \pm } 0 . 0 0 8 ^ { ( 5 ) }$ </td><td> $2 . 3 8 { \pm } 0 . 1 4 3 ^ { ( 2 ) }$ </td><td> $2 . 3 8 { \pm } 0 . 1 4 4 ^ { ( 2 ) }$ </td><td> $\mathbf { 1 . 0 2 \pm 0 . 0 0 1 ^ { ( 1 ) } }$ </td><td> $\mathbf { 0 . 9 9 \pm 0 . 0 0 1 ^ { ( 1 ) } }$ </td><td> $5 . 6 8 { \pm } 0 . 0 3 5 ^ { ( 6 ) }$ </td><td> $5 . 6 5 { \pm } 0 . 0 3 9 ^ { ( 6 ) }$ </td></tr><tr><td>ASTGI</td><td> $\mathbf { 8 . 4 7 \pm 0 . 1 3 7 ^ { ( 1 ) } }$ </td><td> ${ \bf 8 . 2 6 \pm 0 . 1 2 2 ^ { ( 1 ) } }$ </td><td> $3 . 2 8 { \pm } 0 . 2 1 3 ^ { ( 6 ) }$ </td><td> $3 . 2 5 { \pm } 0 . 2 1 1 ^ { ( 6 ) }$ </td><td> $1 . 1 0 { \pm } 0 . 0 0 3 ^ { ( 4 ) }$ </td><td> $1 . 0 4 { \pm } 0 . 0 0 3 ^ { ( 4 ) }$ </td><td> $5 . 4 6 \pm 0 . 1 6 7 ^ { ( 3 ) }$ </td><td> $5 . 4 1 \pm 0 . 1 6 7 ^ { ( 3 ) }$ </td></tr><tr><td>GRU-D</td><td> $1 0 . 2 2 { \pm } 0 . 4 6 4 ^ { ( 1 0 ) }$ </td><td> $9 . 9 1 \pm 0 . 4 6 6 ^ { ( 1 0 ) }$ </td><td> $5 . 2 4 { \pm } 0 . 8 0 0 ^ { ( 8 ) }$ </td><td> $5 . 2 0 { \scriptstyle \pm 0 . 7 9 1 } ^ { ( 8 ) }$ </td><td> $1 . 2 2 { \pm } 0 . 0 0 3 ^ { ( 1 0 ) }$ </td><td> $1 . 1 5 { \pm } 0 . 0 0 3 ^ { ( 1 0 ) }$ </td><td> $8 . 5 6 \pm 1 . 4 9 7 ^ { ( 9 ) }$ </td><td> $8 . 5 6 \pm 1 . 5 1 5 ^ { ( 9 ) }$ </td></tr><tr><td>GraFITi</td><td> $8 . 5 0 { \pm } 0 . 1 1 3 ^ { ( 2 ) }$ </td><td> $8 . 2 8 { \pm } 0 . 1 0 2 ^ { ( 2 ) }$ </td><td> $3 . 2 1 { \pm } 0 . 7 5 5 ^ { ( 4 ) }$ </td><td> $3 . 2 2 { \pm } 0 . 7 6 0 ^ { ( 4 ) }$ </td><td> $1 . 0 9 { \pm } 0 . 0 0 2 ^ { ( 3 ) }$ </td><td> $1 . 0 1 { \pm } 0 . 0 0 2 ^ { ( 2 ) }$ </td><td> $5 . 4 4 { \pm } 0 . 0 3 8 ^ { ( 2 ) }$ </td><td> $5 . 4 0 { \pm } 0 . 0 3 4 ^ { ( 2 ) }$ </td></tr><tr><td>HyperIMTS</td><td> $8 . 9 0 { \pm } 0 . 0 7 3 ^ { ( 7 ) }$ </td><td> $8 . 6 5 { \pm } 0 . 0 6 0 ^ { ( 7 ) }$ </td><td>2.72±0.593(3)</td><td>2.71±0.592(3)</td><td> $1 . 1 8 { \pm } 0 . 0 0 2 ^ { ( 8 ) }$ </td><td> $1 . 0 9 { \pm } 0 . 0 0 2 ^ { ( 8 ) }$ </td><td> ${ \bf 5 . 4 2 \pm 0 . 0 3 7 ^ { ( 1 ) } }$ </td><td> ${ \bf 5 . 3 8 \pm 0 . 0 3 9 ^ { ( 1 ) } }$ </td></tr><tr><td>KAFNet</td><td> $8 . 7 3 { \pm } 0 . 0 1 2 ^ { ( 4 ) }$ </td><td> $8 . 5 4 { \pm } 0 . 0 0 3 ^ { ( 4 ) }$ </td><td> $\mathbf { 2 . 2 3 { \pm } 0 . 0 4 7 ^ { ( 1 ) } }$ </td><td> $\mathbf { 2 . 2 2 \pm 0 . 0 4 3 ^ { ( 1 ) } }$ </td><td> $1 . 0 8 { \pm } 0 . 0 0 5 ^ { ( 2 ) }$ </td><td> $1 . 0 2 { \pm } 0 . 0 0 5 ^ { ( 3 ) }$ </td><td> $5 . 6 2 { \pm } 0 . 0 2 1 ^ { ( 4 ) }$ </td><td> $5 . 5 8 { \pm } 0 . 0 2 5 ^ { ( 4 ) }$ </td></tr><tr><td>NeuralFlow</td><td> $9 . 9 8 { \pm } 0 . 1 0 3 ^ { ( 8 ) }$ </td><td> $9 . 6 9 { \pm } 0 . 1 1 8 ^ { ( 8 ) }$ </td><td> $7 . 3 3 { \pm } 0 . 1 3 8 ^ { ( 1 0 ) }$ </td><td> $7 . 2 9 \pm 0 . 1 4 0 ^ { ( 1 0 ) }$ </td><td> $1 . 1 8 \pm 0 . 0 0 5 ^ { ( 9 ) }$ </td><td> $1 . 0 9 { \pm } 0 . 0 0 5 ^ { ( 9 ) }$ </td><td> $1 7 . 5 3 { \pm } 2 . 5 1 2 ^ { ( 1 1 ) }$ </td><td> $1 7 . 5 1 { \pm } 2 . 5 2 3 ^ { ( 1 1 ) }$ </td></tr><tr><td>SeFT</td><td> $1 0 . 1 1 { \pm } 0 . 1 8 3 ^ { ( 9 ) }$ </td><td> $9 . 7 9 { \scriptstyle \pm 0 . 1 7 7 ^ { ( 9 ) } }$ </td><td> $8 . 7 0 { \pm } 0 . 9 5 8 ^ { ( 1 1 ) }$ </td><td> $8 . 6 6 \pm 0 . 9 5 7 ^ { ( 1 1 ) }$ </td><td> $1 . 1 3 { \pm } 0 . 0 0 3 ^ { ( 6 ) }$ </td><td> $1 . 0 5 { \pm } 0 . 0 0 3 ^ { ( 5 ) }$ </td><td> $9 . 3 8 \pm 0 . 4 2 2 ^ { ( 1 0 ) }$ </td><td> $9 . 3 6 \pm 0 . 4 0 6 ^ { ( 1 0 ) }$ </td></tr><tr><td>Warpformer</td><td> $8 . 5 8 { \pm } 0 . 0 6 7 ^ { ( 3 ) }$ </td><td> $8 . 3 7 { \pm } 0 . 0 5 8 ^ { ( 3 ) }$ </td><td> $3 . 6 6 \pm 0 . 2 1 6 ^ { ( 7 ) }$ </td><td> $3 . 6 3 { \pm } 0 . 2 2 0 ^ { ( 7 ) }$ </td><td> $1 . 3 0 { \pm } 0 . 0 1 6 ^ { ( 1 1 ) }$ </td><td> $1 . 2 0 { \pm } 0 . 0 1 6 ^ { ( 1 1 ) }$ </td><td> $5 . 6 5 { \pm } 0 . 1 4 3 ^ { ( 5 ) }$ </td><td> $5 . 6 1 { \pm } 0 . 1 4 1 ^ { ( 5 ) }$ </td></tr><tr><td>mTAN</td><td> $1 0 . 4 7 { \pm } 0 . 1 5 3 ^ { ( 1 1 ) }$ </td><td> $1 0 . 1 4 { \pm } 0 . 1 0 6 ^ { ( 1 1 ) }$ </td><td> $6 . 1 3 { \pm } 0 . 7 4 3 ^ { ( 9 ) }$ </td><td> $6 . 0 9 { \pm } 0 . 7 3 8 ^ { ( 9 ) }$ </td><td> $1 . 1 2 { \pm } 0 . 0 0 3 ^ { ( 5 ) }$ </td><td> $1 . 0 5 { \pm } 0 . 0 0 3 ^ { ( 6 ) }$ </td><td> $7 . 3 3 { \pm } 0 . 1 0 8 ^ { ( 8 ) }$ </td><td> $7 . 3 0 { \pm } 0 . 1 0 7 ^ { ( 8 ) }$ </td></tr><tr><td>tPatchGNN</td><td> $8 . 8 6 \pm 0 . 1 4 0 ^ { ( 6 ) }$ </td><td> $8 . 6 1 { \pm } 0 . 1 1 4 ^ { ( 6 ) }$ </td><td> $3 . 2 6 \pm 0 . 2 6 9 ^ { ( 5 ) }$ </td><td> $3 . 2 4 \pm 0 . 2 6 9 ^ { ( 5 ) }$ </td><td> $1 . 1 5 { \pm } 0 . 0 0 6 ^ { ( 7 ) }$ </td><td> $1 . 0 8 { \pm } 0 . 0 0 6 ^ { ( 7 ) }$ </td><td> $5 . 8 0 { \pm } 0 . 1 2 1 ^ { ( 7 ) }$ </td><td> $5 . 7 6 { \pm } 0 . 1 3 2 ^ { ( 7 ) }$ </td></tr></table>

target density, we define the Global-Time Squared Error as

$$
\mathrm { G S E } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \frac { \displaystyle \sum _ { i = 1 } ^ { L _ { s } } \ell ( t _ { s , i } , X _ { s } ) \frac { \widehat { p _ { T } } ( t _ { s , i } ) } { \widehat { p } ( t _ { s , i } \mid X _ { s } ) } } { \displaystyle \sum _ { i = 1 } ^ { L _ { s } } \frac { \widehat { p _ { T } } ( t _ { s , i } ) } { \widehat { p } ( t _ { s , i } \mid X _ { s } ) } } .\tag{41}
$$

GSE evaluates each trajectory under the dataset-level marginal timestamp distribution, thereby removing samplespecific sampling effects while retaining population-level temporal non-uniformity. It serves as the empirical counterpart of $R _ { \mathrm { g l o b a l } }$ and an intermediate reference between MSE and CSE.

## D. Benchmark Details

## D.1. Synthetic and Semi-Synthetic Data

Common setup. The controlled benchmark contains two fully synthetic datasets, Synthetic-Regime and Synthetic-Multiscale, and two semi-synthetic datasets constructed from ETTm1 and Weather. Each dataset contains 512 training, 128 validation, and 128 test trajectories. For each trajectory, 16 historical observations are provided as input. Training and validation use 30 future target points, whereas each test realization contains 128 future query timestamps. For fully synthetic data, the historical and future intervals are [0, 0.5] and (0.5, 1], respectively. Gaussian noise with standard deviation 0.002 is added only to the historical observations.

Synthetic-Regime. Each trajectory is controlled by $z _ { 1 } , z _ { 2 } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , 1 )$ . We first define

The smooth and periodic components are

$$
\begin{array} { l } { { r ( t ) = \displaystyle \mathrm { c l i p } \left( \frac { t - 0 . 5 8 } { 0 . 6 8 - 0 . 5 8 } , 0 , 1 \right) , } } \\ { { w ( t ) = r ( t ) ^ { 2 } \{ 3 - 2 r ( t ) \} . } } \end{array}\tag{42}
$$

$$
\begin{array} { r l } & { b _ { 1 } ( t ) = 0 . 1 0 z _ { 1 } + 0 . 0 1 5 \sin ( 2 \pi t ) , } \\ & { b _ { 2 } ( t ) = 0 . 1 0 z _ { 2 } + 0 . 0 1 2 \cos ( 2 \pi t ) , } \\ & { b _ { 3 } ( t ) = 0 . 0 7 ( z _ { 1 } - z _ { 2 } ) + 0 . 0 1 0 \sin ( 4 \pi t ) , } \\ & { p _ { 1 } ( t ) = \left( 0 . 4 8 + 0 . 0 4 z _ { 2 } \right) \sin ( 1 2 \pi ( t - 0 . 6 4 ) ) , } \\ & { p _ { 2 } ( t ) = \left( 0 . 4 4 + 0 . 0 4 z _ { 1 } \right) \cos ( 1 0 \pi ( t - 0 . 6 4 ) ) , } \\ & { p _ { 3 } ( t ) = \{ 0 . 4 0 + 0 . 0 3 ( z _ { 1 } + z _ { 2 } ) \} \sin ( 8 \pi ( t - 0 . 6 4 ) ) . } \end{array}\tag{43}
$$

The final three-dimensional trajectory is

$$
y _ { j } ( t ) = b _ { j } ( t ) + w ( t ) p _ { j } ( t ) , \qquad j \in \{ 1 , 2 , 3 \} .\tag{44}
$$

The transition function $w ( t )$ continuously changes the process from a low-amplitude smooth regime to a highfrequency periodic regime, producing different levels of forecasting difficulty across the future interval.

Synthetic-Multiscale. For $z _ { 1 } , z _ { 2 } , z _ { 3 } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , 1 )$ , we define

$$
\begin{array} { c } { { g _ { 0 } ( t ) = 0 . 2 5 z _ { 1 } + 0 . 1 8 t + 0 . 3 2 \sin ( 2 \pi \cdot 1 . 2 5 t + 0 . 2 0 z _ { 2 } ) , } } \\ { { g _ { h } ( t ) = 0 . 1 4 ( 1 + 0 . 0 8 z _ { 1 } ) \sin ( 2 \pi \cdot 8 t + 0 . 2 5 z _ { 3 } ) , } } \\ { { \mu = 0 . 7 2 + 0 . 0 2 5 \operatorname { t a n h } ( z _ { 3 } ) , } } \\ { { g _ { b } ( t ) = 0 . 3 0 ( 1 + 0 . 0 6 z _ { 2 } ) \exp \Biggl \{ - \displaystyle \frac { ( t - \mu ) ^ { 2 } } { 2 ( 0 . 0 5 5 ) ^ { 2 } } \Biggr \} . } } \end{array}
$$

The output channels are

(45)

$$
\begin{array} { r l } & { y _ { 1 } ( t ) = g _ { 0 } ( t ) + g _ { h } ( t ) + g _ { b } ( t ) , } \\ & { y _ { 2 } ( t ) = 0 . 7 5 g _ { 0 } ( t ) - 0 . 8 0 g _ { h } ( t ) + 0 . 6 5 g _ { b } ( t ) } \\ & { ~ + 0 . 0 8 \cos ( 5 \pi t ) , } \\ & { y _ { 3 } ( t ) = - 0 . 4 5 g _ { 0 } ( t ) + 0 . 5 5 g _ { h } ( t ) - 0 . 7 0 g _ { b } ( t ) } \\ & { ~ + 0 . 1 2 t + 0 . 1 0 \sin ( 1 . 5 \pi t + 0 . 3 0 z _ { 1 } ) . } \end{array}\tag{46}
$$

This construction combines trend, low- and high-frequency periodicity, and a localized transient component.

Table 4. Complete MSE and CSE results on MIMIC-III, RepoHealth, StudentLife, and USHCN. Values are mean ± standard deviation over three random seeds; standard deviations are rounded to three decimal places.
<table><tr><td rowspan="2">Model</td><td colspan="2"> $\mathbf { M I M C - I I I }$   $\times 1 0 ^ { - 1 }$ </td><td colspan="2">RepoHealth  $\times 1 0 ^ { - 1 }$ </td><td colspan="2"> $_ { \mathrm { S t u d e n t L i f e } }$   $\times 1 0 ^ { - 1 }$ </td><td colspan="2">USHCN  $\times 1 0 ^ { - 1 }$ </td></tr><tr><td>MSE</td><td>CSE</td><td>MSE</td><td>CSE</td><td>MSE</td><td>CSE</td><td>MSE</td><td>CSE</td></tr><tr><td>APN</td><td> $6 . 6 6 { \pm } 0 . 0 3 8 ^ { ( 6 ) }$ </td><td> $6 . 7 2 { \scriptstyle \pm 0 . 0 3 8 ^ { ( 6 ) } }$ </td><td> $3 . 3 6 { \pm } 0 . 0 6 9 ^ { ( 2 ) }$ </td><td> $\mathbf { 3 . 5 7 { \pm } 0 . 0 8 0 ^ { ( 1 ) } }$ </td><td> $6 . 6 1 { \pm } 0 . 0 0 9 ^ { ( 2 ) }$ </td><td> $6 . 6 8 { \pm } 0 . 0 0 6 ^ { ( 3 ) }$ </td><td> $6 . 0 4 { \pm } 0 . 5 5 4 ^ { ( 4 ) }$ </td><td> $6 . 7 1 { \pm } 0 . 6 8 6 ^ { ( 5 ) }$ </td></tr><tr><td>ASTGI</td><td> $1 0 . 5 0 { \pm } 0 . 0 0 0 ^ { ( 1 1 ) }$ </td><td> $1 0 . 6 2 { \pm } 0 . 0 0 0 ^ { ( 1 1 ) }$ </td><td> $6 . 2 7 { \pm } 2 . 3 1 0 ^ { ( 7 ) }$ </td><td> $6 . 5 3 { \pm } 2 . 3 7 8 ^ { ( 7 ) }$ </td><td> $6 . 6 5 { \pm } 0 . 0 3 6 ^ { ( 3 ) }$ </td><td> $6 . 6 5 { \pm } 0 . 0 3 8 ^ { ( 2 ) }$ </td><td> $5 . 1 4 { \pm } 0 . 1 6 9 ^ { ( 3 ) }$ </td><td> $5 . 6 6 \pm 0 . 1 7 9 ^ { ( 3 ) }$ </td></tr><tr><td>GRU-D</td><td> $6 . 6 7 { \pm } 0 . 0 4 3 ^ { ( 7 ) }$ </td><td> $6 . 7 5 { \pm } 0 . 0 4 6 ^ { ( 7 ) }$ </td><td> $1 1 . 6 8 { \pm } 0 . 6 9 5 ^ { ( 1 1 ) }$ </td><td> $1 1 . 9 5 { \pm } 0 . 6 7 2 ^ { ( 1 1 ) }$ </td><td>7.21±0.045(8)</td><td> $7 . 2 3 { \pm } 0 . 0 4 5 ^ { ( 8 ) }$ </td><td> $7 . 7 8 \pm 0 . 6 9 0 ^ { ( 1 1 ) }$ </td><td> $8 . 5 9 { \pm } 0 . 7 5 6 ^ { ( 1 1 ) }$ </td></tr><tr><td>GraFITi</td><td> ${ \bf 5 . 8 9 { \pm } 0 . 2 6 4 ^ { ( 1 ) } }$ </td><td> $6 . 0 2 { \pm } 0 . 2 7 8 ^ { ( 2 ) }$ </td><td> ${ \bf 3 . 3 2 \pm 0 . 3 4 1 ^ { ( 1 ) } }$ </td><td> $3 . 6 1 { \pm } 0 . 3 4 2 ^ { ( 2 ) }$ </td><td> $6 . 7 2 { \scriptstyle \pm 0 . 1 6 4 } ^ { ( 6 ) }$ </td><td> $6 . 7 0 { \pm } 0 . 1 6 2 ^ { ( 4 ) }$ </td><td> $\mathbf { 4 . 5 0 \pm 0 . 2 5 2 ^ { ( 1 ) } }$ </td><td> ${ \bf 4 . 8 9 { \pm } 0 . 2 9 9 ^ { ( 1 ) } }$ </td></tr><tr><td>HyperIMTS</td><td> $5 . 9 4 { \pm } 0 . 0 6 8 ^ { ( 2 ) }$ </td><td> ${ \bf 5 . 9 2 \pm 0 . 0 6 8 ^ { ( 1 ) } }$ </td><td> $3 . 7 2 { \scriptstyle \pm 0 . 4 3 7 ^ { ( 3 ) } }$ </td><td> $3 . 9 7 { \pm } 0 . 4 4 4 ^ { ( 3 ) }$ </td><td> ${ \bf 6 . 5 4 } \pm 0 . 0 6 7 ^ { ( 1 ) }$ </td><td> ${ \bf 6 . 5 6 { \pm } 0 . 0 6 2 ^ { ( 1 ) } }$ </td><td> $6 . 6 7 { \pm } 1 . 8 3 7 ^ { ( 8 ) }$ </td><td> $7 . 3 0 { \pm } 2 . 0 7 3 ^ { ( 8 ) }$ </td></tr><tr><td>KAFNet</td><td> $6 . 4 1 \pm 0 . 0 2 1 ^ { ( 3 ) }$ </td><td> $6 . 4 7 { \pm } 0 . 0 2 2 ^ { ( 3 ) }$ </td><td> $5 . 7 2 { \pm } 3 . 0 3 4 ^ { ( 6 ) }$ </td><td> $5 . 9 8 { \pm } 3 . 0 6 1 ^ { ( 6 ) }$ </td><td> $6 . 6 6 { \pm } 0 . 0 2 3 ^ { ( 4 ) }$ </td><td> $6 . 7 2 { \scriptstyle \pm 0 . 0 2 4 } ^ { ( 5 ) }$ </td><td> $7 . 1 4 { \pm } 0 . 4 1 8 ^ { ( 9 ) }$ </td><td> $7 . 8 2 { \pm } 0 . 4 0 6 ^ { ( 9 ) }$ </td></tr><tr><td>NeuralFlow</td><td> $7 . 0 5 { \pm } 0 . 0 4 2 ^ { ( 8 ) }$ </td><td> $7 . 1 9 { \pm } 0 . 0 4 4 ^ { ( 9 ) }$ </td><td> $1 0 . 1 3 { \pm } 0 . 2 8 3 ^ { ( 1 0 ) }$ </td><td> $1 0 . 4 4 { \pm } 0 . 3 2 0 ^ { ( 1 0 ) }$ </td><td> $7 . 8 3 { \pm } 0 . 2 1 8 ^ { ( 1 0 ) }$ </td><td> $7 . 8 6 \pm 0 . 2 1 2 ^ { ( 1 0 ) }$ </td><td> $6 . 0 7 { \pm } 0 . 7 2 0 ^ { ( 5 ) }$ </td><td> $6 . 7 4 { \scriptstyle \pm 0 . 7 6 4 } ^ { ( 6 ) }$ </td></tr><tr><td>SeFT</td><td> $7 . 1 0 { \pm } 0 . 0 6 0 ^ { ( 9 ) }$ </td><td> $7 . 1 5 { \pm } 0 . 0 6 0 ^ { ( 8 ) }$ </td><td> $9 . 1 3 { \pm } 0 . 4 5 4 ^ { ( 8 ) }$ </td><td> $9 . 4 4 \pm 0 . 4 6 3 ^ { ( 8 ) }$ </td><td> $8 . 9 4 { \pm } 1 . 7 0 6 ^ { ( 1 1 ) }$ </td><td> $8 . 9 5 { \pm } 1 . 7 0 0 ^ { ( 1 1 ) }$ </td><td> $7 . 6 4 \pm 0 . 1 3 7 ^ { ( 1 0 ) }$ </td><td> $8 . 4 1 \pm 0 . 1 7 0 ^ { ( 1 0 ) }$ </td></tr><tr><td>Warpformer</td><td> $6 . 4 2 { \pm } 0 . 0 4 1 ^ { ( 4 ) }$ </td><td> $6 . 5 0 { \scriptstyle \pm 0 . 0 4 5 ^ { ( 5 ) } }$ </td><td> $5 . 0 4 { \pm } 0 . 5 5 5 ^ { ( 5 ) }$ </td><td> $5 . 3 3 { \pm } 0 . 5 9 8 ^ { ( 5 ) }$ </td><td> $6 . 7 8 { \scriptstyle \pm 0 . 0 6 5 ^ { ( 7 ) } }$ </td><td> $6 . 8 0 { \pm } 0 . 0 6 6 ^ { ( 7 ) }$ </td><td> $6 . 1 3 { \pm } 0 . 1 6 8 ^ { ( 6 ) }$ </td><td> $6 . 6 7 { \pm } 0 . 1 9 5 ^ { ( 4 ) }$ </td></tr><tr><td>mTAN</td><td> $7 . 4 7 { \pm } 0 . 0 3 4 ^ { ( 1 0 ) }$ </td><td> $7 . 5 4 { \pm } 0 . 0 3 2 ^ { ( 1 0 ) }$ </td><td> $1 0 . 0 9 \pm 0 . 4 3 8 ^ { ( 9 ) }$ </td><td> $1 0 . 3 9 { \pm } 0 . 4 4 3 ^ { ( 9 ) }$ </td><td> $7 . 3 2 { \pm } 0 . 0 6 9 ^ { ( 9 ) }$ </td><td> $7 . 3 4 { \pm } 0 . 0 6 7 ^ { ( 9 ) }$ </td><td> $4 . 8 4 { \pm } 0 . 1 7 4 ^ { ( 2 ) }$ </td><td> $5 . 2 3 { \pm } 0 . 2 3 2 ^ { ( 2 ) }$ </td></tr><tr><td>tPatchGNN</td><td> $6 . 4 4 { \pm } 0 . 0 2 9 ^ { ( 5 ) }$ </td><td> $6 . 4 8 { \pm } 0 . 0 2 1 ^ { ( 4 ) }$ </td><td> $4 . 5 5 \pm 0 . 3 2 2 ^ { ( 4 ) }$ </td><td> $4 . 8 2 \pm 0 . 3 3 5 ^ { ( 4 ) }$ </td><td> $6 . 6 8 { \pm } 0 . 0 5 9 ^ { ( 5 ) }$ </td><td> $6 . 7 3 { \scriptstyle \pm 0 . 0 6 0 ^ { ( 6 ) } }$ </td><td> $6 . 3 6 \pm 1 . 3 0 6 ^ { ( 7 ) }$ </td><td> $7 . 0 5 { \pm } 1 . 3 8 4 ^ { ( 7 ) }$ </td></tr></table>

Semi-synthetic datasets. For ETTm1 and Weather, the original regularly sampled series are chronologically divided into training, validation, and test segments with a ratio of $6 0 \% / 2 0 \% / 2 0 \%$ . Standardization statistics are fitted using the training segment. Each sample contains 96 historical grid points and 512 future grid points. We retain 16 historical points as input and sample 30 future points for training and validation. At test time, 128 future points are selected from the complete 512-point future grid, which is retained to compute the reference continuous-time risk.

For all four datasets, test timestamps are sampled from

$$
p _ { \alpha } ( u ) = \alpha \ \mathrm { B e t a } ( u ; 1 0 , 6 0 ) + ( 1 - \alpha ) \ \mathrm { U n i f o r m } ( 0 , 1 ) ,\tag{47}
$$

where $\alpha \in \{ 0 , 0 . 3 , 0 . 5 , 0 . 9 , 0 . 9 9 \}$ controls the degree of temporal non-uniformity. For each α, the trained model, historical observations, and underlying future trajectory are fixed, while only the test query timestamps are resampled. Each setting is evaluated using 20 timestamp resamples shared across all models.

For fully synthetic data, the reference $R _ { \mathrm { c t } }$ is computed by trapezoidal integration over a dense common time grid. For semi-synthetic data, it is computed as the equally weighted squared error over all 512 future grid points.

## D.2. Real-World Datasets

We evaluate models on eight real-world datasets spanning healthcare, climate science, human sensing, finance, software engineering, international events, and network systems. These datasets exhibit diverse irregularity mechanisms, including event-triggered logging, activity-dependent collection, operational constraints, human scheduling, missing observations, and system jitter (Chang et al., 2025). Only the numerical time-series modalities are used.

The processed data are represented either as wide tables with one row per record and timestamp or as sparse triplets of timestamp, variable index, and value. Sparse observations are converted into multivariate tensors with binary masks. Filled values at unobserved positions are used only for tensor storage and do not contribute to training or evaluation. Record identifiers are deterministically divided into training, validation, and test partitions, and forecasting windows are constructed without crossing record boundaries.

CESNET. CESNET contains network traffic measurements collected from 11 devices, with 10 traffic variables. Its irregular timestamps mainly arise from scheduling delay and logging jitter. Raw timestamps are aggregated to minute resolution, and duplicate observations within the same minute are averaged. We use 1,440 minutes of history to predict the following 1,440 minutes, producing 252, 84, and 126 training, validation, and test windows, respectively.

FNSPID. FNSPID contains six financial variables for 10 entities, including open, high, low, close, adjusted close, and trading volume. Observations are restricted by market operating hours, creating structured gaps between trading periods. Timestamps are converted into relative calendar days without filling non-trading days. Both the historical and forecasting windows contain 30 days, resulting in 496/171/176 train/validation/test samples.

GDELT. GDELT records international events and contains five numerical variables for eight event streams. Since observations are generated when geopolitical events occur, the sampling process is naturally event-driven. Individual events and fractional-day timestamps are retained without daily aggregation. We use 14-day historical and forecasting windows, yielding 208/52/156 train/validation/test samples.

HumanActivity. HumanActivity contains 12 location variables obtained from sensors placed on the body during different human activities. The dataset contains 25 records, and the sensors are observed at asynchronous and slightly different timestamps. Relative timestamps are quantized, while duplicate measurements from the same sensor and timestamp are averaged. The history and forecasting lengths are 3,000 and 1,000 time units, producing 735/264/361 train/validation/test samples.

Table 5. Complete MAE and CAE results on CESNET, FNSPID, GDELT, and HumanActivity. Values are mean ± standard deviation over three random seeds; standard deviations are rounded to three decimal places.
<table><tr><td rowspan="2">Model</td><td colspan="2">CESNET  $\times 1 0 ^ { - 1 }$ </td><td colspan="2">FNSPID  $\times 1 0 ^ { - 1 }$ </td><td colspan="2"> $\mathrm { G D E L T }$   $\times 1 0 ^ { - 1 }$ </td><td colspan="2">HumanActivity  $\times 1 0 ^ { - 1 }$ </td></tr><tr><td>MAE</td><td>CAE</td><td>MAE</td><td>CAE</td><td>MAE</td><td>CAE</td><td>MAE</td><td>CAE</td></tr><tr><td>APN</td><td> $6 . 6 9 { \pm } 0 . 0 0 3 ^ { ( 5 ) }$ </td><td> $6 . 6 2 { \pm } 0 . 0 0 3 ^ { ( 5 ) }$ </td><td> ${ \underline { { 1 . 8 6 } } } { \pm } 0 . 1 7 3 ^ { ( 2 ) }$ </td><td> ${ \underline { { 1 . 8 5 } } } { \pm } 0 . 1 7 3 ^ { ( 2 ) }$ </td><td> ${ \bf 6 . 5 9 { \pm } 0 . 0 6 6 ^ { ( 1 ) } }$ </td><td> $\mathbf { 6 . 5 0 { \pm } 0 . 0 6 7 ^ { ( 1 ) } }$ </td><td> $1 . 4 3 { \pm } 0 . 0 3 1 ^ { ( 6 ) }$ </td><td> $1 . 4 2 { \pm } 0 . 0 3 2 ^ { ( 6 ) }$ </td></tr><tr><td>ASTGI</td><td> ${ \bf 6 . 5 5 { \pm } 0 . 0 6 4 ^ { ( 1 ) } }$ </td><td> ${ \bf 6 . 4 8 \pm 0 . 0 6 7 ^ { ( 1 ) } }$ </td><td> $2 . 1 9 { \pm } 0 . 0 9 2 ^ { ( 6 ) }$ </td><td> $\overline { { 2 . 1 8 } } \pm 0 . 0 9 3 ^ { ( 6 ) }$ </td><td> $6 . 6 9 \pm 0 . 0 8 8 ^ { ( 4 ) }$ </td><td> $6 . 6 0 { \pm } 0 . 0 8 8 ^ { ( 4 ) }$ </td><td> $1 . 3 8 { \pm } 0 . 0 2 4 ^ { ( 3 ) }$ </td><td> $1 . 3 8 { \pm } 0 . 0 2 4 ^ { ( 3 ) }$ </td></tr><tr><td>GRU-D</td><td> $7 . 4 5 \pm 0 . 2 4 9 ^ { ( 1 0 ) }$ </td><td> $7 . 3 7 { \pm } 0 . 2 6 2 ^ { ( 1 0 ) }$ </td><td> $2 . 7 6 \pm 0 . 2 8 6 ^ { ( 8 ) }$ </td><td> $2 . 7 5 { \pm } 0 . 2 8 4 ^ { ( 8 ) }$ </td><td> $6 . 8 1 \pm 0 . 0 3 3 ^ { ( 1 0 ) }$ </td><td> $6 . 7 2 { \pm } 0 . 0 3 2 ^ { ( 1 0 ) }$ </td><td> $2 . 0 6 \pm 0 . 2 5 1 ^ { ( 9 ) }$ </td><td> $2 . 0 6 \pm 0 . 2 5 3 ^ { ( 9 ) }$ </td></tr><tr><td>GraFITi</td><td> $6 . 5 6 { \pm } 0 . 0 3 3 ^ { ( 2 ) }$ </td><td> $6 . 4 9 { \pm } 0 . 0 3 3 ^ { ( 2 ) }$ </td><td> $1 . 9 6 \pm 0 . 0 9 7 ^ { ( 4 ) }$ </td><td> $1 . 9 5 { \pm } 0 . 0 9 6 ^ { ( 4 ) }$ </td><td> $6 . 6 6 \pm 0 . 1 4 6 ^ { ( 3 ) }$ </td><td> $6 . 5 7 { \pm } 0 . 1 4 6 ^ { ( 3 ) }$ </td><td> $1 . 3 7 { \pm } 0 . 0 1 2 ^ { ( 2 ) }$ </td><td> $1 . 3 7 { \pm } 0 . 0 1 3 ^ { ( 2 ) }$ </td></tr><tr><td>HyperIMTS</td><td> $6 . 7 9 { \pm } 0 . 0 5 6 ^ { ( 7 ) }$ </td><td> $6 . 7 5 { \pm } 0 . 0 6 3 ^ { ( 7 ) }$ </td><td> $1 . 9 1 { \pm } 0 . 3 9 1 ^ { ( 3 ) }$ </td><td> $1 . 9 1 { \pm } 0 . 3 9 0 ^ { ( 3 ) }$ </td><td> $6 . 7 4 { \scriptstyle \pm 0 . 0 7 2 ^ { ( 8 ) } }$ </td><td> $6 . 6 5 { \pm } 0 . 0 7 4 ^ { ( 8 ) }$ </td><td> $\mathbf { 1 . 3 6 \pm 0 . 0 1 3 ^ { ( 1 ) } }$ </td><td> $\mathbf { 1 . 3 6 \pm 0 . 0 1 3 ^ { ( 1 ) } }$ </td></tr><tr><td>KAFNet</td><td> $6 . 6 8 { \pm } 0 . 0 2 7 ^ { ( 4 ) }$ </td><td> $6 . 6 1 { \pm } 0 . 0 3 1 ^ { ( 4 ) }$ </td><td> ${ \bf 1 . 7 3 \pm 0 . 0 5 5 ^ { ( 1 ) } }$ </td><td> $\mathbf { 1 . 7 3 { \pm } 0 . 0 5 6 ^ { ( 1 ) } }$ </td><td> $6 . 6 0 { \pm } 0 . 0 9 2 ^ { ( 2 ) }$ </td><td> $6 . 5 1 { \pm } 0 . 0 9 3 ^ { ( 2 ) }$ </td><td> $1 . 3 9 { \pm } 0 . 0 1 0 ^ { ( 4 ) }$ </td><td> $1 . 3 9 { \pm } 0 . 0 1 0 ^ { ( 4 ) }$ </td></tr><tr><td>NeuralFlow</td><td> $7 . 4 1 { \pm } 0 . 1 2 2 ^ { ( 8 ) }$ </td><td> $7 . 3 5 { \pm } 0 . 1 2 7 ^ { ( 8 ) }$ </td><td> $3 . 4 6 \pm 0 . 2 3 4 ^ { ( 1 0 ) }$ </td><td> $3 . 4 5 \pm 0 . 2 3 3 ^ { ( 1 0 ) }$ </td><td> $6 . 7 7 { \pm } 0 . 0 6 4 ^ { ( 9 ) }$ </td><td> $\overline { { 6 . 6 8 } } \pm 0 . 0 6 4 ^ { ( 9 ) }$ </td><td> $3 . 1 1 \pm 0 . 3 0 5 ^ { ( 1 1 ) }$ </td><td> $3 . 1 1 \pm 0 . 3 0 4 ^ { ( 1 1 ) }$ </td></tr><tr><td>SeFT</td><td> $7 . 4 4 { \pm } 0 . 1 1 8 ^ { ( 9 ) }$ </td><td> $7 . 3 6 \pm 0 . 1 1 5 ^ { ( 9 ) }$ </td><td> $4 . 3 3 { \pm } 0 . 8 2 8 ^ { ( 1 1 ) }$ </td><td> $4 . 3 2 { \pm } 0 . 8 2 7 ^ { ( 1 1 ) }$ </td><td> $6 . 7 2 { \scriptstyle \pm 0 . 0 6 2 ^ { ( 6 ) } }$ </td><td> $6 . 6 1 { \pm } 0 . 0 5 9 ^ { ( 5 ) }$ </td><td> $2 . 1 2 { \pm } 0 . 0 4 1 ^ { ( 1 0 ) }$ </td><td> $2 . 1 2 { \pm } 0 . 0 4 1 ^ { ( 1 0 ) }$ </td></tr><tr><td>Warpformer</td><td> $6 . 6 3 { \pm } 0 . 0 5 4 ^ { ( 3 ) }$ </td><td> $6 . 5 6 { \pm } 0 . 0 5 9 ^ { ( 3 ) }$ </td><td> $2 . 2 0 { \pm } 0 . 0 5 3 ^ { ( 7 ) }$ </td><td> $2 . 1 9 { \pm } 0 . 0 5 4 ^ { ( 7 ) }$ </td><td> $7 . 4 8 \pm 0 . 1 1 0 ^ { ( 1 1 ) }$ </td><td> $7 . 3 5 { \pm } 0 . 1 0 9 ^ { ( 1 1 ) }$ </td><td> $1 . 4 1 { \pm } 0 . 0 0 3 ^ { ( 5 ) }$ </td><td> $1 . 4 1 { \pm } 0 . 0 0 3 ^ { ( 5 ) }$ </td></tr><tr><td>mTAN</td><td> $7 . 6 1 { \pm } 0 . 0 1 0 ^ { ( 1 1 ) }$ </td><td> $7 . 5 3 { \pm } 0 . 0 1 6 ^ { ( 1 1 ) }$ </td><td> $2 . 9 6 \pm 0 . 3 1 4 ^ { ( 9 ) }$ </td><td> $2 . 9 4 { \pm } 0 . 3 1 4 ^ { ( 9 ) }$ </td><td> $6 . 7 0 { \scriptstyle \pm 0 . 0 6 2 ^ { ( 5 ) } }$ </td><td> $6 . 6 3 { \pm } 0 . 0 6 3 ^ { ( 6 ) }$ </td><td> $1 . 8 5 { \pm } 0 . 0 1 8 ^ { ( 8 ) }$ </td><td> $1 . 8 5 { \pm } 0 . 0 1 8 ^ { ( 8 ) }$ </td></tr><tr><td>tPatchGNN</td><td> $6 . 7 5 { \pm } 0 . 0 7 0 ^ { ( 6 ) }$ </td><td> $6 . 7 0 { \scriptstyle \pm 0 . 0 8 2 ^ { ( 6 ) } }$ </td><td> $2 . 1 6 { \pm } 0 . 2 4 7 ^ { ( 5 ) }$ </td><td> $2 . 1 6 { \pm } 0 . 2 4 6 ^ { ( 5 ) }$ </td><td> $6 . 7 3 { \scriptstyle \pm 0 . 0 8 3 ^ { ( 7 ) } }$ </td><td> $6 . 6 4 { \pm } 0 . 0 8 3 ^ { ( 7 ) }$ </td><td> $1 . 4 5 { \pm } 0 . 0 3 5 ^ { ( 7 ) }$ </td><td> $1 . 4 4 { \pm } 0 . 0 3 6 ^ { ( 7 ) }$ </td></tr></table>

MIMIC-III. MIMIC-III is a large critical-care database containing de-identified clinical records from ICU patients (Johnson et al., 2016). We combine laboratory, input, output, and prescription events from the first 48 hours of each admission and aggregate them into half-hour bins. The processed data contain 22,406 admissions and 96 variables. The first 72 bins are used as history and the following 24 bins as the forecasting horizon, resulting in 11,156/3,699/3,703 train/validation/test samples. Its irregularity mainly reflects clinician-driven measurements and heterogeneous clinical recording frequencies.

and missing daily observations make the resulting trajectories sporadic. Following the commonly used irregular forecasting protocol, the first 150 time units are used as history and the following 50 units as the forecasting horizon. The resulting split contains 666/220/224 train/validation/test samples.

RepoHealth. RepoHealth describes issue, pull-request, and development activities from four software repositories using 10 numerical variables. Observations become denser during active development periods and sparser during inactive periods. Timestamps are converted into relative days, and duplicated daily observations are averaged. We use 30- day history and forecasting windows, with 97/59/50 train/- validation/test samples.

## D.3. Baselines and Hyperparameters

We compare eleven representative models covering recurrent, continuous-time, set-based, attention-based, graphbased, and pre-alignment forecasting paradigms. To ensure consistent comparisons, each model uses the same configuration across all real-world datasets.

APN. APN aggregates irregular observations into learnable temporal patches through soft boundaries and predicts values at arbitrary query timestamps (Liu et al., 2026b). We use hidden dimension d = 56, two layers, eight attention heads, time-embedding dimension 8, two patches, dropout 0.1, and batch size 32.

StudentLife. StudentLife contains daily mobile-sensing measurements from 20 students, including activity, location, sleep, and phone-use variables. Its nine variables are irregularly observed because data availability depends on individual routines and device usage. Both the historical and forecasting windows contain 30 days, producing 271/89/90 train/validation/test samples.

USHCN. USHCN contains long-term climate observations from 1,114 weather stations across the United States (Menne et al., 2015). Five meteorological variables are used,

ASTGI. ASTGI directly represents each discrete observation as a point in a learnable spatio-temporal space and adaptively constructs causal nearest-neighbor graphs for information propagation (Liu et al., 2026a). We use hidden dimension 128, two propagation layers, time-embedding dimension 128, channel-embedding dimension 64, 96 candidate neighbors, MLP ratio 4, dropout 0.1, and batch size 8.

GRU-D. GRU-D extends recurrent models with trainable decay mechanisms that explicitly incorporate missingness masks and elapsed time between observations (Che et al., 2016). We use one recurrent layer with hidden dimension 128 and batch size 32.

Table 6. Complete MAE and CAE results on MIMIC-III, RepoHealth, StudentLife, and USHCN. Values are mean ± standard deviation over three random seeds; standard deviations are rounded to three decimal places.
<table><tr><td rowspan="2">Model</td><td colspan="2">MIMIC-III  $\times 1 0 ^ { - 1 }$ </td><td colspan="2">RepoHealth  $\times 1 0 ^ { - 1 }$ </td><td colspan="2"> $_ { \mathrm { S t u d e n t L i f e } }$   $\times 1 0 ^ { - 1 }$ </td><td colspan="2">USHCN  $\times 1 0 ^ { - 1 }$ </td></tr><tr><td>MAE</td><td>CAE</td><td>MAE</td><td>CAE</td><td>MAE</td><td>CAE</td><td>MAE</td><td>CAE</td></tr><tr><td>APN</td><td> $4 . 7 8 { \pm } 0 . 0 3 4 ^ { ( 6 ) }$ </td><td> $4 . 7 9 { \pm } 0 . 0 3 4 ^ { ( 6 ) }$ </td><td> $3 . 4 7 { \pm } 0 . 0 4 9 ^ { ( 2 ) }$ </td><td> $3 . 5 5 { \pm } 0 . 0 5 1 ^ { ( 2 ) }$ </td><td> ${ \underline { { 5 . 7 5 } } } { \pm } 0 . 0 1 7 ^ { ( 2 ) }$ </td><td> $5 . 7 8 { \pm } 0 . 0 1 7 ^ { ( 3 ) }$ </td><td> $3 . 5 6 \pm 0 . 2 3 5 ^ { ( 4 ) }$ </td><td> $3 . 7 0 { \pm } 0 . 2 9 1 ^ { ( 5 ) }$ </td></tr><tr><td>ASTGI</td><td> $6 . 7 4 { \pm } 0 . 0 0 1 ^ { ( 1 1 ) }$ </td><td> $6 . 7 6 { \pm } 0 . 0 0 1 ^ { ( 1 1 ) }$ </td><td> $\overline { { 5 . 2 7 } } \pm 1 . 5 5 4 ^ { ( 7 ) }$ </td><td> $\overline { { 5 . 3 6 } } \pm 1 . 5 8 4 ^ { ( 7 ) }$ </td><td> $5 . 7 7 { \pm } 0 . 0 7 5 ^ { ( 3 ) }$ </td><td> $5 . 7 7 { \scriptstyle \pm 0 . 0 7 5 ^ { ( 2 ) } }$ </td><td> $3 . 1 1 \pm 0 . 0 8 0 ^ { ( 3 ) }$ </td><td> $3 . 2 1 { \pm } 0 . 0 8 5 ^ { ( 3 ) }$ </td></tr><tr><td>GRU-D</td><td> $4 . 9 8 { \pm } 0 . 0 1 8 ^ { ( 7 ) }$ </td><td> $5 . 0 0 { \pm } 0 . 0 2 0 ^ { ( 7 ) }$ </td><td> $7 . 5 6 { \pm } 0 . 6 1 4 ^ { ( 1 1 ) }$ </td><td> $7 . 6 5 { \pm } 0 . 6 0 0 ^ { ( 1 1 ) }$ </td><td> $6 . 1 5 { \pm } 0 . 0 2 3 ^ { ( 8 ) }$ </td><td> $6 . 1 6 { \pm } 0 . 0 2 4 ^ { ( 8 ) }$ </td><td> $4 . 5 3 { \pm } 0 . 4 0 8 ^ { ( 1 1 ) }$ </td><td> $4 . 6 9 { \pm } 0 . 4 0 8 ^ { ( 1 1 ) }$ </td></tr><tr><td>GraFITi</td><td> $\mathbf { 4 . 4 0 \pm 0 . 0 7 2 } ^ { ( 1 ) }$ </td><td> $\underline { { 4 . 4 7 } } \pm 0 . 0 7 3 ^ { ( 2 ) }$ </td><td> $\mathbf { 3 . 2 1 } { \pm } 0 . 5 8 8 ^ { ( 1 ) }$ </td><td> $\mathbf { 3 . 2 6 { \pm } 0 . 5 8 7 ^ { ( 1 ) } }$ </td><td> $5 . 8 3 { \pm } 0 . 0 3 5 ^ { ( 6 ) }$ </td><td> $5 . 7 9 { \pm } 0 . 0 3 4 ^ { ( 4 ) }$ </td><td> $\mathbf { 2 . 9 7 { \pm } 0 . 1 7 7 ^ { ( 1 ) } }$ </td><td> $\mathbf { 3 . 0 8 } \pm 0 . 1 9 1 ^ { ( 1 ) }$ </td></tr><tr><td>HyperIMTS</td><td> ${ \underline { { 4 . 4 6 } } } \pm 0 . 0 3 5 ^ { ( 2 ) }$ </td><td> $\mathbf { 4 . 4 1 } { \pm } 0 . 0 3 5 ^ { ( 1 ) }$ </td><td> $3 . 8 9 \pm 0 . 3 7 0 ^ { ( 3 ) }$ </td><td> $3 . 9 7 { \pm } 0 . 3 7 4 ^ { ( 3 ) }$ </td><td> ${ \bf 5 . 7 4 } \pm 0 . 0 2 4 ^ { ( 1 ) }$ </td><td> ${ \bf 5 . 7 6 \pm 0 . 0 2 4 ^ { ( 1 ) } }$ </td><td> $4 . 1 5 { \pm } 0 . 9 6 7 ^ { ( 8 ) }$ </td><td> $4 . 2 9 { \pm } 0 . 9 9 6 ^ { ( 8 ) }$ </td></tr><tr><td>KAFNet</td><td> $4 . 6 0 { \pm } 0 . 0 3 5 ^ { ( 3 ) }$ </td><td> $4 . 6 1 { \pm } 0 . 0 3 3 ^ { ( 3 ) }$ </td><td> $5 . 1 7 { \pm } 1 . 9 7 8 ^ { ( 6 ) }$ </td><td> $5 . 2 5 { \pm } 1 . 9 9 7 ^ { ( 6 ) }$ </td><td> $5 . 7 8 { \pm } 0 . 0 2 3 ^ { ( 4 ) }$ </td><td> $5 . 8 3 { \pm } 0 . 0 2 3 ^ { ( 5 ) }$ </td><td> $4 . 3 1 { \pm } 0 . 2 9 1 ^ { ( 9 ) }$ </td><td> $4 . 4 7 { \pm } 0 . 2 8 8 ^ { ( 9 ) }$ </td></tr><tr><td>NeuralFlow</td><td> $5 . 1 1 { \pm } 0 . 0 1 5 ^ { ( 8 ) }$ </td><td> $5 . 1 2 { \pm } 0 . 0 1 5 ^ { ( 8 ) }$ </td><td> $7 . 3 6 \pm 0 . 5 1 0 ^ { ( 1 0 ) }$ </td><td> $7 . 4 6 \pm 0 . 5 3 1 ^ { ( 1 0 ) }$ </td><td> $6 . 4 9 \pm 0 . 0 7 5 ^ { ( 1 0 ) }$ </td><td> $6 . 5 1 { \pm } 0 . 0 7 3 ^ { ( 1 0 ) }$ </td><td> $3 . 5 8 { \pm } 0 . 6 1 5 ^ { ( 5 ) }$ </td><td> $3 . 7 7 { \pm } 0 . 6 3 8 ^ { ( 6 ) }$ </td></tr><tr><td>SeFT</td><td> $5 . 1 5 { \pm } 0 . 0 4 8 ^ { ( 9 ) }$ </td><td> $5 . 1 6 \pm 0 . 0 4 6 ^ { ( 9 ) }$ </td><td> $6 . 6 3 { \pm } 0 . 1 5 2 ^ { ( 8 ) }$ </td><td> $6 . 7 2 { \scriptstyle \pm 0 . 1 5 2 ^ { ( 8 ) } }$ </td><td> $6 . 9 3 { \pm } 0 . 8 5 1 ^ { ( 1 1 ) }$ </td><td> $6 . 9 4 { \pm } 0 . 8 4 9 ^ { ( 1 1 ) }$ </td><td> $4 . 4 3 \pm 0 . 0 0 8 ^ { ( 1 0 ) }$ </td><td> $4 . 5 9 { \pm } 0 . 0 0 4 ^ { ( 1 0 ) }$ </td></tr><tr><td>Warpformer</td><td> $4 . 6 3 { \pm } 0 . 0 2 5 ^ { ( 4 ) }$ </td><td> $4 . 6 4 { \pm } 0 . 0 2 3 ^ { ( 4 ) }$ </td><td> $4 . 8 6 \pm 0 . 2 5 7 ^ { ( 5 ) }$ </td><td> $4 . 9 8 { \pm } 0 . 2 6 0 ^ { ( 5 ) }$ </td><td> $5 . 8 5 { \pm } 0 . 0 5 2 ^ { ( 7 ) }$ </td><td> $5 . 8 6 \pm 0 . 0 5 2 ^ { ( 7 ) }$ </td><td> $3 . 6 6 { \pm } 0 . 0 2 6 ^ { ( 6 ) }$ </td><td> $3 . 6 7 { \pm } 0 . 0 2 2 ^ { ( 4 ) }$ </td></tr><tr><td>mTAN</td><td> $5 . 3 8 \pm 0 . 0 4 3 ^ { ( 1 0 ) }$ </td><td> $5 . 3 9 \pm 0 . 0 4 5 ^ { ( 1 0 ) }$ </td><td> $6 . 9 6 \pm 0 . 2 4 2 ^ { ( 9 ) }$ </td><td> $7 . 0 5 { \pm } 0 . 2 2 9 ^ { ( 9 ) }$ </td><td> $6 . 1 9 \pm 0 . 1 2 7 ^ { ( 9 ) }$ </td><td> $6 . 2 0 { \pm } 0 . 1 2 5 ^ { ( 9 ) }$ </td><td> $3 . 0 1 { \pm } 0 . 1 0 6 ^ { ( 2 ) }$ </td><td> $3 . 1 3 { \pm } 0 . 1 0 7 ^ { ( 2 ) }$ </td></tr><tr><td>tPatchGNN</td><td> $4 . 6 5 { \pm } 0 . 0 2 3 ^ { ( 5 ) }$ </td><td> $4 . 6 5 { \pm } 0 . 0 2 1 \AA ^ { ( 5 ) }$ </td><td> $4 . 3 6 \pm 0 . 3 2 2 ^ { ( 4 ) }$ </td><td> $4 . 4 3 { \pm } 0 . 3 2 4 ^ { ( 4 ) }$ </td><td> $5 . 8 2 { \pm } 0 . 0 8 1 ^ { ( 5 ) }$ </td><td> $5 . 8 4 { \pm } 0 . 0 8 1 ^ { ( 6 ) }$ </td><td> $3 . 8 2 { \pm } 0 . 7 2 5 ^ { ( 7 ) }$ </td><td> $3 . 9 4 { \pm } 0 . 7 3 8 ^ { ( 7 ) }$ </td></tr></table>

GraFITi. GraFITi represents observed and queried values as nodes in a bipartite graph and performs graph-based information propagation for irregular forecasting (Yalavarthi et al., 2024). We use hidden dimension 128, four graph layers, one attention head, zero dropout, and batch size 32.

HyperIMTS. HyperIMTS constructs hypergraph interactions to model higher-order dependencies among asynchronously observed variables (Li et al., 2025). We use hidden dimension 128, three layers, one attention head, dropout 0.1, and batch size 32.

KAFNet. KAFNet employs kernel aggregation to prealign irregular observations before temporal modeling (Zhou et al., 2026). We use hidden dimension 16, pre-convolution dimension 32, one layer, one attention head, four Gaussian kernels, time-embedding dimension 10, and batch size 32.

NeuralFlow. NeuralFlow parameterizes continuous latent dynamics through an invertible flow, providing an efficient alternative to numerical ODE solvers (Bilos et al., 2021). We use hidden dimension 100, two coupling-flow layers, latent dimension 20, time-encoding hidden dimension 8, three hidden layers, reconstruction dimension 30, and batch size 32.

SeFT. SeFT treats all irregular observations as an unordered set and uses learned set functions and attention pooling to obtain a fixed-dimensional representation (Horn et al., 2020). We use hidden dimension 128, three $\phi / \rho$ layers, two ψ layers, four attention heads, positional dimension 4, dropout 0.1, and batch size 32.

dimension 64, one attention head, key/value dimensions of 8, and batch size 32.

Warpformer. Warpformer applies multiscale temporal warping to transform irregular observations into representations suitable for Transformer-based modeling (Zhang et al., 2023). We use hidden dimension 128, three layers, inner mTAN. mTAN employs multi-time attention to map irregular observations onto continuous reference timestamps (Shukla & Marlin, 2021). We use hidden dimension 128, latent dimension 32, one attention head, 128 reference points, observation standard deviation 0.01, and batch size 32.

tPatchGNN. tPatchGNN organizes irregular observations into transformable temporal patches and models variable interactions using graph neural networks (Zhang et al., 2024). We use hidden dimension 32, one graph layer, one attention head, one Transformer layer, one-hop propagation, nodeembedding dimension 10, time-embedding dimension 10, a linear output head, and batch size 32.

Controlled-experiment baselines. The controlled experiments use SeFT, CRU, GraFITi, tPatchGNN, and KAFNet. SeFT uses hidden dimension 64, two $\phi / \rho$ layers, two ψ layers, two attention heads, and dropout 0.02. CRU uses hidden dimension 20 and 20 basis functions (Schirmer et al., 2022). GraFITi uses hidden dimension 28 and two graph layers. tPatchGNN uses hidden dimension 32, patch length 4, one graph layer, and one Transformer layer. KAFNet uses hidden dimension 64, pre-convolution dimension 32, two layers, eight Gaussian kernels, and time-embedding dimension 16.

## D.4. Training and Evaluation Details

Training protocol. All models are trained using the Adam optimizer and masked observation-point MSE. For prediction mask $m _ { b i d } ,$ the training objective is

$$
\mathcal { L } _ { \mathrm { t r a i n } } = \frac { \sum _ { b , i , d } m _ { b i d } \left( \widehat { y } _ { b i d } - y _ { b i d } \right) ^ { 2 } } { \sum _ { b , i , d } m _ { b i d } } .\tag{48}
$$

![](images/19afa674bef5a0e7002af24ee85d4e4e98d4059d53209822c43157b2c3cb108c.jpg)

![](images/d0bca014e002ad3ba7930476f7b6a45f1790bf15f626514006876442caf4433c.jpg)

![](images/15f2f8c791222b11fe7530a5b409904c4f0e53a5f41565224f9464c4075086dc.jpg)  
(c) ETTm1

![](images/2a719932f542f4b3d3036308f3d73e9d0ea956994efdd3c525961464141dbd1c.jpg)  
(d) Weather  
Figure 5. Relative estimation errors of MSE and CSE with respect to $R _ { \mathrm { c t } }$ under increasing sampling non-uniformity on all synthetic and semi-synthetic datasets.

All baselines are trained for 100 epochs, and the checkpoint with the lowest validation MSE is retained for testing. The learning rate is $1 0 ^ { - 3 }$ for all models. NeuralFlow and mTAN use weight decay $1 0 ^ { - 4 }$ , while the remaining real-world baselines use no weight decay. Models that originally employ probabilistic objectives, including mTAN and NeuralFlow, are also optimized using the common MSE objective to ensure a unified training protocol.

The controlled experiments use learning rate $1 0 ^ { - 3 } .$ , weight decay $1 0 ^ { - 5 }$ , batch size 64, and global gradient-norm clipping at 1.0. Training and validation target timestamps are uniformly sampled from the future interval.

All model–dataset experiments are independently repeated with three random seeds, 2024, 2025, and 2026. We report the mean and standard deviation across the three runs. In the controlled sampling-shift experiment, each trained checkpoint is further evaluated using 20 independent timestamp resamples for each sampling strength α.

Temporal-density estimation. Future timestamps are mapped to the common forecasting interval [0, 1] before density estimation. We use a trajectory-specific leave-one-out Gaussian kernel density estimator. The bandwidth follows a robust Silverman rule:

$$
h _ { s } = \operatorname* { m a x } \left\{ 0 . 9 \operatorname* { m i n } \left( \sigma _ { s } , \frac { \mathrm { I Q R } _ { s } } { 1 . 3 4 } \right) L _ { s } ^ { - 1 / 5 } , \frac { 1 } { \operatorname* { m a x } ( 2 0 , 2 L _ { s } ) } , 1 0 ^ { - 6 } \right\} .
$$

Non-positive scale estimates are excluded when computing the minimum. Estimated densities are lower-bounded by $1 0 ^ { - 8 }$ , and no importance-weight clipping or boundary reflection is applied. Conditional and marginal timestamp densities are estimated only from test timestamps and are shared by all models evaluated on the same dataset.

Metric computation. For each model–dataset–seed combination, the selected checkpoint is loaded once, and MSE, GSE, and CSE are computed from the same test predictions. Let $m _ { s i d }$ indicate whether variable d is observed at future timestamp $t _ { s , i }$ and let $w _ { s , i }$ denote the corresponding temporal weight. The trajectory-level weighted error is

Table 7. Complete model-evaluation results on the two synthetic datasets under $\alpha = 0 . 9$ . Values are mean ± standard deviation over 20 timestamp resamples; $R _ { \mathrm { c t } }$ is computed from the complete trajectory. Superscripts denote ranks.
<table><tr><td rowspan="3">Model</td><td colspan="3">Synthetic-Regime  $\times 1 0 ^ { - 2 }$ </td><td colspan="3">Synthetic-Multiscale  $\times 1 0 ^ { - 2 }$ </td></tr><tr><td>MSE</td><td>CSE</td><td> $R _ { \mathrm { c t } }$ </td><td>MSE</td><td>CSE</td><td> $R _ { \mathrm { c t } }$ </td></tr><tr><td>SeFT</td><td> $1 . 0 8 8 4 { \pm } 0 . 0 2 2 8 ^ { ( 3 ) }$ </td><td> $\mathbf { 8 . 6 9 9 4 } \pm 0 . 3 7 6 0 ^ { ( 1 ) }$ </td><td> ${ \bf 6 . 2 5 5 2 ^ { ( 1 ) } }$ </td><td> $1 . 2 6 5 5 { \pm } 0 . 0 0 5 8 ^ { ( 3 ) }$ </td><td> $1 . 1 1 6 9 { \pm } 0 . 0 6 3 2 ^ { ( 3 ) }$ </td><td> $1 . 1 6 5 9 ^ { ( 3 ) }$ </td></tr><tr><td>CRU</td><td> $\mathbf { 0 . 6 5 2 7 { \overset { . } { \bot } } 0 . 0 2 3 2 { \overset { . } { ( 1 ) } } }$ </td><td> $\underline { { 8 . 8 1 0 5 } } \pm 0 . 3 8 1 4 ^ { ( 2 ) }$ </td><td> ${ \underline { { 6 . 4 8 3 1 } } } ^ { ( 2 ) }$ </td><td> $1 . 3 1 1 8 { \pm } 0 . 0 0 5 5 ^ { ( 4 ) }$ </td><td> $\mathbf { 0 . 8 0 8 5 \pm 0 . 0 3 8 0 ^ { ( 1 ) } }$ </td><td> $\mathbf { 0 . 9 5 7 5 } ^ { ( 1 ) }$ </td></tr><tr><td>GraFITi</td><td> $2 . 5 7 4 1 { \scriptstyle \pm 0 . 0 2 3 3 } ^ { ( 5 ) }$ </td><td> $9 . 8 7 7 6 { \pm } 0 . 4 6 4 4 ^ { ( 5 ) }$ </td><td> $7 . 6 5 8 8 ^ { ( 5 ) }$ </td><td> $1 . 6 8 2 2 { \pm } 0 . 0 0 5 3 ^ { ( 5 ) }$ </td><td> $1 . 4 7 0 8 { \scriptstyle \pm 0 . 0 4 7 7 } ^ { ( 5 ) }$ </td><td> $1 . 5 0 5 9 ^ { ( 5 ) }$ </td></tr><tr><td>tPatchGNN</td><td> $0 . 9 1 3 2 { \pm } 0 . 0 2 4 1 ^ { \left( 2 \right) }$ </td><td> $9 . 3 7 9 8 { \pm } 0 . 3 6 0 3 ^ { ( 4 ) }$ </td><td> $6 . 9 1 1 9 ^ { ( 4 ) }$ </td><td> $0 . 8 9 3 9 { \pm } 0 . 0 0 5 2 ^ { ( 2 ) }$ </td><td> $1 . 1 2 7 2 { \scriptstyle \pm 0 . 0 6 8 7 ^ { ( 4 ) } }$ </td><td> $1 . 1 7 8 9 ^ { ( 4 ) }$ </td></tr><tr><td>KAFNet</td><td> $1 . 5 4 3 5 { \pm } 0 . 0 2 3 2 ^ { ( 4 ) }$ </td><td> $9 . 0 4 3 7 { \pm } 0 . 3 6 0 2 ^ { ( 3 ) }$ </td><td> $6 . 7 5 1 2 ^ { ( 3 ) }$ </td><td> $\mathbf { 0 . 8 0 8 8 \pm 0 . 0 0 5 7 ^ { ( 1 ) } }$ </td><td> $1 . 0 7 5 1 { \pm } 0 . 0 5 1 1 ^ { ( 2 ) }$ </td><td> $\underline { { 1 . 1 5 6 7 } } ^ { ( 2 ) }$ </td></tr></table>

$$
\widehat { R } _ { s } ( w ) = \frac { \sum _ { i , d } m _ { s i d } w _ { s , i } \left( \widehat { y } _ { s i d } - y _ { s i d } \right) ^ { 2 } } { \sum _ { i , d } m _ { s i d } w _ { s , i } } .\tag{50}
$$

The observation-distribution, global-time, and continuoustime weights are

$$
\begin{array} { r } { w _ { s , i } ^ { \mathrm { o b s } } = 1 , } \\ { w _ { s , i } ^ { \mathrm { g l o b a l } } = \frac { \widehat { p } _ { T } ( t _ { s , i } ) } { \widehat { p } ( t _ { s , i } \mid X _ { s } ) } , } \\ { w _ { s , i } ^ { \mathrm { c t } } = \frac { 1 } { \widehat { p } ( t _ { s , i } \mid X _ { s } ) } . } \end{array}\tag{51}
$$

Each trajectory is normalized independently, after which the trajectory-level errors are averaged with equal weight. Padding positions and timestamps without any observed target variables are excluded from both density estimation and metric computation.

Implementation environment. All experiments are implemented in Python 3.12 and PyTorch 2.10.0 with CUDA 12.8. Training and evaluation are conducted on NVIDIA GeForce RTX 4090 GPUs.

## E. Additional Experimental Results

## E.1. Complete Results on Synthetic and Semi-Synthetic Data

Figure 5 reports the relative estimation errors of MSE and CSE with respect to $R _ { \mathrm { c t } }$ on all four controlled datasets. The curves show the mean over 20 independent test-timestamp resamples, and the shaded regions denote one standard deviation. Because the trained models and underlying future trajectories remain fixed across different values of α, the observed changes are caused solely by the shift in the test-time sampling distribution. Under uniform sampling, the inversedensity weights are constant and CSE consequently reduces to MSE. As α increases, observations become increasingly concentrated in a limited portion of the future interval. MSE therefore places disproportionate emphasis on densely sampled regions, and its deviation from $R _ { \mathrm { c t } }$ generally increases.

In contrast, CSE compensates for this concentration through inverse-density weighting and remains substantially closer to $R _ { \mathrm { c t } }$ under strong non-uniform sampling. Although density estimation and importance weighting may introduce additional finite-sample variability, the correction becomes increasingly beneficial as the mismatch between the observed and uniform temporal distributions grows. The consistent behavior on ETTm1 and Weather further shows that this advantage is not restricted to analytically generated trajectories.

Tables 7 and 8 report the complete model-level results under $\alpha = 0 . 9$ . On all four datasets, CSE recovers exactly the same ranking of the five models as $R _ { \mathrm { c t } }$ , whereas MSE produces at least one ranking inversion. This result indicates that the effect of temporal sampling is not limited to a uniform shift in metric values: different models are affected to different degrees because their prediction errors vary differently over time. Correcting the target-time distribution therefore improves both risk estimation and the recovery of model comparisons, supporting the main-text conclusion that CSE more accurately reflects continuous-time predictive performance under non-uniform sampling.

## E.2. Complete Results on Real-World Datasets

Tables 3 and 4 report the complete MSE and CSE results on the eight real-world datasets, including standard deviations over three random seeds. Tables 5 and 6 provide the corresponding MAE and Continuous-Time Absolute Error (CAE) results. The best and second-best mean values are shown in bold and underlined, respectively, and superscripts denote ranks computed from the unrounded means.

Consistent with the observations in the main text, CSE is neither a fixed-direction nor a fixed-ratio transformation of MSE. On CESNET, FNSPID, GDELT, and HumanActivity, CSE is generally lower than MSE, whereas the opposite trend is more common on MIMIC-III, RepoHealth, and USHCN. Moreover, even within the same dataset, the magnitude of the change varies across models. This confirms that the sampling distribution interacts with model-specific temporal error patterns rather than simply applying a datasetlevel scaling factor. The reported standard deviations also distinguish this systematic metric effect from training variability: most model–dataset combinations remain relatively stable across seeds, although datasets such as RepoHealth and USHCN exhibit larger variation for several models.

Table 8. Complete model-evaluation results on the two semi-synthetic datasets under $\alpha = 0 . 9 .$ . Values are mean ± standard deviation over 20 timestamp resamples; $R _ { \mathrm { c t } }$ is computed on the complete future grid. Superscripts denote ranks.
<table><tr><td rowspan="3">Model</td><td colspan="3"> $\mathrm { E T T m 1 }$   $\times 1 0 ^ { 0 }$ </td><td colspan="3">Weather</td></tr><tr><td colspan="3"></td><td colspan="3">×10−1</td></tr><tr><td>MSE</td><td>CSE</td><td> $R _ { \mathrm { c t } }$ </td><td>MSE</td><td>CSE</td><td> $R _ { \mathrm { c t } }$ </td></tr><tr><td>SeFT</td><td> $1 . 1 0 3 2 { \pm } 0 . 0 0 5 9 ^ { ( 4 ) }$ </td><td> $1 . 1 1 4 5 { \pm } 0 . 0 1 5 3 ^ { ( 4 ) }$ </td><td> $1 . 1 1 3 0 ^ { ( 4 ) }$ </td><td> $4 . 4 0 4 6 { \scriptstyle \pm 0 . 0 1 4 6 } ^ { ( 5 ) }$ </td><td> $4 . 9 6 2 9 { \pm } 0 . 0 6 2 9 ^ { ( 5 ) }$ </td><td> $4 . 8 4 9 3 ^ { ( 5 ) }$ </td></tr><tr><td>CRU</td><td> $1 . 3 3 9 4 { \pm } 0 . 0 0 5 8 ^ { ( 5 ) }$ </td><td> $1 . 1 0 0 1 { \scriptstyle \pm 0 . 0 1 5 3 ^ { ( 3 ) } }$ </td><td> $1 . 0 9 9 4 ^ { ( 3 ) }$ </td><td> $\mathbf { 3 . 0 8 3 9 } \pm 0 . 0 1 4 4 ^ { ( 1 ) }$ </td><td> $4 . 6 0 8 7 { \scriptstyle \pm 0 . 0 6 6 7 ^ { ( 3 ) } }$ </td><td> $4 . 4 6 3 1 ^ { ( 3 ) }$ </td></tr><tr><td>GraFITi</td><td> $1 . 0 8 8 6 \pm 0 . 0 0 5 6 ^ { ( 3 ) }$ </td><td> $1 . 4 1 4 0 { \scriptstyle \pm 0 . 0 1 6 3 } ^ { ( 5 ) }$ </td><td> $1 . 4 0 1 6 ^ { ( 5 ) }$ </td><td> $3 . 8 9 1 8 { \pm } 0 . 0 1 4 7 ^ { ( 4 ) }$ </td><td> $4 . 9 2 2 1 { \scriptstyle \pm 0 . 0 6 6 5 } ^ { ( 4 ) }$ </td><td> $4 . 7 6 1 2 ^ { ( 4 ) }$ </td></tr><tr><td>tPatchGNN</td><td> $\underline { { 0 . 9 2 5 1 } } \pm 0 . 0 0 5 6 ^ { ( 2 ) }$ </td><td> $\mathbf { 0 . 9 3 3 0 { \pm 0 . 0 1 5 1 } ^ { ( 1 ) } }$ </td><td> $\mathbf { 0 . 9 3 0 2 ^ { ( 1 ) } }$ </td><td> $3 . 8 7 8 9 { \pm } 0 . 0 1 8 1 ^ { ( 3 ) }$ </td><td> $\mathbf { 3 . 6 8 5 6 } \pm 0 . 0 6 4 5 ^ { ( 1 ) }$ </td><td> $\mathbf { 3 . 5 2 7 4 } ^ { ( 1 ) }$ </td></tr><tr><td>KAFNet</td><td> $\mathbf { 0 . 9 0 9 3 { \scriptstyle \pm 0 . 0 0 5 6 } ^ { ( 1 ) } }$ </td><td> $0 . 9 6 4 0 { \scriptstyle \pm 0 . 0 1 5 5 ^ { ( 2 ) } }$ </td><td> $\underline { { 0 . 9 5 8 4 } } ^ { ( 2 ) }$ </td><td> $3 . 4 8 3 9 { \pm } 0 . 0 1 4 4 ^ { ( 2 ) }$ </td><td> $3 . 7 7 8 3 { \scriptstyle \pm 0 . 0 6 0 7 ^ { ( 2 ) } }$ </td><td> $3 . 6 4 9 6 ^ { ( 2 ) }$ </td></tr></table>

The MAE–CAE comparison exhibits a similar pattern. The direction of the correction is broadly consistent with the MSE–CSE comparison, while its magnitude remains dependent on both the dataset and the model. In particular, the best-performing model on MIMIC-III changes from GraFITi under MAE to HyperIMTS under CAE, accompanied by additional local rank changes on several datasets. Therefore, the discrepancies reported in the main text are not specific to squared error, but arise more generally from the tempo ral distribution under which pointwise prediction losses are aggregated. Since the complete continuous trajectories of real-world datasets are unavailable, these results do not directly identify which metric is exact; rather, the systematic differences suggest that relying solely on observation-point metrics may not fully characterize models’ continuous-time predictive performance.
# LOB-ID: Evaluating Synthetic Market Data by Inception Distances

Andreea Bacalum Simudyne London, United Kingdom

Zhuohan Wang<sup>∗</sup> Simudyne London, United Kingdom

Martin Garaj Simudyne London, United Kingdom

Ollie Olby Simudyne London, United Kingdom

Namid Stillman   
Simudyne   
London, United Kingdom   
namid@simudyne.com

## Abstract

Generative models of limit orderbook (LOB) data have advanced rapidly, but their evaluation often focuses on stylised facts and selected market statistics. These measures provide useful diagnostics but may not capture the joint temporal and cross-level structure of order-book trajectories. We introduce LOB-ID, an embedding-based framework that adapts the Fréchet Inception Distance (FID) and Monge Inception Distance (MIND) to LOB data. To obtain domainspecific embeddings, we train the DeepLOB architecture on four months of Level-2 order-book data for five equities. We show that LOB-ID is stable across time, instruments, and embedding check points, and rises monotonically under controlled distortions. We then construct a moment-matching attack against FID and a deep book perturbation that evades statistic-based evaluation. MIND remains substantially more sensitive to both distortions. Finally, we score five generative LOB models, spanning stochastic baselines and deep learning approaches, and find that LOB-ID ranks them in line with the joint temporal and cross-level structure each captures by construction.

## Keywords

synthetic market data, limit orderbooks, generative model evaluation, embedding-based evaluation, inception networks

## 1 Introduction

Synthetic market data is becoming an increasingly useful tool in quantitative finance [1]. It supports applications such as backtesting, execution analysis, risk measurement, and market-microstructure research, and it reduces reliance on historical datasets that can be costly. It also addresses a more basic limitation of historical data: real markets provide only one realised path of what occurs, whereas generative models produce many alternative scenarios, allowing market behaviour to be studied beyond what has already happened.

A limit orderbook (LOB) records outstanding bids and asks at each price level, with a matching engine pairing incoming orders against resting ones by price-time priority [9]. The book evolves through a stream of messages, so a synthetic LOB model must generate this order flow rather than prices alone.

Recent generative models have substantially improved the realism of synthetic order flow, but progress in generation has outpaced progress in evaluation [6, 14, 15]. Historical data provides a ground-truth reference, yet evaluation cannot be posed as matching a specific sequence. Models are not expected to perfectly reproduce a single realised market path, but to produce plausible alternatives from the same underlying distribution. Evaluation is therefore a question of distributional similarity rather than point-wise accuracy. Hence, most existing validation methods assess generative models of LOB data by checking whether the generated data reproduce selected properties of real markets. Such comparisons are informative but partial, since a sequence can match a wide range of individual statistics and still fail to behave like a coherent market trajectory.

What these comparisons tend to miss is the underlying joint structure of the data. A realistic LOB sequence should preserve the dependence between successive book states, between liquidity at the touch and in the deeper levels, between the bid and ask sides of the book, and between all of these and the prevailing market regime. Such dependencies are hard to capture with a fixed set of hand-designed statistics, as they are spread across time and price levels, and evaluating them benefits from a representation of the order-book trajectory as a whole.

We address this problem by introducing inception distances for LOB (LOB-ID), an embedding-based framework for evaluating synthetic LOB data. To achieve this, we train a DeepLOB network on four months of Level-2 order-book data for five equities listed on the Hong Kong Stock Exchange (HKEX) [20]. After removing the LSTM and classification head, we use the network’s Inception representation as a domain-specific feature extractor. We then apply the Fréchet Inception Distance (FID) and the Monge Inception Distance (MIND) to the resulting embeddings [3, 10]. As the metrics are based on representations constructed from Level-2 order-book windows, they are sensitive to the joint structure that statistic-based comparisons can overlook.

This work makes four contributions. First, we introduce LOB-ID, adapting FID and MIND metrics from image-generation evaluation to representations learned from LOB data. Second, we show that MIND is consistent across nearby trading days, distinguishes between instruments, responds monotonically to controlled perturbations, and stabilises at later stages of DeepLOB training. Third, we analyse the adversarial weaknesses of FID and selected statisticbased LOB evaluations and show that MIND remains sensitive to the loss of joint structure they miss. Fourth, we compare Zero Intelligence, Compound Hawkes, LOBGAN, LOBS5 and DifLOB on a common dataset using LOB-ID, stylised facts, and existing benchmark metrics. We believe our work will support a more rigorous evaluation of synthetic market data by assessing the joint structure of the generated order-book trajectories that standard statistics may miss.

## 2 Related Work

Synthetic LOB models. Early methods for generating synthetic LOB data were typically based on stochastic models. For example, zero-intelligence models generate orders according to exogenously specified stochastic rules, without explicitly modeling strategic optimisation or learning by market participants [8]. Alternatively, other models use Hawkes processes to allow past events to afect the rate of future events, which have been shown to capture clustered and self-exciting order flow [2, 11]. Although interpretable, these models require the event structure and arrival dynamics to be specified and calibrated in advance.

The development of deep learning shifted the focus towards learning representations directly from LOB data. For example, LOB-GAN generates synthetic order flow using a conditional generative adversarial network $[ 5 ,$ 6]. LOBS5 instead treats order-flow generation as an end-to-end autoregressive sequence-modelling task [14]. LOBS5 tokenises message-level LOB data and generates the tokens sequentially using simplified structured state-space layers, which can process long histories eficiently. More recently, DifLOB uses a regime-conditioned difusion model to generate future LOB trajectories under specified market conditions [19]. It conditions on historical LOB states and future regimes defined by trend, volatility, liquidity, and order-flow imbalance. This enables controllable and counterfactual generation, allowing the model to simulate how the orderbook may evolve under hypothetical future conditions.

Evaluation ofsynthetic LOB data. Generative LOB models have often been evaluated using so-called ‘stylised facts’ such as heavytailed returns, volatility clustering, and persistent order-flow dependence [7, 13]. These tests show whether a simulator reproduces well-known properties of real markets. However, the choice of statistics and the way they are measured often difer across studies, which makes direct comparison dificult.

More recently, LOB-Bench introduced a standardised framework for evaluating synthetic LOB data [15]. It compares real and gener ated data through distributions of selected market-microstructure statistics, using both the $L _ { 1 }$ norm and Wasserstein-1 distance. The framework also evaluates price-impact responses across multiple lags and includes a trained discriminator that measures how well real and generated trajectories can be separated. Together, these components provide a broad assessment of the realism and quality of synthetic LOB data. In this work, we use the distributional and price-impact components of LOB-Bench as an evaluation framework and do not directly evaluate its discriminator.

Embedding-based generative metrics. Embedding-based metrics ofer a complementary approach to evaluation. They pass real and generated samples through a fixed, pre-trained network and compare the resulting feature distributions. Similarity is therefore mea sured in a learned representation space rather than directly from raw inputs or hand-selected summary statistics. Past work has considered embedding networks within the context of calibration for generative models of market data but not the evaluation of the resulting synthetic data [17].

The Fréchet Inception Distance (FID) was introduced for evaluating generated images by comparing Gaussian approximations of the Inception-v3 feature distributions of real and generated samples [10]. However, FID captures only the first two moments of these distributions, and its empirical estimate can be biased or unstable when the sample size is small [4].

The Monge Inception Distance (MIND) was introduced as an alternative to FID for evaluating generative models [3]. Rather than approximating the embedding distributions as Gaussian, MIND computes a scaled average of the squared 2-Wasserstein distances between their one-dimensional projections along random unit directions. This allows it to capture diferences beyond the first two moments without estimating high-dimensional covariance matrices.

## 3 Methods

In this section, we outline how we adapt the Fréchet Inception Distance (FID) and the Monge Inception Distance (MIND) to the evaluation of LOB data. Both metrics compare distributions of feature embeddings produced by the pre-trained network, conventionally the Inception-v3 model trained on ImageNet. Since features learned from natural images are inappropriate for financial data, we replace this backbone with a domain-specific feature extractor trained directly on LOB data. We use the DeepLOB model, whose architecture and training are described in Section 3.2.1. The trained network yields an embedding function $\psi _ { w }$ that maps a window of � = 100 successive event-indexed Level-2 book states, $x \in \mathbb { R } ^ { T \times 4 0 }$ , to a fixed 96-dimensional feature vector. This representation plays the same role as the Inception-v3 features in the image setting, while capturing relationships across price levels and over time that are specific to LOB data.

Both real and generated samples are passed through $\psi _ { w }$ to obtain two sets of embeddings, whose distributions are then compared. FID compares Gaussian approximations of the embedding distributions, whereas MIND compares their random one-dimensional projections. Lower values indicate greater similarity in the learned LOB representation space.

All experiments use four months of message-level (L3) data collected between September and December 2025 for five liquid equities listed on the Hong Kong Stock Exchange (HKEX): 0700.HK (Tencent), 1024.HK (Kuaishou), 9999.HK (NetEase), 3690.HK (Meituan), and 9888.HK (Baidu). From the raw event streams, we construct the order messages and corresponding Level-2 books containing the ten best price levels on each side. We split the data chronologically, using 1 September to 12 December for training, 13 to 20 December for validation, and 21 to 31 December for testing, so that all evaluation is performed on market days that follow the training period. We choose this period so that evaluation spans a range of liquidity conditions: the training window covers active trading, while the validation and test windows fall in the year-end holiday period, when participation is materially reduced. This provides a natural distribution shift under which to assess how the LOB-ID metric changes in thin markets. The same split and book-reconstruction pipeline are used throughout the experiments, for both the embedding network and the generative models. Although the source dataset contains message-level data and some generators operate at the message level, all inputs to the LOB-ID feature extractor are Level-2 book-state windows. Accordingly, when discussing joint structure in the context of LOB-ID, we refer specifically to temporal, cross-level, and bid–ask dependencies within Level-2 trajectories. LOB-ID does not directly evaluate the full message-level order-flow process.

## 3.1 Synthetic Market Data

We train and evaluate five synthetic LOB models using the common dataset and temporal split described above: Zero-intelligence (ZI), Compound Hawkes process (CH), LOBGAN, LOBS5 and DifLOB.

Our implementations of the ZI and CH baselines are adapted from the calibration procedures used in Kawawa-Beaudan et al. [12]. The ZI model independently samples order side and action type from empirical categorical distributions, inter-arrival times and volumes from fitted exponential distributions, and price depth from a Gaussian mixture model. It therefore preserves calibrated marginals while removing temporal and conditional dependence. The CH model introduces temporal dependence through an eightdimensional Hawkes process, with one dimension for each action– side combination and sum-of-exponential kernels at four fixed timescales. For each test chunk, it conditions on the preceding 1024 real events and generates the next 1024 events using Ogata’s thinning algorithm [16]. Volumes and price depths are sampled from fitted exponential and Gaussian-mixture mark distributions, respectively.

We also consider three deep generative models for LOB data. LOBGAN follows the conditional architecture of Coletta et al. [6]. Its generator combines an LSTM representation of the previous 30 nine-dimensional market states with a 64-dimensional Gaussian noise vector to generate a seven-feature order representation. During closed-loop training, generated orders update the marketstate features fed back to the generator for up to five unrolled steps. LOBS5 is implemented as a masked-token generator over message-level LOB data [14]. Its message and book branches pro cess 22-token order messages and a 41-dimensional signed volume image using structured state-space layers. Generation proceeds token by token from a real message context, with each completed message and updated book state fed back into the model. DifLOB is a regime-conditioned difusion model that jointly generates future Level-2 price and volume trajectories rather than individual order messages [19]. During training, the conditioning information is randomly dropped with probability 0.5, allowing the model to learn both conditional and unconditional score estimates. We evaluate DifLOB in two settings. First, we condition it on the historical LOB and the realised future trend, volatility, liquidity, and order-flow imbalance to test whether it can reproduce a specified market regime. Secondly, to observe how the model performs without conditioning information, we set all conditioning inputs to null, so the model generates unconditionally from the learned distribution.

A central challenge for LOB models is determining whether the resulting synthetic data are realistic. We evaluate each generator using three complementary approaches. First, we test the 11 stylised facts of financial time series considered by Cont [7]. Second, we compute the distributional metrics introduced by LOB-Bench [15], which compare real and generated data using distances over a standard set of market-microstructure statistics, including spread, imbalance, timing, and volume measures. The full list of statistics is provided by Nagy et al. [15]. Since all generators are evaluated on Level-2 book states, we use 12 of the 21 LOB-Bench measures that do not require message-level data. Finally, we introduce LOB-ID, our own method of evaluating synthetic order-book data, an embedding-based evaluation framework using Inception Distances. We provide a comparison of the relative ranking of models with respect to these diferent evaluation frameworks in Section 4.5.

## 3.2 Inception Distances

3.2.1 Inception Network. An Inception network is a deep convolutional neural network built around the Inception module, introduced in GoogLeNet by Szegedy et al. [18]. The module applies convolutions of multiple sizes in parallel and concatenates their outputs, allowing features to be captured at multiple scales.

DeepLOB combines convolutional layers, an Inception module, and an LSTM to predict short-term mid-price movements from raw LOB data [20]. We follow the architecture, preprocessing, and training procedure of Zhang et al. [20], using the dataset and temporal split described above. Each input contains the most recent $T = 1 0 0$ order-book events, $x \in \mathbb { R } ^ { T \times 4 0 }$ , with price and size information from the ten best bid and ask levels, producing a 96-dimensional feature vector.

An LSTM and linear classification head use these features to predict whether the mid-price will move down, remain stationary, or move up over the next 100 messages. On the held-out test set, the trained DeepLOB classifier achieves an accuracy of 75.44% and a macro- ${ \bf \nabla } \cdot { \cal F } _ { 1 }$ score of 75.29%, indicating balanced predictive performance across the three mid-price movement classes. For LOB-ID, we remove the recurrent and classification layers and use the time-averaged output of the Inception module as the embedding, namely

$$
\psi _ { w } ( x ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathrm { I n c e p t i o n } \bigl ( \mathrm { C o n v } ( x ) \bigr ) _ { : , t } \in \mathbb { R } ^ { 9 6 } .\tag{1}
$$

3.2.2 Fréchet Inception Distance $\left( F I D \right)$ . The Fréchet Inception Distance (FID) compares real and generated embeddings in the feature space of a fixed network [10]. Let $ { p _ { \mathrm { d a t a } } }$ and $\scriptstyle { \mathcal { P } } \theta$ denote their respective distributions. FID fits Gaussians with empirical means $\mu , \tilde { \mu }$ and covariance matrices $\Sigma , \tilde { \Sigma }$ , and computes

$$
\mathrm { F I D } ( p _ { \theta } , \bar { p } _ { \mathrm { d a t a } } ) = \| \mu - \tilde { \mu } \| _ { 2 } ^ { 2 } + \mathrm { t r } \Bigg [ \Sigma + \tilde { \Sigma } - 2 \left( \Sigma ^ { 1 / 2 } \tilde { \Sigma } \Sigma ^ { 1 / 2 } \right) ^ { 1 / 2 } \Bigg ] .\tag{2}
$$

A lower FID indicates greater similarity between the real and generated embedding distributions. However, the Gaussian approximation means that FID depends only on their means and covariances and therefore cannot detect diferences beyond the first two moments.

3.2.3 Monge Inception Distance (MIND). The Monge Inception Distance (MIND) is an embedding-based measure introduced as an alternative to FID [3]. Rather than fitting Gaussian approximations to the embedding distributions, MIND averages the squared 2-Wasserstein distances between their one-dimensional projections along random unit directions. For embeddings in $\mathbb { R } ^ { 9 6 }$ and a unit direction � drawn uniformly from the sphere $\mathbb { S } ^ { 9 5 }$ , MIND is defined as

$$
\begin{array} { r } { \mathrm { M I N D } ( p _ { \theta } , p _ { \mathrm { d a t a } } ) = \alpha \mathbb { E } _ { u \sim \mathcal { U } ( \mathbb { S } ^ { 9 5 } ) } \left[ W _ { 2 } ^ { 2 } \left( \boldsymbol { u } ^ { \top } \boldsymbol { p } _ { \theta } , \boldsymbol { u } ^ { \top } \boldsymbol { p } _ { \mathrm { d a t a } } \right) \right] . } \end{array}\tag{3}
$$

Here, $\boldsymbol { u } ^ { \intercal } \boldsymbol { p }$ denotes the distribution of $u ^ { \top } X$ for $X \sim p ,$ and $W _ { 2 } ^ { 2 }$ is the squared 2-Wasserstein distance. Following the empirical scaling proposed by Berthet et al. [3], we use $\alpha = 3 d = 2 8 8$ , where $d = 9 6$ is the embedding dimension. Note that this construction difers from the Wasserstein-1 distance used by LOB-Bench [15], which compares univariate distributions of individual scalar market statistics rather than projections of the embedding distribution. All reported MIND values estimate the expectation in Eq. (3) using 256 random projections.

For uniformly weighted empirical distributions of equal size, each one-dimensional squared 2-Wasserstein distance can be computed exactly by sorting the projected values. MIND therefore avoids estimating and storing high-dimensional covariance matrices. At the population level, MIND is zero if and only if the two embedding distributions are equal. Matching only their means and covariances is therefore insuficient to make MIND zero. Berthet et al. [3] show empirically that MIND is more resistant, though not immune, to moment-matching attacks than FID. We confirm this behaviour for LOB embeddings in Section 4.3.1, where MIND remains substantially more sensitive than FID after the embedding moments are approximately matched. We therefore use MIND as the primary score within the LOB-ID framework throughout the remainder of the paper.

## 4 Results

In this section, we show that LOB-ID behaves as a reliable evaluation metric for synthetic market data: it is consistent across time and instruments, responds monotonically to perturbations, stabilises across embedding checkpoints, and remains sensitive to adversarial attacks that FID and selected LOB-Bench statistics substantially underestimate or miss. We then use it to score the generative models of Section 3.1. All experiments use the trained DeepLOB Inception layer as the embedding network. The stability analysis and the moment-matching attack use all five stocks over the six test days, since they require only real data. Experiments involving the gen erative models, namely the joint-structure attack, the checkpoint analysis, and the model comparison, use three stocks (9999.HK, 1024.HK, 0700.HK), as each generator must be separately trained and calibrated per instrument.

## 4.1 Stability and Robustness of LOB-ID

We expect samples from the same stock taken on nearby days to have similar LOB structure. We therefore compare MIND across all stock–day pairs in the test period. Figure 1 shows the pairwise MIND distances. The white entries on the main diagonal are zero because each stock–day is compared with itself. The of-diagonal entries within each 6 × 6 stock block are consistently dark, showing that MIND assigns low distances to diferent days from the same instrument. In contrast, the cross-stock blocks are substantially lighter, revealing a clear separation between instruments. A small number of dates depart from this pattern and show larger distances from nearby days, particularly around the Christmas period.

## 4.2 Sensitivity Analysis

We test whether MIND responds to small, gradual corruptions by applying perturbations of increasing strength � to the historical

![](images/d9dfe3cd621ff5158633d5afb696fe797a3e6db53ac3495cab66e6571cdf2fc6.jpg)  
Figure 1: Pairwise MIND distances between all stock-days in the test period (five stocks, six dates each). Each 6×6 diagonal block compares diferent days of the same stock, while the of-diagonal blocks compare diferent stocks. Darker colours indicate smaller distances; the white main diagonal marks each stock-day compared with itself, where MIND is zero.

LOB data. Each input window is $\boldsymbol { X } \in \mathbb { R } ^ { T \times 4 0 }$ , where $T = 1 0 0$ and the features contain the �-scored prices and volumes at the ten best bid and ask levels. We modify the price path directly as well as the temporal structure of the window (through deletions).

Let $\mathcal { P }$ denote the set of 20 price columns. For each price perturbation, we apply the same time-dependent shift $\delta _ { t }$ to every price level,

$$
\widetilde { X } _ { t , j } = X _ { t , j } + \sigma _ { p } \delta _ { t } , \qquad j \in \mathcal { P } ,\tag{4}
$$

where $\sigma _ { p }$ is the mean standard deviation of the price features. The volume columns are unchanged. We draw $s \sim \mathrm { U n i f } \{ - 1 , + 1 \}$ independently for each window so that the perturbations are approximately zero-mean across the dataset.

For $t \in \{ 0 , . . . , T - 1 \}$ , the three price shifts are

$$
\delta _ { t } ^ { \mathrm { t r e n d } } = \varepsilon s \left( \frac { 2 t } { T - 1 } - 1 \right) ,\tag{5}
$$

$$
\delta _ { t } ^ { \mathrm { w a l k } } = \varepsilon \widehat { W } _ { t } ,\tag{6}
$$

$$
\delta _ { t } ^ { \mathrm { j u m p } } = \varepsilon s 1 \{ t \geq \tau \} , \qquad \tau \sim \mathrm { U n i f } [ T / 4 , 3 T / 4 ) .\tag{7}
$$

The linear trend creates a total upward or downward movement of $2 \varepsilon \sigma _ { p }$ across the window. For the random-walk perturbation, $\widehat { W }$ is a Gaussian random walk standardised to zero mean and unit variance within each window. It introduces smooth stochastic movement with scale $\varepsilon \sigma _ { p } .$ The jump perturbation introduces a single step change at a randomly selected point in the middle half of the window.

The fourth perturbation, row deletion, removes each row independently with probability �. The surviving rows retain their order and are shifted to the beginning of the window, while the remaining tail is filled by repeating the final surviving row. Successive retained rows are therefore separated by $1 / ( 1 - \varepsilon )$ original events on average. Thus, � acts on the event index rather than on feature values, and we evaluate this perturbation over a smaller range.

![](images/04e58f24afb7c13e3371881998f3e5f25bb5b3acf7522a1532c30f56435a6b28.jpg)  
Figure 2: MIND as a function of perturbation strength � for the four perturbation mechanisms: a) linear trend, b) random walk, c) price jump, and d) row deletion. Top: example mid-price paths for a single stock-day (0700.HK, 22 December 2025) at increasing �, with the historical unperturbed path in dark green. Bottom: mean MIND (log scale) across stock-days, with shaded bands showing ±1 standard deviation.

Figure 2 shows the mean MIND score for the four perturbation mechanisms, with shaded bands representing one standard deviation across stock-days. For all four perturbations, MIND is zero when $\varepsilon = 0$ and rises steadily as the distortion becomes stronger, with no reversals. The perturbations are deliberately kept subtle to test whether MIND can detect small deviations from the real data. Accordingly, the MIND score increases only moderately for the trend and jump perturbations. The random-walk perturbation introduces more persistent noise throughout each price level and therefore produces a larger increase in MIND. Row deletion has a similarly strong efect because removing events disrupts the temporal structure and spacing of the original order-book sequence.

## 4.3 Adversarial Attacks

4.3.1 Moment-Matching Atack on FID. FID compares embedding distributions using only their means and covariance matrices [10]. It therefore cannot distinguish distributions that share these moments but difer in shape. Motivated by Berthet et al. [3], we construct an attack in the LOB input space that maximises the diference from the real embedding distribution while keeping its mean and covariance close to the real values.

We collect up to 20,000 real embeddings from the five stocks over the test period and, after shufling with a fixed random seed, divide them equally into a target pool $X _ { A }$ and an evaluation pool $X _ { B } .$ . The target mean � and covariance Σ are estimated from the full pool $X _ { A }$

The attack starts from $n = 1 0 2 4$ real LOB windows and treats the input tensor $\boldsymbol { a } \in \mathbb { R } ^ { n \times T \times 4 0 }$ as the optimisation variable while keeping the DeepLOB network fixed. Writing $E = { \mathrm { e m b e d } } ( a )$ , we maximise the shape loss $\mathcal { L } _ { \mathrm { s h a p e } } ( a )$ subject to $\mathcal { L } _ { \mathrm { m o m } } ( a ) \leq \tau$ . The shape loss averages the squared 2-Wasserstein distances between one-dimensional projections of � and a fixed �-sample subset of $X _ { B }$ , using $K = 1$ 28 random directions redrawn at each optimisation step. The moment loss is

$$
\mathcal { L } _ { \mathrm { m o m } } ( a ) = \left\| \bar { E } - \mu \right\| _ { 2 } ^ { 2 } + \frac { 1 } { d } \left\| \Sigma _ { E } - \Sigma \right\| _ { F } ^ { 2 } , \qquad d = 9 6 ,\tag{8}
$$

where �<sup>¯</sup> and $\Sigma _ { E }$ are the mean and covariance of the attacked embeddings, and $\mu$ and Σ are the corresponding target moments estimated from $X _ { A }$ . The threshold � is the moment discrepancy of a genuine real batch of size �, so a feasible attack matches the real moments within the sampling variation of a genuine batch.

We optimise in alternating distortion and polishing phases. The distortion phase increases the shape loss using Adam with learning rate $5 \times 1 0 ^ { - 2 }$ , while an adaptive penalty keeps the moment loss below 2�. The polishing phase minimises only the moment loss, starting with a learning rate of $2 . 5 \times 1 0 ^ { - 2 }$ and reducing it on plateaus to $1 0 ^ { - 4 }$ . Polishing stops when the loss falls below 0.2�. We run three cycles and retain the feasible result with the largest MIND score.

Since empirical FID and MIND depend on sample size, all final comparisons use matched batches of $n = 1 0 2 4$ embeddings. Let $X _ { A } ^ { ( n ) } \subset X _ { A }$ and $X _ { B } ^ { ( n ) } \subset X _ { B }$ denote the real batches used to compute the real–real noise floor. For each metric �, we report

$$
R _ { m } = \frac { d _ { m } \Big ( E , X _ { B } ^ { ( n ) } \Big ) } { d _ { m } \Big ( X _ { A } ^ { ( n ) } , X _ { B } ^ { ( n ) } \Big ) } .\tag{9}
$$

Here, $E , X _ { A } ^ { ( n ) }$ , and $X _ { B } ^ { ( n ) }$ each contain 1024 embeddings. A ratio close to one means that the attacked samples appear as similar to the real data as two genuine real batches, while a larger value indicates that the attack remains detectable.

The optimisation converges with a moment loss of 0.0033, satisfying $\mathcal { L } _ { \mathrm { m o m } } \leq \tau$ . FID increases from a real–real noise floor of

![](images/aa0af3a77baf9d068f07c64e60fd9ad893402168c90777f5787d89b7d5a1a6c2.jpg)

![](images/0eefb2bc5e7ad3e6c931e60a01b0a1b4924692fe9654e3bb6394a6e7d4c669d5.jpg)  
Figure 3: Efect of inception-network training on LOB-ID. Left: DeepLOB training loss (log scale) over training steps, with five evaluation checkpoints C1–C5 at diferent stages of training. Right: MIND (log scale) between historical data and synthetic data from each one of five generative models, computed at the same five checkpoints; error bars show ±1 standard error of the mean.

0.17 to 0.59, giving $R _ { \mathrm { F I D } } = 3 . 5$ . MIND increases from 0.54 to 40.3, yielding $R _ { \mathrm { M I N D } } = 7 4 . 5 .$ , approximately 21 times the FID ratio. Thus, FID does not miss the attack entirely but substantially understates the remaining distributional distortion, whereas MIND places the attacked data far outside the real–real baseline.

4.3.2 Joint-Structure Atack against Statistic-Based LOB Evaluation. LOB-Bench evaluates synthetic LOB data using distributions of selected market statistics, price-impact responses computed from the mid-price, and a trained discriminator [15]. Here, we focus on the first two components. Marginal distributional comparisons do not verify that deep-book liquidity evolves coherently over time, with the Level-1 state, with the opposite side of the book, or with the prevailing market regime. We therefore construct adversarial LOB data that preserve the evaluated marginals and impact response while deliberately breaking these dependencies. We omit the discriminator because it serves a diferent evaluation objective: measuring the empirical separability of real and generated sequences. This experiment instead isolates whether the evaluated statistics and impact response detect the perturbation. We leave discriminator-based evaluation of the attack for future work.

We construct the attack for each real trading day. Prices, timestamps, message features, and quantities at level 1 are left unchanged, while the size and order count columns at levels 2 to 10 are rearranged. For each side ofthe book, we divide the data into contiguous segments separated by price changes in the deep levels and permute only segments of equal length. Reordering the deep-book values does not change their marginal distributions. Most LOB-Bench statistics use only Level-1 data, which the attack leaves unchanged, so their scores are unafected. For every stock-day we report each metric as its percentage above that day’s real–real noise floor, obtained by splitting the day into even and odd row blocks: −100% means the metric sees no diference at all, 0% is the sampling floor, and positive values mean detection.

We find that the distributional statistics and impact-response components from LOB-Bench do not detect it. Averaged over its 12 distributional statistics that do not require message-level data, the attack sits 95% ± 1% below the noise floor across the 18 stock-days. This follows from the construction. Indeed, 9 of the 12 statistics are preserved exactly, whereas the three that shift (total bid volume, total ask volume, and volume per minute) stay below their real–real baselines, and the impact response function is untouched. Every evaluated LOB-Bench statistic rates the adversarial sequence at least as favourably as a second genuine sample of the same day. However, we find that LOB-ID, using MIND, is able to detect the attack. Averaged over the same 18 stock-days, the attack sits 75% ± 17% above the noise floor, and the mean is positive for every stock: +40% for 9999.HK, +144% for 1024.HK, and +42% for 0700.HK. LOB-ID consistently places the attack above genuine sampling variation.

## 4.4 Impact of Inception Network Training

We next examine how the training stage of the DeepLOB Inception network afects the stability and consistency of the MIND score when applied to diferent LOB generative models. We select five checkpoints at progressively later stages of training and, for each, compute LOB-ID between the historical data and synthetic data generated by five fixed models (ZI, CH, LOBGAN, LOBS5 and DifLOB with null and future conditioning). Since the generators are held constant throughout, any change in LOB-ID across checkpoints reflects the evolving embedding network rather than a change in the generated data.

As shown in Figure 3, as DeepLOB training approaches convergence, the MIND scores also converge towards stable plateaus across all generator configurations. Most of the reduction occurs at the earlier checkpoints, while the changes between approximately 40,000 and 50,000 training steps are small. This indicates that, once the DeepLOB representation is suficiently trained, the resulting MIND scores become less sensitive to the particular checkpoint used to construct the embedding space.

## 4.5 Comparison of Generative Models

We compare five generative models using stylised facts, LOB-Bench distances, and LOB-ID, evaluating DifLOB under both null and future conditioning. Table 1 reports the mean ± one standard deviation across three stocks and six test days. The variation therefore reflects diferences across stocks and dates. We separate futureconditioned DifLOB from the other generators in Table 1 because it is an oracle: the trend, volatility, liquidity, and order-flow im balance variables it conditions on are computed from the realised future trajectory, so it is scored on a strictly easier task. We therefore report it as an upper reference point rather than as a competitor, and rank the remaining models among themselves. Future-conditioned DifLOB attains the best values on all three distance measures, with low variability across stocks and dates, and reproduces $7 . 2 2 \pm 1 . 0 0$ stylised facts. Historical data scores $7 . 6 7 \pm 0 . 5 9$ under the same evaluation.

Table 1: Comparison of synthetic LOB generators. Values are means ± one standard deviation across 18 stock-days; lower is better except for stylised facts. Future-conditioned DifLOB<sup>†</sup> uses realised future regime variables and is excluded from the non-oracle ranking. Best and second-best non-oracle results are shown in bold and underlined
<table><tr><td>Model</td><td></td><td>Stylised Facts (of 11) LOB-Bench (Wasserstein-1)</td><td> $\mathrm { L O B - B e n c h } \left( L _ { 1 } \right)$ </td><td> $\mathrm { L O B - I D } \left( \mathrm { M I N D } \right)$ </td></tr><tr><td>Zero Intelligence</td><td> $4 . 5 6 \pm 1 . 8 2$ </td><td> $1 . 5 3 1 5 \pm 0 . 3 6 7 9$ </td><td> $0 . 6 1 5 8 \pm 0 . 0 2 3 5$ </td><td> $2 6 . 5 5 1 4 \pm 1 1 . 3 7 8 5$ </td></tr><tr><td>Compound Hawkes</td><td> $6 . 0 6 \pm 0 . 8 7$ </td><td> $2 . 0 4 9 8 \pm 0 . 5 2 7 8$ </td><td> $0 . 5 1 7 0 \pm 0 . 0 4 3 3$ </td><td> $2 4 . 4 2 5 4 \pm 9 . 8 4 6 4$ </td></tr><tr><td>LOBGAN</td><td> $5 . 6 7 \pm 1 . 2 8$ </td><td> $1 . 1 0 3 8 \pm 0 . 2 4 8 4$ </td><td> $0 . 5 6 4 2 \pm 0 . 0 9 0 3$ </td><td> $1 7 . 9 7 1 1 \pm 8 . 6 2 5 0$ </td></tr><tr><td>LOBS5</td><td> $5 . 2 8 \pm 1 . 7 8$ </td><td> $0 . 7 5 1 5 \pm 0 . 4 1 6 2$ </td><td> $\underline { { 0 . 4 1 9 5 } } \pm 0 . 0 4 5 0$ </td><td> ${ \underline { { 9 . 1 2 0 1 } } } \pm 4 . 2 6 7 3$ </td></tr><tr><td>DiffLOB (null cond)</td><td> $7 . 3 3 \pm 0 . 6 9$ </td><td> $\underline { { 0 . 9 1 6 8 \pm 0 . 7 1 6 7 } }$ </td><td> $\mathbf { 0 . 2 7 3 5 \pm 0 . 0 7 7 6 }$ </td><td> $6 . 0 2 5 1 \pm 3 . 7 4 1 7$ </td></tr><tr><td>DiffLOB (future cond)†</td><td> $7 . 2 2 \pm 1 . 0 0$ </td><td> $0 . 2 1 2 4 \pm 0 . 0 6 8 3$ </td><td> $0 . 1 5 5 8 \pm 0 . 0 2 3 8$ </td><td> $2 . 8 2 0 4 \pm 0 . 9 4 2 4$ </td></tr></table>

Our results show that LOB-ID provides a stable measure of similarity between real and synthetic LOB data. The pairwise comparisons show that the trained embeddings capture persistent diferences between instruments. Samples from the same stock remain close across nearby dates, while diferent stocks are clearly separated. The unusually high distances for a small number of dates may reflect changes in liquidity, volatility, or trading activity. We do not investigate the cause here, but the result suggests that LOB-ID could also be used to study changes in market conditions, unusual trading days, or possible regime shifts.

Among the five non-oracle generators, null-conditioned DiffLOB is strongest overall. It reproduces the most stylised facts, with $7 . 3 3 \pm 0 . 6 9$ , and records the lowest $L _ { 1 }$ and LOB-ID distances, ranking second only on Wasserstein-1. LOBS5 follows, with the lowest Wasserstein-1 distance and second place under both $L _ { 1 }$ and LOB-ID. The separation is clearest under LOB-ID, which orders the models as DifLOB (null), LOBS5, LOBGAN, CH and ZI. Among LOBGAN, CH and ZI, LOBGAN performs best under both LOB-ID and Wasserstein-1. The metrics nevertheless produce diferent rankings, where CH reproduces $6 . 0 6 \pm 0 . 8 7$ stylised facts, more than the other non-DifLOB models, but records the highest Wasserstein-1 distance, while LOBS5 reproduces fewer stylised facts while performing strongly on all three distance measures.

The perturbation experiments support the same conclusion. The changes are deliberately small and remain within the range of normal market variation, yet LOB-ID responds before the altered price paths become visibly unrealistic. Its consistent increase across several diferent perturbations shows that the metric is sensitive to a broad range of structural changes, not only to one specific distortion or to changes that are obvious in the raw data.

The checkpoint experiment indicates that, although the absolute MIND scores depend on the stage of DeepLOB training, the relative ordering of the generator configurations remains unchanged across checkpoints. Once the embedding network is suficiently

## 5 Discussion

trained, the scores stabilise and become less sensitive to checkpoint selection.

The two adversarial experiments expose diferent weaknesses in FID and the evaluated statistic-based components of LOB-Bench. The first moment-matching attack targets FID directly. Because FID depends only on the embedding mean and covariance, it substantially understates the distortion once these moments are matched. MIND still detects a large diference because it compares the projected distributions rather than summarising the embeddings only by their means and covariances.

The joint-structure attack shows a similar limitation in statisticbased benchmarks. The attacked sequence preserves most of the measured marginal distributions and leaves the market-impact response unchanged, but it no longer forms a coherent market trajectory. The attack breaks consistency between the deep-book trajectory and the corresponding order flow, market conditions, and Level-1 state. Although LOB-ID does not observe the message stream directly, it detects the resulting distortion in the temporal, cross-level, and bid–ask structure of the Level-2 book trajectory. Independently rearranging the bid and ask sides also breaks their dependence, while the segment boundaries disrupt continuity in the deep book.

The attacked data therefore appear realistic when individual features are examined separately, but not when the orderbook is viewed as a joint time series. This supports using the two approaches together. Statistic-based benchmarks provide clear and interpretable diagnostics, while LOB-ID captures temporal and cross-level relationships that individual statistics may miss. Beyond complementing statistic-based diagnostics, LOB-ID provides a reusable distance-based alternative to the LOB-Bench discriminator. The discriminator assigns each trajectory a probability of being real, with model quality inferred from how well its outputs separate real and generated samples. In contrast, LOB-ID directly computes a non-negative discrepancy between the real and generated embedding distributions, without requiring a classification threshold. Once the DeepLOB feature extractor is fixed, the same representation and scoring rule can be applied across generators without training an additional real-versus-generated classifier for each comparison.

Finally, LOB-ID produces an ordering consistent with the dependencies represented by the tested models. These mechanisms are not observed directly, but they leave signatures in the temporal and cross-level structure of the book trajectories, which is what LOB-ID measures. Zero Intelligence samples side, type, timing, volume, and depth independently, so no relationship between order flow and book state can arise, and it scores worst. Compound Hawkes lets past events raise the intensity of future arrivals, which introduces dependence in event timing, but volumes and depths remain independently sampled, so the book state is still not explained by its order flow. LOBGAN improves on both because its generator conditions each new order on the current market state, creating genuine flow to book dependence, but that state enters only as a small set of hand-crafted order and book features, so the dependence it can learn is limited to what this summary exposes. LOBS5 improves further because it models the raw message stream directly. Its S5 backbone, a structured state space model, maintains a compressed continuous state of a long message context. DifLOB models the full trajectory jointly across time and price levels, allowing cross-level and bid–ask dependencies to be represented directly rather than carried implicitly through sequential generation. When the other metrics produce a diferent ordering, the disagreement reflects diferences in the properties emphasised by each metric. For CH, its large Wasserstein distance may reflect the more extreme timing distributions produced by self-excitation. Future-conditioned DifLOB receives regime variables computed from the future trajectory, giving it information unavailable to the other generators considered. Its low distances therefore measure regime reproduction rather than unconditional generative quality. However, null-conditioned DifLOB receives no future information. LOB-ID clearly distinguishes these settings, demonstrating that LOB-ID captures both the benefit of regime information and the underlying quality of the generator.

LOB-ID depends on the choice of embedding network. Its absolute scores are therefore comparable only when the same representation, preprocessing, sample size, and projection protocol are used. As the current embedding is constructed from Level-2 book states, it evaluates temporal and cross-level trajectory structure rather than the full message-level process. Our experiments are limited to five liquid HKEX equities, so validation across markets, instruments, and regimes remains future work, although the stable ranking across checkpoints suggests limited sensitivity to check point selection within the chosen representation. A direct empirical comparison between inception distances and trained discriminator scores remains an important direction for future work.

## 6 Conclusion

We introduced LOB-ID, an embedding-based framework for evaluating synthetic limit orderbook data. By comparing real and generated limit order-book data in the representation space learned by the DeepLOB Inception network, LOB-ID provides a summary of temporal and cross-level similarity between Level-2 order-book trajectories. Our experiments demonstrate that LOB-ID is consistent across instruments and trading days, detects subtle perturbations of the data, and yields stable model rankings. Moment matching substantially reduces FID’s sensitivity, while a deep-book attack pre serves the selected LOB-Bench statistics. MIND remains sensitive to both distortions. These results highlight LOB-ID as an efective method for measuring joint temporal and cross-level structure in Level-2 trajectories.

## References

[1] Samuel A. Assefa, Danial Dervovic, Mahmoud Mahfouz, Robert E. Tillman, Prashant Reddy, and Manuela Veloso. 2020. Generating Synthetic Data in Finance: Opportunities, Challenges and Pitfalls. In Proceedings ofthe First ACM International Conference on AI in Finance (ICAIF ’20). 1–8.

[2] Emmanuel Bacry, Iacopo Mastromatteo, and Jean-François Muzy. 2015. Hawkes Processes in Finance. Market Microstructure and Liquidity 1, 1 (2015), 1550005.

[3] Quentin Berthet, Yu-Han Wu, Clement Crepy, Romuald Elie, Klaus Gref, and Michael Eli Sander. 2026. MIND: Monge Inception Distance for Generative Models Evaluation. arXiv:2605.06797

[4] Min Jin Chong and David Forsyth. 2020. Efectively Unbiased FID and Inception Score and Where to Find Them. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6070–6079.

[5] Andrea Coletta, Joseph Jerome, Rahul Savani, and Svitlana Vyetrenko. 2023. Conditional Generators for Limit Order Book Environments: Explainability, Challenges, and Robustness. In Proceedings of the Fourth ACM International Conference on AI in Finance (ICAIF ’23). Association for Computing Machinery, New York, NY, USA, 27–35.

[6] Andrea Coletta, Matteo Prata, Michele Conti, Emanuele Mercanti, Novella Bartolini, Aymeric Moulin, Svitlana Vyetrenko, and Tucker Balch. 2021. Towards Realistic Market Simulations: A Generative Adversarial Networks Approach. In Proceedings of the Second ACM International Conference on AI in Finance (ICAIF ’21). Association for Computing Machinery, New York, NY, USA, 1–9.

[7] Rama Cont. 2001. Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues. Quantitative Finance 1, 2 (2001), 223–236.

[8] J. Doyne Farmer, Paolo Patelli, and Ilija I. Zovko. 2005. The Predictive Power of Zero Intelligence in Financial Markets. Proceedings of the National Academy of Sciences 102, 6 (2005), 2254–2259.

[9] Martin D. Gould, Mason A. Porter, Stacy Williams, Mark McDonald, Daniel J. Fenn, and Sam D. Howison. 2013. Limit Order Books. Quantitative Finance 13, 11 (2013), 1709–1742.

[10] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In Advances in Neural Information Processing Systems, Vol. 30. 6626–6637.

[11] Konark Jain, Nick Firoozye, Jonathan Kochems, and Philip Treleaven. 2024. Limit Order Book Dynamics and Order Size Modelling Using Compound Hawkes Process. Finance Research Letters 69 (2024), 106157.

[12] Maxime Kawawa-Beaudan, Srijan Sood, Kassiani Papasotiriou, Daniel Borrajo, and Manuela Veloso. 2026. TradeFM: A Generative Foundation Model for Tradeflow and Market Microstructure. arXiv:2602.23784 [cs.LG]

[13] Fabrizio Lillo and J. Doyne Farmer. 2004. The Long Memory of the Eficient Market. Studies in Nonlinear Dynamics & Econometrics 8, 3 (2004).

[14] Peer Nagy, Sascha Frey, Silvia Sapora, Kang Li, Anisoara Calinescu, Stefan Zohren, and Jakob N. Foerster. 2023. Generative AI for End-to-End Limit Order Book Modelling: A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network. In Proceedings ofthe Fourth ACM International Conference on AI in Finance (ICAIF ’23). Association for Computing Machinery, New York, NY, USA, 91–99.

[15] Peer Nagy, Sascha Yves Frey, Kang Li, Bidipta Sarkar, Svitlana Vyetrenko, Stefan Zohren, Ani Calinescu, and Jakob Nicolaus Foerster. 2025. LOB-Bench: Benchmarking Generative AI for Finance – an Application to Limit Order Book Data. In Proceedings ofthe 42nd International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 267). PMLR, 45437–45460.

[16] Yosihiko Ogata. 1981. On Lewis’ Simulation Method for Point Processes. IEEE Transactions on Information Theory 27, 1 (1981), 23–31.

[17] Namid R Stillman, Rory Baggott, Justin Lyon, Jianfei Zhang, Dingqui Zhu, Tao Chen, and Perukrishnen Vytelingum. 2023. Deep calibration of market simulations using neural density estimators and embedding networks. In Proceedings of the Fourth ACM International Conference on AI in Finance. 46–54.

[18] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. 2015. Going Deeper with Convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 1–9.

[19] Zhuohan Wang and Carmine Ventre. 2026. DifLOB: Difusion Models for Counterfactual Generation in Limit Order Books. In Proceedings of the 35th International Joint Conference on Artificial Intelligence (IJCAI-ECAI 2026). arXiv:2602.03776 [q-fin.CP] Accepted, in press.

[20] Zihao Zhang, Stefan Zohren, and Stephen Roberts. 2019. DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. IEEE Transactions on Signal Processing 67, 11 (2019), 3001–3012.
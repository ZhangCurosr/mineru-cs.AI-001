# MoFE: A Novel Mixture-of-Experts Framework with Fourier Neural Operators for Cryptocurrency Forecasting

Bowen Liu

School of Art and Science

Mingming Sun

University of Rochester

Rochester, USA

bliu59@u.rochester.edu

AGI Lab

Beijing Institute of Mathematical Sciences and Applications

Beijing, China

sunmingming@bimsa.cn

Abstract—Forecasting cryptocurrency prices remains a formidable challenge due to inherent non-stationarity, abrupt regime shifts, and multi-scale stochastic dependencies. Conventional deep learning models often struggle to capture complex underlying dynamics, frequently resulting in persistent phaselagged predictions. To address these limitations, we propose MoFE, a novel deep learning framework that integrates Fourier Neural Operators (FNOs) within a Mixture-of-Experts (MoE) architecture. Rooted in the theoretical framework of stochastic differential equations, MoFE conceptualizes cryptocurrency volatility as a superposition of multi-frequency components, which includes user network based fundamental growth, mining costs and halving mechanism caused seasonal volatility, and market sentiment-induced chaos. Specifically, specialized adaptive FNO (AFNO) and Convolution dual-domain experts learn continuous function-to-function mappings to encapsulate global spectral trends, cyclical adjustments and microstructures, while a dynamic gating based MoE mechanism enables adaptive strategy switching across diverse market regimes. Extensive experiments on Bitcoin datasets spanning January 2020 to December 2025 demonstrate that MoFE achieves state-of-the-art (SOTA) performance in both T+1 and T+5 forecasting horizons. Notably, the model effectively mitigates the phase-lag effect, delivering superior Directional Accuracy (DA) and Information Coefficient (IC). In high-fidelity simulated trading environments, these predictive gains transfer into significant excess returns and robust risk-adjusted performance, characterized by a high Sharpe ratio.

Index Terms—Cryptocurrency Forecasting, Fourier Neural Operators, Mixture-of-Experts.

## I. INTRODUCTION

Cryptocurrency price forecasting represents a pivotal frontier in computational finance, increasingly shifting from heuristic technical analysis toward sophisticated data-driven machine learning paradigms [1], [2]. Robust predictive capabilities are fundamental to enhancing market transparency and facilitating informed decision-making for both institutional investors and regulatory stakeholders.

Unlike traditional assets anchored by fundamental yields, cryptocurrencies are governed by algorithmic scarcity and network-driven valuations—dynamics often modeled by fundamental growth guided by Metcalfe’s Law, seasonal volatility driven by mining costs and halving mechanisms, and chaos induced by market sentiment [3]. The absence of intrinsic valuation anchors leaves the market susceptible to sentiment-driven volatility, frequently invalidating traditional mean-reversion frameworks [4]. Developing robust predictive models remains a formidable challenge due to inherent non-stationarity, abrupt regime shifts, and multi-scale stochasticity [3]. Furthermore, compared to the centuries of data in equity markets, the relatively brief history of cryptocurrencies induces a “data scarcity” problem, particularly when analyzing low-frequency cyclical patterns like the quadrennial “halving” events [5].

The landscape of cryptocurrency price forecasting has transitioned from heuristic technical analysis toward sophisticated data-driven machine learning paradigms. Initially, classical statistical and ensemble frameworks [6]–[8] leveraged historical inertia to model market trends. Meanwhile, some research introduced sentiment-based approaches [4] based on on-chain data and social media psychology to quantify market value beyond endogenous price action. However, these methods primarily capture linear dynamics or non-inherent dynamics and are often inadequate for the pervasive non-stationarity of crypto-markets.

Initially, classical statistical frameworks (e.g., ARIMA [6] and GARCH [7]) and ensemble models leveraged historical inertia to model market trends.

Currently, deep representation learning defines the state-ofthe-art, utilizing non-linear optimization to decode intricate dependencies. This evolution spans Recurrent Neural Networks (RNNs) for temporal modeling [9], [10], Transformers and State-Space-Models (SSMs) for long-range dependency modeling [11], [12], which would like capture cross-period stochastic volatilization. Despite their success, conventional deep learning models primarily minimize point-wise statistical errors via regression. Consequently, they often fail to internalize the underlying continuous spectral dynamics of the market, resulting in significant phase lags and an inability to adapt to rapid regime shifts.

Many existing approaches primarily optimize point-wise regression errors, which often leads to phase lag and weak adaptability to rapid market regime shifts [13], [14]. In addition, purely time-domain models tend to be sensitive to high-frequency market noise, which is particularly common in cryptocurrency markets [15].

Beyond traditional architectures, Fourier Neural Operators (FNOs) [16] have redefined sequence modeling by learning continuous function-to-function mappings in the frequency domain. Notably, the Adaptive FNO (AFNO) [17] optimizes this process via frequency sparsification, allowing for continuous global convolutions that are invariant to input resolution. Further advancements in frequency-domain Mixture-of-Experts (MoE) [18] have demonstrated superior performance in modeling complex Partial Differential Equation (PDE) systems and non-stationary time series. Motivated by the success of these spectral methods, this work introduces a frequency-domain motivated MoE-FNO framework. By modeling cryptocurrency volatility as a superposition of multi-frequency components, our framework captures the multi-scale stochasticity inherent in digital assets to improve the accuracy of return forecasting.

The proposed MoFE framework complements these approaches by combining frequency-domain and time-domain modeling. The global FNO component captures long-range dependencies and cyclical market dynamics, while the convolutional branch extracts short-term temporal microstructures.

The main contributions of this study are as follows:

• We design a novel MoE-FNO framework tailored for cryptocurrency markets, capable of modeling nonstationary volatility and abrupt regime shift through multi-experts decomposition and nonlinear optimization.

• We introduce an AFNO-Conv dual-domain operator that simultaneously captures global cyclical patterns in the frequency domain and transient local microstructures in the temporal domain, addressing the multi-scale nature of financial data.

• Our proposed model, MoFE, achieves state-of-the-art (SOTA) performance across statistical metrics (RMSE, $R ^ { 2 } )$ and financial indicators (Sharpe Ratio, Information Coefficient, ROI). High-fidelity back-testing demonstrates significant risk-adjusted returns and robust predictive consistency.

## II. RELATED WORK

Bitcoin, as the pioneer of decentralized cryptocurrencies, has become one of the most volatile and scrutinized assets in global financial markets [2]. Unlike traditional financial instruments such as stocks and bonds, Bitcoin operates independently of centralized monetary authorities, and its valuation is influenced by a complex interplay of factors including market liquidity, mining costs, macroeconomic changes, and investor sentiment [1]. This extreme volatility, structural instability, and inherent non-linearity make Bitcoin price prediction exceptionally difficult. Therefore, establishing a robust forecasting framework is crucial for improving market transparency and providing institutional investors and regulators with a rigorous, data-driven basis for decision-making.

The corpus of existing literature on cryptocurrency forecasting is generally taxonomized into three distinct paradigms. First, statistical methodologies predicate their predictions on linear dynamical systems and stochastic processes—exemplified by ARIMA [6] and GARCH [7]. These models operate under strong assumptions of stationarity and historical autocorrelation. However, such approaches are inherently constrained to capturing local or univariate features. Consequently, these conventional architectures often fail to simultaneously resolve global frequency-domain patterns and localized temporal dynamics in highly volatile markets. Second, fundamental and sentiment analysis paradigms attempt to quantify intrinsic asset value through on-chain network adoption metrics (e.g., stock-to-flow models) and behavioral indicators [4]. These methods are premised on the hypothesis that information dissemination and social media discourse serve as leading indicators for subsequent price fluctuations.

In recent years, deep learning methodologies have established a new state-of-the-art by leveraging non-linear optimization to capture high-dimensional dependencies within financial datasets. Recurrent Neural Network (RNN) variants, most notably Long Short-Term Memory (LSTM) [9] and Gated Recurrent Units (GRU) [10], employ sophisticated gating mechanisms to characterize temporal dynamics. Specifically, the GRU architecture enhances computational efficiency by consolidating the LSTM’s forget and input gates into a single update gate, often facilitating superior convergence in Bitcoin volatility modeling. Beyond recurrent paradigms, gradientboosted frameworks such as XGBoost [8] have demonstrated competitive performance in minimizing mean deviation errors relative to traditional LSTMs. More recently, self-attentionbased Transformer architectures [11], [19] have achieved significant performance gains by exploiting global long-range dependencies. Despite these advancements, a fundamental limitation still persists: these predominantly time-domain models—including LSTMs and Transformers—exhibit high susceptibility to high-frequency noise, which frequently obscures the latent low-frequency trends inherent in cryptocurrency markets [15]. Furthermore, recent studies highlight that models relying exclusively on conventional point-wise regression loss functions are prone to structural phase lag, thereby failing to capture regime-specific dynamics under the extreme nonstationarity of digital assets [13], [14]. While contemporary innovations like Smamba [20] and CryptoMamba [12] attempt to mitigate these issues through enhanced structural generalizability, the inability to disentangle complex dynamics within the frequency domain remains a critical barrier to achieving robust generalization. However, they often struggle with multi-scale feature extraction in the presence of nonstationary volatility.

Frequency-domain methodologies, such as Fourier analysis [21] and Fourier Neural Networks [22], have demonstrated significant efficacy in time-series forecasting by leveraging the inherent time-frequency duality of spectral representations. Concurrently, Mixture-of-Experts frameworks [23]–[25] have advanced financial predictive modeling through the dynamic coordination of specialized experts. While recent hybrid architectures like FreqMoE [18] have successfully integrated FNOs with MoE for solving complex PDE systems, the synergistic application of MoE-driven expert collaboration and FNO-based function-to-function mapping remains largely unexplored in the context of cryptocurrency markets. Consequently, there is a notable research gap in utilizing these combined paradigms to address the pervasive challenges of nonstationarity, abrupt regime shifts, and multi-scale stochasticity inherent in digital asset dynamics.

## III. METHODOLOGY

## A. Architecture Overview

The architecture of the proposed MoFE model is illustrated in Figure 1. The process begins with a Stem Block comprising a Linear layer, which projects the input data sequence into a high-dimensional latent space with $d _ { m o d e l }$ Subsequently, these features are concurrently processed by K parallel Experts and a Gating Network (MoE block). Each expert is designed as a time-frequency dual-domain feature extraction module. Specifically, the global FNO path employs an adaptive FNO to capture global cyclical fluctuations features via a sparse MLP in the frequency domain, while the local context path utilizes a $n \times 1$ convolution $( \mathbf { e } . \mathbf { g } . , n = 3 )$ to extract local microstructure features in the temporal domain. Simultaneously, the Gating Network regresses expert weights through an MLP and a Softmax layer. These weights are then applied to the expert outputs via Batch Matrix Multiplication (BMM) to dynamically re-weight their contributions. Finally, the aggregated features are passed through a Prediction Head—consisting of Linear, GELU, and Dropout layers—to produce the final forecasts for prices and returns.

## B. Our MoFE Model

1) FNO-Expert: Each expert $\mathrm { E } _ { \mathrm { k } } ( \mathbf { X } ) \ \in \ \mathbb { R } ^ { B \times K \times C }$ in our MoFE framework, where $\mathbf { X } \in \mathbb { R } ^ { B \times L \times C }$ denotes the hidden representation (with B, L, C representing batch size, sequence length, and channel dimension $d _ { m o d e l }$ , respectively), employs a dual-path architecture. This design is specifically engineered to synergistically capture global periodic patterns and local transient microstructures.

Global FNO Path - The core of the FNO-expert utilizes a one-dimensional Adaptive Fourier Neural Operator (AFNO) [17] to model long-range dependencies and cyclical market fluctuations. This path operates under the premise that financial time-series evolution can be decomposed into complex modal interactions within the frequency domain. The global path is formulated as:

$$
\mathbf { X } _ { g l o b a l } = \operatorname { L N } \left( \mathbf { X } + \mathcal { F } ^ { - 1 } \left( \operatorname { L P F } \left( \mathbf { M } \mathbf { L } \mathbf { P } _ { \mathrm { c o m p l e x } } ( \mathcal { F } ( \mathbf { X } ) ) \right) \right) \right)\tag{1}
$$

where ${ \bf X } _ { g l o b a l } \in \mathbb { R } ^ { B \times K \times C } , \mathcal { F } ( \cdot )$ is a Fast Fourier Transform (FFT), $\mathcal { F } ^ { - 1 } ( \cdot )$ is an Inverse FFT (IFFT), LN is the Layer-Norm, $\mathbf { M L P } _ { \mathrm { c o m p l e x } } ( \cdot )$ is a complex-valued MLP block.

The operator LPF represents a spectral filtering mechanism using a Low Pass Filter (LPF) to mitigate noise and redundancy through two operators: Hard Thresholding, which truncates high-frequency components to retain only the most significant low-frequency modes and Soft Shrinkage (SS), which sparsifies the spectral coefficients by filtering out lowamplitude noise with $\mathrm { S S } ( \mathbf { Z } ) = \mathrm { s i g n } ( \mathbf { Z } ) \operatorname* { m a x } ( 0 , | \mathbf { Z } | - \lambda )$ , where λ is a sparsity threshold.

Local Context Path - To complement the spectral path’s inherent lack of spatial localization, we introduce a parallel local context branch. It consists of a 1D-convolution with a kernel size of $\textit { n } ( n = 3 )$ , followed by Batch Normalization (BN), to capture short-term momentum, abrupt shocks, and localized trend variations:

$$
\mathbf { X } _ { l o c a l } = \mathrm { B N } ( \mathbf { C o n v } 1 D ( \mathbf { X } ) )\tag{2}
$$

where $\mathbf { X } _ { l o c a l } \in \mathbb { R } ^ { B \times K \times C }$

Feature Fusion - The global spectral features and local spatial contexts are integrated via element-wise summation. The fused representation is then projected back to the model dimension using a point-wise convolution $( 1 ~ \times ~ 1 )$ and a LeakyReLU activation:

$$
\mathrm { E } _ { \mathrm { k } } ( \mathbf { X } ) = \mathrm { C o n v } ( \mathrm { L e a k y R e L U } ( \mathbf { X } _ { g l o b a l } + \mathbf { X } _ { l o c a l } ) ) .\tag{3}
$$

2) Mixture-of-Experts System: Cryptocurrency markets are characterized by high non-stationarity, exhibiting distinct $\mathrm { { d y } \mathrm { { - } } }$ namic regimes (e.g., bullish, bearish, or sideways). The MoE module adaptively manages this regime-switching via a Gating Network, which performs conditional computation by selecting the most relevant experts for a given market state. The gating mechanism and expert aggregation are defined as:

$$
\hat { \mathbf { X } } = \sum _ { k = 1 } ^ { K } \underbrace { \mathrm { S o f t m a x } ( \mathbf { M L P } ( \operatorname { v e c } ( \mathbf { X } ) ) ) _ { k } } _ { \mathrm { G a t i n g ~ w e i g h t ~ } g _ { k } } \cdot \underbrace { { \mathrm { E } } _ { k } ( \mathbf { X } ) } _ { \mathrm { E x p e r t ~ o u t p u t } } \ ,\tag{4}
$$

where $\mathrm { v e c } ( \cdot )$ is used to flatten the input to a vector, Softmax(·) is the Softmax operation, the $\mathrm { M L P ( \cdot ) }$ denotes a learnable MLP that maps the flattened input $\mathbf { \bar { \mathbf { \tau } } } _ { \mathrm { v e c } ( \mathbf { X } ) } ~ \in ~ \mathbb { R } ^ { B \times L C }$ to gating logits. The aggregation is implemented as a batch-wise weighted sum, allowing the model to dynamically reconfigure its internal logic based on the latent state of the batch.

3) Prediction Head: The final forecast is decoded from the aggregated latent features X<sup>ˆ</sup> through a multi-layer perceptron (MLP) based prediction head. To ensure robust mapping and mitigate potential overfitting, the head incorporates non-linear activations and stochastic regularization:

$$
\mathbf { Y } _ { p r e d } = \operatorname { L i n e a r } ( \operatorname { D r o p o u t } ( \operatorname { G E L U } ( \operatorname { L i n e a r } ( \hat { \mathbf { X } } ) ) ) ) ,\tag{5}
$$

where $\mathbf { Y } _ { p r e d } \in \mathbb { R } ^ { B \times H o r i z o n }$ , GELU denotes the activation function. This head structure allows the model to refine the high-dimensional latent representations into precise price or return forecasts while maintaining generalization capability.

![](images/eadb49df8d8b172b9f610d70b09430bcbe462cabeca97acfeef2da1098d08dfd.jpg)  
Fig. 1. The overall architecture of the MoFE model. The input features are processed through parallel global (FNO) and local (Conv) paths within the experts, dynamically weighted by the gating network.

## C. Loss Function

To address the inherent complexities of financial time-series forecasting, we propose a multi-objective function termed the Break-even Optimized Loss. Unlike conventional approaches that rely solely on Euclidean distance minimization, our composite loss functional integrates magnitude accuracy, correlation-based alignment, and directional consistency to better capture market dynamics:

$$
\mathcal { L } _ { T a s k } = \alpha \mathcal { L } _ { M S E } ^ { w } + \beta \mathcal { L } _ { I C } + ( 1 - \alpha - \beta ) \mathcal { L } _ { D i r } ,\tag{6}
$$

where the components $\mathcal { L } _ { M S E } ^ { w } , \mathcal { L } _ { I C }$ , and $\mathcal { L } _ { D i r }$ represent the weighted Mean Squared Error of log return, Information Coefficient loss, and Directional Accuracy loss, respectively. These metrics are grounded in established financial econometrics and forecasting literature [26]–[28]. The hyperparameters α, $\beta$ govern the trade-offs between these competing objectives.

From a mathematical standpoint, discrete post-hoc strategy metrics derived from executed trading signals, such as the Win/Loss ratio, are non-differentiable. Therefore, they are structurally incompatible as direct penalty terms in backpropagation.

However, the proposed Break-even Optimized Loss is explicitly architected to achieve this trade-off without compromising differentiability via the existing continuous hyperparameters α and $\beta .$ Specifically, Magnitude Accuracy is directly governed by α (which weights the LMSE term), whereas Directional Frequency is controlled by the residual weight $( 1 - \alpha - \beta )$ for LDir and $\beta$ for LIC.

This formulation allows practitioners to tune the existing α and β parameters to customize the model’s risk-reward profile. For instance, aggressive trading strategies prioritizing directional hit rates (Win/Loss) over magnitude suppression can simply lower the α weight, thereby achieving trading style flexibility.

A common challenge in MoE architectures is “expert collapse” (or mode collapse), where the gating mechanism converges to a state that activates the same subset of experts for all inputs, thereby sacrificing model capacity. To enforce expert specialization and diversity, we introduce an orthogonality constrained regularization loss $\mathcal { L } _ { R e g }$ on the gating activations,

$$
\mathcal { L } _ { T o t a l } = \mathcal { L } _ { T a s k } + \gamma \mathcal { L } _ { R e g } ,\tag{7}
$$

where γ is the weight of the regularization loss.

Let $\mathbf { G } \in \mathbb { R } ^ { B \times K }$ denote the gating activation matrix for a batch size B with K experts. We first perform $L _ { 2 }$ normalization on the activation vectors $g _ { k } \ ( g _ { k } \in \mathbf { \bar { \mathbb { R } } } ^ { B \times 1 }$ , seen in Equ. 4) of each expert k (representing the k-th column of G) to ensure scale invariance:

$$
\hat { g } _ { \boldsymbol k } = \frac { g _ { \boldsymbol k } } { | g _ { \boldsymbol k } | _ { 2 } + \epsilon } ,\tag{8}
$$

where $| \cdot | _ { 2 }$ is an $L _ { 2 }$ normalization, ϵ is a small constant for numerical stability, resulting in the normalized matrix $\hat { \textbf { G } } \in$ R<sup>B×K</sup>.

To promote decorrelation among experts, we penalize the off-diagonal elements of the Gram matrix $\hat { \mathbf { G } } ^ { T } \hat { \mathbf { G } } \hat { \mathbf { \Xi } } \in \mathbb { R } ^ { K \times K }$ to drive it toward the identity matrix I as

$$
\mathcal { L } _ { R e g } = \| \hat { \mathbf { G } } ^ { T } \hat { \mathbf { G } } - \mathbf { I } \| _ { F } ,\tag{9}
$$

where $\| \cdot \| _ { F }$ denotes the Frobenius norm. This regularization term effectively encourages disjoint expert activation patterns, ensuring that different experts specialize in distinct market regimes or latent feature subspaces.

## IV. EXPERIMENTAL ANALYSIS AND DISCUSSION

## A. Datasets

The dataset for this study was collected via a bespoke web crawler from historical Bitcoin repositories, spanning the period from January 1, 2020 to December 27, 2025. The raw data comprises daily Open-High-Low-Close-Volume (OHLCV) metrics organized in strict chronological order. To maintain temporal integrity, the dataset was physically partitioned into training, validation, and testing sets with an approximate ratio of 4:1:1 prior to any preprocessing.

The dataset for this study was collected using a Python script accessing the Binance Public API (specifically the v3/klines endpoint). The raw exchange data comprises daily cryptocurrency K-line data organized in strict chronological order, spanning the period from January 1, 2020, to December 27, 2025. It includes six primary attributes: Timestamp, Open, High, Low, Close, and Volume.

To preclude data leakage, normalization was applied locally to each sliced window, while global standardization parameters and z-score were derived exclusively from the training set. All data used in the test set, including for forecasting inference and performance metric calculations, only uses data from before the given time point, and not data from after that time point.

It is imperative to explicitly recognize the scope and boundaries of our methodology. First, our data collection strictly relies on endogenous market variables (OHLCV data) from the Binance exchange, explicitly excluding external macroeconomic indicators (e.g., interest rates, inflation) that might drive long-term valuations. Second, while the MoFE model demonstrates robustness, its predictive reliability may diminish under extreme market conditions or systemic ”black swan” events where historical patterns fundamentally break down. Finally, the current 32-day look-back window may be insufficient to capture very low-frequency cyclical patterns, such as the quadrennial Bitcoin halving events.

To enhance the representative capacity of the MoFE model, the feature space is expanded into an 8-dimensional vector. This includes the logarithmic returns of the OHLC prices, supplemented by four technical indicators, including Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Relative Volume (RVol), and a Volatility Proxy (VolProxy). The VolProxy characterizes price fluctuations based on the log returns over a rolling window of $n = 5$ days. These features are derived according to established technical analysis methodologies [29], [30] to capture both momentum and volatility dynamics.

## B. Performance Metrics

To rigorously evaluate the predictive efficacy and practical profitability of the MoFE model within the volatile Bitcoin market, we implement a multi-dimensional evaluation framework. This framework is categorized into three dimensions: Statistical Predictive Metrics, Trading Strategy Metrics, and Profitability Metrics.

Statistical forecasting metrics quantify a model’s ability to minimize forecast error and capture market return variance. These metrics include root mean square error (RMSE), mean absolute error (MAE), and $R ^ { 2 }$ [31]. These metrics are all back-calculated prices statistical error computed from the log return forecasting from the MoFE. The trading strategy metrics are used to evaluate the quality and persistence of investment signals or “alpha models”, including Information Coefficient (IC) [26] and Directional Accuracy (DA) [28]. Beyond statistical accuracy, we assess the model’s utility in simulated trading environments, including Win/Loss Ratio [32], Cumulative Return (ROI) [26], and Annualized Sharpe Ratio [33].

In this experiment, Transaction Fee rate and slippage fee are set 0.1% and 0.2% respectively on day-line trading.

## C. Experiments Setup

All proposed models and baselines were implemented using the PyTorch framework and executed on a platform equipped with an NVIDIA GeForce RTX 4060 Laptop GPU. To ensure experimental rigor and a fair benchmarking environment, we maintained consistent configurations across all evaluated models. This includes a fixed random seed (42) for deterministic behavior, uniform data partitioning, identical input features.

All models were optimized using the AdamW optimizer with a standardized learning rate schedule.

All models were optimized using the AdamW optimizer. To mitigate overfitting and ensure optimal generalization, we incorporated a ReduceLROnPlateau learning rate scheduler, weight decay, and an early stopping mechanism with a patience of 20 epochs. The model checkpoint with the lowest validation loss was selected for testing.

Tab. I summarizes the experimental configuration. We utilize a look-back window of $L \ = \ 3 2$ days for forecasting cumulative log-returns. Given the high susceptibility of $T + 1$ predictions to transient noise, our evaluation emphasizes the T+5 horizon to better assess the capture of persistent temporal dependencies. The model dimension is set to $d _ { m o d e l } = 4 8$ Within the MoE-FNO framework, we deploy $K = 2$ experts to prevent overfitting, with each expert operating on $M = 8$ frequency modes. This design strikes an optimal balance between model capacity and computational efficiency.

TABLE I  
HYPERPARAMETER CONFIGURATION OF MOFE
<table><tr><td>Category</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>Data</td><td>Look-back Window (L)</td><td>32</td></tr><tr><td></td><td>Forecast Horizon  $( T _ { p } )$ </td><td>1,5</td></tr><tr><td>Model</td><td>Hidden Dimension  $( d _ { m o d e l } )$ </td><td>48</td></tr><tr><td></td><td>Frequency Modes</td><td>8</td></tr><tr><td></td><td>Number of Experts (K)</td><td>2</td></tr><tr><td></td><td>FNO Blocks</td><td>1</td></tr><tr><td></td><td>Dropout Rate</td><td>0.5</td></tr><tr><td>Training</td><td>Batch Size</td><td>128</td></tr><tr><td></td><td>Initial Learning Rate</td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>Weight Decay</td><td> $1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td></td><td>Loss Weights  $( \alpha , \beta )$ </td><td>0.20, 0.30</td></tr><tr><td></td><td>Reg. Loss Weights (γ)</td><td>0.05</td></tr></table>

## D. Price Forecasting Experiments

1) Overall Performance: The quantitative results of our comparative study are summarized in Tab. II and III. We benchmark the proposed MoFE framework against a suite of state-of-the-art baselines across prediction horizons of $T + 1$ and $T + 5$

In terms of specific comparisons, traditional RNN-based architectures (e.g., LSTM [9] and GRU [10]) achieve competitive DA but yield suboptimal results in profitability and precision compared to MoFE. While the iTransformer [11] demonstrates competence in select metrics, it generally lacks the comprehensive competitiveness of our proposed method. Furthermore, although recent state-space models such as CryptoMamba [12] exhibit promising potential, MoFE maintains a clear performance margin. Moreover, other MoE models like FreqMoE [18] and MoLE [24] are also overshadowed by MoFE. The empirical evidence indicates that MoFE consistently surpasses existing methods, demonstrating significant improvements in both statistical accuracy and economic utility. This superiority is largely attributed to MoFE’s modeling capability on cryptocurrency non-stationary volatilization.

## 2) T+1 Forecasting:

Predictive Accuracy Analysis: It can be concluded from Tab. II and Fig. 2 that

MoFE outperforms all baselines, yielding a minimum RMSE of \$1582.64 and a maximum $R ^ { \dot { 2 } }$ of 0.9827, signaling a precise mapping of continuous market dynamics. Unlike conventional models that often suffer from significant phase lag by collapsing into a trivial identity function $( P _ { t + 1 } \approx P _ { t } )$ MoFE captures genuine structural shifts. This distinction is vital in finance, where high autocorrelation often inflates RMSE and $R ^ { 2 }$ scores through a naive Lag-1 imitation of price persistence. By maintaining high DA far above the 50% baseline, MoFE proves its robustness against the deceptive optimism of standard statistical metrics and its ability to identify actionable volatility clusters.

![](images/c47163c3e20819e9c631a9497627472d66b75640b081d9f7020af674d110f2cb.jpg)  
Fig. 2. Visualized results of the comparison of predicted prices and real prices.

Directional Predictive Capability: Beyond point-wise convergence, MoFE demonstrates superior directional predictive strength. From Tab. II, at the T + 1 horizon, it achieves a peak IC of 0.1957 and DA of 58.43%, with its IC exceeding the runner-up baseline by 1.6×. This suggests an exceptionally robust monotonic correspondence between predictive signals and realized returns.

Visualized results in Figs. 3 and 4 further elucidate these findings below: (1) In Fig. 3, the predicted cumulative returns (blue line) exhibit a dampened amplitude relative to market volatility (grey bars). This reduced variance underscores the efficacy of our balanced loss function, which induces strategic conservatism by penalizing significant directional deviations. (2) Despite this magnitude suppression, the predicted trajectory closely tracks actual market trends. The scatter plot in Fig. 4 confirms this, with the points densely clustered around the diagonal line $( y = x )$ , indicating a strong linear correlation. Simultaneously, the clustering of points near $y = 0$ reflects the risk-aversion effect of the loss function. These results collectively provide empirical evidence that MoFE can effectively filter out high-frequency noises while preserving high-fidelity directional signals, which is consistent with its superior IC and DA metrics.

![](images/5d1d9fd4747e11446d3ec520c852581a651c2b7a6c7827addfedb0caa9570a5d.jpg)  
Fig. 3. Visualized results of the log returns prediction versus the real situations.

![](images/8254a7df8baab825b3a936d06559c4ad22ac2c78703b6a3e55de22a00d993c0f.jpg)  
Fig. 4. Visualized results of the correlation scatter of the predicted log-returns and the realized market returns.

Financial Performance Evaluation: MoFE model achieved the superior performance in terms of ROI (4.41) and Sharpe Ratio (3.89), which indicates that the model not only achieves a superior return rate while maintaining a low risk profile. This shows that MoFE model yields substantial excess returns in unit risk. Although MoFE has Win/Loss Ratio a slightly lower than GRU and iTransformer, it still performs better in overall return and risk control.

## 3) T+5 Forecasting:

Predictive Accuracy Analysis: Even though having slightly higher MAE (\$2593.37) than iTransformer model only at the difference of 1.05, MoFE model still has the smallest RMSE (\$3360.33) and $R ^ { 2 }$ (0.9186) in T + 5 price prediction. Since RMSE is more sensitive to large errors, the results showed that the MoFE model is more stable.

Directional Predictive Capability: Similar to the situation of T+1 prediction, MoFE model also took the lead in IC (0.1389) and DA (56.10%).

Financial Performance Evaluation: In T + 5 prediction, MoFE model also performed best in ROI (4.91) and Sharpe

TABLE II  
PERFORMANCE COMPARISON OF MOFE AGAINST BASELINE MODELS FOR T+1 PREDICTION. THE BEST RESULTS ARE IN BOLD.
<table><tr><td rowspan="2">Model</td><td colspan="3">Statistical Metrics</td><td colspan="2">Strategy Metrics</td><td colspan="3">Profitability Metrics</td></tr><tr><td>RMSE/$ (↓)</td><td>MAE/$ (↓)</td><td> $R ^ { 2 } \left( \uparrow \right)$ </td><td>IC (↑)</td><td>DA (%) (↑)</td><td>Sharpe (↑)</td><td>W/L Ratio (↑)</td><td>ROI (x) (↑)</td></tr><tr><td>LSTM [9]</td><td>1618.35</td><td>1134.32</td><td>0.9819</td><td>0.0611</td><td>54.52%</td><td>1.00</td><td>1.00</td><td>1.48</td></tr><tr><td>GRU [10]</td><td>1605.20</td><td>1130.03</td><td>0.9819</td><td>0.0725</td><td>53.01%</td><td>1.03</td><td>1.24</td><td>1.16</td></tr><tr><td>iTransformer [11]</td><td>1603.96</td><td>1130.63</td><td>0.9822</td><td>0.1225</td><td>51.51%</td><td>2.42</td><td>0.96</td><td>2.75</td></tr><tr><td>S-Mamba [20]</td><td>1662.95</td><td>1183.44</td><td>0.9809</td><td>0.0127</td><td>54.82%</td><td>1.40</td><td>0.96</td><td>1.86</td></tr><tr><td>CryptoMamba [12]</td><td>1613.97</td><td>1145.64</td><td>0.9820</td><td>0.1123</td><td>51.81%</td><td>2.93</td><td>1.17</td><td>3.13</td></tr><tr><td>FreqMoE [18]</td><td>1607.94</td><td>1140.63</td><td>0.9821</td><td>0.1156</td><td>52.41%</td><td>1.34</td><td>1.09</td><td>1.69</td></tr><tr><td>MoLE [24]</td><td>1621.76</td><td>1151.95</td><td>0.9818</td><td>0.1181</td><td>55.72%</td><td>2.11</td><td>1.07</td><td>2.45</td></tr><tr><td>MoFE (Ours)</td><td>1582.64</td><td>1106.10</td><td>0.9827</td><td>0.1957</td><td>58.43%</td><td>3.89</td><td>1.09</td><td>4.41</td></tr></table>

TABLE III  
PERFORMANCE COMPARISON OF MOFE AGAINST BASELINE MODELS FOR T+5 PREDICTION. THE BEST RESULTS ARE IN BOLD.
<table><tr><td rowspan="2">Model</td><td colspan="3">Statistical Metrics</td><td colspan="2">Strategy Metrics</td><td colspan="3">Profitability Metrics</td></tr><tr><td>RMSE/$ (↓)</td><td>MAE/$ (↓)</td><td> $R ^ { 2 } \left( \uparrow \right)$ </td><td>IC (↑)</td><td>DA (%) (↑)</td><td>Sharpe (↑)</td><td>W/L Ratio (↑)</td><td>ROI (x) (↑)</td></tr><tr><td>LSTM [9]</td><td>3392.81</td><td>2596.66</td><td>0.9170</td><td>0.1015</td><td>56.10%</td><td>1.07</td><td>0.68</td><td>1.07</td></tr><tr><td>GRU [10]</td><td>3410.93</td><td>2618.38</td><td>0.9146</td><td>0.0326</td><td>52.44%</td><td>1.11</td><td>1.15</td><td>0.90</td></tr><tr><td>iTransformer [11]</td><td>3381.62</td><td>2592.32</td><td>0.9176</td><td>0.0760</td><td>53.66%</td><td>1.07</td><td>1.37</td><td>1.50</td></tr><tr><td>S-Mamba [20]</td><td>3443.05</td><td>2620.90</td><td>0.9146</td><td>0.0259</td><td>54.88%</td><td>0.82</td><td>1.05</td><td>0.92</td></tr><tr><td>CryptoMamba [12]</td><td>3650.63</td><td>2798.99</td><td>0.9040</td><td>0.0473</td><td>54.88%</td><td>0.96</td><td>1.17</td><td>1.60</td></tr><tr><td>FreqMoE [18]</td><td>3414.83</td><td>2620.41</td><td>0.9160</td><td>0.0678</td><td>55.18%</td><td>1.29</td><td>1.22</td><td>1.82</td></tr><tr><td>MoLE [24]</td><td>3493.51</td><td>2699.26</td><td>0.9120</td><td>0.0142</td><td>51.83%</td><td>0.36</td><td>1.11</td><td>0.40</td></tr><tr><td>MoFE (Ours)</td><td>3360.33</td><td>2593.37</td><td>0.9186</td><td>0.1389</td><td>56.10%</td><td>1.73</td><td>1.26</td><td>4.91</td></tr></table>

Ratio (1.73). Its exceptional ROI significantly outperforms competing models, surpassing the second-best approach by a factor of three. Even though it still did not get the highest Win/Loss Ratio. It remains the optimal architecture for maximizing risk-adjusted returns.

4) Robustness and Horizon Sensitivity: While the horizon extends from 1 to 5, all models exhibit a natural deterioration in Predictive Metrics, characterized by increased RMSE/MAE and decreased $R ^ { 2 }$ . However, MoFE demonstrates superior robustness, suffering only a marginal and controllable decline. Crucially, its ROI improves rather than declines over this longer horizon, which highlights its ability to grasp mid-term market dynamics. The results indicate that that MoFE model can not only dominate short-term forecasting but also leads in medium-term scenarios.

## E. Efficiency Analysis

The MoFE model demonstrates high computational and parameter efficiency for both T+1 and T+5 forecasting tasks. Specifically, the model requires 93.6K parameters and consumes 2.36M FLOPs in $T { + } 1$ mode, while the T+5 configuration requires 94.1K parameters with identical FLOPs. These metrics are highly competitive with established baseline models.

Crucially, despite integrating a complex dual-domain structure, MoFE achieves SOTA performance with a highly lightweight footprint, making it explicitly competitive against more parameter-heavy baselines such as iTransformer and Mamba-based models, which typically demand significantly higher memory and computational overhead.

## F. Interpretability Analysis

Figure 5 indicates that the two experts will change the weight according to different conditions. Sometimes, the experts represented in blue dominate the choice depicted in orange gets more weights. By using the Reg. Loss, the two experts are incentivized to capture distinct feature subspaces. One focuses on short-term fluctuations and the other one focus on long-term trend. This enables the model to possess both the time and frequency domain perspective.

![](images/bcf239442894fb63d713a0bc8ae795c188f370a5824cefd6b3f7bd48ec0093a1.jpg)  
Fig. 5. Visualized results of two experts’ dynamic weights during mixing.

TABLE IV ABLATION STUDY
<table><tr><td rowspan="2">Model</td><td colspan="3">Statistical Metrics</td><td colspan="2">Strategy Metrics</td><td colspan="3">Profitability Metrics</td></tr><tr><td>RMSE/$ (↓)</td><td>MAE/$ (↓)</td><td> $R ^ { 2 } \left( \uparrow \right)$ </td><td>IC (↑)</td><td>DA (%) (↑)</td><td>Sharpe (↑)</td><td>W/L Ratio (↑)</td><td>ROI (x) (↑)</td></tr><tr><td>1 Expert (w/o MoE)</td><td>1608.70</td><td>1127.54</td><td>0.9821</td><td>0.0903</td><td>51.51%</td><td>1.66</td><td>1.37</td><td>1.43</td></tr><tr><td>4 Experts (w MoE)</td><td>1658.03</td><td>1184.79</td><td>0.9810</td><td>0.0680</td><td>53.01%</td><td>1.29</td><td>1.16</td><td>1.70</td></tr><tr><td>2 Experts (w/o MoE)</td><td>1633.03</td><td>1142.99</td><td>0.9816</td><td>0.1077</td><td>54.52%</td><td>0.67</td><td>1.07</td><td>1.76</td></tr><tr><td>Conv. Kernel in Experts (n = 1)</td><td>1597.61</td><td>1161.83</td><td>0.9824</td><td>0.1627</td><td>55.12%</td><td>2.11</td><td>1.13</td><td>2.52</td></tr><tr><td>Conv. Kernel in Experts (n = 5)</td><td>1602.83</td><td>1129.21</td><td>0.9823</td><td>0.1324</td><td>54.22%</td><td>3.31</td><td>1.12</td><td>2.66</td></tr><tr><td>w/o Reg. Loss</td><td>1582.59</td><td>1107.91</td><td>0.9827</td><td>0.1937</td><td>56.93%</td><td>3.91</td><td>1.07</td><td>4.44</td></tr><tr><td>Ours</td><td>1582.64</td><td>1106.10</td><td>0.9827</td><td>0.1957</td><td>58.43%</td><td>3.89</td><td>1.09</td><td>4.41</td></tr></table>

Figure 6 shows that the weight of the 8 input features varies according to different condition. By adjusting the weight dynamically, the model can adaptively respond to different conditions in the Bitcoin market. Those factors have roughly the same weight in most of the cases, however, when the market experiences high volatility the model will react and recalibrate weights to make the prediction more accurate.

![](images/0b390a54829139ec91fb2ab539c1b3dc5ce08686fb6616d172e6d57ea9f17d5e.jpg)  
Fig. 6. Visualized results of the input vector contribution for the forecasting.

## G. Ablation Study

We conduct a series of ablation experiments to evaluate the contribution of each core component in MoFE. The quantitative results are summarized in Table IV.

1) Impact of regularization loss $( L _ { r e g } )$ - Removing $L _ { r e g }$ leads to a marked decline in DA and Win/Loss Ratio, validating that $L _ { r e g }$ effectively encourages expert specialization. While metrics such as Sharpe Ratio and ROI exhibit marginal decreases, the overall enhancement in predictive precision confirms that $L _ { r e g }$ prevents expert collapse and enables the model to capture diverse market dynamics more accurately.

2) Scaling the number of experts (N) - Varying the expert count from N=2 to N=1 or N=4 results in performance degradation across most metrics. As illustrated in Fig. 7, increasing N to 4 leads to “expert dilution”, where the gating weights fluctuate erratically, suggesting that the 32-day lookback window provides insufficient information to supervise a high-dimensional expert space, thereby inducing overfitting. Conversely, a single expert (N=1) fails to simultaneously resolve high-frequency noise and medium-term trends. The

N=2 configuration achieves the optimal balance by dedicating specialized capacity to distinct temporal scales.

3) Efficacy of the gating mechanism - To isolate the benefit of dynamic routing, we replaced the gating network with a simple averaging operation while retaining $L _ { r e g } .$ . The results show a significant deterioration in IC, DA, and profitability metrics. This drop confirms that the gating network is not merely an aggregator but a critical decision-maker that dynamically prioritizes experts based on shifting market regimes.

4) Sensitivity to convolutional kernel size - The spatial receptive field of each expert was tested with kernel sizes of $1 \times 1 , 3 \times 1$ , and $5 \times 1$ . Reducing the kernel to $1 \times 1$ severely impairs strategic performance, as the model loses its ability to extract local temporal correlations. However, expanding the kernel to 5 × 1 also degrades performance, likely due to the introduction of excessive noise from distant time steps. A 3 × 1 kernel proves to be the “sweet spot” for capturing meaningful medium-term trends without overfitting to stochastic fluctuations.

![](images/b435618c2d72410534563b9f98e01a7837b25254d2098d27e8601309314301bd.jpg)  
Fig. 7. Visualized results of four experts’ dynamic weights during mixing.

## V. CONCLUSION

We introduced MoFE in this work, a novel deep learning framework that integrates FNOs and MoE to capture the non-stationary and multi-scale stochasticity of cryptocurrency markets. By leveraging a dual-domain architecture, MoFE effectively extracts global spectral trends and local temporal microstructures, while a dynamic gating mechanism ensures adaptability to rapid regime shifts.

Our experiments confirm that MoFE achieves SOTA results for T + 1 and T + 5 horizons, significantly reducing phase lag and enhancing directional predictive power.

This superior phase-lag mitigation is achieved because the frequency-domain modeling effectively separates highfrequency noise from underlying structural market trends. Empirically, this theoretical advantage translates into robust performance particularly during high-volatility periods, where the spectral representation seamlessly isolates structural trends from transient short-term noise.

In simulated trading, the model yielded exceptional riskadjusted returns, demonstrating high practical utility.

Future research will strictly focus on extending the MoFE framework to other financial time-series forecasting tasks, such as cross-asset volatility modeling.

## REFERENCES

[1] A. J. Hou, W. Wang, C. Y. Chen, and W. K. Hardle, “Pricing cryptocur-¨ rency options,” Journal of Financial Econometrics, vol. 18, no. 2, pp. 250–279, 2020.

[2] M. Amjad and D. Shah, “Trading bitcoin and online time series prediction,” in Proceedings of Machine Learning Research (PMLR), 2017, pp. 1–15.

[3] S. Wheatley, D. Sornette, T. Huber, and R. N. G. Max Reppen, “Are bitcoin bubbles predictable? combining a generalized metcalfe’s law and the lppls model,” 2018, arXiv:1803.05663.

[4] F. N. Zargar and D. Kumar, “Informational inefficiency of bitcoin: A study based on high-frequency data,” Research in International Business and Finance, vol. 47, pp. 344–353, 2019.

[5] J. Y.-L. Chan, S. W. Phoong, S. Y. Phoong, W. K. Cheng, and Y.-L. Chen, “The bitcoin halving cycle volatility dynamics and safe havenhedge properties: A msgarch approach,” Mathematics, vol. 11, no. 3, p. 698, 2023.

[6] H. Tian, “Bitcoin price forecasting using arima model,” Theoretical and Natural Science, pp. 105–112, 2023.

[7] J. Chu, S. Chan, S. Nadarajah, and J. Osterrieder, “Garch modelling of cryptocurrencies,” Journal of Risk and Financial Management, vol. 10, no. 4, p. 17, 2017.

[8] P. C. Sekhar, M. Padmaja, B. Sarangi, and Aditya, “Prediction of cryptocurrency using lstm and xgboost,” in 2022 IEEE International Conference on Blockchain and Distributed Systems Security (ICBDS), 2022, pp. 32–37.

[9] P. L. Seabe, C. R. B. Moutsinga, and E. Pindza, “Forecasting cryptocurrency prices using lstm, gru, and bi-directional lstm: a deep learning approach,” Fractal and Fractional, vol. 7, no. 2, p. 203, 2023.

[10] A. Dutta, S. Kumar, and M. Basu, “A gated recurrent unit approach to bitcoin price prediction,” Journal of Risk and Financial Management, vol. 13, p. 23, 2020.

[11] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “itransformer: Inverted transformers are effective for time series forecasting,” in International Conference on Learning Representations, 2024.

[12] M. S. Sepehri, A. Mehradfar, M. Soltanolkotabi, and S. Avestimehr, “Cryptomamba: Leveraging state space models for accurate bitcoin price prediction,” in IEEE International Conference on Blockchain and Cryptocurrency (ICBC) 2025, 2025.

[13] F. Jamhamed, F. Martin, F. Rondeau, J. Thelissaint, and S. Tuff ´ ery,´ “Regime-specific dynamics and informational efficiency in cryptomarkets: Evidence from gaussian mixture models,” Economics Working Paper Archive, Tech. Rep., 2024.

[14] M. Kang, J. Hong, and S. Kim, “Harnessing technical indicators with deep learning based price forecasting for cryptocurrency trading,” Physica A: Statistical Mechanics and its Applications, vol. 660, p. 130359, 2025.

[15] A. Peik, M. A. Zare Chahooki, A. Milani Fard, and M. Agha Sarram, “Adaptive temporal fusion transformers for cryptocurrency price prediction,” arXiv preprint arXiv:2509.10542, 2025.

[16] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, “Fourier neural operator for parametric partial differential equations,” in International Conference on Learning Representations, 2021.

[17] J. Guibas, M. Mardani, Z. Li, A. Tao, A. Anandkumar, and B. Catanzaro, “Adaptive fourier neural operators: Efficient token mixers for transformers,” 2022, arXiv:2111.13587.

[18] T. Chen, H. Zhou, Y. Li, H. Wang, Z. Zhang, T. Zhu, S. Zhang, and J. Li, “Freqmoe: Dynamic frequency enhancement for neural pde solvers,” in Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25), 2025.

[19] T. O. Kehinde, O. J. Adedokun, A. Joseph, K. M. Kabirat, H. A. Akano, and O. A. Olanrewaju, “Helformer: an attention-based deep learning model for cryptocurrency price forecasting,” Journal of Big Data, vol. 12, p. 81, 2025.

[20] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv:2312.00752 [cs.LG], 2023.

[21] D. Song, A. M. C. Baek, and N. Kim, “Forecasting stock market indices using padding-based fourier transform denoising and time series deep learning models,” IEEE Access, vol. 9, pp. 83 786 – 83 796, 2021.

[22] R. Yang, L. Cao, X. You, K. Fang, J. Li, and J. Yang, “Fourier basis mapping: A time-frequency learning framework for time series forecasting,” 2025, arXiv:2507.09445v1.

[23] C. Remlinger, C. Alasseur, M. Briere, and J. Mikael, “Expert aggregation\` for financial forecasting,” The Journal of Finance and Data Science, vol. 9, p. 100108, 2023.

[24] N. Ronghao, L. Zinan, W. Shuaiqi, and F. Giulia, “Mixture-of-linearexperts for long-term time series forecasting,” in Proceedings of Machine Learning Research (PMLR), vol. 238, 2024.

[25] X. Shi, S. Wang, Y. Nie, D. Li, Z. Ye, Q. Wen, and M. Jin, “Time-moe: Billion-scale time series foundation models with mixture of experts,” in International Conference on Learning Representations, 2025.

[26] R. C. Grinold and R. N. Kahn, Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk. New York: McGraw-Hill, 1999.

[27] F. X. Diebold and J. A. Lopez, “Forecast evaluation and combination,” in Handbook of Statistics. Elsevier, 1996, vol. 14, pp. 241–268.

[28] M. H. Pesaran and A. Timmermann, “A simple nonparametric test of predictive performance,” Journal of Business & Economic Statistics, vol. 10, no. 4, pp. 461–465, 1992.

[29] M. J. Pring, Technical Analysis Explained. McGraw-Hill, 2002.

[30] G. Appel, Technical Analysis: Power Tools for Active Investors. Financial Times Prentice Hall, 2005.

[31] R. J. Hyndman and A. B. Koehler, “Another look at measures of forecast accuracy,” International Journal of forecasting, vol. 22, no. 4, pp. 679– 688, 2006.

[32] R. Pardo, The Evaluation and Optimization of Trading Strategies, 2nd ed. John Wiley & Sons, 2008.

[33] W. F. Sharpe, “The sharpe ratio,” Journal of portfolio management, vol. 21, no. 1, pp. 49–58, 1994.
# Deep Learning Based Relative Transfer Matrix Estimation for Multiple Sources and Multiple Microphones

Oshan A. B. Yalegama <sup>ID</sup> <sup>1,∗∗</sup>, Wageesha N. Manamperi <sup>ID</sup> <sup>1,2</sup>

<sup>1</sup> Department of Electronic and Telecommunication Engineering, University of Moratuwa, Sri Lanka <sup>2</sup> School of Engineering, The Australian National University, Australia

yalegamammoab.20@uom.lk, wageesham@uom.lk

## Abstract

The Relative Transfer Matrix (ReTM), recently introduced as a generalization of the relative transfer function for multiple receivers and sources, shows promising performance when applied to speech enhancement in noisy environments. Estimating the ReTM of sound sources by exploiting the covariance matrices of multichannel recordings is highly beneficial for practical applications and, to date, remains the only proposed approach. This paper investigates deep learning-based ReTM estimation. We propose three novel supervised learning frameworks using time and short-time frequency transform domain convolutional networks, and a Long Short-Term Memory-based recurrent neural network. Experimental results demonstrate that the proposed models achieve more accurate estimation of the ReTM using five objective metrics compared to the covariance-based method. We also show the effectiveness of the proposed frameworks for speech enhancement, achieving performance on par with the baseline method.

Index Terms: Relative transfer matrix, speech enhancement, CNN, LSTM, time domain

## 1. Introduction

Blind estimation of the relative transfer function (ReTF) without requiring positional knowledge of receivers or sound sources is highly attractive in practical applications such as robot audition [1], drone audition [2], teleconferencing [3], and hearing aids [4], which involve tasks such as sound source localization, speech enhancement, speaker separation, and acoustic echo cancellation [5, 6, 7, 8]. However, these applications typically involve multiple simultaneously active sound sources, where the assumption of W-disjoint orthogonality [9] required for ReTF estimation does not hold. To overcome this limitation, the relative transfer matrix (ReTM) [10] has been introduced as a generalization of the ReTF for acoustic environments with multiple simultaneously active sound sources and multiple microphones, providing an essential component in audio signal processing techniques and applications. In this paper, we propose machine learning-based methods for ReTM estimation from microphone recordings.

Many approaches to estimate ReTF for a single active sound source have been developed [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]. Among them, covariancebased methods have gained attention due to their practical feasibility as well as simplicity [11, 12, 13, 14, 15, 16], whereas, machine learning-based approaches offer superior performance at an increased computational cost [19, 20, 21, 22, 23, 24, 25].

In [10], the concept of ReTM was introduced to relate the signal received between two sets of microphone groups with respect to all active sources present in a room. Similar to the ReTF [6], the ReTM is independent of source signals but dependent on the spatial location of the sources and the environment [10]. Recently, various studies have been proposed to utilize ReTM for speech enhancement and speaker separation in multi-source noisy reverberant environments [26, 27, 28, 29, 30], which have shown significant performance improvement. Traditional ReTM estimation approach exploits statistical relationships between two microphone groups using covariance matrices [10]. In contrast to the baseline method that learns a spatial mapping between receiver groups, alternative approaches remain largely unexplored.

In this paper, we introduce three fully supervised machine learning frameworks for ReTM estimation. We use both time and short-time Fourier transform (STFT) domains, deep learning approaches. All three models assume a stationary environment that all sound sources are unmoving. The contributions of this work are as follows: 1) We propose: i) the STFT Convolutional Network (SCoNet) that uses depthwise convolution; ii) the Convolutional Filter and Summation Network (FuS-Net) that applies a set of individual convolutional filters; and iii) the Long Short-Term Memory (LSTM)-based Autoencoder Network (LAeNet) that uses a shared bidirectional-LSTM followed by a feedforward network. 2) We evaluate ReTM estimation accuracy with respect to the covariance-based approach using five qualitative metrics, signal-to-distortion ratio (SDR), mean square error (MSE), log spectral distortion (LSD), average phase distortion (APD), and relative spectrum error (RSE). 3) We demonstrate the merits of the proposed models in speech denoising. The simulation results are comparable to the baseline method overall, while the STFT domain models consistently outperform it.

## 2. Problem formulation

Consider a reverberant environment with concurrently active L sound sources. In the short time Fourier transform (STFT) domain, we denote $S _ { \ell } ( f , t ) , \ell = 1 , \cdots , \mathcal { L }$ as the source signals. Let there be Q arbitrary distributed microphones in the room. We divide them to two groups of microphones, {A} and $\{ B \}$ with $Q _ { A }$ and $Q _ { B }$ microphones, respectively $( Q = Q _ { A } + Q _ { B } ) .$ We denote ${ \bf M } _ { A } ( f , t )$ and ${ \bf M } _ { B } ( f , t )$ as the vector of received signals at microphone groups A and B, respectively. Then the received signals at each microphone group in matrix form as

$$
\mathbf { M } _ { A } ( f , t ) = \mathbf { H } _ { A } ( f ) \mathbf { S } ( f , t ) ,\tag{1}
$$

$$
\mathbf { M } _ { B } ( { f } , t ) = \mathbf { H } _ { B } ( { f } ) \mathbf { S } ( { f } , t ) ,
$$

where $\mathbf { S } ( f , t ) = [ S _ { 1 } ( f , t ) , \ldots , S _ { \mathcal { L } } ( f , t ) ] ^ { T }$ , and $\{ \cdot \} ^ { T }$ is the matrix transpose. Here, $\mathbf { H } _ { A } ( f ) \in \mathbb { C } ^ { Q _ { A } \times \mathcal { L } }$ and $\mathbf { H } _ { B } \bar { ( f ) } \in \mathbb { C } ^ { Q _ { B } \times { \mathcal { L } } }$ are the matrices with elements defined by the acoustic transfer functions. Note that we consider thermal microphone noise to be negligible.

The ReTM, $\mathcal { R } _ { A B } ( f )$ , is defined as in [10]

$$
\pmb { \mathcal { R } } _ { A B } ( f ) \triangleq \mathbf { H } _ { A } ( f ) \mathbf { H } _ { B } ( f ) ^ { \dagger } ,\tag{2}
$$

where $( \cdot ) ^ { \dagger }$ is Moore-Penrose inverse, assuming the validity, i.e., $Q _ { B } \geq { \mathcal { L } }$ . Thus, we can relate the received signal at group {A} and $\{ B \}$ using

$$
\mathbf { M } _ { A } ( { f } , t ) = \pmb { \mathcal { R } } _ { A B } ( { f } ) \mathbf { M } _ { B } ( { f } , t ) .\tag{3}
$$

In time domain Equation 3 is given by

$$
m _ { A } ^ { j } ( t ) = \sum _ { i = 1 } ^ { Q _ { B } } r _ { A B } ^ { j i } ( t ) * m _ { B } ^ { i } ( t ) , \quad j = 1 , \cdots , Q _ { A } ,\tag{4}
$$

where $m _ { A } ^ { j }$ , and m $\boldsymbol { \mathrm { \Sigma } } _ { B } ^ { i }$ denote the signals of the $j ^ { \mathrm { t h } }$ microphone in group $\{ A \}$ , and the $i ^ { \mathrm { t h } }$ microphone in group $\{ B \}$ , respectively, $r _ { A B } ^ { j i }$ denotes the inverse Fourier transform of the $\{ j , i \} ^ { \mathrm { t } }$ element of $\mathcal { R } _ { A B } ( f )$

The aim of this paper is to exploit the spatial properties of sound sources by modeling the ReTM with machine learning algorithms in the time domain and STFT domain as $f _ { B } ( \cdot )$ , and $F _ { B } ( \cdot )$ , respectively, such that

$$
M _ { A } ( f , t ) = F _ { B } ( M _ { B } ( f , t ) ) ,\tag{5}
$$

$$
m _ { A } ( t ) = f _ { B } ( m _ { B } ( t ) ) .\tag{6}
$$

The next section proposes machine learning approaches for the modeling of $f _ { B } ( \cdot )$ and $F _ { B } ( \cdot )$

## 3. Proposed ReTM estimation models

In this section, we present two convolutional networks and one Long Short-Term Memory (LSTM)-based recurrent neural network for the modeling of the ReTM.

## 3.1. STFT convolutional network (SCoNet)

We model $F _ { B } ( \cdot )$ using SCoNet, which applies depthwise convolution operations in the STFT domain, as shown in Figure 1. We input $M _ { B } ( f , t )$ , STFT of microphone signals in group {B}, and train to estimate the microphone signals in group {A}. SCoNet stacks the real and imaginary parts along the channel dimension and performs two-dimensional depthwise convolution across the time and channel axes, allowing the model to learn distinct parameter sets for each frequency component of the relative transfer matrix. The channel-wise outputs of group {A} are then stacked to form the STFT representation of $M _ { A } ( f , t )$

## 3.2. Convolutional filter and summation network (FuSNet)

Motivated by the effectiveness of convolutional neural networks (CNNs) in modeling the ReTF [23], we propose FuSNet, which parameterizes $f _ { B } ( \cdot )$ using $Q _ { A } \times Q _ { E }$ learnable one-dimensional convolutional filters, whose weights correspond to $r _ { A B } ^ { j i } ( t )$ in Equation 4. Figure 2 shows the FuSNet architecture. We input non-overlapping segments of length $L ,$ and a context window of size 3L from the time frames of microphone signals at group {B}. The context window is chosen such that the convolution output matches the original segment length, setting the filter size to $L .$ Note that the window length should be longer than the room’s reverberation time to satisfy the multiplicative transfer function [23, 31]. The convolution outputs are regrouped and summed to obtain the time domain microphone signals at group {A}.

![](images/f837ee0ce3b756adc4ae51241ae2063a13577c9e91ba0922e658438a5444db63.jpg)  
Figure 1: Model architecture of SCoNet, where F and T are the number offrequency bins and timeframes, respectively.

## 3.3. LSTM based autoencoder network (LAeNet)

We propose LAeNet, an LSTM-based autoencoder for modeling of $F _ { B } ( \cdot )$ , as shown in Figure 3. This design exploits LSTMs’ ability to capture narrow-band spatial information [32]. In LAeNet, each frequency bin of the concatenated STFTs $M _ { A } ( f , t )$ and $M _ { B } ( f , t )$ is independently processed by a bidirectional-LSTM (BiLSTM) layer with shared weights. The resulting temporal sequences are passed through layer normalization to stabilize and accelerate training, then averaged over time to yield feature vectors of size $4 ( Q _ { B } + Q _ { A } )$ , where the increase in dimensionality stems from the BiLSTM. Each feature vector is subsequently mapped by a fully connected network $k ( \cdot )$ to estimate the ReTM coefficients, which are finally used to estimate the group {A} microphone signals $M _ { A } ( f , t )$ as in Equation 3.

## 3.4. Loss function and training

To improve the stability and performance of the proposed models, we train using a weighted sum of the negative signal-todistortion ratio (SDR) computed in the time domain, and the relative spectrum error (RSE) measured in the STFT domain as:

$$
{ \cal L } = - \alpha { \cal L } _ { S D R } ( m _ { A } , \hat { m } _ { A } ) + \beta { \cal L } _ { R S E } ( M _ { A } , \hat { M } _ { A } ) ,\tag{7}
$$

where $\hat { m } _ { A }$ , and $\hat { M } _ { A }$ denote the estimated microphone signals at group {A} in the time and STFT domains, respectively. This joint optimization enforces consistency across both representations and improves robustness to noise and artifacts that may manifest differently in the time and frequency domains. Here, α and $\beta$ are weight factors for

$$
L _ { S D R } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } 1 0 \log _ { 1 0 } \left( \frac { | m _ { A } ( t ) | _ { 2 } ^ { 2 } } { | m _ { A } ( t ) - \hat { m } _ { A } ( t ) | _ { 2 } ^ { 2 } } \right)\tag{8}
$$

and

$$
L _ { R S E } = \frac { 1 } { F T } \sum _ { t = 1 } ^ { T } \sum _ { f = 1 } ^ { F } 1 0 \log _ { 1 0 } \left( \frac { | \hat { M } _ { A } ( f , t ) - M _ { A } ( f , t ) | ^ { 2 } } { | { M } _ { A } ( f , t ) | ^ { 2 } } \right) ,\tag{9}
$$

respectively. For training, we employ the Adam optimizer [33] with a learning rate that is adaptively reduced over epochs based on the validation set performance.

![](images/326e5eb5242af56dd53e6e323abce94408e7c57c028d2c7e9c305eb4b682dae6.jpg)  
Figure 2: Model architecture of FuSNetfor the case of $Q _ { B } = 4$ and $Q _ { A } = 3$

![](images/9af22d04c8cb058c98d27919d53375679f7d535e99a704dc133c843e28f6b571.jpg)  
Figure 3: Model architecture ofLAeNet with a shared BiLSTM.

## 4. Experiments

## 4.1. Experimental methodology

We utilize an open-source toolbox [34] to model the room impulse response from the sound sources to irregularly distributed microphones in a $6 \times 7 \times 3$ m rectangular room $( T _ { 6 0 } = 5 0 0$ ms). We assign $Q = 7$ with $Q _ { A } = 3 ,$ , and $Q _ { B } = 4$ number of receivers to group {A} and $\{ B \}$ , respectively. We consider four scenarios. A1: two white Gaussian noise (WGN) sources, A2: two noise sources (air conditioner noise, music), and B: three sources (speech, air conditioner noise, music). Scenario C uses a total of $Q = 1 2$ microphones with $Q _ { A } = 5 ,$ and $Q _ { B } = 7 ,$ drives with the same set of sources as B. The received signals are added with 40 dB SNR of WGN at each microphone and down-sampled to 16 kHz. For Scenario A1, the training, validation, and test recordings are 3 minutes, 1 minute, and 1 minute, respectively. For all other scenarios, 50 s and 10 s recordings are used for training and testing, respectively.

For the model hyperparameter configuration, FuSNet is designed with a window and context size of 8192 samples. For SCoNet and LAeNet, the input recordings are converted to STFT frames using a Hann window of size 8192 with 50% overlap. The window length is selected based on the room’s reverberation time to ensure that it is sufficiently long for the multiplicative transfer function assumption to hold [31]. The weight factors α and β are set to 1 and 10, respectively.

For a comprehensive evaluation of the proposed models, we adopt the covariance-based approach [10] as the baseline. To fairly assess performance across all scenarios, we use five quantitative metrics: (i) SDR, (ii) mean square error (MSE), defined as

$$
\mathrm { M S E } = 1 0 \log _ { 1 0 } \left( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left| m _ { A } ( t ) - \hat { m } _ { A } ( t ) \right| ^ { 2 } \right) ,
$$

(iii) log spectral distortion (LSD), defined as

$$
\mathrm { L S D } = \sqrt { \frac { 1 } { N } \sum _ { \eta = 1 } ^ { N } \left( 1 0 \log _ { 1 0 } | { \cal M } _ { A } ( f _ { \eta } , t ) | ^ { 2 } / | \hat { \cal M } _ { A } ( f _ { \eta } , t ) | ^ { 2 } \right) , ^ { 2 } }
$$

(iv) average phase distortion (APD), defined as

$$
\mathrm { A P D } = \frac { 1 } { N } \sum _ { \eta = 1 } ^ { N } \left| \angle M _ { A } ( f _ { \eta } , t ) - \angle \hat { M } _ { A } ( f _ { \eta } , t ) \right| ,
$$

and (v) RSE.

## 4.2. Results and discussion

The ReTM estimation accuracy results are given in Table 1. From the results in scenario A1, we observe that FuSNet achieves the highest performance across both time and frequency domain metrics. This improvement of FuSNet can be attributed to the use of WGN sources that provide a uniform spectral distribution and thereby facilitate more accurate frequency domain estimation. The other proposed methods, SCoNet and LAeNet, also achieve high estimation accuracy, performing comparably to the baseline method.

In scenario A2, FuSNet again achieves the highest overall performance across most evaluation metrics, with noticeable degradation in LSD and APD. The key difference between scenarios A1 and A2 lies in the replacement of WGN with specific noise sources, which exhibit non-uniform frequency distributions. This indicates that FuSNet performs best with uniform spectra but degrades under non-uniform conditions. Moreover, SCoNet yields performance on par with LAeNet and surpasses the state-of-the-art method, excluding the RSE score, where the baseline approach shows the second-best performance.

From scenarios B and C, we observe that all four methods exhibit improved ReTM estimation accuracy as the number of microphones in both groups increases, consistently across all five evaluation metrics. The traditional method exhibits significantly reduced accuracy, and is seen to be the worst performing method among all methods, except for the LSD and RSE measures. Compared to the baseline, both SCoNet and FuSNet, show substantially superior performance. SCoNet performs best in scenario B with an MSE score lower than FuSNet. Whereas, FuSNet achieves the highest performance in scenario C, excluding the LSD and APD measures.

Overall, it can be verified that although the baseline method achieves satisfactory results, it is clearly outperformed by the proposed deep learning-based models, particularly in the time domain measures, e.g., MSE and SDR.

Table 2 compares the computational efficiency of the proposed models in terms of the parameter count, alongside the inference latency measured on an NVIDIA GeForce RTX 3090 GPU. Across both scenarios, FuSNet demonstrates the lowest latency and the smallest parameter count, whereas LAeNet exhibits the highest latency and the largest number of parameters. Notably, FuSNet and SCoNet benefit from faster training due to their comparatively low latency, although their memory usage increases linearly with the window length. In contrast, LAeNet maintains nearly constant memory usage as the window length increases, owing to its shared LSTM and feedforward network architecture. This design allows efficient parameter utilization for longer windows but results in substantially longer training times due to higher latency.

Table 1: ReTM estimation accuracy for various scenarios (Channel 1/Average).
<table><tr><td rowspan=1 colspan=1>Scenario</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=3>SDR (dB)↑</td><td rowspan=1 colspan=2>MSE (dB)↓</td><td rowspan=1 colspan=1>LSD (dB)↓</td><td rowspan=1 colspan=1>APD (rad)↓</td><td rowspan=1 colspan=2>RSE (dB)↓</td></tr><tr><td rowspan=4 colspan=1>A1</td><td rowspan=4 colspan=1>BaselineSCoNetFuSNetLAeNet</td><td rowspan=3 colspan=3>26.56/26.124.76/24.2429.13/28.46</td><td rowspan=2 colspan=1>-43.84-40.88</td><td rowspan=1 colspan=1>43.56</td><td rowspan=1 colspan=1>1.21/1.19</td><td rowspan=2 colspan=1>0.1/0.10.13/0.14</td><td rowspan=2 colspan=2>-24.76// - 24.10-25.62// - 24.89</td></tr><tr><td rowspan=1 colspan=1>/ - 40.54</td><td rowspan=1 colspan=1>0.89/0.96</td></tr><tr><td rowspan=1 colspan=1>-59.25</td><td rowspan=1 colspan=1>-58.75/</td><td rowspan=1 colspan=1>0.5/0.49</td><td rowspan=1 colspan=1>0.08/0.09</td><td rowspan=2 colspan=2>-48.61/ - 48.4-20.25/i - 19.74</td></tr><tr><td rowspan=1 colspan=3>25.12/24.09</td><td rowspan=1 colspan=1>-55.24</td><td rowspan=1 colspan=1>/ - 54.38/</td><td rowspan=1 colspan=1>1.03/1.13</td><td rowspan=1 colspan=1>0.12/0.13</td></tr><tr><td rowspan=3 colspan=1>A2</td><td rowspan=3 colspan=1>BaselineSCoNetFuSNetLAeNet</td><td rowspan=3 colspan=3>21.2/20.526.26/26.5930.96/29.6228.07/26.21</td><td rowspan=1 colspan=1>-36.25</td><td rowspan=1 colspan=1>36.57</td><td rowspan=1 colspan=1>2.02/2.03</td><td rowspan=1 colspan=1>0.21/0.21</td><td rowspan=2 colspan=2>-25.28/24.65-24.61/− 24.06</td></tr><tr><td rowspan=2 colspan=1>-43.08/-55.2/-52.32</td><td rowspan=1 colspan=1>/ - 42.47</td><td rowspan=1 colspan=1>1.22/1.23</td><td rowspan=1 colspan=1>0.18/0.19</td></tr><tr><td rowspan=1 colspan=1>-54.94-51.53</td><td rowspan=1 colspan=1>2.47/2.591.68/1.70</td><td rowspan=1 colspan=1>0.23/0.20.22/0.21</td><td rowspan=1 colspan=2>-39.81/-33.78-18.17/-17.67</td></tr><tr><td rowspan=5 colspan=1>B</td><td rowspan=5 colspan=1>BaselineSCoNetFuSNetLAeNet</td><td rowspan=5 colspan=3>16.1/16.2922.42/21.9622.51/21.9122.36/21.88</td><td rowspan=1 colspan=1>-36.65</td><td rowspan=1 colspan=1>一37.49</td><td rowspan=1 colspan=1>2.48/2.5</td><td rowspan=1 colspan=1>0.23/0.24</td><td rowspan=1 colspan=2>-17.39-16.85</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>-42.78</td><td rowspan=1 colspan=1>/ - 42.98</td><td rowspan=1 colspan=1>1.53/1.54</td><td rowspan=1 colspan=1>0.19/0.19</td><td rowspan=1 colspan=2>-20.76// -20.33</td></tr><tr><td rowspan=2 colspan=1>-45.12/</td><td rowspan=2 colspan=1>- 45.17</td><td rowspan=2 colspan=1>4.89/4.96</td><td rowspan=2 colspan=1>0.71/0.72</td><td rowspan=2 colspan=2>-19.60// - 19.16</td></tr><tr><td rowspan=2 colspan=1>-45.01</td></tr><tr><td rowspan=1 colspan=1>/-45.18</td><td rowspan=1 colspan=1>1.78/1.88</td><td rowspan=1 colspan=1>0.2/0.21</td><td rowspan=1 colspan=1>-17.58</td><td rowspan=1 colspan=1>-17.21</td></tr><tr><td rowspan=3 colspan=1>C</td><td rowspan=3 colspan=1>BaselineSCoNetFuSNetLAeNet</td><td rowspan=3 colspan=1>23.05/29.59/40.33/29.35/</td><td rowspan=1 colspan=2>23.25</td><td rowspan=1 colspan=1>-43.6</td><td rowspan=1 colspan=1>/-43.35</td><td rowspan=1 colspan=1>1.03/1.09</td><td rowspan=1 colspan=1>0.15/0.15</td><td rowspan=1 colspan=1>-34.03</td><td rowspan=1 colspan=1>-33.22</td></tr><tr><td rowspan=1 colspan=2>27.85</td><td rowspan=1 colspan=1>-44.12/</td><td rowspan=1 colspan=1>− 43.27</td><td rowspan=1 colspan=1>1.06/1.17</td><td rowspan=1 colspan=1>0.14/ 0.16</td><td rowspan=1 colspan=1>-28.63/</td><td rowspan=1 colspan=1>- 27.43</td></tr><tr><td rowspan=1 colspan=2>38.8828.57</td><td rowspan=1 colspan=2>-62.89/-61.98-52/-51.77</td><td rowspan=1 colspan=1>1.95/2.101.39/1.49</td><td rowspan=1 colspan=1>0.15/0.160.14/0.15</td><td rowspan=1 colspan=1>-34.07/-20.15/</td><td rowspan=1 colspan=1>-33.27- 19.97</td></tr></table>

Table 2: Computational complexity of each proposed model.
<table><tr><td rowspan="2">Method</td><td> $\overline { { Q A = 3 , Q _ { B } = 4 } }$ </td><td> $\overline { { Q _ { A } = 5 , Q _ { B } = 7 } }$ </td></tr><tr><td>Param. Latency</td><td>Param. Latency</td></tr><tr><td>SCoNet</td><td>196.7k 6.77ms</td><td>573.6k 6.61ms</td></tr><tr><td>FuSNet</td><td>98.3k 1.07ms</td><td>286.8k 2.41ms</td></tr><tr><td>LAeNet</td><td>263.5k 1.80s</td><td>652.5k 1.78s</td></tr></table>

## 5. Application into speech enhancement

Motivated by the computational efficiency of the proposed architectures, in this section we further evaluate the ReTM estimation accuracy in a speech denoising application [27].

Denote $\mathcal { R } _ { A B } ^ { ( N ) }$ be the noise sources ReTM. In brief, [27] proposed to enhance the speech from the noisy speech recordings with the known $\mathcal { R } _ { A B } ^ { ( N ) }$ , which can be estimated using the noise-only recordings, as

$$
\hat { \mathbf { S } } \triangleq \mathbf { M } _ { A } - \pmb { \mathcal { R } } _ { A B } ^ { ( N ) } \mathbf { M } _ { B } = \left[ \mathbf { h } _ { A } - \pmb { \mathcal { R } } _ { A B } ^ { ( N ) } \mathbf { h } _ { B } \right] S ,\tag{10}
$$

where $\mathbf { h } _ { A }$ and $\mathbf { h } _ { B }$ be the acoustic transfer function vectors from the speech source to group $\{ A \}$ and {B} microphones, respectively. We note that $\hat { \bf S }$ is a $Q _ { A } \times 1$ vector consists $Q _ { A }$ copies of estimated target speech signal S.

We evaluate speech denoising performance in scenarios B and C (Sec. 4), each containing a single speech source, where the ReTM is estimated from 1-minute noise-only recordings. For analysis we use the SDR [35], and the Short-Time Objective Intelligibility (STOI) [36].

Table 3 depicts the speech enhancement accuracy for both B and C scenarios. The results confirm that enhanced performance (+0.5 dB in SDR, 13% in STOI) for all methods. Although FuSNet significantly outperforms ReTM estimation on most metrics, its speech denoising performance degrades, confirming the advantage of time-frequency domain ReTM estimation over the time domain approach. Compared to FuSNet, both

Table 3: Evaluation results on the scenarios B & C.
<table><tr><td rowspan="2">Method</td><td colspan="2">B</td><td colspan="2">C</td></tr><tr><td>SDR</td><td>STOI</td><td>SDR</td><td>STOI</td></tr><tr><td>Noisy</td><td>-2.70</td><td>0.54</td><td>-2.70</td><td>0.54</td></tr><tr><td>Baseline</td><td>6.06</td><td>0.87</td><td>4.07</td><td>0.85</td></tr><tr><td>SCoNet</td><td>6.45</td><td>0.87</td><td>4.96</td><td>0.87</td></tr><tr><td>FuSNet</td><td>2.21</td><td>0.80</td><td>-1.19</td><td>0.67</td></tr><tr><td>LAeNet</td><td>8.67</td><td>0.92</td><td>7.03</td><td>0.91</td></tr></table>

LAeNet and SCoNet, demonstrate significantly superior performance. LAeNet achieves the highest performance in both scenarios B and C, with an average SDR of +10.55 dB and an average STOI of 37%. Whereas, SCoNet ranks second in both scenarios, yielding an average SDR of +8.41 dB and an average STOI 33%. In contrast, the traditional method demonstrates comparable performance, although the proposed models, except FuSNet, consistently achieve better results in both scenarios. From informal listening, we find that FuSNet’s denoised speech remains significant echo noise, leading to lower performance, which could be improved through dereverberation algorithms [37, 38] that we will investigate in future work. The audio samples of this work can be found on GitHub.<sup>1</sup>

## 6. Conclusion

We proposed SCoNet, FuSNet, and LAeNet to estimate the ReTM using machine learning. Our proposed approaches use either time or time-frequency domains to model the inter-group spatial relationship. Experimental results confirmed that FuS-Net consistently outperformed other approaches in terms of accuracy, while SCoNet and LAeNet achieved performance comparable to the baseline method. We further validated the effectiveness of the proposed methods in a speech denoising application. In future work, we aim to extend these models to source separation, speech dereverberation, and investigate optimal microphone grouping strategies for ReTM-based applications.

## 7. Acknowledgements

This work was supported by the Accelerating Higher Education Expansion and Development (AHEAD) operation (Grant No. 6026-LK/8743-LK, World Bank) and a data scholarship from the Linguistic Data Consortium, University of Pennsylvania. The authors thank Prof. Thushara Abhayapala for fruitful discussions and support on this work.

## 8. Use of Generative AI Disclosure

Generative AI tools were used to assist in drafting and refining portions of the scripts. Additionally, grammar-checking models were utilized to improve language clarity, correctness, and overall readability.

## 9. References

[1] K. Nakadai, T. Lourens, H. G. Okuno, and H. Kitano, “Active audition for humanoid,” in Natl. Conf. Artif. Intell., Jul. 2000, pp. 832–839.

[2] W. Manamperi, T. D. Abhayapala, J. A. Zhang, and P. N. Samarasinghe, “Drone audition: Sound source localization using onboard microphones,” IEEE/ACM Trans. on Audio, Speech, and Lang. Process., vol. 30, pp. 508 – 519, Jan. 2022.

[3] S. Araki, M. Fujimoto, K. Ishizuka, H. Sawada, and S. Makino, “Speaker indexing and speech enhancement in real meetings/conversations,” in Proc. IEEE Int. Conf. on Acoust., Speech and Signal Process., Mar. 2008, pp. 93–96.

[4] D. Marquardt, E. Hadad, S. Gannot, and S. Doclo, “Incorporating relative transfer function preservation into the binaural multichannel wiener filter for hearing aids,” in Proc. IEEE Int. Conf. on Acoust., Speech and Signal Process., Mar. 2016, pp. 6500–6504.

[5] S. Gannot, D. Burshtein, and E. Weinstein, “Signal enhancement using beamforming and nonstationarity with applications to speech,” IEEE Trans. on Signal Process., vol. 49, no. 8, pp. 1614– 1626, Aug. 2001.

[6] R. Talmon, I. Cohen, and S. Gannot, “Relative transfer function identification using convolutive transfer function approximation,” IEEE Trans. on Audio, Speech, and Lang. Process., vol. 17, no. 4, pp. 546–555, Mar. 2009.

[7] E. Warsitz, A. Krueger, and R. Haeb-Umbach, “Speech enhancement with a new generalized eigenvector blocking matrix for application in a generalized sidelobe canceller,” in Proc. IEEE Int. Conf. on Acoust., Speech and Signal Process., Mar. 2008, pp. 73– 76.

[8] W. N. Manamperi, T. D. Abhayapala, L. Brinie, J. Zhang, and P. N. Samarasinghe, “Drone audition: On measurements and modeling of drone-related transfer functions,” IEEE/ACM Trans. on Audio, Speech, and Lang. Process., vol. 33, pp. 1775 – 1786, Apr. 2025.

[9] O. Yilmaz and S. Rickard, “Blind separation of speech mixtures via time-frequency masking,” IEEE Signal Process. Mag., vol. 52, no. 7, pp. 1830–1847, Jun. 2004.

[10] T. D. Abhayapala, L. Birnie, M. Kumar, D. Grixti-Cheng, and P. N. Samarasinghe, “Generalizing the relative transfer function to a matrix for multiple sources and multichannel microphones,” in Proc. Eur. Signal Process. Conf., Sep. 2023, pp. 336–340.

[11] R. Serizel, M. Moonen, B. Van Dijk, and J. Wouters, “Lowrank approximation based multichannel wiener filter algorithms for noise reduction with application in cochlear implants,” IEEE/ACM Trans. on Audio, Speech, and Lang. Process., vol. 22, no. 4, pp. 785–799, Feb. 2014.

[12] R. Varzandeh, M. Taseska, and E. A. P. Habets, “An iterative mul tichannel subspace-based covariance subtraction method for relative transfer function estimation,” in Proc. Joint Workshop Handsfree Speech Comm. and Microphone Arrays, Mar. 2017, pp. 11– 15.

[13] S. Markovich, S. Gannot, and I. Cohen, “Multichannel eigenspace beamforming in a reverberant noisy environment with multiple interfering speech signals,” IEEE/ACM Trans. on Audio, Speech, and Lang. Process., vol. 17, no. 6, pp. 1071–1086, Jun. 2009.

[14] W. Middelberg, H. Gode, and S. Doclo, “Relative transfer function vector estimation for acoustic sensor networks exploiting covariance matrix structure,” in Proc. IEEE Workshop on Applications ofSignal Process. to Audio and Acoust., Oct. 2023, pp. 1–5.

[15] E. Warsitz and R. Haeb-Umbach, “Blind acoustic beamforming based on generalized eigenvalue decomposition,” IEEE Trans. on Audio, Speech, and Lang. Process., vol. 15, no. 5, pp. 1529–1539, Jun. 2007.

[16] S. Markovich-Golan, S. Gannot, and W. Kellermann, “Performance analysis of the covariance-whitening and the covariancesubtraction methods for estimating the relative transfer function,” in Proc. Eur. Signal Process. Conf., Sep. 2018, pp. 2499–2503.

[17] O. Shalvi and E. Weinstein, “System identification using nonstationary signals,” IEEE Trans. on Signal Process., vol. 44, no. 8, pp. 2055–2063, Aug. 1996.

[18] I. Cohen, “Relative transfer function identification using speech signals,” IEEE Trans. on Speech and Audio Process., vol. 12, no. 5, pp. 451–459, Aug. 2004.

[19] A. Brendel, J. Zeitler, and W. Kellermann, “Manifold learningsupported estimation of relative transfer functions for spatial filtering,” in Proc. IEEE Int. Conf. on Acoust., Speech and Signal Process., May 2022, pp. 8792–8796.

[20] A. Sofer, T. Kounovsky, J.\` Cmejla, Z. Koldovsk<sup>ˇ</sup> y, and S. Gannot,\` “Robust relative transfer function identification on manifolds for speech enhancement,” in Proc. Eur. Signal Process. Conf., Aug. 2021, pp. 401–405.

[21] B. Yang, X. Li, and H. Liu, “Supervised direct-path relative transfer function learning for binaural sound source localization,” in Proc. IEEE Int. Conf. on Acoust., Speech and Signal Process., Jun. 2021, pp. 825–829.

[22] B. Yang, H. Liu, and X. Li, “Learning deep direct-path relative transfer function for binaural sound source localization,” IEEE/ACM Trans. on Audio, Speech, and Lang. Process., vol. 29, pp. 3491–3503, Oct. 2021.

[23] L. Birnie, P. Samarasinghe, T. Abhayapala, and D. Grixti-Cheng, “Noise retf estimation and removal for low snr speech enhancement,” in ’IEEE Workshop Mach. Learning Signal Process.’, Oct. 2021, pp. 1–6.

[24] Z. Wang and D. Wang, “Mask weighted stft ratios for relative transfer function estimation and its application to robust asr,” in Proc. IEEE Int. Conf. on Acoust., Speech and Signal Process., Apr. 2018, pp. 5619–5623.

[25] D. Levi, A. Sofer, and S. Gannot, “peerrtf: Robust mvdr beamforming using graph convolutional network,” IEEE Trans. on Audio, Speech, and Lang. Process., Mar. 2025.

[26] W. N. Manamperi and T. D. Abhayapala, “Relative transfer matrix for drone audition applications: Source enhancement,” in Proc. Asia-Pacific Signal and Inf. Process. Assoc. Annu. Summit and Conf., Dec. 2024, pp. 1–6.

[27] M. Kumar, L. Birnie, T. Abhayapala, S. A. Holzinger, A. Bastine, D. Grixti-Cheng, and P. Samarasinghe, “Speech denoising in multi-noise source environments using multiple microphone devices via relative transfer matrix,” in Proc. Eur. Signal Process. Conf., Sep. 2024, pp. 336–340.

[28] W. N. Manamperi and T. D. Abhayapala, “Relative transfer matrix estimator using covariance subtraction,” arXiv preprint arXiv:2510.19439, Oct. 2025.

[29] ——, “Successive speaker relative transfer function estimation through relative transfer matrix in noisy reverberant environ ments,” in Proc. Asia-Pacific Signal and Inf. Process. Assoc. Annu. Summit and Conf., Dec. 2024, pp. 1–6.

[30] W. N. Manamperi, “Multiple speaker separation in reverberant rooms under low snr conditions using the relative transfer matrix,” acceptedfor publication in Appl. Acoust., Jun. 2026.

[31] Y. Avargel and I. Cohen, “On multiplicative transfer function approximation in the short-time fourier transform domain,” IEEE Signal Process. Letters, vol. 14, no. 5, pp. 337–340, Apr. 2007.

[32] Y. Yang, C. Quan, and X. Li, “Mcnet: Fuse multiple cues for multichannel speech enhancement,” in Proc. IEEE Int. Conf. on Acoust., Speech and Signal Process., Jun. 2023, pp. 1–5.

[33] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[34] E. A. Habets, “Room impulse response (RIR) generator,” 2006, [Online]. Available: https://www.audiolabserlangen.de/fau/ professor/habets/software/rir-generator.

[35] E. Vincent, R. Gribonval, and C. Fevotte, “Performance measure-´ ment in blind audio source separation,” IEEE Trans. on Audio, Speech, and Lang. Process., vol. 14, no. 4, pp. 1462–1469, Jun. 2006.

[36] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, “An algorithm for intelligibility prediction of time-frequency weighted noisy speech,” IEEE/ACM Trans. on Audio, Speech, and Lang. Process., vol. 19, no. 7, pp. 2125–2136, Feb. 2011.

[37] P. A. Naylor and N. D. Gaubitch, Speech dereverberation. Springer, Jun. 2010.

[38] Z.-Q. Wang and D. Wang, “Deep learning based target cancellation for speech dereverberation,” IEEE/ACM Trans. on Audio, Speech, and Lang. Process., vol. 28, pp. 941–950, Feb. 2020.
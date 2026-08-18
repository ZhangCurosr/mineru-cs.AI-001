# OceanLight: Eficient Global Ocean Forecasting via Geometry-Adaptive Unstructured Mesh Representation

Wei Wu<sup>1</sup>, Xiang Wang<sup>2\*</sup>, Hongze Leng<sup>2</sup>, Qingye Min<sup>3</sup>, Junxing Zhu<sup>2</sup>, Junqiang Song<sup>12\*</sup>

<sup>1</sup>College of Computer Science and Technology, National University of Defense Technology, Deya Street, Changsha, 410073, Hunan, China. <sup>2</sup>College of Meteorology and Oceanography, National University of Defense Technology, Deya Street, Changsha, 410073, Hunan, China.   
<sup>3</sup>Hunan Institute of Advanced Technology, Qingshan Street, Changsha, 410205, Hunan, China.

\*Corresponding author(s). E-mail(s): xiangwangcn@nudt.edu.cn; junqiang@nudt.edu.cn;

## Abstract

Reliable global ocean forecasting is critical for climate monitoring, marine navigation, and extreme event early warning. Physics-based ocean forecasting models impose prohibitive computational costs, while existing deep learning approaches predominantly rely on structured-grid architectures, incurring unnecessary computation on masked land cells and enforcing uniform resolution across dynamically heterogeneous ocean regions regardless of local flow complexity. Here we present OceanLight, an eficient global ocean forecasting framework innovatively combining geometry-adaptive unstructured mesh tokenization with a graph neural network (GNN) backbone. OceanLight achieves pointwise forecast accuracy and kinetic energy spectral fidelity exceeding both operational numerical analyses and state-of-the-art AI-based models, while surpassing all AI-based ocean models in geostrophic balance consistency. Furthermore, OceanLight demonstrates reliable mesoscale eddy representation, capturing coherent ocean structures beyond pointwise statistical optimization. These capabilities are delivered with a 62% reduction in GPU memory consumption and 70% reduction in FLOPs relative to structured-grid baselines. Our unstructured mesh representation establishes a generalizable paradigm for scalable data-driven oceanography.

## 1 Introduction

The ocean plays a fundamental role in Earth’s climate system by storing and transporting heat, freshwater, and momentum [1, 2], thereby influencing weather patterns, sea level variability, and marine ecosystems [3, 4]. Reliable ocean forecasting is essential for a broad range of scientific and societal applications, including climate monitoring and prediction, marine navigation and fisheries management [5], as well as early warning for extreme events such as storm surges and coastal flooding[6, 7]. In addition, ocean forecasts provide critical boundary and initial conditions for coupled Earth system models, and their accuracy directly afects the performance of weather and climate predictions[8].

State-of-the-art operational ocean forecasting relies on physics-based general circulation models (GCMs) such as NEMO [9], MOM6 [10], and HYCOM [11, 12], which discretize the governing primitive equations onto structured latitude–longitude or tripolar grids. While these models encode well-established oceanic dynamics, their computational demands are prohibitive: global eddy-resolving simulations at high resolution require large-scale HPC resources[13–15] and impose substantial wall-clock and throughput costs[16–18], limiting ensemble size, reanalysis throughput and real-time applicability [19, 20].

The success of deep learning in atmospheric forecasting has stimulated growing interest in datadriven alternatives for ocean prediction. Models such as FourCastNet [21], Pangu-Weather [22], and GraphCast [23] have demonstrated that neural networks trained on reanalysis data can produce skillful medium-range atmospheric forecasts at a fraction of the cost of numerical weather prediction. Inspired by these results, recent eforts have begun applying deep learning architectures to ocean state prediction [24–28]. However, directly transferring the atmospheric paradigm to the global ocean domain introduces several structural challenges that are absent or less severe in the atmosphere. The ocean domain is geometrically complex, multiply-connected, and characterized by sharp bathymetric gradients [29]; applying structured-grid neural architectures to irregular ocean boundaries may introduce systematic discretization errors and require costly preprocessing (masking, interpolation, or padding) which risks distorting the learned spatial relationships, particularly near coastlines and islands [30–32]. Furthermore, uniform structured tokenization allocates equal computational resources to dynamically quiescent regions (the abyssal plain) and energetically active regions (boundary currents, eddy fields), leading to unnecessary memory and compute overhead [33, 34]. Notably, few prior works have proposed a data-driven global ocean forecasting framework based on an unstructured spatial representation, which can naturally conform both to ocean geometry and to spatial heterogeneity [35–37].

To address the aforementioned limitations of structured-grid ocean forecasting paradigms, we develop OceanLight, a novel global ocean forecasting framework that integrates principled unstructured mesh tokenization with a graph neural network (GNN) backbone [38, 39]. Taking comprehensive 0.25° ocean state variables—including sea surface height, as well as temperature, salinity, and current velocities across 23 vertical depth levels—as inputs, OceanLight produces deterministic forecasts for lead times of 1–10 days, with individual models optimized for each forecasting horizon. The framework of OceanLight is demonstrated in figure 1.

The key innovations of this work are threefold. First, we design OceanLight as a novel climatologyaware unstructured mesh forecasting framework for global ocean prediction. Quantitative results confirm its superiority over traditional numerical models and the state-of-the-art GLONET baseline [26]. OceanLight matches numerical simulations in geostrophic balance consistency, leads all competitors in kinetic energy spectral fidelity, and accurately reconstructs coherent mesoscale eddies. Second, we introduce an unstructured mesh representation derived from global ocean climatology. This representation naturally adapts to complex ocean boundaries and dynamic heterogeneity, refining resolution in dynamically active regions and coarsening resolution in tranquil basins automatically. Third, by equipping the GNN framework with unstructured mesh representation, OceanLight achieves a 73% reduction in FLOPs and a 62% drop in GPU memory consumption relative to the original structured-grid GNN-based GraphCast, thus facilitating eficient training for global ocean forecasting on a single consumer-grade GPU.

![](images/90d7603c3d6e5e97cd9a9ec536498b44ddf7734fde19f6fe3937b4e00191be8b.jpg)  
Fig. 1: Overall architecture of OceanLight. OceanLight couples a geometry-adaptive unstructured mesh representation with a graph neural network (GNN) built on the encode–process–decode paradigm to forecast global ocean states. Input ocean variables at 0.25° resolution are encoded onto a compact unstructured mesh, propagated through an L-layer message-passing processor, and decoded back onto the original grid to produce the output ocean variables.(A) Overall forecasting framework. The encoder embeds grid nodes, mesh nodes, and grid-to-mesh edges via separate MLPs and passes information from grid to mesh through an Interaction Network. The processor applies L unshared message-passing layers, each performing edge and node updates followed by residual connection and LayerNorm on the mesh representation. The decoder embeds mesh-to-grid edges via an MLP and uses an Interaction Network to propagate the processed mesh representation back to the grid nodes, yielding the output ocean variables.(B) Unstructured mesh representation. Starting from ocean climatology data preprocessing, an initial partition of the global ocean is obtained via a region-growing algorithm guided by gradient-based seed selection. Large regions area are further refined through HDBSCAN clustering to produce the final refined partition. Each resulting region is treated as a mesh node, and mesh graph construction connects nodes based on geographical adjacency and multistep (k-hop) connectivity, yielding the final mesh graph used by the GNN backbone.(C) Interaction Network. For each directed edge connecting a sender and a receiver node (e.g., from grid to mesh), 1. Message Passing concatenates the sender, edge, and receiver features and passes them through an MLP to produce an updated edge feature (Edge ). 2. Node Update aggregates the updated edge features incoming to each receiver node (by summation) and combines the aggregated features with the original receiver node feature through a second MLP to produce the updated receiver node representation.

## 2 Results

## 2.1 Forecast accuracy across variables and lead times

We compare our model with GLONET and graphcast, using oceanbench [40] as inputs and make predictions across diferent lead times. We calculate RMSE and ACC to evaluate basic forecast accuracy across variables and lead times. The result of RMSE and ACC of diferent variables is shown in figure 2.

![](images/8408a2284abe76bf42b3bb20641597c6767656a549204cf963189f6541a77119.jpg)

B ACC of different models across lead time ranging from 1 to 10  
![](images/5810a6ec929da0c5bd259cc29ad53d741b151c9134532763bac3ae23e7a15552.jpg)  
Fig. 2: Forecast accuracy of ocean state variables across GLONET, OceanLight, Graph-Cast, and GLO12. Root-mean-square error (RMSE) and anomaly correlation coeficient (ACC) are evaluated for five key ocean variables as a function of forecast lead time (1–10 days). (A) RMSE as a function of lead time for sea surface height (top left), temperature (top middle), salinity (top right), zonal current (bottom left), and meridional current (bottom right). Lower RMSE indicates higher forecast accuracy. (B) ACC as a function of lead time for the same five variables, arranged in the same layout as in (a). Higher ACC indicates stronger correspondence between forecast and observed anomalies. Data are shown as mean values for each model across the forecast period of 2024.

Overall, our model achieves consistently lower RMSE and higher ACC than GLONET across all variables and lead times, while showing performance comparable to or slightly better than GraphCast in most cases. For sea surface height (SSH), our model obtains the lowest RMSE at nearly all forecast horizons and maintains competitive ACC values, especially at short- and medium-range lead times. In temperature prediction, our method significantly reduces RMSE compared with GLONET and GLO12 and achieves similar ACC performance to GraphCast, indicating better stability during longterm forecasting. For salinity, both RMSE and ACC results demonstrate that our model substantially outperforms GLONET, GLO12 and closely matches GraphCast, particularly at 1–7 day lead times. In the prediction of zonal and meridional currents, our model consistently achieves lower RMSE and higher ACC than GLONET and GLO12, while remaining competitive with GraphCast. Although the forecasting accuracy of all models gradually decreases as the lead time increases, our method shows a slower degradation trend, especially for dynamic variables such as ocean currents. These results demonstrate that the proposed model has strong forecasting capability and generalization performance on multiple ocean variables under the IV-TT evaluation framework.

We also conducted a depth-stratified evaluation of temperature and salinity using the IVTT metric and plotted their vertical profiles. The vertical profiles of temperature and salinity are shown in supplementary Figure S1. We have also implemented the IVTT evaluation of ocean temperature, salinity, current velocity, and sea surface height, broken down by season and region. The experimental results are presented in the figures in supplementary Figure S2-S3.

## 2.2 Increased dynamical consistency between geostrophic and surface flows

The geostrophic surface currents were diagnosed from the model-predicted sea surface height above geoid (SSH) via the geostrophic balance equations and compared with the model-predicted surface currents. Dynamical consistency between the two was quantified using the RMSE and correlation coeficient; lower RMSE and higher correlation coeficients indicate better geostrophic consistency. The geostrophic surface currents $( u _ { g }$ and $v _ { g } )$ were derived from the model-predicted sea surface height above geoid; the calculation details are given in the supplementary note 1. The comparison results are presented in figure 3.

![](images/9018bbcb970e6cc7e2d22979111e8fe037b08d49e91ded944a1e884b0f29d3ef.jpg)

![](images/0fe68fb621131f1327c4d90a98fb5d4b45c6230fe3d12a0de4dac626b04b7655.jpg)

![](images/ca065979a76dbbebf2bb13bfb23cf636287a1e101a7a11291ff3b702da78ca24.jpg)  
Fig. 3: Internal geostrophic consistency of GLONET, OceanLight, and GraphCast surface current forecasts. For each model, surface geostrophic currents are derived from the forecasted sea surface height(SSH) field and compared against the model’s directly forecasted surface zonal(u) and meridional (v) velocity components, as a function of forecast lead time (1-10 days). (A) Root-mean-square error (RMSE) between SSH-derived geostrophic surface currents and forecasted surface currents (Geo RMSE Surface) as a function of lead time. (B) Correlation coeficient between SSH-derived geostrophic zonal current and forecasted zonal current (Geo U Corr Surface) as a function of lead time. (C) Correlation coeficient between SSH-derived geostrophic meridional current and forecasted meridional current (Geo V Corr Surface) as a function of lead time. These metrics quantify the degree to which each model’s forecasted surface velocity fields are dynamically consistent with its own forecasted SSH field under geostrophic balance.

The experimental results show that the our model achieves the lowest RMSE between the geostrophic currents and the predicted surface currents, indicating that the predicted dynamical height field is more physically reasonable. Since geostrophic currents are directly determined by the spatial gradients of sea surface height, the lower RMSE suggests that the proposed method produces more accurate SSH gradient structures and reconstructs more realistic ocean dynamical patterns, rather than merely reducing point-wise prediction errors.

In addition, our model achieves the highest correlation coeficients for both the zonal and meridional velocity components, demonstrating stronger consistency in the flow direction between the derived geostrophic currents and the predicted surface currents. In particular, the improvement in the meridional component (v) is especially significant, indicating that the model more efectively captures the two-dimensional spatial structure of SSH gradients and mesoscale ocean dynamical features.

These results demonstrate that the proposed model not only fits sea surface height fields more accurately, but also better preserves the underlying ocean dynamical consistency and geostrophic balance relationships.

## 2.3 Improved energy consistency and realistic kinetic energy distribution

To further evaluate the dynamical characteristics of the predicted ocean surface circulation, the total kinetic energy (KE), mean kinetic energy (MKE), and eddy kinetic energy (EKE) were calculated based on the predicted velocity components (u, v) of 0.49m, and the calculation formulas are provided in the supplementary note 5.

We use the AVISO dataset as the observational reference to calculate KE, MKE and EKE. The same quantities are also computed from the model-predicted surface velocity fields. To quantitatively evaluate the consistency between the model-derived and AVISO-derived energy fields, several statistical metrics are calculated, including the RMSE, variance ratio, amplitude ratio, and area integral ratio, with the computational formulas presented in the supplementary note 3. We demonstrate the KE results and its visualization in figure 4. The results of EKE, MKE are demonstrated in supplementary S14-S15.

![](images/9df974feca3b5525e7eba2773c7300085b057c783cad8356c31b2c1c893532d2.jpg)

![](images/191c2466c307934b4ffefe6992989707f90621f013038a5040ece6fc6c39b607.jpg)

A KE Forecast Metrics Comparison Across Models  
![](images/a98d00269642f709e95b87a09d700887b4de7697cfc76e169ff7f759316164bb.jpg)

![](images/21e18add6541d3678a527f901fa6c8b04bbae81ea72d22f051a7b0ab43c706af.jpg)

![](images/967fea2fca3fad98baee7bf5ab226a2a3637fe3028480de3acf11bf858afe57b.jpg)  
B Comparison of 10-day forecast sea surface kinetic energy  
Fig. 4: Comparison of model-derived and AVISO-derived surface kinetic energy fields across forecast lead times. For each model, kinetic energy (KE) is computed from the forecasted surface velocity fields $( \mathrm { K E } = \textstyle { \frac { 1 } { 2 } } ( u ^ { 2 } + v ^ { 2 } )$ , in $\mathrm { m } ^ { 2 } / \mathrm { s } ^ { 2 } )$ and compared against the corresponding energy field derived from AVISO altimetry. (A) KE forecast skill as a function of forecast lead time, evaluated using four metrics: root-mean-square error (RMSE, leftmost panel) and three ratio-based metrics (right three panels), namely |1 − Variance Ratio|, |1 − Amplitude Ratio|, and |1 − Area Integral Ratio|, expressed as absolute deviations from unity such that lower values indicate closer agreement with the AVISOderived reference field. Four models are compared: GLONET, OceanLight, GraphCast, and GLO12. (B) Spatial maps of 10-day-averaged sea surface kinetic energy from three models—OceanLight, GraphCast, and GLONET—initialized using the OceanBench input dataset. The displayed fields represent the mean of each model’s 10-day forecasts, and are compared against AVISO satellite altimetry-derived observations.

For RMSE, our model achieves the lowest errors across KE, MKE, and EKE, demonstrating substantially improved prediction accuracy compared with both GLONET and GraphCast. For the ratio-based metrics, including variance ratio, amplitude ratio, and area integral ratio, values closer to 1 indicate better agreement with the reference data. Variance ratio represents the magnitude o spatial variation in an energy field, a value smaller than 1 indicates overly smooth predictions, while a value larger than 1 suggests exaggerated spatial fluctuations. The amplitude ratio reflects whether the average energies of the two fields are matched, and the area integral ratio indicates whether the total energies of the two fields are consistent. The results show that our model achieves most of the ratio-based metrics closer to 1, indicating better agreement with the reference fields. In contrast, GLONET exhibits substantial overestimation of energy intensity and spatial variability, as all ratio-based metrics are significantly larger than 1. This suggests that GLONET tends to excessively amplify the geostrophic energy fields and produces overly exaggerated spatial fluctuations.

## 2.4 Evaluation of Mesoscale Eddy Detection

To evaluate the mesoscale eddy forecasting performance, the predicted Sea Surface Height above Geoid (SSH) was first converted into Sea Level Anomaly (SLA) by subtracting the Mean Dynamic Topography (MDT). Mesoscale eddies were then identified from the resulting SLA fields using the Py-Eddy-Tracker (PET) algorithm. As the reference, mesoscale eddies were extracted from the AVISO satellite-observed SLA using the same PET algorithm, ensuring a consistent eddy identification procedure for both forecasts and observations. For eddy-based verification, a forecasted eddy was matched to an observed eddy when either of the following criteria was satisfied: the core position of the forecasted eddy fell within the efective closed contour of the observed eddy, or the distance between the forecasted eddy core and the observed eddy core was less than or equal to the efective radius of the observed eddy. Based on the matched forecasted and observed eddies, four commonly used dichotomous verification metrics, namely Probability of Detection (POD), False Alarm Ratio (FAR), Bias, and Critical Success Index (CSI) [41], were calculated to comprehensively evaluate the detection capability, reliability, systematic prediction tendency, and overall forecasting skill of our model and the baseline methods. The calculation formulas are provided in supplementary note 4. Since the AVISO altimetry product used as the observational reference for eddy detection has a spatial resolution of 0.25°, the mesoscale eddy detection comparison is restricted to baselines at matching resolution, namely GLONET and GraphCast. GLO12 is excluded from this comparison due to its higher resolution of 1/12°, as GLO12 resolves finer-scale eddies that fall below what AVISO can detect, making direct comparison against the 0.25° reference inherently unfair to its true eddy-resolving skill.The results are shown in figure 5.

Western boundary current regions are among the most dynamically active areas of the global ocean. The Kuroshio Extension, Gulf Stream, and Brazil Current are characterized by strong currents, sharp sea surface height gradients, intense baroclinic instability, and vigorous air–sea interactions. These processes favor the frequent generation, propagation, and deformation of mesoscale eddies, making such regions rich in eddy activity but also particularly challenging for ocean forecasting. Therefore, in addition to global quantitative evaluation, we further conduct regional visualization and IVTT-based assessment [42] over these three representative western boundary current regions to examine whether the model can accurately capture mesoscale eddy structures and SLA evolution in highly energetic oceanic regimes.

Thus, mesoscale eddy visualizations were conducted in three major western boundary current regions, including the Kuroshio and its extension region, the Gulf Stream, and the Brazil Current. The initial condition was taken from the OceanBench dataset on 2 January 2024, and a 10-day lead-time forecast was performed. The AVISO observational data on 12 January 2024 were used as the reference state. Mesoscale eddies were identified from both the forecasted and observed SLA fields using the PET algorithm [43], and the corresponding detection results are shown in supplementary Figure S4-S6.

Forecast skills of mesoscale eddies by GLONET, OceanLight, and GraphCast

![](images/7ceff77fe9a048ebcb6f33fc873f11842b4273f55c73a381879f02dd162f007a.jpg)

![](images/382ec6b17c6a20d9793f8a19d676fa8bddccbfd5faa7681f0c6393546c4159fb.jpg)

![](images/37fe99ce42306e1c984b249fb9d0a8b4a5de1c144037869328bbb6251d8eaaee.jpg)

![](images/0e2d2e4ee4cc53de723bbca26dffcd6a0dc987b10af2d7e649fefd42a255e9fb.jpg)  
Fig. 5: Verification of forecasted mesoscale eddies against AVISO altimetry across forecast lead times. For each lead time, forecasted sea surface height (SSH) fields from three forecast systems were converted to sea level anomaly (SLA) by subtracting the mean dynamic topography (MDT), and mesoscale eddies were then identified from the resulting SLA fields using the py-eddy-tracker (PET) algorithm. The same detection procedure was applied to satellite-observed AVISO SLA fields, which serve as the reference for eddy identification. Forecast-detected eddies at each lead time were matched against the AVISO reference eddies, and four categorical verification metrics were computed as a function of lead time (1-10) for GLONET (blue circles), our model (OceanLight; orange squares) and GraphCast (green triangles). (A) Absolute deviation of the eddy-count bias ratio from unity, |1−Bias|, where Bias is the ratio of the number of forecast-detected eddies to the number of AVISO-detected eddies; values near zero indicate agreement in eddy abundance between forecast and observations. (B) Critical success index (CSI), which jointly penalizes missed and falsely detected eddies; higher values indicate better overall detection skill. (C) False alarm ratio (FAR), the fraction of forecast-detected eddies with no matching observed eddy; lower values are better. (D) Probability of detection (POD), the fraction of AVISO-observed eddies correctly recovered by the forecast; higher values are better.

## 2.5 Enabling eficient and scalable ocean forecasting with reduced computational demands

To ensure a fair comparison with GraphCast, our unstructured mesh-based graph neural network model adopts identical hyperparameter settings, i.e., an embedding size of 512 and blocks number of 16.

Table 1: Comparison of Nodes and Edges Between OceanLight and GraphCast
<table><tr><td>Model</td><td>Mesh nodes</td><td>Mesh edges</td><td>grid→mesh edges</td><td>mesh→grid edges</td></tr><tr><td>Graphcast</td><td>40,962</td><td>2,941,920</td><td>1,539,243</td><td>2,941,920</td></tr><tr><td>OceanLight</td><td>1212</td><td>4120</td><td>681,067</td><td>681,067</td></tr></table>

As shown in table 1, compared with GraphCast, our model reduces the numbers of mesh nodes and mesh edges by approximately 97.0% and 99.9%, respectively, and decreases the grid→mesh edges and mesh→grid edges by approximately 55.7% and 76.8%, respectively. The significant reduction in the number of edges and nodes greatly decreases both the FLOPs and the memory usage during model operation, as shown in table 2. Specifically, FLOPs are reduced by 73.0%, while Max Memory Reserved and Max Memory Allocated are reduced by 62.4% and 62.3%, respectively. The decrease in FLOPs also greatly reduces the time cost of model training, and the time decomposition of model training is illustrated in the figure 6.

Table 2: Comparison of FLOPs and max memory during training Between OceanLight and GraphCast
<table><tr><td>Model</td><td>Graphcast</td><td>OceanLight</td></tr><tr><td>FLOPs</td><td>27673.207G</td><td>7481.553G</td></tr><tr><td>Max Memory Reserved</td><td>77.70GB</td><td>29.19 GB</td></tr><tr><td>Max Memory Allocated</td><td>69.09GB</td><td>26.04GB</td></tr></table>

Compared to GraphCast, our model reduces the training time of the encoder, processor, decoder, and backpropagation by 40.3%, 92.2%, 61.4%, and 71.2%, respectively.

![](images/d33bb9a1b4e63cb0694a3a1cf862588cce4ab270450f0e9e58f62c1c1644108d.jpg)

Training Time Breakdown  
![](images/84313a8ed2b9835347445b37b748f5dc153502924e721548c4868059e4eaac19.jpg)

![](images/3545e10d87c6d4314b5b53bd4bd53b7a8d9bed1c753618bd1b0c5943499251b7.jpg)  
Fig. 6: Training time comparison between GraphCast and OceanLight. (A) Total training time per iteration, showing an overall 3.4× speedup for OceanLight compared to GraphCast. (B) Percomponent breakdown of the forward pass (Encoder, Processor, Decoder): OceanLight reduces Encoder time by 40.3%, Processor time by 92.2%, and Decoder time by 61.4%, with the Processor showing the largest relative reduction. (C) Backward pass time, which dominates the total training cost for both models; OceanLight reduces backward time by 71.2% compared to Graphcast.

## 3 Methods

This section presents the methodology for constructing an unstructured mesh representation for global ocean forecasting at 0.25° resolution using Graph Neural Networks (GNNs). Figure 1 illustrates the overall methodological framework.

## 3.1 Overall Forecasting Framework

The core architecture follows the established encode-process-decode paradigm used in GraphCast. Given an input ocean state at time t, the model predicts the ocean state at time $t + \Delta t$ . The key distinction lies in the graph topology: instead of uniform refinement to an icosahedral mesh, our model operates on unstructured mesh comprising 1,212 regions based on ocean climatology data. This results in a substantially sparser graph structure, with only 4,120 edges connecting mesh nodes, and 681,067 bidirectional edges linking grid and mesh nodes. Compared to GraphCast’s dense mesh, our unstructured representation reduces the number of mesh nodes and edges by over an order of magnitude, cutting GPU memory usage by over 50% while preserving—and even slightly improving—the model’s capacity to resolve regionally varying ocean dynamics. Based on the graph structure constructed from the above process, we use ${ \bf v } _ { G } , { \bf v } _ { M } , { \bf e } _ { G 2 M } , { \bf e } _ { M 2 G } ,$ and ${ \bf e } _ { M 2 M }$ to denote the features of grid nodes, mesh nodes, edges from grid to mesh, edges from mesh to grid, and edges from mesh to mesh, respectively. The encoder embeds grid nodes, mesh nodes, and grid-to-mesh edge features into a latent space using separate MLPs. Then it performs a gnn-based message-passing step from grid nodes to mesh nodes. In this step, the $0 . 2 5 ^ { \circ }$ ocean data is encoded into the unstructured mesh representation:

$$
\tilde { \mathbf { v } } _ { G } = \mathrm { M L P } ( \mathbf { v } _ { G } ) , \quad \tilde { \mathbf { v } } _ { M } = \mathrm { M L P } ( \mathbf { v } _ { M } ) , \quad \tilde { \mathbf { e } } _ { G 2 M } = \mathrm { M L P } ( \mathbf { e } _ { G 2 M } )
$$

$$
\begin{array} { c } { { { \bf { v } } _ { M } ^ { \prime } = \tilde { { \bf { v } } } _ { M } + \mathrm { { I n t e r a c t i o n N e t w o r k } } \left( { \tilde { \bf { v } } _ { G } } , \ { \tilde { \bf { v } } _ { M } } , \ { \tilde { \bf { e } } _ { G 2 M } } \right) } } \\ { { { \bf { v } } _ { G } ^ { \prime } = \tilde { \bf { v } } _ { G } + { \mathrm { M L P } } ( \tilde { \bf { v } } _ { G } ) } } \end{array}
$$

The processor consists of 16 unshared message-passing layers that operate on the mesh representation. Each layer updates edges and nodes sequentially, followed by residual connections for both nodes and edges. For layer $l = 1 , \ldots , L$ (unshared parameters), where l denotes the index of the current

processor layer among the L stacked message-passing layers:

$$
\mathbf { e } _ { M 2 M } ^ { ( l ) } , ~ \mathbf { v } _ { M } ^ { ( l ) } = \mathrm { E d g e U p d a t e } \mathcal { E } \mathrm { X N o d e U p d a t e } \left( \mathbf { v } _ { M } ^ { ( l - 1 ) } , ~ \mathbf { e } _ { M 2 M } ^ { ( l - 1 ) } \right)
$$

$$
\mathbf { v } _ { M } ^ { ( l ) } \gets \left( \mathbf { v } _ { M } ^ { ( l - 1 ) } + \mathbf { v } _ { M } ^ { ( l ) } \right) , \quad \mathbf { e } _ { M 2 M } ^ { ( l ) } \gets \mathbf { e } _ { M 2 M } ^ { ( l - 1 ) } + \mathbf { e } _ { M 2 M } ^ { ( l ) }
$$

$$
{ \bf v } _ { M } ^ { \mathrm { o u t } } = { \bf v } _ { M } ^ { ( L ) }
$$

The decoder uses MLPs to embed mesh-to-grid edge features and performs gnn-based message-passing step from mesh nodes back to grid nodes. MLPs are used for the message passing flow, converting the information contained in the meshes to grids and output the prediction:

$$
\tilde { \mathbf { e } } _ { M 2 G } = \mathrm { M L P } ( \mathbf { e } _ { M 2 G } )
$$

$$
\begin{array} { r } { { \bf v } _ { G } ^ { \prime \prime } = { \bf v } _ { G } ^ { \prime } + \mathrm { I n t e r a c t i o n N e t w o r k } \left( { \bf v } _ { M } ^ { \mathrm { o u t } } , ~ { \bf v } _ { G } ^ { \prime } , ~ \tilde { { \bf e } } _ { M 2 G } \right) } \\ { \hat { \bf v } _ { G } ^ { \mathrm { o u t } } = \mathrm { M L P } ( { \bf v } _ { G } ^ { \prime \prime } ) \qquad } \end{array}
$$

## 3.2 Data Preprocessing

We obtained monthly ocean climatological data from the Copernicus Marine Environment Monitoring Service (CMEMS). For each grid point, we extract a multi-dimensional feature vector comprising various oceanographic variables, including temperature, salinity, zonal and meridional ocean current velocity components at 23 discrete depth levels (i.e., 0.49m, 2.65m, 5.08m, 7.93m, 11.41m, 15.81m, 21.60m, 29.44m, 40.34m, 55.76m, 77.85m, 92.32m, 109.73m, 130.67m, 155.85m, 186.13m, 222.48m, 266.04m, 318.13m, 380.21m, 453.94m, 541.09m and 643.57m) and sea surface height. By averaging the climatological fields of all twelve months, we construct the annual mean ocean climatology. This dataset represents the long-term mean state of the ocean and reflects its stable background characteristics. The annual mean climatological dataset provides a robust representation of the ocean’s integrated mean state on long time scales, serving as the foundational dataset for subsequent calculation and ocean graph construction.

Global ocean data at high resolution contains fine-scale details that can introduce challenges for constructing a tractable and representative adaptive mesh. The high-resolution data may contain noise, and in regions such as land-sea boundaries or strong oceanic fronts, fine-scale spatial gradients can be excessively sharp, potentially leading to an overly fragmented adaptive mesh with an impractically large number of regions. To address these issues, we employ a hierarchical multi-resolution strategy that balances computational eficiency, noise reduction, and the preservation of meaningful spatial structure for mesh construction. We regridded the data from its original resolution to $\mathrm { ~ a ~ } 1 . 4 0 6 2 5 ^ { \circ } \times 1 . 4 0 6 2 5 ^ { \circ }$ latitude-longitude grid using the Climate Data Operators (CDO) [44] with bilinear interpolation (remapbil operator).

Denote the horizontal grids of 1.40625° ocean climatology as

$$
\Omega _ { c } = \{ ( i , j ) | i = 1 , \dots , N _ { l a t } ^ { c } , j = 1 , \dots , N _ { l o n } ^ { c } \}
$$

At each grid point $( i , j ) \in \Omega _ { c } .$ , the multivariate climatological state is represented by a feature vector $\mathbf { x } _ { i , j } \in \mathbb { R } ^ { d }$ which includes temperature, salinity, zonal and meridional velocities across multiple depth levels, as well as sea surface height. To extract dominant mode of variability, principal component analysis (PCA) [45, 46] is applied to the multivariate feature vectors, reducing it to a one-dimensional principal component. Formally, the reduced scalar value at each grid point is given by

$$
z _ { i , j } = \mathcal { P } ( \mathbf { x } _ { i , j } ) , ~ z _ { i , j } \in \mathbb { R }
$$

where $\mathcal { P } ( \cdot )$ denotes projection onto the first principal component via PCA. The resulting scalar field is

$$
Z = \{ z _ { i , j } \} \in \mathbb { R } ^ { N _ { l a t } ^ { c } \times N _ { l o n } ^ { c } }
$$

Then the spatial gradient of Z is computed using the sobel operator [47]. Let ${ \bf K } _ { x }$ and $\mathbf { K } _ { y }$ denote the standard Sobel kernels in the zonal and meridional directions respectively, which are defined as:

$$
\mathbf { K } _ { x } = { \left[ \begin{array} { l l } { - 1 \mathrm { ~ 0 ~ } 1 } \\ { - 2 \mathrm { ~ 0 ~ } 2 } \\ { - 1 \mathrm { ~ 0 ~ } 1 } \end{array} \right] } \qquad \mathbf { K } _ { y } = { \left[ \begin{array} { l l l } { - 1 \mathrm { ~ - 2 ~ } - 1 } \\ { 0 } & { 0 } & { 0 } \\ { 1 } & { 2 } & { 1 } \end{array} \right] }
$$

The discrete gradients at each grid point (i, j) are calculated as:

$$
G _ { x } ( i , j ) = ( { \bf K } _ { x } * Z ) ( i , j ) = \sum _ { p = - 1 } ^ { 1 } \sum _ { q = - 1 } ^ { 1 } K _ { x } ( p , q ) Z ( i + p , j + q )
$$

$$
G _ { y } ( i , j ) = ( { \bf K } _ { y } * Z ) ( i , j ) = \sum _ { p = - 1 } ^ { 1 } \sum _ { q = - 1 } ^ { 1 } K _ { y } ( p , q ) Z ( i + p , j + q )
$$

The gradient magnitude is then given by

$$
G ( i , j ) = \sqrt { G _ { x } ( i , j ) ^ { 2 } + G _ { y } ( i , j ) ^ { 2 } } ,
$$

This resulting in the gradient map $G \in \mathbb { R } ^ { N _ { l a t } ^ { c } \times N _ { l o n } ^ { c } }$ , and the gradient field is used exclusively to guide the selection of seed points in the subsequent region-growing procedure.

## 3.3 Region Growing Partition

Inspired by work of adaptive multi-granularity graph representation [48], we first adopt the region growing method for initial region delineation. For any pair of adjacent grid points $( i , j )$ and (m, n), the feature-space diference is defined as

$$
D ( ( i , j ) , ( m , n ) ) = \| \mathbf { x } _ { i , j } - \mathbf { x } _ { m , n } \| _ { 2 }
$$

Let $\mathcal { N } ( i , j )$ denote the 8-connected neighborhood of $( i , j ) , \tau$ denote the threshold as the domain is partitioned into spatially connected regions using the threshold-based region-growing algorithm. Let $\Omega \subseteq \Omega _ { u }$ denote the set of unlabeled grid points. At each iteration, a new region is initialized from the point with the minimum gradient magnitude:

$$
( i _ { k } , j _ { k } ) = \arg \operatorname* { m i n } _ { ( i , j ) \in \Omega _ { u } } G ( i , j ) .
$$

Starting from this seed, the region $R _ { k }$ is expanded by recursively including neighboring points that satisfy the similarity criterion:

$$
D ( ( i , j ) , ( m , n ) ) \leq \tau , \quad ( m , n ) \in \mathcal { N } ( i _ { k } , j _ { k } ) .
$$

The expansion is implemented using depth-first traversal until no further points can be added. This process is repeated until all grid points are assigned, yielding a partition

$$
\Omega _ { c } = \bigcup _ { k = 1 } ^ { K } R _ { k } , \qquad R _ { k } \cap R _ { \ell } = \emptyset ( k \neq \ell ) .
$$

While the region growing algorithm preliminary partitioning method efectively handles regions such as land-sea boundaries and areas with steep gradients, it still generates large mesh regions that exceed over 20% of the total ocean grid points. Such large regions are problematic because they aggregate diverse oceanic features into a single mesh node, reducing the model’s ability to resolve spatial heterogeneity and potentially degrading forecast skill.

## 3.4 HDBSCAN Clustering

To further resolve large-scale structures, regions occupying more than 20% of the domain are refined using HDBSCAN clustering. For each region $R _ { k }$ , define its area fraction:

$$
\alpha _ { k } = \frac { | R _ { k } | } { | \Omega _ { c } | } .
$$

Regions satisfying $\alpha _ { k } > 0 . 2 0$ are selected for refinement. For each such region, feature vectors $\mathbf { x } _ { i , j }$ in the region $R _ { k }$ forms a sample set

$$
X _ { k } = \{ \mathbf { x } _ { i , j } \mid ( i , j ) \in R _ { k } \}
$$

HDBSCAN partitions $X _ { k }$ into clusters, inducing a decomposition $\textstyle R _ { k } = \bigcup _ { l = 1 } ^ { L } R _ { k , l }$ , where each $R _ { k , l }$ denotes a refined subregion. Since HDBSCAN operates in feature space, a given cluster $R _ { k , l }$ may consist of multiple geographically disconnected components. To ensure spatial consistency, each cluster is further decomposed into spatially connected subregions based on grid adjacency. Specifically, for each $R _ { k , l } ,$ we define its connected components as

$$
R _ { k , l } \longrightarrow \{ R _ { k , l } ^ { ( 1 ) } , R _ { k , l } ^ { ( 2 ) } , \ldots \} ,
$$

where each $R _ { k , l } ^ { ( \ell ) }$ is a maximal connected subset under the neighborhood relation $\mathcal { N } ( i , j )$ . These connected components are treated as independent regions in the final segmentation, and subclusters replace the original region, yielding a refined segmentation of the global ocean.

## 3.5 Mesh Graph Construction

Following the segmentation, each region $R _ { k }$ is treated as a mesh node. The set of mesh nodes is defined as

$$
{ \mathcal { V } } _ { M } = \{ R _ { 1 } , R _ { 2 } , \ldots , R _ { k } \}
$$

Each mesh node $R _ { k }$ is associated with a representative location in the spatial domain, defined as the centroid of the region:

$$
( \bar { i } _ { k } , \bar { j } _ { k } ) = \frac { 1 } { | R _ { k } | } \sum _ { ( i , j ) \in R _ { k } } ( i , j ) .
$$

Two meshes $R _ { k }$ and $R _ { \ell }$ are connected if they share at least one pair of neighboring grid points:

$$
( R _ { k } , R _ { \ell } ) \in \mathcal { E } _ { 1 } \ \Longleftrightarrow \ \exists ( i , j ) \in R _ { k } , ( m , n ) \in R _ { \ell } \ \mathrm { s . t . } \ ( m , n ) \in \mathcal { N } ( i , j ) .
$$

Here, $\mathcal { E } _ { 1 }$ denotes the set of first-order geographical adjacency edges. To capture higher-order spatial relationships, edges are further extended based on multi-step connectivity. Let

$$
\mathcal { G } _ { 1 } = ( \mathcal { V } _ { M } , \mathcal { E } _ { 1 } )
$$

denote the first-order adjacency graph. For a given integer $k \geq 1$ , define

$$
\xi _ { M } = \{ ( R _ { i } , R _ { j } ) \mid \mathrm { d i s t } _ { \mathcal { G } _ { 1 } } ( R _ { i } , R _ { j } ) \leq k , R _ { i } \neq R _ { j } \} ,
$$

where $\mathrm { d i s t } _ { \mathcal { G } _ { 1 } }$ is the shortest-path distance. In this study, we set $k = 2$ . The final mesh graph is defined as

$$
\mathcal { G } _ { M } = ( \nu _ { M } , \mathcal { E } _ { M } )
$$

where $\mathcal { E } _ { M }$ includes both first-order and higher-order (k-hop) spatial connectivity.

## 3.6 Interaction Network

Let

$$
\Omega _ { f } = \{ ( u , v ) \ | \ u = 1 , \ldots , N _ { l a t } ^ { f } , v = 1 , \ldots , N _ { l o n } ^ { f } \}
$$

denote the fine-resolution spatial grid at $0 . 2 5 ^ { \circ }$ , on which the subsequent gnn forecasting model is applied. The set of fine-grid nodes is defined as

$$
\mathcal { V } _ { G } = \Omega _ { f }
$$

Since the mesh partition is obtained on the coarse grid, each 0.25°grid point is assigned to a mesh region based on its spatial location. Specifically, a 0.25°grid node $( u , v )$ is associated with region $R _ { k }$ if its geographical coordinates fall within the spatial extent of $R _ { k }$ . Let

$$
\phi : \Omega _ { f } \to \mathcal { V } _ { M }
$$

denote this assignment, where $\phi ( u , v ) = R _ { k }$ if the grid point $( u , v )$ lies within the spatial domain covered by region $R _ { k }$ . The grid–mesh edge set is then defined as

$$
\mathcal { E } _ { G M } = \{ ( ( u , v ) , R _ { k } ) , ( R _ { k } , ( u , v ) ) ~ | ~ \phi ( u , v ) = R _ { k } \} .
$$

Thus, for each 0.25°grid node $( u , v )$ , two directed edges are constructed:

$$
( u , v )  R _ { k } , \qquad R _ { k }  ( u , v ) , \qquad { \mathrm { i f ~ } } \phi ( u , v ) = R _ { k }
$$

thereby enabling bidirectional information exchange between grid nodes and mesh nodes. The complete graph is defined as

$$
\mathcal { G } = ( \nu , \mathcal { E } ) ,
$$

where

$$
\mathcal { V } = \mathcal { V } _ { G } \cup \mathcal { V } _ { M } , \quad \mathcal { E } = \mathcal { E } _ { M } \cup \mathcal { E } _ { G M }
$$

Given the constructed edge set $\mathcal { E } _ { G M }$ , information is then propagated between nodes via an Interaction Network. For an arbitrary directed edge $( s , r ) \in \mathcal { E }$ , where $s \in \nu$ denotes the sender node and $r \in \mathcal V$ the receiver node, let $x _ { s }$ and $x _ { r }$ denote the corresponding node features and $e _ { s r }$ the edge feature. In the Interaction Network, the receiver node representation is updated through message passing, followed by aggregation and node update.

$\mathrm { A s }$ for the message passing step, for each edge $( s , r ) \in \mathcal { E } _ { : }$ , a message is computed by concatenating the sender feature, the receiver feature, and the edge feature, followed by a multi-layer perceptron:

$$
e _ { s r } ^ { \prime } = \mathrm { M L P } \left( \left[ \boldsymbol { x } _ { s } \parallel \boldsymbol { x } _ { r } \parallel e _ { s r } \right] \right) ,
$$

where $[ \cdot \| \cdot \| \cdot ]$ denotes feature concatenation.

As for the aggregation and node update step, for each receiver node r, the messages from all its incoming edges are aggregated by summation, and the receiver representation is updated via a second multi-layer perceptron:

$$
x _ { r } ^ { \prime } = \mathrm { M L P } \left( \left[ \boldsymbol { x } _ { r } \parallel \sum _ { s : ( s , r ) \in \mathcal { E } } e _ { s r } ^ { \prime } \right] \right) .
$$

This topological design of grid-mesh edges reduces the number of edges while preserving the essential edges for physical information transmission. Thus, the enhancement in computational eficiency is achieved without compromising forecasting performance.

## 4 Discussion

In this work, we present a lightweight ocean forecasting framework based on unstructured mesh representation, which is built according to ocean climatology. The construction of the unstructured mesh integrates the region growing method and the HDBSCAN clustering approach, enabling a relatively coarse mesh representation in stable and gentle open ocean areas, while employing a finer mesh in regions with drastic changes, such as land-sea boundaries. This mesh generation strategy efectively balances simulation accuracy and computational eficiency. Building on this eficient spatial representation, our framework achieves high forecasting accuracy and maintains high physical consistency with regard to geostrophic flow balance and kinetic energy fidelity, while significantly reducing memory consumption and time overhead compared to GraphCast and GLONET. The method also demonstrates the ability to identify and generate mesoscale eddies, indicating that the reduction in computational cost does not adversely afect its eddy-resolving capability.

The reduction in GPU memory usage during training allows our model to be trained on a single 40GB GPU, in contrast to GLONET, which requires multiple 40GB GPUs and a model-parallel training strategy. In addition to the reduced memory footprint, our model’s better time eficiency further contributes to a significant decrease in training cost.

Despite these advantages, OceanLight does have some limitations. The unstructured mesh used in this model is constructed based on climatological state, meaning that it primarily reflects long-term averaged oceanic conditions rather than capturing daily dynamical variability. Consequently, the mesh does not account for transient, short-term ocean processes that could be relevant for high-resolution forecasting. Future research will investigate methods for rapidly constructing unstructured meshes according to daily ocean conditions, thereby enabling more eficient and accurate ocean forecasting. Beyond the mesh construction strategy, another limitation lies in the forecasting paradigm itself: the current forecasting system is deterministic and does not provide probabilistic predictions. Leveraging the substantial reduction in computational cost aforded by the unstructured mesh representation, future work will further explore ensemble forecasting strategies, thereby enabling reliable uncertainty quantification for operational ocean prediction.

Looking beyond these immediate directions for improvement, we emphasize that the core contribution of this work extends beyond the specific GNN implementation evaluated here: the proposed unstructured mesh representation strategy constitutes an architecture-agnostic spatial representation for ocean state forecasting. Because the unstructured mesh is inherently sparse and irregular, each cell-level token interacts with only a geometrically local neighborhood, making the representation directly compatible with sparse transformer architectures [49, 50] and other attention-based models that exploit irregular token connectivity. This decoupling of spatial discretization from the model backbone establishes a generalizable paradigm: future work may substitute the GNN with sparse self-attention mechanisms, neural operators, or hybrid physics–data-driven architectures [51–53] without modifying the underlying mesh representation. By framing the ocean state as an unstructured token graph rather than a structured image, this approach reconciles the computational eficiency demonstrated in our experiments with the physical fidelity—captured through eddy-resolving skill and dynamical consistency—required for scientific credibility, opening a new direction for scalable, physics-aware data-driven oceanography.

## 5 Data availability

We download reanalysis data from the oficial website of Copernicus Marine Service at https://data. marine.copernicus.eu/product/GLOBAL MULTIYEAR PHY 001 030/download, and the Aviso satellite altimetry observations at https://data.marine.copernicus.eu/product/SEALEVEL GLO PHY CLIMATE L4 MY 008 057/download. We download input datasets for OceanBench challenger evaluation following the document of Oceanbench at https://oceanbench.readthedocs.io/en/latest/ input-datasets-for-oceanbench-challenger-evaluation.html.

## 6 Code availability

The mesh construction code is available at https://github.com/Rowena-929/OceanLight. The GNN-based forecasting model is implemented using an open-source PyTorch reimplementation of GraphCast, available at https://github.com/openclimatefix/graph weather. Our complete codebase, which integrates the proposed unstructured mesh representation with the above GraphCast-style architecture, is hosted at https://github.com/Rowena-929/OceanLight.

## Acknowledgements

This research is partially supported by National Key R&D Program of China (2024YFC3109200), National Natural Science Foundation of China (Grant No. 62372460), Hunan Provincial Natural Science Foundation of China (2024JJ4042), and the science and technology innovation Program of Hunan Province (2024RC3134).

## Author Contributions Statement

X.W and W.W designed the project. W.W performed model training. W.W and Q.Y.M performed model evaluation. W.W, J.X.Z, X.W., H.Z.L. and J.Q.S. wrote and revised the manuscript.

## Competing interests

The authors declare no competing interests.

## References

[1] Schuckmann, K., Mini\`ere, A., Gues, F., Gues, F., et al.: Heat stored in the earth system 1960–2020: where does the energy go? Earth System Science Data, 1675–1709 (2023)

[2] Cheng, L. LJ (Cheng, Abraham, J. J (Abraham, Hausfather, Z. Z (Hausfather, Trenberth, K.E.. KE (Trenberth: How fast are the oceans warming? Science, 128–129 (2019)

[3] Belyaev, K., Kuleshov, A.A., Tuchkova, N.: Estimation of the meridional heat and mass transport in the south atlantic by using the joint atmosphere and ocean circulation model with data assimilation and visualization facilities(article). Scientific Visualization, 119–138 (2019)

[4] Le Traon, P.Y., Reppucci, A., Fanjul, A., et al.: From observation to information and users: The copernicus marine service perspective(review). Frontiers in Marine Science (2019)

[5] Tommasi, D., Stock, C.A., Hobday, A.J., Methot, R., Kaplan, I.C., Eveson, J.P., Holsman, K., Miller, T.J., Gaichas, S., Gehlen, M., et al.: Managing living marine resources in a dynamic environment: the role of seasonal to decadal climate forecasts. Progress in Oceanography 152, 15–49 (2017)

[6] Ferreira\*, O.: A review of early warning systems for storm-induced coastal flooding and erosion on wave-dominated open coasts. Cambridge Prisms: Coastal Futures, 7 (2026)

[7] Le Traon, P.Y., Reppucci, A., Alvarez Fanjul, E., Aouf, L., Behrens, A., Belmonte, M., Bentamy, A., Bertino, L., Brando, V.E., Kreiner, M.B., et al.: From observation to information and users: The copernicus marine service perspective. Frontiers in marine science 6, 234 (2019)

[8] Tonani, M., Balmaseda, M., Bertino, L., Blockley, E., Brassington, G., Davidson, F., Drillet, Y., Hogan, P., Kuragano, T., Lee, T., et al.: Status and future of global and regional ocean prediction systems. Journal of Operational Oceanography 8(sup2), 201–220 (2015)

[9] Madec, G., NEMO System Team: NEMO Ocean Engine Reference Manual. (2024). https://doi. org/10.5281/zenodo.1464816 . https://doi.org/10.5281/zenodo.1464816

[10] Adcroft, A., Anderson, W., Balaji, et al.: The gfdl global ocean and sea ice model om4.0: Model description and simulation features. Journal of Advances in Modeling Earth Systems 11(10), 3167–3211 (2019) https://doi.org/10.1029/2019MS001726 https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2019MS001726

[11] Bleck, R.: An oceanic general circulation model framed in hybrid isopycnic-cartesian coordinates. Ocean modelling 4(1), 55–88 (2002)

[12] Chassignet, E.P.e.m.e., Smith, L.T., Halliwell, G.R., Bleck, R.: North atlantic simulations with the hybrid coordinate ocean model (hycom): Impact of the vertical coordinate choice, reference pressure, and thermobaricity. Journal of Physical Oceanography, 2504–2526 (2003)

[13] Smith, R.D., Maltrud, M.E., Bryan, F.O., Hecht, M.W.: Numerical simulation of the north atlantic ocean at 1/10°. Journal of Physical Oceanography 30(7), 1532–1561 (2000) https://doi. org/10.1175/1520-0485(2000)030⟨1532:NSOTNA⟩2.0.CO;2

[14] Hurlburt, H.E., Chassignet, E.P., Cummings, J.A., Kara, A.B., Metzger, E.J., Shriver, J.F., Smedstad, O.M., Wallcraft, A.J., Barron, C.N.: Eddy-Resolving Global Ocean Prediction, pp. 353–381. American Geophysical Union (AGU), ??? (2008). https://doi.org/10.1029/177GM21 . https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/177GM21

[15] Hewitt, H.T., Bell, M.J., Chassignet, E.P., Czaja, A., Ferreira, D., Grifies, S.M., Hyder, P.,

McClean, J.L., New, A.L., Roberts, M.J.: Will high-resolution global ocean models benefit coupled predictions on short-range to climate timescales? Ocean Modelling 120, 120–136 (2017) https://doi.org/10.1016/j.ocemod.2017.11.002

[16] Koldunov, N., Wang, Q., Sidorenko, D., Horvat, C., Fox-Kemper, B., Chassignet, E.P., Bozec, A., Xu, X., Yeager, S.G., Castruccio, F., Kim, W.M., Danabasoglu, G., Sein, D.V., Liu, H., Lin, P., Li, Y.: Impact of horizontal resolution on global ocean–sea ice model simulations based on the experimental protocols of the ocean model intercomparison project phase 2 (omip-2). Geoscientific Model Development (2020)

[17] Chassignet, E.P., Xu, X.: On the importance of high-resolution in large-scale ocean models. Advances in Atmospheric Sciences 38(10), 1621–1634 (2021)

[18] Seo, H., O’Neill, L.W., Bourassa, M.A., Czaja, A., Drushka, K., Edson, J.B., Fox-Kemper, B., Frenger, I., Gille, S.T., Kirtman, B.P., et al.: Ocean mesoscale and frontal-scale ocean– atmosphere interactions and influence on large-scale climate: A review. Journal of climate 36(7), 1981–2013 (2023)

[19] Jean-Michel, L., Eric, G., Romain, B.-B., Gilles, G., Ang´elique, M., Marie, D., Cl´ement, B., Mathieu, H., Olivier, L.G., Charly, R., et al.: The copernicus global 1/12 oceanic and sea ice glorys12 reanalysis. Frontiers in Earth Science 9, 698876 (2021)

[20] Lellouche, J.-M., Le Galloudec, O., Dr´evillon, M., R´egnier, C., Greiner, E., Garric, G., Ferry, N., Desportes, C., Testut, C.-E., Bricaud, C., et al.: Evaluation of global monitoring and forecasting systems at mercator oc´ean. Ocean Science 9(1), 57–81 (2013)

[21] Kurth, T., Subramanian, S., Harrington, P., Pathak, J., Mardani, M., Hall, D., Miele, A., Kashinath, K., Anandkumar, A.: Fourcastnet: Accelerating global high-resolution weather forecasting using adaptive fourier neural operators (2022)

[22] Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., Tian, Q.: Accurate medium-range global weather forecasting with 3d neural networks. Nature, 533–538 (2023)

[23] Lam, R., Sanchez-Gonzalez, A., Willson, M., Wirnsberger, P., Fortunato, M., Alet, F., Ravuri, S., Ewalds, T., Eaton-Rosen, Z., Hu, W., et al.: Learning skillful medium-range global weather forecasting. Science 382(6677), 1416–1421 (2023)

[24] Wang, X., Wang, R., Hu, N., Wang, P., Huo, P., Wang, G., Wang, H., Wang, S., Zhu, J., Xu, J., Yin, J., Bao, S., Luo, C., Zu, Z., Han, Y., Zhang, W., Ren, K., Deng, K., Song, J.: Xihe: A data-driven model for global ocean eddy-resolving forecasting (2024)

[25] Cui, Y., Wu, R., Zhang, X., Zhu, Z., Liu, B., Shi, J., Chen, J., Liu, H., Zhou, S., Su, L., Jing, Z., An, H., Wu, L.: Forecasting the eddying ocean with a deep neural network. Nature communications, 2268 (2025)

[26] Aouni, A.E., Gaudel, Q., Regnier, C., VanGennip, S., Galloudec, O.L., Drevillon, M., Drillet, Y., Lellouche, J.: Glonet: Mercator’s end-to-end neural global ocean forecasting system. Journal of Geophysical Research: Machine Learning and Computation (2025)

[27] Yang, N., Wang, C., Zhao, M., Zhao, Z., Zheng, H., Zhang, B., Wang, J., Li, X.: LangYa: Revolutionizing Cross-Spatiotemporal Ocean Forecasting (2025). https://arxiv.org/abs/2412. 18097

[28] Han, Y., Jia, H., Wang, X., et al.: Oceanforecastbench: A benchmark data set for data-driven global ocean forecasting. Journal of Geophysical Research: Machine Learning and Computation 3(4), 2025–000838 (2026) https://doi.org/10.1029/2025JH000838

[29] Wang, Q., Danilov, S., Sidorenko, D.: The finite element sea ice-ocean model (fesom) v.1.4: formulation of an ocean general circulation model. Geoscientific Model Development, 663–693 (2014)

[30] Furner\*, R., Haynes, P., Jones, D.C., Munday, D., Paige, B., Shuckburgh, E.: The challenge of land in a neural network ocean model. Environmental Data Science, 40 (2025)

[31] Zhang, C., Perezhogin, P., Adcroft, A., Zanna, L.: Addressing out-of-sample issues in multilayer convolutional neural-network parameterization of mesoscale eddies applied near coastlines. Journal of Advances in Modeling Earth Systems, 2024–004819 (2025)

[32] Cuervo-Londono, G.A., Reyes, J.G., Rodriguez-Santana, A., Sanchez, J.: Voronoi-induced artifacts from grid-to-mesh coupling and bathymetry-aware meshes in graph neural networks for sea surface temperature forecasting. ELECTRONICS (2025)

[33] Ringler, T., Petersen, M., Higdon, R.L., Jacobsen, D., Jones, P.W., Maltrud, M.: A multiresolution approach to global ocean modeling(article). Ocean Modelling, 211–232 (2013)

[34] Hoch[1], K.E., Petersen[2], M.R., Brus[3], S.R., Engwirda[4], D., Roberts[5], A.F., Rosa[6], K.L., Wolfram[7], P.J.: Mpas ocean simulation quality for variable resolution north american coastal meshes. Journal of Advances in Modeling Earth Systems, 2019–001848 (2020)

[35] Hirabayashi, Y., Matusoka, D., Kimura, K.: Eddy-Resolving Global Ocean Forecasting with Multi-Scale Graph Neural Networks (2026). https://arxiv.org/abs/2601.12775

[36] Holmberg, D., Clementi, E., Epicoco, I., Roos, T.: Accurate mediterranean sea forecasting via graph-based deep learning. arXiv (2025)

[37] Shi, N., Xu, J., Wurster, S.W., Guo, H., Woodring, J., Van Roekel, L.P., Shen, H.-W.: Gnnsurrogate: A hierarchical and adaptive graph neural network for parameter space exploration of unstructured-mesh ocean simulations. IEEE Transactions on Visualization and Computer Graphics 28(6), 2301–2313 (2022)

[38] Gori, M., Monfardini, G., Scarselli, F.: A new model for learning in graph domains. In: Proceedings. 2005 IEEE International Joint Conference on Neural Networks, 2005., vol. 2, pp. 729–734 (2005). IEEE

[39] Scarselli, F., Gori, M., Tsoi, A.C., Hagenbuchner, M., Monfardini, G.: The graph neural network model. IEEE transactions on neural networks 20(1), 61–80 (2008)

[40] Aouni, A.E., Gaudel, Q., Johnson, J.E., Charly, R., Sommer, J.L., Gennip, Fablet, R., Drevillon, M., DRILLET, Y., Traon, P.Y.L.: Oceanbench: A benchmark for data-driven global ocean forecasting systems. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2026). https://openreview.net/forum?id=wZGe1Kqs8G

[41] Roebber, P.J.: Visualizing multiple measures of forecast quality. Weather and Forecasting, 601– 608 (2009)

[42] Ryan, A., Regnier, C., Divakaran, P., Spindler, T., Mehra, A., Smith, G., Davidson, F., Hernandez, F., Maksymczuk, J., Liu, Y.: Godae oceanview class 4 forecast verification framework: global ocean inter-comparison. Journal of Operational Oceanography 8(sup1), 98–111 (2015)

[43] Mason, E., Pascual, A., McWilliams, J.C.: A new sea surface height–based code for oceanic mesoscale eddy tracking. Journal of atmospheric and oceanic technology 31(5), 1181–1188 (2014)

[44] Schulzweida, U., Mueller, R., Kornblueh, L.: Climate Data Operators (CDO). Max Planck

Institute for Meteorology (2023). https://code.mpimet.mpg.de/projects/cdo

[45] Pearson, K.: Liii. on lines and planes of closest fit to systems of points in space. The London, Edinburgh, and Dublin philosophical magazine and journal of science 2(11), 559–572 (1901)

[46] Hotelling, H.: Analysis of a complex of statistical variables into principal components. Journal of educational psychology 24(6), 417 (1933)

[47] Seger, O.: Generalized and separable sobel operators. Machine vision for three-dimensional scenes, 347 (2012)

[48] Dai, D., Chen, F., Xia, S., Yang, L., Wang, G., Wang, G., Gao, X.: An adaptive multigranularity graph representation of image via granular-ball computing. IEEE Transactions on Image Processing 34, 2986–2999 (2025)

[49] Child, R., Gray, S., Radford, A., Sutskever, I.: Generating Long Sequences with Sparse Transformers (2019). https://arxiv.org/abs/1904.10509

[50] Price, I., Sanchez-Gonzalez, A., Alet, F., Andersson, T.R., El-Kadi, A., Masters, D., Ewalds, T., Stott, J., Mohamed, S., Battaglia, P., Lam, R., Willson, M.: Probabilistic weather forecasting with machine learning. Nature, 84–90 (2025)

[51] Raissi, M., Perdikaris, P., Karniadakis, G.E.: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational physics 378, 686–707 (2019)

[52] Shu, R., Zhong, X., Huang, Q., Gou, R., Gao, T., Li, H., Huang, X.: Hybridom: Hybrid physicsbased and data-driven global ocean modeling with eficient spatial downscaling (2026)

[53] Shu, R., Gou, R., Xiang, Y., Huang, X.: Ocean-E2E: Hybrid Physics-Based and Data-Driven Global Forecasting of Marine Heatwaves with End-to-End Neural Assimilation (2026). https: //openreview.net/forum?id=soeWR3TyfK

## Supplementary

## Supplementary Note 1-Geostrophic current calculation

The geostrophic surface currents $( u _ { g }$ and $v _ { g } )$ were derived from the model-predicted sea surface height above geoid (SSH, denoted here as $\eta )$ based on the geostrophic balance equations:

$$
u _ { g } = - \frac { g } { f } \frac { \partial \eta } { \partial y }
$$

$$
v _ { g } = \frac { g } { f } \frac { \partial \eta } { \partial x }
$$

where $g$ is the gravitational acceleration, $f$ is the Coriolis parameter, and $\partial \eta / \partial x$ and $\partial \eta / \partial y$ are the zonal and meridional gradients of SSH.

The geostrophic and model-predicted current speeds were calculated as

$$
S _ { g } = \sqrt { u _ { g } ^ { 2 } + v _ { g } ^ { 2 } }
$$

$$
S _ { m } = \sqrt { u ^ { 2 } + v ^ { 2 } }
$$

## Supplementary Note 2-Basic Evaluation Metrics

## RMSE (Root Mean Square Error)

We use RMSE to measure pointwise prediction accuracy. It is formulated as the following:

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( \hat { y } _ { i } - y _ { i } ) ^ { 2 } }
$$

where $N$ denotes the number of grids, $\hat { y } _ { i } ^ { \phantom { } }$ , denotes the predicted value, and $y _ { i }$ denotes the real value. ACC(Anomaly Correlation Coeficient)

We use ACC to assess the spatial pattern correlation with respect to climatological anomalies. It is formulated as the following:

$$
A C C = \frac { \sum _ { i = 1 } ^ { N } ( \hat { y } _ { i } - c _ { i } ) ( y _ { i } - c _ { i } ) } { \sqrt { \sum _ { i = 1 } ^ { N } ( \hat { y } _ { i } - c _ { i } ) ^ { 2 } } \sqrt { \sum _ { i = 1 } ^ { N } ( y _ { i } - c _ { i } ) ^ { 2 } } }
$$

where $N$ denotes the number of grids, ${ \hat { y } } _ { i } ,$ denotes the predicted value, and $y _ { i }$ denotes the real value, and $c _ { i }$ denotes the corresponding climatology value.

## Corr(Correlation Coeficient)

We use the Pearson correlation coeficient (Corr) to evaluate the consistency between the geostrophic currents derived from sea surface height and the predicted surface currents. The correlation coeficients are calculated separately for the zonal and meridional velocity components. It is formulated as follows:

$$
\mathrm { C o r r } = \frac { \sum _ { i = 1 } ^ { N } ( \hat { u } _ { i } - \overline { { \hat { u } } } ) ( u _ { i } - \overline { { u } } ) } { \sqrt { \sum _ { i = 1 } ^ { N } ( \hat { u } _ { i } - \overline { { \hat { u } } } ) ^ { 2 } } \sqrt { \sum _ { i = 1 } ^ { N } ( u _ { i } - \overline { { u } } ) ^ { 2 } } }
$$

where $N$ denotes the number of grids, $\hat { u } _ { i }$ denotes the geostrophic current component derived from sea surface height, $u _ { i }$ denotes the corresponding predicted surface current component, and $\bar { \hat { u } }$ and u denote their spatial mean values, respectively. The same formulation is applied to the meridional velocity component.

## Supplementary Note 3-Energy Evaluation Metrics

To quantitatively evaluate the consistency between the model-predicted and observation-derived ocean surface kinetic energy fields, three spatial ratio metrics are applied to each energy variable — total kinetic energy (KE), mean kinetic energy (MKE), and eddy kinetic energy (EKE). These metrics assess diferent aspects of energy agreement: the spatial variability amplitude, the mean energy intensity, and the total integrated energy magnitude, respectively. For a given energy field $X \in \{ \mathrm { K E } , \mathrm { M K E } , \mathrm { E K E } \}$ , the three metrics are defined as follows.

## Variance ratio

We use the variance ratio to evaluate whether the spatial variability amplitude of the model-predicted field is realistically reproduced. It is formulated as follows:

$$
\mathrm { V a r i a n c e ~ R a t i o } ( X ) = \frac { \mathrm { V a r } ( X _ { \mathrm { m o d e l } } ) } { \mathrm { V a r } ( X _ { \mathrm { o b s } } ) } = \frac { \displaystyle \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } \bigl ( X _ { \mathrm { m o d e l } } ( \mathbf { x } _ { i j } ) - \overline { { X } } _ { \mathrm { m o d e l } } \bigr ) ^ { 2 } } { \displaystyle \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } \bigl ( X _ { \mathrm { o b s } } ( \mathbf { x } _ { i j } ) - \overline { { X } } _ { \mathrm { o b s } } \bigr ) ^ { 2 } } ,
$$

where $\mathbf { x } _ { i j }$ denotes the grid point at latitude index i and longitude index $j ,$ with $i = 1 , \ldots , N$ and $j = 1 , \dots , M . \ X _ { \mathrm { m o d e l } } ( \mathbf { x } _ { i } )$ and $X _ { \mathrm { o b s } } ( \mathbf { x } _ { i } )$ denote the model-derived and observation-derived energy values at grid point $\mathbf { x } _ { i j }$ , and ${ \overline { { X } } } _ { \mathrm { m o d e l } } , { \overline { { X } } } _ { \mathrm { o b s } }$ are their respective spatial means. A variance ratio closer to 1 indicates better agreement in spatial variability amplitude between the model and observations. Values smaller than 1 indicate that the modelled field is overly smooth, while values larger than 1 indicate excessive variability.

## Amplitude ratio

We use the amplitude ratio to evaluate whether the mean intensity of the modeled field is consistent with the observations. It is formulated as follows:

$$
{ \mathrm { A m p l i t u d e ~ R a t i o } } ( X ) = { \frac { \overline { { X } } _ { \mathrm { m o d e l } } } { \overline { { X } } _ { \mathrm { o b s } } } } = { \frac { { \frac { 1 } { N \times M } } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } X _ { \mathrm { m o d e l } } ( \mathbf { x } _ { i j } ) } { { \frac { 1 } { N \times M } } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } X _ { \mathrm { o b s } } ( \mathbf { x } _ { i j } ) } } ,
$$

where ${ \overline { { X } } } _ { \mathrm { m o d e l } }$ and $\overline { { X } } _ { \mathrm { o b s } }$ denote the spatial mean values of the model-derived and observation-derived energy fields, respectively. An amplitude ratio closer to 1 indicates better agreement in the mean energy intensity between the model and observations.

## Area Integral Ratio

We use the area integral ratio to evaluate the consistency of the total integrated energy between the modeled and observed fields. It is formulated as follows:

$$
\mathrm { A r e a \ I n t e g r a l \ R a t i o } ( X ) = \frac { \displaystyle \int X _ { \mathrm { m o d e l } } \mathrm { d } A } { \displaystyle \int X _ { \mathrm { o b s } } \mathrm { d } A } = \frac { \displaystyle \sum _ { i , j } A _ { i j } X _ { \mathrm { m o d e l } } \bigl ( { \bf x } _ { i j } \bigr ) } { \displaystyle \sum _ { i , j } A _ { i j } X _ { \mathrm { o b s } } \bigl ( { \bf x } _ { i j } \bigr ) } ,
$$

where $A _ { i j }$ is the spherical area of grid cell (i, j), computed as

$$
A _ { i j } = R ^ { 2 } \Delta \lambda \left[ \sin \left( \phi _ { i } + \frac { \Delta \phi } { 2 } \right) - \sin \left( \phi _ { i } - \frac { \Delta \phi } { 2 } \right) \right] ,
$$

with R being the Earth’s radius, $\phi _ { i }$ the latitude of grid point i, and $\Delta \phi , \Delta \lambda$ the grid spacings in latitude and longitude, respectively. An area integral ratio closer to 1 indicates better agreement between the modeled and observed fields. An area integral ratio closer to 1 indicates better agreement in the total integrated energy between the model and observations.

## Supplementary Note 4-Mesoscale Eddies Evaluation Metrics

Based on the principles outlined in measures of forecast quality[41] for dichotomous (yes–no) forecasts, we employ Bias, Critical Success Index (CSI), False Alarm Ratio (FAR), and Probability of Detection (POD) to evaluate the forecast accuracy of mesoscale eddies. To evaluate the forecast quality of mesoscale eddies, we define three categorical outcomes based on the correspondence between forecasts and observations:

A = number of eddies correctly forecast (hit)

B = number of eddies forecast but not observed (false alarm),

C = number of eddies observed but not forecast (miss),

From these, the following standard quality measures are defined:

## Bias

$\textstyle \mathrm { B i a s } \ = \ { \frac { A + B } { A + C } }$ evaluates the systematic tendency of the forecast to overpredict or underpredict mesoscale eddy occurrence. A bias value of 1 represents an unbiased forecast, values greater than 1 indicate overforecasting, and values less than 1 indicate underforecasting.

## Critical Success Index (CSI)

$\textstyle \mathrm { C S I } = { \frac { A } { A + B + C } }$ provides an overall measure of forecast accuracy by jointly considering hits, misses, and false alarms while excluding correct negatives. A higher CSI indicates better overall forecast performance.

## False Alarm Ratio (FAR)

$\textstyle \mathrm { F A R } = { \frac { B } { A + B } } $ quantifies the proportion of forecasted mesoscale eddies that did not actually occur. A lower FAR indicates fewer false detections and therefore higher forecast reliability.

## Probability of Detection (POD)

$\textstyle \operatorname { P O D } = { \frac { A } { A + C } }$ is the fraction of observed eddies that are successfully predicted and measures the ability of the forecasting system to correctly identify observed mesoscale eddies. A higher POD indicates that a larger proportion of actual eddies are successfully detected, reflecting the forecast’s sensitivity to eddy occurrence.

## Supplementary Note 5-Energy Calculation

The total kinetic energy (KE) represents the overall intensity of ocean surface currents, including both the large-scale mean circulation and mesoscale eddy variability. Let $u _ { n } ( \mathbf { x } )$ and $v _ { n } ( \mathbf { x } )$ denote the zonal and meridional velocity components at grid point x on day n $( n = 1 , 2 , \ldots , N )$ . The time-mean kinetic energy per unit mass at each grid point is defined as:

$$
\mathrm { K E } ( \mathbf { x } ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \frac { 1 } { 2 } \left[ u _ { n } ^ { 2 } ( \mathbf { x } ) + v _ { n } ^ { 2 } ( \mathbf { x } ) \right]
$$

The mean kinetic energy (MKE) represents the strength of the large-scale mean circulation. It is calculated from the temporally averaged velocity field, and the mean velocity components are

$$
\overline { { u } } ( \mathbf { x } ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } u _ { n } ( \mathbf { x } )
$$

$$
\overline { { v } } ( \mathbf { x } ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } v _ { n } ( \mathbf { x } )
$$

Then the mean kinetic energy (MKE) is defined as

$$
\mathrm { M K E } ( \mathbf { x } ) = \frac { 1 } { 2 } ( \overline { { u } } ^ { 2 } ( \mathbf { x } ) + \overline { { v } } ^ { 2 } ( \mathbf { x } ) )
$$

where u and v denote the temporal mean zonal and meridional velocity components, respectively. The eddy kinetic energy (EKE) represents the intensity of mesoscale eddies and reflects mesoscale variability, flow instability, and energy cascade processes. It is defined as

$$
\mathrm { E K E } ( \mathbf { x } ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \frac { 1 } { 2 } \left[ ( u _ { n } ( \mathbf { x } ) - \overline { { u } } _ { n } ( \mathbf { x } ) ) ^ { 2 } + ( v _ { n } ( \mathbf { x } ) - \overline { { v } } _ { n } ( \mathbf { x } ) ) ^ { 2 } \right]
$$

where $u - \overline { { u } }$ and $v - { \overline { { v } } }$ represent the velocity anomalies relative to the temporal mean flow.

A Temperature profile  
![](images/96052a7d1493e8ec6662a9272a2788bd592f6c6bafa5c5b09c9b98c6fac89d22.jpg)  
Fig. S1: Vertical RMSE profiles of (A) temperature and (B) salinity as a function of depth for lead times of 1, 3, 5, 7, and 10 days, comparing GLONET, OceanLight(our model), GraphCast, and the GLO12 baseline. For each variable, RMSE generally increases near the surface and in the upper thermocline/halocline (0–150 m) before decreasing with depth, with the magnitude of nearsurface errors growing progressively with longer lead times. OceanLight and GraphCast track each other closely across all depths and lead times, consistently achieving lower RMSE than GLONET, particularly in the subsurface temperature maximum region (50–300 m), while remaining comparable to or better than the GLO12 baseline. Salinity RMSE profiles show a similar depth-dependent pattern across models, with smaller inter-model diferences than for temperature.

![](images/7493437fa0525fe5c4cad2788e9f1cb8e1c68b82e931abea9d7c4d6ad30b86f1.jpg)

![](images/ef1e400119980c5aafea447e21081fd28b733cbccfdba39d33b985b65af6e5a6.jpg)

![](images/0345ac01163a63937b090d9d2befd7a6daf2a3bed2b5952ad2d5ec032fe3e36f.jpg)

![](images/f192aad9186a142942f8d9902fa42b2814ee8414f826bbbc39e32a30d61dd199.jpg)

![](images/c3db61eb35d23c761ac0db261b6b6df9535cef768aaea4e12a1113e46b52aa9a.jpg)  
Fig. S2: Regional RMSE comparison for 10-day forecasts across five variables — temperature, salinity, zonal velocity (U), meridional velocity (V), and sea surface height — evaluated over eight ocean basins. Values are shown for the GLO12 numerical baseline (grey, top row) and three data-driven models: OceanLight (ours), GraphCast, and GLONET. Cell colors indicate RMSE relative to the GLO12 baseline within each variable panel, with blue denoting improvement and red denoting degradation. OceanLight achieves consistent RMSE reductions relative to GLO12 across nearly all regions and variables, with performance closely matching GraphCast and generally surpassing GLONET.

![](images/50cedf07712d274ce7158eceaf69910321b8dc8081fd0d280899967da125e052.jpg)  
Fig. S3: RMSE comparison of GLO12, OceanLight, GraphCast, and GLONET across four seasonal evaluation periods (s1–s4), for temperature, salinity, zonal/meridional current velocity, and sea surface height. Current velocity metrics are unavailable for s4 (N/A) due to missing observational data for that period.

![](images/bbfdde5668ab77f490f522177537a006330673694645adc576a62751cbca7122.jpg)  
Fig. S4: Mesoscale eddy visualization in the Kuroshio and its extension region. Forecasts are initialized from OceanBench of January 2nd, 2024 with a 10-day lead time, and validated against AVISO observations of January 12th, 2024. Eddy structures are detected using the PET algorithm applied consistently to both forecasted and observed SLA fields.

![](images/31130a785d854327c9e5c1c3483a2f72e770ef0feaff2fe182e10d75302aae3f.jpg)  
Fig. S5: Mesoscale eddy visualization in the Gulf Stream. Forecasts are initialized from OceanBench of January 2nd, 2024 with a 10-day lead time, and validated against AVISO observations of January 12th, 2024. Eddy structures are detected using the PET algorithm applied consistently to both forecasted and observed SLA fields.

![](images/e279ad7d6302376e383035e457f61323e6df828ca96388595e8a89ffd1d6b577.jpg)  
Fig. S6: Mesoscale eddy visualization in the Brazil Current. Forecasts are initialized from Ocean-Bench of January 2nd, 2024 with a 10-day lead time, and validated against AVISO observations of January 12th, 2024. Eddy structures are detected using the PET algorithm applied consistently to both forecasted and observed SLA fields.

## 10-day forecast vs. GLORYS Reanalysis Temperature Comparison

![](images/09d51833c6fb07099efe4379413ad105a00b47750c737f4149de502e80a3e4f5.jpg)  
Fig. S7: Global sea surface temperature fields averaged over 10-day forecasts for all of 2024, using OceanBench-derived initial conditions, for OceanLight (ours), GraphCast, and GLONET, compared against the corresponding time-averaged GLORYS reanalysis field.

![](images/f8ab8543a45539438a8d7805a3bfbb0b77abb61eb27f32b228fd294b84014ba4.jpg)  
Fig. S8: Global sea surface salinity fields averaged over 10-day forecasts for all of 2024, using OceanBench-derived initial conditions, for OceanLight (ours), GraphCast, and GLONET, compared against the corresponding time-averaged GLORYS reanalysis field.

## 10-day forecast vs. AVISO Zonal Current Velocity Comparison

![](images/a7fdcedb5d8a014e41ca38d43b9121fd6bf60458beb92e26c41ff06d095a12f8.jpg)  
Fig. S9: Global sea surface zonal current velocity fields averaged over 10-day forecasts for all of 2024, using OceanBench-derived initial conditions, for OceanLight (ours), GraphCast, and GLONET, compared against the corresponding time-averaged GLORYS reanalysis field.

![](images/f67088adb6ac06871d2a4c6b110be804d0c97729d83e079664057433bb34108e.jpg)  
Fig. S10: Global sea surface meridional current velocity fields averaged over 10-day forecasts for all of 2024, using OceanBench-derived initial conditions, for OceanLight (ours), GraphCast, and GLONET, compared against the corresponding time-averaged GLORYS reanalysis field.

## C 10-day forecast vs. GLORYS Reanalysis Sea Surface Height Comparison

![](images/3e644bb56930a7735f41ad39952c8018caaa6b121f50c30db89a5cc29d2fbaa7.jpg)  
Fig. S11: Global sea surface height fields averaged over 10-day forecasts for all of 2024, using OceanBench-derived initial conditions, for OceanLight (ours), GraphCast, and GLONET, compared against the corresponding time-averaged GLORYS reanalysis field.

## 10-day forecast vs. AVISO Velocity Magnitude Comparison

![](images/b0a947b4c5bde807d3dc24f5925ce1d56e97981c4bc2141c3fd6729d0e13c31b.jpg)  
Fig. S12: Comparison of 10-day forecast sea surface current speed (magnitude of the horizontal velocity, $| \mathbf { u } | = \sqrt { u ^ { 2 } + v ^ { 2 } } )$ from three models—OceanLight, GraphCast, and GLONET—against AVISO satellite altimetry-derived observations.

A Profile Observation Points Distribution (2024-01-12)  
![](images/7e888b8119fdd54f9913091365a502ed1631fc692c6b14a60980b81e7ddbb4bf.jpg)

B Current Observation Points Distribution (2024-01-12)  
![](images/b3acea04a43d8f6c5bce69e62796e2235d05cee3e87422b786a30a0c1f62c7c1.jpg)

C SLA Observation Points Distribution (2024-01-12)  
![](images/1e0b67a0367a6e0a24f2268525b3a974ba1251d83e51e4f7e0cc071a30b6fd40.jpg)  
Fig. S13: Spatial distribution of IV-TT Class4 reference observations used for global ocean forecast verification, obtained from the OceanPredict (formerly GODAE OceanView) Intercomparison and Validation Task Team (IV-TT) archive hosted on the NCI THREDDS server, for 2024-01-12. (A) Sub-surface profile observations of 682 points, sparsely distributed across the global ocean. (B) In situ current (velocity) observations of 26,821 points, showing denser sampling concentrated along subtropical gyres and western boundary current regions. (C) Along-track satellite altimeter sea level anomaly (SLA) observations of 109,677 points, exhibiting the characteristic crosshatch pattern of overlapping ascending/descending orbital passes with near-global coverage.

![](images/1856f304a850b5912f4d9fb3f2262b9cd08aab8a5b5221d087e13edc4641a98d.jpg)

MKE Forecast Metrics Comparison Across Models  
![](images/bde101586ee089c8c51c8ad62be7414cb9f1bbe7f307bc315f33413862282abd.jpg)

![](images/2914cea663ea81fa5993d722e23549ded25a87fdf43107f152897b306e8e278d.jpg)

![](images/aebfce089b90b3087ee6b161068cdb90460fa77ef951ee07d10cc0f4bb1ed06e.jpg)  
Fig. S14: Mean kinetic energy (MKE) are computed from the forecasted surface velocity fields and compared against the corresponding energy fields derived from AVISO altimetry, as a function of forecast lead time. For each energy component, four metrics are evaluated: root-mean-square error (RMSE), |1 − Variance Ratio|, |1 − Amplitude Ratio|, and |1 − Area Integral Ratio|, where the three ratio-based metrics are expressed as absolute deviations from unity such that lower values indicate closer agreement with the AVISO-derived reference fields.

![](images/1416e91c68eea6ba28d239e060ae9b4595c0fb66fd8711d05a521639a5d37bcf.jpg)

![](images/b3ae3de5b84a6d2e57ddb53a6b4d4a65b7da41363c22acc53162ae67fb643712.jpg)  
EKE Forecast Metrics Comparison Across Models

![](images/844cabe54ce19a8558dc6d682c414e98e9ff3e110b471dcd93232874408cbc5f.jpg)

![](images/a24dbb7ae0bbbd41942492aa1670dc1e1c4589126ae9ece5a13469c6c99e2a9f.jpg)  
Fig. S15: Eddy kinetic energy (EKE) are computed from the forecasted surface velocity fields and compared against the corresponding energy fields derived from AVISO altimetry, as a function of forecast lead time.
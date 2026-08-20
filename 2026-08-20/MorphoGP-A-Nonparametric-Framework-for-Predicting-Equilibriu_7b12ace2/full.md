# MorphoGP: A Nonparametric Framework for Predicting Equilibrium Beach Profiles Under Tidal Influence

Xi Wu<sup>∗</sup>, Yanqing Wei<sup>∗</sup>, Hang Yin, Pengze Li, Hongshuai Qi<sup>†</sup>, and Xi Chen<sup>†</sup>, Member, IEEE

Abstract—The prediction of equilibrium beach profiles under tidal influence is of fundamental importance for sustainable coastal development, informing shoreline protection strategies and managing coastal ecosystems under changing environmental conditions. However, it remains challenging due to the highly nonlinear interactions among wave, tide, and sedimentary processes. Traditional empirical and numerical models often exhibit limited adaptability across diverse coastal environments, with especially pronounced limitations in beach systems where tidal processes are important . To improve data-driven prediction under these conditions, this study proposes MorphoGP, a unified categoryspecific Gaussian process framework for predicting equilibrium beach profiles (EBPs) under tidal influence. The framework first introduces a ContourCluster model based on contrastive learning to classify tide-influenced beach morphologies automatically. Within each morphological category, a specialized Gaussian process expert learns statistical associations between environmental descriptors including waves, tides, and sediments and the beach profile’s shape. A Gating Net then integrates the outputs of all experts through a probabilistic weighting mechanism to produce the final prediction. It should be noted that MorphoGP is a data-driven predictive framework and does not explicitly resolve the full wave–tide–sediment transport dynamics. Instead, it incorporates physically relevant descriptors to support prediction and model interpretation. Evaluated on data from over 180 beach profiles from tide-influenced coasts along the Chinese coast, MorphoGP achieves improved predictive performance compared with conventional and deep learning models, reducing the test RMSE by about 59.3% compared with the best baseline and achieving a final RMSE of 0.297 m. Additionally, feature relevance analysis suggests that tidal parameters are strongly associated with equilibrium morphology, alongside wave and sediment characteristics. The proposed framework provides a physically informed, data-driven tool for equilibrium beachprofile prediction under tidal influence and coastal management,

while stronger process-level physical coupling remains an important direction for future development. The source code is available at https://github.com/Ch1hyaAnon/MorphoGP.git.

Index Terms—Tide-influenced beach, Equilibrium beach profile, Gaussian Process, Contrastive learning, Coastal morphodynamics.

## I. INTRODUCTION

QUILIBRIUM beach profiles [1] provide an idealized adjustment of beach morphology to hydrodynamic forcing and sediment transport processes [2], [3]. Their development is influenced by local factors such as wave climate, sediment supply, and geological-geomorphological setting. Because these factors vary in time and space, natural beaches are generally in continuous adjustment rather than fixed equilibrium. Therefore, equilibrium profiles should be regarded as theoretical constructs rather than universally applicable representations of beach morphology. Nevertheless, they are widely used to assess shoreline stability, evaluate beach nourishment performance, and examine coastal resilience under changing environmental conditions. Accurate prediction of equilibrium beach profiles is essential for understanding coastal morphodynamics and informing effective coastal management. While substantial progress has been made in modeling equilibrium profiles for wave-dominated beaches, the applicability of existing approaches to tide-influenced environments remains limited.

In particular, macrotidal beaches, as representative tideinfluenced systems, characterized by large tidal ranges and strong tidal currents [4], exhibit complex and highly variable morphological responses [5] that differ fundamentally from those observed on microtidal coasts. These environments are widespread, occurring along extensive coastlines in northwestern Europe, East Asia, and parts of Australia, and they are socioeconomically important because of their roles in coastal protection, ecosystem services, and human activities. Despite this importance, tidal effects have received comparatively less attention in equilibrium-profile research, leaving a critical gap between scientific understanding and practical modeling needs.

Traditional equilibrium beach profile (EBP) models are largely rooted in parametric or semi-empirical formulations that emphasize wave-driven processes. Classic theories, such as the Dean profile and its extensions, relate profile shape to wave energy dissipation and sediment characteristics, under the implicit assumption that wave forcing dominates morphological adjustment [1], [6]–[11]. These approaches have demonstrated considerable success for wave-dominated beaches, where tidal effects are relatively weak or can be treated as secondary influences.

However, the assumptions underlying conventional models are often violated when tidal processes become important [12], [13]. For example, on macrotidal beaches, tidal currents and wave–tide interactions can play leading roles in sediment transport and profile evolution, giving rise to multiple equilibrium-like states under similar wave conditions. The resulting profiles are often heterogeneous and multimodal, reflecting the coexistence of distinct morphological regimes controlled by tidal phase, sediment redistribution pathways, and hydrodynamic coupling. Consequently, singleregime parametric models struggle to represent the diversity and uncertainty inherent in beach profiles affected by tidal processes.

In response to the limitations of traditional formulations, machine learning (ML) and deep learning (DL) techniques have increasingly been applied to coastal morphodynamic modeling. Data-driven approaches, including neural networks, random forests, and convolutional architectures, have shown promising performance in predicting shoreline change, beach states, and profile evolution by learning complex nonlinear relationships directly from observations [14]–[16]. These methods reduce reliance on explicit physical parameterization and can leverage large, heterogeneous datasets.

Recent studies have also made important progress in improving the physical relevance and reliability of ML-based coastal predictions. For example, some studies have incorporated wave-, tide-, and sediment-related descriptors into data-driven models, while others have adopted probabilistic or Gaussian-process-based formulations to provide uncertaintyaware predictions [17]–[20]. Nevertheless, the scope of many existing ML- and DL-based studies is still often limited in ways that hinder their applicability to beach-profile prediction under tidal influence. Rather than being trained and validated across diverse coastal settings, a substantial portion of prior work focuses on time-series prediction using a small number of beaches and a few repeatedly surveyed profiles, which provides limited coverage of morphological variability and hydrodynamic forcing conditions. More importantly, existing uncertainty-aware or physically informed ML models have not fully addressed the regime heterogeneity of equilibrium beach profiles under tidal influence, where different morphological categories may respond differently to wave–tide–sediment forcing. When such models are transferred to tide-influenced coasts, they often face challenges in simultaneously representing morphological regimes, quantifying predictive uncertainty, and maintaining interpretability across heterogeneous sites. In particular, modeling all beach profiles affected by tidal forcing as samples from a single statistical distribution can obscure the presence of multiple morphological regimes driven by distinct wave–tide interactions, yielding overly smoothed profile estimates and reducing generalization across sites and environmental conditions. Although probabilistic ML methods can provide uncertainty estimates, a single global model may still produce poorly resolved uncertainty when the underlying beach profiles are drawn from multiple morphological regimes. Therefore, a category-specific probabilistic framework is needed to combine morphological regime identification, uncertainty-aware prediction, and feature relevance analysis for equilibrium beach-profile prediction under tidal influence.

These limitations highlight the need for a modeling framework that explicitly accounts for the heterogeneous, multiregime nature of equilibrium beach profiles under tidal in fluence while retaining predictive flexibility and uncertainty awareness. In this study, the term “multiple equilibrium states” refers to the coexistence of distinct, observed equilibriumprofile types under different combinations of wave, tide, and sediment conditions, rather than to a single universal profile form. In representative macrotidal settings, beaches may exhibit different quasi-stable profile geometries, such as relatively steep and narrow profiles, wide low-gradient intertidal profiles, or multi-slope profiles with pronounced cross-shore variability, depending on tidal range, wave exposure, sediment properties, and antecedent morphodynamic conditions. Rather than relying on a single global model or restrictive parametric assumptions, such a framework should be capable of identifying observed morphological regimes in the data, learning localized relationships within distinct regimes, and integrating these components into a coherent probabilistic prediction.

To this end, we propose a nonparametric framework for equilibrium beach-profile prediction under tidal influence. By combining morphology-driven clustering, category-specific Gaussian process experts, and probabilistic model averaging, the proposed approach directly addresses the challenges posed by tidal influence, regime diversity, and predictive uncertainty. Compared with conventional Gaussian process or ensemble models that typically learn a global input–output mapping, MorphoGP improves physical interpretability by conditioning the prediction on observed profile types. Each Gaussian pro cess expert is associated with a morphology-derived regime, and the probabilistic gating weights indicate the relative contribution of different regimes to the final prediction. This design allows the relationships between environmental forcing features and equilibrium-profile geometry to be interpreted within specific morphological contexts, instead of being averaged over all beach types.

The main contributions of this work are threefold:

1) We present the first large-scale, data-driven investigation for tide-influenced beaches along the Chinese coast, covering over 180 beaches, including macrotidal, mesotidal, and microtidal sites, and compile a unified dataset that supports systematic model development and evaluation. The dataset enables the observed diversity of tide-influenced equilibrium-profile types to be analyzed in a consistent cross-site framework.

2) To account for heterogeneous tide-influenced beach morphodynamics, we develop a category-specific nonparametric modeling framework based on Gaussian process experts, enabling flexible and localized equilibriumprofile prediction across distinct morphological regimes. Unlike a single global Gaussian process or a generic ensemble predictor, the proposed framework first identifies morphology-derived profile categories and then learns regime-conditioned forcing–profile relationships, thereby linking predictive components to physically interpretable beach-profile types.

3) We analyze the correlations between environmental forcing features and equilibrium beach-profile predictions. This analysis highlights the pronounced influence of tide-related factors in equilibrium beach-profile prediction, underscoring the limitations of wave-centric equilibrium-profile models. By conducting feature relevance analysis within a probabilistic, category-specific framework, MorphoGP provides more interpretable evidence on how wave-, tide-, and sediment-related descriptors are associated with different equilibrium-profile regimes.

The remainder of this article is organized as follows. Section II reviews related work; Section III presents the proposed MorphoGP framework; Section IV describes the experimental setup and results; and Section V concludes the paper.

## II. RELATED WORKS

## A. Progress in Equilibrium Beach Profile Research

The EBP not only reflects the long-term effects of hydrodynamics but also contains information on sediment properties and geomorphic erosion and deposition [21], [22]. Early empirical models, notably Bruun (1954) and Dean (1991), established power-law formulations to describe cross-shore profile geometry, primarily based on data from microtidal to mesotidal environments where wave action dominates. Subsequent refinements, such as the exponential model proposed by Bodge (1992) and the segmented equilibrium profile by Larson et al. (1999), further advanced the analytical representation of EBP under varying wave conditions. However, these classical frameworks largely overlooked the role of tides, particularly in macrotidal settings where tidal ranges exceed 4 m and exert comparable or greater morphodynamic influence on beaches.

In recognition of this gap, several studies began to incorporate tidal forcing into beach-state classification. Masselink and Short (1993) proposed the Ω–RTR framework, which combines the dimensionless fall velocity (Ω) with the relative tidal range (RTR) to extend the reflective–dissipative continuum to tidal environments. These morphodynamic studies provide an instructive qualitative taxonomy of beach-profile morphologies; however, they do not offer a quantitative expression for the EBP. More importantly, the predictive skill of the Ω–RTR approach tends to deteriorate in macrotidal settings, where strong tidal currents and wave–tide coupling promote tide-modulated, spatially heterogeneous morphodynamics that depart from the wave-dominated assumptions embedded in the model. Such limitations have been documented along the Shandong Peninsula and the western Guangdong coasts in China [23], [24]. Consistently, megatidal beaches in northern France (e.g., Merlimont) exhibit persistent intertidal bar– trough systems and pronounced tidal modulation of sediment transport [13], conditions that further complicate the mapping between Ω–RTR parameters and observed beach states. Recent field observations along the western coast of the Taiwan Strait likewise demonstrate that existing classification schemes struggle to capture terrace-type and dissipative-type profiles shaped by macrotidal forcing, underscoring the need for more flexible, data-driven approaches for quantitative EBP characterization in tide-influenced environments [12].

## B. Process-Based Numerical Beach Morphology Models

Process-based numerical models have been extensively developed to simulate coastal morphodynamics by explicitly resolving the physical processes governing wave–current interactions, sediment transport, and energy dissipation. DELFT3D [25] implemented and validated sediment transport formulations that allow for coupled simulation of hydrodynamics and sediment transport, making it one of the most widely used coastal process models. CSTM-ROMS [26] extended the Regional Ocean Modeling System (ROMS) [27] by implementing three-dimensional sediment transport modeling that accounts for multiple noncohesive sediment classes and dynamic bed morphology. XBeach [28] was developed to simulate nearshore hydrodynamics and morphological changes, providing insight into wave run-up, overwash, and dune erosion processes under storm conditions. COAST2D [29] was designed to model cross-shore and longshore sediment transport, enabling the prediction of shoreline evolution driven by combined wave and current. These process-based models have been widely applied to simulate coastal morphodynamics and beach profile evolution.

However, these process-based models are constrained by substantial computational demands, extensive parameterization needs, and reliance on detailed initial and boundary conditions [30]. More critically, their ability to accurately predict equilibrium profiles diminishes in environments with strong spatial heterogeneity and tidal action, especially along macrotidal coasts. The complexity of representing macrotidal ranges, sediment redistribution under combined wavecurrent actions, and varying sediment supplies hinders their generalizability across diverse and data-sparse settings. These limitations underscore the need for alternative approaches that can leverage growing observational datasets and offer greater adaptability. Data-driven machine learning methods present a compelling pathway, as they can learn complex morphodynamic relationships directly from data, require fewer explicit physical assumptions, and provide scalable predictions with computational efficiency—addressing key gaps left by traditional numerical models.

C. Machine Learning and Deep Learning for Beach Profile Prediction

With the advent of deep learning, data-driven methods have emerged as powerful alternatives to traditional modeling approaches for predicting beach profile evolution. Before the widespread adoption of deep learning–based spatiotemporal representation learning, Hsu et al. [31] introduced a twodimensional empirical eigenfunction model—an early machine learning approach—for modeling beach profile changes. Subsequently, Hashemi et al. [14] employed an Artificial Neural Network (ANN) to predict seasonal beach profile variations along Tremadoc Bay, using beach geometry, wind data, and local wave conditions as input features. Later, de Melo et al. [15] applied the Random Forest (RF) algorithm to simulate the seasonal morphological evolution of several Portuguese beaches, while Khan et al. [16] utilized Convolutional Neural Networks (CNNs) to predict long-term coastal changes driven by sea-level rise and wave conditions. Disdier et al. [32] developed an ANN model with three hidden layers to predict beach profiles from offshore wave spectra. Although this model achieved high predictive accuracy, its performance was constrained by reliance on synthetic training data generated from FUNWAVE-TVD modeling, which limited its real-world applicability. More recently, Kim et al. [33] proposed spatiotemporal Graph Neural Networks (GNNs) to improve the prediction of beach morphological evolution by capturing nonlinear relationships and spatiotemporal dependencies among multiple environmental variables, incorporating satellite optical imagery as an additional remote-sensing input.

Despite these advances, most existing studies have focused on time-series predictions for beaches without adequately considering tidal influence on beach processes, and several rely on synthetically generated datasets. These limitations are particularly evident in macrotidal settings, where tides can play an important role in shaping beach morphology. Furthermore, many machine learning models suffer from the “black-box” problem, meaning that they provide accurate predictions but lack interpretability and transparency regarding the relationships between environmental variables and beach profile characteristics.

In contrast, our approach seeks to bridge this gap by utilizing Gaussian Process (GP) [34], which provides a more interpretable and probabilistic solution to beach profile prediction. GPs can naturally account for uncertainty and provide insight into the importance of different environmental drivers, making them well suited for modeling the complex and highly nonlinear relationships in equilibrium beach-profile prediction under tidal influence.

## D. Unsupervised Morphological Classification and Regime-Specific Ensemble Modelling

Morphodynamic regime classification has long provided an important basis for interpreting beach-profile variability. A widely used framework for tide-influenced beaches is the Masselink–Short Ω–RTR model, which classifies beach states using the dimensionless fall velocity Ω and the relative tide range RTR [35]. In this framework, Ω reflects the relative effects of wave forcing and sediment settling, whereas RTR represents the relative importance of tidal range compared with breaking wave height. By combining these two physically meaningful parameters, the Masselink–Short framework links beach morphology to wave, tide, and sediment conditions and distinguishes morphodynamic states such as reflective, intermediate, dissipative, low-tide terrace, and tide-dominated beach types. It therefore provides an essential physical reference for tide-influenced beach classification.

In addition to parameter-based morphodynamic classifications, shape-based learning methods have provided another important route for profile and contour classification. Shapeletbased methods are particularly relevant because they identify discriminative local subsequences or sub-shapes that are representative of different classes [36]. By transforming each sequence into distances to a set of informative shapelets, the shapelet transform enables local, phase-independent shape comparison and can improve the interpretability of timeseries or profile classification [37], [38]. Later studies further extended this idea by learning shapelets directly from data through optimization-based formulations [39]. For coastal profile analysis, such methods are conceptually attractive because local geometric patterns, such as beachface steepening, terrace-like flattening, bar–trough structures, or changes in profile curvature, may serve as diagnostic indicators of different morphodynamic states.

Nevertheless, most shapelet-based methods are designed primarily for supervised classification or discriminative feature extraction, whereas the present study requires unsupervised discovery of morphology-derived regimes and their integration with downstream probabilistic EBP prediction. ContourCluster is therefore developed not simply to classify profiles, but to learn morphology-aware categories that can define regime-specific Gaussian process experts and support uncertainty-aware prediction across heterogeneous tideinfluenced beaches.

However, the Ω–RTR framework was originally designed as a conceptual morphodynamic classification scheme rather than as a morphology-driven representation for profile prediction. Its classes are primarily determined from a small set of hydrodynamic and sediment-related parameters and may not fully capture the geometric diversity of complete cross-shore equilibrium profiles, especially when beaches with similar Ω and RTR values exhibit different profile curvature, intertidal width, bar/terrace expression, or multi-slope structure. For equilibrium beach-profile prediction, this limitation is important because the subsequent regression model requires profile categories that are not only physically interpretable but also geometrically coherent in the observed profile space.

A growing body of work has explored unsupervised learning techniques to classify beach profiles based on their intrinsic morphological features, thereby making the strong heterogeneity between beach states explicit. Classical approaches include k-means clustering and principal component analysis (PCA), which group profiles according to low-dimensional shape descriptors. For example, Riazi [40] applied unsupervised clustering techniques to 916 subaerial beach profiles collected from six beaches along the Maine coast between 2005 and 2018 and successfully grouped them into multiple categories based on their shape characteristics. This study demonstrated the potential of data-driven methods to capture the inherent variability of beach morphologies beyond traditional parameter-based classifications.

Once distinct morphological regimes have been identified, it is natural to model each regime with a specialized predictor rather than enforcing a single global relationship for all profiles. In statistics and machine learning, this idea appears in various forms such as piecewise regression [41], latentclass or regime-switching models [42], and ensemble methods [43] in which several sub-models are trained on different parts of the input space and their predictions are combined through data-dependent weights. Such heterogeneity-aware frameworks have been shown to improve both predictive performance and interpretability in complex systems with multiple operating states.

![](images/814bd2d0cd02a51e864aea84a095974c1b3b70ce8ee4ca399a985f0528940973.jpg)  
Fig. 1. Overview of the proposed MorphoGP framework. Given environmental forcing descriptors $\mathbf { c } { } ~ = { } \ [ \mathrm { w a v e } { } ,$ , tide, sediment] and the cross-shore coordinate s, MorphoGP predicts the equilibrium beach profile $y ( s )$ . ContourCluster first groups observed profiles into K morphodynamic clusters. A set of cluster-specific Gaussian-process experts $\{ G _ { t } ( \mathbf { c } , s ) \} _ { t = 1 } ^ { K }$ models profile variability within each cluster, while a gating network $g _ { \Phi } ( \mathbf { c } )$ outputs mixture weights $\mathbf { q } = \mathrm { s o f t m a x } ( g _ { \Phi } ( \mathbf { c } ) )$ ). The final prediction is obtained by fusing the expert outputs via $\begin{array} { r } { \hat { y } ( s ) = \sum _ { t = 1 } ^ { K } q _ { t } \bar { G } _ { t } \big ( \mathbf { c } , s \big ) } \end{array}$ , yielding an uncertainty-aware profil estimate (inset).

For coastal morphodynamics, this perspective is particularly appealing: different beach states, including reflective, dissipative, low-tide terrace, and tide-dominated morphologies [35], are associated with distinct combinations of wave, tide, and sediment conditions and may obey different effective morphodynamic relationships. Nevertheless, a key unresolved question is whether a conceptual parameter-based classification such as the Masselink–Short Ω–RTR scheme provides sufficiently coherent categories for data-driven equilibriumprofile prediction, or whether profile-shape-based clustering can better represent the observed geometric structure of macrotidal EBPs. However, the combination of unsupervised morphology-based classification with category-specific probabilistic regression models has not yet been systematically explored for equilibrium beach-profile modeling, especially under macrotidal conditions. In this work, we address this gap by integrating a contour-based unsupervised classifier with a set of Gaussian Process models and a probabilistic weighting mechanism that selects and combines them according to the inferred morphodynamic category, enabling morphologyaware, uncertainty-quantified EBP prediction across heterogeneous tide-influenced beaches. To directly assess the novelty and utility of the clustering module, we further compare ContourCluster with the standard Masselink–Short Ω–RTR classification in terms of both morphological consistency and downstream prediction performance.

## III. METHODOLOGY

## A. Overview

Problem setup and notation. Let $\mathcal { D } = \{ ( \mathbf { c } _ { i } , \mathbf { y } _ { i } ) \} _ { i = 1 } ^ { N }$ denote a dataset of N surveyed equilibrium beach profiles. For profile $i , \mathbf { c } _ { i } \in \mathbb { R } ^ { d }$ collects the environmental drivers (wave, tide, and sediment descriptors), and $\mathbf { y } _ { i } ~ \in ~ \mathbb { R } ^ { M }$ is the corresponding profile sampled on a fixed cross-shore grid $\lbrace s _ { m } \rbrace _ { m = 1 } ^ { M }$ , i.e., $\mathbf { y } _ { i } = [ y _ { i } ( s _ { 1 } ) , \dots , y _ { i } ( s _ { M } ) ] ^ { \top }$ , where $y _ { i } ( s _ { m } )$ denotes bed elevation (or depth) at coordinate $s _ { m }$ under a consistent vertical datum. For notational simplicity, we drop sample indices and write y and c for generic profile and forcing vectors. Given a new forcing vector $\mathbf { c } _ { * } ,$ our goal is to predict the conditional distribution

$$
p ( \mathbf { y } \mid \mathbf { c } ) ,\tag{1}
$$

and to provide both a point estimate (e.g., posterior mean profile) and a coherent predictive uncertainty. In implementation, MorphoGP predicts each ordinate $y ( s _ { m } )$ using the augmented input $\mathbf { z } = [ \mathbf { c } , s _ { m } ]$ (thus $D = d + 1 )$ and assembles the full profile by evaluating the model for $m = 1 , \ldots , M$

This work introduces MorphoGP, a framework for predicting equilibrium beach profiles under tidal influence from key environmental drivers, including wave, tide, and sediment characteristics. As illustrated in Fig. 1, the framework consists of three components: (1) a ContourCluster module that assigns each profile to one of K morphology categories based on contour-derived shape features; (2) K categoryspecific Gaussian process (GP) experts trained to model $p ( y ( s ) \mid \mathbf { c } , s , \kappa = k )$ for each category, where $\kappa \in \{ 1 , \ldots , K \}$ denotes the morphology category; and (3) a gating network that maps c to input-dependent mixture weights and combines the expert predictive distributions into $p ( \mathbf { y } \mid \mathbf { c } )$ with uncertainty quantification. When encountering a new beach characterized by $\mathbf { c } _ { \ast }$ , MorphoGP evaluates the gating network to obtain expert weights and produces the final profile prediction by aggregating expert outputs across all $s _ { m }$

The following sections detail these components.

## B. ContourCluster Model

To establish objective, data-driven domains for the expert models, we categorize beach profiles based on morphological features derived from their cross-shore elevation sequences $\mathbf { y } _ { i } .$ The ContourCluster module comprises two stages: (i) a beachshapelet extractor that embeds profile subsequences into a discriminative representation space via contrastive learning and (ii) k-means clustering to obtain K morphology categories.

This work leverages shapelets [39], which are discriminative subsequences of time series that capture local patterns and serve as interpretable features for classification. As shown in Fig. 2, the highlighted subsequence illustrates a shapelet S, and its distance profile is obtained by sliding a window $w _ { u }$ across a reference series r. Before computing the distance,

![](images/9630452c1928dd84dc8d735221624e0a4a7ea3fd0cefe9fa5f9a943e7a321609.jpg)  
Fig. 2. Illustration of shapelet-based matching on a representative Low Tide Terrace equilibrium beach profile [35]. Top: a discriminative subsequence (shapelet) S highlighted on the reference profile $r ( x )$ . Middle: z-normalized Euclidean distance profile $d _ { u }$ obtained by sliding a window $w _ { u }$ of length L along r. Bottom: normalized similarity weights $\chi _ { u }$ computed via the softmin transform in (4), where larger $\chi _ { \tau }$ <sub>u</sub> indicates higher local morphological similarity.

both the shapelet and each window are z-normalized [44] to eliminate the influence of amplitude differences:

$$
S ^ { \prime } = \frac { S - \mathrm { m e a n } ( S ) } { \mathrm { s t d } ( S ) } , \quad w _ { u } ^ { \prime } = \frac { w _ { u } - \mathrm { m e a n } ( w _ { u } ) } { \mathrm { s t d } ( w _ { u } ) } .\tag{2}
$$

Let $S ^ { \prime } \in \mathbb { R } ^ { L }$ and $w _ { u } ^ { \prime } \in \mathbb { R } ^ { L }$ denote the z-normalized subsequences of length L. The normalized Euclidean distance is computed as

$$
d _ { u } = \sqrt { \sum _ { j = 1 } ^ { L } ( S _ { j } ^ { \prime } - w _ { u , j } ^ { \prime } ) ^ { 2 } } .\tag{3}
$$

To emphasize regions of highest morphological similarity, the distances $\{ d _ { u } \}$ are transformed into normalized similarity weights $\{ \chi _ { u } \}$ using a soft-min function:

$$
\chi _ { u } = \frac { \exp ( - \gamma d _ { u } ) } { \sum _ { u ^ { \prime } = 1 } ^ { N _ { w } } \exp ( - \gamma d _ { u ^ { \prime } } ) } ,\tag{4}
$$

where $\gamma > 0$ is a temperature parameter and $N _ { w }$ is the number of sliding windows. Finally, k-means clustering is applied to group profiles into different categories according to their similarity weights $\{ \chi _ { u } \}$ and slope characteristics.

Beach Shapelets Extractor. We employ contrastive learning to extract discriminative shapelets—i.e., representative and diagnostic segments of the beach profile or outline—from different types of beaches. The workflow is detailed below and illustrated in Fig. 3.

a) Shapelets Candidate Extraction: Since the raw beach profile data consist of discrete samples $\{ ( s _ { m } , y _ { m } ) \} _ { m = 1 } ^ { M }$ , we perform linear interpolation to obtain an elevation sequence $\overline { { u } } = \{ u _ { n } \} _ { n = 1 } ^ { N _ { s } }$ of fixed length $N _ { s }$ . To ensure comparability across samples, the sequence is rescaled by min–max normalization:

$$
\tilde { u } _ { n } = \frac { u _ { n } - \operatorname* { m i n } ( u ) } { \operatorname* { m a x } ( u ) - \operatorname* { m i n } ( u ) } , \quad n = 1 , 2 , \ldots , N _ { s } .\tag{5}
$$

From each normalized profile, subsequences of different lengths $L \in \mathcal { L } = \{ L _ { 1 } , L _ { 2 } , . . . , L _ { Q } \}$ are cropped as shapelet candidates:

$$
a _ { u , L } = \tilde { u } _ { u : u + L - 1 } , \quad u = 1 , 2 , \dots , N _ { s } - L + 1 .\tag{6}
$$

b) Transformer-based Embedding Layer: As illustrated in Fig. 4, each shapelet candidate $\boldsymbol { a } _ { u , L }$ is mapped into a lowdimensional embedding vector through a Transformer-based encoder [45]. Before entering the encoder, positional encoding is added to retain sequential order. The embedding process is denoted as:

$$
v _ { u , L } = f _ { \theta } ( a _ { u , L } ) ,\tag{7}
$$

where $f _ { \theta } ( \cdot )$ is the Transformer encoder parametrized by $\theta ,$ and $v _ { u , L } \in \mathbb { R } ^ { d _ { e } }$ is the fixed-size embedding vector of dimension $d _ { e }$

c) Contrastive Training with Triplet Loss: To construct a discriminative embedding space, we employ contrastive learning with triplet loss [46].

Triplet construction is guided by local geometric descriptors computed from each shapelet candidate. Positive and negative samples are defined based on similarity in this descriptor space, rather than purely random sampling or augmentation alone.

For each candidate shapelet $\boldsymbol { a } _ { u , L }$ , we compute a local morphology descriptor

$$
\begin{array} { r } { \mathbf { g } _ { u , L } = \left[ \bar { x } _ { u , L } , \bar { z } _ { u , L } , \mu _ { \nabla y } ^ { u , L } , \sigma _ { \nabla y } ^ { u , L } , \mu _ { \kappa } ^ { u , L } , \sigma _ { \kappa } ^ { u , L } \right] , } \end{array}\tag{8}
$$

where the elements describe the normalized cross-shore location, relative elevation, and statistical properties of local slope and curvature within the subsequence. This descriptor provides a compact representation of local geometric structure.

Positive samples are defined as (i) augmentation-based variants generated by weak perturbations that preserve local structure, and (ii) descriptor-nearest neighbors satisfying

$$
D _ { g } ( a _ { u , L } , a _ { p , L _ { p } } ) < \tau _ { p } ,\tag{9}
$$

where $D _ { g } ( \cdot , \cdot )$ measures Euclidean distance in the standardized descriptor space and $\tau _ { p }$ is a threshold.

Negative samples are selected from candidates with sufficiently different descriptor values, defined as

$$
D _ { g } ( a _ { u , L } , a _ { n , L _ { n } } ) > \tau _ { n } ,\tag{10}
$$

where $\tau _ { n } > \tau _ { p }$ . Semi-hard negatives are also included when their embedding similarity is high but descriptor distance exceeds $\tau _ { n }$ , improving robustness of the embedding space.

Let $v _ { a } = f _ { \theta } ( a _ { u , L } ) , v _ { p } = f _ { \theta } ( a _ { p } )$ , and $v _ { n } = f _ { \theta } ( a _ { n } )$ . The triplet loss is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t r i p l e t } } = \operatorname* { m a x } \Big ( 0 , \| v _ { a } - v _ { p } \| _ { 2 } ^ { 2 } - \| v _ { a } - v _ { n } \| _ { 2 } ^ { 2 } + m \Big ) , } \end{array}\tag{11}
$$

where $m > 0$ is a margin hyperparameter.

The overall training objective is computed by averaging over all valid triplets in a mini-batch:

$$
\mathcal { L } = \frac { 1 } { | T | } \sum _ { ( a , p , n ) \in \mathcal { T } } \mathcal { L } _ { \mathrm { t r i p l e t } } ( a , p , n ) ,\tag{12}
$$

where $\tau$ denotes the set of descriptor-guided triplets in the batch. Compared with random sampling, this strategy introduces a structured constraint on triplet formation based on local geometric descriptors.

d) Cluster and Choose: After obtaining the embedding representations $\{ v _ { u , L } \}$ of all shapelet candidates, we apply $k \mathrm { - }$ means to group them into K clusters based on morphological similarity in the embedding space. Formally, the clustering objective is:

$$
\operatorname* { m i n } _ { \{ C _ { k } \} _ { k = 1 } ^ { K } } \sum _ { k = 1 } ^ { K } \sum _ { v _ { u , L } \in C _ { k } } \| v _ { u , L } - \varepsilon _ { k } \| _ { 2 } ^ { 2 } ,\tag{13}
$$

where $C _ { k }$ denotes the k-th cluster and $\varepsilon _ { k }$ is its centroid. From each cluster, we select the shapelet that is closest to the centroid as the representative shapelet:

$$
s _ { k } ^ { * } = \arg \operatorname* { m i n } _ { a _ { u , L } \in C _ { k } } \| f _ { \theta } ( a _ { u , L } ) - \varepsilon _ { k } \| _ { 2 } , \quad k = 1 , 2 , \ldots , K .\tag{14}
$$

The final set of discriminative shapelets is

$$
S ^ { * } = \{ s _ { 1 } ^ { * } , s _ { 2 } ^ { * } , . . . , s _ { K } ^ { * } \} .\tag{15}
$$

e) Post-hoc Projection and Geomorphic Interpretation of Learned Shapelets: To further examine the learned shapelets, we perform a post-hoc projection analysis after the shapelet selection step. This analysis is not used in model training and is intended to visualize where each shapelet best matches observed cross-shore profiles. For each representative shapelet $s _ { k } ^ { * } ,$ we slide it along each normalized profile and compute the similarity weight $\chi _ { i , k , u }$ between $s _ { k } ^ { * }$ and the profile window starting at cross-shore position u on the i-th profile. The bestmatching location is identified as

$$
\boldsymbol { u } _ { i , k } ^ { * } = \arg \operatorname* { m a x } _ { \boldsymbol { u } } \chi _ { i , k , \boldsymbol { u } } ,\tag{16}
$$

![](images/78cc3c97f5d7b398a98fc51b5bf4e8b154b91b6cf460d5773041162663593a2a.jpg)  
Fig. 3. Overall workflow for obtaining shapelets. The workflow integrates candidate extraction, a transformer-based embedding layer, and a selection strategy.

where $\boldsymbol { u } _ { i , k } ^ { * }$ denotes the location with maximum similarity between the shapelet and the observed profile segment. The matched segment is then mapped back to the original elevation–distance coordinate system for visualization.

We then summarize the local geometric properties of the matched segments, including relative cross-shore position, elevation variation, slope, and curvature. These descriptors are used to characterize the geometric patterns captured by each shapelet. For example, some matched segments correspond to low-gradient regions, while others exhibit sharper slope transitions or convex–concave structures along the profile.

These descriptions should be interpreted as geometric pattern characterizations rather than explicit geomorphological or process-based interpretations. The goal of this analysis is to provide an intuitive understanding of the local profile structures encoded by the shapelets, rather than to assign definitive physical labels or infer underlying sediment transport mechanisms.

## C. Category-Specific Experts

The Category-Specific Experts consist of K independent regression models. Expert $t \in \{ 1 , \ldots , K \}$ is responsible for predicting profile elevations for the t-th morphology category inferred in Section III-B. We use Gaussian processes (GPs) [34] with automatic relevance determination (ARD) [47] to model the mapping from environmental drivers to profile elevation while quantifying predictive uncertainty.

For a given cross-shore coordinate s and forcing vector $\mathbf { c \in }$ $\mathbb { R } ^ { d }$ , we define the augmented input

$$
\mathbf { z } = [ \mathbf { c } , s ] \in \mathbb { R } ^ { D } , \qquad D = d + 1 .\tag{17}
$$

The t-th expert models the elevation as

$$
\begin{array} { r } { y = g _ { t } ( \mathbf { z } ) + \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , } \end{array}\tag{18}
$$

with GP prior

$$
g _ { t } ( \mathbf { z } ) \sim \mathcal { G P } \big ( m ( \mathbf { z } ) , k _ { t } ( \mathbf { z } , \mathbf { z } ^ { \prime } ) \big ) ,\tag{19}
$$

where $m ( \mathbf { z } )$ is set to zero and $k _ { t } ( \cdot , \cdot )$ is the covariance function. We adopt an ARD squared-exponential kernel:

$$
\begin{array} { r } { k _ { \mathrm { A R D } } ( \mathbf { z } , \mathbf { z } ^ { \prime } ) = \sigma _ { f } ^ { 2 } \exp \left( - \frac { 1 } { 2 } ( \mathbf { z } - \mathbf { z } ^ { \prime } ) ^ { \top } \Lambda ^ { - 1 } ( \mathbf { z } - \mathbf { z } ^ { \prime } ) \right) , } \end{array}\tag{20}
$$

where $\boldsymbol { \Lambda } = \operatorname { d i a g } ( \ell _ { 1 } ^ { 2 } , \dots , \ell _ { D } ^ { 2 } )$ and $\ell _ { d }$ is the length scale for the d-th input dimension. Given training data $\mathcal { D } _ { t } = \{ ( \mathbf { z } _ { i } , y _ { t , i } ) \} _ { i = 1 } ^ { n _ { t } }$ from category t, the predictive distribution at z is Gaussian,

$$
p ( y _ { * } \mid \mathbf { z } _ { * } , \mathcal { D } _ { t } , M _ { t } ) = \mathcal { N } \big ( \mu _ { t } ( \mathbf { z } _ { * } ) , \sigma _ { t } ^ { 2 } ( \mathbf { z } _ { * } ) \big ) ,\tag{21}
$$

![](images/a0ec7f309e2076e92efc7330b023579a43f152ebfee2688003910ecbd7944e1c.jpg)  
Fig. 4. Structure of the Transformer encoder. The encoder comprises three stacked Transformer layers, each equipped with three multi-head attention modules.

where $\mu _ { t } ( \cdot )$ and $\sigma _ { t } ^ { 2 } ( \cdot )$ are given by the standard GP posterior. Hyperparameters $\theta _ { t } = \bar { \{ \sigma _ { t } ^ { 2 } , \{ \ell _ { d } \} _ { d = 1 } ^ { D } , \sigma ^ { 2 } \} }$ are learned by maximizing the log marginal likelihood (LML). Let $\mathbf { Z } _ { t } =$ $[ \bar { \mathbf { z } } _ { 1 } , \dots , \mathbf { z } _ { n _ { t } } ] ^ { \top } \in \mathbb { R } ^ { n _ { t } \times D }$ and $\mathbf { y } _ { t } = [ y _ { t , 1 } , \dotsc , y _ { t , n _ { t } } ] ^ { \top } \in \mathbb { R } ^ { n _ { t } }$ denote the training inputs and outputs for expert t. The LML is

$$
\begin{array} { r l } & { \log p ( \mathbf { y } _ { t } \mid \mathbf { Z } _ { t } , \boldsymbol { \theta } _ { t } ) = - \frac { 1 } { 2 } \mathbf { y } _ { t } ^ { \top } ( \mathbf { K } _ { t } + \boldsymbol { \sigma } ^ { 2 } \mathbf { I } ) ^ { - 1 } \mathbf { y } _ { t } } \\ & { ~ - ~ \frac { 1 } { 2 } \log \left| \mathbf { K } _ { t } + \boldsymbol { \sigma } ^ { 2 } \mathbf { I } \right| - \frac { n _ { t } } { 2 } \log 2 \pi , } \end{array}\tag{22}
$$

where $( \mathbf { K } _ { t } ) _ { i j } = k _ { \mathrm { A R D } } ( \mathbf { z } _ { i } , \mathbf { z } _ { j } )$

## D. Gating Net

To integrate the K category-specific GP experts, we employ a gating network that produces cross-shore-dependent mixture weights. In the actual implementation, the gating network takes both the environmental forcing vector c and the crossshore coordinate s as inputs. For a given forcing vector c and cross-shore coordinate $s ,$ the mixture predictive distribution is

$$
p ( \boldsymbol { y } \mid s , \mathbf { c } ) = \sum _ { t = 1 } ^ { K } p ( \boldsymbol { y } \mid s , \mathbf { c } , M _ { t } ) q _ { \phi } ( t \mid \mathbf { c } , s ) ,\tag{23}
$$

where each expert $M _ { t }$ provides a Gaussian prediction $p ( y$ $s , \mathbf { c } , M _ { t } ) = \mathcal { N } \big ( y ; \mu _ { t } ( \mathbf { z } ) , \sigma _ { t } ^ { 2 } ( \mathbf { z } ) \big )$ with $\mathbf { z } = [ \mathbf { c } , s ]$ , and $q _ { \phi } ( t \mid$ $\mathbf { c } , s )$ denotes the cross-shore-dependent gating weight.

Motivated by Bayesian model averaging, we parameterize the gating weight by combining an evidence term and a learned contextual term:

$$
l _ { t } ( \mathbf { c } , s ) = \beta \log \widetilde { p ( \mathcal { D } _ { t } \mid M _ { t } ) } + g _ { \phi } ( \mathbf { c } , s ) _ { t } ,\tag{24}
$$

$$
q _ { \phi } ( t \mid \mathbf { c } , s ) = \frac { \exp ( l _ { t } ( \mathbf { c } , s ) ) } { \sum _ { j = 1 } ^ { K } \exp ( l _ { j } ( \mathbf { c } , s ) ) } ,\tag{25}
$$

where $g _ { \phi } ( \underline { { \mathbf { c } } } , s ) _ { t }$ is the t-th logit output of an MLP with input [c, s], log $p ( \mathcal { D } _ { t } \mid M _ { t } )$ denotes a normalized log-evidence term computed from the GP marginal likelihood for expert t, and β controls the trade-off between evidence and contextual weighting. The inclusion of s allows the mixture weights to vary between the upper beach, beachface, intertidal zone, and lower foreshore, thereby reflecting spatially varying morphodynamic controls.

The resulting predictive mean and variance are

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } [ \hat { y } ] = \sum _ { t = 1 } ^ { K } q _ { \phi } ( t \mid \mathbf { c } , s ) \mu _ { t } ( \mathbf { z } ) , } \\ & { \displaystyle \mathrm { V a r } [ \hat { y } ] = \sum _ { t = 1 } ^ { K } q _ { \phi } ( t \mid \mathbf { c } , s ) \big ( \sigma _ { t } ^ { 2 } ( \mathbf { z } ) + \mu _ { t } ^ { 2 } ( \mathbf { z } ) \big ) } \\ & { \displaystyle \phantom { \frac { K } { K } } - \Big ( \sum _ { t = 1 } ^ { K } q _ { \phi } ( t \mid \mathbf { c } , s ) \mu _ { t } ( \mathbf { z } ) \Big ) ^ { 2 } . } \end{array}\tag{26}
$$

All parameters are learned by maximizing the conditional mixture log-likelihood:

$$
\begin{array} { r } { \mathcal { L } ( \{ \theta _ { t } \} , \phi ) = \displaystyle \sum _ { i = 1 } ^ { N } \sum _ { m = 1 } ^ { M } \log \biggr ( \sum _ { t = 1 } ^ { K } q _ { \phi } ( t \mid \mathbf { c } _ { i } , s _ { m } ) \mathcal { N } \bigl ( y _ { i } ( s _ { m } ) ; } \\ { \mu _ { t } ( \mathbf { z } _ { i , m } ) , \sigma _ { t } ^ { 2 } ( \mathbf { z } _ { i , m } ) \bigr ) \biggr ) , \qquad } \end{array}\tag{27}
$$

where $\mathbf { z } _ { i , m } = [ \mathbf { c } _ { i } , s _ { m } ] , \{ \theta _ { t } \}$ are the GP hyperparameters, and ϕ are the gating-network parameters.

This formulation avoids using a single fixed mixture over the entire profile and instead learns a spatially varying expert combination at each cross-shore coordinate. Because the expert predictions are conditioned on the augmented input $\mathbf { z } = [ \mathbf { c } , s ]$ , the cross-shore coordinate is involved in both the GP posterior and the gating weights. As a result, the predicted profile is shaped by two complementary sources of spatial dependence: the covariance-based variation of each GP expert along s, and the gradual change of expert contributions across different cross-shore zones. This design is consistent with the continuous organization of beach-profile morphology, while still allowing local variations associated with berms, terraces, or bar–trough structures.

## IV. EXPERIMENTAL RESULTS AND DISCUSSION

## A. Study area and Data

1) Study Area: We analyzed 222 cross-shore profiles from   
183 sandy beaches along the Chinese coast, distributed along

![](images/a0765d70ab026b1c0a28413e45f46a47563476e2489a1169b8ec93ba9511c335.jpg)  
Fig. 5. Study area and locations of beach-profile surveys along the Chinese coast.

open-coast settings and covering a broad range of hydrodynamic and sedimentary conditions. For each beach, we compiled basic geographic and morphological information to support spatial analysis and to link local morphology with hydrodynamics and sediment characteristics.

2) Beach profile and sediment data: We surveyed beach profiles during the period of lowest tide using a STONEX S9II PRO RTK-GPS. The measured profiles generally extended from the backshore to the low-water level and were referenced to a common elevation datum. Field surveys were conducted between April 2023 and September 2024 using consistent survey procedures across all sites. Although the field surveys included part of the supratidal zone, the present analysis focused specifically on the intertidal portion of the profiles rather than the full active beach profile extending from the frontal dune to the closure depth. The supratidal zone was excluded from the analysis because it is generally weakly affected by normal tidal and wave processes and is typically modified only during extreme events such as storm surges or exceptionally high tides. The subtidal zone was not included because accurate field measurement in submerged environments is strongly constrained by water depth, wave disturbance, and the operational limitations of RTK-GPS surveys. Consequently, the analyzed profiles primarily represent the morphodynamic characteristics of the intertidal beach under tidal influence. In this study, these lowest-tide intertidal profiles are therefore treated as practical approximations of equilibrium-like intertidal beach profiles, rather than as complete long-term averaged equilibrium beach profiles over the entire active cross-shore domain. We acknowledge that true equilibrium beach profiles ideally represent long-term averaged morphology, whereas the profiles used here were measured during individual low-tide surveys. This distinction should be considered when interpreting the predicted equilibrium profiles, as the results represent equilibrium-like intertidal profiles under tidal influence and may not fully capture the complete active beach profile from the frontal dune to the closure depth or longer-term temporal variability.

Surface sediment sampling was carried out synchronously with the profile measurements. For each beach, we selected one or more representative cross-shore profiles based on coastal morphology and sediment distribution, and we preferentially sampled in the mid-tidal zone. At least one surface sample (5–20 cm depth) was collected per beach, with additional samples where needed to capture alongshore or crossshore variability. Each sample had a minimum mass of 500 g.

In the laboratory, we analyzed grain size of the 2023 sediment samples using a POWTEQ SS2000 vibrating sieve shaker. Approximately 50 g of dried sediment (oven-dried at 60 <sup>◦</sup>C for at least 24 h) was gently disaggregated when necessary and sieved through a stack of standard sieve meshes (aperture range: 0.063–4.00 mm) at an amplitude of 0.5Φ until the sediment mass on each sieve reached a stable state. The mass retained on each sieve was weighed and recorded. From the grain-size distributions, we calculated standard statistical parameters such as mean grain size, sorting, skewness, and kurtosis following classical sedimentological procedures [48].

3) Wave data: Wave conditions along the study coast were analyzed with TOMAWAC [49]. The computational domain was discretized with an unstructured triangular mesh, and bathymetry was derived from the GEBCO global dataset combined with CMAP coastal bathymetric data. The model was applied at 183 nearshore locations, hereinafter referred to as hydrodynamic observation points, which were set in the offshore area adjacent to beaches and can characterize the regional hydrodynamic conditions of the corresponding beaches.

The model performance was systematically validated through multiple approaches, demonstrating reliable accuracy. First, simulations of nine historical typhoon events (e.g., Mangkhut-1822 and Rammasun-1409) were compared with in situ observations from 11 wave stations. The results showed a mean bias in significant wave height of no more than 0.14 m (relative error $< 8 . 8 \% )$ and a mean bias in wave period of no more than 0.57 s (relative error $< 8 . 6 \% )$ ).

These validations confirm the robustness of the TOMAWAC model under both extreme and routine wave conditions. Based on the validated model, time series of significant wave height and period for the year 2023 were obtained at observation points. The maximum significant wave height and period were extracted over 12-hour intervals, and annual mean values of wave parameters were statistically analyzed. The nearshore breaking wave height was calculated (Komar, 1973) and was adopted as the wave descriptor in this study.

4) Tidal data: Tidal conditions at the same 183 points were simulated using the MIKE 21 global tide model, which was validated against long-term tide gauge data from four national marine observation stations (Pingtan, Chongwu, Xiamen, and Dongshan) along the western coast of the Taiwan Strait. The model produced hourly water-level time series from 00:00 on 1 January 2023 to 00:00 on 1 January 2024. Based on the modeling results, we calculated the mean spring tidal range (MSR) and mean tidal range (MTR) for each observation point, and MSR was used as the primary tidal indicator in our analysis.

5) Derived parameters: Combining the beach profile, sediment, wave, and tidal datasets, we derived a suite of morphodynamic and hydrodynamic parameters for each beach. Morphological descriptors include beach slope, cross-shore width, and the presence/absence of the dry beach (i.e., the supratidal zone between the mean spring high water level and the foredune toe or vegetation, which remains subaerially exposed under normal hydrodynamic conditions). Sedimentological descriptors include sediment composition and grainsize parameters (e.g., mean grain size, sorting) derived from the sieve data [48]. Hydrodynamic descriptors, such as the breaker wave height (Hb), the relative tidal range (RTR), and the dimensionless fall velocity (Ω) linking wave and sediment properties, were calculated following published formulations [50]. Together, these variables provide a consistent set of environmental drivers for the MorphoGP framework and provide the basis for the category-specific equilibrium beach profile analysis.

## B. Implementation Details

All experiments were executed on a single NVIDIA A100 Tensor Core GPU. We adopted five-fold cross-validation, with splits performed at the beach level to avoid spatial leakage. Reported results are averaged over the five test folds.

For reproducibility, a fixed random seed was used for data splitting, parameter initialization, and mini-batch sampling. All models were trained under the same optimization and stopping protocol for a fair comparison. We used the Adam optimizer with a learning rate of $1 \times 1 0 ^ { - 3 }$ . Training was run for up to 200 epochs with early stopping based on validation performance, and a batch size of 16 was used throughout.

## C. Performance

To validate the effectiveness of the proposed framework, we conduct a comparative evaluation against a diverse set of baselines. All methods are assessed under the same evaluation protocol, and performance is reported using three standard regression metrics: the coefficient of determination $( R ^ { 2 } )$ , root mean square error (RMSE) [51], and mean absolute error (MAE).

We compare MorphoGP with representative data-driven regressors, including K-Nearest Neighbors (KNN) [52], Linear Regression, a Multilayer Perceptron (MLP), Random Forest (RF) [53], Support Vector Regression (SVR) [54], and a Transformer-based model [45]. In addition, two physicsinspired analytical baselines are implemented using classical one-dimensional equilibrium-profile formulations: (i) a Bruun/Dean-type profile $y ~ = ~ A x ^ { 2 / 3 }$ [1], [6], and (ii) an exponential profile $y = B ( 1 - e ^ { - k x } )$ [10]. For these analytical baselines, the shape parameters are not fitted separately for each profile. Instead, the coefficients A, B, and k are predicted from the same set of wave, tidal, and sediment descriptors used by the transformer-based models: the coefficients are estimated on the training set via regression and then used to generate predicted equilibrium profiles on the test set, from which the target morphodynamic indicator is derived.

As shown in Table I, MorphoGP achieves the highest $R ^ { 2 }$ (0.942), indicating the strongest ability to explain the variance of the observations, while also attaining the lowest RMSE (0.297) and MAE (0.209). The improvement over the best-performing conventional data-driven baseline (RF, $R ^ { 2 } = 0 . 6 5 4 , \mathrm { R M S E } = 0 . 7 3 0 , \mathrm { M A E } = 0 . 4 6 2 )$ suggests that the proposed nonparametric probabilistic formulation enhances both predictive accuracy and generalization for macrotidal equilibrium-profile prediction. The analytical baselines provide a reference to earlier theoretical work by imposing classical profile shapes through $y = A x ^ { 2 / 3 }$ and $y = B ( 1 - e ^ { - k x } )$ their results further highlight the limitations of tide-insensitive parametric forms when applied to diverse macrotidal beaches where tidal processes strongly influence profile morphology.

Process-based numerical morphodynamic models, such as XBeach and Delft3D, are not included as direct baselines in this comparison. These models primarily target time-evolving simulations of coastal change under prescribed forcing and typically require detailed, site-specific boundary conditions, long-term hydrodynamic and sediment time series, and extensive calibration for each location. Such inputs are not consistently available for the large, multi-site dataset considered here. Moreover, MorphoGP is a reduced-order framework that predicts static morphodynamic indicators from aggregated environmental descriptors, whereas process-based models are designed to resolve transient processes. A direct benchmark would therefore be both computationally prohibitive and conceptually mismatched. We instead focus on baselines that operate on the same input feature space and are intended for fast, large-sample prediction, while reporting per-profile analytical fits separately as curve-fitting references.

![](images/18b8b956c6d37ed65bb07bb127c498523c3212150c07a20f313bce9cc7d0977a.jpg)

TABLE I  
PERFORMANCE COMPARISON OF DIFFERENT MODELS ON THE TEST SET. VALUES ARE REPORTED AS MEAN ± STANDARD DEVIATION ACROSS THE FIVE CROSS-VALIDATION FOLDS. ARROWS INDICATE DESIRABLE METRIC DIRECTIONS (↑: HIGHER IS BETTER; ↓: LOWER IS BETTER).
<table><tr><td>Model</td><td> $R ^ { 2 } \left( \uparrow \right)$ </td><td>RMSE (↓)</td><td>MAE (↓)</td></tr><tr><td>KNN</td><td> $0 . 5 3 3 \pm 0 . 0 4 1$ </td><td> $0 . 8 4 8 \pm 0 . 0 6 3$ </td><td> $0 . 5 9 5 \pm 0 . 0 4 8$ </td></tr><tr><td>Linear Regression</td><td> $0 . 6 0 9 \pm 0 . 0 3 8$ </td><td> $0 . 7 7 6 \pm 0 . 0 5 2$ </td><td> $0 . 5 9 5 \pm 0 . 0 4 5$ </td></tr><tr><td>MLP</td><td> $0 . 4 7 9 \pm 0 . 0 5 7$ </td><td> $0 . 8 9 6 \pm 0 . 0 7 1$ </td><td> $0 . 6 0 1 \pm 0 . 0 5 9$ </td></tr><tr><td>Random Forest</td><td> $0 . 6 5 4 \pm 0 . 0 3 5$ </td><td> $0 . 7 3 0 \pm 0 . 0 4 9$ </td><td> $0 . 4 6 2 \pm 0 . 0 3 7$ </td></tr><tr><td>SVR</td><td> $0 . 5 6 9 \pm 0 . 0 4 4$ </td><td> $0 . 8 1 5 \pm 0 . 0 5 8$ </td><td> $0 . 5 5 0 \pm 0 . 0 4 6$ </td></tr><tr><td>Transformer</td><td> $0 . 6 0 0 \pm 0 . 0 5 0$ </td><td> $0 . 7 8 4 \pm 0 . 0 6 1$ </td><td> $0 . 5 2 7 \pm 0 . 0 5 2$ </td></tr><tr><td>Bruun-type (transformer-based models)</td><td> $0 . 6 2 3 \pm 0 . 0 3 9$ </td><td> $0 . 7 7 9 \pm 0 . 0 5 4$ </td><td> $0 . 4 9 8 \pm 0 . 0 4 1$ </td></tr><tr><td>Exponential (transformer-based models)</td><td> $0 . 5 1 7 \pm 0 . 0 4 6$ </td><td> $0 . 8 3 2 \pm 0 . 0 6 7$ </td><td> $0 . 5 9 9 \pm 0 . 0 5 5$ </td></tr><tr><td>MorphoGP (Ours)</td><td> $\mathbf { 0 . 9 4 2 \pm 0 . 0 2 2 }$ </td><td> $\mathbf { 0 . 2 9 7 \pm 0 . 0 4 1 }$ </td><td> $\mathbf { 0 . 2 0 9 \pm 0 . 0 3 8 }$ </td></tr></table>

![](images/c9e63c651414f99faa13c7abc36b2b9fb5f4cb3512fb28ee6e704457deea26a2.jpg)

![](images/f0f5f9c4759fa140bd4a0067bb3f63b6bdedbc465f5c5b49c75639c0e81cd9da.jpg)

![](images/da08cb9373ecc3e8756cb7405fac3c382b88d73562666a1ec794dd1901f380dc.jpg)

![](images/aebb806ff471a6f58a8b8ed093c6026fa1ec953e9728ccb7c0a3e2dcd2eaec5c.jpg)  
Fig. 6. Expert decomposition of the predicted cross-shore profile. (a) Predicted cross-shore profile with uncertainty. The solid curve shows the MorphoGP prediction, while the dashed curve denotes the observed elevation. The shaded band represents predictive uncertainty, providing an overall assessment of profile-scale accuracy. (b) Continuous expert mixture proportions along the profile. The stacked areas show the normalized mixture weights of individual experts, obtained by applying a softmax transformation to the pre-softmax gating logits, illustrating how expert responsibility varies smoothly in space. Local expert contributions at representative cross-shore locations. At selected positions along the transect, pie charts visualize the corresponding expert weight distributions, highlighting heterogeneous regions where different experts dominate the prediction.

Beyond aggregate error metrics, Fig. 6 offers qualitative evidence that MorphoGP captures cross-shore morphological structure. Across representative transects, the predicted profiles reproduce the observed equilibrium shapes, while the 95% predictive confidence intervals adapt to location-dependent variability along the profile, indicating where predictions are more or less reliable.

TABLE II

ABLATION STUDY ON KEY COMPONENTS OF MORPHOGP. THE REFERENCE IS THE FULL CONFIGURATION (CONTOURCLUSTER + GP PREDICTOR + RBF-ARD KERNEL). EACH VARIANT MODIFIES ONLY THE COMPONENT INDICATED BY ITS CATEGORY WHILE KEEPING ALL OTHER COMPONENTS   
IDENTICAL TO THE REFERENCE. THE RANDOM GROUPING CONTROL KEEPS THE SAME NUMBER OF GROUPS, GP EXPERTS, AND GATING STRUCTURE AS   
MORPHOGP, BUT REPLACES CONTOURCLUSTER LABELS WITH RANDOM GROUP ASSIGNMENTS. RESULTS FOR RANDOM GROUPING ARE REPORTED AS MEAN ± STANDARD DEVIATION OVER FIVE RANDOM SEEDS. BOLDFACE INDICATES THE BEST RESULT ACROSS ALL ROWS.

<table><tr><td>Category</td><td>Variant</td><td> $\overline { { R ^ { 2 } \left( \uparrow \right) } }$ </td><td>RMSE (↓)</td><td>MAE (↓)</td></tr><tr><td>Reference</td><td>Full MorphoGP (ContourCluster + GP + RBF-ARD)</td><td>0.942</td><td>0.297</td><td>0.209</td></tr><tr><td>Classification</td><td>w/o ContourCluster (Single GP, RBF-ARD)</td><td>0.664</td><td>0.718</td><td>0.463</td></tr><tr><td rowspan="2">Predictor</td><td>Random grouping (GP experts + Gating Net)</td><td>0.612</td><td>0.772</td><td>0.507</td></tr><tr><td>MLP (ContourCluster; replaces GP)</td><td>0.876</td><td>0.436</td><td>0.299</td></tr><tr><td rowspan="2">Kernel</td><td>Transformer (ContourCluster; replaces GP)</td><td>0.920</td><td>0.350</td><td>0.257</td></tr><tr><td>Matérn-ARD (ν = 3/2) (ContourCluster + GP)</td><td>0.935</td><td>0.316</td><td>0.224</td></tr><tr><td></td><td>Rational Quadratic-ARD (ContourCluster + GP)</td><td>0.930</td><td>0.321</td><td>0.232</td></tr></table>

TABLE III

COMPARISON BETWEEN THE MASSELINK–SHORT Ω–RTR CLASSIFICATION AND CONTOURCLUSTER. NMI MEASURES THE LABEL AGREEMENT BETWEEN THE TWO SCHEMES; THE SILHOUETTE SCORE EVALUATES COMPACTNESS AND SEPARATION IN THE RAW PROFILE-GEOMETRY SPACE.
<table><tr><td>Metric</td><td>Masselink-Short</td><td>ContourCluster</td></tr><tr><td>NMI with the other scheme</td><td>0.104</td><td>0.104</td></tr><tr><td>Silhouette score</td><td>-0.355</td><td>0.191</td></tr></table>

## D. Morphological Validity of ContourCluster

To examine whether the learned ContourCluster categories represent meaningful profile-geometry regimes rather than arbitrary numerical partitions, we compare them with the classical Masselink–Short Ω–RTR morphodynamic classification. The Ω–RTR framework remains an important process-oriented classification for tide-influenced beaches because it links beach state to dimensionless fall velocity and relative tide range. However, it is not designed to directly optimize the geometric compactness of complete cross-shore profiles.

As shown in Table III, the two classification schemes show low label agreement, with an NMI of 0.104. This indicates that ContourCluster does not simply reproduce the traditional Ω–RTR classes. In the raw profile-geometry space, ContourCluster achieves a higher silhouette score than the Masselink–Short classification (0.191 versus -0.355), suggesting more compact and better separated profile groups. These results suggest that the Masselink–Short framework remains physically meaningful as a process-oriented morphodynamic classification, while ContourCluster provides a complementary morphology-driven partition that is more suitable for defining geometrically coherent profile regimes in the present dataset.

We further inspect representative learned shapelets by projecting them back to the profile segments where they show high similarity. Fig. 7 shows three representative local subsequences. These examples suggest that the learned shapelets are associated with recognizable local profile-geometry patterns: S0 corresponds to a low-gradient concave transition, S5 resembles a convex slope-break transition, and S6 captures a steep beachface-like segment with relatively stable slope.

These interpretations should be considered cautious and geometry-based. The learned shapelets are statistical subsequences extracted from observed profiles, and they should not be interpreted as deterministic geomorphological units or direct process-level representations. Nevertheless, their correspondence with local profile patterns helps reduce the black-box nature of ContourCluster and supports its use as a morphology-aware clustering module.

![](images/ba7a15882ce663e510fec5db6c88364479f7b3a11ca4b78a3c3c4161b861f38a.jpg)  
Fig. 7. Representative learned shapelets projected back to observed crossshore profile segments. S0 shows a low-gradient concave transition, S5 shows a convex slope-break-like transition, and S6 shows a steep beachface-like segment. These interpretations describe local profile-geometry patterns associated with learned subsequences and should not be regarded as deterministic geomorphological labels or direct process-level explanations.

## E. Ablation and Sensitivity Analysis

To investigate the contribution of each component in the proposed MorphoGP framework, we conducted a series of ablation experiments focusing on three aspects: (1) the contourbased morphodynamic classification module, (2) the choice of prediction architecture, and (3) the selection of covariance kernel within the Gaussian Process (GP) predictor. All experiments used the same data split and training protocol for a fair comparison. The results are summarized in Table II.

Removing the ContourCluster module and training a single GP expert across all beaches (“w/o ContourCluster”) leads to a clear drop in performance compared with the full MorphoGP configuration, with $R ^ { 2 }$ decreasing from 0.942 to 0.664 and RMSE increasing from 0.297 to 0.718. This result indicates that a single global GP is insufficient to represent the heterogeneous forcing–profile relationships across different beach morphologies. To further examine whether the improvement comes from morphology-aware grouping rather than simply from using multiple experts, we added a random grouping control. In this variant, the number of groups, GP experts, and gating structure are kept unchanged, while the Contour-Cluster labels are replaced by random group assignments.

Across five random seeds, the random grouping variant obtains $R ^ { 2 } \ = \ 0 . 6 1 2 , \ \mathrm { R M S E } \ = \ 0 . 7 7 2$ , and $\mathrm { { M A E } = 0 . 5 0 7 }$ , which are substantially worse than the full MorphoGP results. This suggests that the learned contour-based partition provides more coherent domains for category-specific regression than arbitrary grouping.

To assess the impact of the prediction architecture, we replaced the GP with a multilayer perceptron (MorphoGP-MLP) and a Transformer-based regressor (MorphoGP-Trans), while keeping the input features and the ContourCluster module unchanged. Both deep-learning variants achieve reasonably strong performance, but they remain consistently inferior to the GP-based MorphoGP in all metrics. The GP predictor attains the highest $R ^ { 2 }$ and the lowest RMSE and MAE, suggesting that its probabilistic structure provides smoother regression and better uncertainty handling in data-sparse regions, which is crucial for morphodynamic prediction.

Finally, we examined the influence of the GP kernel by testing three widely used covariance functions: the Radial Basis Function (RBF), a Matern kernel with´ $\nu = 3 / 2$ , and the Rational Quadratic kernel. These kernels span a spectrum from very smooth priors (RBF) to rougher local variability (Matern) and multi–scale behavior (Rational Quadratic). As´ shown in Table II, MorphoGP with the RBF kernel yields the best overall trade-off among $R ^ { 2 }$ , RMSE, and MAE, and is therefore adopted as the default kernel in the final model. The relatively small performance differences across kernels indicate that MorphoGP is robust to reasonable choices of kernel smoothness and length scale, rather than relying on a highly tuned, exotic kernel.

In addition to component ablations, we analyze the sensitivity of the ContourCluster module to the number of clusters $k ,$ which controls the granularity of the morphodynamic partition. We sweep $k \in \{ 2 , 3 , \ldots , 8 \}$ under identical beach-level crossvalidation splits and compute the fold-averaged Silhouette coefficient for each k. Across folds, the Silhouette score increases from small k, peaks at $k = 5$ , and then decreases for larger $k ,$ indicating that overly coarse or overly fine partitions degrade clustering compactness and separation. To avoid selection bias, the optimal $k ^ { \star }$ is selected using only the training split within each fold and then fixed when evaluating the corresponding test split. Under this protocol, the best average performance is achieved at $k ^ { \star } ~ = ~ 5$ with a foldaveraged Silhouette coefficient of 0.4235.

## F. Region-Held-Out Transfer and OOD Uncertainty Assessment

Global error metrics such as RMSE and $R ^ { 2 }$ provide an overall indication of predictive skill, but they do not reveal whether the model can recognize situations in which its predictions are less reliable. This is particularly important when transferring equilibrium beach-profile models to new coastal settings where forcing–morphology combinations may differ from those represented during training. We therefore assess regional transferability and uncertainty response using a region-held-out evaluation protocol and test whether MorphoGP assigns higher predictive uncertainty under geographic domain shift. It should be noted that this setting represents a moderate regional domain shift within the South China coastal dataset, rather than a strict extrapolation to entirely different coastal systems. For any forcing vector c (wave, tide, sediment, and associated descriptors) and cross-shore coordinate $x ,$ MorphoGP yields a predictive distribution $p ( y \mid x , c )$ with mean ${ \hat { y } } ( x , c )$ and variance $\mathrm { V a r } [ \hat { y } ( x , c ) ]$ (see (24)). We use the predictive variance as an uncertainty score,

TABLE IV  
REGION-HELD-OUT OOD EVALUATION FOR MORPHOGP. IN EACH RUN, ONE REGION IS WITHHELD FOR TESTING AND THE MODEL IS TRAINED ON THE REMAINING REGIONS. REPORTED RMSE AND MEAN PREDICTIVE VARIANCE $\overline { { E } } _ { \mathrm { u n c } }$ ARE COMPUTED OVER ALL TEST POINTS IN THE WITHHELD REGION; “OVERALL” DENOTES THE RANDOM-SPLIT REFERENCE.
<table><tr><td>Evaluation setting</td><td>RMSE (↓)</td><td> $\overline { { E } } _ { \mathrm { u n c } } \left( \uparrow \right)$ </td></tr><tr><td>Overall (random split / test folds)</td><td>0.297</td><td>0.071</td></tr><tr><td>Hainan held out</td><td>0.328</td><td>0.105</td></tr><tr><td>Fujian held out</td><td>0.351</td><td>0.113</td></tr><tr><td>Guangdong held out</td><td>0.314</td><td>0.097</td></tr></table>

$$
E _ { \mathrm { u n c } } ( x , c ) = \mathrm { V a r } [ \hat { y } ( x , c ) ] ,\tag{28}
$$

where larger values indicate lower model confidence. For reporting at the dataset/region level, we summarize uncertainty by the mean predictive variance $\begin{array} { r } { \overline { { E } } _ { \mathrm { u n c } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } E _ { \mathrm { u n c } , i } } \end{array}$ over all test points.

To mimic external validation, we perform leave-one-regionout evaluation over the three geographic groups with sufficient samples (Hainan, Fujian, and Guangdong). In each run, all profiles from one region are withheld as a test set and the model is trained using only the remaining regions. Profiles from other provinces (e.g., Zhejiang and Shandong) are included in the dataset but are not treated as a separate held-out group due to the limited number of samples. All preprocessing steps are fitted on the training regions only and then applied to the heldout region to prevent information leakage.

Table IV reports RMSE and $\overline { { E } } _ { \mathrm { u n c } }$ for each held-out region, together with the overall random-split evaluation for reference. Compared with the overall setting, all region-held-out runs exhibit higher ${ \overline { { E } } } _ { \mathrm { u n c } } ,$ suggesting that MorphoGP expresses reduced confidence when applied to unseen geographic regions. At the same time, the RMSE values remain relatively low, increasing only from 0.297 in the random-split setting to $0 . 3 1 4  – 0 . 3 5 1$ in the region-held-out tests. This indicates that the three held-out regions are not completely dissimilar from the training regions in the feature and morphology spaces.

The relatively stable transfer performance can be explained by two factors. First, although Hainan, Fujian, and Guangdong are geographically distinct, they belong to the broader South China coastal setting and may share overlapping ranges of wave, tide, sediment, and morphological descriptors. Second, MorphoGP does not rely on geographic labels directly; instead, it organizes profiles according to learned morphological regimes and then applies category-specific Gaussian process experts. Therefore, if a held-out region contains beach-profile types and forcing combinations that are already represented in the training regions, the model can transfer reasonably well despite the geographic split.

![](images/a5fe65a78b3be493b2d2bbf3bfd5da4450153b98a138324e004de4bf03c49b4e.jpg)  
Fig. 8. Pearson correlation heatmap of the 16 selected input variables. The analysis was conducted using all profile-point samples. Strong correlations ar mainly observed within tidal-range, wave-height, and wave-period variable groups, indicating potential redundancy that should be considered when interpretin ARD-based feature relevance.

However, this result should not be interpreted as evidence of universal OOD generalization. The current region-heldout protocol is limited by the available spatial coverage of the dataset and cannot fully evaluate model behavior under stronger domain shifts, such as regions with tidal ranges outside the training distribution, substantially different sediment supply, reef-controlled or engineered coastlines, stormdominated disequilibrium profiles, or beach types absent from the learned ContourCluster categories. Under such conditions, both the predicted mean profiles and the uncertainty estimates may become less reliable. Therefore, the present evaluation demonstrates promising transferability under moderate regional shifts, while broader validation using more geographically and morphodynamically diverse datasets remains necessary for operational deployment.

## G. Relevance Analysis

To further interpret the internal decision-making process of the proposed MorphoGP model and to identify the dominant predictive descriptors of equilibrium beach morphology, we conducted a feature relevance analysis based on the Automatic Relevance Determination (ARD) mechanism within each Gaussian Process expert.

In this analysis, each input variable is assigned an independent characteristic length scale, which is optimized jointly with other GP hyperparameters during training. We report the learned ARD length scales, where smaller values indicate higher sensitivity of the model output to the corresponding input dimension.

To support interpretation, we examine pairwise linear correlations among the 16 input variables using Pearson correlation analysis (Fig. 8).

The correlation analysis reveals clear intra-group dependencies among several physically related variables. Strong correlations are observed within wave- and sediment-related descriptors, including Breaker Wave Height and Deepwater Wave Height $( r \ : = \ : 0 . 9 9 3 )$ , Wave Period and Annual Mean Wave Period (r = 0.935), and Mean Grain Size and High-tide Sediment Settling Velocity (r = 0.879). These results indicate that variables within the same physical category tend to share overlapping information and may partially represent similar underlying processes.

In contrast, correlations across different physical groups are generally weak. Tidal-range-related variables exhibit only weak linear relationships with wave- and sediment-related descriptors, with most correlation coefficients typically below |r| < 0.30. This suggests limited linear redundancy between tidal forcing and other environmental descriptors in the dataset, and indicates that different physical drivers provide largely independent information.

These results suggest that the high ARD relevance of tidalrange-related variables is unlikely to be solely driven by linear collinearity with wave or sediment descriptors. Instead, tidal variables appear to provide complementary predictive information within the overall environmental descriptor space. The Pearson correlation analysis is used here solely to assess redundancy and dependence structure among input variables, thereby providing context for interpreting the ARD-based importance ranking.

Fig. 9 illustrates the feature-wise mean ARD values with standard deviation across 50 independent training runs, sorted in ascending order to emphasize the most influential features.

![](images/3fb0f303cb0d31423a11018e99b3fed505e489864970a705425af1f510d1c0ba.jpg)  
Fig. 9. Feature-wise mean ARD values with variance for all input parameters in the MorphoGP framework. Smaller ARD values indicate higher feature relevance. The most relevant variables are mainly associated with tidal range, directional wave climate, and sediment/morphological descriptors. Because several input variables are strongly correlated, the ARD results are interpreted at the level of physically meaningful variable groups rather than as strict causal evidence for isolated variables.

As shown in the figure, Annual Mean Spring Tidal Range has the smallest ARD value, followed by Annual Mean Tidal Range and Frequency of Dominant Wave Direction, indicating that these variables show the highest sensitivity within the trained MorphoGP model. Combined with the Pearson correlation analysis, this result suggests that tidal-range-related descriptors provide strong and relatively independent predictive information for equilibrium beach-profile prediction, rather than simply acting as linear substitutes for wave- or sedimentrelated descriptors.

From the perspective of the main driving factors, the prediction of equilibrium beach profiles (EBP) for macrotidal beaches is broadly consistent with geomorphological expectations. Tidal range determines the extent of the intertidal zone and the frequency with which different parts of the beach are exposed to hydrodynamic forcing, thereby constraining the spatiotemporal scales of coastal processes acting on the profile. The presence or absence of a dry beach further reflects whether the equilibrium profile is in a self-stable state or in a stressed state under sediment deficit. Beach slope and crossshore width collectively modulate wave-energy dissipation and the associated sediment redistribution across the intertidal zone, which are key processes governing the evolution of equilibrium profiles. However, these interpretations should be understood as physically plausible explanations of model sensitivity and predictive relevance, not as definitive causal attribution derived solely from ARD.

A second tier of variables, including Frequency of Dominant Wave Direction, the Dimensionless Fall Velocity, Dominant Wave Direction, and the Sorting Coefficient, also shows relatively low ARD values, suggesting appreciable but more nuanced roles in shaping the equilibrium state. These parameters primarily characterize the dominant wave climate and sediment properties, which adjust the detailed profile shape around the broader hydrodynamic and morphological template. Their relevance indicates that wave and sediment descriptors still provide useful predictive information, although their individual ARD rankings may be affected by correlations within each descriptor group.

In contrast, variables such as the cross-shore coordinate (x), Relative Tidal Range (RTR), Mean Grain Size (mm), deepwater wave height $( H _ { d } ) _ { \cdot }$ , and breaker wave height $( H _ { b } )$ appear toward the upper end of the ARD spectrum, indicating weaker direct sensitivity in the predictive model. Nevertheless, these variables provide important contextual information about spatial geometry, offshore hydrodynamics, and sediment characteristics, and they may still influence morphodynamic responses indirectly through nonlinear feature interactions within the mixture-of-experts architecture. In particular, the lower ARD relevance of $H _ { d }$ and $H _ { b }$ should be interpreted cautiously because these two variables are highly correlated with each other, and their predictive information may be partly represented by other wave-related descriptors.

The variance of ARD values across training runs provides further insight into the stability of feature influence. Tidal and morphological parameters such as Annual Mean Spring Tidal Range and Beach Slope exhibit relatively low variance, implying a consistent and robust contribution to the model predictions under different random initializations and resampling. By contrast, several wave-related descriptors, such as Mean Wave Height and Wave Period, display higher variance in their ARD values, likely reflecting interdependence with other hydrodynamic or sedimentary factors and greater sensitivity to sampling variability.

Overall, the ARD-based analysis suggests that tidal-rangerelated and morphological descriptors provide strong and stable predictive information for macrotidal equilibrium beachprofile prediction in MorphoGP. The Pearson correlation analysis further supports the reliability of this interpretation by showing that tidal descriptors are not highly linearly correlated with most wave- and sediment-related descriptors. Therefore, the high ARD relevance of tidal variables is unlikely to be caused solely by cross-category multicollinearity. This modelbased finding is consistent with physical expectations for macrotidal environments, where strong tides and the associated morphological configuration influence nearshore hydrodynamics and sediment transport in ways that are more complex than in predominantly wave-dominated coasts.

## V. CONCLUSION

This study proposes MorphoGP, a category-specific Gaussian Process framework for equilibrium beach-profile prediction under varying tidal conditions. The framework integrates unsupervised contour-based classification, probabilistic GP regression, and a gating mechanism for expert aggregation, enabling morphology-conditioned prediction from environmental descriptors.

MorphoGP organizes beach profiles into data-driven morphological regimes and performs probabilistic prediction of equilibrium-like intertidal profiles from wave, tide, and sediment descriptors. Rather than explicitly simulating full coastal morphodynamic processes, it learns statistical relationships between environmental forcing and observed profile geometry.

Experiments on 183 beaches along the Chinese coast demonstrate that MorphoGP consistently outperforms conventional machine learning and deep learning baselines in terms of $R ^ { 2 }$ , RMSE, and MAE. The ContourCluster module yields compact and well-separated morphological groups, and ablation studies confirm the effectiveness of both the clustering and Gaussian process expert components.

In addition, MorphoGP provides structured uncertainty estimates, where predictive variance increases under region-heldout evaluation, indicating reduced confidence under moderate domain shift. Feature relevance analysis further suggests that tidal-range and morphological descriptors play a dominant role in explaining variability in equilibrium profile shapes, while wave and sediment variables contribute complementary information.

The learned shapelets and expert representations capture recurring local geometric patterns in cross-shore profiles, providing an interpretable view of morphological variability at the sub-profile scale. However, these representations should be interpreted as statistical descriptors of observed geometry rather than direct process-based physical units.

Overall, MorphoGP provides a probabilistic and morphology-aware framework for equilibrium beach-profile prediction with quantified uncertainty. Its main contributions lie in morphology-driven regime decomposition, GP-based predictive modeling, and spatially varying expert aggregation.

Future work will focus on extending the framework to fully continuous spatio-temporal profile evolution, improving physical constraints in the predictive model, and incorporating broader cross-regional datasets to enhance generalization under stronger domain shifts.

## ACKNOWLEDGMENT

The computations in this research were performed using the CFFF platform of Fudan University.

## APPENDIX A

PER-PROFILE FITTED ANALYTICAL REFERENCES

TABLE V  
PER-PROFILE LEAST-SQUARES FITTED ANALYTICAL REFERENCES. THESE RESULTS ARE REPORTED AS ORACLE-STYLE CURVE-FITTING REFERENCES RATHER THAN PREDICTIVE BASELINES.
<table><tr><td>Model</td><td> $R ^ { 2 }$ </td><td>RMSE (m)</td><td>MAE (m)</td></tr><tr><td>Bruun/Dean-type model</td><td>0.9855</td><td>0.1494</td><td>0.1040</td></tr><tr><td>Exponential model</td><td>0.9963</td><td>0.0750</td><td>0.0473</td></tr></table>

To further clarify the distinction between profile prediction and post-hoc curve fitting, we provide additional per-profile least-squares fitting results for the classical analytical formulations. Unlike MorphoGP, which predicts unseen equilibrium beach profiles from environmental descriptors without using the target profile elevations during inference, the per-profile fitted analytical models use each observed target profile to estimate their parameters. Therefore, these results are reported as oracle-style curve-fitting references rather than strict predictive baselines.

For each observed test profile, the Bruun/Dean-type parameter A and the exponential parameters B and k were optimized independently by least squares. The fitted profiles were then evaluated using the same metrics as in the main experiments. As shown in Table V, per-profile fitting substantially improves the analytical formulations, indicating that these parametric models have strong shape-fitting capacity when the target profile is available. However, this setting differs from the predictive task considered in the main comparison, where the target profile is unavailable at inference time.

TABLE VI  
DEFINITIONS OF THE INPUT PARAMETERS USED IN THIS STUDY.
<table><tr><td>Input parameter</td><td>Symbol</td><td>Unit</td><td>Definition / calculation</td></tr><tr><td>Cross-shore coordinate</td><td>x</td><td>m</td><td>Horizontal coordinate perpendicular to the shoreline.</td></tr><tr><td>Mean spring tidal range</td><td>MSR</td><td>m</td><td>Vertical difference between mean high water springs and mean low water springs.</td></tr><tr><td>Mean tidal range</td><td>MTR</td><td>m</td><td>Average vertical difference between adjacent high and low tides.</td></tr><tr><td>Relative tidal range</td><td>RTR</td><td></td><td> $\mathrm { R T R } = \mathrm { M S R } / H _ { b } .$ </td></tr><tr><td>Breaker wave height</td><td> $H _ { b }$ </td><td>m</td><td> $H _ { b } = 0 . 3 9 g ^ { 1 / 5 } ( T H _ { d } ^ { 2 } ) ^ { 2 / 5 } .$ </td></tr><tr><td>Deep-water wave height</td><td> $H _ { d }$ </td><td>m</td><td>Wave height in deep-water conditions.</td></tr><tr><td>Wave period</td><td> $T$ </td><td>S</td><td>Time interval between two consecutive wave crests or troughs.</td></tr><tr><td>Dominant wave direction θ</td><td></td><td>0</td><td>Wave direction with the highest integrated energy.</td></tr><tr><td>Frequency of dominant wave direction</td><td></td><td>%</td><td>Occurrence frequency of waves from the dominant direction.</td></tr><tr><td>Dimensionless fall ve- Ω locity</td><td></td><td></td><td> $\Omega = H _ { b } / ( \omega _ { s } T ) .$ </td></tr><tr><td>High-tide sediment fall</td><td> $\omega _ { s }$ </td><td> $\mathrm { ~ m ~ s ~ } ^ { - 1 }$ </td><td>Sediment fall velocity at high-tide level.</td></tr><tr><td>velocity Mean grain size</td><td> $M _ { z } / d _ { m }$ </td><td>Φ/mm</td><td>Average sediment particle size.</td></tr><tr><td>Sorting coefficient</td><td> $\delta$ </td><td></td><td> $\delta = ( \Phi _ { 8 4 } - \Phi _ { 1 6 } ) / 4 + ( \Phi _ { 9 5 } - \Phi _ { 5 } ) / 6 . 6 .$ </td></tr><tr><td>Skewness</td><td> $\mathrm { S k }$ </td><td></td><td>Grain-size distribution asymmetry.</td></tr><tr><td>Kurtosis</td><td> $K$ </td><td></td><td>Peakedness of the grain-size distribution.</td></tr><tr><td>Beach slope</td><td> $\tan \beta$ </td><td></td><td>Intertidal beach slope derived from the surveyed profile.</td></tr></table>

Note: $\Phi _ { x }$ denotes the grain-size value at which the cumulative percentage reaches $x \% .$ . Due to field-survey limitations, the lowest surveyed elevation may not always reach the mean low water spring level.

## APPENDIX B

## DEFINITION OF INPUT PARAMETERS

Table VI summarizes the input parameters used in this study. Detailed derivations of standard morphodynamic parameters, such as Ω and RTR, follow conventional definitions.

## REFERENCES

[1] R. G. Dean, “Equilibrium beach profiles: Characteristics and applications,” Journal of Coastal Research, vol. 7, no. 1, pp. 53–84, 1991.

[2] D. W. Johnson, Shore Processes and Shoreline Development. New York: Prentice Hall, 1919, p. 584.

[3] P. Cornaglia, “Delle spiaggie,” Atti della Classe di Scienze Fisiche, Matematiche e Naturali, vol. 5, pp. 284–304, 1889.

[4] A. Short, “Macro-meso tidal beach morphodynamics: an overview,” Journal of Coastal Research, pp. 417–436, 1991.

[5] L. Wright, P. Nielsen, A. Short, and M. Green, “Morphodynamics of a macrotidal beach,” Marine geology, vol. 50, no. 1-2, pp. 97–127, 1982.

[6] P. Bruun, Coast erosion and the development of beach profiles. US Beach Erosion Board, 1954, vol. 44.

[7] R. G. Dean, Equilibrium beach profiles: US Atlantic and Gulf coasts. Department of Civil Engineering, University of Delaware Newark, Delaware, 1977, vol. 12.

[8] K. R. Bodge, “Representing equilibrium beach profiles with an exponential expression,” Journal of Coastal Research, vol. 8, no. 1, pp. 47–55, 1992.

[9] P. Z.-F. Lee, “The submarine equilibrium profile: A physical model,” Journal of Coastal Research, vol. 10, no. 1, pp. 1–17, 1994.

[10] M. Larson, N. C. Kraus, and R. A. Wise, “Equilibrium beach profiles under breaking and non-breaking waves,” Coastal Engineering, vol. 36, no. 1, pp. 59–85, 1999.

[11] C. Stokes, M. Davidson, and P. Russell, “Observation and prediction of three-dimensional morphology at a high-energy macrotidal beach,” Geomorphology, vol. 243, pp. 1–13, 2015.

[12] Y. Wei, H. Qi, G. Liu, F. Cai, Y. He, K. Hua, and S. Zhao, “Analysis of the morphodynamic characteristics of macrotidal beaches on the western coast of the taiwan strait,” Regional Studies in Marine Science, p. 104429, 2025.

[13] E. Anthony, F. Levoy, and O. Monfort, “Morphodynamics of intertidal bars on a megatidal beach, merlimont, northern france,” Marine geology, vol. 208, no. 1, pp. 73–100, 2004.

[14] M. Hashemi, Z. Ghadampour, and S. Neill, “Using an artificial neural network to model seasonal changes in beach profiles,” Ocean Engineering, vol. 37, no. 14-15, pp. 1345–1356, 2010.

[15] W. W. de Melo, J. Pinho, and I. Iglesias, “A data model to forecast the morphological evolution of multiple beach profiles,” Coastal Engineering, vol. 192, p. 104574, 2024.

[16] A. R. Khan, M. S. B. Ab Razak, B. B. Yusuf, H. Z. B. M. Shafri, and N. B. Mohamad, “Future prediction of coastal recession using convolutional neural network,” Estuarine, Coastal and Shelf Science, vol. 299, p. 108667, 2024.

[17] T. Beuzen, E. B. Goldstein, and K. D. Splinter, “Ensemble models from machine learning: an example of wave runup and coastal dune erosion,” Natural Hazards and Earth System Sciences, vol. 19, no. 10, pp. 2295– 2309, 2019.

[18] S. Adusumilli, N. Cirrito, L. Engeman, J. W. Fiedler, R. Guza, A. M. Lange, M. A. Merrifield, W. O’Reilly, and A. P. Young, “Predicting shoreline changes along the california coast using deep learning applied to satellite observations,” Journal of Geophysical Research: Machine Learning and Computation, vol. 1, no. 3, p. e2024JH000172, 2024.

[19] W. W. de Melo, J. Pinho, and I. Iglesias, “A data model to forecast the morphological evolution of multiple beach profiles,” Coastal Engineering, vol. 192, p. 104574, 2024.

[20] S. K. Muroi, E. Bertone, N. Cartwright, and F. Alvarez, “Machine learning methods for predicting shoreline change from submerged breakwater simulations,” Engineering Applications of Artificial Intelligence, vol. 152, p. 110726, 2025.

[21] E. J. Anthony and J. D. Orford, “Between wave-and tide-dominated coasts: the middle ground revisited,” Journal of Coastal Research, no. 36, pp. 8–15, 2002.

[22] H. Karunarathna, J. Horrillo-Caraballo, Y. Kuriyama, H. Mase, R. Ranasinghe, and D. E. Reeve, “Linkages between sediment composition, wave climate and beach profile variability at multiple timescales,” Marine Geology, vol. 381, pp. 194–208, 2016.

[23] L. Zhang, H. Xing, X. Zhang, H. Li, Z. Liu, H. Shi, and Z. You, “Classification of beach dynamic geomorphic features in the shandong peninsula,” Mar Sci, vol. 47, no. 02, pp. 10–19, 2023.

[24] Y. Ding, J. Yu, and H. Cheng, “Beach morphodynamic characteristics and classifications on the straight coastal sectors in the west guangdong,” Journal of Geographical Sciences, vol. 30, no. 7, pp. 1179–1194, 2020.

[25] G. R. Lesser, J. v. Roelvink, J. T. M. van Kester, and G. Stelling, “Development and validation of a three-dimensional morphological model,” Coastal engineering, vol. 51, no. 8-9, pp. 883–915, 2004.

[26] J. C. Warner, C. R. Sherwood, R. P. Signell, C. K. Harris, and H. G. Arango, “Development of a three-dimensional, regional, coupled wave, current, and sediment-transport model,” Computers & geosciences, vol. 34, no. 10, pp. 1284–1306, 2008.

[27] D. B. Haidvogel, H. Arango, W. P. Budgell, B. D. Cornuelle, E. Curchitser, E. Di Lorenzo, K. Fennel, W. R. Geyer, A. J. Hermann, L. Lanerolle et al., “Ocean forecasting in terrain-following coordinates: Formulation and skill assessment of the regional ocean modeling system,” Journal of computational physics, vol. 227, no. 7, pp. 3595–3624, 2008.

[28] D. Roelvink, A. Reniers, A. Van Dongeren, J. Van Thiel de Vries, J. Lescinski, and R. McCall, “Xbeach model description and manual,” Unesco-IHE Institutefor Water Education, Deltares and Delft University of Tecnhology. Report June, vol. 21, no. 2010, p. 2, 2010.

[29] Y. Du, S. Pan, and Y. Chen, “Modelling the effect of wave overtopping on nearshore hydrodynamics and morphodynamics around shore-parallel breakwaters,” Coastal Engineering, vol. 57, no. 9, pp. 812–826, 2010.

[30] A. Kroon, J. C. Christiaanse, A. P. Luijendijk, M. A. d. Schipper, and R. Ranasinghe, “Parameter uncertainty in medium-term coastal morphodynamic modeling,” Scientific Reports, vol. 15, no. 1, p. 18471, 2025.

[31] T.-W. Hsu, S.-R. Liaw, S.-K. Wang, and S.-H. Ou, “Two-dimensional empirical eigenfunction model for the analysis and prediction of beach profile changes,” in Coastal Engineering 1986, 1986, pp. 1180–1195.

[32] E. Disdier, R. Almar, R. Benshila, M. Al Najar, R. Chassagne, D. Mukherjee, and D. G. Wilson, “Predicting beach profiles with machine learning from offshore wave reflection spectra,” Environmental Modelling & Software, vol. 183, p. 106221, 2025.

[33] J. Kim, T. Kim, S. Chang, J. Kim, and I. Kim, “Prediction of beach profile changes using spatiotemporal graph neural networks for beach morphological evolution modeling,” Ocean Engineering, vol. 340, p. 122282, 2025.

[34] C. K. Williams and C. E. Rasmussen, Gaussian processes for machine learning. MIT press Cambridge, MA, 2006, vol. 2, no. 3.

[35] G. Masselink and A. D. Short, “The effect of tide range on beach morphodynamics and morphology: a conceptual beach model,” Journal of coastal research, pp. 785–800, 1993.

[36] L. Ye and E. Keogh, “Time series shapelets: A new primitive for data mining,” in Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2009, pp. 947– 956.

[37] J. Lines, L. M. Davis, J. Hills, and A. Bagnall, “A shapelet transform for time series classification,” in Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2012, pp. 289–297.

[38] J. Hills, J. Lines, E. Baranauskas, J. Mapp, and A. Bagnall, “Classification of time series by shapelet transformation,” Data Mining and Knowledge Discovery, vol. 28, no. 4, pp. 851–881, 2014.

[39] J. Grabocka, N. Schilling, M. Wistuba, and L. Schmidt-Thieme, “Learning time-series shapelets,” in Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, 2014, pp. 392–401.

[40] A. Riazi and P. A. Slovinsky, “Subaerial beach profiles classification: An unsupervised deep learning approach,” Continental Shelf Research, vol. 226, p. 104508, 2021.

[41] J. Toms and M. Lesperance, “Piecewise regression: A tool for identifying ecological thresholds,” Ecology, vol. 84, no. 8, pp. 2034–2041, 2003.

[42] J. Hamilton, “A new approach to the economic analysis of nonstationary time series and the business cycle,” Econometrica, vol. 57, no. 2, pp. 357–384, 1989.

[43] T. Dietterich, “Ensemble methods in machine learning,” in Multiple Classifier Systems. Springer, 2000, pp. 1–15.

[44] T. Rakthanmanon, B. Campana, A. Mueen, G. Batista, B. Westover, Q. Zhu, J. Zakaria, and E. Keogh, “Addressing big data time series: Mining trillions of time series subsequences under dynamic time warping,” ACM Transactions on Knowledge Discovery from Data (TKDD), vol. 7, no. 3, pp. 1–31, 2013.

[45] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[46] A. Hermans, L. Beyer, and B. Leibe, “In defense of the triplet loss for person re-identification,” arXiv preprint arXiv:1703.07737, 2017.

[47] D. J. MacKay, “Bayesian interpolation,” Neural computation, vol. 4, no. 3, pp. 415–447, 1992.

[48] R. L. Folk and W. C. Ward, “Brazos river bar [texas]; a study in the significance of grain size parameters,” Journal of sedimentary research, vol. 27, no. 1, pp. 3–26, 1957.

[49] M. Benoit, F. Marcos, and F. Becq, “Development of a third generation shallow-water wave model with unstructured spatial meshing,” in Proceedings of the 25th International Conference on Coastal Engineering (ICCE), Orlando, Florida, USA, 1996, pp. 465–478.

[50] L. D. Wright and A. D. Short, “Morphodynamic variability of surf zones and beaches: a synthesis,” Marine geology, vol. 56, no. 1-4, pp. 93–118, 1984.

[51] C. J. Willmott and K. Matsuura, “Advantages of the mean absolute error (mae) over the root mean square error (rmse) in assessing average model performance,” Climate research, vol. 30, no. 1, pp. 79–82, 2005.

[52] T. Cover and P. Hart, “Nearest neighbor pattern classification,” IEEE transactions on information theory, vol. 13, no. 1, pp. 21–27, 1967.

[53] L. Breiman, “Random forests,” Machine learning, vol. 45, no. 1, pp. 5–32, 2001.

[54] C. Cortes and V. Vapnik, “Support-vector networks,” Machine learning, vol. 20, no. 3, pp. 273–297, 1995.
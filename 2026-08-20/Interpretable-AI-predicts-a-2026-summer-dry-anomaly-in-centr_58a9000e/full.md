# Interpretable AI predicts a 2026 summer dry anomaly in central China

Anran WANG <sup>1†</sup>, Wen SHI <sup>1†</sup>, Yong LUO <sup>1</sup>, Jianbin HUANG <sup>2,3,4</sup>, Lijuan CHEN <sup>5,6</sup>, Junhu ZHAO <sup>6</sup>, Weixin JIN <sup>7</sup>, and Huihui YUAN <sup>1</sup>

1 Department of Earth System Science, Tsinghua University, Beijing 100084

2 College of Resources and Environment, University of Chinese Academy of Sciences,

Beijing 101408

3 Beijing Yanshan Earth Critical Zone National Research Station, University ofChinese Academy ofSciences, Beijing 101408

4 College of Resources and Environment, University of Chinese Academy of Sciences, Beijing 100049

5 State Key Laboratory ofClimate System Prediction and Risk Management, National

Climate Center, China Meteorological Administration, Beijing 100081

6 China Meteorological Administration Key Laboratory for Climate Prediction Studies, National Climate Center, China Meteorological Administration, Beijing 100081 7 Microsoft, Beijing 100080

## Abstract

Seasonal precipitation anomalies are largely regulated by atmospheric circulation, which dynamical models predict with greater reliability than precipitation itself. Here, we employ a deep learning model that translates dynamical circulation predictions into precipitation estimates. Predictions initialized from March to May consistently indicate a dry anomaly over central China in summer 2026. Retrospective evaluations revealed higher predictive skill in the analogue years, which also tended to feature central equatorial Pacific warming persisting from the preceding winter into summer. This warming favors an anomalous cyclonic circulation over the western North Pacific–South China Sea–South China region, which induces northerly winds and moisture divergence that jointly suppress rainfall over central China. Supporting

this mechanism, layer-wise relevance propagation (LRP) independently identifies these northerly winds as the dominant driver of the prediction among all model inputs. Perturbation tests supported this attribution: removing LRP-identified features effectively eliminates the dry anomaly. Our framework thus provides physically interpretable explanations for AI-derived regional climate projections, facilitating evidence-based assessment before observational data become available.

## Introduction

Credible and skillful precipitation predictions issued ahead of the summer flood season in China could provide actionable lead time for water management, agricultural planning, and drought and flood preparedness<sup>1,2</sup>. Yet regional summer precipitation remains one of the most challenging targets of seasonal prediction owing to complex interactions among atmospheric, oceanic and land-surface processes<sup>3–6</sup>. Seasonal precipitation skill is generally modest and varies substantially with the prevailing climate state<sup>7</sup>, creating intermittent windows of opportunity for more skillful prediction<sup>8–11</sup>. Averaging skill over a hindcast period can obscure this state-dependent predictability and provide only limited guidance on the credibility of an individual real-time prediction<sup>12</sup>. Therefore, tracing a prediction to physically meaningful predictive signals supported by the historical record can help assess whether it provides a credible basis for issuing an early warning.

Such predictive signals can be investigated through historical climate diagnostics and analyses of model behavior. Historical analogue analyses identify past cases resembling the target anomaly, and composite analyses assess whether they share recurrent patterns of atmospheric circulation, moisture transport, or oceanic conditions<sup>13–15</sup>. At the model level, feature attribution methods in explainable artificial intelligence (XAI) identify the input regions and variables that contribute to a particular prediction<sup>9,16–22</sup>. Because attribution results can vary among methods, their reliability is commonly assessed through cross-method comparisons<sup>23,24</sup>. Perturbation tests examine whether changing the identified features alters the prediction as expected<sup>25,26</sup>. Evidence for the credibility of an individual prediction is strengthened when historical diagnostics and feature attribution converge on the same physically interpretable signals.

Dynamical–statistical bridging provides a suitable framework in which these forms of evidence can be brought together. Such approaches translate dynamical predictions of relatively predictable large-scale climate modes into regional precipitation anomalies through statistical relationships<sup>27–30</sup>. More recently, deep-learning models have expanded this framework by learning nonlinear relationships between multivariable circulation fields and regional precipitation<sup>31–33</sup>. This explicit circulation-to-precipitation mapping allows feature attribution to identify physically interpretable circulation signals underlying an individual prediction. Yet this interpretive capability has rarely been used to support physically grounded assessment when seasonal precipitation predictions are issued in real time<sup>11,34</sup>.

Here, we use a circulation-to-precipitation bridging model to generate real-time predictions of summer 2026 precipitation over China. Predictions initialized in March, April and May consistently indicate a dry anomaly over central China. We assess the credibility of the predicted anomaly by evaluating model skill in historical analogue years and examining whether the climate conditions associated with these years are also evident in 2026. We then use feature attribution to determine whether the model highlights circulation signals consistent with these climate features. We evaluate the stability of the attributions across methods and use LRP-guided perturbation tests to assess the relevance of the identified features to the predicted anomaly. Together, these analyses establish a framework for assessing AI-based seasonal precipitation predictions through convergent historical, physical and model-centered evidence available ahead of the target season.

## Results

## 1. Consistent predictions point to a central China dry anomaly in summer 2026

We generated real-time predictions of summer 2026 precipitation over China by applying the circulation-to-precipitation bridging model to dynamical circulation predictions initialized in March, April and May. All three predictions showed a broadly consistent spatial pattern, with below-normal precipitation across central China and above-normal precipitation over South China and parts of North China (Fig. 1a–c). The central China dry anomaly was the most prominent regional feature, with regional means of −34%, −29% and −38% in the three predictions, respectively. Although central China showed the largest anomaly magnitude, its inter-initialization range was comparatively small, whereas larger ranges occurred over parts of South and Northeast China (Fig. 1d). The location, spatial extent and magnitude of the central China dry anomaly therefore remained relatively stable as the initialization approached the target season, indicating the anomaly was not specific to a particular initialization month.

Cross-validation over 1993–2025 showed comparable skill between the bridging-model and dynamical multi-model ensembles (MMEs; Fig. 1e, f). The bridging-model MME yielded anomaly correlation coefficient (ACC) values of 0.140–0.161 and prediction scores (Ps) of 73.27–74.25% across the three initialization months. Relative to the dynamical MME, its largest gains occurred for the March initialization, whereas the two MMEs performed similarly in May. These results motivate a case-specific credibility assessment of the predicted dry anomaly over central China.

Initialized in March  
![](images/b426bda521d642a676b511cce4b2e92e85d5ad5e0e4a0ab51c2e437fbc1ba114.jpg)

![](images/7f3f8e955e8429fcdd55f33c309bc78dc9e288034eabd5d79e8a29e746ffa2ae.jpg)

Initialized in May  
![](images/9d79f097fdaf937c5134fde895accc039bf11519c8a0caca94d6d88d657e6474.jpg)

![](images/df3a7380b8b11eec78359a15e01bec16945fc88f9bfbf6d292dcd02e5c857e73.jpg)

![](images/9cfbc0c376976f5fd8ac08e0629e0a81d2528ec60031985bb1b5fae3c3fe9a3f.jpg)

(f)  
![](images/9a08d96b51e9256816c625536b972f117d48070fb1407ffefff8fc57f05be18b.jpg)  
Fig. 1: Summer 2026 precipitation predictions over China and cross-validated hindcast skill. a–c, Precipitation anomalies (%) relative to the 1991–2020 climatology predicted by the bridging-model multi-model ensemble (MME) for June–August 2026, based on the March, April and May initializations, respectively. d, Inter-initialization range at each station, calculated as the difference between the maximum and minimum anomalies among the three predictions. Red boxes in a–d delineate central China $( 2 8 ^ { \circ } - 3 6 ^ { \circ } \mathrm { N }$ $1 0 3 ^ { \circ } \mathrm { - } 1 1 3 ^ { \circ } \mathrm { E } )$ , and the values in a–c indicate the corresponding regional-mean precipitation anomalies. e, f, Historical cross-validated anomaly correlation coefficient (ACC; e) and prediction score (Ps; f) for summer precipitation predictions initialized in March, April and May during 1993–2025. Gray and hatched bars represent the raw dynamical MME and the bridging-model MME, respectively. Error bars indicate bootstrap 95% confidence intervals for the mean

cross-validated scores. Values above the bars indicate the differences between the bridging-model and raw dynamical MMEs.

## 2. Higher prediction skill in historical dry-anomaly years

The historical cases most closely resembling the predicted 2026 precipitation pattern were highly consistent across initializations. For each initialization, the observed summers during 1993–2025 were ranked separately using four metrics of similarity to the corresponding prediction. The four ranks were then summed for each year, and the seven years with the lowest rank sums were retained for analysis (Methods and Supplementary Table 1). Six years—1994, 1997, 2001, 2006, 2011 and 2015—were common to all three sets. The April and May predictions yielded identical sets, additionally including 2002, whereas the March prediction included 2022 instead.

We next compared cross-validated prediction skill between the selected dry-anomaly years and the remaining 26 years to examine how the model had performed in historical cases resembling the 2026 prediction (Fig. 2). Across all three initializations and both year groups, ERA5-driven outputs attained markedly higher ACC and Ps than predictions driven by dynamical-model circulation, suggesting that errors in the predicted circulation partly constrained real-time prediction skill. Despite their different overall skill levels, both input settings showed a common conditional pattern: median skill was higher in the dry-anomaly years in every comparison except ACC for the May-initialized, dynamically driven predictions, for which the dry-year median was slightly lower and the two distributions overlapped substantially. These comparisons suggest that the model may perform better for precipitation states resembling the predicted 2026 dry anomaly than its average historical skill would imply. Because the bridging-model predictions are derived entirely from atmospheric circulation, the higher skill in the dry-anomaly years suggests that a recurrent circulation pattern may underlie the model’s prediction of central China drying.

![](images/de1e61ad0adfc370287632aea0ebbc7f57291c5b1f1d7377e41c08a2a7885dd8.jpg)

![](images/03f1bf1a57e43133d4ecc14011b07029df5718e58069e5c2641a40207114afca.jpg)

![](images/90b911d07fcb53c4955b7e72c5a8ba0c05aa207c3f62ef24166b569489bffc44.jpg)

![](images/7aab0f6318cd27f678c2a0d8a35ab43bbbb355379dd395fa9479c1694b255b49.jpg)

![](images/eb2bc5b5eef46dee07974ff0a00be692f1f68da84108c64ba4adf426d4900463.jpg)

![](images/88a529c6d01ed9c22f79bb9a39918fce8061f4b3c79ce9761d88c8e04c42ae60.jpg)  
Fig. 2: Prediction skill in analogue years for the predicted 2026 dry anomaly. a–c, ACC of summer precipitation outputs for the March, April and May initializations during 1993–2025, grouped into the 7 dry-anomaly analogue years selected for each initialization and the remaining 26 years. d–f, corresponding Ps. Gray boxplots represent bridging-model outputs driven by ERA5 circulation, whereas blue boxplots represent bridging-model predictions driven by dynamical-model circulation. Within each input setting, darker and lighter shades denote analogue years and other years, respectively.

## 3. Climate conditions conducive to a dry anomaly over central China

We composited the summer circulation across the seven analogue years selected for the May-initialized bridging-model prediction and compared them with the May-initialized dynamical MME circulation prediction for summer 2026 (Fig. 3). The March- and April-initialized predictions showed similar spatial patterns (Supplementary Figs. 1 and 2).

In the upper troposphere, the analogue composite showed a northward displacement of the East Asian subtropical westerly jet (EASWJ), with strengthened 200-hPa zonal winds north of the climatological jet axis and weakened winds to its south (Fig. 3e). The May-initialized prediction reproduced this meridional dipole (Fig.

3i). Previous observational studies have associated a poleward-displaced EASWJ with reduced summer rainfall over the Yangtze–Huaihe River Valley and a redistribution of rainfall toward South and Northeast China<sup>35–37</sup>.

In the middle troposphere, the composite featured a pronounced positive 500-hPa geopotential-height anomaly extending from Mongolia to south of Lake Baikal (Fig. 3f). This type of continental high anomaly has been associated with enhanced subsidence south of Lake Baikal, anomalous low-level northerly winds over eastern China, and a weakened East Asian summer monsoon with reduced moisture transport<sup>38</sup>. The May-initialized prediction reproduced the positive-height anomaly but placed its center farther west (Fig. 3j).

At 700 hPa, both the dry-anomaly composite and the 2026 prediction featured a cyclonic circulation anomaly over the western North Pacific that extended westward into the South China Sea–South China region (Fig. 3g, k). Central China lay on the northwestern flank of this circulation, where anomalous northeasterly flow was stronger in the 2026 prediction than in the historical composite. Consistent with this circulation, the 850–500-hPa layer-integrated moisture-flux anomaly opposed the climatological moisture transport into central China, and the target region showed anomalous moisture-flux divergence (Fig. 3d, h, l). The 2026 prediction also showed anomalous moisture-flux convergence over South China, consistent with the above-normal precipitation predicted there (Figs. 3l and 1).

Among these circulation anomalies, the western North Pacific–South China Sea –South China region cyclonic anomaly most directly favored central China drying by weakening moisture transport into the target region, with the stronger anomalous winds associated with this circulation in the 2026 prediction suggesting that its drying influence may be stronger than in the composite of the seven analog years.

![](images/f0eeddf8bccaf185a198f079fb5f8262d0927e05f69c85f7b5921522f32d82dd.jpg)  
Fig. 3: Circulation patterns associated with the predicted central China dry anomaly. a–d, ERA5 JJA climatology for $1 9 9 3 { - } 2 0 1 6 . ~ \mathbf { e { - } h } .$ , Composite anomalies for the seven historical dry-anomaly years selected for the May initialization. i–l, May-initialized dynamical MME anomalies for JJA 2026. Columns show 200-hPa zonal wind $( \mathrm { U } 2 \quad 0 \quad 0 \quad ;$ shading and contours; $\mathrm { ~ m ~ s ^ { - 1 } ) }$ , 500-hPa geopotential height $( Z _ { 5 } \quad \ : _ { 0 } \quad \ : _ { 0 } \quad ;$ shading and contours; gpm), 700-hPa specific humidity $\left( \mathsf { q } ^ { _ { 7 } } \quad 0 \quad 0 \right)$ ; shading; g $\mathbf { k g ^ { - 1 } } )$ and horizontal winds (vectors; m $\mathbf { S } ^ { - 1 } )$ , and 850–500-hPa integrated vapor transport calculated from the 850-, 700- and 500-hPa levels (IVT; vectors; kg $\mathrm { m } ^ { - 1 } \ \mathrm { s } ^ { - 1 } )$ and its divergence (div.; shading; $1 0 ^ { - 5 } \quad \mathrm { k g ~ m ^ { - 2 } ~ s ^ { - 1 } } )$ . Red boxes delineate the central China target region. Blue 5880-gpm contours in b, f and j delineate the western North Pacific subtropical high. In d, IVT vectors are shown only where their magnitude is $\ge 5 0$ kg $\mathrm { m ^ { - 1 } ~ s ^ { - 1 } }$ . In h, vectors are retained where their magnitude is ${ \ge } 6$ kg $\mathrm { m ^ { - 1 } ~ s ^ { - 1 } }$ and at least one horizontal component differs from zero in a two-sided one-sample t-test $( \mathbf { p } ~ < ~ 0 . 2 0 )$ ; divergence shading is restricted to grid points satisfying the same significance threshold. In l, vectors are shown where their magnitude is $\geq 6 \mathrm { \ k g \ m ^ { - 1 } \ s ^ { - 1 } } .$ , and divergence shading is retained where at least six of the seven models agree on the sign of the MME-mean divergence anomaly.

The recurrent circulation pattern was accompanied by a tendency towards warmer conditions in the central equatorial Pacific. Niño-4 SST increased markedly from the preceding winter into summer in six of the seven historical dry-anomaly years, whereas the central equatorial Pacific was already anomalously warm in the preceding winter and remained so throughout 2015 (Fig. 4). In 2026, Niño-4 rose continuously from a negative anomaly in December 2025 to above $1 ^ { \circ } \mathrm { C }$ by May 2026. Recent climate monitoring indicates that El Niño conditions have developed in the tropical Pacific<sup>39–41</sup>, and forecasts further suggest that the developing El Niño intensity could become exceptionally strong<sup>42</sup>. Although the strongest warming in 2026 was located farther east, the warming extended into the Niño-4 region. Previous studies found that the relationship between developing El Niño and reduced summer rainfall over central China became significant after the late 1980s, owing to the increased frequency of central-Pacific El Niño events<sup>43</sup>. When SST warming is centered over the central equatorial Pacific, the associated Walker circulation anomaly shifts westward, extending its descending branch to the Maritime Continent and favoring a lower-tropospheric cyclonic circulation anomaly over the western North Pacific–South China Sea<sup>44</sup>. The cyclonic circulation weakens northward monsoon moisture transport<sup>43</sup>. The pronounced Niño-4 warming in 2026 is consistent with this mechanism and the predicted reduction in summer rainfall over central China.

![](images/75651ee7afe875bc40ef967f30c9403cdcd39f231c99b9eebe310cf5ca13090d.jpg)

![](images/c9e81fd36df1ada623970a0eb60653d7bad29810dcd917f3483e1cac5cf129ca.jpg)

![](images/de88399448e15e6bfdf9fecad941c318b80a875a6d9b31800c8d4989c3856da7.jpg)

![](images/d948c3f14a442515f81f5e9d5f49c4a63fad8a0aafcda635865ffd70f7078b18.jpg)

![](images/f5ef3c97bef8867e229c3f36cbdf97cd1dfec7039c2846e1860f616c231c7a62.jpg)

![](images/cd9baf28dd4a131e8adb22f6b607fc5a061a3f45dd80f372aa0eb1e3afe08733.jpg)

![](images/cf32c19a5d045b241c5f902e8db5acd7ac9f3a5c3d77f2778bcac7c6bd6b6505.jpg)

![](images/fd3d270ab28959871664f8be73d31bffa9e46890b02078324a256977049ca144.jpg)  
Fig. 4: Niño-4 SST anomaly evolution in 2026 and historical analogue years. Monthly Niño-4 SST anomalies (°C) from December of the preceding year to August of each of the seven historical dry-anomaly years selected for the May initialization. The 2026 series spans December 2025–May 2026. Gray shading marks the JJA target season.

## 4. Physically coherentprediction signals identified by LRP

To examine whether the bridging model drew on the circulation features highlighted in the previous section, we applied LRP to the May-initialized prediction for summer 2026 (Fig. 5). We defined the attribution target as the predicted regional-mean precipitation anomaly across the central China stations. The linear mapping from the predicted principal component (PC) coefficients through empirical orthogonal function (EOF) reconstruction and regional averaging was incorporated as the final target layer for relevance propagation, with each mode weighted by its contribution to the regional anomaly. This formulation ties the resulting attribution directly to the regional prediction of interest, avoiding separate interpretation and subsequent combination of mode-specific attribution maps.

A physically coherent attribution pattern was evident in the meridional wind fields at 500, 700 and 850 hPa (Fig. 5d, g, j, n). At all three levels, dry-supporting relevance was concentrated on the northerly anomalies along the northwestern flank of the western North Pacific–South China Sea–South China region cyclonic circulation, with a particularly pronounced signal at 700 hPa. Together, the three meridional-wind predictors accounted for 37.5% of the area-weighted dry-supporting relevance. These northerly anomalies oppose the climatological moisture transport into central China. The predicted fields showed widespread positive specific-humidity anomalies at 500, 700 and 850 hPa, while the corresponding predictors received little dry-supporting relevance (Fig. 5e, h, k, n). These results indicate that the model associated the predicted rainfall deficit primarily with circulation-induced weakening of moisture transport into central China rather than with limited moisture availability. The spatial distribution of this attribution closely matched the circulation pathway identified independently from the analogue composites and the 2026 dynamical prediction.

Beyond the meridional-wind signals, dry-supporting relevance in 2-m air temperature (T2m) was concentrated over an anomalously warm region upstream of central China (Fig. 5m, n). This spatial correspondence indicates that the model associated the temperature signal with the predicted dry anomaly. The 500-hPa geopotential height (Z500) field also received dry-supporting relevance (Fig. 5b, n), concentrated mainly along strong geopotential-height gradients on the WNPSH periphery. However, the alignment of its edges with the Vision Transformer (ViT) patch grid suggests that some of the spatial structure may arise from tokenization artifacts.

![](images/047bce4b48057abaa13eff89f5e599c51c07f6fdee0f18df2da9009f27c47e76.jpg)  
Fig. 5: LRP-identified circulation signals supporting the predicted summer 2026 dry anomaly over central China. a–m, Spatial distributions of dry-supporting negative LRP relevance for the 13 circulation predictors, calculated for the regional-mean precipitation anomaly over central China from the May initialization. High negative relevance indicates a stronger contribution toward the predicted dry anomaly; only negative, dry-supporting relevance is shown, and the fields are smoothed using a Gaussian filter with σ=1 grid cell. Contours denote the corresponding predicted anomalies for scalar fields, and vectors show horizontal wind anomalies at 500, 700 and 850 hPa in the paired u and v panels. The red box

Integrated Gradients (IG) and Guided Integrated Gradients (Guided IG) broadly reproduced the May-initialized wind-field attribution patterns (Supplementary Figs. 3 and 4). Cross-method agreement in predictor rankings remained high across all three initializations (Supplementary Table 2).

marks the target region. n, Relative contribution of each predictor, calculated from the spatially integrated negative relevance. Here, u, v, q and z denote zonal wind, meridional wind, specific humidity and geopotential height, with pressure levels indicated by the subscripts; msl and t2m denote mean sea-level pressure and 2-m air temperature.

## 5. Perturbation tests support the faithfulness of LRP explanations

To test whether the circulation signals highlighted by LRP materially influenced the predicted dry anomaly, we conducted LRP-guided Removal and Retention tests for the May initialization (Fig. 6). Perturbation responses were evaluated through the regional LRP target, $\Phi _ { R }$ , which is equivalent to the mean standardized precipitation anomaly across central China stations; positive values indicate wet anomalies and negative values indicate dry anomalies. For 2026 and its seven analog years, case-specific masks were constructed from the variable–grid-cell features with the top 5% of dry-supporting relevance and their neighboring grid cells (Fig. 6a). The masked features were zeroed out in the Removal test and retained exclusively in the Retention test. Each test was repeated with 100 random masks perturbing the same fraction of the input. The prespecified one-sided comparisons tested whether Removal yielded a higher $\Phi _ { R }$ and Retention a lower $\Phi _ { R }$ than the corresponding random-mask controls.

LRP-guided Removal reversed the sign of $\Phi _ { R }$ from negative to positive for 2026 and all seven analog years, whereas random Removal left $\Phi _ { R }$ close to the baselines (Fig. 6b). Conversely, LRP-guided Retention preserved negative $\Phi _ { R }$ and produced more negative values than the corresponding baselines in every case, while random Retention yielded values clustered near zero (Fig. 6c). None of the 100 random perturbations produced a response more extreme than the LRP-guided result in the prespecified direction for any of the eight cases (one-sided empirical tests, all $P < \theta . O 7 )$ . The disappearance of the dry signal after Removal and its persistence under Retention indicate that the circulation signals supporting the predicted dry anomaly were concentrated in the input features highlighted by LRP, providing evidence that the attribution faithfully reflected the model’s dependence on these inputs.

(a)  
![](images/90dbd4f1f79f37a3d281e48e68bd36a7f9444aa90e2b24f8c64f9dad59cf16f5.jpg)

![](images/9577d5761c57e35efc52183c55c8a448c7ca707826b747632e6c76e474699397.jpg)

![](images/93895bc6614231a17c4f5274a951ce7c980111515375b6609df6022d05824573.jpg)

![](images/4bb28735ee23f222d7b766e986e684d4b864b39edab2c06af4fcfaaaba62f558.jpg)

(c)  
![](images/cd180cc64e48ada31a95cf8d11a226d3a4e14ea182749101f22e5ebe6248ab96.jpg)  
Fig. 6: LRP-guided perturbation tests of central China dry-anomaly predictions. a, Schematic of the zero-out and keep-only masks applied across the 13 circulation input fields for the May initialization. The LRP-guided masks comprise the variable–grid-cell features with the top 5% of dry-supporting relevance and their neighboring grid cells; the red box delineates central China. b, c, $\Phi _ { R }$ for 2026 and its seven analog years under the Removal (b) and Retention (c) tests. Open red circles show the baselines, colored dots the LRP-guided results, and gray boxplots the distributions from 100 random-mask controls perturbing the same fraction of the input. $\Phi _ { R }$ denotes the regional LRP target.

This study provides a prospective, case-specific assessment of the predicted summer 2026 dry anomaly over central China using only information available at each prediction start time. Historical evaluation showed generally higher skill in the selected analogue years than in other years. The predicted circulation also reproduced features associated with suppressed precipitation over central China, including a lower-tropospheric cyclonic anomaly over the western North Pacific–South China Sea –South China region, northerly anomalies on its northwestern flank, weakened monsoon moisture transport, and moisture-flux divergence over central China. LRP indicated that the model relied strongly on these northerly anomalies. In perturbation tests, removing the LRP-selected features reversed the predicted anomaly from dry to wet, whereas retaining these features alone intensified the dry anomaly, supporting the faithfulness of the LRP attribution. In this study, we defined the LRP target as the predicted regional-mean precipitation anomaly over central China. The network outputs 512 PC coefficients, and the regional anomaly is reconstructed from all 512 associated EOF modes. Attributing each PC separately would not directly explain why central China was predicted to be dry and would require combining many mode-specific maps. Thus, we included the linear reconstruction from all 512 PCs to the regional mean in the relevance-propagation pathway. The resulting map integrates all modes into a single region-specific attribution that can be compared directly with the circulation patterns. This formulation may also be useful for regional interpretation of other spatial prediction models with reduced-order outputs. LRP attributes the model output to input features but does not establish causal relationships in the climate system. Evidence was strongest for the mid-to-lower-tropospheric northerly signal, which agreed with climate diagnostics and was broadly reproduced by IG and Guided IG. Secondary signals require caution. In particular, T2m is a rapidly responding component of the coupled land–atmosphere

## Discussion

system. Reduced precipitation and cloud cover, together with diminished soil moisture and evaporative cooling, can increase near-surface temperature, producing a coupled warm–dry surface state. The bridging model can exploit this covariation, and the dry-supporting T2m relevance over northwestern China may indicate that such a coupled warm–dry state contributed predictive information. It does not demonstrate that warming over northwestern China directly caused the predicted precipitation deficit over central China. The patch-aligned Z500 relevance also suggests that its detailed spatial pattern may partly reflect ViT tokenization. The perturbation tests confirmed the joint importance of the selected features to the model prediction, but did not establish them as causal drivers in the climate system.

The framework examines whether similar historical cases were more predictable, whether the predicted circulation supports a plausible physical pathway, and whether the model relies on features associated with that pathway. For summer 2026, all three analyses provided supportive evidence. The value of this framework lies in making the evidential basis of a prediction explicit before observations become available, rather than guaranteeing that the prediction will verify. The large-scale conditions captured by the framework may modulate subseasonal rainfall variability and the likelihood of extreme events, but the model does not explicitly resolve the occurrence, timing or intensity of individual events. The predicted dry anomaly therefore represents a seasonal tendency toward drier conditions and does not rule out the possibility of heavy rainfall that could substantially affect the observed summer precipitation total. More broadly, it provides physically grounded explanations for seasonal predictions of potentially high-impact climate anomalies and supports the transparent and evidence-based use of AI in pre-season climate-risk assessment.

Method

1. Data

Monthly station precipitation observations for 1961–2025 were calculated from the China Precipitation Daily Dataset $( V 3 . 0 ) ^ { 4 5 }$ , provided by the National Meteorological Information Center of the China Meteorological Administration. These observations served as the predictand for transfer learning, the reference for cross-validation evaluation, and the basis for identifying historical analogue years. Monthly Niño-4 index were obtained from the NOAA Physical Sciences Laboratory climate-index archive. The index is produced by the NOAA Climate Prediction Center from ERSSTv5<sup>46</sup> and represents the area-mean SST anomaly over $5 ^ { \circ } \mathrm { S } { - } 5 ^ { \circ } \mathrm { N } .$ $1 6 0 ^ { \circ } \mathrm { E } { - } 1 5 0 ^ { \circ } \mathrm { W }$

Monthly mean atmospheric fields for 1961–2025 were obtained from $\mathrm { E R A } 5 ^ { 4 7 }$ The fields were interpolated onto 1° × 1°grid covering $6 4 . 5 ^ { \circ } - 1 5 9 . 5 ^ { \circ } \mathrm { E }$ and $3 . 5 ^ { \circ } \mathrm { S } { - } 5 9 . 5 ^ { \circ } \mathrm { N }$ . We used 13 variables: zonal wind at 200 hPa; geopotential height, zonal and meridional winds, and specific humidity at 500 hPa; zonal and meridional winds and specific humidity at 700 and 850 hPa; mean sea-level pressure; and 2-m air temperature. ERA5 circulation fields served as model inputs during transfer learning and retrospective evaluation. They were also used to diagnose circulation and moisture transport in the analogue years.

Seasonal prediction data were obtained from the monthly pressure-level and single-level datasets in the Copernicus Climate Change Service (C3S) Climate Data Store<sup>48,49</sup>. The hindcast for 1993–2025 comprised 27 forecast-system versions from eight forecasting centers; the systems and their ensemble sizes are listed in Table 1, with ensemble sizes given in parentheses. The same 13 atmospheric variables and the corresponding precipitation formed paired inputs and targets for pretraining. Because the operational system composition changed during spring 2026, the March initialization used seven systems (BOM-2, CMCC-4, DWD-22, ECCC-5, ECMWF-51, Météo-France-9 and UKMO-605), whereas the April and May initializations used eight (BOM-2, CMCC-4, DWD-22, ECCC-5, ECMWF-51, JMA-4, Météo-France-9 and UKMO-610). Each system was processed separately,

and the resulting precipitation predictions were averaged with equal system weights to form the MME prediction. All monthly data were converted to overlapping three-month means, yielding 12 seasonal samples per year.
<table><tr><td>forecasting center</td><td>model</td></tr><tr><td>ECMWF</td><td>SEAS5 (25), SEAS5.1 (25)</td></tr><tr><td>MF</td><td>Météo-France System 6 (25), 7 (25), 8 (25), 9 (31)</td></tr><tr><td rowspan="3">UKMO</td><td>GloSea5-GC2 12 (28), 13 (28), GloSea5-GC2-LI 14 (28),</td></tr><tr><td>15 (28), GloSea6 600 (28), 601 (28), 602 (28), 603 (28), 604</td></tr><tr><td>(28)</td></tr><tr><td>CMCC</td><td>CMCC-SPS 3 (40), 3.5 (40), 4 (30)</td></tr><tr><td>DWD</td><td>GCFS 2.0 (30), 2.1 (30), 2.2(30)</td></tr><tr><td>ECCC</td><td>GEM5-NEMO (10), CanESM5.1p1bc (20),</td></tr><tr><td></td><td>GEM5.2-NEMO (20)</td></tr><tr><td>JMA</td><td>JMA-CPS 2 (10), 3 (10)</td></tr><tr><td>BOM</td><td>ACCESS-S2 (27)</td></tr></table>

Table 1. Forecasting centers and C3S seasonal prediction systems used for model pretraining.

## 2. Bridging-model development and evaluation

The circulation-to-precipitation bridging model maps 13 seasonal circulation-anomaly fields to station precipitation anomalies (Fig. 7). The model is an updated version of the circulation-to-precipitation bridging model developed by Jin et al.<sup>32</sup>, with a ViT architecture replacing the original convolutional backbone. The multichannel fields were divided into patches and transformed into spatial tokens through convolutional patch embedding and bottleneck projection. The target three-month season was embedded as a classification (CLS) token, which was prepended to the spatial-token sequence and used to condition feature extraction through adaptive layer normalization. After positional embeddings were added, the sequence was processed by six ViT blocks. The updated CLS token was passed through root mean square normalization, dropout and two fully connected layers to predict the PC coefficients of the leading 512 EOF modes. Station precipitation anomalies were then reconstructed from the predicted coefficients and the EOF basis.

Because each three-month window yields only one seasonal sample, the observation dataset is relatively small. The model was therefore pretrained using paired circulation and precipitation anomalies from dynamical models. During transfer learning, the final two linear layers were fitted by ridge regression using ERA5 circulation anomalies and observed precipitation anomalies.

![](images/10d7af9bd5096ad50295cd6f3875cba63bce001482004ea80651cbd8f1ea4add.jpg)  
Fig. 7: Architecture of the circulation-to-precipitation bridging model. Spatial information is integrated within each ViT block via Multi-Head Self-Attention (MHA), while target-season information from the CLS token modulates the main branch of the neural network through shifting, scaling, and gating. Purple trapezoids indicate the EOF projection used to derive the PC targets and reconstruct station precipitation anomalies. $P _ { m } ,$ , positional embedding; CLS, classification; PC, principal component.

Cross-validation for 1993–2025 used six contiguous year blocks. In each fold, one block was held out for testing, the preceding block in cyclic order for validation, and the remaining four for training. The same splits were applied during pretraining and transfer learning. Predictions for the six test blocks formed the complete cross-validated record.

Prediction performance was evaluated using the spatial ACC and $P _ { s }$ . For � stations, ACC was calculated as

$$
A C C = \frac { \sum _ { i = 7 } ^ { N } ( p _ { i } - \overline { { p } } ) ( o _ { i } - \overline { { o } } ) } { \sqrt { \sum _ { i = 7 } ^ { N } ( p _ { i } - \overline { { p } } ) ^ { 2 } \sum _ { i = 7 } ^ { N } ( o _ { i } - \overline { { o } } ) ^ { 2 } } }
$$

where $p _ { i }$ and $o _ { i }$ are the predicted and observed precipitation anomaly percentages at station �, and the overbars denote spatial means. Following the National Climate Center specification<sup>50</sup>, the $P _ { s }$ is an empirical operational metric that combines anomaly-grade accuracy with grade-dependent weighting of precipitation anomalies:

$$
P _ { s } = \frac { 2 \times N _ { o } + 2 \times N _ { \mathrm { ~ } \mathrm { ~ } \mathrm { ~ + ~ } } 4 \times N _ { \mathrm { ~ } \mathrm { ~ } ^ { ~ } } } { N + N _ { o } + 2 \times N _ { \mathrm { ~ } \mathrm { ~ } \mathrm { ~ + ~ } } 4 \times N _ { \mathrm { ~ } \mathrm { ~ } ^ { ~ } } + \mathsf { M } } \times \mathrm { ~ } 7 0 0 \%
$$

Here, � is the total number of stations; $N _ { O }$ denotes correct anomaly-sign predictions, $N _ { \mathit { 1 } }$ and $N _ { 2 }$ denote correctly predicted level-1 and level-2 anomalies, and � denotes the number of stations at which the observed precipitation anomaly was at least 100%, but the predicted anomaly failed to reach the second-level positive-anomaly category. Level-2 and level-1 anomalies correspond to $2 0 \% \leq$ $| \Delta R | < 5 0 \%$ and $| \Delta R | \ge 5 \time 1 0 \%$

## 3. Identification ofhistorical analogue years

For each initialization, observed JJA precipitation anomalies during 1993–2025 were compared with the corresponding 2026 prediction using four metrics: all-station ACC $\left( \mathsf { A C C } _ { \mathsf { a l l } } \right)$ , Central China ACC $\left( \mathsf { A C C } _ { \mathsf { C C } } \right)$ , dry coverage ( DC ), and dry-intensity distance ( DID ). Let $P _ { y }$ denote the observed precipitation anomaly percentage at Central China stations in year $y , \widehat { P }$ the corresponding prediction for 2026. DC and DID were calculated as

$$
\mathsf { D C } _ { \mathsf { y } } = \frac { N _ { y } ^ { - } } { N _ { C C } } \times \ 7 0 0 \% , \ \mathsf { D I D } _ { \mathsf { y } } = \left| P _ { y } - \widehat { P } \right|
$$

where $N _ { y } ^ { - }$ is the number of Central China stations with negative anomalies and $N _ { C C }$ is the total number of stations in the region. Years were ranked separately for

each metric, with rank 1 assigned to the closest match: larger values for the two ACCs and DC, but smaller values for DID. The overall rank sum was

$$
S _ { y } = r _ { y } ( \mathsf { A C C } _ { \mathsf { a l l } } ) + r _ { y } ( \mathsf { A C C } _ { \mathsf { c c } } ) + r _ { y } ( \mathsf { D C } ) + r _ { y } ( \mathsf { D l D } )
$$

The seven years with the smallest $S _ { y }$ were retained as historical analogues, closely corresponding to the top 20% of ranked 33 years. Ties were resolved by smaller DID, followed by higher Central China and all-station ACCs.

## 4. Layer-wise relevance propagation and perturbation tests

LRP attributes a scalar neural-network output to its input features by redistributing relevance backward through the network using layer-specific rules<sup>51</sup>. This redistribution approximately conserves total relevance between adjacent layers, yielding signed scores that indicate features supporting or opposing the selected output. In climate science, LRP has been used to interpret the spatial patterns underlying neural-network predictions, identify indicators of externally forced change, and diagnose climate states associated with enhanced predictability<sup>52–54,9</sup>.

Here, we used LRP to trace the circulation signals contributing to the predicted central China dry anomaly. To accommodate the ViT architecture, we adopted the CP-LRP relevance-propagation scheme<sup>55</sup>. Within each self-attention layer, attention weights were held fixed and relevance was propagated through the value path, without redistribution through the query–key scores or softmax operation. Normalization and positional encoding were treated as pass-through operations. We used a γ-rule for the convolutional patch-embedding layers and an ε-rule for linear layers. LRP requires a scalar explanation target, whereas the bridging model predicts $K = 5 7 2$ precipitation PC coefficients. We therefore defined the target as the predicted mean standardized precipitation anomaly over the central China region �:

$$
\Phi _ { R } \big ( x \big ) = \frac { \eta } { N _ { R } } \sum _ { s \in R } \widehat { P } _ { s } ( x ) = \sum _ { k = \eta } ^ { K } \omega _ { R , k } \widehat { y } _ { k } ( \mathsf { x } )
$$

$$
\omega _ { R , k } = \frac { \jmath } { N _ { R } } \sum _ { s \in R } E O F _ { s , k }
$$

where $N _ { R }$ is the number of stations within �, � denotes the circulation input, $y _ { k }$ is the predicted coefficient of the �th EOF mode, and $\omega _ { R , k }$ is the mean loading of that mode across the $N _ { R }$ stations in �. This reconstruction was implemented as a fixed linear layer, allowing relevance to propagate from the regional target through the PC outputs to the circulation inputs. For visualization, only negative relevance values were retained, because they contributed toward a more negative $\Phi _ { R }$ . The relevance field for each predictor was smoothed independently using a two-dimensional Gaussian filter with $\sigma = I \ g r i d$ cell. Predictor contributions were calculated by cosine-latitude-weighted integration of the dry-supporting relevance and normalized across the 13 predictors.

LRP-guided perturbation tests were conducted for the May-initialized 2026 prediction and the seven analogue years. Dry-supporting relevance values were pooled across all grid cells and all 13 predictors, and the strongest 5% were selected using a single global threshold. The resulting mask was expanded by one grid cell in each spatial direction. In the Removal test, the selected input elements were set to zero; in the Retention test, only the selected elements were retained. The increase in $\Phi _ { R }$ after Removal and the retained $\Phi _ { R }$ in the Retention test were compared with results from $B = 7 0 0$ random masks containing the same number of input elements as the expanded LRP mask. One-sided empirical � values were calculated as

$$
P = ( r + 7 ) / ( B + 7 )
$$

For the Removal test, � was the number of random masks producing an equal or larger increase in $\Phi _ { R }$ than the LRP-guided mask; for the Retention test, it was the number producing an equal or lower retained $\Phi _ { R }$ . The smallest attainable � value was therefore 1/101 (<0.01).

## 5. Integrated Gradients and Guided Integrated Gradients

Like LRP, IG and Guided IG produce signed feature-level attributions for a specified scalar output. They were therefore used to assess whether the main dry-supporting signals identified by LRP were reproduced by methods based on different attribution principles. IG integrates gradients along a straight path from a reference state to the input<sup>56</sup>. Guided IG follows an adaptive path designed to reduce the accumulation of noisy gradients<sup>57</sup>. Both methods were applied to the frozen ViT using the regional attribution target $\Phi _ { R }$ . Because the inputs were standardized anomalies, a zero field represented the climatology of all predictors. This field served as the common baseline for evaluating how input anomalies changed the regional prediction. The same baseline and attribution procedures were used for predictions initialized in March, April and May. Negative values indicate dry-supporting contributions. The IG and Guided IG attribution fields in Supplementary Figs. 3 and 4 were smoothed for visualization in the same manner as the LRP fields. Cross-method agreement in predictor rankings and attribution patterns is shown in Supplementary Table 2.

## Reference

1. Golding, N. et al. Co-development of a seasonal rainfall forecast service: Supporting flood risk management for the Yangtze River basin. Clim. Risk Manag. 23, 43–49 (2019).

2. Bruno Soares, M., Daly, M. & Dessai, S. Assessing the value of seasonal climate forecasts for decision-making. Wiley Interdiscip. Rev. Clim. Change 9, e523 (2018).

3. Shi, P. et al. Significant land contributions to interannual predictability of East Asian summer monsoon rainfall. Earths Future 9, e2020EF001762 (2021).

4. Wang, B. et al. Advancing Asian monsoon climate prediction under global change: Progress, challenges, and outlook. Adv. Atmos. Sci. 43, 1–29 (2026).

5. Ma, J. et al. Skillful seasonal predictions of continental East-Asian summer rainfall by integrating its spatio-temporal evolution. Nat. Commun. 16, 273 (2025).

6. He, C. et al. How much of the interannual variability of East Asian summer rainfall is forced by SST? Clim. Dyn. 47, 555–565 (2016).

7. Pegion, K. & Kumar, A. Does an ENSO-conditional skill mask improve seasonal predictions? Mon. Weather Rev. 141, 4515–4533 (2013).

8. Mariotti, A. et al. Windows of opportunity for skillful forecasts subseasonal to seasonal and beyond. Bull. Am. Meteorol. Soc. 101, E608–E625 (2020).

9. Mayer, K. J. & Barnes, E. A. Subseasonal forecasts of opportunity identified by an explainable neural network. Geophys. Res. Lett. 48, e2020GL092092 (2021).

10. Arcodia, M. C. et al. Assessing decadal variability of subseasonal forecasts of opportunity using explainable AI. Environ. Res. Clim. 2, 045002 (2023).

11. Dunstone, N. et al. Windows of opportunity for predicting seasonal climate extremes highlighted by the Pakistan floods of 2022. Nat. Commun. 14, 6544 (2023).

12. Borchert, L. F., Düsterhus, A., Brune, S., Müller, W. A. & Baehr, J. Forecast-Oriented Assessment of Decadal Hindcast Skill for North Atlantic SST. Geophys. Res. Lett. 46, 11444–11454 (2019).

13. Duan, S., Ullrich, P. & Boos, W. R. Meteorological Drivers of North American Monsoon Extreme Precipitation Events. J. Geophys. Res. Atmos. 129, e2023JD040535 (2024).

14. Li, L. & Dolman, A. J. On the reliability of composite analysis: an example of wet summers in North China. Atmos. Res. 292, 106881 (2023).

15. Yang, Y., Zhai, P., Li, J. & Wang, Q. Rainbelt Properties of Persistent Heavy Precipitation over the Yangtze River Basin and Associated Three-Dimensional Circulations. Weather Forecast. 40, 689–702 (2025).

16. Pegion, K., Becker, E. J. & Kirtman, B. P. Understanding Predictability of Daily Southeast U.S. Precipitation Using Explainable Machine Learning. Artif. Intell. Earth Syst. 1, e220011 (2022).

17. Kalashnikov, D. A. et al. Predicting Cloud-To-Ground Lightning in the Western United States From the Large-Scale Environment Using Explainable Neural Networks. J. Geophys. Res. Atmos. 129, e2024JD042147 (2024).

18. Camps-Valls, G. et al. Artificial intelligence for modeling and understanding extreme weather and climate events. Nat. Commun. 16, 1919 (2025).

19. Straaten, C. van, Whan, K., Coumou, D., Hurk, B. van den & Schmeits, M. Correcting Subseasonal Forecast Errors with an Explainable ANN to Understand Misrepresented Sources of Predictability of European Summer Temperatures. Artif. Intell. Earth Syst. 2, e220047 (2023).

20. Mamalakis, A. Unraveling Winter Precipitation Predictability over CONUS via Deep Learning and Explainable Artificial Intelligence. Artif. Intell. Earth Syst. 5, 250105 (2026).

21. Liu, Q. et al. Deep-learning post-processing of short-term station precipitation based on NWP forecasts. Atmos. Res. 295, 107032 (2023).

22. Martin, Z. K., Barnes, E. A. & Maloney, E. Using Simple, Explainable Neural Networks to Predict the Madden-Julian Oscillation. J. Adv. Model. Earth Syst. 14, e2021MS002774 (2022).

23. Krell, E., Mamalakis, A., King, S. A., Tissot, P. & Ebert-Uphoff, I. The influence of correlated features on neural network attribution methods in geoscience. Environ. Data Sci. 4, e29 (2025).

24. Mamalakis, A., Barnes, E. A. & Ebert-Uphoff, I. Investigating the Fidelity of Explainable Artificial Intelligence Methods for Applications of Convolutional Neural Networks in Geoscience. Artif. Intell. Earth Syst. 1, e220012 (2022).

25. Bommer, P. L., Kretschmer, M., Hedström, A., Bareeva, D. & Höhne, M. M.-C. Finding the Right XAI Method—A Guide for the Evaluation and Ranking of Explainable AI Methods in Climate Science. Artif. Intell. Earth Syst. 3, e230074 (2024).

26. Fong, R. C. & Vedaldi, A. Interpretable Explanations of Black Boxes by Meaningful Perturbation. in 3429–3437 (2017).

27. Prein, A. F. et al. Sub-Seasonal Predictability of North American Monsoon Precipitation. Geophys. Res. Lett. 49, e2021GL095602 (2022).

28. Strazzo, S. et al. Application of a Hybrid Statistical–Dynamical System to Seasonal Prediction of North American Temperature and Precipitation. Mon. Weather Rev. 147, 607–625 (2019).

29. Peng, Z. et al. Statistical calibration and bridging of ECMWF System4 outputs for forecasting seasonal precipitation over China. J. Geophys. Res. Atmos. 119, 7116–7135 (2014).

30. Li, Y., Xü, K., Wu, Z., Zhu, Z. & Wang, Q. J. A statistical–dynamical approach for probabilistic prediction of sub-seasonal precipitation anomalies over 17 hydroclimatic regions in China. Hydrol. Earth Syst. Sci. 27, 4187–4203 (2023).

31. Lyu, Y. et al. Improving subseasonal-to-seasonal prediction of summer extreme precipitation over southern China based on a deep learning method. Geophys. Res. Lett. 50, e2023GL106245 (2023).

32. Jin, W. et al. Deep learning for seasonal precipitation prediction over China. J. Meteorol. Res. 36, 271–281 (2022).

33. Gibson, P. B. et al. Training machine learning models on climate model output yields skillful interpretable seasonal precipitation forecasts. Commun. Earth Environ. 2, 159 (2021).

34. Yang, R. et al. Interpretable machine learning for weather and climate prediction: A review. Atmos. Environ. 338, 120797 (2024).

35. Wang, S., Zuo, H., Zhao, S., Zhang, J. & Lu, S. How East Asian westerly jet’s meridional position affects the summer rainfall in Yangtze-Huaihe River Valley? Clim. Dyn. 51, 4109–4121 (2018).

36. Yan, Y., Li, C. & Lu, R. Meridional Displacement of the East Asian Upper-tropospheric Westerly Jet and Its Relationship with the East Asian Summer Rainfall in CMIP5 Simulations. Adv. Atmos. Sci. 36, 1203–1216 (2019).

37. Ling, S., Lu, R., Liu, H. & Yang, Y. Interannual Meridional Displacement of the Upper-Tropospheric Westerly Jet over Western East Asia in Summer. Adv. Atmos. Sci. 40, 1298–1308 (2023).

38. TAN, G., SUN, Z., LIN, Z. & JIA, J. Land high over area south to Lake Baikal and its relation with East Asian summer monsoon and climate anomalies of China. Clim. Environ. Res. 13, 791–799 (2008).

39. El Niño is forecast to intensify, increasing likelihood of extreme weather. World Meteorological Organization https://wmo.int/news/media-centre/el-nino-forecast-intensify-increasing-likelihood -of-extreme-weather (2026).

40. Climate Prediction Center: ENSO Diagnostic Discussion. https://www.cpc.ncep.noaa.gov/products/analysis\_monitoring/enso\_advisory/ensod isc.shtml?utm\_source=chatgpt.com.

41. El Nino Monitoring and Outlook / TCC. https://ds.data.jma.go.jp/tcc/tcc/products/elnino/outlook.html?utm\_source=chatgpt. com.

42. Dinneen, J. This El Niño is set to be the largest on record by a ‘mind-blowing margin’. Nature d41586-026-02293-y (2026) doi:10.1038/d41586-026-02293-y.

43. Chen, L. et al. Interdecadal change in the influence of El Niño in the developing stage on the central China summer precipitation. Clim. Dyn. 59, 1265–1282 (2022).

44. Wang, H. & Wang, C. Large-Scale Anomalous Cyclone in the Western North Pacific. J. Clim. 36, 5895–5906 (2023).

45. Zhihua, R., Yu, Y., Fengling, Z. & Yan, X. Quality detection of surface historical basic meteorological data. J. Appl. Meteorol. Sci. 23, 739–747 (2012).

46. Huang, B. et al. Extended Reconstructed Sea Surface Temperature, Version 5 (ERSSTv5): Upgrades, Validations, and Intercomparisons. J. Clim. 30, 8179–8205 (2017).

47. Hersbach, H. et al. The ERA5 global reanalysis. Q. J. R. Meteorol. Soc. 146, 1999–2049 (2020).

48. Copernicus Climate Change Service, Climate Data Store. Seasonal forecast monthly statistics on single levels. https://doi.org/10.24381/cds.68dd14c3 (2018).

49. Copernicus Climate Change Service, Climate Data Store. Seasonal forecast monthly statistics on pressure levels. https://doi.org/10.24381/cds.0b79e7c5 (2018).

50. National Climate Center, China Meteorological Administration. Forecast evaluation methods and parameters.

51. Bach, S. et al. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. PloS One 10, e0130140 (2015).

52. Martin, Z. K., Barnes, E. A. & Maloney, E. Using Simple, Explainable Neural Networks to Predict the Madden-Julian Oscillation. J. Adv. Model. Earth Syst. 14, e2021MS002774 (2022).

53. Toms, B. A., Barnes, E. A. & Ebert-Uphoff, I. Physically Interpretable Neural Networks for the Geosciences: Applications to Earth System Variability. J. Adv. Model. Earth Syst. 12, e2019MS002002 (2020).

54. Barnes, E. A. et al. Indicator Patterns of Forced Change Learned by an Artificial Neural Network. J. Adv. Model. Earth Syst. 12, e2020MS002195 (2020).

55. Ali, A. et al. XAI for transformers: Better explanations through conservative propagation. in International conference on machine learning 435–451 (PMLR, 2022).

56. Sundararajan, M., Taly, A. & Yan, Q. Axiomatic attribution for deep networks. in International conference on machine learning 3319–3328 (PMLR, 2017).

57. Kapishnikov, A. et al. Guided integrated gradients: An adaptive path method for removing noise. in 2021 IEEE/CVF conference on computer vision and pattern recognition (CVPR) 5048–5056 (IEEE, 2021).

# Interpretable AI predicts a 2026 summer dry anomaly in central China

Anran WANG <sup>1†</sup>, Wen SHI <sup>1†</sup>, Yong LUO <sup>1</sup>, Jianbin HUANG <sup>2,3,4</sup>, Lijuan CHEN <sup>5,6</sup>, Junhu ZHAO <sup>6</sup>, Weixin JIN <sup>7</sup>, and Huihui YUAN <sup>1</sup>

1 Department of Earth System Science, Tsinghua University, Beijing 100084

2 College of Resources and Environment, University of Chinese Academy of Sciences,

Beijing 101408

3 Beijing Yanshan Earth Critical Zone National Research Station, University ofChinese Academy ofSciences, Beijing 101408

4 College of Resources and Environment, University of Chinese Academy of Sciences, Beijing 100049

5 State Key Laboratory ofClimate System Prediction and Risk Management, National Climate Centre, China Meteorological Administration, Beijing 100081

6 China Meteorological Administration Key Laboratory for Climate Prediction Studies, National Climate Center, China Meteorological Administration, Beijing 100081 7 Microsoft, Beijing 100080

Contents of this file Supplementary Tables 1 to 2 Supplementary Figs. 1 to 4

Supplementary Table 1. Historical analogue years selected for the summer 2026 predictions initialized in March, April and May. For each initialization, 33 summers during 1993–2025 were ranked by all-station ACC (ACC<sub>all</sub>), central China ACC (ACC<sub>CC</sub>), dry coverage (DC) and dry-intensity distance (DID). The seven years with the smallest sum of metric ranks (S<sub>y</sub>) were retained. DC is reported as a percentage and DID in percentage points. Six years were common to all three selections; April and May yielded identical sets, whereas March replaced 2002 with 2022.
<table><tr><td>Initialization</td><td>Year</td><td>ACCall</td><td>ACCcc</td><td>DC (%)</td><td>DID</td><td> $S _ { y }$ </td><td>Final rank</td></tr><tr><td rowspan="7">March</td><td>1997</td><td>0.47</td><td>0.86</td><td>93</td><td>1.5</td><td>4</td><td>1</td></tr><tr><td>2006</td><td>0.38</td><td>0.57</td><td>81</td><td>10.3</td><td>9</td><td>2</td></tr><tr><td>2001</td><td>0.30</td><td>0.59</td><td>79</td><td>16.0</td><td>12</td><td>3</td></tr><tr><td>2015</td><td>0.18</td><td>0.51</td><td>77</td><td>19.4</td><td>21</td><td>4</td></tr><tr><td>2022</td><td>0.15</td><td>0.42</td><td>74</td><td>15.5</td><td>25</td><td>5</td></tr><tr><td>2011</td><td>0.20</td><td>0.37</td><td>75</td><td>19.0</td><td>25</td><td>6</td></tr><tr><td>1994</td><td>0.11</td><td>0.54</td><td>77</td><td>19.7</td><td>26</td><td>7</td></tr><tr><td rowspan="7">April</td><td>1997</td><td>0.51</td><td>0.88</td><td>93</td><td>6.4</td><td>5</td><td>1</td></tr><tr><td>2006</td><td>0.35</td><td>0.49</td><td>81</td><td>5.4</td><td>12</td><td>2</td></tr><tr><td>2001</td><td>0.38</td><td>0.58</td><td>79</td><td>11.1</td><td>12</td><td>3</td></tr><tr><td>2015</td><td>0.22</td><td>0.59</td><td>77</td><td>14.5</td><td>17</td><td>4</td></tr><tr><td>1994</td><td>0.18</td><td>0.50</td><td>77</td><td>14.8</td><td>23</td><td>5</td></tr><tr><td>2002</td><td>0.27</td><td>0.50</td><td>69</td><td>18.4</td><td>24</td><td>6</td></tr><tr><td>2011</td><td>0.09</td><td>0.35</td><td>75</td><td>14.1</td><td>30</td><td>7</td></tr><tr><td rowspan="6">May</td><td>1997</td><td>0.57</td><td>0.87</td><td>93</td><td>3.2</td><td>4</td><td>1</td></tr><tr><td>2006</td><td>0.38</td><td>0.55</td><td>81</td><td>15.0</td><td>9</td><td>2</td></tr><tr><td>2001</td><td>0.34</td><td>0.60</td><td>79</td><td>20.7</td><td>12</td><td>3</td></tr><tr><td>2015</td><td>0.23</td><td>0.54</td><td>77</td><td>24.1</td><td>20</td><td>4</td></tr><tr><td>1994</td><td>0.16</td><td>0.54</td><td>77</td><td>24.4</td><td>24</td><td>5</td></tr><tr><td>2002</td><td>0.26</td><td>0.45</td><td>69</td><td>28.0</td><td>26</td><td>6</td></tr><tr><td></td><td>2011</td><td>0.16</td><td>0.40</td><td>75</td><td>23.7</td><td>27</td><td>7</td></tr></table>

Supplementary Fig. 1. Circulation patterns associated with the March-initialized prediction. Same as Fig. 3, but for the seven historical analogue years selected for the March initialization and the March-initialized dynamical MME prediction for JJA 2026. The analogue composite and 2026 prediction both show a western North Pacific–South China Sea–South China cyclonic circulation anomaly and anomalous moisture-flux divergence over central China.

![](images/0535aed3a59e2e568679bdcbc3e2386f2a6077c441b600f38636026b78d57710.jpg)

Supplementary Fig. 2. Circulation patterns associated with the April-initialized prediction. Same as Supplementary Fig. 1, but for the seven historical analogue years selected for the April initialization and the April-initialized dynamical MME prediction for JJA 2026.  
![](images/149080edc74876b721ee13349a63be5e4dc88705a49ce37e908254ee9445508f.jpg)

Supplementary Fig. 3. Integrated Gradients attribution for the May-initialized prediction. Same as Fig. 5, but showing dry-supporting negative attributions from Integrated Gradients (IG). IG broadly reproduces the dry-supporting meridional-wind signals at 500, 700 and 850 hPa identified by LRP, while assigning the largest predictor contribution to T2m.

![](images/f8ef729c6eef4ebc43e2e8252d2a81fff4b11ff649e15a62f2740c289618c0f4.jpg)

![](images/4a872c4058a7ae876b920ce561561255236020c3f26405e3fd9ada5cc34aa6eb.jpg)

Supplementary Fig. 4. Guided Integrated Gradients attribution for the May-initialized prediction. Same as Supplementary Fig. 3, but using Guided Integrated Gradients (Guided IG). The dry-supporting meridional-wind signals at 500, 700 and 850 hPa are again evident, while T2m has the largest predictor contribution.

![](images/0011d0291cb10b7b347af5e74f31d6a3dab2cb597164d7a487970c93d0c50cef.jpg)

![](images/31549319aa271004f339cb41239919efb04ceddf07c06976d3df53b25196f324.jpg)

Supplementary Table 2. Cross-method agreement in dry-supporting attributions. Spearman rank correlation assesses agreement in the ranking of area-weighted dry-supporting contributions across the 13 predictors. Area-weighted full-tensor cosine similarity measures agreement in the unsmoothed attribution patterns across all predictors and grid cells. For both metrics, values closer to 1 indicate stronger agreement. The high rank correlations (0.88–0.98) indicate that the overall predictor hierarchy was consistent across methods. Full-tensor similarity was lower between LRP and the two gradient-based methods, indicating greater method dependence in the detailed spatial patterns. Agreement with LRP was strongest for the May initialization examined in the main text. LRP, layer-wise relevance propagation; IG, Integrated Gradients; Guided IG, Guided Integrated Gradients.
<table><tr><td rowspan="2">Initialization</td><td rowspan="2">Method pair</td><td rowspan="2">Spearman rank correlation</td><td rowspan="2">Area-weighted full-tensor cosine similarity</td></tr><tr><td></td></tr><tr><td rowspan="3">March</td><td>LRP vs IG</td><td>0.91</td><td>0.47</td></tr><tr><td>LRP vs Guided IG</td><td>0.88</td><td>0.40</td></tr><tr><td>IG vs Guided IG</td><td>0.95</td><td>0.92</td></tr><tr><td rowspan="3">April</td><td>LRP vs IG</td><td>0.92</td><td>0.48</td></tr><tr><td>LRP vs Guided IG</td><td>0.90</td><td>0.43</td></tr><tr><td>IG vs Guided IG</td><td>0.96</td><td>0.92</td></tr><tr><td rowspan="3">May</td><td>LRP vs IG</td><td>0.95</td><td>0.64</td></tr><tr><td>LRP vs Guided IG</td><td>0.98</td><td>0.56</td></tr><tr><td>IG vs Guided IG</td><td>0.97</td><td>0.90</td></tr></table>
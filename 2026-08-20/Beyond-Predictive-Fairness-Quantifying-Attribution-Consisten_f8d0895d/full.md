# Beyond Predictive Fairness: Quantifying Attribution Consistency Across Demographic Groups in Diabetic Retinopathy Screening

Kerol Djoumessi<sup>1(B)[0009−0004−1548−9758]</sup> and Philipp Berens<sup>1,2[0000−0002−0199−4727]</sup>

<sup>1</sup> Hertie Institute for AI in Brain Health, University of Tübingen, Germany kerol.djoumessi-donteu@uni-tuebingen.de https://hertie.ai/ <sup>2</sup> Tübingen AI Center, University of Tübingen, Germany

Abstract. Fairness in medical imaging is commonly evaluated through subgroup performance metrics, yet it remains unclear whether models rely on consistent visual evidence across demographic groups. This work introduces the Explanation Consistency Score (ECS), a fairness-aware metric based on Jensen–Shannon divergence that quantifies the similarity of attribution maps across subgroups. Using diabetic retinopathy screening as a case study, ECS is evaluated globally and within disease severity. Experiments reveal that while predictive performance differs across ethnic groups, explanation consistency remains relatively high and shows no significant association with performance disparities. These findings suggest that predictive fairness and explanation consistency capture complementary dimensions of model behavior, motivating fairness evaluations that extend beyond predictive performance.

Keywords: Algorithmic Fairness · Explainable AI · Explanation Consistency · Medical Image Analysis · Diabetic Retinopathy Screening.

## 1 Introduction

Deep learning systems have achieved strong performance in automated medical image diagnosis across a wide range of clinical tasks [23], including diabetic retinopathy (DR) screening from retinal fundus images [3, 9]. Alongside these advances, fairness and explainability have become increasingly important concerns in medical imaging due to the reported performance disparities across demographic subgroups [2, 23]. Existing fairness studies primarily evaluate subgroup disparities using predictive metrics such as sensitivity, specificity, and area under the receiver operating characteristic curve (AUC) [15,17]. While such evaluations quantify diferences in predictive outcomes, they provide limited insight into whether models rely on similar visual evidence across patient groups.

Recent studies show that medical imaging models can encode demographic information despite being trained for clinical tasks [20, 25]—in ophthalmology, demographic attributes can even be inferred directly from fundus images [14,

20]. Consequently, two models may show similar subgroup performance while relying on diferent visual evidence, or vice versa, making predictive metrics alone insuficient for trustworthy deployment.

Explainability methods such as Class Activation Mapping (CAM) are widely used in ophthalmology to improve interpretability by highlighting retinal regions contributing to model predictions [4, 21, 24]. These methods have been used to verify whether models attend to clinically relevant retinal structures and lesions, thereby supporting the interpretation of automated screening systems. Despite known limitations and sensitivity to model architecture [1,24], attribution maps remain one of the most widely adopted approaches for understanding model behavior in medical imaging [4, 5]. However, existing studies have primarily leveraged explainability qualitatively to inspect subgroup failures, identify potential sources of bias, or visualize fairness-related concerns [26,27]. Quantitative analyses of attribution diferences across demographic groups remain limited [22], and their relationship with predictive fairness is not yet well understood. This raises an important question: do predictive disparities necessarily imply diferences in model reasoning across demographic groups?

In this work, we investigate the demographic consistency of visual attribution patterns in early diabetic retinopathy detection using retinal fundus images from the EyePACS dataset. An Explanation Consistency Score (ECS) is introduced as a Jensen–Shannon divergence-based metric to quantify attribution similarity across demographic groups. To account for potential confounding efects arising from diferences in disease severity, a conditional formulation of ECS is additionally proposed. By jointly analyzing predictive performance and explanation consistency across demographic groups, this work provides a quantitative framework for fairness-aware attribution analysis in medical imaging.

## 2 Methods

## 2.1 Dataset and Demographic Subgroups

Experiments were conducted on the EyePACS diabetic retinopathy dataset [6], which provides retinal fundus images, DR grades (0–4), and patient ethnicity metadata. Following the provided quality-control protocol, 90,595 good-quality images were retained (Tab. 1). Data were split at the patient level into 70% training, 15% validation, and 15% test sets using iterative multi-label stratification based on ethnicity and DR grade to preserve subgroup and disease distributions.

The fairness analysis focused on the five largest ethnicity groups: Latin American, African Descent, Indian Origin, Caucasian, and Asian. Samples with unspecified ethnicity or underrepresented groups were retained for training but excluded from subgroup analyses due to missing or limited data.

## 2.2 DR classification model

A ResNet-50 [12] was trained for early diabetic retinopathy (DR) detection by distinguishing healthy eyes (grade 0) from eyes exhibiting any stage of DR (grade

Table 1. Demographic distribution and DR grade composition of the EyePACS cohort.
<table><tr><td>Ethnicity</td><td># Samples</td><td>DR0</td><td>DR1</td><td>DR2</td><td>DR3</td><td>DR4</td><td>DR1-4</td><td>Prev.</td></tr><tr><td>Latin American African descent Indian origin Caucasian</td><td>53,960 5,097 3,341 9,542</td><td>42,155 3,597 1,822 7,818</td><td>5,787 747 472 794</td><td>5,502 663 930 884</td><td>285 40 51 24</td><td>231 50 66 22</td><td>11,805 1,500 1,519 1,724</td><td>21.9% 29.4% 45.5% 18.1%</td></tr><tr><td>Asian Not specified Underrepresented</td><td>3,995 12,253 2,407</td><td>3,335 9,825 1,816</td><td>321 1,228 244</td><td>314 1,114 324</td><td>4 60 17</td><td>21 26 6</td><td>660 2,428 591</td><td>16.5% 19.8% 24.5%</td></tr></table>

1–4). This binary setting was selected because early disease detection requires identifying subtle retinal lesions [9] and therefore provides a suitable benchmark for studying potential demographic diferences in model explanations.

## 2.3 Attribution Map Generation

To analyze the visual evidence underlying model predictions, attribution maps were generated using SmoothGradCAM++ [21] and Score-CAM [24], representing gradient-based and gradient-free attribution methods, respectively. Using both methods enables evaluating the robustness of the proposed explanation consistency analysis across diferent attribution mechanisms. Each map $A \in \mathbb { R } ^ { N \times M }$ was normalized to sum to one $( \tilde { A } _ { i , j } = A _ { i , j } / \sum _ { u , v } A _ { u , v } )$ , yielding a spatial probability distribution.

## 2.4 Explanation Consistency Score

To quantify attribution consistency across demographic groups, an Explanation Consistency Score (ECS) is introduced. For a demographic group g, the mean attribution distribution is defined as

$$
\mu _ { g } = \mathbb { E } _ { \boldsymbol { x } \sim P ( \boldsymbol { x } | \boldsymbol { g } ) } [ \tilde { A } ( \boldsymbol { x } ) ] \approx \frac { 1 } { | D _ { g } | } \sum _ { \boldsymbol { x } \in D _ { g } } \tilde { A } ( \boldsymbol { x } ) ,
$$

where $D _ { g }$ denotes the set of samples belonging to group g. Attribution distributions are compared using Jensen–Shannon (JS) divergence [18], a symmetric and bounded measure of similarity between probability distributions. The pairwise ECS between demographic groups $g _ { 1 }$ and $g _ { 2 }$ is defined as

$$
\begin{array} { r } { \mathrm { E C S } ( g _ { 1 } , g _ { 2 } ) = 1 - \mathrm { J S } ( \mu _ { g _ { 1 } } \| \mu _ { g _ { 2 } } ) . } \end{array}
$$

ECS ranges from 0 to 1, where larger values indicate greater similarity between attribution distributions. To summarize the consistency of a demographic group relative to all remaining groups, a multi-group ECS is computed as

$$
\operatorname { A v g E C S } ( g ) = { \frac { 1 } { | { \mathcal { G } } | - 1 } } \sum _ { g ^ { \prime } \neq g } \operatorname { E C S } ( g , g ^ { \prime } ) ,
$$

where G denotes the set of demographic groups. Disease severity may act as a confounding factor because attribution patterns can vary across DR grades. To control this efect, ECS is additionally computed conditionally on DR grade. For demographic group $g$ and DR grade $y ,$ the conditional attribution distribution is defined as

$$
\mu _ { g } ^ { ( y ) } = \mathbb { E } _ { x \sim P ( x \mid g , y ) } [ \tilde { A } ( x ) ] .
$$

The conditional ECS for grade y is then

$$
\begin{array} { r } { \mathrm { E C S } ^ { ( y ) } ( g _ { 1 } , g _ { 2 } ) = 1 - \mathrm { J S } \left( \mu _ { g _ { 1 } } ^ { ( y ) } \lVert \boldsymbol { \mu } _ { g _ { 2 } } ^ { ( y ) } \right) , } \end{array}
$$

and the final conditional score is obtained by averaging across grades:

$$
\mathrm { C o n d E C S } ( g _ { 1 } , g _ { 2 } ) = \frac { 1 } { | \mathcal { V } | } \sum _ { y \in \mathcal { V } } \mathrm { E C S } ^ { ( y ) } ( g _ { 1 } , g _ { 2 } ) .
$$

In practice, conditional analyses were restricted to DR grades 1 and 2, corresponding to the early disease stages considered in this study.

## 3 Experiments and Results

## 3.1 Experimental Setup

Data Preprocessing and Augmentation. Images were preprocessed by circular field-of-view cropping [19], resized to $5 1 2 \times 5 1 2$ , and normalised using training set statistics. Augmentation followed [13], including random cropping, color jittering, and rotation.

Model Training. Following prior work [9, 13], the $\mathrm { m o d e l ^ { 3 } }$ was initialized with ImageNet-pretrained weights and optimized using cross-entropy loss. Training used SGD with an initial learning rate of $1 0 ^ { - 3 }$ and a clipped cosine schedule, selecting the model with highest validation AUC after 100 epochs.

Attribution Processing. Attribution maps were generated using the Torch-CAM library [11] at the resulting native low resolution (16 × 16 pixels), normalized for all consistency analyses, and upsampled for qualitative visualization.

## 3.2 Predictive Performance Across Demographic groups

Subgroup predictive performance was assessed using AUC, sensitivity, and specificity (Tab. 2). The latter two directly reflecting clinical screening relevance.

Performance diferences were observed across demographic groups. Patients of Indian origin achieved the highest $\mathrm { A U C } \left( 0 . 9 2 \pm 0 . 0 0 1 \right)$ and sensitivity (0.83 ± 0.003), whereas Caucasian patients exhibited the lowest $\mathrm { A U C } \left( 0 . 8 3 \pm 0 . 0 0 3 \right)$ and sensitivity $( 0 . 5 4 \pm 0 . 0 1 8 )$ , corresponding to a sensitivity gap of 23%. In contrast, specificity remained relatively stable across groups (0.94–0.97), indicating that subgroup diferences primarily arise from the detection of positive DR cases rather than healthy patients.

Table 2. Predictive performance and attribution consistency across demographic groups. Results are reported as mean ± standard deviation over three random seeds.
<table><tr><td></td><td colspan="3">Predictive performance</td><td colspan="4">Explanation consistency</td></tr><tr><td>Ethnicity</td><td>AUC</td><td>Sens.</td><td>Spec.</td><td></td><td></td><td></td><td>ECS-SCWG-ECSECS-SGDR1-ECS</td></tr><tr><td>Latin American</td><td>.85 ± .002</td><td>.57 ± .013.97 ± .004</td><td></td><td> $\overline { { . 9 2 \pm . 0 0 2 } }$ </td><td> $\overline { { 5 3 \pm . 0 8 } }$ </td><td> $\overline { { . 8 8 \pm . 0 1 0 } }$ </td><td> $\overline { { . 8 7 \pm 0 0 6 } }$ </td></tr><tr><td>African descent</td><td>|.88 ± .002</td><td> $. 6 4 \pm . 0 2 3$ </td><td> $9 7 \pm . 0 0 2$ </td><td> $. 9 1 \pm 0 0 5$ </td><td> $. 5 6 \pm . 0 8$ </td><td> $. 8 8 \pm . 0 0 8$ </td><td> $. 8 5 \pm . 0 1 4$ </td></tr><tr><td>Indian origin</td><td>.92 ± .001|.77 ± .022|</td><td></td><td>|.94 ± .007</td><td> $. 9 0 \pm 0 0 6$ </td><td> $. 5 7 \pm . 0 8$ </td><td> $. 8 6 \pm . 0 1 0 $ </td><td> $. 8 5 \pm 0 0 9$ </td></tr><tr><td>Caucasian</td><td>.83 ± .003|.54 ± .018|.97 ± .004</td><td></td><td></td><td> $. 9 1 \pm 0 0 4$ </td><td> $. 5 7 \pm . 0 7$ </td><td> $. 8 6 \pm . 0 1 4$ </td><td> $. 8 5 \pm 0 0 8$ </td></tr><tr><td>Asian</td><td>.90 ± .006|.68 ± .023|.97 ± .005</td><td></td><td></td><td> $. 8 9 \pm . 0 0 4$ </td><td> $. 5 4 \pm . 0 8$ </td><td> $. 8 5 \pm . 0 0 8$ </td><td> $. 7 9 \pm . 0 1 1$ </td></tr><tr><td>All</td><td></td><td> $\overline { { . 8 6 \pm . 0 0 1 | . 5 9 \pm 0 1 5 | . 9 7 \pm 0 0 4 } }$ </td><td></td><td></td><td></td><td></td><td></td></tr></table>

Interestingly, predictive performance did not follow subgroup size (Tab. 1). Although Latin American patients represent the largest demographic group, they did not achieve the strongest performance. Conversely, Indian-origin and Asian patients, among the smallest groups considered, obtained the highest AUC and sensitivity values. Neither subgroup size nor disease prevalence alone explains the observed diferences: the Asian subgroup achieved strong performance despite the lowest prevalence, and Indian-origin patients outperformed the much larger Latin American group. Low standard deviations across all metrics indicate that these trends were consistent across random initializations. These findings reveal measurable predictive disparities across ethnicities and motivate the subsequent analysis of whether such diferences are reflected in model attribution behavior.

## 3.3 Attribution Consistency Across Demographic Groups

To assess whether model explanations difer across demographic groups, the proposed Explanation Consistency Score (ECS) for a given model was computed from correctly classified test samples using ScoreCAM (ECS-SC) and smooth-GradCAM++ (ECS-SG). This design choice—restricting to correct predictions— isolates explanation consistency under successful decision-making, as misclassified examples may exhibit heterogeneous failure modes that complicate crossgroup comparisons. Multi-group ECS values are reported in Table 2, while average attribution maps are shown in Figure 1. High explanation consistency was observed across all demographic groups. ECS values ranged from 0.89 to 0.92 for ScoreCAM and from 0.85 to 0.88 for SmoothGradCAM++, indicating that attribution patterns remained largely similar across ethnicities. These findings are consistent with the qualitative visualizations (Fig. 1), where both attribution methods consistently focus on similar retinal regions despite minor diferences in attribution concentration and spatial distribution.

Compared with predictive performance, attribution consistency exhibited substantially lower variability across groups. While sensitivity varies by 23%, ECS varies by only 0.03 for both attribution methods. Notably, the subgroup achieving the highest predictive performance (Indian origin) did not exhibit the highest ECS, whereas groups with lower predictive performance, such as Caucasian patients, remained highly consistent with the attribution patterns observed in other groups. These findings suggest that predictive disparities are not necessarily accompanied by comparable diferences in attribution behavior. To further investigate this relationship, Spearman correlations (ρ) were computed between subgroup performance and ECS. Moderate negative correlations were observed for both AUC and sensitivity $( \rho ~ = ~ - 0 . 8 0$ for Score-CAM, $\rho = - 0 . 6 0$ for SmoothGradCAM++), though neither reached statistical significance—unsurprising given the limited number of demographic groups $( n = 5 )$ , preventing meaningful evidence.

![](images/443c360d8955e94774e6af13c645fee3103ccb2f73ae536dc40c11e703da10ff.jpg)  
Fig. 1. Explanation consistency across demographic groups. Average Score-CAM and SmoothGradCAM++ attribution maps for correctly classified DR-positive samples across demographic groups. Similar attention patterns are observed across ethnicities, indicating high explanation consistency.

To assess whether group-level ECS values could be explained by diferences in within-group attribution variability, the similarity between each attribution map and its corresponding group-average attribution map was additionally measured. Within-group consistency (WG.ECS) was comparable across demographic groups (ScoreCAM: 0.53˘0.57; SmoothGradCAM++: 0.43˘0.50), indicating comparable levels of attribution variability and supporting the validity of group-level ECS comparisons. This suggests that the high ECS values are unlikely to be driven by subgroup-specific diferences in attribution variability.

## 3.4 Conditional Attribution Consistency Analysis

Diferences in disease prevalence and severity across demographic groups may influence attribution patterns and potentially confound the interpretation of explanation consistency. To control for this efect, conditional ECS was computed separately for correctly classified DR grades 1 and 2, thereby comparing attri bution maps only among patients with the same disease severity.

![](images/37a57c9c2d58302fdfd00336f8b7601f1aecdbeb127980d6f2293af7e8eca6e6.jpg)  
Fig. 2. Conditional attribution consistency across demographic groups. Average attribution maps for DR grades 1 (top two rows) and 2 (bottom two rows) across demographic groups using ScoreCAM and SmoothGradCAM++. Attribution patterns remain broadly consistent within each disease grade, supporting the robustness of the proposed ECS analysis after controlling for disease severity.

High explanation consistency was preserved after conditioning on disease severity (Fig. 2). For ScoreCAM, multi-group ECS ranged from 0.79 to 0.87 for grade 1 (DR1-ECS, Tab. 2) and from 0.87 to 0.90 for grade 2 across demographic groups. Similarly, trends were observed for SmoothGradCAM++ produced ECS values ranging between 0.77–0.82 for grade 1 and 0.81–0.86 for grade 2. The highest consistency was generally observed for Latin American patients, whereas Asian patients exhibited the lowest ECS values across both grades and attribution methods. Across all demographic groups and attribution methods, ECS was consistently higher for grade 2 than for grade 1. For example, the ScoreCAMbased ECS of Asian patients increased from $0 . 7 9 \pm 0 . 0 1$ (grade 1) to $0 . 8 7 \pm 0 . 0 0 2$ (grade 2). This trend suggests that attribution patterns become more stable as diabetic retinopathy lesions become more visually apparent, whereas early-stage disease (grade 1) is associated with greater attribution variability.

Importantly, high ECS values persisted after controlling for disease severity, indicating that the consistency observed in the global analysis (Fig. 1) is not solely explained by diferences in disease prevalence or grade distribution across across demographic groups. Overall, these findings provide further evidence that the model relies on broadly similar visual evidence across demographic groups even when comparisons are restricted to patients with the same disease stage.

## 4 Discussion and Conclusion

This work introduced the Explanation Consistency Score (ECS) to quantify the similarity of attribution patterns across demographic groups in diabetic retinopathy screening and investigated whether demographic diferences in predictive performance can be explained by diferences in model attribution patterns. Although predictive performance varied across ethnicities, attribution consistency remained uniformly high for both ScoreCAM and SmoothGradCAM++, indicating that predictive fairness and explanation consistency capture complementary rather than equivalent aspects of model behavior. In particular, performance disparities were substantially larger than attribution disparities, and no significant association was observed between ECS and subgroup predictive performance.

Several observations may help explain this decoupling. First, subgroup performance did not appear to be determined solely by sample size: the Indianorigin subgroup achieved the highest AUC and sensitivity despite being among the smallest groups, whereas the largest subgroup did not achieve the strongest performance. Second, disease prevalence may partially contribute to these differences, as higher prevalence generally leads to higher sensitivity, although the strong performance of the Asian subgroup despite its low prevalence suggests that prevalence alone is insuficient to explain subgroup variation. These findings point toward additional factors, such as diferences in disease manifestation, image characteristics, or subgroup-specific data distributions [2, 17, 20]. At the same time, attribution consistency remained high across all groups, including those with lower predictive performance, suggesting that the model relies on broadly similar visual evidence despite diferences in predictive outcomes. The conditional analysis further supported this conclusion, as high ECS values persisted after controlling for disease severity. Interestingly, grade-2 images exhibited higher ECS than grade-1 images, suggesting that more advanced disease stages produce more stable attribution patterns.

Several limitations should be acknowledged. First, ECS was computed on correctly classified samples to characterize attribution consistency under successful predictions. Consequently, subgroup-specific failure modes remain unexplored and should be investigated in future work. Second, ECS is a group-level measure based on mean attribution distributions. Although an additional withingroup analysis revealed comparable attribution variability across demographic groups, suggesting that the observed ECS values are not merely averaging artifacts, ECS may not fully capture individual-level explanation variability. Future work could complement ECS with image-level analyses of attribution variability. Third, attribution maps were analyzed at their native CAM resolution, which may limit sensitivity to fine-grained spatial diferences. Fourth, ECS relies on Jensen–Shannon divergence; alternative similarity measures such as cosine similarity, Wasserstein distance, or structural similarity may capture complementary notions of attribution similarity [16]. Finally, the study was conducted on a single dataset and focused exclusively on ethnicity, using posthoc methods, motivating future evaluation across additional modalities, datasets, protected attributes, and self-explainable models [7, 8, 10].

Acknowledgments. This project was supported by the Hertie Foundation and by the Deutsche Forschungsgemeinschaft under Germany’s Excellence Strategy with the Excellence Cluster 2064 “Machine Learning – New Perspectives for Science”, and a regular grant (project number 571331899).

Disclosure of Interests. The authors declare no competing interests.

## References

1. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., Kim, B.: Sanity checks for saliency maps. Advances in neural information processing systems 31 (2018)

2. Alloula, A., Mustafa, R., McGowan, D.R., Papież, B.W.: On biases in a uk biobankbased retinal image classification model. In: MICCAI Workshop on Fairness of AI in Medical Imaging. pp. 140–150. Springer (2024)

3. Alyoubi, W.L., Shalash, W.M., Abulkhair, M.F.: Diabetic retinopathy detection through deep learning techniques: A review. Informatics in medicine unlocked 20, 100377 (2020)

4. Ayhan, M.S., Kuemmerle, L.B., Kuehlewein, L., Inhofen, W., Aliyeva, G., Ziemssen, F., Berens, P.: Clinical validation of saliency maps for understanding deep neural networks in ophthalmology. Medical Image Analysis 77, 102364 (2022)

5. Bhati, D., Neha, F., Amiruzzaman, M.: A survey on explainable artificial intelligence (xai) techniques for visualizing deep learning models in medical imaging. Journal of Imaging 10(10), 239 (2024)

6. Cuadros, J., Bresnick, G.: Eyepacs: an adaptable telemedicine system for diabetic retinopathy screening. Journal of diabetes science and technology 3(3), 509–516 (2009)

7. Djoumessi, K., Bah, B., Kühlewein, L., Berens, P., Koch, L.: This actually looks like that: Proto-bagnets for local and global interpretability-by-design. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 718–728. Springer (2024)

8. Djoumessi, K., Berens, P.: Softcam: Making black box models self-explainable for medical image analysis. In: Medical Imaging with Deep Learning. pp. 433–467. PMLR (2026)

9. Djoumessi, K., Huang, Z., Kühlewein, L., Rickmann, A., Simon, N., Koch, L.M., Berens, P.: An inherently interpretable ai model improves screening speed and accuracy for early diabetic retinopathy. PLOS Digital Health 4(5), e0000831 (2025)

10. Donteu, K.R.D., Ilanchezian, I., Kühlewein, L., Faber, H., Baumgartner, C.F., Bah, B., Berens, P., Koch, L.M.: Sparse activations for interpretable disease grading. In: Medical Imaging with Deep Learning (2023)

11. Fernandez, F.G.: Torchcam: class activation explorer. https://github.com/ frgfm/torch-cam (March 2020)

12. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

13. Huang, Y., Lin, L., Cheng, P., Lyu, J., Tam, R., Tang, X.: Identifying the key components in resnet-50 for diabetic retinopathy grading from fundus images: a systematic investigation. Diagnostics 13(10), 1664 (2023)

14. Ilanchezian, I., Kobak, D., Faber, H., Ziemssen, F., Berens, P., Ayhan, M.S.: Interpretable gender classification from retinal fundus images using bagnets. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 477–487. Springer (2021)

15. Larrazabal, A.J., Nieto, N., Peterson, V., Milone, D.H., Ferrante, E.: Gender imbalance in medical imaging datasets produces biased classifiers for computer-aided diagnosis. Proceedings of the National Academy of Sciences 117(23), 12592–12594 (2020)

16. Levy, A., Shalom, B.R., Chalamish, M.: A guide to similarity measures and their data science applications. Journal of Big Data 12(1), 188 (2025)

17. Mehta, R., Shui, C., Arbel, T.: Evaluating the fairness of deep learning uncertainty estimates in medical image analysis. In: Medical Imaging with Deep Learning. pp. 1453–1492. PMLR (2024)

18. Menéndez, M.L., Pardo, J.A., Pardo, L., Pardo, M.d.C.: The jensen-shannon divergence. Journal of the Franklin Institute 334(2), 307–318 (1997)

19. Mueller, S., Heidrich, H., Koch, L.M., Berens, P.: fundus circle cropping. https://doi.org/10.5281/zenodo.10137935, https://github.com/berenslab/ fundus\_circle\_cropping

20. Müller, S., Koch, L.M., Lensch, H., Berens, P.: A generative model reveals the influence of patient attributes on fundus images. In: Medical Imaging with Deep Learning (2022)

21. Omeiza, D., Speakman, S., Cintas, C., Weldermariam, K.: Smooth grad-cam++: An enhanced inference level visualization technique for deep convolutional neural network models. arXiv preprint arXiv:1908.01224 (2019)

22. Pfohl, S., Harris, N., Nagpal, C., Madras, D., Mhasawade, V., Salaudeen, O., Dieng, A., Sequeira, S., Arciniegas, S., Sung, L., et al.: Understanding challenges to the interpretation of disaggregated evaluations of algorithmic fairness. Advances in Neural Information Processing Systems 38, 41887–41948 (2026)

23. Vrudhula, A., Kwan, A.C., Ouyang, D., Cheng, S.: Machine learning and bias in medical imaging: opportunities and challenges. Circulation: Cardiovascular Imaging 17(2), e015495 (2024)

24. Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S., Mardziel, P., Hu, X.: Score-cam: Score-weighted visual explanations for convolutional neural networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops. pp. 24–25 (2020)

25. Yang, Y., Zhang, H., Gichoya, J.W., Katabi, D., Ghassemi, M.: The limits of fair medical imaging ai in real-world generalization. Nature medicine 30(10), 2838–2848 (2024)

26. Zhao, Y., Wang, Y., Derr, T.: Fairness and explainability: Bridging the gap towards fair model explanations. In: Proceedings of the AAAI conference on artificial intelligence. vol. 37, pp. 11363–11371 (2023)

27. Zhou, J., Chen, F., Holzinger, A.: Towards explainability for ai fairness. In: International workshop on extending explainable AI beyond deep models and classifiers. pp. 375–386. Springer (2020)
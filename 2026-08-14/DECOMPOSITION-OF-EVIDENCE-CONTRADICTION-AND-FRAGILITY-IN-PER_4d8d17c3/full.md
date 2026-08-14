# DECOMPOSITION OF EVIDENCE, CONTRADICTION, AND FRAGILITY IN PERTURBATION RESPONSES

Lei You

Technical University of Denmark

leiyo@dtu.dk

## ABSTRACT

Perturbation methods explain model decisions by measuring prediction changes under altered inputs, but response magnitude tells us only how much a model reacts, not what that reaction means. The same magnitude can support the final factual– counterfactual difference, oppose it, or arise strongly along the perturbation path yet vanish at the endpoint. We therefore track how the contrast develops as paired inputs are progressively revealed, using the final contrast to interpret the trajectory. We introduce DECAF (Decomposition of Evidence, Contradiction, And Fragility), which routes aligned, opposed, and endpoint-null responses into evidence E, contradiction C, and fragility F. The decomposition preserves ordinary magnitude exactly, Abs = E + C + F, and is unique under endpoint-relative axioms. Across controlled vision and tabular settings, the three components track independently measured behavior. In a 72-model ImageNet-9 audit, we compare cases with nearly identical response magnitude but different independently measured behaviors. The largest DECAF component agrees with an observed behavior in 96.4% of cases, compared with 35.0% for magnitude alone. Changing only the reveal path increases total response by nearly 80%, yet evidence barely changes while fragility grows by more than 4×. On FunnyBirds and ImageNet-1k, short forward-only DECAF trajectories outperform the tested general-purpose attribution baselines. On a 1Bscale DINOv2 model, a short trajectory matches a strong gradient-based baseline with 4.75× lower wall time and 2.36× lower peak memory.

§ Code: https://github.com/youlei202/decaf

## 1 INTRODUCTION

Post-hoc explanation asks what aspects of an input matter to a model’s prediction. A natural way to answer this question is contrastively: compare the prediction with what would happen if a feature, region, concept, or context were changed. Perturbation-based and counterfactual methods instantiate this comparison by constructing alternative inputs and measuring the resulting change in model output (Fong & Vedaldi, 2017; Dhurandhar et al., 2018a; Goyal et al., 2019; Covert et al., 2021a). This has the basic form of an effect measurement: change a factor, compare two responses, and summarize the difference. In practice, that response is often reduced to a scalar magnitude.

A magnitude answers an important question: how much did the model react? It leaves a second question open: what did the reaction mean? Consider an input and a paired counterfactual that changes one factor. Their prediction difference measures how strongly the model responds to that change. This endpoint comparison, however, shows only the net effect after inputs are fully observed. It does not show how the contrast emerges as information becomes available. We therefore consider a paired reveal: the paired inputs are progressively revealed along matched trajectories, so that their prediction difference can be observed at the same level of revealed information.

As the pair is revealed from an uninformative state, the contrast can evolve in different ways. It may align with the final effect, oppose it, or become large even when the fully observed pair differs little. These distinct behaviors can have identical absolute responses. Treating these responses as equivalent therefore hides behavior that matters for interpretation.

![](images/51402424209de4f6ccc6b3e2904b8474d1ba9b0ef7e4e165408066fbbe0dd568.jpg)  
Figure 1: DECAF at a glance. The paired reveal measures a signed stage response, while the clean endpoint supplies an activity gate and orientation. DECAF routes the same response into endpoint-aligned evidence, endpoint-opposed contradiction, or endpoint-null fragility, with exact conservation. The compact summary reports the two headline consequences: mechanism recovery after ordinary magnitude is matched, and substantially more stable cross-protocol diagnostics.

We introduce DECAF in Figure 1, short for Decomposition of Evidence, Contradiction, And Fragility. DE-CAF resolves this ambiguity by using the final response between the fully observed pair as a semantic reference for the entire reveal process. Its magnitude determines whether the factor has a meaningful final effect, while its sign provides the reference direction. Intermediate responses that agree with this direction are routed to evidence, while responses that oppose it are routed to contradiction. If the fully observed pair shows little response, there is no meaningful final effect to align with, and intermediate responses are routed to fragility. We call this semantic routing: the observed responses are assigned different roles according to their relation to the final contrast, without changing the trajectories themselves. The routing is lossless, so evidence, contradiction, and fragility sum exactly to the ordinary response magnitude. DECAF is model-agnostic, therefore explaining any black-box model that returns a scalar score.

Relation to prior work. Gradient and path methods assign contributions to input coordinates or internal units (Simonyan et al., 2014; Selvaraju et al., 2017; Sundararajan et al., 2017; Shrikumar et al., 2017; Bach et al., 2015). Removal and sampling methods measure prediction changes under masking or replacement (Fong & Vedaldi, 2017; Fong et al., 2019; Petsiuk et al., 2018; Covert et al., 2021b). Counterfactual and causal approaches define meaningful contrasts (Goyal et al., 2019; Chattopadhyay et al., 2019; Janzing et al., 2020). Evaluation work shows that baselines, removal operators, and off-manifold inputs can alter explanation behavior (Adebayo et al., 2018; Hooker et al., 2019; Rong et al., 2022; Jethani et al., 2023). DECAF begins after a paired response has been measured. It refines the response by its relation to the clean endpoint. Unlike positive–negative feature attribution, it also contains an endpoint-null branch. Unlike amortized explainers such as FastSHAP, it requires no learned explanation model (Jethani et al., 2022). A detailed comparison with feature attribution, counterfactual explanation, evaluation frameworks, and efficient black-box methods appears in Appendix A.

## We make five contributions:

• Problem. We identify a semantic gap in perturbation explanation: response magnitude tells us how strongly a model reacts, but not what that reaction means.

• Observation interface. We introduce paired reveal, which tracks a factual–counterfactual contrast along matched trajectories as information becomes available.

• Algorithm. We introduce DECAF, which uses the final contrast to route intermediate responses into evidence, contradiction, and fragility. The decomposition is lossless and unique under simple axioms, and adds no model queries beyond the paired trajectories.

• Behavioral meaning. Controlled vision and tabular experiments show that the three components track learned reliance, effect reversal, and fragility across neural, linear, and tree-based models.

• Practical consequence. On ImageNet-9, DECAF recovers response semantics after magnitude is matched and diagnoses what changes across perturbation paths. It also outperforms the tested general-purpose attribution baselines on FunnyBirds and ImageNet-1k and scales efficiently to a large DINOv2 vision transformer.

## 2 PAIRED INFORMATION REVEALING: FORMAL SETUP

We formalize paired information revealing. Let $q : \mathcal { X } $ R denote the model score. An input $\mathbf { x } ^ { + }$ and a factor-level counterfactual $\mathbf { x } ^ { - }$ define the final contrast

$$
d = q ( \mathbf { x } ^ { + } ) - q ( \mathbf { x } ^ { - } ) .
$$

The pair need not be pre-specified: in feature attribution, $\mathbf { x } ^ { - }$ can be induced from $\mathbf { x } ^ { + }$ by a removal or replacement operator.

A paired reveal starts from a common uninformative state $\mathbf { x } _ { \mathrm { 0 } }$ and produces matched trajectories $( { \bf x } ^ { \dagger } ( t ) , { \bf x } ^ { - } ( t ) )$ for $t \in \mathcal { T } = [ 0 , 1 ]$ , with

$$
\mathbf { x } ^ { + } ( 0 ) = \mathbf { x } ^ { - } ( 0 ) = \mathbf { x } _ { 0 } , \qquad \mathbf { x } ^ { + } ( 1 ) = \mathbf { x } ^ { + } , \qquad \mathbf { x } ^ { - } ( 1 ) = \mathbf { x } ^ { - } .
$$

The signed response is $r ( t ) = q ( \mathbf { x } ^ { + } ( t ) ) - q ( \mathbf { x } ^ { - } ( t ) )$ , with $d = r ( 1 )$ . Ordinary magnitude $| r ( t ) |$ preserves size but discards its relation to the final contrast; Section 3 uses it to assign semantic roles.

## 3 DECAF: A SEMANTIC DECOMPOSITION OF PAIRED RESPONSES

Section 2 gives us two quantities: the response $r ( t )$ at each reveal level and the final contrast $d = r ( 1 )$ DECAF uses the final contrast only as a semantic reference for interpreting the intermediate responses. If the final contrast is substantial, its direction tells us which responses agree with the model’s eventual behavior and which oppose it. If the final contrast is negligible, intermediate responses cannot be interpreted as supporting or opposing a meaningful final effect.

Choose a practical threshold $\epsilon > 0$ and define

$$
a = { \bf 1 } \{ | d | \geq \epsilon \} , \quad \quad s = \mathrm { s i g n } ( d ) .
$$

The gate a determines whether the final contrast is active, while s orients the trajectory when it is active. Let

$$
z ( t ) = s r ( t ) ,
$$

and write $z ^ { + } = \operatorname* { m a x } \{ z , 0 \}$ and $z ^ { - } = \operatorname* { m a x } \{ - z , 0 \}$ . DECAF routes the response as

$$
\begin{array} { r } { ( e ( t ) , c ( t ) , f ( t ) ) = \big ( a z ^ { + } ( t ) , a z ^ { - } ( t ) , ( 1 - a ) | r ( t ) | \big ) . } \end{array}
$$

An active response aligned with the final contrast becomes evidence; an active response pointing in the opposite direction becomes contradiction; and a response on a final-contrast-null pair becomes fragility. This logic is illustrated earlier in Figure 1.

Let $\mu$ be a normalized measure over stages and let the outer expectation cover paired examples and protocol randomness. We report

$$
M = \mathbb { E } | d | , \qquad ( E , C , F ) = \mathbb { E } \int _ { \mathcal { T } } ( e ( t ) , c ( t ) , f ( t ) ) \mathrm { d } \mu ( t ) , \qquad \mathrm { A b s } = E + C + F .\tag{1}
$$

The finite-grid implementation replaces the integral by quadrature. No additional model query is needed once the signed paired trajectory is available. See Appendices B and C for finite-grid estimation, conditional summaries, threshold sensitivity, and streaming accumulation.

Forward-only implementation. The model enters only through evaluations of q. DECAF needs no gradients, parameters, or internal activations. It can run against a neural network, a tree ensemble, or a remote score API. Examples, stages, factors, paths, and models are independent batching dimensions.

Table 1: Model calls per image in ImageNet-9.
<table><tr><td>Method</td><td>Fwd.</td><td></td><td>Bwd. Internal Learned</td></tr><tr><td>DECAF-9</td><td>18</td><td>0</td><td>No No</td></tr><tr><td>Input× Grad</td><td>1</td><td>1</td><td>Yes No</td></tr><tr><td>IG</td><td>16</td><td>16 Yes</td><td>No</td></tr><tr><td>SmoothGrad</td><td>16</td><td>16</td><td>Yes No</td></tr><tr><td>BlurIG</td><td>12</td><td>12</td><td>Yes No</td></tr><tr><td>Occlusion</td><td>49</td><td>0</td><td>No No</td></tr><tr><td>RISE</td><td>256</td><td>0</td><td>No No</td></tr></table>

## Algorithm 1: DECAF in four steps.

Require: score q, paired path $( { \bf x } ^ { + } ( t ) , { \bf x } ^ { - } ( t ) ) , \varepsilon$   
1: d $ q ( \mathbf { x } ^ { + } ) - q ( \mathbf { x } ^ { - } ) ; a  \mathbf { 1 } \{ | d | \geq \varepsilon \} ; s  \mathrm { s i g n } ( d )$   
2: for all $t \in \tau$ do ▷ r(t = 1) = d is cached and reused   
3: $r ( t ) \gets q ( \mathbf { x } ^ { + } ( t ) ) - q ( \mathbf { x } ^ { - } ( t ) )$   
4: $( e , c , f ) \gets \big ( a ( s r ) ^ { + } , a ( s r ) ^ { - } , ( 1 - a ) | r | \big )$   
5: end for   
6: return $M , E , C , F$ by averaging over stages and ex  
amples

Scope of the semantics. DECAF assigns observable, endpoint-relative roles to paired responses; it does not infer the latent cause of a trajectory response. These response roles are defined relative to the score $q ,$ the chosen factual–counterfactual pair, the reveal trajectory, and the practical threshold ϵ. In particular, fragility denotes response on a pair whose final contrast is negligible under this specification. It does not by itself distinguish off-manifold artifacts, boundary uncertainty, calibration effects, or other possible causes. An active pair may still be path-sensitive, but its response is described through aligned and opposed mass rather than $F .$

## 4 THEORETICAL FOUNDATIONS OF SEMANTIC ROUTING

Section 3 defines a simple routing rule from a stage response $r ( t )$ and final contrast d to evidence, contradiction, and fragility. We first ask whether this routing is arbitrary.

To formalize this routing, we establish three operational axioms that reflect practical explanation goals. First, conservation ensures that no response magnitude is lost or invented. Second, endpoint gating isolates sensitivity on pairs whose final contrast is negligible under the chosen threshold. Third, directional support distinguishes intermediate responses that move toward the model’s final decision from those that oppose it.

Theorem 1 (Unique hard-gated semantic routing). Fix $a \in \{ 0 , 1 \}$ , endpoint orientation $s \in \{ - 1 , 1 \}$ on the active branch, and response $r \in \mathbb { R }$ . The unique nonnegative triple satisfying conservation, $e + c + f = | r | ;$ endpoint gating, $f = 0$ for $a = 1$ and $e = c = 0$ for $a = 0 ,$ ; and directional support, $c = 0$ when $s r \geq 0$ and $e = 0$ when $s r \leq 0 ,$ , is

$$
( e , c , f ) = \big ( a ( s r ) ^ { + } , a ( s r ) ^ { - } , ( 1 - a ) | r | \big ) .
$$

Theorem 1 removes tunable mixtures once the endpoint gate and orientation are specified. It establishes uniqueness within the stated endpoint-relative semantics, not uniqueness of hard endpoint gating itself. Under these operational axioms, the routing is uniquely determined. See Appendix Bfor the proofand the positive–negative (Jordan) decomposition interpretation.

Theorem 1 shows that the routing is unique once the endpoint-relative semantics are fixed. We next ask whether retaining the three routed components provides information beyond ordinary magnitude.

Theorem 2 (DECAF strictly refines response magnitude). Ordinary response magnitude is a deterministicfunction ofthe DECAF profile,

$$
{ \mathrm { A b s } } = E + C + F ,
$$

so every decision rule based on Abs can also be implemented from $( E , C , F )$ . The reverse recovery fails in general: for every $m > 0 ,$ , the distinct profiles $( m , 0 , 0 ) , ( 0 , m , 0 )$ , and $( 0 , 0 , m )$ all produce $\mathrm { A b s } = m$ . Hence the refinement is strict for any decision problem that distinguishes two such profiles with positive probability.

Thus DECAF is a lossless refinement of ordinary magnitude: any analysis based on Abs can be reproduced from $( E , C , F )$ , while the reverse recovery is impossible in general. The refinement is strict whenever a decision problem distinguishes response profiles that share the same magnitude. For example, if evidence, contradiction, and fragility are equally likely and all produce magnitude $m ,$ , every magnitude-only classifier receives the same observation and cannot exceed $1 / 3$ accuracy, whereas the full profile distinguishes the three roles. Section 6 tests this consequence empirically after explicitly matching ordinary response magnitude. See Appendix B for the Bayes limit, strict information refinement, and unequal priors.

Theorem 2 shows that magnitude can collapse distinct response roles. A practically important case is the difference between an effect that disappears and one that reverses. Both can reduce aligned evidence, but only reversal should create contradiction.

Suppose the final contrast is active with magnitude $m > 0$ . Let $\eta \in [ 0 , 1 ]$ denote the probability that this effect is altered by context. With probability $1 - \eta ,$ , the response remains aligned with magnitude $m ;$ with probability $\eta ,$ it is either suppressed to zero (attenuation) or reversed to −m (inversion). We consider three simple response regimes: preservation, where the effect keeps its direction; attenuation, where it weakens toward zero; and inversion, where it reverses direction.

Proposition 1 (Contradiction separates attenuation from inversion). Under equal aligned and opposed magnitudes,
<table><tr><td>Regime</td><td>E</td><td>C</td><td>Abs</td></tr><tr><td>Preservation</td><td>m</td><td>0</td><td>m</td></tr><tr><td>Attenuation to zero</td><td> $( 1 - \eta ) m$ </td><td>0</td><td> $( 1 - \eta ) m$ </td></tr><tr><td>Inversion</td><td> $( 1 - \eta ) m$ </td><td>ηm</td><td>m</td></tr></table>

In the inversion regime, $C / ( E + C ) = \eta .$

Attenuation and inversion can remove the same amount of aligned evidence, but only inversion creates contradiction. Thus C distinguishes loss of an effect from reversal of that effect. The controlled experiment in Section 5. See Appendix B for the proof, unequal magnitudes, and continuous mixtures.

When contradiction and fragility vanish, conservation gives $\mathbf { A b s } = E$ . Thus DECAF reduces exactly to ordinary magnitude when the response is already fully aligned; it introduces additional distinctions only when the observed behavior contains them.

## 5 BEHAVIORAL VALIDATION ACROSS MODELS AND MODALITIES

The preceding theory defines endpoint-relative response roles. We now test whether these roles correspond to model behavior measured independently of DECAF. We begin with the 3D Shapes dataset, whose generative factors can be changed one at a time while the remaining factors are held fixed (Burgess et al., 2018). This gives exact factual–counterfactual pairs and lets us formulate three controlled learning problems in which reliance, off-path sensitivity, and effect reversal can be measured independently of DECAF. Across the controlled suite, we use ResNet-18 and a small ViT. The base grid contains 30 trained models and 180 model–factor units, with additional dedicated checkpoints and training variants for the three behavioral tests. Comparing DECAF’s decomposition with these independent behavioral measurements then tests whether each component carries the intended response semantics.

Evidence. We train object-shape classifiers in an environment where background wall color is correlated with the label, so models may rely on either this shortcut or the object itself. Reversing the wall–label correlation at test time gives an independent measure of shortcut reliance: a wall-dependent model loses accuracy, while a shape-dependent model remains stable. Across 52 training checkpoints, the evidence margin $E _ { \mathrm { w a l l } } - E _ { \mathrm { s h a p e } }$ correlates 0.936 with this reversal vulnerability, with a 90% bootstrap interval of [0.885, 0.961] (Figure 2(a)). Figure 2(b) shows the same relation during a within-model strategy transition; complete trajectories appear in Appendix E.

Fragility. We construct models for which floor color has little effect once the image is fully revealed, but different effects before full reveal. To create this difference, we expose models during training to partially revealed inputs while preserving the original shape task. Infragile training, the model is encouraged to react to changes in floor color at these partial states. In robust training, it is instead encouraged to remain unchanged, while neutral training uses only the original clean task without either objective. We then measure the resulting sensitivity independently using held-out floor-color interventions. If F captures endpoint-null path sensitivity, it should track this prediction-change rate. It does so closely (Figure 2(c)); intervention separation and cross-geometry checks appear in Appendix E.

Contradiction. We construct three tasks with the same object-color effect in the original wall context but different behavior when the wall context changes. In Direct, the effect is preserved; in Gate, it disappears; and in Invert, it reverses. The resulting label behavior—preservation, collapse, or swap—therefore provides an independent measure of what happened to the effect. If C captures opposition rather than mere weakening, it should remain near zero for Direct and Gate and increase only when the effect reverses. Across 30 models and all mismatch levels, C tracks the pairwise label-swap rate with $\rho = 0 . 9 6 1$ , whereas ordinary magnitude does not $( \rho = - 0 . 0 3 6 ;$ Figure 2(d)). Detailed regime separation and calibration appear in Appendix E.

Transfer across model classes and modalities. We next ask whether the same response roles survive beyond controlled vision models. We use a balanced 240,000-example subset of Covertype (Blackard, 1998) with 54 natural features and append a binary context and a binary candidate factor. In one set of tasks, the candidate factor has the same effect in the reference context, while changing the context either preserves, removes, or reverses that effect. In a second set, the factor remains nearly irrelevant in the reference context but can affect predictions under the alternate context. We train 135 classifiers spanning linear, tree-based, and neural models.

![](images/55eae11afd922930100ab340f6ab462f3100d684f360fee9bd26cf11baa06c2b.jpg)

![](images/f7476de292f1337e306c116a858b60220c94112beff000d386073e501fb70289.jpg)  
(a) Population-level re- (b) Within-model transition. (c) Endpoint-null sensitivliance. ity.

![](images/f1cab89dfdeb4081ccc630a7328244b3c1f5d22a2c2c642b834c460f81847fa9.jpg)

![](images/9ee567ab5a689b7991b4ba9751eae9d6ce68997ebc19ac177df7de4d1a574968.jpg)  
(d) Label-level reversal.  
Figure 2: The three response roles track independently measured model behavior. (a–b) Evidence follows shortcut reliance across checkpoints and through a within-model strategy transition. (c) Fragility follows offpath prediction change on endpoint-null pairs. (d) Contradiction follows pairwise label reversal. Additional intervention-separation, architecture-specific, and cross-geometry diagnostics appear in Appendix E.

Table 2: Cross-model transfer of the three response roles on Covertype. Entries are Spearman correlations with held-out realized behavior across 135 classifiers. The DECAF column uses E for preservation, C for actual inversion, and F for alternate-context prediction change restricted to endpoint-null pairs. Brackets give 95% joint family/seed bootstrap intervals. † SHAP interaction is available only for tree models. Complete baselines, model-family results, threshold-conditioned analyses, and measured cost appear in Appendix I.
<table><tr><td>Held-out behavior</td><td>Role</td><td>DECAF ρ</td><td>Abs</td><td>M</td><td>Native SHAP</td><td>SHAP inter.</td></tr><tr><td>Preservation</td><td>E</td><td>0.864 [0.832,0.895]</td><td>0.804</td><td>0.657</td><td>0.781</td><td>-0.251†</td></tr><tr><td>Actual inversion</td><td>C</td><td>0.987 [0.973,0.997]</td><td>-0.207</td><td>-0.001</td><td>-0.205</td><td>0.093†</td></tr><tr><td>Endpoint-null change</td><td>F</td><td>0.974 [0.942,0.988]</td><td>0.588</td><td>0.148</td><td>0.481</td><td>0.069†</td></tr></table>

Crucially, these task constructions specify what the training data attempt to induce, not what the model necessarily learns. We therefore determine preservation, inversion, and endpoint-null sensitivity directly from held-out predictions. If the response roles transfer, E should track realized preservation, C realized inversion, and F endpoint-null prediction change. See Table 2.

Remark that some models trained under the inversion construction instead suppress the candidate effect, while the fragility construction becomes ordinary endpoint evidence for logistic regression. DECAF follows the behavior realized by the trained model rather than the behavior intended by the data generator; the family-level audit appears in Appendix I.

## 6 REAL-WORLD AUDIT: RESPONSE SEMANTICS ON IMAGENET-9

Section 5 established the meanings of E, C, and F in controlled learning problems. We now ask whether the same response semantics describe behavior on natural images, including models that were never trained for our audit. ImageNet-9 is useful for this purpose because its background variants keep the foreground object fixed while changing only the background (Xiao et al., 2021). This gives natural factual–counterfactual pairs in which we can ask whether a model’s response to background change behaves as evidence, contradiction, or endpoint-null sensitivity.

Setup. We evaluate two complementary groups of models on these same background interventions. The first consists of 24 off-the-shelf ImageNet-1k classifiers. These models were trained independently of our experiment, so they test whether the response semantics established in Section 5 remain meaningful for existing natural-image models. Their 1,000-way predictions are aggregated into the nine ImageNet-9 superclasses before evaluation.

The second group contains 48 models trained directly on ImageNet-9. We fine-tune six backbones— ResNet-50, ConvNeXt-Tiny, EfficientNet-B3, RegNetY-8GF, Swin-T, and ViT-B/16—on four versions of the training data: the original images, images with randomly reassigned backgrounds, images with a fixed next-class background, and foreground-only images. These training conditions deliberately produce models with different degrees and directions of background dependence. This group therefore tests whether the same response semantics continue to track behavior as the model’s learned use of background changes. Together, the two groups form a 72-model zoo: one provides models trained independently of our audit, while the other provides controlled diversity in the behavior being audited.

![](images/05241b47dd8f4b1b26bfdccc35163971d91b443b70d443a33520e7ec87eead83.jpg)  
(a) Behavioral alignment.

![](images/315d93a4e27fcb1462ab577a497baf7602e116490102130084cc187655b40b2e.jpg)  
(b) Response-role accuracy.

![](images/7e27606154cfead82d34653cd645d1b0d1cc4f5cca490aa2dc56eb261cf4eb76.jpg)  
Abs quantile bin  
(c) Across magnitude bins.

Figure 3: Response semantics transfer to ImageNet-9. (a) Evidence and fragility align with independently measured natural-image behaviors. (b) Nearly identical response magnitudes can correspond to different response semantics, which DECAF preserves. (c) The advantage persists across magnitude bins.  
![](images/02e00331ef71ae1bb48324439b47e44f9e8ad6c3ba41ab0f39cdcf1a89ddbf1c.jpg)  
● Same–Rand Same–Next

![](images/5c2767b89e510fd6c9f5b84f617943348dea38fc0e626d9e14ecbf2878987a72.jpg)  
(a) Response composition under patch reveal.  
(b) Model-rank transfer across reveal paths.  
Figure 4: Reveal paths change response composition and ranking. (a) Patch reveal increases ordinary magnitude mainly through contradiction and fragility rather than evidence. (b) Ordinary-magnitude rankings transfer poorly across reveal paths, while evidence and contradiction remain substantially more stable.

## 6.1 RESPONSE SEMANTICS SURVIVE NATURAL IMAGES

The first question is whether the response roles validated in Section 5 remain connected to independently measured behavior on natural images. We consider two behaviors. Background reliance asks whether changing only the background substantially disrupts the model’s prediction. Endpoint-null sensitivity asks whether a pair with little final background effect can nevertheless remain sensitive to held-out background corruptions. Both behaviors are defined independently of DECAF. If the response semantics transfer, E should identify the former and F the latter.

For background reliance, we mark a pair as positive when the model correctly classifies the image with a same-class background, but replacing that background with a random-class one either changes the predicted class or lowers the true-class probability by at least 0.20. We then ask whether these behavior-positive pairs tend to receive larger response scores. Using E directly as a ranking score gives an AUROC of 0.930: a value of 0.5 corresponds to random ranking and 1 to perfect separation. Ordinary response magnitude reaches 0.900, while endpoint magnitude reaches 0.960 (Figure 3(a)). The strong endpoint result is expected because this behavioral target is itself defined by a prediction change at the endpoint.

For endpoint-null sensitivity, we first restrict attention to pairs whose final background effect is negligible, |d| < 0.02. We then apply a separate set of background corruptions that are not used to construct the reveal path. A pair is marked positive if any of these held-out corruptions changes the predicted class or shifts the true-class probability by at least 0.20. We ask whether these independently identified sensitive pairs tend to receive larger F values than the remaining endpoint-null pairs. F reaches AUROC 0.878, compared with 0.433 for ordinary response magnitude, 0.316 for endpoint magnitude, and 0.635 for SmoothGrad (Figure 3(a)).

Together, the two tests show complementary behavior: E tracks consequential background use when the final effect is present, whereas F exposes sensitivity that remains when that final effect is negligible. Appendix F gives the full pair construction, held-out corruption set, behavioral-label definitions, and baseline details.

## 6.2 SAME RESPONSE MAGNITUDE, DIFFERENT RESPONSE SEMANTICS

We next ask whether the decomposition reveals information that ordinary response magnitude does not already contain. We reuse the independently defined evidence and fragility indicators from Section 6.1 and add a contradiction indicator: contradiction behavior is present when changing only the background makes the model switch from the foreground class to the class associated with the new background. These three behavioral indicators may overlap.

We then compare cases with different behavioral patterns but nearly identical ordinary response magnitudes. Requiring their Abs values to differ by at most 5% yields 8,289 matched comparisons. Because a case may exhibit more than one behavior, we do not force it into a single ground-truth class. Instead, we ask whether the largest of $E , C ,$ and $F$ corresponds to a behavior that is actually present; overlap and ties are handled as detailed in Appendix F.

Under this test, a single magnitude provides no basis for preferring evidence, contradiction, or fragility. Abs reaches a role-agreement accuracy of 0.350, whereas DECAF reaches 0.964 (Figure 3(b)). Thus, nearly identical response magnitudes can accompany substantially different observed behaviors, while their decomposition preserves this distinction.

The conclusion also holds beyond the matched cases. Within narrow bins of ordinary response magnitude, macro-AUROC rises from 0.517 for Abs to 0.677 for DECAF, and DECAF exceeds Abs in 15 of the 16 bins that contain all three behavioral indicators (Figure 3(c)). Appendix F gives the exact matching rule, overlap and tie handling, per-bin support, and complete baseline comparisons.

## 6.3 WHAT CHANGES WHEN THE REVEAL PATH CHANGES?

We finally ask whether the same endpoint information can produce different conclusions when it is revealed differently. We keep the model and both factual–counterfactual endpoints fixed. In one path, the whole image gradually changes from a common blurred image toward each endpoint; in the other, the same endpoint information is revealed region by region. We call them the blend and patch paths.

Ordinary response magnitude increases by about 1.8× under the patch path, but the increase is not uniform across response roles (Figure 4(a)). Evidence remains close to its blend value, contradiction grows by about 1.8×, and fragility by more than 4×. This is consistent with their semantics: with the endpoints fixed, evidence remains oriented by the same final contrast, whereas fragility explicitly measures sensitivity along the chosen reveal path. The pattern is not guaranteed by the definition, but here the larger ordinary response comes primarily from fragility rather than additional evidence.

We also ask whether the reveal path changes conclusions about how models compare. For each model, we average E, C, F, and Abs over test images and rank the models separately by each quantity under the blend and patch paths. Spearman correlation measures the agreement between the two rankings. Ordinary-magnitude rankings are highly unstable $( \rho = 0 . 1 7 / 0 . 2 6$ for Same–Rand/Same–Next), whereas evidence rankings remain at 0.86/0.77 and contradiction rankings near 0.93. Fragility is intermediate at 0.69/0.71, consistent with its greater path dependence (Figure 4(b)). Patch-order and threshold checks are reported in Appendix G.

## 7 EXTERNAL ATTRIBUTION BENCHMARKS AND LARGE-MODEL SCALING

Sections 5–6 validate the response semantics of DECAF. We now ask whether it also works as ordinary feature attribution under targets defined without $E , C ,$ , or F. Here the pair is constructed from the feature intervention: $\mathbf { x } ^ { + }$ is the original image and $\mathbf { x } ^ { - k }$ changes only part or patch k using a fixed removal or replacement rule; $E _ { k }$ ranks the features. FunnyBirds evaluates semantic parts under two separate replacement interventions, while ImageNet-1k IDSDS evaluates 16 fixed patches by single-patch deletion. Quality is the per-image Spearman correlation between the attribution ranking and this intervention-based ranking. Both datasets use ResNet-50, VGG-16, and ViT-B/16, with every method evaluated on the same eligible images within each model. DECAF-3/5/9 use three, five, or nine reveal stages.

Table 3: External attribution quality and measured ImageNet-1k compute. Quality is macro-averaged equally over the three architectures; time and memory are ImageNet-1k measurements. KernelSHAP is shown separately because it directly queries the same patch-deletion intervention used to define the ImageNet-1k evaluation target, giving it evaluation-specific information unavailable to the general baselines and DECAF.
<table><tr><td></td><td colspan="5">FunnyBirds ImageNet-1k ms/img Peak GiB</td></tr><tr><td>Method</td><td>Access</td><td>ρ↑</td><td> $\rho \uparrow$ </td><td>↓</td><td>↓</td></tr><tr><td colspan="6">DECAF</td></tr><tr><td>DECAF-3</td><td>Forward</td><td>0.372</td><td>0.359</td><td>22.1</td><td>11.2</td></tr><tr><td>DECAF-5</td><td>Forward</td><td>0.403</td><td>0.367</td><td>36.7</td><td>11.2</td></tr><tr><td>DECAF-9</td><td>Forward</td><td>0.406</td><td>0.379</td><td>65.9</td><td>11.2</td></tr><tr><td colspan="6">General-purpose attribution</td></tr><tr><td>DeepLIFT</td><td>Backward</td><td>0.197</td><td>0.341</td><td>3.1</td><td>6.6</td></tr><tr><td>IG-32</td><td>Backward</td><td>0.271</td><td>0.242</td><td>28.7</td><td>50.4</td></tr><tr><td>IG-U-32</td><td>Backward</td><td>0.200</td><td>0.295</td><td>28.7</td><td>50.4</td></tr><tr><td>RISE-512</td><td>Sampling</td><td>0.302</td><td>0.179</td><td>216.6</td><td>14.4</td></tr><tr><td colspan="6">Endpoint-only reference</td></tr><tr><td>Endpoint M</td><td>Endpoint</td><td>0.324</td><td>0.371</td><td></td><td></td></tr><tr><td colspan="6">Benchmark-aligned reference (uses the evaluation intervention)</td></tr><tr><td>KernelSHAP-512 Sampling</td><td></td><td>0.299</td><td>0.447</td><td>216.4</td><td>14.4</td></tr></table>

Table 4: DINOv2 ViT-g/14 on 238 PartImageNet strict-common-support images.
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td rowspan="2">Spearman sec/img Peak GiB ↓</td><td rowspan="2">↓</td></tr><tr><td>↑</td></tr><tr><td>DECAF-3</td><td>0.208</td><td>0.190</td><td>5.53</td></tr><tr><td>DECAF-5</td><td>0.215</td><td>0.300</td><td>5.66</td></tr><tr><td>DECAF-9</td><td>0.220</td><td>0.526</td><td>6.33</td></tr><tr><td>IG-16</td><td>0.222</td><td>0.710</td><td>9.01</td></tr><tr><td>IG-32</td><td>0.213</td><td>1.425</td><td>13.39</td></tr><tr><td>GradientSHAP</td><td>0.171</td><td>0.407</td><td>27.67</td></tr><tr><td>SmoothGrad-16</td><td>0.042</td><td>0.438</td><td>26.43</td></tr><tr><td>DeepLIFT</td><td>0.068</td><td>0.084</td><td>7.19</td></tr></table>

Table 5: Gain by the trajectory beyond endpoint-only M. Entries are paired $\dot { \Delta \rho } =$ $\rho ( \bar { \mathrm { D E C A F } } ) - \rho ( M )$ with 95% confidence intervals. Larger values indicate better.
<table><tr><td>Contrast</td><td>FunnyBirds different intervention</td><td>ImageNet-1k same deletion</td></tr><tr><td>DECAF-3 -M</td><td>+.049 [.033, .064]</td><td>-.012[−.014, −.010]</td></tr><tr><td>DECAF-5 -M</td><td>+.080 [.064, .096]</td><td>-.004 [−.007, −.002]</td></tr><tr><td>DECAF-9 -M</td><td>+.083 [.067, .099]</td><td>+.007 [.004, .010]</td></tr></table>

All three DECAF trajectories exceed every listed general-purpose baseline on both datasets (Table 3). KernelSHAP reaches 0.447 on ImageNet-1k, but it directly queries the same deletion game used to define that target and is therefore shown separately. The full 50,000-image ImageNet check preserves the ordering $\mathrm { \bar { D E C A F  – 5 } > I G - U - 3 2 > \bar { I G } - 3 2 }$ . Appendix H also reruns the official FunnyBirds RISE-6000 Single Deletion protocol and reproduces the published score within 0.005 on all three architectures (Hesse et al., 2023)

The compute advantage becomes clearer at larger model scale. On the 1B-parameter model DINOv2 $V i T – g / l 4 ,$ DECAF-5 and IG-32 have nearly identical quality (0.215 versus 0.213), while IG-32 requires 4.75× the wall time and 2.36× the peak allocated memory (Table 4).

Table 5 clarifies what the trajectory adds. On FunnyBirds, evaluation changes parts differently from the operation used to form $\dot { \mathbf { x } } ^ { - k }$ ; the trajectory adds about 0.08 Spearman beyond M. ImageNet-1k instead evaluates the same patch deletion used to form $\mathbf { x } ^ { - k }$ , so M is already aligned with the target; five stages nearly match it, while nine stages add only 0.007. Endpoint information therefore carries most of the attribution signal when evaluation repeats the same intervention. The trajectory adds its clearest value when the ranking must transfer to a different intervention. Complete intervals, per-architecture results, and boundary cases are in Appendix H.

## 8 DISCUSSION AND CONCLUSION

Perturbation magnitude tells us how strongly a model responds, but responses of the same size can reflect different behaviors. DECAF uses the final factual–counterfactual contrast to separate intermediate responses into evidence, contradiction, and fragility, while preserving ordinary magnitude exactly, ${ \mathrm { A b s } } = E + C + F$

The experiments show that these distinctions are not artifacts of the definitions. They track independently measured behavior in controlled vision and tabular settings, remain distinguishable on natural images after magnitude is controlled, and remain useful under external attribution criteria. ImageNet further shows why this matters: changing only the reveal path can greatly increase total response without increasing evidence.

These response roles are relative to the chosen counterfactual pair, score, and reveal path; they are not claims about hidden causal mechanisms. The endpoint most directly describes its own intervention, while the trajectory is most useful beyond that intervention. Magnitude tells us how much a model reacts; DECAF preserves that quantity while revealing what kind of response produced it.

## APPENDICES

A Detailed Related Work 10   
B Theoretical Foundations and Proofs 11   
C Estimators, Complexity, and Implementation 13   
D Controlled Experimental Setup 14   
E Selected Controlled Results 16   
F ImageNet-9 Experimental Details 19   
G ImageNet-9 Protocol Robustness Checks 21   
H Forward-Only Attribution: Complete Results and Boundaries 22   
Covertype Experimental Details and Complete Results 25

## A DETAILED RELATED WORK

The closest literatures all study model response, but they attach different objects and semantics to that response. We organize the comparison around the object being explained and the question it answers.

## A.1 GRADIENT, PATH, AND SIGNED ATTRIBUTION

Gradient explanations measure local sensitivity, while path methods accumulate sensitivity from a baseline to the input (Simonyan et al., 2014; Selvaraju et al., 2017; Sundararajan et al., 2017; Smilkov et al., 2017; Xu et al., 2020; Kapishnikov et al., 2021; You et al., 2026). Layer-wise relevance propagation, DeepLIFT, FullGrad, and SHAP instead decompose a prediction into additive feature contributions under conservation, reference, or game-theoretic principles (Bach et al., 2015; Shrikumar et al., 2017; Srinivas & Fleuret, 2019; Lundberg & Lee, 2017). Some of these methods preserve positive and negative contributions (Dhurandhar et al., 2018b).

DECAF studies a different object. It does not distribute one prediction across input coordinates. It decomposes the scalar response of an already specified paired intervention. The clean endpoint supplies the orientation, and the endpoint-null branch separates fragility from positive or negative evidence. Spatial saliency and DECAF are therefore complementary: saliency indicates where a response is localized, whereas DECAF indicates what semantic role the paired response plays.

## A.2 PERTURBATION, COUNTERFACTUAL, AND CAUSAL EXPLANATION

Occlusion, Meaningful Perturbations, Extremal Perturbations, RISE, LIME, and removal-based Shapley methods construct interventions and summarize the resulting model changes (Zeiler & Fergus, 2014; Fong & Vedaldi, 2017; Fong et al., 2019; Petsiuk et al., 2018; Ribeiro et al., 2016; Covert et al., 2021b). Counterfactual and contrastive methods retrieve or synthesize changes that alter a decision or support a contrast (Goyal et al., 2019; Chang et al., 2019; Dhurandhar et al., 2018b; You et al., 2025; Gu et al., 2026). Causal attribution and concept-intervention methods interpret relevance through explicit interventions on inputs, representations, or human-interpretable concepts (Chattopadhyay et al., 2019; Janzing et al., 2020; Goyal et al., 2020; Kim et al., 2018; Koh et al., 2020).

DECAF takes the paired intervention as input rather than searching for it. Counterfactual construction defines the factor and identification assumptions; the paired trajectory measures the response; endpoint orientation then decomposes that response into evidence, contradiction, and fragility. The same decomposition can therefore sit on top of different removal operators or counterfactual generators.

## A.3 BASELINES, OFF-MANIFOLD EFFECTS, AND FAITHFULNESS EVALUATION

Baseline and path choices can substantially change attribution (Sturmfels et al., 2020; Kindermans et al., 2019), and perturbations can move samples away from the data manifold (Frye et al., 2020).

Sanity checks, retraining-based evaluations, infidelity measures, and debiased removal metrics test whether explanations reflect the model rather than artifacts of the evaluation protocol (Adebayo et al., 2018; Hooker et al., 2019; Yeh et al., 2019; Rong et al., 2022). Other work documents label leakage, explanation fragility, and adversarial manipulation (Jethani et al., 2023; Ghorbani et al., 2019; Dombrowski et al., 2019; Slack et al., 2020).

DECAF does not claim path invariance. It keeps the intervention protocol explicit and asks which component changes when the protocol changes. This distinction matters in ImageNet-9: nested-patch reveal raises total response by roughly 80%, but the increase comes primarily from contradiction and fragility rather than evidence. DECAF addresses a semantic ambiguity that remains even when a paired response has been measured faithfully.

## A.4 EFFICIENCY, BLACK-BOX ACCESS, AND BENCHMARK CONTEXT

Black-box perturbation methods trade model access for repeated queries; RISE and dense occlusion can require many forward evaluations, while amortized methods such as FastSHAP add a separate explainer-training problem (Petsiuk et al., 2018; Jethani et al., 2022). DECAF instead reuses the signed scores already computed by a paired perturbation curve. The routing step adds no model queries, requires no gradients or internal activations, and can be batched across examples, stages, factors, and models.

ImageNet-9 is a standard setting for studying background reliance and distribution shift (Xiao et al., 2021; Geirhos et al., 2020). DECAF asks a finer question than whether a model responds to background: is that response endpoint-aligned evidence, endpoint-opposed contradiction, or sensitivity that appears only when the endpoint effect is null?

## B THEORETICAL FOUNDATIONS AND PROOFS

This appendix develops the compact theory used in the main text. We begin with the pointwise decomposition, then lift it to a trajectory, a population, and a decision problem. The proofs require only integrability of the signed response and do not assume differentiability, smoothness, or a particular model class.

## B.1 POSITIVE AND NEGATIVE PARTS

For a real number u, define $u ^ { + } = \operatorname* { m a x } \{ u , 0 \}$ and $u ^ { - } = \mathrm { m a x } \{ - u , 0 \}$ . Then

$$
u = u ^ { + } - u ^ { - } , \qquad | u | = u ^ { + } + u ^ { - } , \qquad u ^ { + } u ^ { - } = 0 .\tag{2}
$$

For an integrable function $z : \mathcal { T }  \mathbb { R }$ , the signed measure $\textstyle \nu ( A ) = \int _ { A } z ( t ) \mathrm { d } \mu ( t )$ admits the Jordan decomposition $\nu = \nu ^ { + } - \nu ^ { - }$ , where $\mathrm { d } \nu ^ { + } = z ^ { + } \mathrm { d } \mu$ and $\mathrm { d } \nu ^ { - } = z ^ { - } \mathrm { d } \mu$ . DECAF applies this decomposition to the endpoint-oriented response on the active branch and reserves a separate branch for endpoint-null pairs.

## B.2 CANONICALITY

ProofofTheorem 1. Fix $a , s ,$ and r. If $a = 0$ , endpoint gating requires $e = c = 0$ . Conservation then forces $f = | r |$ . Hence the triple is unique.

Now suppose $a = 1$ . Endpoint gating gives $f = 0 . { \mathrm { L e t } } z = s r . { \mathrm { I f } } z \geq 0$ , directional support requires $c = 0$ and conservation gives $e = | r | = z = z ^ { + }$ $\mathrm { I f } ~ z \le 0$ , directional support requires $e = 0$ and conservation gives $c = | r | = - z = z ^ { - }$ . Thus

$$
( e , c , f ) = \big ( a ( s r ) ^ { + } , a ( s r ) ^ { - } , ( 1 - a ) | r | \big ) .\tag{3}
$$

The construction is nonnegative, conserves $| r | ,$ , obeys the gate, and has the stated support. Therefore it is the unique triple satisfying the axioms. □

The theorem also yields a projection interpretation. On an active pair, e is the magnitude of the Euclidean projection of r onto the endpoint-aligned ray $\{ s \lambda : \lambda \ge 0 \}$ , while c is the magnitude of the projection onto the opposite ray. The null branch is not a projection. It records that the endpoint does not provide a stable orientation at the chosen threshold.

## B.3 MAGNITUDE NON-IDENTIFIABILITY AND STRICT REFINEMENT

ProofofTheorem 2. Fix $m > 0$ . Consider three responses. First, let the endpoint be active with $s = 1$ and let $r = m$ . Then $( e , c , f ) = ( m , 0 , 0 )$ . Second, keep the same endpoint and let $r = - m$ Then $( e , c , f ) = ( 0 , m , 0 )$ . Third, let the endpoint be null and let $r = m$ . Then $( e , c , f ) = ( 0 , 0 , m )$ In every case, the observed magnitude is $| r | = m$ . A statistic that observes only |r| therefore cannot distinguish the three mechanisms. □

Corollary 1 (Strict response refinement). The DECAF profile is at least as informative as ordinary magnitude for every decision problem, because ${ \mathrm { A b s } } = { \bar { E } } + { \cal C } + { \cal F }$ is a deterministicfunction ofthe profile. It is strictly more informative for any mechanism-identification problem that assigns positive probability to two distinct profiles with the same magnitude.

Proof. Any rule that uses Abs can be composed with the map $( E , C , F ) \mapsto E + C + F$ . Theorem 2 gives distinct profiles that share one magnitude. A decision problem that rewards correct mechanism identification separates those profiles, so no magnitude-only rule can match the best rule that observes the full profile. This is the standard logic of strict Blackwell refinement (Blackwell, 1953). □

A useful finite case makes the gap concrete. Suppose evidence, contradiction, and fragility are equally likely and all produce magnitude $m .$ . Every magnitude-only classifier receives the same observation, so its best accuracy is $1 / 3$ . Under unequal priors, the best magnitude-only accuracy is the largest prior. The DECAF profile identifies the mechanism exactly in this idealized construction.

## B.4 PRESERVATION, ATTENUATION, AND INVERSION

Proof of Proposition 1. Let the endpoint orientation be s. Under preservation, the stage response is sm, so $E = m , C = 0$ , and $\mathrm { A b s } = m$

Under attenuation, the response is sm with probability $1 - \eta$ and zero with probability $\eta .$ Taking expectations gives $E = ( 1 - \eta ) m , C = 0$ , and $\mathrm { A b s } = ( 1 - \eta ) m$

Under inversion, the response is sm with probability $1 - \eta$ and $- s m$ with probability η. The first branch contributes m to evidence, while the second contributes m to contradiction. Therefore $E = ( 1 - \eta ) m , C = \eta m$ , and $\mathrm { A b s } = m$ . The ratio is $C / ( E + C ) = \eta$ □

The equal-magnitude assumption isolates the inversion probability. If the aligned and opposed branches have magnitudes $m _ { + }$ and $m _ { - }$ , then

$$
E = ( 1 - \eta ) m _ { + } , \qquad C = \eta m _ { - } , \qquad \frac { C } { E + C } = \frac { \eta m _ { - } } { ( 1 - \eta ) m _ { + } + \eta m _ { - } } .\tag{4}
$$

Thus the opposed fraction measures probability mass only when the two branches have comparable effect size. In general, $C$ measures opposed causal mass.

## B.5 POPULATION FACTORIZATION

Let $\pi = \operatorname* { P r } ( | d | \geq \varepsilon )$ . Define branch-conditional summaries whenever the corresponding branch has positive probability. Since evidence and contradiction vanish on the null branch, while fragility vanishes on the active branch,

$$
E = \pi E ^ { \mathrm { a c t i v e } } , \qquad C = \pi C ^ { \mathrm { a c t i v e } } , \qquad F = ( 1 - \pi ) F ^ { \mathrm { n u l l } } .\tag{5}
$$

Unconditional quantities combine branch prevalence and conditional intensity. Conditional quantities answer a different question and become unstable when their branch is rare. The factorization is therefore useful when an external outcome is defined only within the active or null branch, but the primary experiments report unconditional population summaries.

## B.6 INVARIANCES

Endpoint swap. Swapping $\mathbf { x } ^ { + }$ and $\mathbf { x } ^ { - }$ multiplies both d and $r ( t ) \ \mathsf { b y - 1 }$ . The oriented response $s r ( t )$ is unchanged. Hence ${ \check { M } } , E , C , F ,$ and Abs are invariant.

Score translation. Replacing q by $q + b$ leaves every difference unchanged.

Positive affine score reparameterization. Let $q ^ { \prime } = \lambda q + b$ with $\lambda > 0$ . Then $d ^ { \prime } = \lambda d , r ^ { \prime } ( t ) =$ $\lambda r ( t )$ , and $s ^ { \prime } = s$ . If the endpoint threshold is transformed consistently as $\varepsilon ^ { \prime } = \lambda \varepsilon$ , the gate is unchanged and

$$
( M ^ { \prime } , E ^ { \prime } , C ^ { \prime } , F ^ { \prime } , \mathrm { A b s } ^ { \prime } ) = \lambda ( M , E , C , F , \mathrm { A b s } ) .
$$

Hence all normalized component proportions are invariant.

With a numerically fixed threshold ε, however, the gate obeys

$$
a _ { \varepsilon } ^ { \prime } = { \bf 1 } \{ | d | \geq \varepsilon / \lambda \} .
$$

Consequently,

$$
E _ { \varepsilon } ^ { \prime } = \lambda E _ { \varepsilon / \lambda } , \qquad C _ { \varepsilon } ^ { \prime } = \lambda C _ { \varepsilon / \lambda } , \qquad F _ { \varepsilon } ^ { \prime } = \lambda F _ { \varepsilon / \lambda } .
$$

Thus positive scaling preserves component proportions only when gate membership is preserved.   
Otherwise, the discrepancy is controlled by the threshold-mass bound in Appendix B.8.

Negative score scaling. A negative scaling reverses the semantic meaning of the target score. The endpoint orientation still makes the component magnitudes invariant after multiplying by |λ|, but the analyst has changed the behavior being explained. We therefore define the score direction before analysis.

## B.7 PROTOCOL REPARAMETERIZATION

Let $h : { \tilde { \mathcal { T } } }  \mathcal { T }$ be an increasing bijection and define the reparameterized path $\widetilde r ( u ) = r ( h ( u ) )$ Pointwise components are reindexed:

$$
\widetilde { e } ( u ) = e ( h ( u ) ) , \quad \widetilde { c } ( u ) = c ( h ( u ) ) , \quad \widetilde { f } ( u ) = f ( h ( u ) ) .\tag{6}
$$

If the integration measure is pushed forward consistently, $\widetilde { \mu } = h _ { \# } ^ { - 1 } \mu ,$ , then the integrated profile is invariant. If both paths are instead integrated with a uniform coordinate measure, their $\mathbf { A U C s }$ may differ. Dynamic DECAF summaries are therefore protocol-relative, and every experiment reports the path and stage measure.

## B.8 THRESHOLD STABILITY

Let $0 \leq \varepsilon _ { 1 } < \varepsilon _ { 2 }$ and suppose the per-example path summary is bounded by B. The two gates differ only on pairs satisfying $\varepsilon _ { 1 } \leq | d | < \varepsilon _ { 2 }$ . Consequently, for each component $G \in \{ E , C , \bar { F } \}$ ,

$$
\vert G _ { \varepsilon _ { 2 } } - G _ { \varepsilon _ { 1 } } \vert \le B \operatorname* { P r } ( \varepsilon _ { 1 } \le \vert d \vert < \varepsilon _ { 2 } ) .\tag{7}
$$

The threshold is stable when little endpoint mass lies near the boundary. We report sensitivity over multiple thresholds rather than treating one numerical cutoff as universal.

## C ESTIMATORS, COMPLEXITY, AND IMPLEMENTATION

DECAF reuses the signed scores of a paired perturbation curve. Once those scores are available, the decomposition is elementwise.

## C.1 FINITE-GRID ESTIMATORS

Let $t _ { 1 } , \dots , t _ { T }$ be stage points with nonnegative quadrature weights $w _ { 1 } , \ldots ,$ w that sum to one. For pair i, define

$$
E _ { i } = \sum _ { j = 1 } ^ { T } w _ { j } e _ { i } ( t _ { j } ) , \quad C _ { i } = \sum _ { j = 1 } ^ { T } w _ { j } c _ { i } ( t _ { j } ) , \quad F _ { i } = \sum _ { j = 1 } ^ { T } w _ { j } f _ { i } ( t _ { j } ) .\tag{8}
$$

For N pairs, the empirical estimates are

$$
\widehat { E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } E _ { i } , \quad \widehat { C } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } C _ { i } , \quad \widehat { F } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } F _ { i } , \quad \widehat { M } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | d _ { i } | .\tag{9}
$$

Every implementation checks the exact sample-level identity $| r _ { i } ( t _ { j } ) | = e _ { i } ( t _ { j } ) + c _ { i } ( t _ { j } ) + f _ { i } ( t _ { j } )$ We use trapezoidal weights on ordered grids unless stated otherwise. When the protocol contains randomness, factual and counterfactual branches share that randomness; repeated paths are averaged or retained as clusters for uncertainty estimation.

## C.2 QUERY COMPLEXITY

Suppose we evaluate K factors, T stages, J counterfactual maps per factor, and R protocol repetitions. If factual branch evaluations are shared across factors, the number of model evaluations per example is

$$
R T ( 1 + K J ) .\tag{10}
$$

For one factor and one map this is 2RT. DECAF adds no model evaluations to the ordinary signed paired trajectory; its arithmetic cost is linear in the number of scalar responses.

The ImageNet-9 main path uses T = 9, so it requires 18 forward evaluations per pair. Under our configurations, Integrated Gradients and SmoothGrad each use 16 forward–backward evaluations, BlurIG uses 12, Occlusion uses 49 forward evaluations, and RISE uses 256. These counts describe access and query structure rather than hardware-independent wall time.

## C.3 MEMORY, BATCHING, AND BLACK-BOX MODELS

A streaming implementation stores the endpoint sign, current stage scores, and three accumulators. Its additional memory is O(BK) scalars for batch size B and K factors, beyond the model and input batch. Examples, branches, stages, factors, counterfactual maps, repetitions, and models provide independent batching dimensions.

DECAF requires no differentiability. It applies to neural networks, tree ensembles, simulators, and remote APIs whenever a stable real-valued score is available. For stochastic models, branches should share random seeds or use repeated queries. Hard labels are formally sufficient, but probabilities, logits, margins, regression values, or action values produce more informative endpoint effects.

## D CONTROLLED EXPERIMENTAL SETUP

The controlled suite gives DECAF an identifiable target before we move to natural images. Every experiment uses exact factor interventions in 3D Shapes, shared protocol randomness across branches, and two architectures. The experiments differ in the behavior they are designed to activate.

## D.1 DATASET AND FACTORS

3D Shapes contains 480,000 rendered scenes generated from six factors: floor color, wall color, object color, object size, object shape, and object orientation (Burgess et al., 2018). The Cartesian product is fully enumerated. This structure gives us exact counterfactuals: to intervene on one factor, we change its index and hold the other five indices fixed.

We use binary prediction tasks derived from the factors. The base benchmark contains five tasks:

• Object color: a binary split of the object-color values.

• Wall color: a binary split of the wall-color values.

• Object shape: the two shapes that are common to all model-response supports.

• Color–shape XOR: the exclusive-or of binary object color and shape.

• Context gate: a color decision whose active factor depends on wall context.

For each task we train ResNet-18 and a small ViT with three random seeds. The resulting 30 models are crossed with all six factors, producing 180 model–factor units.

## D.2 CLEAN ENDPOINT PAIRS

For a factual scene $\mathbf { x } ^ { + }$ and target factor $k ,$ the counterfactual $\mathbf { x } ^ { - }$ changes only factor k. Binary factors are flipped. Multivalued color, size, and orientation factors use fixed involutive maps so that applying the map twice returns the original value. We use two independent maps for multivalued factors and average their measurements.

The clean endpoint effect is computed on 8,192 held-out factual–counterfactual pairs. The main threshold classifies a pair as active when the absolute target-score difference exceeds the experiment specific ε. The endpoint classification is a property of the model and pair, not of the task label.

## D.3 PRIMARY REVEAL PROTOCOL

The main controlled protocol is covariance-matched Gaussian reveal. Let $\pmb { \mu }$ and $\hat { \Sigma }$ be the empirical mean and covariance of the image distribution. For $\alpha \in [ 0 , 1 ]$

$$
{ \bf x } _ { \alpha } = { \pmb { \mu } } + \sqrt { \alpha } ( { \bf x } - { \pmb { \mu } } ) + \sqrt { 1 - \alpha } { \pmb { \eta } } , \qquad { \pmb { \eta } } \sim \mathcal { N } ( \mathbf { 0 } , { \widehat { \Sigma } } ) .\tag{11}
$$

The factual and counterfactual branch share the same η. $\mathrm { A t } \alpha = 1$ the path reaches the clean input; at $\alpha = 0$ the state is independent of the individual sample under the fitted Gaussian reference. The covariance match makes the stimulus second-order neutral in whitened data coordinates. DECAF does not depend on this particular path; it only consumes the paired scores that the path produces.

The base benchmark uses 4,096 dynamic factual pairs, three noise seeds, two counterfactual maps, and 21 points along a trace-matched covariance family. The primary CMMR endpoint is accompanied by a trace-matched pixel-Gaussian endpoint. A diagonal covariance and a trace-normalized covariancepower path provide held-out geometries.

## D.4 ALTERNATIVE PROTOCOLS

The covariance interpolation is

$$
\mathbf { { \Gamma } } \mathbf { { \Gamma } } \mathbf { { \Gamma } } \mathbf { { \Gamma } } \mathbf { { \Gamma } } \mathbf { { \Gamma } } \mathbf { { \Gamma } } = ( 1 - \lambda ) \widehat { \pmb { \Sigma } } + \lambda \tau \mathbf { { I } } , \qquad \tau = \frac { \mathrm { t r } ( \widehat { \pmb { \Sigma } } ) } { D } ,
$$

where $D$ is the pixel dimension. The two endpoints use data covariance and trace-matched isotropic pixel covariance. The diagonal held-out path uses dia $\mathrm { g } ( \widehat { \pmb { \Sigma } } )$ . The power path raises covariance eigenvalues to a power and renormalizes the trace. These paths test whether a semantic conclusion depends on one second-order geometry.

We use the protocol family as an audit, not as a learned worst-case score. The main paper reports CMMR-based DECAF and uses the alternatives to characterize transfer.

## D.5 EVIDENCE VALIDATION THROUGH TRAINING TRAJECTORIES

The evidence experiment trains shape classifiers in an environment where wall color is strongly correlated with the label. We use two training correlations, 0.95 and 0.99, two architectures, two seeds, and save checkpoints throughout training. This creates eight trajectories and 52 selected checkpoints.

The external target is shortcut-reversal vulnerability,

$$
V _ { \mathrm { r e v } } = \mathrm { A c c } ( p _ { \mathrm { t e s t } } = 0 . 9 5 ) - \mathrm { A c c } ( p _ { \mathrm { t e s t } } = 0 . 0 5 ) .
$$

A shortcut-dominant checkpoint performs well when wall color remains aligned and fails when the correlation reverses. $\mathbf { A }$ shape-dominant checkpoint is stable. We compare $V _ { \mathrm { r e v } }$ with the evidence margin $E _ { \mathrm { w a l l } } - E _ { \mathrm { s h a p e } }$

## D.6 FRAGILITY VALIDATION

The fragility experiment uses a binary object-shape task and treats floor color as the candidate endpoint-null factor. We train robust, neutral, and fragile variants for each architecture and three seeds, producing 18 models.

The neutral variant uses clean training only. The robust variant is encouraged to keep its output stable under floor-color counterfactuals at intermediate states. The fragile variant is encouraged to respond to floor color away from the endpoint while preserving the clean shape task. The intended comparison is restricted to models for which clean accuracy remains high and floor color remains endpoint-null.

The behavioral target is the prediction-change rate under held-out floor-color interventions at intermediate states. We also measure randomized-floor stability and transfer across covariance geometries.

## D.7 CONTRADICTION VALIDATION

The contradiction benchmark uses two binary variables: object color A and wall context G. The factual endpoint is evaluated in the context $G = 1$ , where all three tasks share the clean rule $Y = A$ Under the swapped context $G = 0$

$$
{ \mathrm { D i r e c t : ~ } } Y = A , \qquad { \mathrm { G a t e : ~ } } Y = H , \qquad { \mathrm { I n v e r t : ~ } } Y = 1 - A ,
$$

where H is object shape. Direct preserves the object-color effect. Gate removes it. Invert reverses it.

We train 30 models: three tasks, two architectures, and five seeds. Balanced validation accuracy is at least 0.9987. A binary symmetric context channel swaps the wall context with probability $\eta \in [ 0 , 0 . 5 ]$ All intermediate images remain in the support of the generated dataset. Two independent wall-color involutions test map transfer.

The label-level outcomes are preserve, collapse, and swap. The key contrast is between Gate and Invert: both reduce aligned evidence, but only Invert should create endpoint-opposed mass.

## D.8 STATISTICAL SUMMARIES

The base benchmark uses stratified bootstrap over task, architecture, and seed. The evidence experi ment bootstraps complete trajectories. The fragility and contradiction experiments report seed-level variation and architecture-stratified summaries. Bootstrap intervals use 500 repetitions unless stated otherwise. We treat the model or training trajectory as the main statistical unit rather than counting every image as independent evidence.

## E SELECTED CONTROLLED RESULTS

This appendix expands the behavioral validation in Section 5 without repeating its main tests. We report five positive extensions: a matched-magnitude example, the complete response atlas, all evidence-training trajectories, fragility checks across interventions and reveal geometries, and contradiction checks across regimes, architectures, seeds, and counterfactual maps.

## E.1 MATCHED-MAGNITUDE RESPONSE ROLES

Figure 5 gives a concrete example of the scalar ambiguity discussed in the main text. In an objectshape model, the true shape factor and an endpoint-null floor-color factor produce nearly equal ordinary response magnitude, but their response roles differ sharply. The same distinction matters across the controlled benchmark: ordinary magnitude can rank an endpoint-null factor above a supported factor, whereas endpoint-oriented routing removes these false-null reversals.

## E.2 COMPLETE DECOMPOSITION ATLAS

Figure 6 displays every task–factor combination. Endpoint-null factors concentrate near the fragility corner, direct color tasks concentrate in evidence, and interaction tasks carry more contradiction and mixed mass. Across 125 endpoint-null units, 96.8% of ordinary response is fragility on average. Across 48 endpoint-supported units, the mean composition is 64.1% evidence, 12.5% contradiction, and 23.4% null fragility.

## E.3 EVIDENCE ACROSS TRAINING TRAJECTORIES

Figure 7 places exact shortcut vulnerability and the DECAF evidence margin on the same eight checkpoint trajectories. Six trajectories exhibit nonconstant vulnerability, and all six show positive

![](images/d175ab077f6acefd35ccc14641e7f5a8bef552ee2e0c11c7c777b080f2074737.jpg)  
(a) Nearly equal magnitude, different response roles.

![](images/48837b8660d4be48152c2373d8cb0e4f213977786dbb4c8185b7c1a1a0f641fc.jpg)  
(b) Orientation removes false-null reversals.

Figure 5: Equal response magnitude can hide different behavior. (a) The true shape factor and an endpointnull floor-color factor have nearly identical ordinary magnitude, yet DECAF separates evidence from fragility. (b) Across the benchmark, ordinary magnitude ranks an endpoint-null factor above a supported factor in 7.32% of comparable pairs, whereas endpoint-oriented routing reduces this rate to zero.  
![](images/6659acd272041d8ec1d1cda8ef1c1088fb63d0c02a46adb7de1949e0042a19ec.jpg)  
Evidence � Contradiction � Fragility �  
Figure 6: Complete controlled decomposition atlas. Results cover five tasks, two architectures, three seeds, and six factors. Endpoint-null factors are dominated by fragility, while task-relevant and interaction factors contain evidence and contradiction in different proportions.

within-trajectory association between evidence margin and shortcut vulnerability. The two Small-ViT trajectories trained at the strongest shortcut correlation remain shortcut-dominant, and DECAF correspondingly keeps their evidence margin high.

## E.4 FRAGILITY: INTERVENTION SEPARATION AND REVEAL GEOMETRY

The main text validates F against independently measured off-path prediction change. Figure 8 provides three complementary checks. First, F cleanly orders robust, neutral, and fragile interventions. Second, similar total response can correspond to different endpoint-relative roles across architectures. Third, the robust–neutral–fragile ordering transfers from the primary covariance-matched reveal to diagonal covariance, trace-matched pixel Gaussian, and three covariance-power geometries. The controlled fragility result is therefore not tied to one second-order reveal geometry, although the numerical summaries remain protocol-relative.

![](images/7a2acc2901c5dfa27863b3724c38ab9adf52a04c0462c045eb03f8c42c58e9cd.jpg)  
(a) Exact shortcut vulnerability.

![](images/edafb5e5ed89c1d75365695e616c4d4f5e72114f6031baa22f3d659b431fadd2.jpg)  
(b) Evidence margin.  
Figure 7: Evidence follows shortcut reliance across complete training trajectories. The two panels show the same eight checkpoint sequences through an external behavioral measure and the DECAF evidence margin.

![](images/8c1f8bafcd8bebb1573f00e7ac62008dba856356335c7d07487022c8033a8d86.jpg)  
(a) Intervention separation.

![](images/ba76101a021136dd9cdc96cddbbc62b28fecc574fd893c152103882d2bcf324d.jpg)  
(b) Similar magnitude, different roles.

![](images/fc4211417ed448807f155d9bab4716ccdf24a9871c368c0afa92b81b918d10bd.jpg)  
(c) Cross-geometry rank transfer.

Figure 8: Additional fragility checks. (a) F separates robust, neutral, and fragile training. (b) Similar ordinary response can be endpoint evidence in one model and endpoint-null fragility in another. (c) The controlled ordering transfers across held-out reveal geometries.  
![](images/1bd11e87ddb44a4060e848093e1b1c4f90d1a60eace5ce1e452afccaf55c6900.jpg)  
(a) Attenuation vs. inversion.

![](images/9d9f02a5324acac55c6b6e48e316cb81965107e1c7edc56defa0ea32e7788a11.jpg)  
(b) Independent label behavior.

![](images/3619757c1201bea1228ab77cd721ad9037f9aa2a4489c47741729e10cdc14c80.jpg)  
(c) C succeeds where Abs fails.

![](images/952eeb0693409c13c3667b6b538f38bf2745c3cc2cfa561a45ce70bab6e330e9.jpg)  
(d) Opposed-fraction calibration.

![](images/587dc9cb6a37bfaa4ca0f60ae0e18bdb26b5ee0a9dc6c359a032000e54559d11.jpg)  
(e) Architecture and seed consistency.

![](images/bb045646c30cf2f1a678fc26e5d313dded4ba1d023c68c25b6e1c1189d0740d5.jpg)  
(f) Counterfactual-map transfer.  
Figure 9: Additional contradiction checks. Top: contradiction separates attenuation from inversion, agrees with independently measured label behavior, and succeeds where ordinary magnitude does not. Bottom: the opposed fraction is calibrated and stable across architectures, seeds, and alternative wall-color counterfactual maps.

## E.5 CONTRADICTION: REGIME SEPARATION AND CALIBRATION

The main text shows that C tracks independently measured label reversal. Figure 9 expands this result. Direct preserves the effect, Gate suppresses it, and Invert reverses it; only inversion creates substantial opposed mass. The induced label behavior independently separates into preservation, collapse, and swap. Across architectures, seeds, and wall-color maps, the opposed fraction remains well calibrated. Raw sign changes can be frequent when the stage effect is numerically close to zero, but the corresponding opposed mass remains small, separating magnitude-aware contradiction from sign counting.

## F IMAGENET-9 EXPERIMENTAL DETAILS

This appendix gives the construction and secondary diagnostics behind Section 6. The main text keeps only the tests needed for the central argument. Here we specify the data splits, model zoo, paired background interventions, independent behavioral indicators, magnitude-controlled evaluation, spatial-attribution baselines, and cross-architecture comparison.

## F.1 DATA VARIANTS AND DISJOINT SPLITS

We use the official ImageNet-9 Backgrounds Challenge variants (Xiao et al., 2021): Original, Mixed Same, Mixed-Rand, Mixed-Next, Only-FG, and the challenge backgrounds. The variants preserve foreground identity while changing how the background is constructed. We match variants by foreground identifier and retain 4,050 foregrounds for which the required variants are available.

A deterministic split keeps the roles of the data separate. We reserve 1,644 paired foregrounds for the broad model-level response scan and an 820-foreground deep pool for the sample-level benchmark; 768 foregrounds from this pool are used for the expensive baseline comparison. The remaining foregrounds are not used in that sample-level benchmark. No foreground appears in more than one split.

## F.2 MODEL ZOO AND COMMON SCORE SPACE

The 72-model zoo contains two complementary groups. The first consists of 24 off-the-shelf ImageNet-1k classifiers from torchvision and timm. These models were trained independently of our audit and span ResNet, ResNeXt, WideResNet, DenseNet, EfficientNet, RegNet, ConvNeXt, DeiT, ViT, BEiT, Swin, MaxViT, and CoAtNet architectures.

The second group contains 48 models trained directly on ImageNet-9. We fine-tune six backbones— ResNet-50, ConvNeXt-Tiny, EfficientNet-B3, RegNetY-8GF, Swin-T, and ViT-B/16—on four versions of the training data: Original, Mixed-Rand, Mixed-Next, and Only-FG, using two random seeds. This $6 \times 4 \times 2$ design broadens the range of learned background dependence while retaining the same ImageNet-9 prediction task.

We train the fine-tuned models for eight epochs with AdamW, cosine decay, one warm-up epoch, BF16 computation, random resized crops, and horizontal flips. ResNet-50, EfficientNet-B3, and RegNetY-8GF use batch size 256 and learning rate $3 \times 1 0 ^ { - 4 }$ . ConvNeXt-Tiny, Swin-T, and ViT-B/16 use batch size 128 and learning rate $1 0 ^ { - 4 }$

All models are evaluated in the same nine-class score space. For an ImageNet-1k model, we compute its 1,000-way softmax and sum the probabilities belonging to each official ImageNet-9 superclass. We do not sum logits or apply a second softmax. The ImageNet-9 fine-tuned models output nine logits directly. In both groups, the scalar analyzed by DECAF is the probability assigned to the true ImageNet-9 superclass.

## F.3 PAIRED BACKGROUND INTERVENTIONS AND REVEAL PATHS

For each foreground, we construct two paired background interventions. Same–Rand compares Mixed-Same with Mixed-Rand: the foreground is unchanged, while a same-class background is replaced by a random-class background. This pair is used for the evidence and fragility analyses. Same–Next compares Mixed-Same with Mixed-Next, where the replacement background comes from superclass $( y + 1 )$ mod 9. This pair creates a directional background change for the contradiction analysis.

The primary reveal begins from a shared blurred midpoint. For endpoints $\mathbf { x } ^ { + }$ and $\mathbf { x } ^ { - }$ ,

$$
\mathbf { x } ^ { 0 } = \operatorname { B l u r } \left( { \frac { \mathbf { x } ^ { + } + \mathbf { x } ^ { - } } { 2 } } \right) ,\tag{12}
$$

and each branch is linearly revealed from $\mathbf { x } ^ { 0 }$ to its endpoint. We evaluate nine stages, $t \in$ $\{ 0 , 0 . 1 2 5 , \ldots , 1 \}$ . The main endpoint threshold is $\varepsilon = 0 . 0 2$ in true-class probability.

To test whether conclusions depend on the reveal path, we also use nested-patch reveal on an $8 \times 8$ grid. Patches are ordered by endpoint-difference energy, with a fixed random tie-break. Two independently tie-broken orders are evaluated. Both branches use the same patch order and the same neutral starting image. Appendix G reports the corresponding path-order and threshold checks.

## F.4 BEHAVIORAL INDICATORS DEFINED WITHOUT DECAF

The sample-level benchmark defines three observable behavioral indicators directly from model predictions. None uses E, C, or F, and the indicators are allowed to overlap.

Evidence behavior. Set $Y _ { E } = 1$ when Mixed-Same is classified correctly and replacing its background with Mixed-Rand either changes the predicted class or lowers the true-class probability by at least 0.20. This records a consequential change caused by replacing the background.

Contradiction behavior. Set $Y _ { C } = 1$ when changing only the background makes the model switch from the foreground class to the class associated with the new Mixed-Next background. This is an independently observed directional reversal in the model’s prediction.

Fragility behavior. We first restrict attention to Same–Rand pairs whose fully revealed endpoints produce little difference, $| d | < 0 . 0 2$ . The endpoint pair alone cannot tell us whether such a case is otherwise sensitive, so we test it separately with additional background changes that are not part of the DECAF reveal. These changes use Gaussian blur, Gaussian noise, pixelation, color shift, and patch shuffle, each at two severities. Set $Y _ { F } = 1$ if any of them changes the predicted class or shifts the true-class probability by at least 0.20.

These indicators are external behavioral checks, not definitions of the DECAF components. Their purpose is to ask whether the response roles obtained from the paired trajectory agree with behavior measured independently of that trajectory decomposition.

## F.5 MAGNITUDE-CONTROLLED EVALUATION

The deep benchmark uses 32 models and 768 foregrounds, giving 24,576 model–image units. We use two complementary ways to control ordinary response magnitude.

First, we divide the units into 20 quantile bins of Abs. Within a bin, a behavioral AUROC is computed only when both positive and negative examples are present for that indicator. The main text reports both the valid-bin summary and the stricter common-support comparison in which all three indicators are represented.

Second, we construct one-to-one matched comparisons between cases with different behavioral indicator patterns but nearly identical ordinary response magnitudes. We require the relative difference in Abs to be at most 5%, yielding 8,289 matched comparisons.

Because $Y _ { E } , Y _ { C } ,$ , and $Y _ { F }$ may overlap, the matched evaluation does not force every case into one ground-truth class. For a case with at least one active indicator, we examine the largest score among the three response-role coordinates. The case receives full credit when the largest coordinate corresponds to an active behavioral indicator. If several coordinates tie for the maximum, credit is divided across the tied coordinates. The reported matched-pair accuracy averages this role-agreement score over the matched cases. Ordinary magnitude has only one scalar value and therefore cannot prefer evidence, contradiction, or fragility.

The implementation also records a separate pairwise-ranking diagnostic: when an independent behavioral indicator differs across a matched pair, it asks whether the corresponding response-role score changes in the same direction. This ranking diagnostic is distinct from the role-agreement accuracy reported in Section 6.2.

![](images/bbcb88774b67bed5c43656ae4b7eeec0a34eb0a38952add73b916cca5e4e8d64.jpg)  
Figure 10: Cross-architecture transfer and spatial complementarity. All learned results hold out one architecture family during fitting. Response features from DECAF nearly match the supervised responsestatistics reference, while adding spatial attribution provides a further gain and approaches the full supervised reference.

## F.6 SPATIAL-ATTRIBUTION BASELINES

The deep benchmark includes six standard spatial-attribution methods: Input×Gradient, Integrated Gradients with 16 steps, SmoothGrad with 16 noise samples, BlurIG with 12 blur levels, a $7 \times 7$ Occlusion grid, and RISE with 256 masks. For each method, we sum absolute attribution inside the foreground–background difference region. These scores describe where the prediction is sensitive; they are not used to define the three behavioral indicators above.

## F.7 CROSS-ARCHITECTURE COMPARISON AND SPATIAL COMPLEMENTARITY

For the learned comparison, every representation is evaluated with the same supervised protocol: train on all but one architecture family and evaluate on the held-out family. The DECAF representation uses $( M , E , C , F )$ . A response-statistics reference uses (M, Abs, Net, SignFlip, active rate). The spatial representation uses the six attribution scores above. We also evaluate DECAF combined with the spatial scores and a full reference combining response statistics with spatial attribution.

Figure 10 contains the panel moved from the main text. Using response information alone, DECAF reaches leave-one-architecture-family-out macro-AUROC 0.862, close to the supervised responsestatistics reference at 0.867. Spatial attribution alone reaches 0.737. Combining DECAF with spatial attribution reaches 0.918, close to the full supervised reference at 0.921. This is best read as a complementarity check: the response decomposition and spatial attribution retain different information.

## F.8 STATISTICAL REPORTING

For direct behavioral comparisons, we report AUROC and AUPRC. For learned comparisons, we report leave-one-architecture-family-out results so that the evaluated architecture family is not used to fit the probe. Undefined AUROCs remain missing rather than being replaced by chance. Where bootstrap intervals are reported, models—rather than individual model–image rows—are the resampling unit.

## G IMAGENET-9 PROTOCOL ROBUSTNESS CHECKS

The protocol audit in Section 6 changes the reveal path while keeping the factual–counterfactual endpoints fixed. We retain two checks that test whether the reported effect is caused by an arbitrary implementation choice.

First, the nested-patch reveal is evaluated with two independently tie-broken patch orders. Their response summaries agree almost perfectly, showing that the blend-to-patch contrast is not an accident of one patch sequence. Second, we recompute the decomposition at $\varepsilon \in \{ 0 . 0 1 , 0 . 0 2 , 0 . 0 5 \}$ . Changing

![](images/bed27093e2142d07dbb88e4657d4a3f173eba1b75e78e2b2f005a05a5a0459cc.jpg)  
(a) Patch-order stability.

![](images/31d38b26034662e503c3eb6dc726a63e6dbfb91f80a87440df2a61b5424343ae.jpg)  
(b) Endpoint-threshold sensitivity.  
Figure 11: Robustness of the ImageNet-9 protocol audit. (a) The two independently tie-broken nested-patch orders agree almost perfectly for every response component. (b) Varying the endpoint threshold changes the allocation between active and endpoint-null response, but not the qualitative protocol conclusion.

the threshold reallocates response between the active and endpoint-null branches as expected, but preserves the qualitative conclusion that patch reveal increases total response without increasing evidence.

## H FORWARD-ONLY ATTRIBUTION: COMPLETE RESULTS AND BOUNDARIES

This appendix expands the compact comparison in Section 7. It records the strict common-support protocol, complete baseline results, an official-protocol FunnyBirds reproduction check, the endpointversus-trajectory ablation, full-scale ImageNet validation, measured compute, large-model scaling, and the PartImageNet boundary case.

## H.1 PROTOCOL AND STRICT COMMON SUPPORT

FunnyBirds and ImageNet-1k use the same three architectures: ResNet-50, VGG-16, and ViT-B/16. FunnyBirds evaluates semantic-part rankings against two held-out operators, Telea inpainting and background-texture replacement. ImageNet-1k IDSDS partitions every image into a fixed $\bar { 4 } \times 4$ grid and evaluates the ranking of 16 patch scores against in-domain single-deletion effects. We first filter to images classified correctly by each model. We then intersect image IDs across the methods included in the strict comparison. The resulting support contains 499, 497, and 488 FunnyBirds images, and 7,663, 7,189, and 8,285 ImageNet images, respectively.

For each image, the primary metric is Spearman correlation between feature attribution and the benchmark target. We average within each model and then macro-average equally across architectures. All intervals use 1,000 paired image-cluster bootstrap replicates. We never average the FunnyBirds and ImageNet columns into one leaderboard.

Endpoint $M _ { k } = | d _ { k } |$ is an endpoint-only reference. On ImageNet IDSDS, the endpoint audit gives max $_ k \left| d _ { k } - g _ { k } \right| = 0$ and confirms that every stored endpoint effect uses the same deletion pair as the evaluation target. The direct target-derived $\dot { \left| g _ { k } \right| }$ and independently persisted DECAF $M _ { k }$ differ only by a mean numerical discrepancy of $6 . 0 \times 1 0 ^ { - 6 }$

## H.2 COMPLETE CROSS-DATASET METHOD COMPARISON

Published-benchmark sanity check. On the original full-scale IDSDS ResNet-50 setting, our IG and IG-U scores are 0.194 and 0.252, closely matching the published values of 0.196 and 0.255 (Hesse et al., 2024). This agreement provides an external check on our ImageNet implementation. Our FunnyBirds evaluation instead uses held-out intervention operators and strict common support, so it is not intended to reproduce the dataset’s native part-based evaluation protocol (Hesse et al., 2023). We therefore compare its baseline values only within our registered protocol.

The stronger IG baseline changes across datasets. IG-32 exceeds IG-U-32 on FunnyBirds, while IG-U-32 is stronger on IDSDS. DECAF-5 exceeds both variants on both datasets. KernelSHAP exhibits the opposite transfer pattern: it is strongest on IDSDS, whose target is the deletion game it queries, but is weaker than endpoint M and DECAF under the held-out FunnyBirds operators.

Table 6: Complete strict-common-support attribution results. Entries are equal-architecture macro-average Spearman correlations with 95% paired image-cluster bootstrap intervals. Endpoint M is an endpoint-only reference. KernelSHAP-512 is a deletion-game reference.
<table><tr><td>Method</td><td>Access</td><td>FunnyBirds ρ [95% CI]</td><td>ImageNet-1k IDSDS ρ [95% CI]</td></tr><tr><td colspan="4">DECAF</td></tr><tr><td>DECAF-3</td><td>Forward only</td><td>0.372 [0.353, 0.392]</td><td>0.359 [0.355, 0.364]</td></tr><tr><td>DECAF-5</td><td>Forward only</td><td>0.403 [0.385, 0.422]</td><td>0.367 [0.362, 0.371]</td></tr><tr><td>DECAF-9</td><td>Forward only</td><td>0.406 [0.388, 0.425]</td><td>0.379 [0.375, 0.383]</td></tr><tr><td colspan="4">General-purpose attribution</td></tr><tr><td>Input×Gradient</td><td>Backward</td><td>0.019 [-0.002, 0.042]</td><td>0.098 [0.094, 0.101]</td></tr><tr><td>IG-16</td><td>Backward</td><td>0.266 [0.246, 0.285]</td><td>0.238 [0.235, 0.242]</td></tr><tr><td>IG-32</td><td>Backward</td><td>0.271 [0.251, 0.290]</td><td>0.242 [0.238, 0.245]</td></tr><tr><td>IG-U-32</td><td>Backward</td><td>0.200 [0.178, 0.222]</td><td>0.295 [0.292, 0.299]</td></tr><tr><td>DeepLIFT</td><td>Backward</td><td>0.197 [0.176, 0.217]</td><td>0.341 [0.338, 0.344]</td></tr><tr><td>GradientSHAP</td><td>Backward</td><td>0.226 [0.204, 0.246]</td><td>0.236 [0.233, 0.240]</td></tr><tr><td>SmoothGrad-16</td><td>Backward</td><td>-0.045 [-0.065, -0.023]</td><td>-0.015[-0.018, -0.011]</td></tr><tr><td>RISE-512</td><td>Sampling</td><td>0.302 [0.283, 0.321]</td><td>0.179 [0.175, 0.183]</td></tr><tr><td>RISE-U-512</td><td>Sampling</td><td>0.294 [0.274, 0.315]</td><td>0.111 [0.107, 0.115]</td></tr><tr><td colspan="4">Endpoint-only reference</td></tr><tr><td>Endpoint M</td><td>Endpoint only</td><td>0.324 [0.303, 0.344]</td><td>0.371 [0.366, 0.377]</td></tr><tr><td colspan="4">Deletion-game reference</td></tr><tr><td>KernelSHAP-512 Sampling</td><td></td><td>0.299 [0.278, 0.319]</td><td>0.447 [0.443, 0.451]</td></tr></table>

Table 7: FunnyBirds native Single Deletion reproduction and native-target audit. Top: the official RISE-6000 setting reproduces the published Single Deletion scores within 0.005 on all three architectures. Bottom: raw Spearman correlation with the same native part-removal target. Endpoint M is included to show the strong target alignment of this evaluation. These native-target results are a reproduction and boundary check, not a replacement for the held-out FunnyBirds comparison in Table 3.
<table><tr><td>Architecture</td><td>Published RISE SD</td><td>Reproduced RISE SD</td><td>|∆|</td></tr><tr><td>ResNet-50</td><td>0.560</td><td>0.558</td><td>0.002</td></tr><tr><td>VGG16</td><td>0.730</td><td>0.726</td><td>0.004</td></tr><tr><td>ViT-B/16</td><td>0.790</td><td>0.788</td><td>0.002</td></tr><tr><td></td><td colspan="3">Native-target raw Spearman ρ</td></tr><tr><td>Method</td><td>ResNet-50</td><td>VGG16</td><td>ViT-B/16</td></tr><tr><td>Endpoint M</td><td>0.757</td><td>1.000</td><td>0.983</td></tr><tr><td>DEČAF-3</td><td>0.484</td><td>0.926</td><td>0.909</td></tr><tr><td>DECAF-5</td><td>0.565</td><td>0.898</td><td>0.864</td></tr><tr><td>DECAF-9</td><td>0.616</td><td>0.901</td><td>0.850</td></tr><tr><td>RISE-6000</td><td>0.115</td><td>0.452</td><td>0.576</td></tr></table>

Native-protocol reproduction check. The FunnyBirds column above evaluates attribution transfer to two held-out intervention operators, rather than the dataset’s native Single Deletion protocol. We therefore ran a separate implementation check using the official FunnyBirds setting: the released checkpoints, native semantic part removals, and RISE with 6,000 masks, an 8 × 8 coarse grid, and mask probability $p = 0 . 1$ (Hesse et al., 2023). As shown in Table 7, the reproduced RISE scores differ from the published values by less than 0.005 for every architecture. This check supports the implementation fidelity of the RISE baseline used in our benchmark; it does not imply that the held-out score of 0.302 should numerically match the native Single Deletion scores, because the two evaluations use different intervention targets and RISE configurations.

The native and held-out FunnyBirds experiments answer different questions. The native target is defined by the same semantic part-removal contrast used to construct the endpoint, whereas the main benchmark evaluates transfer to held-out inpainting and texture-replacement interventions. The native audit therefore verifies baseline implementation and exposes the target-aligned boundary; the held-out benchmark remains the test of attribution transfer.

Table 8: Paired endpoint-versus-trajectory tests. Differences are DECAF minus endpoint M. Positive values indicate ordinary-attribution information beyond the endpoint.
<table><tr><td>Dataset</td><td>Contrast</td><td>Mean ∆</td><td>95% CI</td><td>Model wins</td></tr><tr><td>FunnyBirds</td><td>DECAF-3 -M</td><td>+0.0488</td><td>[0.0334, 0.0642]</td><td>2/3</td></tr><tr><td>FunnyBirds</td><td>DECAF-5 -M</td><td>+0.0797</td><td>[0.0637, 0.0962]</td><td>2/3</td></tr><tr><td>FunnyBirds</td><td>DECAF-9 -M</td><td>+0.0828</td><td>[0.0671, 0.0993]</td><td>2/3</td></tr><tr><td>ImageNet-1k</td><td>DECAF-3 -M</td><td>-0.0119</td><td>[-0.0142, -0.0096]</td><td>1/3</td></tr><tr><td>ImageNet-1k</td><td>DECAF-5 -M</td><td>-0.0045</td><td>[-0.0072, -0.0017]</td><td>1/3</td></tr><tr><td>ImageNet-1k</td><td>DECAF-9 -M</td><td>+0.0073</td><td>[0.0045, 0.0101]</td><td>2/3</td></tr></table>

Table 9: Endpoint-versus-trajectory ablation by architecture. Best ∆ is the largest DECAF-3/5/9 Spearman minus endpoint M.
<table><tr><td>Dataset</td><td>Model</td><td>Endpoint M</td><td>DECAF-3</td><td>DECAF-5</td><td>DECAF-9</td><td>Best ∆</td></tr><tr><td>FunnyBirds</td><td>ResNet-50</td><td>0.388</td><td>0.425</td><td>0.447</td><td>0.451</td><td>+0.064</td></tr><tr><td>FunnyBirds</td><td>VGG-16</td><td>0.306</td><td>0.283</td><td>0.298</td><td>0.293</td><td>-0.008</td></tr><tr><td>FunnyBirds</td><td>ViT-B/16</td><td>0.277</td><td>0.409</td><td>0.465</td><td>0.475</td><td>+0.197</td></tr><tr><td>ImageNet-1k</td><td>ResNet-50</td><td>0.384</td><td>0.362</td><td>0.370</td><td>0.387</td><td>+0.002</td></tr><tr><td>ImageNet-1k</td><td>VGG-16</td><td>0.671</td><td>0.640</td><td>0.640</td><td>0.644</td><td>-0.027</td></tr><tr><td>ImageNet-1k</td><td>ViT-B/16</td><td>0.058</td><td>0.077</td><td>0.090</td><td>0.105</td><td>+0.047</td></tr></table>

## H.3 ENDPOINT VERSUS TRAJECTORY

FunnyBirds is the clean trajectory-value test because its evaluation operators differ from the explanation endpoint. All three DECAF grids significantly exceed M. Five stages capture 96.3% of the nine-stage gain over the endpoint. IDSDS evaluates the same deletion contrast that defines $d _ { k }$ Endpoint M is therefore already highly aligned with its target. Three and five stages preserve most of that signal, while nine stages add a small but stable gain.

The trajectory gain is architecture-dependent in the tested models. It is largest for ViT-B/16 on both datasets, modest for ResNet-50, and negative for VGG-16. This pattern is empirical rather than a general architecture law, but it shows that the value of intermediate responses is not uniform across model classes.

The separate native FunnyBirds audit in Appendix H.2 provides the complementary endpoint-aligned case: when the evaluation target is the exact native part-removal contrast, endpoint M reaches ρ = 0.913 and exceeds all trajectory summaries.

Together, the two FunnyBirds evaluations isolate the role of the trajectory: the endpoint best recovers its own intervention effect, while intermediate responses add value when attribution must transfer to a different intervention.

## H.4 FULL-SCALE IMAGENET VALIDATION

The 10,000-image conclusions are not a favorable-subset artifact. The full validation set preserves the ordering M > DECAF-5 > IG-U-32 > IG-32. The full-scale paired difference between DECAF-5 and M is −0.0075 with a 95% interval of [−0.0086, −0.0062].

## H.5 MEASURED COMPUTE AND LARGE-MODEL SCALING

DECAF-3 is the speed-oriented Pareto point. Relative to IG-32, it has higher quality on both datasets, is 1.30× faster, and uses 4.50× less peak memory. DECAF-5 trades additional latency for higher quality while keeping the same inference-scale memory. KernelSHAP-512 is 5.89× slower than DECAF-5 and uses six times as many forward rows.

## H.6 PARTIMAGENET AS A TASK-ALIGNED BOUNDARY CASE

PartImageNet supplies semantic part masks and evaluates held-out part-removal effects. Direct part removal and coalition methods are therefore unusually aligned with the evaluation target. We retain the benchmark as a boundary case rather than using it to define the main general-purpose comparison.

Table 10: Full 50,000-image ImageNet validation scale check. Values are equal-architecture macro-average IDSDS over 115,876 correctly classified model–image units.
<table><tr><td>Method</td><td>Macro ρ [95% CI]</td><td>ResNet-50</td><td>VGG-16</td><td>ViT-B/16</td></tr><tr><td>Endpoint M</td><td>0.3708 [0.3685, 0.3731]</td><td>0.3778</td><td>0.6731</td><td>0.0615</td></tr><tr><td>DECAF-5</td><td>0.3633 [0.3613, 0.3653]</td><td>0.3609</td><td>0.6395</td><td>0.0894</td></tr><tr><td>IG-U-32</td><td>0.2942 [0.2926, 0.2958]</td><td>0.2522</td><td>0.4939</td><td>0.1366</td></tr><tr><td>IG-32</td><td>0.2397 [0.2380, 0.2411]</td><td>0.1941</td><td>0.4002</td><td>0.1247</td></tr></table>

Table 11: Measured ImageNet-1k compute. Rows per image count model-forward input rows after batching. Timing is compute-only and macro-averaged across ResNet-50, VGG-16, and ViT-B/16.
<table><tr><td>Method</td><td>Access</td><td>Rows/image</td><td>Backward?</td><td>ms/image ↓</td><td>Peak GiB↓</td></tr><tr><td>DECAF-3</td><td>Forward only</td><td>51</td><td>No</td><td>22.1</td><td>11.2</td></tr><tr><td>DECAF-5</td><td>Forward only</td><td>85</td><td>No</td><td>36.7</td><td>11.2</td></tr><tr><td>DECAF-9</td><td>Forward only</td><td>153</td><td>No</td><td>65.9</td><td>11.2</td></tr><tr><td>Input×Gradient</td><td>Backward</td><td>1</td><td>Yes</td><td>1.4</td><td>2.3</td></tr><tr><td>DeepLIFT</td><td>Backward</td><td>2</td><td>Yes</td><td>3.1</td><td>6.6</td></tr><tr><td>GradientSHAP</td><td>Backward</td><td>16</td><td>Yes</td><td>14.4</td><td>25.9</td></tr><tr><td>IG-32</td><td>Backward</td><td>32</td><td>Yes</td><td>28.7</td><td>50.4</td></tr><tr><td>IG-U-32</td><td>Backward</td><td>32</td><td>Yes</td><td>28.7</td><td>50.4</td></tr><tr><td>SmoothGrad-16</td><td>Backward</td><td>16</td><td>Yes</td><td>16.1</td><td>25.9</td></tr><tr><td>RISE-512</td><td>Sampling</td><td>512</td><td>No</td><td>216.6</td><td>14.4</td></tr><tr><td>KernelSHAP-512</td><td>Sampling</td><td>512</td><td>No</td><td>216.4</td><td>14.4</td></tr></table>

Table 12: PartImageNet strict-common-support Spearman. Part-removal and coalition methods exploit the supplied semantic part groups and are directly aligned with the held-out part-removal target.
<table><tr><td>Method</td><td>Spearman [95% CI]</td></tr><tr><td>Part-LIME-1000 Part Occlusion Exact Part-Shapley KernelSHAP-512</td><td>0.478 [0.458, 0.497] 0.477 [0.458, 0.496] 0.433 [0.412, 0.454] 0.426 [0.405, 0.447]</td></tr><tr><td>Endpoint M DECAF-9</td><td>0.364 [0.342, 0.385] 0.363 [0.341, 0.384]</td></tr><tr><td>DECAF-5</td><td>0.358 [0.337, 0.379]</td></tr><tr><td>DECAF-3</td><td>0.350 [0.329, 0.370] 0.299 [0.276, 0.320]</td></tr><tr><td>RISE-512</td><td>0.289 [0.270, 0.311]</td></tr><tr><td>IG-32</td><td></td></tr></table>

This result marks the method boundary clearly. When perfect semantic parts are already supplied and the target is itself a part-removal effect, direct part interventions can be stronger. Even in that setting, the DECAF trajectories remain above standard gradient-path attribution and preserve their forward-only memory advantage.

## I COVERTYPE EXPERIMENTAL DETAILS AND COMPLETE RESULTS

This appendix reports the complete Covertype audit behind Section 5. The experiment uses a natural tabular base and controlled context–factor channels. Its purpose is not to assume that every treatment succeeds. It measures which response mechanism each trained model actually realizes.

## I.1 SETUP AND OPERATIONAL OUTCOMES

We balance classes 1 and 2 from Covertype and retain 240,000 examples. The 54 natural features are augmented by one binary context and one binary candidate factor. The direction module uses context G and factor Z; the fragility module uses context H and factor U. Natural features and data splits are shared across all mechanism variants.

For the direction module, the endpoint is $G = + 1$ . We query the factor effect under $G = + 1$ and $G = - 1$ , then define preservation, collapse, and inversion from the held-out signed responses. For the fragility module, the endpoint is $H \stackrel { - } { = } + 1$ . The primary behavioral target used in Section 5 is the prediction-change rate under $H = - 1$ among examples satisfying the endpoint-null gate. This target was stored by the formal experiment and matches the definition of $F$

Table 13: Covertype benchmark design. Each model family uses three seeds. Direction experiments additionally use shortcut strengths $p \in \{ 0 . 7 5 , 0 . 9 5 \}$ .
<table><tr><td>Module</td><td>Regimes</td><td>Held-out operational behavior</td><td>Families</td><td>Models</td></tr><tr><td>Direction</td><td>Direct, Gate, Invert</td><td>preserve / collapse / invert</td><td>5</td><td>90</td></tr><tr><td>Fragility</td><td>Robust, Mild, Fragile</td><td>endpoint-null prediction change</td><td>5</td><td>45</td></tr><tr><td>Total</td><td></td><td></td><td></td><td>135</td></tr></table>

Table 14: Spearman correlation with realized behavior. SHAP interaction is available for random forests and XGBoost; KernelSHAP and LIME use their fixed formal subsets. F values use the endpoint-null behavior target.
<table><tr><td>Method</td><td>Preservation  $\rho$ </td><td>Actual inversion  $\rho$ </td><td>Endpoint-null change  $\rho$ </td></tr><tr><td>DECAF component</td><td>0.864</td><td>0.987</td><td>0.974</td></tr><tr><td>Endpoint M</td><td>0.657</td><td>-0.001</td><td>0.148</td></tr><tr><td>Abs</td><td>0.804</td><td>-0.207</td><td>0.588</td></tr><tr><td>Signed net</td><td>0.878</td><td>-0.353</td><td>0.043</td></tr><tr><td>SignFlip</td><td>-0.919 (78)</td><td>0.888 (78)</td><td>0.842 (39)</td></tr><tr><td>OppMass</td><td>-0.412</td><td>0.974</td><td>0.965</td></tr><tr><td>Native SHAP</td><td>0.781</td><td>-0.205</td><td>0.481</td></tr><tr><td>SHAP interaction</td><td>-0.251 (54)</td><td>0.093 (54)</td><td>0.069 (27)</td></tr><tr><td>KernelSHAP</td><td>0.811 (30)</td><td>-0.219 (30)</td><td>0.441 (15)</td></tr><tr><td>Global PFI</td><td>0.664</td><td>-0.029</td><td>0.676</td></tr><tr><td>Context-conditioned PFI</td><td>0.298</td><td>0.311</td><td>0.739</td></tr><tr><td>PDP/ALE interaction</td><td>-0.100</td><td>0.827</td><td>0.937</td></tr><tr><td>LIME</td><td>0.845 (30)</td><td>-0.384 (30)</td><td>0.506 (15)</td></tr></table>

Table 15: Model-family audit. Direction columns average the Invert regime across two strengths and three seeds. Fragility columns average the Fragile regime across three seeds. “Active” is the endpoint-active fraction; “Null change” is the alternate-context prediction-change rate on endpoint-null examples.
<table><tr><td></td><td colspan="3">Invert treatment</td><td colspan="5">Fragile treatment</td></tr><tr><td>Model family</td><td> $C$ </td><td>Invert rate</td><td>Collapse rate</td><td>M</td><td> $E$ </td><td> $F$ </td><td>Active</td><td>Null change</td></tr><tr><td>HistGradientBoosting</td><td>0.000</td><td>0.000</td><td>1.000</td><td>0.020</td><td>0.070</td><td>0.180</td><td>0.340</td><td>0.623</td></tr><tr><td>Logistic regression</td><td>0.000</td><td>0.000</td><td>1.000</td><td>0.299</td><td>0.299</td><td>0.0001</td><td>0.989</td><td>0.000</td></tr><tr><td>MLP</td><td>0.088</td><td>0.743</td><td>0.257</td><td>0.028</td><td>0.084</td><td>0.080</td><td>0.350</td><td>0.219</td></tr><tr><td>Random forest</td><td>0.024</td><td>0.816</td><td>0.169</td><td>0.057</td><td>0.170</td><td>0.062</td><td>0.673</td><td>0.327</td></tr><tr><td>XGBoost</td><td>0.000</td><td>0.002</td><td>0.998</td><td>0.020</td><td>0.091</td><td>0.163</td><td>0.346</td><td>0.552</td></tr></table>

## I.2 COMPLETE BEHAVIOR ALIGNMENT

Table 14 reports the complete rank comparison. Evidence uses preservation, contradiction uses actual inversion, and fragility uses endpoint-null alternate-context prediction change. The first two columns use 90 direction models; the last uses 45 fragility models. Parenthetical sample counts mark methods with partial coverage.

For the two headline semantic coordinates, joint family/seed cluster bootstrap gives

$$
\rho ( C , V _ { \mathrm { i n v } } ) = 0 . 9 8 6 8 , \qquad 9 5 \% \mathrm { C I } = [ 0 . 9 7 2 5 , 0 . 9 9 6 7 ] ,
$$

and

$$
\rho ( F , V _ { \mathrm { n u l l } } ) = 0 . 9 7 4 1 , \qquad 9 5 \% \mathrm { C I } = [ 0 . 9 4 1 6 , 0 . 9 8 8 3 ] .
$$

Evidence reaches $\rho ( E , V _ { \mathrm { k e e p } } ) = 0 . 8 6 4 2$ with 95% interval [0.8318, 0.8955].

## I.3 THE SAME TREATMENT PRODUCES DIFFERENT REALIZED MECHANISMS

Table 15 shows why treatment labels cannot serve as mechanism ground truth. The Invert generator produces genuine reversal in random forests and multilayer perceptrons, but mostly collapse in the remaining families. The Fragile generator becomes endpoint evidence in logistic regression, while nonlinear families retain substantial endpoint-null mass.

Table 16: Mechanism identification with fixed semantics and after conditioning on ordinary magnitude. Learned references use mechanism labels and are not access-matched to fixed DECAF.
<table><tr><td>Method</td><td>Fixed macro-AUROC</td><td>Fixed macro-AUPRC</td><td>Within-Abs macro-AUROC</td><td>Within-Abs macro-AUPRC</td></tr><tr><td>DECAF / M / Abs /</td><td>0.816</td><td>0.768</td><td>0.956</td><td>0.948</td></tr><tr><td>Net / OppMass / SignFlip Endpoint M</td><td>0.487</td><td>0.437</td><td>0.565</td><td>0.666</td></tr><tr><td>Abs</td><td>0.518</td><td>0.450</td><td>0.495</td><td>0.585</td></tr><tr><td>Native SHAP</td><td>0.523</td><td>0.437</td><td>0.461</td><td>0.570</td></tr><tr><td>SHAP interaction</td><td>0.540</td><td>0.440</td><td></td><td></td></tr><tr><td>Context-conditioned PFI</td><td>0.556</td><td>0.545</td><td>0.468</td><td>0.634</td></tr><tr><td>PDP/ALE interaction</td><td>0.539</td><td>0.438</td><td>0.559</td><td>0.739</td></tr><tr><td>Strong tabular reference (supervised)</td><td>0.987</td><td>0.968</td><td>0.983</td><td>0.994</td></tr><tr><td>DECAF probe (supervised)</td><td>0.902</td><td>0.859</td><td>0.742</td><td>0.779</td></tr><tr><td>Combined empirical reference (supervised)</td><td>0.985</td><td>0.965</td><td>0.983</td><td>0.994</td></tr></table>

Within the Invert treatment alone, C still tracks the amount of realized inversion with $\rho = 0 . 9 2 2$ Among families with nonconstant inversion behavior, the correlations are 0.997 for histogram gradient boosting, 0.992 for random forests, 0.971 for XGBoost, and 0.940 for the MLP. Logistic regression has zero inversion throughout, so its within-family correlation is undefined.

## I.4 FIXED SEMANTIC READOUT AND MAGNITUDE CONDITIONING

The fixed semantic benchmark assigns E, C, and F to Evidence, Contradiction, and Fragility without fitting a meta-classifier. Table 16 reports both the unconditional result and the within-Abs-bin result. The latter controls response magnitude through 12 quantile bins.

The fixed DECAF mechanism accuracy is 0.587. The supervised references use leave-one-modelfamily-out calibration, mechanism labels, and up to 15 input summaries. They bound cross-family decodability but do not replace the label-free comparison.

## I.5 MEASURED COST AND SHAP-INTERACTION AUDIT

Table 17 reports the registered cumulative worker time. Relative cost divides worker-seconds per model by the DECAF value. Coverage differs for methods whose formal budget uses a model subset.

Table 17: Complete measured method cost. SHAP interaction uses 128 stratified examples per tree model, split into four 32-example shards.
<table><tr><td>Method</td><td>Models</td><td>Worker-s/model</td><td>Relative to DECAF</td><td>Predicted rows</td></tr><tr><td>DECAF / M / Abs / Net / OppMass / SignFlip</td><td>135</td><td>1.42</td><td>1.0×</td><td>25,920,000</td></tr><tr><td>PDP/ALE interaction</td><td>135</td><td>2.22</td><td>1.57×</td><td>38,880,000</td></tr><tr><td>LIME</td><td>45</td><td>8.38</td><td>5.92×</td><td>11,796,480</td></tr><tr><td>PFI / context PFI</td><td>135</td><td>13.32</td><td>9.41×</td><td>207,360,000</td></tr><tr><td>KernelSHAP</td><td>45</td><td>45.30</td><td>31.99×</td><td>377,501,760</td></tr><tr><td>Native SHAP</td><td>135</td><td>387.65</td><td>273.8×</td><td></td></tr><tr><td>Retraining reference</td><td>135</td><td>19.70</td><td>13.91×</td><td></td></tr><tr><td>SHAP interaction</td><td>54</td><td>15097.63</td><td>10662×</td><td></td></tr></table>

The formal SHAP-interaction run completed 216 shards over 54 tree models. It consumed at least 811,875 CPU-seconds (225.5 CPU-hours), used up to 32 concurrent shards, and required 17.71 hours of elapsed stage time. Its inversion correlation was only 0.093 with a 95% interval spanning zero.

## I.6 MATCHED-PAIR AUDIT AND ITS LIMITS

The automatic pair search produced only 11 protocol pairs. They cover 29.3% of primary units, contain no same-family pair, and have a median relative endpoint-magnitude difference of 0.870 (maximum 0.945), although their Abs difference is small. DECAF reaches 0.727 mechanism accuracy on these pairs, but this set does not support a headline claim that both Abs and M are matched. We therefore use the much better populated within-Abs-bin analysis in Table 16 and retain the 11-pair result only as an audit.

## REFERENCES

Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim. Sanity checks for saliency maps. In Advances in Neural Information Processing Systems, volume 31, 2018.

Sebastian Bach, Alexander Binder, Gregoire Montavon, Frederick Klauschen, Klaus-Robert M´ uller,¨ and Wojciech Samek. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. PLOS ONE, 10(7):e0130140, 2015.

Jock Blackard. Covertype. UCI Machine Learning Repository, 1998.

David Blackwell. Equivalent comparisons of experiments. The Annals of Mathematical Statistics, 24 (2):265–272, 1953.

Christopher P. Burgess, Irina Higgins, Arka Pal, Loic Matthey, Nick Watters, Guillaume Desjardins, and Alexander Lerchner. Understanding disentangling in β-vae. arXiv preprint arXiv:1804.03599, 2018.

Chun-Hao Chang, Elliot Creager, Anna Goldenberg, and David Duvenaud. Explaining image classifiers by counterfactual generation. In International Conference on Learning Representations, 2019.

Aditya Chattopadhyay, Piyushi Manupriya, Anirban Sarkar, and Vineeth N. Balasubramanian. Neural network attributions: A causal perspective. In Proceedings of the 36th International Conference on Machine Learning, pp. 981–990, 2019.

Ian Covert, Scott Lundberg, and Su-In Lee. Explaining by removing: A unified framework for model explanation. Journal ofMachine Learning Research, 22(209):1–90, 2021a.

Ian Covert, Scott Lundberg, and Su-In Lee. Explaining by removing: A unified framework for model explanation. Journal ofMachine Learning Research, 22(209):1–90, 2021b.

Amit Dhurandhar, Pin-Yu Chen, Ronny Luss, Chun-Chen Tu, Paishun Ting, Karthikeyan Shanmugam, and Payel Das. Explanations based on the missing: Towards contrastive explanations with pertinent negatives. In Advances in Neural Information Processing Systems, volume 31, 2018a.

Amit Dhurandhar, Pin-Yu Chen, Ronny Luss, Chun-Chen Tu, Paishun Ting, Karthikeyan Shanmugam, and Payel Das. Explanations based on the missing: Towards contrastive explanations with pertinent negatives. In Advances in Neural Information Processing Systems, volume 31, pp. 590–601, 2018b.

Ann-Kathrin Dombrowski, Maximilian Alber, Christopher Anders, Marcel Ackermann, Klaus-Robert Muller, and Pan Kessel. Explanations can be manipulated and geometry is to blame. In¨ Advances in Neural Information Processing Systems, volume 32, 2019.

Ruth Fong, Mandela Patrick, and Andrea Vedaldi. Understanding deep networks via extremal perturbations and smooth masks. In Proceedings ofthe IEEE International Conference on Computer Vision, pp. 2950–2958, 2019.

Ruth C. Fong and Andrea Vedaldi. Interpretable explanations of black boxes by meaningful perturbation. In Proceedings of the IEEE International Conference on Computer Vision, pp. 3429–3437, 2017.

Christopher Frye, Colin Rowat, and Ilya Feige. Shapley explainability on the data manifold. In International Conference on Learning Representations, 2020.

Robert Geirhos, Jorn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias¨ Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2:665–673, 2020.

Amirata Ghorbani, Abubakar Abid, and James Zou. Interpretation of neural networks is fragile. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 33, pp. 3681–3688, 2019.

Yash Goyal, Ziyan Wu, Jan Ernst, Dhruv Batra, Devi Parikh, and Stefan Lee. Counterfactual visual explanations. In Proceedings of the 36th International Conference on Machine Learning, pp. 2376–2384, 2019.

Yash Goyal, Amir Feder, Uri Shalit, and Been Kim. Explaining classifiers with causal concept effect. arXiv preprint arXiv:1907.07165, 2020.

Yikai Gu, Lele Cao, Bo Zhao, Lei Lei, and Lei You. Discover: A solver for distributional counterfactual explanations. arXiv preprint arXiv:2603.16436, 2026.

Robin Hesse, Simone Schaub-Meyer, and Stefan Roth. Funnybirds: A synthetic vision dataset for a part-based analysis of explainable ai methods. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3981–3991, 2023.

Robin Hesse, Simone Schaub-Meyer, and Stefan Roth. Benchmarking the attribution quality of vision models. In Advances in Neural Information Processing Systems, volume 37, 2024.

Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, and Been Kim. A benchmark for interpretabil ity methods in deep neural networks. In Advances in Neural Information Processing Systems, volume 32, 2019.

Dominik Janzing, Lenon Minorics, and Patrick Blobaum. Feature relevance quantification in ex-¨ plainable ai: A causal problem. In Proceedings ofthe Twenty Third International Conference on Artificial Intelligence and Statistics, pp. 2907–2916, 2020.

Neil Jethani, Mukund Sudarshan, Ian Connick Covert, Su-In Lee, and Rajesh Ranganath. Fastshap: Real-time shapley value estimation. In International Conference on Learning Representations, 2022.

Neil Jethani, Adriel Saporta, and Rajesh Ranganath. Don’t be fooled: Label leakage in explanation methods and the importance of their quantitative evaluation. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings ofMachine Learning Research, pp. 8925–8953, 2023.

Andrei Kapishnikov, Subhashini Venugopalan, Besim Avci, Ben Wedin, Michael Terry, and Tolga Bolukbasi. Guided integrated gradients: An adaptive path method for removing noise. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5050–5058, 2021.

Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, and Rory´ Sayres. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors. In Proceedings of the 35th International Conference on Machine Learning, pp. 2668–2677, 2018.

Pieter-Jan Kindermans, Sara Hooker, Julius Adebayo, Maximilian Alber, Kristof T. Schutt, Sven¨ Dahne, Dumitru Erhan, and Been Kim. The (un)reliability of saliency methods. In¨ Explainable AI: Interpreting, Explaining and Visualizing Deep Learning, pp. 267–280. Springer, 2019.

Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In Proceedings of the 37th International Conference on Machine Learning, pp. 5338–5348, 2020.

Scott M. Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems, volume 30, 2017.

Vitali Petsiuk, Abir Das, and Kate Saenko. Rise: Randomized input sampling for explanation of black-box models. In Proceedings ofthe British Machine Vision Conference, 2018.

Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. Why should i trust you?: Explaining the predictions of any classifier. In Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1135–1144, 2016.

Yao Rong, Tobias Leemann, Vadim Borisov, Gjergji Kasneci, and Enkelejda Kasneci. A consistent and efficient evaluation strategy for attribution methods. In Proceedings of the 39th International Conference on Machine Learning, pp. 18770–18795, 2022.

Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings ofthe IEEE International Conference on Computer Vision, pp. 618–626, 2017.

Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. Learning important features through propagating activation differences. In Proceedings of the 34th International Conference on Machine Learning, pp. 3145–3153, 2017.

Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. International Conference on Learning Representations Workshop, 2014.

Dylan Slack, Sophie Hilgard, Emily Jia, Sameer Singh, and Himabindu Lakkaraju. Fooling lime and shap: Adversarial attacks on post hoc explanation methods. Proceedings ofthe AAAI/ACM Conference on AI, Ethics, and Society, pp. 180–186, 2020.

Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viegas, and Martin Wattenberg. Smoothgrad:´ Removing noise by adding noise. arXiv preprint arXiv:1706.03825, 2017.

Suraj Srinivas and Franc¸ois Fleuret. Full-gradient representation for neural network visualization. In Advances in Neural Information Processing Systems, volume 32, 2019.

Pascal Sturmfels, Scott Lundberg, and Su-In Lee. Visualizing the impact of feature attribution baselines. Distill, 5(1):e22, 2020.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings ofthe 34th International Conference on Machine Learning, pp. 3319–3328, 2017.

Kai Xiao, Logan Engstrom, Andrew Ilyas, and Aleksander Madry. Noise or signal: The role of image backgrounds in object recognition. In International Conference on Learning Representations, 2021.

Shawn Xu, Subhashini Venugopalan, and Mukund Sundararajan. Attribution in scale and space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9680–9689, 2020.

Chih-Kuan Yeh, Cheng-Yu Hsieh, Arun Sai Suggala, David I. Inouye, and Pradeep Ravikumar. On the (in)fidelity and sensitivity of explanations. In Advances in Neural Information Processing Systems, volume 32, 2019.

Lei You, Lele Cao, Mattias Nilsson, Bo Zhao, and Lei Lei. Distributional counterfactual explanations with optimal transport. In Yingzhen Li, Stephan Mandt, Shipra Agrawal, and Emtiyaz Khan (eds.), Proceedings ofThe 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings ofMachine Learning Research, pp. 1135–1143. PMLR, 03–05 May 2025.

Lei You, Yijun Bian, and Lele Cao. Joint distribution–informed shapley values for sparse counterfactual explanations. In International Conference on Learning Representations, volume 2026, pp. 100183–100205, 2026.

Matthew D. Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In European Conference on Computer Vision, pp. 818–833, 2014.
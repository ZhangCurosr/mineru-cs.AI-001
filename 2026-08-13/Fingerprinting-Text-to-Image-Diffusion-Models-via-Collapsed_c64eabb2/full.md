# Fingerprinting Text-to-Image Diffusion Models via Collapsed Generation

Yuanmin Huang, Chen Chen, Geng Hong, Xiaoyu You, Hui Xue, Zhenxing Qian, Mi Zhang, and Min Yang

Abstract—Proprietary text-to-image diffusion models are increasingly distributed as hosted services and downloadable checkpoints, making their intellectual property (IP) protection an increasingly critical concern when model leakage, copying, or unauthorized fine-tuning is disputed. In this work, we present a non-invasive model fingerprinting framework based on collapsed generation, a phenomenon where certain input conditions produce highly consistent images across multiple stochastic seeds. We show that collapsed generation is an intrinsic, model-dependent property of the learned generation process. These collapse-prone conditions therefore expose model-specific behavioral signatures, enabling reliable ownership verification without embedding invasive watermarks. After preparing conditions on the source model, the framework verifies a suspect model under two access settings: (1) white-box pipeline access, where optimized continuous embeddings can be injected into the generation process, and (2) blackbox API-only access, where natural language prompts are queried through the service interface. In both cases, ownership evidence is measured by whether the suspect model reproduces the source model’s collapse behavior across stochastic samplings. Extensive experiments across UNet- and transformer-based diffusion models show that collapsed generation fingerprints can distinguish different source models with low confusion. These fingerprints remain verifiable in fine-tuned derivatives and under common and adaptive model- or query-level obfuscations, while requiring only a modest verification query budget. Together, these results establish collapsed generation as a reliable intrinsic evidence source for non-invasive diffusion model ownership verification.

Index Terms—Text-to-Image Diffusion Models, Intellectual Property Protection, Model Fingerprinting, Ownership Verification.

## I. INTRODUCTION

The rapid advancement of text-to-image (T2I) diffusion models has profoundly transformed generative AI, enabling high-fidelity image synthesis directly from natural language prompts [1], [2]. These models now serve as foundational components in creative industries and commercial pipelines [3]– [5]. As proprietary checkpoints are increasingly released, shared, fine-tuned, or deployed behind commercial APIs, disputes over model leakage, unauthorized reuse, and derivative ownership have become a practical intellectual property (IP)

concern [6], [7]. This growing tension underscores the need for dedicated IP protection mechanisms for diffusion models.

Existing IP protection methods for diffusion models generally fall into two categories: invasive model watermark ing [8]–[10] and non-invasive model fingerprinting [11]–[14]. The former embeds ownership signals by altering model parameters, training objectives, or generation outputs, which often introduces additional computational overhead and may compromise generation quality. In contrast, non-invasive fingerprinting methods attempt to identify models through their inherent behaviors under specific inputs, without modifying model parameters or retraining, thereby preserving model integrity and usability.

However, existing non-invasive fingerprints only partially address the verification needs arising from leaked checkpoints, unauthorized derivatives, and hosted T2I services. Methods based on internal representations or denoising trajectories can be useful when the verifier obtains a checkpoint or a controllable generation pipeline, but their verification evidence is tied to model-internal readouts or specialized sampling procedures [13], [14]. Such assumptions do not hold when the suspect model is exposed only as a hosted T2I service that returns final images. Conversely, query-based methods are closer to the API-only setting, but existing designs often rely on optimized prompt suffixes whose linguistic abnormality and induced semantic deviation make the verification queries easier to distinguish as non-ordinary service interactions [12]. These limitations motivate a different type of fingerprint signal: one whose evidence is expressed in the model’s generation behavior, can be actively elicited when the generation pipeline is controllable, and can still be observed through natural prompts when only API outputs are available.

In this work, we exploit an intrinsic behavior of diffusion models, termed collapsed generation, as a non-invasive ownership signature. Under ordinary prompts, independent stochastic samplings usually produce diverse images. Under certain model-dependent input conditions, however, the sampling stochasticity is suppressed and different generation trajectories converge to visually similar outputs [15], [16], as illustrated in Fig. 1 (a). Such collapse-prone conditions are shaped by the learned data distribution, optimization dynamics, and model architecture [17], [18], and therefore can expose behavioral signatures that are difficult for independently trained models to reproduce.

Based on this observation, we formulate collapse-prone conditions as model-intrinsic fingerprints for ownership verification. A model owner first constructs a fingerprint set on the source model with the resulting conditions as reference fingerprints, as illustrated in Fig. 1 (b). During verification, the verifier applies these conditions to a suspect model and measures whether its independent stochastic generations reproduce the characteristic cross-seed consistency of the source model. We consider two verification interfaces for the suspect model. Under white-box pipeline access, the verifier can control the generation process and inject continuous prompt embeddings, whereas under black-box API-only access, the verifier submits text prompts and observes the final generated images.

![](images/cc42850d099f9ce42899d78b7a550903ec32351a87801c5ffbb8bfdced32b347.jpg)  
Fig. 1: Collapsed generation as a non-invasive ownership signal for diffusion models. (a) Collapsed generation. Under ordinary input conditions, different stochastic seeds usually produce diverse images, whereas a collapse-prone condition yield highly consistent outputs on the source model. Such collapse behavior is model-dependent and therefore can serve as a behavioral fingerprint. (b) Ownership verification. The model owner prepares such conditions on the source model, and the verifier tests whether a suspect model reproduces the corresponding cross-seed consistency. The framework supports both pipeline-access and API-only verification, and remains effective under model obfuscations.

In ownership disputes involving leaked checkpoints or unauthorized derivative models (e.g., through LoRA finetuning), the suspect model may be available as a controllable generation pipeline. Such pipeline access allows the verifier to actively elicit model-specific collapsed generation by searching the continuous text embedding space. However, directly searching this space can be computationally demanding, as it requires repeated differentiation through the iterative generation process. We find that the collapse signature becomes distinguishable during the early denoising stage, well before the final image is produced. This observation motivates a truncated optimization strategy that efficiently synthesizes reproducible fingerprint embeddings for checkpoint-level ownership verification.

In other cases, an unauthorized model may be deployed behind a hosted T2I service and exposed only through an endto-end API. Verification must therefore rely on text prompts and the resulting generated images. Consequently, the fingerprint conditions must be realized as text prompts rather than optimized continuous embeddings. The challenge is to identify prompts that intrinsically trigger model-specific collapse without exhaustively screening a vast prompt space through repeated generation. We find that naturally collapsed prompts are strongly concentrated among training samples with unusually low loss. This relationship provides an efficient way to mine natural fingerprint prompts from training dynamics for service-level ownership verification.

Despite their different construction mechanisms, both realizations rely on the same verification principle. A suspect model is associated with the source model when the prepared fingerprint conditions reproduce statistically abnormal consistency across independent generations. This shared criterion provides a unified behavioral basis for ownership verification.

Our main contributions are summarized as follows:

• We identify collapsed generation as a model-intrinsic behavioral signature and formulate its abnormal consistency across stochastic generations as a non-invasive fingerprint signal for T2I model ownership verification.

• We develop a unified framework that bridges white-box checkpoint-level and black-box service-level ownership verification through the same collapsed-generation signature. Building on insights from early denoising behavior and training dynamics, the framework constructs continuous embeddings for controllable pipelines and natural prompts for hosted APIs.

• We evaluate the framework across multiple T2I model families spanning UNet- and transformer-based architectures. The results demonstrate clear discrimination among independently trained models, while the fingerprints remain detectable after fine-tuning, pruning, quantization, and adaptive query-time interventions with a modest verification query budget.

## II. RELATED WORK

## A. Model Ownership Verification

Model ownership verification aims to determine whether a suspect model originates from, or is derived from, a source model claimed by an owner. Existing approaches generally establish ownership through model watermarks or model fingerprints. Model watermarking proactively embeds a secret ownership signal into the protected model through its parameters, training objective, or trigger-response behavior [19]–[21]. Model fingerprinting instead characterizes a trained model using distinctive decision boundaries, adversarial responses, or internal behaviors, without modifying the protected model [22]–[24]. For reliable ownership verification, the verification signal should remain reproducible after model obfuscations, such as fine-tuning, pruning, and quantization, while being sufficiently specific to distinguish the claimed source from independently trained or unrelated models. The realization of these requirements depends on the architecture and deployment interface of the protected model, motivating specialized IP protection methods for diffusion models.

TABLE I: Methodological comparison of non-invasive diffusion model fingerprints. White-box and black-box refer to the verifier’s access to the suspect model. The table compares the input conditions and evidence used for verification, while further specifying the model control or observations required by each method.
<table><tr><td>Method</td><td>Verification Access</td><td>Input Condition for Verification</td><td>Verification Evidence</td><td>Required Control or Observation</td><td>Natural Query</td></tr><tr><td>FingerInv [13]</td><td>White-box</td><td>Optimized latent noise</td><td>QR-code recovery from the final image</td><td>DDPM-compatible denoising control and intermediate denoising outputs</td><td>N/A</td></tr><tr><td>DiffIP [14]</td><td>White-box</td><td>Inputs for representation extraction</td><td>Representation- fingerprint similarity</td><td>Internal denoising representations of both models</td><td>N/A</td></tr><tr><td>Ours</td><td>White-box</td><td>Optimized continuous prompt embedding</td><td>Cross-seed output consistency</td><td>Text-embedding injection and final-image observation</td><td>N/A</td></tr><tr><td>TVN [12]</td><td>Black-box</td><td>Text prompt with an optimized adversarial suffix</td><td>Target-specific semantic deviation</td><td>Text-query API and final-image observation</td><td>No</td></tr><tr><td>Ours</td><td>Black-box</td><td>Natural collapsed prompt</td><td>Cross-seed output consistency</td><td>Text-query API and final-image observation</td><td>Yes</td></tr></table>

## B. Diffusion Model IP Protection

Invasive Model Watermarking. A prominent line of diffusion model IP protection relies on watermarking, which deliberately implants a predefined ownership signal before deployment and later verifies whether the signal can be recovered from the model’s generated outputs. By embedding an ownercontrolled signal that can be recovered through prescribed queries or output analysis, these methods provide explicit evidence for subsequent ownership verification. Zhao et al. [8] introduce watermark-bearing examples during training so that the resulting model reproduces designated watermark patterns, while Liu et al. [10] use secret trigger prompts to activate predefined watermark responses. Stable Signature [9] follows a different route by fine-tuning the latent decoder so that generated images carry an invisible signature without requiring a special trigger at inference time. More recent methods incorporate watermark conditioning or specialized training objectives to improve the persistence of ownership signals after downstream adaptation such as fine-tuning [25], [26]. Despite their different embedding mechanisms, these methods require an ownership signal to be deliberately introduced through the training data, optimization procedure, or model components before release. Such intervention introduces additional training and deployment overhead and may perturb the model’s original output distribution, potentially affecting its normal generation behavior. It also limits retrospective protection, since an already trained and unmarked model must be fine-tuned or retrained before watermark-based verification can be applied.

Non-Invasive Model Fingerprinting. The limitations of model watermarking motivate non-invasive model fingerprinting, which derives ownership evidence from behaviors already present in a trained model without modifying its parameters or training objective to implant an additional signal. Existing diffusion model fingerprints can be organized primarily according to the verifier’s access to the suspect model during ownership verification. This access constrains both the fingerprint conditions that can be applied to the suspect model and the evidence available for the final ownership decision. Note that white-box and black-box here refer to verification-time access to the suspect model rather than the owner’s access to the source model during fingerprint construction. Table I summarizes the access assumptions and verification evidence of these methods.

Under white-box access, the verifier can control the generation pipeline or inspect intermediate model information. FingerInv [13] constructs optimized latent-noise fingerprints through DDPM inversion and requires a controllable DDPMcompatible denoising process with access to intermediate denoising information. Its final ownership decision is based on whether a predefined QR-code pattern can be recovered from the generated image. DiffIP [14] extracts representation fingerprints from internal denoising features and verifies ownership by comparing the representations of the source and suspect models. The two methods therefore rely on different ownership evidence despite both operating under white-box access.

Under black-box access, the verifier can only submit text queries and observe the final generated images. TVN [12] realizes model fingerprints as text prompts with optimized non-transferable suffixes and verifies ownership from targetspecific semantic deviations in the resulting images. Although it supports end-to-end API verification, its optimized suffixes may exhibit linguistic abnormality. Moreover, since the generated outputs can deviate from the semantics of ordinary user queries, the verification queries may be more easily detected or filtered.

Built on collapsed generation, our framework supports both white-box and black-box ownership verification using the same behavioral evidence: statistically abnormal cross-seed consistency in the final generated images. For white-box verification, although pipeline control is required to apply optimized continuous conditions, the ownership decision relies only on the final outputs rather than intermediate denoising states or internal representations, avoiding the internal-information dependence of existing white-box methods. For black-box verification, our framework uses naturally collapsed prompts instead of optimized adversarial queries, making the verification queries closer to ordinary service interactions and less distinguishable by query detection or filtering mechanisms.

## C. Collapsed Generation in Diffusion Models

Prior studies have shown that diffusion models can memorize and reproduce training examples under particular input conditions [15], [17], [27], [28]. Carlini et al. [15] demonstrated that text-to-image diffusion models can regenerate individual training examples with high fidelity, enabling largescale training data extraction. Subsequent studies investigated the factors underlying such behavior, including data duplication [17], the geometry of the learned probability distribution [29], [30], training dynamics [18], and localized model parameters or subspaces [16], [31].

To identify memorized content, existing extraction methods search for prompts that trigger repeatable generations and compare the resulting images with reference training samples using pixel- or feature-level similarity measures [15], [27], [32]. Together, these findings suggest that particular input conditions may induce highly concentrated generation behavior, under which independent stochastic samplings yield visually similar outputs. We refer to this phenomenon as collapsed generation. Its dependence on the learned data distribution, training dynamics, and model parameters motivates our investigation of collapsed generation as a model-dependent behavioral signature for ownership verification.

## III. PRELIMINARIES

## A. Threat Model

We consider an IP protection scenario in which a model owner develops a proprietary T2I diffusion model, referred to as the source model. An adversary may obtain the source model without authorization and redistribute it directly, produce a modified or obfuscated derivative, or deploy it as a commercial service. The model under examination is referred to as the suspect model. The objective of the owner, or an authorized verifier acting on the owner’s behalf, is to determine whether the suspect model originates from or is derived from the source model.

A complete model fingerprinting framework involves both fingerprint construction and ownership verification. During fingerprint construction, we assume that the owner has full access to the source model and its training process.

To address potential ownership ambiguity arising from competing fingerprint claims, we assume a standard commitmentbased verification protocol [20]. Before deployment, the owner commits the fingerprint set to a trusted third party. A valid commitment with a timestamp preceding competing claims establishes the priority of the owner’s fingerprint evidence.

For ownership verification, we consider two settings that differ in the verifier’s access to the suspect model:

White-box verification. In this setting, the suspect model is available as a checkpoint, for example, after unauthorized open-source redistribution [11] or when it is lawfully obtained for forensic examination [19], [21]. The verifier has access to the model parameters and architecture and can observe and control the inference pipeline, including its latent variables, conditioning inputs, and intermediate denoising states.

Black-box verification. In this setting, the suspect model is deployed as an end-to-end T2I API or online service [9], [20], [25]. The verifier can submit text prompts and observe the corresponding generated images, but has no access to the model parameters, architecture or intermediate activations.

We assume that the adversary may apply model-level obfuscations, such as fine-tuning, pruning, and quantization, or, in the black-box setting, query-time interventions to evade verification, while preserving acceptable generation quality for normal use.

## B. Principles of Model Fingerprinting

Following prior work [13], we consider three desirable properties for diffusion model fingerprints:

Uniqueness. The fingerprint should be sufficiently distinctive to uniquely identify the target model among others. In particular, a fingerprint constructed from one source model should not be validated on an unrelated suspect model.

Robustness. The fingerprint should remain verifiable even after the model undergoes obfuscations. This property prevents adversaries from invalidating the verification.

Stealthiness. For black-box verification, fingerprint queries should appear natural and resemble ordinary service interactions. This reduces the likelihood that the queries are detected or filtered by the suspect service.

## C. Diffusion Models

Diffusion models generate data by iteratively denoising a random noise sample through a learned reverse process [33]. Let ${ \hat { x } } _ { 0 }$ denote the clean image, and $\{ x _ { t } \} _ { t = 1 } ^ { T }$ represent its progressively noised versions obtained via a forward diffusion process controlled by a variance schedule $\{ \beta _ { t } \}$

$$
x _ { t } = \sqrt { \bar { \alpha } _ { t } } \hat { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , I ) ,\tag{1}
$$

where $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } ( 1 - \beta _ { i } ) } \end{array}$

The reverse denoising process is parameterized by a neural network $\epsilon _ { \theta }$ that predicts the noise added at each step. Given a noisy sample $x _ { t } .$ , a single reverse step reconstructs $x _ { t - 1 }$ as:

$$
x _ { t - 1 } = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( x _ { t } - \frac { 1 - \alpha _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \theta } ( x _ { t } , t ) \right) + \sigma _ { t } z _ { t } ,\tag{2}
$$

![](images/6be343ea19555bcb58ea42781c15bd26f723a57cb35b4567547ea16aef552ec4.jpg)

![](images/1a6abb803eb8ec88f425198785650697d04486e09b6633ee416b9f8bde0b1a8e.jpg)  
(a)

![](images/f1a18ebd80125fc199629c0077ee6505d368f3b6dc19b90b2b5444e3258faecb.jpg)  
(b)  
Fig. 2: (a): Distribution of average pairwise similarity scores $\bar { s } ( p , K = 4 )$ for normal prompts vs. collapsed prompts on SD 1.4. (b): Distribution of $\bar { s } ( p , K = 4 )$ for SD 1.4 collapsed prompts vs. SD 2.1 collapsed prompts on SD 1.4 (left) and SD 2.1 (right), respectively. Collapsed generations not only differ from normal ones but also exhibit model-specific characteristics, enabling effective fingerprinting.

![](images/bbee5f17d4792def63440d7a3484de718bbd700cf5f34c7b7bc7ff76f269e3a0.jpg)  
Fig. 3: Different model architectures, training data, and optimization dynamics lead to distinct generative manifolds, resulting in model-specific collapsed generation behaviors.

where $\sigma _ { t }$ is the step-dependent noise scale and $z _ { t } \sim \mathcal { N } ( 0 , I )$ is random noise.

In T2I diffusion models [1], additional condition c is derived from a text encoder $E _ { \mathrm { t e x t } }$ that maps a prompt p to an embedding $c = E _ { \mathrm { t e x t } } ( p )$ . This augments the denoiser $\epsilon _ { \theta }$ with text embedding $c ,$ enabling conditional generation from a prompt $p .$ The overall sampling process from noise $x _ { T } \sim \mathcal { N } ( 0 , I )$ to the generated image $x _ { 0 }$ can be expressed as a function of the prompt:

$$
x _ { 0 } = G _ { \theta } ( x _ { T } , c = E _ { \mathrm { t e x t } } ( p ) ) ,\tag{3}
$$

where $x _ { T }$ controls the stochasticity of the generation.

For a given prompt $p ,$ different initial samples $x _ { T }$ typically yield diverse outputs, whereas consistent and low-diversity outputs indicate a collapsed generation behavior. This behavior forms the foundation for analyzing model-specific generative characteristics, which are exploited in our fingerprinting framework.

## IV. METHODOLOGY

## A. Collapsed Generation in Diffusion Models

Recent studies have shown that diffusion models can memorize and reproduce a subset of their training data [15], [17]. One manifestation of such memorization is reduced generation diversity. For certain prompts associated with memorized samples, independent stochastic samplings can produce highly similar outputs [29], [30]. This behavior contrasts with typical diffusion generation, where different initial noises are expected to yield diverse images under the same prompt. Although diffusion models are generally less prone to mode collapse than GANs [34], [35], this prompt-specific concentration exhibits a similar loss of output diversity. We term this phenomenon collapsed generation and formalize it through cross-seed consistency.

Definition 1 (Collapsed Generation). Let c denote the conditioning embedding supplied to a diffusion model $G _ { \theta } .$ For a text prompt p, $c = E _ { t e x t } ( p )$ . Given K independently sampled initial noises $\{ x _ { T } ^ { ( k ) } \} _ { k = 1 } ^ { K } ,$ , let

$$
x _ { 0 } ^ { ( k ) } ( c ) = G _ { \theta } ( x _ { T } ^ { ( k ) } , c )
$$

denote the corresponding generated images. Their cross-seed consistency is measured as

$$
\bar { s } _ { \theta } ( c , K ) = \frac { 2 } { K ( K - 1 ) } \sum _ { i = 1 } ^ { K - 1 } \sum _ { j = i + 1 } ^ { K } s \Bigl ( x _ { 0 } ^ { ( i ) } ( c ) , x _ { 0 } ^ { ( j ) } ( c ) \Bigr ) ,\tag{4}
$$

where $s ( \cdot , \cdot )$ is a perceptual similarity metric. The generation conditioned on c is considered collapsed $i f$

$$
\bar { s } _ { \theta } ( c , K ) \ge \tau _ { s } ,\tag{5}
$$

where $\tau _ { s }$ is the collapse detection threshold.

When the model is clear from context, we omit the subscript θ. For a text prompt $p ,$ we use ${ \bar { s } } ( p , K )$ as shorthand for $\bar { s } _ { \theta } ( E _ { \mathrm { t e x t } } ( p ) , K )$ .

In this work, we instantiate $s ( \cdot , \cdot )$ using the Self-Supervised Copy Detection (SSCD) score [32]. SSCD computes the cosine similarity between image representations extracted by a self-supervised model trained for image copy detection. Its sensitivity to visual and structural correspondence makes it suitable for measuring output consistency.

Collapsed generation represents an atypical yet repeatable behavior of the learned generation process. As shown in Fig. 2(a), prompts that induce collapse on SD 1.4 form a highconsistency distribution that is clearly separated from that of normal prompts. This separation makes the model’s intrinsic collapse behavior observable through cross-seed similarity, suggesting its potential as a behavioral fingerprint. We next investigate whether such collapse patterns are specific to the model that produces them.

![](images/b83a56fa4fb4ba6b22a15df58fd7313532fb94ecc6870595718f0ff91d4b4aac.jpg)  
Fig. 4: Overview of fingerprint construction. (a) Continuous-Embedding Synthesis. For pipeline-access verification, the owner optimizes conditioning embeddings on the source model. Starting from an initial embedding $c _ { i }$ and K independent noises $\{ \overset { \cdot } { x } _ { T } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ , truncated optimization minimizes the dispersion among early intermediate latents to produce $c _ { i } ^ { * }$ . Repeating this procedure yields the embedding fingerprint set $ { \mathcal { C } } _ { f } .$ . (b) Natural-Prompt Mining. For API-only verification, per-sample denoising loss is used to identify a low-loss candidate pool $\mathcal { P } _ { \mathrm { c a n d } }$ . The candidates are then evaluated by their cross-seed consistency on the source model, and the strongest prompts form the natural-prompt fingerprint set $\mathcal { P } _ { f }$

## B. Model Fingerprint via Collapsed Generation

For collapsed generation to serve as a model fingerprint, the collapse-inducing condition should be specific to the learned generation process. Given a source model, a fingerprint condition is expected to produce high cross-seed consistency on the source model while eliciting substantially weaker consistency on independently trained models. This property corresponds to the uniqueness requirement of model fingerprinting.

We first examine this property using naturally occurring collapsed prompts from SD 1.4 and SD 2.1. Specifically, we apply the collapsed prompts identified for each source model to both models and measure their cross-seed consistency. As shown in Fig. 2 (b), matched prompt–model pairs produce substantially higher consistency scores than mismatched pairs. The same prompt therefore induces different collapse responses across models, showing that collapsed generation reflects characteristics of the learned model rather than the input condition alone.

This model dependence arises from differences in the learned conditional generation mappings. Training data, model architecture, and optimization dynamics jointly shape how a conditioning input is mapped to the generative manifold [17], [27], [30]. Consequently, a condition located in a collapsed region of one model may produce diverse generations in another, as conceptualized in Fig. 3. These model-dependent collapsed regions provide the behavioral basis for constructing model fingerprints.

The comparison above involves T2I models that may differ in their training data or model configurations. For ownership verification, a stronger requirement is that collapsed fingerprints distinguish independently trained model instances even when these factors are controlled. We examine this requirement in Sec. V-A, where the model architecture, training data, and optimization procedure are held fixed, while only the parameter initialization is varied. The resulting models exhibit distinguishable collapsed fingerprints, showing that the learned collapse behavior can be specific to an individual training outcome.

This result is also relevant to models trained from shared public resources. Even when independently trained models use the same public dataset, different parameter initializations can lead to different learned parameters and memorization patterns, thereby producing distinct collapsed regions [15], [17]. The same mechanism provides a basis for differentiating modern large-scale T2I models, while practical differences in data curation, architecture, and training pipelines introduce further variation into their learned generation behavior. Such training-dependent collapse behavior provides the granularity required for ownership verification. Having established the observability and model specificity of collapsed generation, we next describe how the corresponding fingerprint conditions are constructed.

## C. Fingerprint Construction

Having established the model specificity of collapsed generation, the next challenge is to turn this behavioral signal into reproducible fingerprint conditions. The inputs that induce collapsed generations are not known a priori, and their admissible form is determined by the interface available during verification. With access to the inference pipeline, the continuous conditioning space can be actively searched to synthesize fingerprint embeddings. Under end-to-end T2I API-only access, collapse must instead be elicited by text prompts. We address these two settings through pipelineaccess continuous embedding synthesis and API-only naturalprompt mining, respectively.

Continuous-Embedding Synthesis. In checkpoint-level ownership verification, white-box access gives the verifier control over the suspect model’s inference pipeline, allowing continuous conditioning embeddings to be supplied directly. This makes continuous embeddings a viable representation for fingerprint conditions. The owner therefore constructs such fingerprints by optimizing embeddings on the source model to induce collapsed generation, as illustrated in Fig. 4 (a).

![](images/5eef0011763bb86cc5fb109fc87acaf598b3b681e33dd3c19bf4b669ac563f3f.jpg)  
Fig. 5: Visualization of predicted $x _ { 0 }$ along the denoising process for a representative collapsed and normal example in SD 1.4. For the collapsed example (top), stable and recognizable output content emerges within the first few denoising steps. For the normal example (bottom), the prediction remains ambiguous at the early stage and becomes progressively clearer as denoising proceeds.

We construct multiple embedding fingerprints for joint evaluation during verification. For the i-th fingerprint, we initialize $c _ { i } ~ = ~ E _ { \mathrm { t e x t } } ( p _ { i } )$ and sample $K$ independent initial noises $\{ x _ { T } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ . We then optimize $c _ { i }$ such that the corresponding sampling trajectories suppress the variation introduced by different initial noises and converge toward similar outputs.

A direct approach would maximize the final-image consistency $\bar { s } _ { \theta } ( c _ { i } , K )$ defined in Eq. 4. However, each optimization iteration would require differentiating through the complete denoising process, decoding all resulting latents into images, and evaluating their perceptual similarity. This makes direct optimization computationally and memory intensive, particularly when multiple trajectories are jointly optimized [36], [37].

In this work, our key observation is that the characteristic output content of collapsed generation becomes established early in the denoising process. As illustrated in Fig. 5, for a collapsed example, the predicted clean image rapidly develops a stable and recognizable structure within the first few denoising steps. In contrast, for a normal example, the prediction remains ambiguous at the early stage and becomes progressively clearer as denoising proceeds. This contrast suggests that the generation tendency underlying collapse is already reflected in early intermediate states. Motivated by this observation, we adopt truncated optimization, which minimizes the dispersion among early intermediate latents from independent initial noises as a surrogate for final-image consistency.

Let $x _ { T - t _ { \mathrm { t r u n c } } } ^ { ( k ) } ( c _ { i } )$ denote the intermediate latent obtained from $x _ { T } ^ { ( k ) }$ after the first $t _ { \mathrm { t r u n c } }$ denoising steps, and let

$$
\bar { x } _ { T - t _ { \mathrm { t r u n c } } } ( c _ { i } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } x _ { T - t _ { \mathrm { t r u n c } } } ^ { ( k ) } ( c _ { i } )\tag{6}
$$

![](images/354f3dea3728813d3d38456c5d72d0bd5f4d09bba397b96da9af7c6d77c590fe.jpg)  
Fig. 6: Cumulative distribution of training loss for collapsed and normal samples in SD 1.4. The loss is computed as the MSE between the ground-truth noise ϵ and the predicted noise ϵˆ at intermediate timesteps $( t \in [ 4 5 0 , 5 5 0 ] )$ [38]. Collapsed samples are predominantly concentrated in the lowloss region. Consequently, applying a low-loss threshold yields a significantly higher proportion of collapsed samples (e.g., a likelihood ratio of roughly 12.5 at a threshold of 0.02), enabling efficient extraction of fingerprint prompts.

denote their mean. We define the collapse loss as

$$
\mathcal { L } _ { \mathrm { C o l l a p s e } } ( c _ { i } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left. x _ { T - t _ { \mathrm { t r u n c } } } ^ { ( k ) } ( c _ { i } ) - \bar { x } _ { T - t _ { \mathrm { t r u n c } } } ( c _ { i } ) \right. _ { 2 } ^ { 2 } .\tag{7}
$$

Minimizing this objective drives the early denoising trajectories toward a shared latent region. Since gradients are propagated through only the first $t _ { \mathrm { t r u n c } }$ steps, the remaining denoising process, VAE decoding, and image-level similarity computation are excluded from the optimization loop, thereby reducing both computational and memory costs.

Repeating this procedure from M initial embeddings produces the fingerprint set

$$
\mathcal { C } _ { f } = \left\{ c _ { i } ^ { * } \mid c _ { i } ^ { * } = \arg \operatorname* { m i n } _ { c _ { i } } \mathcal { L } _ { \mathrm { C o l l a p s e } } ( c _ { i } ) , i = 1 , \ldots , M \right\} .\tag{8}
$$

After optimization, the owner evaluates each $c _ { i } ^ { * }$ through complete sampling on the source model and measures its final-image cross-seed consistency using Eq. 4. The detailed optimization procedure is provided in Appendix A.

Natural-Prompt Mining. When the suspect model is exposed only through an end-to-end T2I API, the continuous embeddings constructed above cannot be directly replayed. In black-box verification, fingerprint conditions must therefore be represented as text prompts. We further construct them from naturally occurring prompts associated with the source model’s training data, allowing the resulting fingerprint queries to retain the form of ordinary API inputs.

The main challenge is to efficiently identify prompts that induce collapsed generation among the large-scale training data of modern T2I models. A direct search would require generating K images for every candidate prompt and evaluating their cross-seed consistency. Applying this procedure to a large training collection would incur a substantial number of complete sampling runs.

We address this search cost by exploiting the relationship between collapsed generation and sample-level training loss [39]. As shown in Fig. 6, training samples associated with collapsed generation are substantially enriched in the low-loss region. The model owner can therefore record persample denoising losses during training and retain the prompts associated with low-loss samples as a candidate pool $\mathcal { P } _ { \mathrm { c a n d } }$ as illustrated in Fig. 4 (b).

Only this reduced candidate pool requires output-level evaluation. For each $p \in \mathcal { P } _ { \mathrm { c a n d } }$ , the owner generates K samples on the source model and computes the cross-seed consistency ${ \bar { s } } ( p , K )$ ). The owner selects the M prompts with the highest consistency scores to form $\mathcal { P } _ { f } ~ = ~ \{ p _ { i } ^ { * } \} _ { i = 1 } ^ { M }$ . The resulting prompts can subsequently be submitted to the suspect model through its standard text interface.

The two construction strategies yield interface-compatible fingerprint sets $\mathcal { C } _ { f }$ and $\mathcal { P } _ { f } ,$ , both of which are evaluated through the cross-seed consistency test described next.

## D. Fingerprint Verification

Observing a high consistency score on the suspect model is not by itself sufficient for ownership verification. Normal prompts may occasionally produce similar outputs, while an individual fingerprint condition may exhibit an atypical response. The main challenge is therefore to determine whether the collapse induced by the fingerprint set is statistically distinguishable from the source model’s normal generation behavior. We address this challenge through source-referenced statistical calibration and set-level aggregation.

The same verification principle applies to both embedding and text fingerprints. Let $G _ { \theta ^ { \prime } }$ denote the suspect model. For an embedding fingerprint $c _ { m } ^ { * } \in { \mathcal { C } } _ { f }$ , the verifier supplies the embedding directly to the suspect pipeline; for a naturalprompt fingerprint $p _ { m } ^ { * } \in \mathcal { P } _ { f }$ , the verifier submits the corresponding text prompt. In either case, K independent outputs are collected and their cross-seed consistency is computed using Eq. 4. The responses of the M fingerprints are then aggregated as

$$
\bar { s } _ { f } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \bar { s } _ { \theta ^ { \prime } } ( c _ { m } ^ { * } , K )\tag{9}
$$

for $\mathcal { C } _ { f }$ , or equivalently $\begin{array} { r } { \bar { s } _ { f } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \bar { s } _ { \theta ^ { \prime } } ( p _ { m } ^ { * } , K ) } \end{array}$ for $\mathcal { P } _ { f }$ .

Before deployment, the owner establishes a normalgeneration reference sample on the source model using a set of normal prompts, which are generated independently by GPT-4 in our implementation. Let $\{ u _ { n } = \bar { s } _ { \theta } ( p _ { n } , K ) \} _ { n = 1 } ^ { N }$ denote the resulting reference scores, and let $\mu _ { 0 }$ and $\sigma _ { 0 }$ be their sample mean and standard deviation. Treating the aggregate fingerprint response $\bar { s } _ { f }$ as a new observation relative to this reference sample, we perform a right-tailed predictive t-test:

$$
t _ { f } = \frac { \bar { s } _ { f } - \mu _ { 0 } } { \sigma _ { 0 } \sqrt { 1 + 1 / N } } , \qquad p _ { \mathrm { v a l } } = 1 - F _ { t _ { N - 1 } } ( t _ { f } ) ,\tag{10}
$$

where $F _ { t _ { \nu } }$ denotes the Student’s t CDF with ν degrees of freedom.

A small $p _ { \mathrm { v a l } }$ indicates that the aggregate fingerprint response lies in the upper tail of the source model’s normal-generation reference, providing evidence that the suspect model reproduces the source model’s collapsed fingerprint behavior. Given a decision threshold $\tau _ { f }$ , we accept the ownership claim when

$$
p _ { \mathrm { v a l } } < \tau _ { f } .\tag{11}
$$

The model specificity established above makes this decision selective across models: independently trained models are not expected to reproduce the same set-level response, providing the uniqueness required for ownership verification. Aggregating multiple fingerprint responses further reduces the influence of an anomalous individual condition. Moreover, the normalgeneration reference can be prepared before deployment and reused without requiring additional normal-prompt queries to the suspect model. The detailed procedure is provided in Appendix A.

For API-only verification, an additional practical challenge is to prevent the fingerprint queries from being readily distinguished from normal service interactions, i.e., stealthiness. Our natural-prompt fingerprints preserve the linguistic form of normal user inputs, avoiding the conspicuous token patterns introduced by adversarially optimized prompts, as evaluated in Sec. V-D. Beyond the appearance of individual prompts, stealthiness also depends on the pattern in which repeated queries are issued. The verification statistic requires K independent generations for each prompt, but these generations need not be obtained consecutively or within a single session. They may be collected over temporally separated interactions and, when appropriate, pooled across multiple verification clients before applying the same statistical test. Under the default configuration of $M = 4$ and $K = 4$ , this amounts to a total of 16 fingerprint queries. Together, the natural prompt form, modest query budget, and flexible collection schedule jointly support stealthy verification. We further evaluate robustness against query-time interventions in Sec. V-D.

## V. EXPERIMENTS

Following the principles in Sec. III-B, we organize our evaluation around four research questions.

RQ1: Uniqueness. Are collapsed-generation fingerprints specific to their source models and distinguishable from independently trained models?

RQ2: Robustness. Do the fingerprints remain verifiable after model modifications and adaptive query-time obfuscations?

RQ3: Stealthiness. In the API-only setting, do fingerprint queries resemble ordinary user prompts and remain difficult to detect?

RQ4: Efficiency. Can collapsed-generation fingerprints be constructed with practical computational overhead?

In the following, we first present a controlled preliminary study using conditional DDPMs to validate the uniqueness of collapsed-generation fingerprints. We then describe the common experimental setup for T2I models and evaluate fingerprint uniqueness and robustness under both white-box pipeline access and black-box API-only access. For the latter, we additionally assess the stealthiness of fingerprint queries.

![](images/bbc85f74164e9e4220788691359e3ade68e44c951f2b8fe157f739870b5274aa.jpg)

![](images/9ea24bfcaa59e1ffbefa36e79a5123f2efce55574414bb8587c17de5d9b1010f.jpg)  
(a) Confusion matrix  
(b) Representative generations  
Fig. 7: Controlled validation of fingerprint uniqueness across four independently trained conditional DDPMs. (a) Crossmodel verification matrix. Rows represent the source models of the fingerprint sets, columns represent the queried models, and lower $p _ { \mathrm { v a l } }$ indicate stronger matches. (b) Representative cross-seed generations. The left panel shows one fingerprint condition from each model evaluated on its corresponding source model. The right panel applies the first fingerprint condition from the left panel to all four models, producing consistent outputs on the matched source model and diverse outputs on the three mismatched models.

Finally, we analyze the computational cost of fingerprint construction.

## A. Preliminary Study: Model-Specific Collapse

To answer RQ1, we conduct a controlled study using independently trained conditional DDPMs on CIFAR-10 to examine whether collapsed-generation fingerprints remain specific to individual training runs.

Experimental Settings. We construct a conditional DDPM for CIFAR-10 in which each training image is assigned an instance-specific condition composed of its class label and image index. The class label and image index are encoded separately and then fused as the conditioning input to the U-Net. This conditioning design allows collapsed generation to be measured at the level of individual training samples.

We independently train four model instances with the same architecture, dataset, optimizer, and training schedule, using different random seeds for parameter initialization and training stochasticity. To accelerate the emergence of pronounced collapse cases, we select the same 500 CIFAR-10 training images as an upweighted subset for all four models and sample them approximately 10× more frequently than the remaining images. The identical exposure schedule enables a controlled comparison of collapse patterns across independent training runs. Each model is trained for 40,000 steps with a batch size of 128 and evaluated using the final checkpoint, the same DDPM sampler, a classifier-free guidance scale of 1.5, and four fixed sampling seeds shared across all models.

Fingerprint Construction and Verification. Following the low-loss mining strategy described in Sec. IV-C, we rank the training set for each model according to sample-level denoising loss during training and retain the low-loss conditions as fingerprint candidates. For each candidate, we generate four images and measure cross-seed diversity using the average pairwise $\ell _ { 2 }$ distance among the generated images, where a smaller distance indicates stronger collapsed generation. The four candidates with the lowest distances form the fingerprint set of each model.

To establish the normal-generation reference, we randomly sample 50 non-upweighted training conditions for each model and compute their cross-seed $\ell _ { 2 }$ distances. Following the setlevel verification procedure in Sec. IV-D, we evaluate each four-condition fingerprint set on all four models and obtain a $4 \times 4$ matrix of lower-tail $p _ { \mathrm { v a l } }$ scores.

Results. Fig. 7 (a) exhibits a clear diagonal separation, with matched fingerprint–model pairs yielding markedly lower $p _ { \mathrm { v a l } }$ scores than all mismatched pairs. This result indicates that the collapse-prone conditions identified from one training run are not reproduced with comparable strength by the other independently trained models. The qualitative examples in Fig. 7 (b) corroborate this distinction: each representative fingerprint produces consistent cross-seed generations on its source model, whereas the fingerprint selected from the first source model yields substantially more diverse outputs on the three mismatched models. Together, these results show that model-specific collapse patterns emerge across independent training runs under matched architectures, training data, upweighted samples, and optimization configurations, supporting the uniqueness of collapsed-generation fingerprints.

## B. Experimental Setup

Having established model-specific collapse in the controlled CIFAR-10 setting, we next evaluate the proposed fingerprinting framework on representative T2I diffusion models. The following experiments use SSCD-based cross-seed similarity and the T2I fingerprint construction and verification procedures described in Secs. IV-C and IV-D. We describe the experimental setup below.

Models. We consider both white-box pipeline-access and black-box API-only verification settings. Under pipeline access, we evaluate Stable Diffusion 1.4 (SD 1.4) [1], DeciDiffusion v1.0 (Deci) [3], PixArt-α (PixArt) [5], and Stable Diffusion 3 (SD 3) [40]. These models span both U-Net-based and diffusion Transformer (DiT)-based architectures, with PixArt and SD 3 representing the latter. In particular, SD 3 combines a flow-matching formulation with training-data deduplication, providing a challenging case for assessing whether collapsedgeneration fingerprints remain effective when naturally occurring collapse may be less prevalent. Under API-only access, we evaluate SD 1.4 and Stable Diffusion 2.1 (SD 2.1) [2], for which prior work provides public prompt collections that support reproducible candidate construction. All pretrained checkpoints are obtained from Hugging Face [41], with their exact versions reported in Appendix B.

Fingerprint Construction. For pipeline-access verification, we construct the fingerprint set $\mathcal { C } _ { f }$ by optimizing continuous conditioning embeddings independently on each source model. For API-only verification, we draw candidates from the public prompt collections released by [28] for studying black-box training-data extraction. Each collection contains 500 prompts for which the corresponding model was shown to reproduce training examples. We use these prompts as an enriched candidate pool for collapsed generation, screen them according to their cross-seed generation consistency, and select the top-M prompts to form $\mathcal { P } _ { f }$ . Unless otherwise specified, both $\mathcal { C } _ { f }$ and $\mathcal { P } _ { f }$ contain $M = 4$ fingerprint conditions, and each condition is evaluated using $K = 4$ generations.

Inference Configuration. For SD 1.4, SD 2.1, and Deci, we use DDIM sampling with 50 denoising steps and a classifierfree guidance scale of 7.5. PixArt uses its default DPM-Solver scheduler with 20 denoising steps, while SD 3 uses its default 28-step sampler and a guidance scale of 7.0. Four fixed sampling seeds are shared across all applicable models and conditions.

Obfuscations. To evaluate robustness against model-level modifications, we consider pruning, quantization, and finetuning. For pruning, we remove 10%, 20%, and 30% of the smallest-magnitude weights from the model backbone, including the U-Net or Transformer blocks. For quantization, we evaluate bfloat16, int8, and fp4 parameter representations. For fine-tuning, we consider three representative derivatives of SD 1.4: Stable Diffusion 1.5 (SD 1.5) [1], Deliberate v4 [42], and Realistic Vision $\mathrm { { v } } 2 . 0$ [43]. In the API-only setting, we additionally evaluate four adaptive query-time obfuscations: GPT rewriting, Random Token Addition (RTA) [17], Embedding Optimization (EO) [18], and Sharpness-Aware Initialization for Latent Diffusion (SAIL) [29], whose configurations are detailed in Appendix B.

Baselines. Under pipeline access, we compare our method with FingerInv [13], which derives a model-specific latent code through DDPM inversion of a reference QR-code image and verifies model identity through its recovery on the queried model. Under API-only access, we compare with TVN [12], which searches for non-transferable adversarial prompt suffixes that induce target-specific semantic deviations. Following its original configuration, TVN optimizes from 10 initial prompts for each target model and retains the best-performing prompt as its fingerprint. Detailed baseline implementations are provided in Appendix B.

Metrics. For our method, we compute SSCD-based crossseed generation consistency and derive the set-level $p _ { \mathrm { v a l } }$ defined in Sec. IV-D. For each source model, we construct the normal-generation reference using $ { N _ { \mathrm { ~ ~ \scriptsize ~ = ~ } } } 5 0$ prompts independently generated by GPT-4. We use $\tau _ { f } = 1 0 ^ { - 4 }$ as the verification threshold throughout the T2I experiments, with $p _ { \mathrm { v a l } } < \tau _ { f }$ indicating a fingerprint match. TVN measures the CLIP similarity between the generated image and its input prompt, whereas FingerInv measures the $\ell _ { 2 }$ distance between the recovered and reference QR-code images. For all three methods, lower reported values provide stronger evidence of a match under their respective verification rules. For APIonly stealthiness, we additionally measure prompt perplexity using Mistral-7B [44], where lower perplexity indicates a more natural prompt. Additional optimization hyperparameters and computational resources are reported in Appendix B.

## C. White-box Pipeline-access Verification

Optimization Details. In the white-box setting, we induce collapsed generations by optimizing the prompt embedding. In the main experiments, we initialize the optimization from text embeddings of candidate prompts for SD v1 models. The initialization analysis in Appendix C further shows that both low-diversity and random prompt initializations yield verifiable fingerprints. Further details and hyper-parameters are available in Appendix B.

![](images/c0b7cf572ea22aaec08e1d9589341b3ccccedc80949274378e3fe9aa1ee4305b.jpg)

![](images/ad5953912bc4513776440e03dbfb5fa108474689654db23d4ffcba882b35e288.jpg)  
(a) FingerInv [13]  
(b) Ours

Fig. 8: Confusion matrices (white-box). Blue indicates higher likelihood that the suspect model is identified as the corresponding source model. Due to architectural differences, our fingerprints of PixArt and SD 3 cannot be used with other models, resulting in empty cells.  
![](images/38e6a19d739039e8ee5e00af439f036380a144e436e9db600c1146e8616326f5.jpg)  
Fig. 9: Case study for robustness analysis (white-box). FingerInv demonstrates degradation in the quality of generated QR codes under strong obfuscations, which may hinder verification. In contrast, our fingerprint relies on the consistency within the generated batch, maintaining strong robustness.

Uniqueness Analysis. Fig. 8 reports the cross-model verification results of our method and FingerInv under pipeline access. Note that FingerInv is formulated upon DDPM inversion trajectories, making it incompatible with the underlying ODE mechanisms of flow-matching models like SD 3.

The results show that both our method and FingerInv exhibit excellent discriminative ability, with significantly low metric values (indicated by blue cells) appearing only along the main diagonal. Cross-model verification is performed only between models with compatible conditioning interfaces. Because PixArt and SD 3 use embedding spaces incompatible with the other evaluated pipelines, the corresponding crossarchitecture entries are left undefined. Under pipeline access, these interface incompatibilities can be identified before statistical fingerprint verification.

Robustness Analysis. Both our method and FingerInv demonstrate strong verification performance under obfuscation attacks. The complete quantitative results are reported in $\mathsf { A p - }$ pendix C, while Fig. 9 provides representative qualitative examples.

![](images/036e40b635f83e8f71035962e62cc639a5e19c7cbad7847443ae14eca943f575.jpg)  
(a) TVN [12]

![](images/69189a5f6dbdaa4b73b676c213d819ba5db62c373d31fb0337de6f554ce351bb.jpg)  
(b) Ours  
Fig. 10: Confusion matrices (black-box). Scores are displayed on a logarithmic scale, with lower values indicating stronger matches. Our method (b) shows clear unique identification for each suspect model, while TVN (a) exhibits significant confusion among different models.

TABLE II: Robustness (black-box, quantization and pruning). Green and red cells indicate successful and failed verification, respectively. † indicates false alarm.
<table><tr><td rowspan="2">Obfuscation</td><td colspan="2">TVN [12] (CLIP score)</td><td colspan="2">Ours  $( p _ { \mathrm { v a l } } )$ </td></tr><tr><td>SD 1.4</td><td>SD 2.1</td><td>SD 1.4</td><td>SD 2.1</td></tr><tr><td>Quant. BF16</td><td>21.8</td><td>20.0†</td><td> $9 . 7 7 \times 1 0 ^ { - 2 1 }$ </td><td> $3 . 4 4 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Quant. INT8</td><td>21.7</td><td>21.4†</td><td> $1 . 0 5 \times 1 0 ^ { - 2 0 }$ </td><td> $2 . 7 4 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Quant. FP4</td><td>20.5</td><td>19.8†</td><td> $2 . 0 5 \times 1 0 ^ { - 2 0 }$ </td><td> $1 . 4 9 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Prune 10%</td><td>22.0†</td><td>20.6†</td><td> $1 . 1 9 \times 1 0 ^ { - 2 0 }$ </td><td> $5 . 0 0 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Prune 20%</td><td>20.5†</td><td>19.8†</td><td> $8 . 0 8 \times 1 0 ^ { - 2 0 }$ </td><td> $6 . 3 5 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Prune 30%</td><td>24.3</td><td>18.9†</td><td> $2 . 9 8 \times 1 0 ^ { - 1 6 }$ </td><td> $6 . 2 6 \times 1 0 ^ { - 1 0 }$ </td></tr></table>

As illustrated in the figure, FingerInv and our method exhibit different degradation behaviors. FingerInv relies on recovering a specific reference pattern, and its generated QR codes exhibit visible degradation as model obfuscation becomes stronger. In contrast, our verification signal is based on cross-seed consistency rather than the reconstruction quality of a prescribed image, and the generated batches retain consistent content under the same obfuscations. This highlights the stability of the collapse phenomenon induced by our optimized embeddings, ensuring robust performance even under challenging conditions.

## D. Black-box API-only Verification

We next consider the more restrictive API-only setting, where fingerprint conditions must be submitted as text prompts and verification relies solely on the returned images.

Uniqueness Analysis. We evaluate the distinctiveness of our method compared to TVN in Fig. 10. Pairwise crossmodel verification should yield low scores only for matched fingerprint–model pairs along the main diagonal. The results reveal that TVN suffers from a false positive (top-right cell), where SD 2.1 is misidentified as SD 1.4. In contrast, our method exhibits clearer diagonal separation, thereby ensuring reliable model identification.

Robustness Analysis. Before deployment, adversaries may apply obfuscation techniques, such as pruning, quantization, or fine-tuning, to modify model parameters and evade fingerprint verification. In Table II and Table III, we present the verification results under these attacks. The results show that TVN exhibits poor robustness, frequently misidentifying SD 2.1 as SD 1.4 across all attack types. Moreover, pruning SD 1.4 leads to the same false positive, which only ceases at 30% pruning. This is likely because the model’s ability to comprehend prompts and generate corresponding images is severely degraded at higher pruning levels. In contrast, our method remains effective across all attack types and intensities. This demonstrates that the proposed collapsed prompts induce a highly stable and consistent generation, where minor parameter modifications are insufficient to disrupt the established generation behavior. These findings validate the robustness of our approach under realistic adversarial conditions.

TABLE III: Robustness (black-box, fine-tuning). Green and red cells indicate successful and failed verification, respectively. † indicates false alarm.
<table><tr><td>Model</td><td>TVN [12] (CLIP score)</td><td> $\mathbf { O u r s } \ ( p _ { \mathrm { v a l } } )$ </td></tr><tr><td>SD 1.5</td><td>21.4</td><td> $1 . 5 5 \times 1 0 ^ { - 2 0 }$ </td></tr><tr><td>Deliberate</td><td>17.8†</td><td> $6 . 5 0 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Realistic</td><td>21.9</td><td> $5 . 0 0 \times 1 0 ^ { - 2 1 }$ </td></tr></table>

TABLE IV: Robustness (black-box, adaptive obfuscations).
<table><tr><td></td><td> $p _ { \mathrm { v a l } } \ ( \downarrow )$ </td><td>CLIP Score (↑)</td><td>Time (s) (↓)</td><td>Success?</td></tr><tr><td>No Obfuscation</td><td> $8 . 6 3 \times 1 0 ^ { - 2 1 }$ </td><td>31.41</td><td>1.744</td><td></td></tr><tr><td>GPT Rewriting</td><td> $1 . 3 4 \times 1 0 ^ { - 1 5 }$ </td><td>31.75</td><td>2.590</td><td></td></tr><tr><td>RTA [17]</td><td> $6 . 9 3 \times 1 0 ^ { - 1 6 }$ </td><td>29.00</td><td>1.753</td><td></td></tr><tr><td>EO [18]</td><td> $2 . 3 3 \times 1 0 ^ { - 6 }$ </td><td>28.21</td><td>2.466</td><td></td></tr><tr><td>SAIL [29]</td><td> $2 . 2 4 \times 1 0 ^ { - 6 }$ </td><td>30.25</td><td>3.426</td><td></td></tr></table>

Adaptive Obfuscations. We further consider scenarios where an adversary is aware of our fingerprinting method and attempts to bypass verification using adaptive attacks. Assuming that the verification query is detected (which is non-trivial as discussed in Sec. IV-D), the adversary may try to disrupt generation consistency. To evaluate this, we test four collapse-mitigation strategies: GPT rewriting, Random Token Addition (RTA) [17], Embedding Optimization (EO) [18] and Sharpness-Aware Initialization for Latent Diffusion (SAIL) [29] on SD 1.4. RTA inserts R random tokens at arbitrary positions within the prompt, while EO and SAIL reduce the classifier-free guidance magnitude at the initial denoising step through prompt embedding or latent optimization. As shown in Table IV, our fingerprints remain effective against all obfuscation strategies, with $p _ { \mathrm { v a l } }$ consistently staying well below the threshold. Given the difficulty of detecting verification queries, adversaries are forced to apply these mitigations routinely. However, some interventions reduce semantic quality, while all introduce additional processing latency. This demonstrates that our method has a stable collapse effect, ensuring robustness even against adaptive obfuscations.

Stealthiness Analysis. To evade verification, an adversary may attempt to detect fingerprint prompts. A common detection approach involves analyzing prompt perplexity, as unnatural or garbled text often indicates abnormal queries. We evaluate stealthiness by measuring the average perplexity of the fingerprint prompts. Results in Table V show that TVN’s prompts exhibit extremely high perplexity (all exceeding 200) due to their suffix-based design. As an adversary, one can further locate and remove the suspicious suffixes to evade verification. In contrast, our fingerprints retain the form of naturally occurring, human-readable prompts, resulting in substantially lower perplexity.

<table><tr><td>Method</td><td>SD 1.4</td><td>SD 2.1</td></tr><tr><td>TVN [12]</td><td>294.96</td><td>243.83</td></tr><tr><td>Ours</td><td>41.85</td><td>51.32</td></tr></table>

TABLE VI: Optimization time (s/prompt).
<table><tr><td>Method</td><td>TVN [12]</td><td>FingerInv [13]</td><td>Ours</td></tr><tr><td>Time (↓)</td><td>369.57</td><td>24.91</td><td>25.03</td></tr></table>

## E. Efficiency

To answer RQ4, we examine the computational cost of fingerprint construction. Table VI reports the per-fingerprint runtime of optimization-based construction procedures under their respective access settings. Under pipeline access, our truncated embedding synthesis requires 25.03 seconds, comparable to the 24.91 seconds required by FingerInv. For reference, TVN requires 369.57 seconds to construct one API-compatible fingerprint through iterative prompt-suffix optimization. These results show that our active fingerprint synthesis incurs practical construction overhead comparable to the pipeline-access baseline.

## VI. CONCLUSION

In this work, we formulate collapsed generation, characterized by unusually high cross-seed consistency under particular input conditions, as a model-intrinsic behavioral signature for non-invasive ownership verification of T2I diffusion models. Under pipeline access, we synthesize modelspecific continuous embedding fingerprints through truncated optimization. Under API-only access, we construct naturalprompt fingerprints by screening collapse-prone candidates. Both fingerprint representations are evaluated through a unified source-referenced statistical test that determines whether a suspect model reproduces the characteristic collapse response of the source model.

A controlled study of independently trained conditional DDPMs and evaluations across representative U-Net-, DiT-, and flow-matching-based T2I models show clear separation between matched and mismatched fingerprint–model pairs, supporting fingerprint uniqueness. The fingerprints remain verifiable under most evaluated fine-tuned derivatives, model modifications, and adaptive query-time interventions, demonstrating robustness across the considered conditions. In the API-only setting, the natural fingerprint prompts remain human-readable and exhibit substantially lower perplexity than adversarially optimized prompt suffixes, supporting stealthier service-level verification. Meanwhile, truncated embedding synthesis maintains construction cost comparable to the pipeline-access baseline. Together, these results establish collapsed generation as a practical source of intrinsic behavioral evidence for diffusion model ownership verification across checkpoint- and service-level disputes.

[1] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[2] S. AI, “Stable diffusion v2.1 and dreamstudio updates,” Blog post / Release notes, Dec 2022, stable Diffusion v2.1 release (Dec 7, 2022). [Online]. Available: https: //stability.ai/news/stablediffusion2-1-release7-dec-2022

[3] D. R. Team, “Decidiffusion 1.0,” 2023. [Online]. Available: [https://huggingface.co/deci/decidiffusion-v1-0](https: //huggingface.co/deci/decidiffusion-v1-0)

[4] Z. Li, J. Zhang, Q. Lin, J. Xiong, Y. Long, X. Deng, Y. Zhang, X. Liu, M. Huang, Z. Xiao, D. Chen, J. He, J. Li, W. Li, C. Zhang, R. Quan, J. Lu, J. Huang, X. Yuan, X. Zheng, Y. Li, J. Zhang, C. Zhang, M. Chen, J. Liu, Z. Fang, W. Wang, J. Xue, Y. Tao, J. Zhu, K. Liu, S. Lin, Y. Sun, Y. Li, D. Wang, M. Chen, Z. Hu, X. Xiao, Y. Chen, Y. Liu, W. Liu, D. Wang, Y. Yang, J. Jiang, and Q. Lu, “Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding,” 2024. [Online]. Available: https://arxiv.org/abs/2405.08748

[5] J. Chen, J. YU, C. GE, L. Yao, E. Xie, Z. Wang, J. Kwok, P. Luo, H. Lu, and Z. Li, “Pixart-\$\alpha\$: Fast training of diffusion transformer for photorealistic text-to-image synthesis,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=eAKmQPe3m1

[6] W. Jiang, H. Li, G. Xu, T. Zhang, and R. Lu, “A comprehensive defense framework against model extraction attacks,” IEEE Transactions on Dependable and Secure Computing, vol. 21, no. 2, pp. 685–700, 2023.

[7] T. Miura, T. Shibahara, and N. Yanai, “Megex: Data-free model extraction attack against gradient-based explainable ai,” in Proceedings of the 2nd ACM Workshop on Secure and Trustworthy Deep Learning Systems, 2024, pp. 56–66.

[8] Y. Zhao, T. Pang, C. Du, X. Yang, N.-M. Cheung, and M. Lin, “A Recipe for Watermarking Diffusion Models,” Oct. 2023. [Online]. Available: http://arxiv.org/abs/2303.10137

[9] P. Fernandez, G. Couairon, H. Jegou, M. Douze, and T. Furon,´ “The stable signature: Rooting watermarks in latent diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 22 466–22 477. [Online]. Available: http://openaccess.thecvf.com/content/ICCV2023/html/Fernandez The Stable Signature Rooting Watermarks in Latent Diffusion Models ICCV 2023 paper.html

[10] Y. Liu, Z. Li, M. Backes, Y. Shen, and Y. Zhang, “Watermarking Diffusion Model,” May 2023. [Online]. Available: http://arxiv.org/abs/ 2305.12502

[11] B. Zeng, L. Wang, Y. Hu, Y. Xu, C. Zhou, X. Wang, Y. Yu, and Z. Lin, “Huref: Human-readable fingerprint for large language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 126 332– 126 362, 2024.

[12] J. Guo, W. Jiang, R. Zhang, G. Lu, and H. Li, “One Prompt to Verify Your Models: Black-Box Text-to-Image Models Verification via Non-Transferable Adversarial Attacks,” May 2025. [Online]. Available: http://arxiv.org/abs/2410.22725

[13] H. Teng, Y. Quan, C. Wang, J. Huang, and H. Ji, “Fingerprinting Denoising Diffusion Probabilistic Models,” 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[14] Z. Li, H. Qu, J. Kuen, J. Gu, Q. Ke, J. Liu, and H. Rahmani, “Diffip: Representation fingerprints for robust ip protection of diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 17 035–17 045.

[15] N. Carlini, J. Hayes, M. Nasr, M. Jagielski, V. Sehwag, F. Tramer, B. Balle, D. Ippolito, and E. Wallace, “Extracting training data from diffusion models,” in 32nd USENIX Security Symposium (USENIX Security 23), 2023, pp. 5253–5270. [Online]. Available: https://www.usenix.org/conference/usenixsecurity23/presentation/carlini

[16] D. Hintersdorf, L. Struppek, K. Kersting, A. Dziedzic, and F. Boenisch, “Finding nemo: Localizing neurons responsible for memorization in diffusion models,” Advances in Neural Information Processing Systems, vol. 37, pp. 88 236–88 278, 2024. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2024/hash/ a102dd5931da01e1b40205490513304c-Abstract-Conference.html

[17] G. Somepalli, V. Singla, M. Goldblum, J. Geiping, and T. Goldstein, “Understanding and Mitigating Copying in Diffusion Models,” Advances in Neural Information Processing Systems, 2023.

[18] Y. Wen, Y. Liu, C. Chen, and L. Lyu, “Detecting, Explaining, and Mitigating Memorization in Diffusion Models,” The Twelfth International Conference on Learning Representations, 2024.

[19] Y. Uchida, Y. Nagai, S. Sakazawa, and S. Satoh, “Embedding watermarks into deep neural networks,” in Proceedings of the 2017 ACM on International Conference on Multimedia Retrieval, ser. ICMR ’17. ACM, Jun. 2017, p. 269–277. [Online]. Available: http://dx.doi.org/10.1145/3078971.3078974

[20] Y. Adi, C. Baum, M. Cisse, B. Pinkas, and J. Keshet, “Turning your weakness into a strength: Watermarking deep neural networks by backdooring,” 2018. [Online]. Available: https://arxiv.org/abs/1802. 04633

[21] B. D. Rouhani, H. Chen, and F. Koushanfar, “Deepsigns: A generic watermarking framework for ip protection of deep learning models,” 2018. [Online]. Available: https://arxiv.org/abs/1804.00750

[22] N. Lukas, Y. Zhang, and F. Kerschbaum, “Deep neural network fingerprinting by conferrable adversarial examples,” arXiv preprint arXiv:1912.00888, 2019.

[23] X. Cao, J. Jia, and N. Z. Gong, “Ipguard: Protecting intellectual property of deep neural networks via fingerprinting the classification boundary,” in Proceedings of the 2021 ACM asia conference on computer and communications security, 2021, pp. 14–25.

[24] Y. Zheng, S. Wang, and C.-H. Chang, “A dnn fingerprint for nonrepudiable model ownership identification and piracy detection,” IEEE Transactions on Information Forensics and Security, vol. 17, pp. 2977– 2989, 2022.

[25] R. Min, S. Li, H. Chen, and M. Cheng, “A watermark-conditioned diffusion model for ip protection,” 2024. [Online]. Available: https://arxiv.org/abs/2403.10893

[26] Z. Wang, J. Guo, J. Zhu, Y. Li, H. Huang, M. Chen, and Z. Tu, “Sleepermark: Towards robust watermark against finetuning text-to-image diffusion models,” 2025. [Online]. Available: https://arxiv.org/abs/2412.04852

[27] G. Somepalli, V. Singla, M. Goldblum, J. Geiping, and T. Goldstein, “Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2023, pp. 6048–6058. [Online]. Available: https://ieeexplore.ieee.org/document/10203298

[28] R. Webster, “A Reproducible Extraction of Training Images from Diffusion Models,” May 2023. [Online]. Available: http://arxiv.org/abs/ 2305.08694

[29] D. Jeon, D. Kim, and A. No, “Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes,” Proceedings ofthe 42nd International Conference on Machine Learning, 2025.

[30] B. L. Ross, H. Kamkari, T. Wu, R. Hosseinzadeh, Z. Liu, G. Stein, J. C. Cresswell, and G. Loaiza-Ganem, “A geometric framework for understanding memorization in generative models,” in The Thirteenth International Conference on Learning Representations, 2025.

[31] R. Chavhan, O. Bohdal, Y. Zong, D. Li, and T. Hospedales, “Memorization is localized within a small subspace in diffusion models,” in International Conference on Machine Learning (ICML)-Workshop on Generative AI and Law, 2024.

[32] E. Pizzi, S. D. Roy, S. N. Ravindra, P. Goyal, and M. Douze, “A selfsupervised descriptor for image copy detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 14 532–14 542.

[33] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” 2020. [Online]. Available: https://arxiv.org/abs/2006.11239

[34] Z. Zhang, M. Li, and J. Yu, “On the convergence and mode collapse of gan,” in SIGGRAPH Asia 2018 Technical Briefs, 2018, pp. 1–4.

[35] Z. Miao, J. Wang, Z. Wang, Z. Yang, L. Wang, Q. Qiu, and Z. Liu, “Training diffusion models towards diverse image generation with reinforcement learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 10 844–10 853.

[36] Z. Chen, B. Li, S. Wu, K. Jiang, S. Ding, and W. Zhang, “Contentbased unrestricted adversarial attack,” Advances in Neural Information Processing Systems, vol. 36, pp. 51 719–51 733, 2023.

[37] Y. Lin, J. Zhang, Y. Chen, and H. Li, “Sd-nae: Generating natural adversarial examples with stable diffusion,” arXiv preprint arXiv:2311.12981, 2023.

[38] S. Zhai, H. Chen, Y. Dong, J. Li, Q. Shen, Y. Gao, H. Su, and Y. Liu, “Membership inference on text-to-image diffusion models via conditional likelihood discrepancy,” Advances in Neural Information Processing Systems, vol. 37, pp. 74 122–74 146, 2024.

[39] T. Bonnaire, R. Urfin, G. Biroli, and M. Mezard, “Why diffusion models don’t memorize: The role of implicit dynamical regularization in

training,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[40] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Muller, H. Saini,¨ Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in Forty-first international conference on machine learning, 2024.

[41] “Hugging Face – The AI community building the future.” [Online]. Available: https://huggingface.co/

[42] XpucT, “Deliberate v4,” 2023. [Online]. Available: [https://huggingface. co/XpucT/Deliberate](https://huggingface.co/XpucT/Deliberate)

[43] SG161222, “Realistic vision v2.0,” 2023. [Online]. Available: [https://huggingface.co/SG161222/Realistic Vision V2.0](https: //huggingface.co/SG161222/Realistic Vision V2.0)

[44] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed, “Mistral 7b,” 2023. [Online]. Available: https://arxiv.org/abs/2310.06825

[45] Y. Wen, N. Jain, J. Kirchenbauer, M. Goldblum, J. Geiping, and T. Goldstein, “Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery,” Advances in Neural Information Processing Systems, vol. 36, pp. 51 008–51 025, 2023.

[46] A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, and M. Fredrikson, “Universal and transferable adversarial attacks on aligned language models,” arXiv preprint arXiv:2307.15043, 2023.

APPENDIX A Algorithm 2 Fingerprint Verification   
ALGORITHMS Input: Fingerprint set $\mathcal { F } ~ = ~ \{ f _ { m } \} _ { m = 1 } ^ { M }$ , where $\mathcal { F }$ is either   
$\mathcal { C } _ { f }$ or $\mathcal { P } _ { f } ;$ suspect model $G _ { \theta ^ { \prime } } ;$ source-model reference   
Algorithm 1 Continuous-Embedding Fingerprint Synthesis scores $\mathcal { U } _ { 0 } = \{ u _ { n } \} _ { n = 1 } ^ { N } ;$ number of generations $K ;$ decision   
Input: Initial prompt set $\mathcal { P } _ { \mathrm { i n i t } } = \{ p _ { i } \} _ { i = 1 } ^ { M }$ , source model $G _ { \theta } ,$ threshold $\tau _ { f }$   
text encoder $E _ { \mathrm { t e x t } } .$ number of noise samples $K ,$ number of Output: Verification decision d and statistical evidence $p _ { \mathrm { v a l } }$   
truncated denoising steps $t _ { \mathrm { t r u n c } } ,$ number of optimization 1: $\begin{array} { r } { \bar { \mu } _ { 0 }  \frac { 1 } { N } \sum _ { n = 1 } ^ { N } u _ { n } } \end{array}$   
iterations $I ,$ learning rate η 2: $\begin{array} { r } { \sigma _ { 0 }  \sqrt { \frac { 1 } { N - 1 } \sum _ { n = 1 } ^ { N } ( u _ { n } - \mu _ { 0 } ) ^ { 2 } } } \end{array}$   
Output: Optimized fingerprint embedding set $\mathcal { C } _ { f } = \{ c _ { i } ^ { * } \} _ { i = 1 } ^ { M }$ 3: $S _ { f } \gets \dot { \emptyset }$   
1: $\bar { \boldsymbol { \mathcal { C } } } _ { f } \gets \bar { \boldsymbol { \emptyset } }$ 4: for $m = 1 \ \mathrm { t o } \ M$ do   
2: for $i = 1$ to M do 5: Obtain $K$ independent outputs $\{ x _ { 0 } ^ { ( k ) } ( f _ { m } ) \} _ { k = 1 } ^ { K }$ from   
3: $c _ { i } ^ { ( 0 ) } \gets E _ { \mathrm { t e x t } } ( p _ { i } )$ ▷ Initialize the i-th embedding $G _ { \theta ^ { \prime } }$ conditioned on $f _ { m }$ $\triangleright \ f _ { m }$ is supplied as an   
4: Sample independent noises $\{ x _ { T } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ embedding or a text prompt   
5: for $r = 0$ to $I - 1$ do 6: Compute the cross-seed consistency $\bar { s } _ { \theta ^ { \prime } } ( f _ { m } , K )$ using   
6: Run the first $t _ { \mathrm { t r u n c } }$ denoising steps with $c _ { i } ^ { ( r ) }$ to $\operatorname { E q . }$ 4   
obtain $\{ x _ { T - t _ { \mathrm { t r u n c } } } ^ { ( k ) } ( c _ { i } ^ { ( r ) } ) \} _ { k = 1 } ^ { K }$ 7: $S _ { f } \gets S _ { f } \cup \{ \bar { s } _ { \theta ^ { \prime } } ( f _ { m } , K ) \}$   
7: $\begin{array} { r } { \bar { x } _ { T - t _ { \mathrm { t r u n c } } } ( c _ { i } ^ { ( r ) } ) \longleftarrow \frac { 1 } { K } \sum _ { k = 1 } ^ { K } x _ { T - t _ { \mathrm { t r u n c } } } ^ { ( k ) } ( c _ { i } ^ { ( r ) } ) } \end{array}$ 8: end for   
8: $\mathcal { L } _ { \mathrm { C o l l a p s e } } ( c _ { i } ^ { ( r ) } )$ ← 9: $\begin{array} { r } { \bar { s } _ { f }  \frac { 1 } { M } \sum _ { s \in S _ { f } } s } \end{array}$   
<sup>1</sup><sub>K</sub> $\begin{array} { r l } { \mathrm { ~ } } & { { } \sum _ { k = 1 } ^ { K } \left\| x _ { T - t _ { \mathrm { t r u n c } } } ^ { ( k ) } ( c _ { i } ^ { ( r ) } ) - \bar { x } _ { T - t _ { \mathrm { t r u n c } } } ( c _ { i } ^ { ( r ) } ) \right\| _ { \gamma } ^ { 2 } } \end{array}$ 10: $\begin{array} { r } { t _ { f } \gets \frac { \bar { s } _ { f } - \mu _ { 0 } } { \sigma _ { 0 } \sqrt { 1 + 1 / N } } } \end{array}$   
9: $c _ { i } ^ { ( \ddot { r } + 1 ) }  c _ { i } ^ { ( r ) } - \eta \nabla _ { c _ { i } ^ { ( r ) } } \mathcal { L } _ { \mathrm { C o l l a p s e } } ( c _ { i } ^ { ( r ) ^ { \ast } } )$ 11: 12: if $\begin{array} { r l } { p _ { \mathrm { v a l } }  \bar { 1 } ^ { \check { } } - \check { F _ { t _ { N - 1 } } } ( t _ { f } ) } & { { } \triangleright F _ { t _ { N - 1 } } } \end{array}$ $p _ { \mathrm { v a l } } < \tau _ { f }$ then is the Student’s t CDF   
10: end for 13: $d  \mathtt { T r u e } \ \triangleright$ The suspect model matches the source   
11: $c _ { i } ^ { * } \gets c _ { i } ^ { ( I ) }$ fingerprint   
12: $\bar { \mathcal { C } } _ { f }  \dot { \mathcal { C } } _ { f } \cup \{ \boldsymbol { c } _ { i } ^ { * } \}$ 14: else   
13: end for 15: $d  \mathrm { F a l } s \mathrm { e }$ ▷ The fingerprint match is not   
14: return $\mathcal { C } _ { f }$ established   
16: end if   
17: return $( d , p _ { \mathrm { v a l } } )$

## APPENDIX B EXPERIMENT SETTINGS

## A. Datasets

• Black-box prompts: For SD 1.4 and SD 2.1, we use 500 prompts each from [28]. From these collections we pick prompts that produce highly collapsed generations to form the fingerprint set $\mathcal { P } _ { f }$ , following the filtering procedure described in Sec. IV-C.

• White-box prompts: For SD 1.4, we initialize with the template verbatim prompts from [28]. For Deci and PixArt, we first apply the black-box prompt extraction procedure to the 1, 000 prompts to identify candidates exhibiting collapsed generations, then sample several of these candidates as initializations. We also evaluate that using low-diversity prompts and random initialization can also converge to effective fingerprints, as shown in Appendix D.

• GPT prompts: For Fig. 2 (a) and normal-generation modeling in the main experiments, we use GPT-4 to generate 50 prompts in plain text.

## B. Models

All pretrained diffusion model checkpoints are obtained from HuggingFace [41], including:

• Target models: SD 1.4 (CompVis/ stable-diffusion-v1-4), SD 2.1 (stabilityai/ stable-diffusion-2-1), Deci (Deci/ DeciDiffusion-v1-0), SD 3 (stabilityai/ stable-diffusion-3-medium-diffusers), and PixArt (PixArt-alpha/PixArt-XL-2-512x512).

• Fine-tune models: SD 1.5 (runwayml/ stable-diffusion-v1-5), Deliberate (XpucT/Deliberate), and Realistic (SG161222/ Realist\_Vision\_V2.0).

## C. Implementation Details

## Baselines.

• FingerInv [13]: We adopt the pipeline from the official open-source implementation provided by [13]. Specifically, a QR code image of size $3 2 \times 3 2$ is used to obtain a modelspecific DDPM latent code via FingerInv. This latent code is then applied to a suspect model to perform denoising. We subsequently examine whether the final generated image can still be successfully scanned. The quantitative measure is the L2 distance between the original and recovered QR codes. Note that this method is not applicable to flow-based models such as SD 3, which do not utilize DDPM-based sampling.

• TVN [12]: This method identifies non-transferable adversarial prompts as fingerprints. Following the experimental setup in [12], we start with 10 initial prompts provided in the original work. For each tested model, we optimize a 5-token prompt suffix using the NSGA-II black-box optimization algorithm. During optimization, the target model is treated as the positive example, while substitute models and a pretrained CLIP model serve as negative examples. The best-performing optimized prompt is selected as the fingerprint prompt for the target model. Subsequently, we run the generation with 10 different random seeds and compute $\mu + 3 \sigma$ of the CLIP scores as the threshold.

TABLE VII: Hyperparameters for white-box fingerprint optimization.
<table><tr><td></td><td>SD 1.4</td><td>Deci</td><td>PixArt</td><td>SD 3</td></tr><tr><td>Truncated step  $t _ { \mathrm { t r u n c } }$ </td><td>5</td><td>5</td><td>10</td><td>10</td></tr><tr><td>Learning rate η</td><td>0.01</td><td>0.01</td><td>0.002</td><td>0.01</td></tr><tr><td>Optimization iters I</td><td>15</td><td>30</td><td>8</td><td>60</td></tr></table>

## Adaptive obfuscations.

• GPT rewriting: We employ GPT-4o to rewrite the original prompts using the following instruction: ”You are a prompt enhancer. For the prompt that will be input into the textto-image model below, you can make some rewrites and optimizations to improve the image quality, but you must ensure that the semantics remain unchanged. Prompt to be rewritten: {prompt}.

• RTA [17]: Following the implementation in [18], we randomly add four tokens for each prompt using the pipeline’s tokenizer.

• EO [18]: Following the official source code’s approach, we optimize the text embeddings for 10 iterations at the first denoising step to reduce the classifier-free guidance magnitude, setting $l _ { t a r g e t } = 1$

• SAIL [29]: Following the original settings, we optimize the initial noise using an optimization threshold of 8.2 and a maximum of 10 iterations to steer the generation starting point away from the sharp regions described in the paper.

Our method. Our method consists of two verification settings: white-box and black-box. In the white-box setting, we apply the original text encoder of each diffusion model as the text encoder for prompt embedding extraction. In the blackbox setting, we directly use the pipeline of each model for generation. We use different hyperparameters for different models due to their varying architectures and parameter scales. We detail the hyperparameters in Table VII.

Metrics.

• SSCD: We use the SSCD [32] model (sscd\_disc\_ large) provided in the GitHub repository of the original paper for collapsed generation verification.

• CLIP Score: We use the openai/ clip-vit-base-patch32 model from the transformers library to measure the semantic similarity between generated images and text prompts on embeddings.

Computational resources. All the experiments are conducted on a machine equipped with 8 NVIDIA RTX 4090 GPUs, each with 24 GB of VRAM, except for white-box fingerprint optimization on PixArt, which is performed on a single NVIDIA A100 GPU with 80 GB of VRAM.

TABLE VIII: Robustness (white-box, quantization and pruning). Green and red cells indicate successful and failed verification, respectively.
<table><tr><td>Obfuscation</td><td>SD 1.4</td><td>Deci</td><td>PixArt</td><td>SD 3</td></tr><tr><td colspan="5">FingerInv [13]  $( L _ { 2 } { \mathrm { ~ d i s t a n c e } } )$ </td></tr><tr><td>Quant. BF16</td><td> $3 . 7 6 \times 1 0 ^ { - 3 }$   $\smash { 3 . 7 6 \times 1 0 ^ { - 4 } }$ </td><td> $2 . 8 4 \times 1 0 ^ { - 2 }$ </td><td> $9 . 2 5 \times 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>Quant. INT8</td><td>4.52 × 10</td><td> $1 . 9 6 \times 1 0 ^ { - 3 }$ </td><td> $6 . 3 1 \times 1 0 ^ { - 4 }$ </td><td></td></tr><tr><td>Quant. FP4</td><td> $2 . { \overset { \cdot } { 1 0 } } \times 1 0 ^ { - 4 }$ </td><td> $4 . 1 2 \times 1 0 ^ { - 3 }$ </td><td> $1 . 3 8 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>Prune 10%</td><td> $1 . 6 3 \times 1 0 ^ { - 4 }$ </td><td> $1 . 4 5 \times 1 0 ^ { - 2 }$ </td><td> $1 . 9 7 \times 1 0 ^ { - 4 }$ </td><td></td></tr><tr><td>Prune 20%</td><td> $2 . 4 3 \times 1 0 ^ { - 4 }$ </td><td> $1 . 1 1 \times 1 0 ^ { - 1 }$ </td><td> $5 . 0 9 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>Prune 30%</td><td> $1 . 3 0 \times 1 0 ^ { - 2 }$ </td><td> $3 . 4 1 \times 1 0 ^ { - 1 }$ </td><td> $2 . 1 6 \times 1 0 ^ { - 2 }$ </td><td></td></tr><tr><td colspan="5">Ours  $( p _ { \mathrm { v a l } } )$ </td></tr><tr><td>Quant. BF16</td><td> $1 . 2 3 \times 1 0 ^ { - 1 9 }$ </td><td> $6 . 1 1 \times 1 0 ^ { - 2 0 }$ </td><td> $8 . 4 6 \times 1 0 ^ { - 8 }$ </td><td> $8 . 3 8 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>Quant. INT8</td><td> $3 . 9 2 \times 1 0 ^ { - 1 9 }$ </td><td> $6 . 2 3 \times 1 0 ^ { - 2 0 }$ </td><td> $3 . 1 6 \times 1 0 ^ { - 7 }$ </td><td> $1 . 3 9 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>Quant. FP4</td><td> $4 . 6 9 \times 1 0 ^ { - 1 8 }$ </td><td> $4 . 8 9 \times 1 0 ^ { - 1 9 }$ </td><td> $5 . 7 2 \times 1 0 ^ { - 9 }$ </td><td>1.53×10-9</td></tr><tr><td>Prune 10%</td><td> $3 . 1 1 \times 1 0 ^ { - 1 9 }$ </td><td> $8 . 3 7 \times 1 0 ^ { - 2 0 }$ </td><td> $3 . 5 6 \times 1 0 ^ { - 1 0 }$ </td><td> $2 . 5 1 \times 1 0 ^ { - 1 2 }$ </td></tr><tr><td>Prune 20%</td><td> $2 . 0 0 \times 1 0 ^ { - 2 0 }$ </td><td> $2 . 9 6 \times 1 0 ^ { - 1 8 }$ </td><td> $6 . 4 1 \times 1 0 ^ { - 6 }$ </td><td> $6 . 0 8 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>Prune 30%</td><td> $7 . 1 8 \times 1 0 ^ { - 9 }$ </td><td> $4 . 5 7 \times 1 0 ^ { - 2 }$ </td><td> $7 . 0 8 \times 1 0 ^ { - 1 1 }$ </td><td> $1 . 9 0 \times 1 0 ^ { - 1 1 }$ </td></tr></table>

TABLE IX: Robustness (white-box, fine-tuning). Green and red cells indicate successful and failed verification, respectively.
<table><tr><td>Model</td><td>FingerInv [13] (L2 distance)</td><td>Ours  $( p _ { \mathrm { v a l } } )$ </td></tr><tr><td>SD 1.5</td><td> $5 . 4 7 \times 1 0 ^ { - 4 }$ </td><td> $1 . 9 5 \times 1 0 ^ { - 1 8 }$ </td></tr><tr><td>Deliberate</td><td> $2 . 6 6 \times 1 0 ^ { - 4 }$ </td><td> $6 . 0 9 \times 1 0 ^ { - 1 4 }$ </td></tr><tr><td>Realistic</td><td> $6 . 9 8 \times 1 0 ^ { - 4 }$ </td><td> $1 . 1 8 \times 1 0 ^ { - 1 8 }$ </td></tr></table>

## APPENDIX C

## ADDITIONAL EXPERIMENT RESULTS

## A. Robust Analysis in the White-box Setting

Table VIII and IX present a comprehensive robustness analysis under the white-box setting. The results demonstrate that both our method and FingerInv remain robust against the majority of obfuscation techniques, including quantization, pruning, and fine-tuning.

However, both fingerprinting methods fail on lightweight architectures such as Deci. This failure is attributed to the severe performance degradation caused by 30% pruning, which renders the model incapable of generating semantically meaningful images. Notably, such drastic obfuscation is unlikely to be applied by attackers in realistic threat scenarios.

## B. Ablation Study

We investigate the impact of varying the number of fingerprint prompts (embeddings) M and number of generations K on ownership verification. We evaluate the $p _ { \mathrm { v a l } }$ derived from average pairwise similarity scores using $M = \{ 2 , 4 , 8 \}$ and $K = \{ 2 , 4 , 8 \}$ . The ablation results for black-box and whitebox settings are presented in Fig. 11 and Fig. 12, respectively. While increasing the number of prompts/embeddings to 8 yields the optimal $p _ { \mathrm { v a l } } .$ , the performance gain is marginal. Conversely, using 4 seeds is sufficient to maintain a significantly low $p _ { \mathrm { v a l } }$ , since inducing collapsed generations over 4 random initializations is already distinctive enough for ownership verification. To balance performance with the query budget, we select $M = 4$ prompts/embeddings and K = 4 seeds as the default configuration.

## C. Case Study

Fig. 13 provides qualitative examples for the black-box and white-box uniqueness analysis. Notably, only the image groups along the main diagonal maintain a high degree of visual consistency, while off-diagonal entries exhibit varying levels of significant diversity. This validates the strong discriminative ability of our proposed average pairwise similarity score.

![](images/637e9dc8ee8797b7b68ceecca4d59d81d52366e92e61920e984d5b8995facc5e.jpg)

![](images/7bb76ba905001ef63e427fcc1937baabc62d21cb04ea5835e9ee52455b547a76.jpg)  
Number of Prompts

![](images/ec6d9e4eeea3ec5e28e0feb25d7e996279307e98b006f67df0f1fe78e9bb5ae6.jpg)  
Fig. 11: $p _ { \mathrm { v a l } }$ using different number of prompts/generations (black-box).

![](images/c4ce3905200f95fa5a046844c8982735e068c3e10c603f4b299fc3b205069555.jpg)

![](images/ee9a652988c3279e0eecac097cf114f299878af190b54b4ac3790f9cc26d696a.jpg)  
Fig. 12: $p _ { \mathrm { v a l } }$ using different number of embeddings/generations (white-box).

Furthermore, Fig. 14 and 15 demonstrate the robustness of our method in both settings. Taking SD 1.4 as the target model, the generated images retain high similarity scores even when subjected to standard obfuscations (quantization, pruning, finetuning) or adaptive attacks (e.g., RTA, EO). Moreover, such consistency is not affected even when the quality of the generated images is degraded. These results confirm that our method is highly robust and applicable to diverse real-world threat scenarios.

## APPENDIX D FURTHER ANALYSIS

## A. Scalability of Black-Box Fingerprints in Recent Models

Following early diffusion models, recent models (e.g., SD 3 [40]) have heavily employed rigorous data deduplication pipelines to explicitly suppress the phenomenon of collapsed generation. One might question the scalability of finding natural collapsed text prompts in such heavily deduplicated models. However, while deduplication effectively mitigates simple memorization caused by exact data duplication, recent studies [30], [39] reveal that collapsed generation is also driven by other factors, including intrinsic data characteristics (e.g., outlier samples) and training dynamics (e.g., overtraining). This ensures that a sparse set of natural candidates inevitably persists even in highly curated datasets. Because these natural collapsed prompts carve deep, specific basins in the loss landscape during the extensive training phase, they inherently exist as discrete text tokens. Importantly, our framework only requires a minimal set of fingerprints (e.g., $| \mathcal { P } _ { f } | ~ = ~ 4 )$ for reliable verification, making this sparse availability more than sufficient. Furthermore, model owners can efficiently capture these rare collapsed prompts “on the fly” during training via loss monitoring [39] as illustrated in Sec. IV-C, bypassing the need for expensive post-hoc scanning.

SD 1.4  
SD 2.1  
SD 1.4  
![](images/316d35611502d60ab86dabd337b7fb596c6869872a04a4a0cfb8632c9f196516.jpg)  
Deci  
Fig. 13: Case study for uniqueness analysis. Left: black-box, right: white-box.  
SAIL  
Fig. 14: Case study for robustness analysis (SD 1.4, blackbox).

## B. Efficacy of White-Box Optimization on Recent Models

While natural collapsed generation is reduced in recent models, our white-box optimization provides a proactive mechanism to deliberately induce such behaviors. Unlike black-box prompts that rely on naturally occurring collapsed generation stemming from training outliers and artifacts, white-box optimization actively forces the model’s latent trajectory into a collapsed manifold via gradient descent in the continuous embedding space. As demonstrated in our experiments with PixArt and SD 3, we can still reliably opti mize continuous embeddings that trigger collapsed generations in these recent models, ensuring the framework’s long-term applicability.

Realistic

![](images/c1f2bfe0cfffb920a09739df11d179e41b3b13410d7432fe8faadaa229c2bfa1.jpg)  
SD 1.5  
Deliberate  
Fig. 15: Case study for robustness analysis (SD 1.4, whitebox).

TABLE X: Embeddings obtained via discrete fingerprint optimization fail verification.
<table><tr><td>Strategy</td><td>Post-hoc Proj.</td><td>PEZ [45]</td><td>GCG [46]</td></tr><tr><td>pval</td><td> $1 . 6 8 \times 1 0 ^ { - 4 }$ </td><td> $4 . 2 3 \times 1 0 ^ { - 2 }$ </td><td> $2 . 3 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Success?</td><td>X</td><td>X</td><td>X</td></tr></table>

## C. Challenges in Discrete Fingerprint Optimization

Given the success of continuous white-box optimization, an intuitive extension is to map these optimized embeddings back to discrete text tokens to enable purely black-box verification. However, we find such discretization highly non-trivial in practice.

The fundamental reason is that collapsed modes reside in extremely narrow and precise regions within the latent space [30]. While natural black-box prompts can successfully trigger collapse because the model explicitly learned their discrete mappings during training, our white-box optimized embeddings locate artificial collapsed points. When applying post-hoc projection (mapping optimized embeddings to the nearest vocabulary tokens) or gradient-based discrete optimization (e.g., PEZ [45] or GCG [46]), the necessary discretization step severely disrupts the precise continuous coordinates required to stay within these narrow basins, resulting in a significant drop in SSCD scores (leading to $p _ { \mathrm { v a l } } ~ > ~ 1 0 ^ { - 4 }$ in Table X). Consequently, we maintain the white-box fingerprints in the continuous embedding space and leave the discretization of collapsed embeddings as a valuable direction for future work.

TABLE XI: Verification efficacy and time cost of different initialization strategies.
<table><tr><td>Init.</td><td>Random</td><td>Low Diversity</td><td>Collapsed</td></tr><tr><td>pval (↓)</td><td> $2 . 8 3 \times 1 0 ^ { - 6 }$ </td><td> $4 . 2 6 \times 1 0 ^ { - 8 }$ </td><td> $1 . 2 3 \times 1 0 ^ { - 1 9 }$ </td></tr><tr><td>Time (s)</td><td>47.77</td><td>24.81</td><td>25.03</td></tr><tr><td>Success?</td><td>L</td><td>L</td><td>L</td></tr></table>

D. Different Initialization Strategies in White-Box Optimization

We evaluate the impact of different initialization prompts on white-box optimization in Table XI. While warm starts based on collapsed prompts accelerate convergence, they are not strictly necessary. Low-diversity prompts generated by LLMs (e.g., pure textual) are also sufficient. Even with random initialization, the optimization successfully converges, although it requires more steps.

## APPENDIX E

## PROMPTS USED IN VISUALIZATION

## A. Figure 1

• The No Limits Business Woman Podcast

## B. Figure 2

• GPT Generated Prompt: A glass teapot house in a foggy forest

• SD 1.4 Collapsed Prompt: <i>The Long Dark</i>Gets First Trailer, Steam Early Access

• SD 2.1 Collapsed Prompt: Pencil pleat curtain in collection Linen, fabric: 392-05

## C. Figure 4

• Lilah Teal Blue Area Rug by Andover Mills

## D. Figure 5

• Collapsed: Thumbprintz Splatter No I Red Floor Pillow

• Normal: Happy woman in hat with shopping bags showing credit card

## E. Figure 13

• SD 1.4 (black-box): <em>Bloodborne<em>Video: Sony Explains the Game’s Procedurally Generated Dungeons

• SD 2.1: Pencil pleat curtains in collection Jupiter, fabric: 127-00

• SD 1.4 (white-box), Deci: N/A<sup>1</sup>

## F. Figure 14

• <em>Bloodborne<em>Video: Sony Explains the Game’s Procedurally Generated Dungeons
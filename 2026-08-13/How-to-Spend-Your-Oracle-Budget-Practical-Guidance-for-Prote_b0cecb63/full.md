# How to Spend Your Oracle Budget: Practical Guidance for Protein Structure Prediction Models

Aleksandra Kalisz <sup>1</sup> <sup>2</sup> Jack Simons <sup>1</sup> Krisztina Sinkovics <sup>1</sup> Noam Ghenassia <sup>1</sup> Shikha Surana <sup>1</sup> Henry Moss <sup>3</sup> Paul Duckworth <sup>1</sup>

## Abstract

Foundation models for protein structure prediction remain unreliable on certain targets. External oracles can flag and correct these failures, but biological oracles are expensive, making oracle budget a critical constraint. Existing guidance methods, such as FK-steering, DPO, and Best K-of-N sampling, differ in how they spend this budget, yet no systematic comparison exists to guide method selection. To bridge this gap, we benchmark these methods alongside the recently proposed Optimisation Over Outputs (O3), which applies off-theshelf optimisers within a generative model’s latent subspace. We extend the usage of O3 to protein structure prediction models. Overall, our work provides the first practical reference for oracle budget-aware guidance. Our evaluation on two protein targets, calmodulin (1CLL) and E. coli aspartate transcarbamoylase (9EEH), reveals that no single method consistently dominates across all budgets and oracles. Specifically, O3 proves most effective at low oracle budgets, while FK-steering and DPO demonstrate improved performance as the budget increases. We distil these findings into actionable recommendations for practitioners operating under real-world oracle-budget constraints.

## 1. Introduction

Foundation models for structure prediction such as AlphaFold3 (Abramson et al., 2024), Chai-1 (Chai Discovery et al., 2024), and Boltz-2 (Passaro et al., 2025) can accurately predict structures from sequence information alone. Downstream applications, including drug discovery and antibody therapeutic design, rely on these predictions being correct. In practice these predictions are not always reliable: a model may collapse onto a single conformation when several are functionally relevant (Wayment-Steele et al., 2023), produce physically implausible geometries (Buttenschoen et al., 2024), or assemble large biomolecular complexes incorrectly (Yin & Pierce, 2023). A typical procedure to flag unreliable predictions is via the usage of external oracles, which assign a score to a given prediction, e.g. a molecular dynamics simulation of conformational stability or binding energy (Hollingsworth & Dror, 2018). A natural research direction is whether such oracles can be leveraged at training or inference time to return higher scoring structures, a process referred to as guidance. There are two dominant classes of guidance approaches, inference-time steering and fine-tuning; however their relative effectiveness is poorly understood, a gap in the literature we seek to fill.

Inference-time methods, such as Feynman-Kac steering (FKsteering) (Singhal et al., 2025), sample multiple interacting particles through the generative process and resample at intermediate steps using oracle scores. Fine-tuning methods, such as Direct Preference Optimisation (DPO) (Rafailov et al., 2023), instead update model parameters using oraclescored samples. An even simpler approach to guidance with an oracle is Best K-of-N sampling, whereby we draw N samples and return the K with the highest oracle scores. All three approaches improve outputs given enough oracle evaluations, but biological oracles are usually expensive: a single molecular dynamics simulation can take days of GPU compute (Hollingsworth & Dror, 2018), and wet-lab assays are slower still.

As well as benchmarking the methods listed above, we also provide the first application of the newly proposed Optimisation Over Outputs (O3) framework of Willis et al. (2025) to a protein structure prediction model. O3 uses a small set of example model generations to build a low-dimensional subspace of the latent space over which standard optimisers can be directly applied. More details of the precise methodology are given in Willis et al. (2025).

We analyse the relative performances of O3, FK-steering, DPO, and Best K-of-N sampling applied to Boltz-2 (Passaro et al., 2025) across a range of budgets, with the goal to improve the quality of the predicted protein structures. We find that our new application of O3 to Boltz-2 dominates at low-to-mid budgets that we evaluate on, while FK-steering and DPO perform better with larger budgets. This qualitative trade-off is demonstrated in our experimental study reported in Section 5. The primary goal of this paper is straightforward: provide the first practical reference for which guidance strategies are effective under different oracle evaluation budgets.

![](images/47d383f57da1dc016e342587982aedfed736d0ef8ec283354cbc9e62cd394582.jpg)  
Figure 1. Bayesian Optimisation via O3 in a Boltz-2. d = 3 surrogate subspace created from three Calmodulin seed samples (top row), showing the 2D U space (bottom row), the GP training data (white dots) and mean predictive posterior over U, the max acquisition point (red cross) for two acquisition rounds, and a 25x25 exhaustive ground-truth grid directly evaluating the oracle (bottom right).

## 2. Related Work

Protein structure prediction. Recent foundation models for protein structure prediction, including AlphaFold3 (Abramson et al., 2024), Chai-1 (Chai Discovery et al., 2024), and Boltz-2 (Passaro et al., 2025), share a common architecture: an attention-based trunk, typically a Pairformer, that processes the input sequence into single and pair representations, which is then used as conditioning information to a diffusion block over atomic coordinates. Throughout our experiments we use the open-source model Boltz-2.

Latent space optimisation. Black-box optimisation describes optimisation of an objective function which is explicitly unknown a priori but can be queried through function evaluations. These evaluations can be expensive, noisy, and derivative-free. A popular approach to such problems is Bayesian Optimisation (BO) (see e.g. Garnett (2023)). However, standard BO approaches begin to fail in highdimensional or highly structured spaces. To tackle this shortcoming, several Latent space Bayesian Optimisation (LSBO) approaches have been proposed which instead perform BO in the latent space associated with a generative model. Examples of latent space Bayesian optimisation include VAE-BO (Gomez-Bombarelli et al.´ , 2018), weighted retraining (Tripp et al., 2020), LOL-BO (Maus et al., 2022), and COWBOYS (Moss et al., 2025). Generative models often have latent spaces that are too large to apply these methods directly, fortunately, recent work from Bodin et al. (2024) and Willis et al. (2025) proposes an alternative approach that extracts low-dimensional subspaces from generative models, over which standard optimisers can be applied.

Inference-time guidance. Inference-time guidance biases the sampling trajectory of a diffusion model toward a target distribution without modifying its weights. Classifier guidance (Dhariwal & Nichol, 2021) perturbs the sampling trajectory by adding the gradient of an externally-trained noise-aware classifier to the denoising step, while classifierfree guidance (Ho & Salimans, 2022) trains a single model on both conditional and unconditional inputs and combines their score estimates at inference time. We choose Feynman-Kac steering (Singhal et al., 2025) as a baseline for this work since it does not require propagating gradients from the oracle, and does not require an auxiliary network.

Reward-based fine-tuning. DDPO (Black et al., 2023) casts sampling as a multi-step Markov decision process and applies policy-gradient updates with the oracle as reward; DRaFT (Clark et al., 2023) backpropagates the oracle gradient through the sampling chain when the oracle is differentiable. DPO (Rafailov et al., 2023) sidesteps explicit reward modelling by fine-tuning on pairs of oracle-scored samples to increase the relative likelihood of the preferred one, and Diffusion-DPO (Wallace et al., 2023) adapts this objective to the diffusion likelihood. We choose DPO as our fine-tuning baseline because it requires only sample pairs and oracle scores.

Typically, biological oracles are non-differentiable, meaning that gradient information is not available, making many common guidance approaches unsuitable.

## 3. Guidance Methods

We denote our frozen generative model $g : { \mathcal { Z } }  { \mathcal { X } }$ , that maps random noise $z \sim p _ { 0 }$ from within a latent space $z \in { \mathcal { Z } }$ to structures $x \sim p _ { 1 }$ in data space $x \in \mathcal { X }$ . An oracle $r :$ $\mathcal { X } $ R scores each generated output. In our experiments $g$ is the protein structure prediction model Boltz-2 (Passaro et al., 2025) with $\mathcal { Z } = \mathcal { X } = \mathbb { R } ^ { 3 n }$ for an n-atom system and $p _ { 0 } = \mathcal { N } ( 0 , I )$ . The goal of guidance is to use this model to generate a batch of $K$ candidate outputs whose scores under r should be as high as possible, under the constraint that we only make N oracle queries. We will now introduce three guidance methods, explaining the key modifications and design choices that were required to apply them to protein structure prediction models.

## 3.1. Bayesian Optimisation via O3

We now describe how the O3 framework of Willis et al. (2025) is used to guide the generative process of Boltz-2. The algorithm relies on two key components, each necessitating a number of oracle queries: (i) the construction of a suitable low-dimensional subspace U within which generation is constrained (the first M queries), and (ii) a resourceefficient optimization procedure to effectively explore this subspace (the remaining $N - M$ queries).

(i) Constructing U. Given a total budget of N, we first sample $M < N$ initial structures from $g$ and score them with $^ { r } \cdot$ Exploiting the determinism of $^ { g , }$ we collect the d latents corresponding to the d highest scoring structures<sup>1</sup> as seeds, $Z = [ z _ { 1 } , \ldots , z _ { d } ] ^ { \top }$ . Here d determines the dimension of the resulting optimisation space, d−1. We then follow the guidelines of O3 (Willis et al., 2025), and construct $\mathcal { U } \subset [ 0 , \bar { 1 } ] ^ { d - 1 }$ via a surrogate chart $\phi : \mathcal { U } \to \mathcal { Z }$ that decodes each $u \in \mathcal { U }$ to a corresponding latent z through two maps (see Algorithm 1). The first, a Knothe–Rosenblatt transform (Knothe, 1957; Rosenblatt, 1952) $\phi : \mathcal { U } \to \mathbb { S } _ { + } ^ { d - 1 }$ , maps u to a nonnegative weight vector w on the simplex. The second, a LOL projection (Bodin et al., 2024) $\ell ( w , Z ) : = w ^ { \top } Z$ , projects these weight vectors back onto the support of our isotropic Gaussian prior $p _ { 0 }$ . This is effective because the original seeds themselves were drawn from $p _ { 0 }$ and the weights are on the simplex. Once d seeds are chosen, the subspace is fixed, and never re-built. Optimisation occurs within $u ,$ where U has dimensionality $d - 1$ and crucially is independent of the dimensionality of Z, thus opening up opportunities for efficient optimization. In Appendix Fig. A.2 we select two seeds $d = 2 ,$ , and linearly interpolate between them using 5 intermediate weights, displaying their corresponding generations $g ( \cdot ) \sim p _ { 1 }$

Algorithm 1 Mapping from U to protein structures   
$\mathbf { l } \colon u = [ u _ { 1 } , \ldots , u _ { d - 1 } ] ^ { \intercal }$ // selected by optimiser   
$2 \colon w  \phi ( u )$ // Knothe–Rosenblatt transform   
$3 \colon z  \ell ( w , Z ) : = w ^ { \top } Z$ // LOL projection   
4: $x \gets g ( z )$ // decoded structure, ready for scoring

(ii) Bayesian optimization over U. Given a subspace U, we use Bayesian optimisation (Garnett, 2023, BO) — the gold-standard for low-dimensional black-box optimisation — to efficiently utilise our remaining N − M oracle budget. We begin by fitting a Gaussian process (GP) (Williams & Rasmussen, 2006) surrogate model on $d + 2$ initial points, where d of them are the seed latents projected onto U and the remaining 2 are sampled randomly from within U. Each subsequent round uses Log Expected Improvement (Ament et al., 2023) to drive data acquisition, each time updating the GP model with the newly acquired point. In order to provide additional training data, we also project the $d$ seeds into the U subspace using the surrogate chart $\phi ,$ and include those as additional training data points (as demonstrated in Fig. 1). We use an RBF kernel, constant mean function, single task GP, and Monte Carlo acquisition function sampler from BOTorch (Balandat et al., 2020).

## 3.2. FK-Steering

Feynman-Kac steering (Singhal et al., 2025) is an inferencetime method which uses the signal from an oracle to alter the sampling trajectory towards high-reward regions of the target distribution via Sequential Monte Carlo. The procedure generates M interacting stochastic processes called particles. Particles are resampled proportionally to weights w calculated for each particle i using the oracle reward r evaluated on the denoised predictions from $\boldsymbol { x } _ { t } ^ { i }$ (for denoising timestep t):

$$
w ( x _ { t } ^ { i } ) = \frac { g ( x _ { t } ^ { i } \mid x _ { t _ { \mathrm { p r e v } } } ^ { i } ) } { g _ { \mathrm { p r o p o s a l } } ( x _ { t } ^ { i } \mid x _ { t _ { \mathrm { p r e v } } } ^ { i } ) } \lambda \exp ( r ( x _ { t } ^ { i } ) - r ( x _ { t _ { \mathrm { p r e v } } } ^ { i } ) )
$$

where λ controls the scale of the reward signal, and $g _ { \mathrm { p r o p o s a l } } ( x _ { t } \ \vert \ x _ { t _ { \mathrm { p r e v } } } )$ is the proposal transition kernel. In our experiments, since most high-quality biological oracles are non-differentiable, we use the original BOLTZ-2 transition kernel $g \big ( x _ { t } ^ { i } \mid x _ { t _ { \mathrm { p r e v } } } ^ { i } \big )$ , demonstrating the gradient-free nature of FK-steering. The rewards are calculated only at the resampling steps and there are several ways to combine them across the reward trajectory (difference, max, average);

in this work we use the reward difference, which favours particles with increasing rewards.

To restrict to an oracle budget of N and return a batch of K distinct samples, we require $M \geq K$ particles, each of which has access to $N / M$ oracle calls. Practically, this ratio determines the number of resampling steps in our algorithm. Boltz-2 employs a particular noising schedule, which stops injecting noise in the last quarter of the trajectory in order to avoid atomic jitter. Since FK-steering relies on the diversity of particles, we only resample at intermediate denoising steps where Boltz-2 adds positive noise. We call the oracle once at the start of the denoising process, once at the end, and spread the remainder of the $N / M - 2$ budget across the first $3 / 4$ of the denoising trajectory.

## 3.3. DPO

Unlike our other approaches, Direct Preference Optimisation (DPO) (Rafailov et al., 2023) fine-tunes and directly modifies the generative model parameters based on oraclescored sample pairs, with the aim of increasing the likelihood of generating high-scoring structures. DPO does not require a differentiable oracle (no explicit reward modeling is required, rather just pairs of structures and their relative oracle rankings) making it amenable to common biological oracles. However, we find that a large oracle budget is required to support effective fine-tuning at foundation model-scale.

Diffusion-DPO training objective. DPO updates the model’s weights θ in a direction that maximises the margin between the implicit rewards of preferred and rejected samples under a Kullback-Leibler regularised policy, as measured by the DPO loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D P O } } ( \boldsymbol { \theta } ) = - \mathbb { E } _ { ( \mathbf { x } _ { w } , \mathbf { x } _ { l } ) } \left[ \log \sigma \left( \beta \left( \log \frac { p _ { \boldsymbol { \theta } } ( \mathbf { x } _ { w } ) } { p _ { \mathrm { r e f } } ( \mathbf { x } _ { w } ) } - \log \frac { p _ { \boldsymbol { \theta } } ( \mathbf { x } _ { l } ) } { p _ { \mathrm { r e f } } ( \mathbf { x } _ { l } ) } \right) \right) \right] , } \end{array}
$$

where $\beta > 0$ controls the strength of the KL penalty against moving the new model $p _ { \theta }$ too far from the pre-trained reference model $p _ { \mathrm { r e f } }$ (initialized at the pre-trained Boltz-2 checkpoint), and σ represents the sigmoid function. Preference pairs always contain a preferred structure $\mathbf { x } _ { w }$ and a rejected structure $\mathbf { x } _ { l } ,$ each sampled from our N scored structures and lie above and below the median oracle score respectively. Obtaining log-likelihoods from diffusion models is prohibitively expensive (Song et al., 2020), we follow Wallace et al. (2023), and approximate the log-likelihood ratio in the DPO loss using differences in the individual training losses $( \mathcal { L } _ { \mathrm { D S M } } )$ of the reference and fine-tuned model as $\log ( p _ { \theta } ( \mathbf { x } ) / p _ { \mathrm { r e f } } ( \mathbf { x } ) ) \approx \mathcal { L } _ { \mathrm { D S M } } ^ { \mathrm { r e f } } ( \mathbf { x } ) - \mathcal { L } _ { \mathrm { D S M } } ^ { \theta } ( \mathbf { x } )$

Online and offline DPO. We evaluate two variants of DPO that differ in how oracle queries are distributed across training. The offline variant draws all N structures in a single batch, forms preference pairs once, and trains to convergence on the resulting fixed dataset. The online variant interleaves sampling and training, at each of the E epochs: we generate $N / E$ structures from the current model, score the structures with the oracle, form preference pairs, update parameters for one epoch; until a total budget of $N = ( N / E ) \times E$ queries has been utilised. Critically, the reference model $p _ { \mathrm { r e f } }$ is reset to the current policy at the start of each epoch, thus softening the KL regularisation. However, this resetting weakens the influence of the pre-trained model — a tradeoff we investigate in Section 5.4.

## 4. Experimental Setup

As our main experiment, we benchmark the four methods by comparing the predicted structures for the calmodulin protein from its amino acid sequence (of length 144). As ground truth, we compare against the 1CLL structure (Chattopadhyaya et al., 1992) consisting of 1184 atom coordinates from the Protein Data Bank (PDB) (Berman et al., 2000). Additionally, we benchmark on E. coli aspartate transcarbamoylase (PDB: 9EEH), with results in Appendix B. For all experiments, the base model we use, g, is Boltz-2 (Passaro et al., 2025) and for $1 { \mathrm { C L L } } \ Z = \ \mathbb { R } ^ { 3 5 5 2 }$ , and 9EEH $\mathcal { Z } = \mathbb { R } ^ { 2 1 , 6 9 6 }$ . All methods are permitted N oracle calls in total and return a batch of K candidate structures. Next, we describe the TM-score oracle used in the main experiment.

![](images/29aeeba699a7343c8085fe60f6d926c8a9af4dd9a67729d6cbdd64e9dcfdf0d7.jpg)  
Figure 3. TM-score on 1CLL. Left: the 1CLL crystal structure (blue). Centre: a vanilla Boltz-2 sample (orange) overlaid on the ground truth (grey) at TM-score 0.45. Right: a guided sample at TM-score 0.80. Structure pairs with TM-score above 0.5 are considered to share the same fold.

Oracle. The reward, r, is the TM-score (Zhang & Skolnick, 2004) of a generated structure x against the groundtruth 1CLL backbone $x _ { \mathrm { G T } }$ . To compare x with $x _ { \mathrm { G T } }$ , we find the optimal rigid alignment, namely the rotation and translation of x that minimise the distances to the corresponding residues of $x _ { \mathrm { G T } }$ . The TM-score is then

$$
r ( x ) = \mathrm { m a x } \frac { 1 } { n } \sum _ { i = 1 } ^ { n _ { \mathrm { a l i } } } \frac { 1 } { 1 + ( d _ { i } ( x , x _ { \mathrm { G T } } ) / C ) ^ { 2 } } ,
$$

where n is the number of residues in the ground truth structure (144 for 1CLL); $n _ { \mathrm { a l i } } \leq n$ is the number of residues from $x _ { \mathrm { G T } }$ matched to a residue in x by the optimal alignment, with residues that have no nearby counterpart excluded from the sum; $d _ { i }$ is the distance between the i-th aligned residue pair<sup>2</sup>; $C = 1 . 2 4 \sqrt [ 3 ] { n - 1 5 } - 1 . 8$ is a lengthdependent normalisation chosen so that random structure pairs yield approximately constant TM-score; and the maximum is over the rigid alignments described above. Scores lie in (0, 1], with values above 0.5 indicating the same fold and below 0.2 indicating random pairs; Fig. 3 shows examples on 1CLL. We use TM-score because it is relatively fast to compute, thus enabling our extensive experimentation. However, we emphasise that our analysis exposes trade-offs between the studied methodologies in settings where indeed the oracle is expensive or non-differentiable, a setting common in biology.

![](images/6a04e164e0149523fe8d2dec4e7a5f413567908bdbd62594426215cd63fd0080.jpg)  
Figure 2. Mean-of-K TM-score across oracle budgets. Comparison of FK-steering, DPO (in the online setting), Best K-of-N, and O3 on 1CLL across the six (N, K) configurations of Section 4. Bars are the mean over 5 random seeds; error bars are ±1 std. Higher is better. O3 method dominates at budgets evaluated, while FK-steering and DPO improve steadily with N.

For PDB: 9EEH experiments, we use the MolProbity oracle, described in Appendix B.1.

Budget settings. We evaluate each method at six (N, K) configurations spanning low to moderate oracle-call budgets: (20, 2), (50, 5), (100, 10), (200, 20), (500, 50), and (1000, 100). We are specifically interested in K > 1 settings throughout because downstream stages of biological design pipelines, such as wet-lab assay screening, often test batches of candidates rather than committing to a single proposal.

Evaluation metrics. At each (N, K) configuration, we report two performance metrics: the max-of-K score, the best single structure returned, and the mean-of-K score, the average score across the returned batch. The former measures single-shot design quality; the latter measures batched-screening quality. Each plot in Section 5 reports one of the two, with the complementary results presented in our appendices. All experiments are repeated over 5 random seeds unless stated otherwise.

## 5. Results

Section 5.1 discusses the comparison of the four methods on PDB: 1CLL target, across six (N, K) configurations introduced in Section 4. Sections 5.2 to 5.4 elaborate on method-specific findings and ablations for O3, FK-steering, and DPO, respectively. PDB: 9EEH results are provided in Appendix B.

## 5.1. Model Comparison Across Budgets

Fig. 2 reports the mean-of-K TM-score for our four methods (and Appendix Section A.1 presents the max-of-K results). O3 outperforms all baselines at low-mid budgets (N ≤ 1000) with the mean value plateauing at ∼ 0.81. Best K-of-N is roughly flat at ∼ 0.60 across all budgets. FKsteering and DPO improve with increased N, but neither are competitive with simple Best K-of-N at low budgets (N ≤ 100), with FK-steering achieving only ∼ 0.55 at N = 20. From N = 200 to N = 1000, FK-steering performs better, reaching ∼ 0.73 at N = 1000; DPO improves with N but does not match FK-steering or O3 within our range of settings, reaching ∼ 0.71 at N = 1000. O3 is the only method that meaningfully improves on the Best K-of-N baseline at low oracle budgets (N ≤ 100), supporting the case for example-defined latent-subspace optimisation when oracle calls are scarce. Appendix B.2 reports a model comparison across budgets on a different protein (PDB:9EEH) and MolProbity oracle (Appendix B.1).

## 5.2. O3

Fig. 4 isolates the optimisation step in O3, comparing Bayesian optimisation in the subspace against uniform random sampling in the same subspace for N = 100. We create the subspace U by selecting to best d seeds from M. Random sampling in U is shown to improve over the Best

K-of-N sampling baseline, demonstrating the value of the low-dimensional subspace. Furthermore, the superior performance of O3 seen in Fig. 2 comes from a combination of both the optimiser and the subspace construction.

![](images/519f5ba20523903c7680a2f3dc9bdda0f255c4d5f6aa664d19e218d1808b89c7.jpg)  
Figure 4. Bayesian optimisation versus random sampling in the O3 subspace. Mean-of-K TM-score at $N = 1 0 0$ for O3 with Bayesian optimisation and O3 with uniform random sampling in the same subspace, on 1CLL. The Best K-of-N baseline is invariant under our fixed $K / N$ ratio. Bars are means over 3 random seeds; error bars are ±1 std.

In the experiment summarised by Fig. 5 we sweep across subspace dimensions d for different available budgets. The key takeaway is that best-performing d varies with the budget — different oracle-call regimes favour different subspace dimensions. For example, when $N = 2 0$ the bestperforming dimension is $d = 6 ;$ however, for $N = 2 0 0$ dimension $d = 1 0$ yields better predictions. It is clear there is a trade-off between the richness of the O3 subspace, guaranteed by increasing the number of seed latents d, and the resulting difficulty of the optimisation task in that subspace. Additional O3 ablations and results are presented in Appendices A.2 and B.3.

![](images/825ff98284e6fc1eb5047bc63494ef08668c3449d26b6ee05f09033854a0eac8.jpg)  
Figure 5. Performance for different O3 subspace dimensions. Max-of-K TM-score for O3 with Bayesian optimisation, across subspace dimensions d for different budgets, on 1CLL. The bestperforming d varies with the budget. Bars are means over 3 random seeds; error bars are ±1 std.

## 5.3. FK-steering

We ablate three of FK-steering’s hyperparameters: the number of particles, K, the number of resampling steps $N / K$ and the reward-scaling coefficient λ. Fig. 6 shows that, with K and N fixed, increasing λ improves performance across all budgets, however, it likely hinders output diversity. A high $\lambda = 5 0$ amplifies the signal from the oracle, forcing the algorithm to greedily resample the best-performing particles across the trajectory, driving the mean-of-K and max-of-K (see Fig. A.7) closer. In Fig. A.8 we vary K per fixed budgets of N and find that increasing the number of resampling steps $N / K$ at the cost of particle population K generally yields better performance at higher oracle budgets. Additional FK-Steering ablations are available in Appendix A.3.

![](images/dab51873a5303b106c5a49da732d8d94e069c4904d3a1f871098626488b273aa.jpg)  
Figure 6. Effect of FK-steering λ hyperparameter across different budgets N. Mean-of-K TM-score achieved by FK-steering on 1CLL. Bars are means over 10 random seeds; error bars are ±1 std. Higher λ increases the signal from the rewards and leads to better performance across budgets.

## 5.4. DPO

Fig. 7 compares online and offline DPO across oracle budgets. (Max-of-K counterpart of this result is reported in Appendix A.4). Online DPO improves steadily with N, reaching a mean TM-score of 0.708 and a max TM-score of 0.783 at $N = 1 0 0 0$ . In contrast, offline DPO shows little sensitivity to budget, plateauing at 0.55–0.57 across all budgets. The gap between the two variants widens with scale: negligible at $N = 1 0 0 .$ , it becomes substantial at $N = 5 0 0$ These results suggest that the gains from DPO arise primarily from adaptive on-policy resampling, rather than from additional optimisation steps alone, consistent with prior observations that online preference optimisation methods can significantly outperform offline counterparts under a fixed budget (Tang et al., 2024). We also find that training hyperparameters such as the preference batch size and the number of samples per epoch can meaningfully affect performance at large N, though a controlled study isolating each factor is left for future work.

![](images/990e4821a5523aa4939033e9cc6dbd150bfdaa528caded5cec580a78e508d28b.jpg)  
Figure 7. Online vs. offline DPO across oracle budgets. Meanof-K TM-score achieved by online and offline DPO on 1CLL, for oracle budgets $N \in \{ 1 0 0 , 5 0 0 , 1 0 0 0 \}$ . Bars are means over 5 random seeds; error bars show ±1 std. Online DPO improves consistently with N, while offline DPO remains flat across budgets.

## 6. Conclusions

Notably, our work represents the first application of O3 to protein structure prediction. We demonstrate that this novel approach equips practitioners with a highly effective tool, yielding exceptional results particularly in budgetconstrained tasks. Using Boltz-2 and two protein targets along with the TM-score and MolProbity oracles, we compare O3, FK-steering, DPO, and Best K-of-N sampling under oracle-budget constraints, thus providing a practical reference for budget-aware guidance in biological settings. Across the budgets we evaluated, O3 dominates the performance. $\mathbf { A t } \leq 1 0 0$ queries, it is the only method that meaningfully improves on the Best K-of-N baseline, while FK-steering and DPO both steadily improve with N. As the only method that updates the model parameters, DPO is the natural candidate at substantially larger budgets than we evaluate here.

The conclusion from this work can be summarised in the following piece of practical advice: use O3 at constrained oracle budgets. FK-steering can be competitive at moderate budgets, and reach for DPO once budgets are large enough to fine-tune. An additional practical consideration is batch size and required batch diversity. DPO amortises its training budget into the model weights, making sampling large subsequent batches cheap, whereas inference-time and search methods usually discard oracle values afterwards. Future work will include additional protein targets paired with other, potentially expensive and non-differentiable, oracles.

## Impact Statement

Our work is a methodological comparison of guidance methods for protein-structure foundation models, aimed at helping practitioners spend limited biological oracle budgets effectively, including for downstream applications such as drug discovery and therapeutic design. As with all generalpurpose tools for protein design, the same techniques could in principle be applied to harmful targets; our experiments use only the publicly deposited calmodulin structure 1CLL and E. coli aspartate transcarbamoylase structure 9EEH and pose no specific dual-use risk in themselves.

## References

Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A. J., Bambrick, J., Bodenstein, S. W., Evans, D. A., Hung, C.- C., O’Neill, M., Reiman, D., Tunyasuvunakool, K., Wu, Z., Zemgulyt<sup>ˇ</sup> e, A., Arvaniti, E., Beattie, C., Bertolli, O.,˙ Bridgland, A., Cherepanov, A., Congreve, M., Cowen-Rivers, A. I., Cowie, A., Figurnov, M., Fuchs, F. B., Gladman, H., Jain, R., Khan, Y. A., Low, C. M. R., Perlin, K., Potapenko, A., Savy, P., Singh, S., Stecula, A., Thillaisundaram, A., Tong, C., Yakneen, S., Zhong, E. D., Zielinski, M., Z<sup>ˇ</sup> ´ıdek, A., Bapst, V., Kohli, P., Jaderberg, M., Hassabis, D., and Jumper, J. M. Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature, 630(8016):493–500, May 2024. ISSN 1476-4687. URL http://dx.doi.org/10.1038/ s41586-024-07487-w.

Ament, S., Daulton, S., Eriksson, D., Balandat, M., and Bakshy, E. Unexpected improvements to expected improvement for bayesian optimization. Advances in neural information processing systems, 36:20577–20612, 2023.

Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B., Wilson, A. G., and Bakshy, E. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. In Advances in Neural Information Processing Systems 33, 2020. URL http://arxiv.org/abs/1910. 06403.

Berman, H. M., Westbrook, J., Feng, Z., Gilliland, G., Bhat, T. N., Weissig, H., Shindyalov, I. N., and Bourne, P. E. The Protein Data Bank. Nucleic Acids Research, 28(1): 235–242, January 2000. ISSN 1362-4962. URL http: //dx.doi.org/10.1093/nar/28.1.235.

Black, K., Janner, M., Du, Y., Kostrikov, I., and Levine, S. Training Diffusion Models with Reinforcement Learning, 2023. URL https://arxiv.org/abs/2305. 13301.

Bodin, E., Stere, A., Margineantu, D. D., Ek, C. H., and Moss, H. Linear combinations of latents in generative

models: subspaces and beyond, 2024. URL https: //arxiv.org/abs/2408.08558.

Buttenschoen, M., Morris, G. M., and Deane, C. M. PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences. Chemical Science, 15(9):3130–3139, 2024. ISSN 2041-6539. URL http://dx.doi.org/10. 1039/d3sc04185a.

Chai Discovery, Boitreaud, J., Dent, J., McPartlon, M., Meier, J., Reis, V., Rogozhnikov, A., and Wu, K. Chai-1: Decoding the molecular interactions of life. October 2024. URL http://dx.doi.org/10.1101/ 2024.10.10.615955.

Chattopadhyaya, R., Meador, W. E., Means, A. R., and Quiocho, F. A. Calmodulin structure refined at 1.7 a reso-˚ lution. Journal ofMolecular Biology, 228(4):1177–1192, December 1992. ISSN 0022-2836. URL http://dx. doi.org/10.1016/0022-2836(92)90324-D.

Chen, V. B., Arendall, W. B., Headd, J. J., Keedy, D. A., Immormino, R. M., Kapral, G. J., Murray, L. W., Richardson, J. S., and Richardson, D. C. MolProbity: allatom structure validation for macromolecular crystallography. Acta Crystallographica Section D: Biological Crystallography, 66(1):12–21, 2010. doi: 10.1107/ S0907444909042073.

Clark, K., Vicol, P., Swersky, K., and Fleet, D. J. Directly Fine-Tuning Diffusion Models on Differentiable Rewards, 2023. URL https://arxiv.org/abs/ 2309.17400.

Dhariwal, P. and Nichol, A. Diffusion Models Beat GANs on Image Synthesis, 2021. URL https://arxiv. org/abs/2105.05233.

Garnett, R. Bayesian optimization. Cambridge University Press, 2023.

Gomez-Bombarelli, R., Wei, J. N., Duvenaud, D.,´ Hernandez-Lobato, J. M., S ´ anchez-Lengeling, B., She-´ berla, D., Aguilera-Iparraguirre, J., Hirzel, T. D., Adams, R. P., and Aspuru-Guzik, A. Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules. ACS Central Science, 4(2):268–276, January 2018. ISSN 2374-7951. URL http://dx.doi.org/ 10.1021/acscentsci.7b00572.

Ho, J. and Salimans, T. Classifier-Free Diffusion Guidance, 2022. URL https://arxiv.org/abs/ 2207.12598.

Hollingsworth, S. A. and Dror, R. O. Molecular Dynamics Simulation for All. Neuron, 99(6):1129–1143, September 2018. ISSN 0896-6273. URL http://dx.doi.org/ 10.1016/j.neuron.2018.08.011.

Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating the Design Space of Diffusion-Based Generative Models, 2022. URL https://arxiv.org/abs/2206. 00364.

Knothe, H. Contributions to the theory of convex bodies. Michigan Mathematical Journal, 4(1), January 1957. ISSN 0026-2285. URL http://dx.doi.org/10. 1307/mmj/1028990175.

Maus, N., Jones, H. T., Moore, J. S., Kusner, M. J., Bradshaw, J., and Gardner, J. R. Local Latent Space Bayesian Optimization over Structured Inputs, 2022. URL https: //arxiv.org/abs/2201.11872.

Miller, R. C., Patterson, M. G., Bhatt, N., Pei, X., and Ando, N. Cooperativity in e. coli aspartate transcarbamoylase is tuned by allosteric breathing. Nature Communications, 17(1):4285, mar 2026. ISSN 2041-1723. doi: 10.1038/ s41467-026-70909-y. URL https://doi.org/10. 1038/s41467-026-70909-y.

Moss, H., Ober, S. W., and Diethe, T. Return of the latent space cowboys: Re-thinking the use of vaes for bayesian optimisation of structured spaces. In International Conference on Machine Learning, pp. 44956–44970. PMLR, 2025.

Passaro, S., Corso, G., Wohlwend, J., Reveiz, M., Thaler, S., Somnath, V. R., Getz, N., Portnoi, T., Roy, J., Stark, H., Kwabi-Addo, D., Beaini, D., Jaakkola, T., and Barzilay, R. Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction. June 2025. URL http://dx.doi.org/ 10.1101/2025.06.14.659707.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. Direct Preference Optimization: Your Language Model is Secretly a Reward Model, 2023. URL https://arxiv.org/abs/2305.18290.

Rosenblatt, M. Remarks on a Multivariate Transformation. The Annals of Mathematical Statistics, 23(3):470–472, September 1952. ISSN 0003-4851. URL http://dx. doi.org/10.1214/aoms/1177729394.

Singhal, R., Horvitz, Z., Teehan, R., Ren, M., Yu, Z., McKeown, K., and Ranganath, R. A General Framework for Inference-time Scaling and Steering of Diffusion Models, 2025. URL https://arxiv.org/abs/2501. 06848.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Tang, Y., Guo, D. Z., Zheng, Z., Calandriello, D., Cao, Y., Tarassov, E., Munos, R., Avila Pires, B., Valko, M.,<sup>´</sup> Cheng, Y., and Dabney, W. Understanding the performance gap between online and offline alignment algorithms. arXiv preprint arXiv:2405.08448, 2024.

Tripp, A., Daxberger, E., and Hernandez-Lobato, J. M.´ Sample-Efficient Optimization in the Latent Space of Deep Generative Models via Weighted Retraining, 2020. URL https://arxiv.org/abs/2006.09191.

Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., and Naik, N. Diffusion Model Alignment Using Direct Preference Optimization, 2023. URL https://arxiv. org/abs/2311.12908.

Wayment-Steele, H. K., Ojoawo, A., Otten, R., Apitz, J. M., Pitsawong, W., Homberger, M., Ovchinnikov,¨ S., Colwell, L., and Kern, D. Predicting multiple conformations via sequence clustering and AlphaFold2. Nature, 625(7996):832–839, November 2023. ISSN 1476-4687. URL http://dx.doi.org/10.1038/ s41586-023-06832-9.

Williams, C. J., Headd, J. J., Moriarty, N. W., Prisant, M. G., Videau, L. L., Deis, L. N., Verma, V., Keedy, D. A., Hintze, B. J., Chen, V. B., Jain, S., Lewis, S. M., Arendall, W. B., Snoeyink, J., Adams, P. D., Lovell, S. C., Richardson, J. S., and Richardson, D. C. MolProbity: More and better reference data for improved all-atom structure validation. Protein Science, 27(1):293–315, 2018. doi: 10.1002/pro.3330.

Williams, C. K. and Rasmussen, C. E. Gaussian processes for machine learning, volume 2. MIT press Cambridge, MA, 2006.

Willis, S., Stere, A. I., Margineantu, D. D., Oldroyd, H. T., Fozard, J. A., Ek, C. H., Moss, H., and Bodin, E. Defining latent spaces by example: optimisation over the outputs of generative models, 2025. URL https://arxiv. org/abs/2509.23800.

Yin, R. and Pierce, B. G. Evaluation of AlphaFold antibody– antigen modeling with implications for improving predictive accuracy. Protein Science, 33(1), December 2023. ISSN 1469-896X. URL http://dx.doi.org/10. 1002/pro.4865.

Zhang, Y. and Skolnick, J. Scoring function for automated assessment of protein structure template quality. Proteins: Structure, Function, and Bioinformatics, 57(4):702–710, October 2004. ISSN 1097-0134. URL http://dx. doi.org/10.1002/prot.20264.

## A. Additional Results on PDB: 1CLL

A.1. Complementary Model Comparison Across Budgets

![](images/b96696a867cbda889f271a6a72b1b125c8efefa597e684431375341c31b7630a.jpg)  
Figure A.1. Max-of-K TM-score across oracle budgets. Comparison of FK-steering, DPO, Best K-of-N, and O3 on 1CLL across the six $( N , K )$ configurations of Section 4, reporting the maximum TM-score across the returned batch of K candidate structures. Bars are the mean over 5 random seeds; error bars are ±1 std. Higher is better.

## A.2. Additional O3 Details and Results on PDB: 1CLL

## O3 EXPERIMENTAL DETAILS

Table A.1. Experimental configurations for O3 protein structure optimisation (Boltz-2 generator, TM-score oracle on 1CLL). N: total Oracle budget; K: top-K batch selection; M: initial samples filtered to create subspace U; d: number of seed latents used to create the d − 1 dimensional $u ; d + 2 \colon$ initial points used to fit the Gaussian process in U; n<sub>rounds</sub>: number of BO acquisition rounds is set to $N - M - 2$ for filtered seed-latents (and $N - \left( d + 2 \right)$ for random seed latents); Wall-clock time of experiments run using a H100 GPU with 80Gb RAM.
<table><tr><td>Task</td><td>N</td><td>K</td><td>M</td><td>d</td><td> $n _ { \mathrm { r o u n d s } }$ </td><td>wall-clock time (mins)</td></tr><tr><td>n20_k2</td><td>20</td><td>6</td><td>10</td><td>5</td><td>8</td><td>4.5 (±2.7)</td></tr><tr><td>n50_k5</td><td>50</td><td>7</td><td>25</td><td>7</td><td>23</td><td>6.1 (±0.4)</td></tr><tr><td>n100_k10</td><td>100</td><td>10</td><td>50</td><td>5</td><td>48</td><td>10.6 (± 1.4)</td></tr><tr><td>n200_k20</td><td>200</td><td>10</td><td>100</td><td>10</td><td>98</td><td>20.6 (±8.6)</td></tr><tr><td>n500_k50</td><td>500</td><td>7</td><td>250</td><td>30</td><td>248</td><td>48.4588 (±2.2)</td></tr><tr><td>n1000_k100</td><td>1000</td><td>100</td><td>400</td><td>50</td><td>598</td><td>153.5 (±1.6)</td></tr></table>

## LOL INTERPOLATION

![](images/aea842ceda8184e0d68a5cc43a71fe29853ec5b7ff88f03c8a4c0cc701d51c40.jpg)  
Figure A.2. LOL interpolation trajectory in a Boltz-2 d = 2 surrogate subspace of two Calmodulin examples, using 5 intermediate values of $( w , 1 - w )$ for w $\in ( 0 , 1 )$ . Structures vary smoothly along the interpolation while remaining within the support of Boltz-2. For more details see (Bodin et al., 2024).

## COMPLEMENTARY O3 METRICS

![](images/10c9e3d0984db6793514f65212421af72ca12ee5f785dcf2000a4bbdce8ecac5.jpg)

(b)  
![](images/404eda6fb25dfaa34d5e31e1196b02830c3c8335f08b28fa0ecdbec89661da24.jpg)  
Figure A.3. Complementary O3 metrics. (a) Max-of-K counterpart to Fig. 4: BO versus random sampling at $N = 1 0 0$ and $K = 1 0$ (b) Mean-of-K counterpart to Fig. 5: subspace dimension sweep across budgets.

## O3 ABLATE BUDGET ALLOCATION

![](images/d892240a42f6250750ad1be66dd38d2412c9da35b5b1377c364e3bcc18b67602.jpg)

(b)  
![](images/6d5d89923d11c71cc7fd84cb9432a4a7c373b1f26aa60748974d6b684429595a.jpg)  
Figure A.4. O3 Ablation: Budget allocation M. (a) Mean-of-K TM Score reported for a fixed budget $N = 1 0 0 ,$ varying the initial number of samples used to create the U subspace as per Section 3.1, resulting in varying remaining BO budget, $N - M$ acquisition rounds. (b) Max-of-K counterpart. In all cases we keep the subspace-dimension constant $d = 7 , \bar { K } = 1 0 .$ , and draw $j = 1 0$ uniform random u samples as initial GP training data. Bars are the mean over 3 random seeds; error bars are ±1 std. Higher is better. Experiment uses a previous set of hyperparameters.

## O3 EFFECT OF GP COVARIANCE FUNCTION

![](images/356ba24b435240fdd13b63c3adec5f1232b22fd03cbcdbe96fc1dfbc3d256bd9.jpg)

![](images/e2899cac19209b3a21753a4dde5c3a275f09ce6fe60d2cf334a3dff0c85b4000.jpg)  
Figure A.5. O3 Ablation: GP covariance function. (a) Mean-of-K TM Score across different subspace dimensionality for an RBF kernel vs Matern 5/2 kernel as Gaussian process covariance function. (b) Max-of-K counterpart to (a). In all cases we keep the budge $N = 1 0 0$ , batch size $K = 1 0 .$ , with $j = \bar { 1 } 0$ uniform random u samples, as per Section 3. Bars are the mean over 5 random seeds; error bars are ±1 std. Higher is better. Experiment uses a previous set of hyperparameters.

## O3 EFFECTS OF SEED SELECTION STRATEGY

![](images/2c18bf9f8c0e422d367d4fbf5ab5ed9d0286b066d68521b797dbbabaaaaa4902.jpg)

(b)  
![](images/cad86ab62b283477bae9670705e208cdcd26fdcd17e155d6dd12a9e339d7147b.jpg)  
Figure A.6. O3 Ablation: Choice of seed selection strategy. (a) Mean-of-K TM Score on 1CLL for different approaches to construct the U subspace: either the “best d seeds” or “random d seeds” are used, across a sweep of subspace dimensionalities. Using random seeds means $\bar { \mathbf { M } } = 0 ,$ so number of BO rounds is $N - \left( d + 2 \right)$ , whereas best d seeds has fewer $N - \mathbf { \hat { { M } } } - 2$ . (b) Max-of-K counterpart to (a). In all cases we keep the budget $N = 1 0 0$ and batch size $K = 1 0$ . Bars are the mean over 3 random seeds; error bars are ±1 std. Higher is better.

## A.3. Additional FK-Steering Results on PDB: 1CLL

COMPLEMENTARY FK-STEERING METRICS

![](images/8639542ff48ee152eb68464bd2ab0fc91eaeb33e120cf7d5984a1a55a37a5723.jpg)  
Figure A.7. Effect of FK-steering λ hyperparameter across different budgets N. Max-of-K TM-score achieved by FK-steering on 1CLL. Higher λ increases the signal from the rewards and leads to better performance across budgets. Bars are means over 10 random seeds; error bars are ±1 std.

## FK-STEERING K VS $N / K$ TRADEOFF

![](images/77e5528312eb81c078b8c5c30394707befe3fed1862e605e3224509fc658f14f.jpg)

(b)  
![](images/d3bb17424bcb44f6a9af3110f1ebc2e5b2fb5c9511930e52550de6ef7263fc10.jpg)  
Figure A.8. FK-steering K vs N/K tradeoff for a set budget. (a) Mean-of-K TM Score for different approaches to allocate K particles and $N / K$ resampling steps per set budget of $N$ oracle calls. (b) Max-of-K counterpart to (a). For a given budget N, increasing the number of resampling steps $N / K$ at the cost of particle population K generally improves performance in the higher budget groups. In all cases $\lambda = 5 0$ . Bars are the mean over 10 random seeds; error bars are ±1 std. Higher is better.

## FK-STEERING REWARD TRAJECTORIES

![](images/85a0e84f616d49378b5a3dac5147e4766fe5f82a3bc30cb8ff17d1e8048b63d8.jpg)

(b)  
![](images/2b6376603460f485c30e110606923d375ca57b78a74dff2ff55effd582205b9b.jpg)  
Figure A.9. FK-steering reward trajectory for $K = 1 0$ particles with different number of resampling steps $N / K$ . (a) $N / K = 1 0$ resampling steps with a budget of $N = 1 0 0$ oracle calls. (b) $N / K = 5 0$ resampling steps with a budget of $N = 5 0 0$ oracle calls. In both cases $\lambda = 5 0$ . Colors preserve $N / K$ colorscheme from Fig. A.8. Resampling more frequently at highe $N / K$ propagates the reward signal better and reduces reward variance in the final resampling steps.

## A.4. Additional DPO Results on PDB: 1CLL

![](images/5d08989ebb6264f5e792f39146edc4d5b510460e7945cc3f532fff217389b367.jpg)  
Figure A.10. Online vs. offline DPO across oracle budgets. Max-of-K TM-score achieved by online and offline DPO on 1CLL, for oracle budgets $N \in \{ 1 0 0 , 5 0 0 , 1 0 0 0 \}$ . Bar are means over 5 random seeds; error bars show ±1 std. Online DPO improves consistently with N, while offline DPO remains flat across budgets.

## A.5. Compute resources used in PDB: 1CLL Experiments

Table A.2. Compute resources used by the Boltz-2 guidance PDB: 1CLL experiment reported in this paper. ‘Evals/run’ counts oracle (TM-score) evaluations, each of which corresponds to one full generation through Boltz-2. The FK-steering rows aggregate over the 3 values of λ swept per budget.
<table><tr><td>Experiment</td><td>Model / pipeline</td><td>Hardware</td><td>Evals/run</td><td># runs</td><td>Per-run wall-clock</td><td>Total wall-clock</td></tr><tr><td>O3 (Willis et al., 2025)</td><td>BOLTZ-2 (PF-ODE)</td><td>H100</td><td>6 budgets up to N=1000</td><td>5 per budget</td><td>~ 0.1 h</td><td>~3 h</td></tr><tr><td></td><td></td><td></td><td>N=20 N=50</td><td></td><td>2.7 m 2.8 m</td><td>13 m 14 m</td></tr><tr><td></td><td></td><td></td><td>N=100</td><td>5 5</td><td>3.1 m</td><td>16 m</td></tr><tr><td></td><td></td><td></td><td>N=200</td><td>5</td><td>2.9 m</td><td>15 m</td></tr><tr><td></td><td></td><td></td><td>N=500</td><td>5</td><td>3.4 m</td><td>17 m</td></tr><tr><td></td><td></td><td></td><td></td><td>5</td><td>18 m</td><td>1 h 31 m</td></tr><tr><td>Best K-of-N</td><td>BOLTZ-2</td><td>H100</td><td>N=1000</td><td></td><td>~ 0.1 h</td><td>~3 h</td></tr><tr><td></td><td></td><td></td><td>6 budgets up to N=1000</td><td>5 per budget</td><td>2.7 m</td><td>13 m</td></tr><tr><td></td><td></td><td></td><td>N=20</td><td>5</td><td>2.8 m</td><td>14 m</td></tr><tr><td></td><td></td><td></td><td>N=50 N=100</td><td>5 5</td><td>3.1 m</td><td>16 m</td></tr><tr><td></td><td></td><td></td><td>N=200</td><td>5</td><td>2.9 m</td><td>15 m</td></tr><tr><td></td><td></td><td></td><td>N=500</td><td>5</td><td>3.4 m</td><td>17 m</td></tr><tr><td></td><td></td><td></td><td>N=1000</td><td>5</td><td>18 m</td><td>1 h 31 m</td></tr><tr><td>FK-steering (Singhal et al., 2025)</td><td>BOLTZ-2</td><td>H100</td><td>6 budgets, 3 λ values each</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>N=20, 3 λ vals</td><td>10</td><td>3.3 m</td><td>1 h 39 m</td></tr><tr><td></td><td></td><td></td><td>N=50, 3 λ vals</td><td>10</td><td>3.8 m</td><td>1 h 54 m</td></tr><tr><td></td><td></td><td></td><td>N=100, 3 λ vals</td><td>10</td><td>4.1 m</td><td>2h3m</td></tr><tr><td></td><td></td><td></td><td>N=200, 3 λ vals</td><td>10</td><td>5 m</td><td>2 h 30 m</td></tr><tr><td></td><td></td><td></td><td>N=500, 3 λ vals</td><td>10</td><td>6.9 m</td><td>3 h 27 m</td></tr><tr><td></td><td></td><td></td><td>N=1000, 3λ vals</td><td>10</td><td>11 m</td><td>5 h 30 m</td></tr><tr><td>Online DPO (Rafailov et al., 2023; Wallace et al., 2023)</td><td>BOLTZ-2 (fine-tuned)</td><td>H100</td><td>6 budgets up to N=1000</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>N=20</td><td>4</td><td>4m</td><td>16 m</td></tr><tr><td></td><td></td><td></td><td>N=50</td><td>4</td><td>6 m</td><td>24 m</td></tr><tr><td></td><td></td><td></td><td>N=100</td><td>5</td><td>11 m</td><td>55 m</td></tr><tr><td></td><td></td><td></td><td>N=200</td><td>3</td><td>17 m</td><td>51 m</td></tr><tr><td></td><td></td><td></td><td>N=500</td><td>5</td><td>44 m</td><td>3 h 40 m</td></tr><tr><td></td><td></td><td></td><td>N=1000</td><td>4</td><td>1 h 31 m</td><td>6 h 04 m</td></tr><tr><td>Offline DPO (Rafailov et al., 2023; Wallace et al., 2023) BoLTZ-2 (fine-tuned)</td><td></td><td>H100</td><td>6 budgets up to N=1000</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>N=20</td><td>4</td><td>4 m</td><td>16 m</td></tr><tr><td></td><td></td><td></td><td>N=50</td><td>5</td><td>7m</td><td>35 m</td></tr><tr><td></td><td></td><td></td><td>N=100</td><td>4</td><td>18 m</td><td>1 h 12 m</td></tr><tr><td></td><td></td><td></td><td>N=200</td><td>5</td><td>56 m</td><td>4h 40 m</td></tr><tr><td></td><td></td><td></td><td>N=500</td><td>4</td><td>5 h40m</td><td>22 h 40 m</td></tr><tr><td></td><td></td><td></td><td>N=1000</td><td>3</td><td>21 h 57 m</td><td>65 h 51 m</td></tr></table>

## B. Experimental Setup and Results on PDB: 9EEH

## B.1. MolProbity Oracle

TM-score (Section 4) requires a ground-truth structure, which is usually unavailable in the design settings this paper targets. We therefore also evaluate against a reference-free oracle that scores the physical plausibility of a generated structure, built on MolProbity (Chen et al., 2010; Williams et al., 2018). The oracle runs six checks. Backbone dihedral quality comes from a Ramachandran analysis and Cβ deviation count via mmtbx. Steric quality is the clashscore, the number of serious all-atom overlaps (probe “bad overlap“ contacts with a gap below −0.4 A) per 1000 atoms, computed after adding and<sup>˚</sup> optimising all hydrogens with reduce. Covalent geometry is scored as RMS Z-scores of backbone bond lengths and bond angles against the Engh-Huber reference values, and peptide planarity contributes counts of cis-nonPro and twisted ω angles. These terms are combined into a single log-weighted penalty following the MolProbity score formulation, which we flip and normalise to a validity score in [0, 1], with higher values indicating fewer physical violations.

## B.2. PDB: 9EEH Results

We repeat the experiment of Section 5.1 on E. coli aspartate transcarbamoylase (PDB: 9EEH) (Miller et al., 2026), scored with the reference-free MolProbity oracle described above. There are three key differences between this setup and the one with the TM-score oracle on the calmodulin protein (PDB: 1CLL). First, 9EEH was deposited to the PDB after the training cutoff of Boltz-2, thus it is likely an out-of-distribution task for the base model, whereas 1CLL is not. Second, 9EEH is a 7,232-atom complex, which is much larger than 1CLL with 1,184 atoms. Third, the MolProbity score measures physical plausibility rather than similarity to a reference structure, and its effective range is a lot narrower, with most structures typically scoring between 0.5 and 0.7 under our MolProbity implementation.

Fig. B.1 reports the mean-of-K MolProbity score. O3 scores highest at all but the smallest budget, where it matches Best K-of-N. Best K-of-N has a relatively constant performance across budgets, as expected under fixed K/N ratio. DPO improves with N, although it performs significantly worse than Best K-of-N at low budgets. FK-steering performs poorly throughout and does not improve with budget, in contrast to 1CLL where it does better than Best K-of-N for budgets above N = 100. In Fig. B.2 under max-of-K, Best K-of-N is strongest at large budgets and both it and FK-steering improve with N, while O3 is unchanged.

![](images/717b982cb6cd985ddb019ec4c84e4458c4af40ed81c2d675d64d77cd8a16490f.jpg)  
Figure B.1. Mean-of-K MolProbity score across oracle budgets. Comparison of FK-steering, DPO (in the online setting), Best K-of-N, and O3 on 9EEH across the six (N, K) configurations of Section 4. Bars are the mean over 3 random seeds; error bars are ±1 std. Higher is better.

![](images/7e4c4129ae39061c3b9773941a155547d48f61fb9ae1f434f032137cdcc3d293.jpg)  
Figure B.2. Max-of-K MolProbity score across oracle budgets. Comparison of FK-steering, DPO, Best K-of-N, and O3 on 9EEH across the six (N, K) configurations of Section 4, reporting the maximum MolProbity score across the returned batch of K candidate structures. Bars are the mean over 3 random seeds; error bars are ±1 std. Higher is better.

Under max-of-K, O3 performs worse than Best K-of-N, and worse than all methods at the highest budget. We hypothesise that this is because it is the only method that requires sampling Boltz-2 with the probability-flow ODE while the other three methods use the stochastic Boltz-2 sampler which improves the output diversity. Under mean-of-K this yields worse performance overall because the lower-scoring samples drag down the average, however it improves the max-of-K scores, where only the best sample matters and increasing N gives better scores even without any guidance signal (i.e. Best K-of-N baseline).

FK-steering is the only method that scores intermediate denoised predictions, and it is the weakest method under the MolProbity oracle. TM-score varies smoothly with structural accuracy: the same global fold already scores above 0.5, and refinement moves the score toward 1, so intermediate predictions receive informative scores. MolProbity instead measures local geometry, in which residual noise dominates until the final denoising steps. Since the quality of the denoised predictions is much worse at the intermediate steps than at the final ones, this leaves noise-sensitive intermediate MolProbity scores nearly uninformative. Moreover, since FK-steering relies on a positive noise scale for particle diversity, in Boltz-2 resampling is concentrated in the first three quarters of the denoising trajectory, which exacerbates the issues with signal quality from the MolProbity oracle at intermediate steps.

Finally, the differences between methods on this target are small, and the error bars over three seeds overlap at severa budgets. Three of the four methods behave much as they do on 1CLL — O3 scores highest, DPO improves with N, and Best K-of-N is relatively constant — with FK-steering the exception, for the reason above. We therefore read these orderings as consistent trends across budgets rather than as individually meaningful differences, and note that guidance is less effective on this target than on 1CLL with TM-score.

## B.3. Additional O3 Details and Results on PDB: 9EEH

## O3 EXPERIMENTAL DETAILS

Table B.1. Experimental configurations for O3 protein structure optimisation (Boltz-2 generator, MolProbity oracle on 9EEH). N: total Oracle budget; K: top-K batch selection; M: initial samples filtered to create subspace U; d: number of seed latents used to create the $d - 1$ dimensional $u ; d + 2 \colon$ initial points used to fit the Gaussian process in $\mathcal { U } ; n _ { \mathrm { r o u n d s } } \mathrm { : }$ number of BO acquisition rounds is set to $N - M - 2$ for filtered seed-latents (and $N - \left( d + 2 \right)$ for random seed latents); Wall-clock time of experiments run using a H100 GPU with 80Gb RAM.
<table><tr><td>Task</td><td>N</td><td>K</td><td>M</td><td>d</td><td> $n _ { \mathrm { r o u n d s } }$ </td><td>wall-clock time (mins)</td></tr><tr><td>n20_k2</td><td>20</td><td>2</td><td>10</td><td>5</td><td>8</td><td> $9 . 1 \left( \pm 0 . 1 \right)$ </td></tr><tr><td>n50_k5</td><td>50</td><td>5</td><td>25</td><td>5</td><td>23</td><td> $1 9 . 2 \ : ( \pm 0 . 3 ) $ </td></tr><tr><td>n100_k10</td><td>100</td><td>10</td><td>50</td><td>20</td><td>48</td><td> $4 0 . 0 \left( \pm 0 . 7 \right)$ </td></tr><tr><td>n200_k20</td><td>200</td><td>20</td><td>100</td><td>20</td><td>98</td><td> $7 5 . 9 \ : ( \pm 2 . 0 )$ </td></tr><tr><td>n500_k50</td><td>500</td><td>50</td><td>250</td><td>30</td><td>248</td><td> $2 2 4 . 2 \ : ( \pm 9 . 0 )$ </td></tr><tr><td>n1000_k100</td><td>1000</td><td>100</td><td>400</td><td>50</td><td>598</td><td> $3 6 8 . 0 ( \pm 7 . 4 ) $ </td></tr></table>

O3 EFFECTS OF SEED SELECTION STRATEGY  
(a)  
![](images/77a9ae0e6a8b5738e760c86b8b631600abf248192a1aa21d1eec17d377f4f886.jpg)

(b)  
![](images/40ba4cd36933b84d799ed0de47e8042990bb3dcc1df1c09054acbba393e17086.jpg)  
Figure B.3. O3 Ablation: Choice of seed selection strategy. (a) Mean-of-K MolProbity Score on 9EEH for different approaches to construct the U subspace: either the “best d seeds” or “random d seeds” are used, across a sweep of subspace dimensionalities. Using random seeds means $\mathbf { \bar { M } } = 0 ,$ , so the number of BO rounds is $N - \left( d + 2 \right)$ , whereas best d seeds has fewer $N - M - 2 . \left( \mathbf { b } \right)$ Max-of-K counterpart to (a). In all cases, we keep the budget $N = 1 0 0$ and batch size $K = 1 0$ . Bars are the mean over 3 random seeds; error bars are ±1 std. Higher is better.
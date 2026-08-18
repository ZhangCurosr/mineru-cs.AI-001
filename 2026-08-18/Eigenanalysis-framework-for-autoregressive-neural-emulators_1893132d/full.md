# Eigenanalysis framework for autoregressive neural emulators of multi-scale chaotic dynamics

Conrad Ainslie<sup>1</sup>, Pedram Hassanzadeh<sup>2,3</sup>, Michael W. Mahoney<sup>4,5,6</sup> and Ashesh Chattopadhyay <sup>1∗</sup>

<sup>1</sup>Department of Applied Mathematics, University of California, Santa Cruz, Santa Cruz, 95064, CA

<sup>2</sup>Department of Geophysical Sciences, University of Chicago, Chicago, IL 60637

<sup>3</sup>Committee on Computational and Applied Mathematics, University of Chicago, Chicago, IL 60637

<sup>4</sup>International Computer Science Institute, Berkeley, CA 94720

<sup>5</sup>Lawrence Berkeley National Laboratory, Berkeley, CA 94720

<sup>6</sup>Department of Statistics, University of California, Berkeley, CA 94720

## Abstract

Neural autoregressive models have rapidly emerged as powerful emulators of high-dimensional chaotic systems, yet their long-term instability and error growth remain poorly understood, leading to ad-hoc solutions. Here, we develop an eigenanalysis framework that reveals the dynamical origin of this error growth. By analyzing the Jacobian of the learned one-step update map with respect to the state, we show how inference-time error growth, and thus model stability, is governed by its spectral radius. Direct-step architectures (models that predict the next state from the previous one) generically admit unstable eigenvalues with magnitudes exceeding one, explaining the rapid divergence of these widely used models. In contrast, integration-constrained models (where the time-derivative is estimated and integrated with a higher-order integrator) collapse their eigenspectrum onto the unit circle, yielding neutral stability and a universal linear error-scaling law. The largest eigenvalue of this Jacobian provides an architecture-agnostic, a priori diagnostic of shortterm skill, long-term stability, and spectral bias, without requiring an expensive rollout. Leveraging this theory, we introduce a stability-promoting loss that explicitly regularizes Jacobian-driven error amplification, improving both forecast accuracy and dynamical robustness. Demonstrated across 29 models spanning two architectures, several explicit and implicit integrators, and multiple loss functions on the Kuramoto–Sivashinsky system, our results establish a theoretical foundation for the design and evaluation of neural emulators of chaotic multi-scale dynamics. More broadly, our framework is a step toward the kind of a priori stability analysis that numerical analysis provides for discretizations of diferential equations, and that scientific machine learning currently lacks.

## Significance Statement

Neural networks that step a physical system forward in time now rival traditional simulations of chaotic, multi-scale flows such as the atmosphere, climate, and ocean, and they can run far faster than traditional numerical methods. They can also drift or blow up over long rollouts. Remedies are currently found by trial and error, each tested by running the simulation that the network was built to replace. Here, we develop an eigenanalysis theory of these networks, which shows that one quantity, obtained from a single derivative of the trained model, predicts how fast its error grows, before any simulation is run. We find that networks that predict the next state directly amplify errors far faster than the physics allows, while networks that instead integrate a learned rate of change do not.

## 1 Introduction

Neural autoregressive models have become efective data-driven emulators of high-dimensional chaotic dynamical systems [Fan et al., 2020, Floryan, 2024, Li et al., 2022]. Their most visible success is in Earth system modeling, where they now rival or outperform operational numerical weather prediction [Bi et al., 2023, Lam et al., 2022, Pathak et al., 2022, Price et al., 2025].

Despite this progress, long-term emulation remains dificult. Many models become unstable or excessively difusive over long rollouts [Chattopadhyay et al., 2023b, Keisler, 2022, Lai et al., 2025, Lippe et al., 2023, Pedersen et al., 2025].

A growing number of models are now stable over long horizons, but this stability is typically achieved through extensive hyperparameter optimization; and even carefully tuned models can still blow up. Each such failure tends to be addressed with a new, problem-specific fix rather than by identifying the mechanism that produced it, leaving the field without a principled account of when and why these models lose stability [Guan et al., 2025, Pedersen et al., 2025, Sambamurthy and Chattopadhyay, 2025, Watt-Meyer et al., 2025].

A leading explanation for this instability in multi-scale systems, such as turbulent flows with a decaying energy spectrum, is the spectral bias of the network [Chattopadhyay et al., 2023b, Yu et al., 2024], whereby a model trained to predict a single step fails to capture the high-wavenumber dynamics. This error is thought to grow during autoregressive rollout through the coupling between small and large scales until it corrupts the resolved dynamics. If spectral bias is the underlying cause, however, its consequences are realized only through the way one-step errors accumulate from step to step. The role of this error propagation, as distinct from the one-step error itself, has not been made precise, and most stabilization strategies remain disconnected from any quantitative theory of how errors grow during rollout inference.

One line of work points to the integration scheme used to advance the model in time. Krishnapriyan et al. [2023] showed that constraining a model with a higher-order integrator, such as fourth-order Runge–Kutta, tends yield convergent integration analogous to classical numerical schemes. Chattopadhyay et al. [2023b] combined such a hard constraint with a spectral regularizer and reported reduced error growth and longer stable rollouts, and subsequent work has scaled this idea to coupled climate and high-resolution ocean models, producing stable rollouts over centuries of integration [Chattopadhyay et al., 2024, Guan et al., 2025, Lupin-Jimenez et al., 2025]. Related stabilization strategies include refinement [Lippe et al., 2023], noise-based regularization [Stachenfeld et al., 2021, Wikner et al., 2022], and stochastic modeling [Chattopadhyay et al., 2023a, Pedersen et al., 2025, Sambamurthy and Chattopadhyay, 2025].

Despite this empirical evidence, it remains unclear why an integration constraint should help. By analogy with numerical analysis, it is natural to assume that incorporating physical insight into a constraint inspired by time integration may aid stability; however, intuitions about combining physical reasoning and machine learning can be misleading [Krishnapriyan et al., 2021, 2023, Sakarvadia et al., 2025], and learned spatiotemporal models can behave counterintuitively [Yu et al., 2025], including how performance scales with choice of the discrete time step [Bi et al., 2023, Chattopadhyay et al., 2022]. A general theory of inference-time stability for these models is still lacking, with rigorous results largely confined to the linear case [Floryan, 2024], and the efect of the integration scheme, loss function, and architecture on accuracy and stability has not been studied systematically.

This raises a set of questions. Does an integration constraint control the rate at which errors grow during rollout, and does it also improve short-term accuracy? Can a single quantity, computed before any long rollout, predict both? Useful answers should not depend on the particular architecture, integration scheme, loss function, or system, and they should not require long emulations from many initial conditions.

We answer these questions through an eigenanalysis of the Jacobian of the learned one-step update map (Fig. 1). This Jacobian is taken with respect to the system state, which is the derivative that sets the stability of a discrete map in dynamical systems theory [Floryan, 2024], and not with respect to the network parameters, which is the derivative returned by backpropagation and the one whose spectrum is usually studied in machine learning [Liao and Mahoney, 2021]. The two derivatives have diferent arguments and diferent roles, and only the state derivative enters the error dynamics at inference. Linearizing the error dynamics shows that inference-time error growth is governed by the spectral radius of this Jacobian, so its largest eigenvalue is an a priori diagnostic that predicts error growth during inference. This can be accomplished without resorting to expensive long rollouts, and it is independent of architecture, integration scheme, and loss function. This single quantity diagnoses the amplification of error during rollout; together with the one-step error, it accounts for short-term accuracy and the growth of spectral bias (Section 3.3). The analysis also explains the empirical record. Direct-step models, which predict the next state directly, generically admit eigenvalues with magnitude well above one and have no mechanism to control them, which accounts for their rapid divergence. Integration-constrained models, in which the network estimates a time derivative that is advanced by a numerical integrator, have a Jacobian whose eigenvalues collapse onto the unit circle and yield neutral stability together with a linear error-scaling law. Building on this, we introduce a stability-promoting loss that directly regularizes the Jacobian-driven amplification of errors and improves both accuracy and stability.

We develop this theory for the Kuramoto–Sivashinsky (KS) system, and we evaluate it across the full suite of models studied here, spanning two architectures, several explicit and implicit integration schemes, and loss functions with and without spectral regularization. The remainder of the paper presents the linear stability theory, the empirical results, the limits of the eigenvalue-only view and its distinctions from classical linear stability analysis, and the methods, including all the explicit integration schemes and an implicit integration scheme based on implicit-layer and deep-equilibrium models [Kawaguchi, 2021]. The full model suite is listed in SI, Section 1. The results obtained with the implicit constraint, together with the derivation of its Jacobian, are reported in SI, Section 2. The Fourier-space analysis of spectral bias is given in SI, Section 3, and the normality analysis that supports the eigenvalue approximation is given in SI, Section 4.

## 2 Linear stability analysis

We analyze the inference-time stability of a trained neural autoregressive model by linearizing its error dynamics during emulation, following the linear stability analysis of discrete dynamical systems.

## 2.1 Learned dynamics and the Jacobian

Let the true system advance one step through an operator G,

$$
{ \bf u } _ { T } ( t + \Delta t ) = { \bf G } \left[ { \bf u } _ { T } ( t ) \right] ,\tag{1}
$$

with $\mathbf { u } _ { T }$ the true state, usually available only at the initial condition $t = t _ { 0 }$ . A neural emulator approximates G by a learned operator $\tilde { \mathbf { G } }$ and advances its own predicted state $\mathbf { u } _ { p }$

$$
\mathbf { u } _ { p } ( t + \Delta t ) = \tilde { \mathbf { G } } \left[ \mathbf { u } _ { p } ( t ) \right] .\tag{2}
$$

Both states are functions of time. From here on we suppress the time argument whenever it is not needed and write $\mathbf { u } _ { T }$ and $\mathbf { u } _ { p } .$ , restoring it only where the rollout step matters.

The learned operator is built from a neural network $\mathcal { N } ( \cdot ; \theta )$ with parameters θ (see Section 6.2). A direct-step model sets $\tilde { \mathbf { G } } [ \mathbf { u } _ { T } ] = \mathcal { N } ( \mathbf { u } _ { T } ; \theta )$ , predicting the next state in one shot. An integrationconstrained model instead estimates a time derivative with the network and advances it with a numerical integrator H,

$$
\tilde { \bf G } [ { \bf u } _ { T } ] = { \bf u } _ { T } + \Delta t { \bf H } [ \mathcal { N } ( { \bf u } _ { T } ; \boldsymbol { \theta } ) ] .\tag{3}
$$

Equation (3) is the explicit form of the constraint. The implicit form, in which the network is evaluated at the updated state, is given in Section 6.4.

The central object of our analysis is the Jacobian of the learned update map with respect to the state,

$$
\mathbf { J } = \nabla \tilde { \mathbf { G } } [ \mathbf { u } _ { T } ] .\tag{4}
$$

Throughout, ∇ denotes the gradient with respect to the state ${ \bf u } _ { T } .$ evaluated at the current true state, and not the gradient with respect to the network parameters θ. In machine learning, the derivative of a trained model most often refers to the derivative with respect to $\theta ,$ which is what backpropagation returns and what the optimizer acts on. Equation (4) is the other derivative, taken at fixed $\theta ,$ and it is the one that appears in the error recursion below. The Jacobian is the sensitivity of the one-step prediction to a perturbation of its input, which is the quantity that governs how errors propagate during rollout. It is distinct from the parameter gradients used in training, and it is rarely examined when these models are evaluated.

![](images/e0e582e4ea113ec3c7543d0026d2ee1b0c3ec6c984f4c4140c3a3f397bf6055c.jpg)  
Figure 1: Schematic representation of the main contributions of this paper. An eigenvalue decomposition of the Jacobian of the learned one-step update map, taken with respect to the system state and not with respect to the network parameters, yields a linear theory for the role the Jacobian plays in short-term error growth, the rate of error accumulation over rollout, and the spectral fidelity of the emulation. Direct-step models, which predict the next state in one shot, admit eigenvalues outside the unit circle and diverge during rollout, whereas integration-constrained models, in which the network estimates a time derivative that is advanced by a numerical integrator, collapse their eigenspectrum onto the unit circle and remain neutrally stable, so that a perturbation is neither amplified nor damped from one step to the next (Section 2.3). The resulting stability-promoting loss regularizes the Jacobian-driven amplification of error, reducing the rollout error-growth rate: for some models this appears directly as eigenvalues pulled toward the unit circle (illustrated), while more generally it suppresses the realized, direction-dependent amplification of the error without necessarily changing $\lvert \lambda _ { \mathrm { m a x } } \rvert$ (Section 3.4).

## 2.2 Error dynamics

Define the prediction error ${ \bf e } ( t ) = { \bf u } _ { T } ( t ) - { \bf u } _ { p } ( t )$ . Subtracting Eq. (2) from Eq. (1) gives

$$
\mathbf { e } ( t + \Delta t ) = \mathbf { G } \left[ \mathbf { u } _ { T } \right] - \tilde { \mathbf { G } } \left[ \mathbf { u } _ { p } \right] .\tag{5}
$$

Linearizing $\tilde { \mathbf { G } }$ about the true state yields

$$
\mathbf { e } ( t + \Delta t ) = \underbrace { \mathbf { G } \left[ \mathbf { u } _ { T } \right] - \tilde { \mathbf { G } } \left[ \mathbf { u } _ { T } \right] } _ { \epsilon ( t ) } + \underbrace { \nabla \tilde { \mathbf { G } } \left[ \mathbf { u } _ { T } \right] } _ { \mathbf { J } } \mathbf { e } ( t ) + \mathcal { O } \bigl ( \| \mathbf { e } ( t ) \| _ { 2 } ^ { 2 } \bigr ) .\tag{6}
$$

The first term $\epsilon ( t )$ is what we call the generalization error, the one-step error of the learned operator at the true state, and the quantity that autoregressive models are trained to minimize. The second term, $\mathbf { J e } ( t )$ , is what we call the propagated error, the accumulated error carried forward through the linearized map, and J in this role is the error propagator. Training and validation control only $\epsilon ( t )$ whereas J, which sets the size of the propagated error, is rarely examined.

At the initial condition the two error notions coincide. Since $\mathbf { u } _ { p } ( t _ { 0 } ) = \mathbf { u } _ { T } ( t _ { 0 } )$ , the first step incurs only the generalization error, so $\mathbf { e } ( t _ { 0 } + \Delta t ) = \pmb { \epsilon } ( t _ { 0 } )$ . Writing $t _ { n } = t _ { 0 } + n \Delta t$ and unrolling Eq. (6) over n steps, we obtain

$$
\mathbf { e } ( t _ { n } ) = \epsilon ( t _ { n - 1 } ) + \sum _ { i = 0 } ^ { n - 2 } \left( \prod _ { j = i + 1 } ^ { n - 1 } \mathbf { J } ( t _ { j } ) \right) \epsilon ( t _ { i } ) + { \mathcal { O } } \bigl ( \| \mathbf { e } ( t _ { n - 1 } ) \| _ { 2 } ^ { 2 } \bigr ) ,\tag{7}
$$

with $\mathbf { J } ( t _ { j } ) ~ = ~ \nabla \tilde { \mathbf { G } } [ \mathbf { u } _ { T } ( t _ { j } ) ]$ the Jacobian at step $j .$ Total error growth therefore has two sources. The generalization error ϵ reflects intrinsic model quality and can be reduced through better data, architecture, or capacity. The product of Jacobians is the error propagator over the intervening steps, and it determines whether those errors are amplified or damped, thus controlling stability during model rollout. Equation (7) cleanly separates model fidelity, carried by ϵ, from dynamical stability, carried by J. Characterizing J across architectures, loss functions, and integration constraints, and relating it to model quality and stability during emulation, is the main contribution of this work.

## 2.3 Eigenvalues and the Jacobian of each model class

The Jacobian admits the eigendecomposition

$$
\mathbf { J } = \mathbf { V } \pmb { \Sigma } \mathbf { V } ^ { - 1 } ,\tag{8}
$$

with V the eigenvector matrix and Σ the diagonal matrix of eigenvalues $\lambda _ { 1 } , \ldots , \lambda _ { d }$ for a system of dimension d. The magnitude of the largest eigenvalue $\lvert \lambda _ { \mathrm { m a x } } \rvert$ sets the local stability. The dynamics are locally unstable when $| \lambda _ { \operatorname* { m a x } } | > 1$ , in which case a perturbation of the state is amplified at every step and grows geometrically in the step count, and neutrally stable when $\left| \lambda _ { \operatorname* { m a x } } \right| = 1$ , in which case a perturbation is neither amplified nor damped from one step to the next, so an error already present is carried forward at fixed size. For non-normal J, as is typical of neural-network Jacobians, the eigenvalues bound the asymptotic growth of perturbations but not their transient amplification, which also depends on the orientation of the error relative to the eigenvectors; Section 3.3 quantifies this efect. SI, Section 4 shows that the integration constraint renders J near-normal, with a departure from normality of $\mathcal { O } ( \Delta t ^ { 2 } )$ , so the eigenvalue approximation used below is valid.

The two model classes have very diferent Jacobians. For a direct-step model,

$$
\mathbf { J } = \nabla \mathcal { N } ( \mathbf { u } _ { T } ; \boldsymbol { \theta } ) ,\tag{9}
$$

which places no intrinsic constraint on the spectrum. On the other hand, diferentiating the integrationconstrained map of Eq. (3) yields

$$
\mathbf { J } = \mathbf { I } + \Delta t \nabla \mathbf { H } [ \mathcal { N } ( \mathbf { u } _ { T } ; \boldsymbol { \theta } ) ] .\tag{10}
$$

Because $\Delta t \ll 1$ , the eigenvalues of Eq. (10) of an integration-constrained model stay close to one, and the model is near-neutrally stable by design. Direct-step models lack this control, since their Jacobian has no explicit dependence on $\Delta t ,$ and are thus prone to instability. Equation (10) is the Jacobian of the explicitly constrained map. Diferentiating the implicit map instead gives $\dot { \mathbf { J } } = ( \mathbf { I } - \Delta t \nabla \mathbf { H } [ \mathcal { N } ] ) ^ { - 1 }$ which reduces to Eq. (10) to first order in $\Delta t$ . SI, Section 2.2 derives this expression and shows that the inverse form places an eigenvalue outside the unit circle only when the learned tangent operator has an expanding direction.

The constraint fixes the form of J but not the size of the learned tangent operator $\nabla { \bf H } [ \mathcal { N } ( { \bf u } _ { T } ; \theta ) ]$ and the boundedness of that operator is a property the network acquires during training, rather than being an algebraic consequence of Eq. (10). For the true KS tangent operator at the resolution used here, $\Delta t k _ { \operatorname* { m a x } } ^ { 4 } = 1 . 0 7 \times 1 0 ^ { 3 }$ , so an explicit Euler step applied to the exact linearization would be unstable by three orders of magnitude. The learned operators instead satisfy $\| \Delta t \nabla \mathbf { H } [ \mathcal { N } ] \| \sim 1 0 ^ { - 3 }$ read from the spectral spread of Fig. 2(d). The near-unit spectrum therefore reflects the smoothness of the learned time derivative at high wavenumbers, which is the same property that produces the spectral bias analyzed in SI, Section 3, and not the algebra of $\mathbf { I } + \Delta t \nabla \mathbf { H }$ alone.

## 2.4 Error scaling for integration-constrained models

We first derive the error growth over a single step, and we then extend it to p steps for as long as the linear approximation holds.

From the linearized recursion of Eq. (6), the error advances by one step as ${ \bf e } ( t _ { k + 1 } ) = \epsilon ( t _ { k } ) +$ $\mathbf { J } ( t _ { k } ) \mathbf { e } ( t _ { k } ) + \mathcal { O } ( \| \mathbf { e } ( t _ { k } ) \| _ { 2 } ^ { 2 } )$ ), with $\mathbf { J } ( t _ { k } )$ the Jacobian at step k. For an integration-constrained model $\mathbf { J } ( t _ { k } ) = \mathbf { I } + \Delta t \nabla \mathbf { H } [ \mathcal { N } ( \mathbf { u } _ { T } ( t _ { k } ) ) ]$ from Eq. (10), so

$$
\mathbf { e } ( t _ { k + 1 } ) = \epsilon ( t _ { k } ) + \left( \mathbf { I } + \Delta t \nabla \mathbf { H } [ \mathcal { N } ( \mathbf { u } _ { T } ( t _ { k } ) ) ] \right) \mathbf { e } ( t _ { k } ) + \mathcal { O } \big ( \| \mathbf { e } ( t _ { k } ) \| _ { 2 } ^ { 2 } \big ) .\tag{11}
$$

At the initial condition ${ \bf u } _ { p } ( t _ { 0 } ) = { \bf u } _ { T } ( t _ { 0 } ) , \mathrm { s o } { \bf e } ( t _ { 0 } ) = { \bf 0 }$ and the first step incurs only the generalization error, $\mathbf { e } ( t _ { 1 } ) = \epsilon ( t _ { 0 } )$ . Applying Eq. (11) once more, at $k = 1$

$$
\mathbf { e } ( t _ { 2 } ) = \epsilon ( t _ { 1 } ) + ( \mathbf { I } + \Delta t \nabla \mathbf { H } [ \mathcal { N } ( \mathbf { u } _ { T } ( t _ { 1 } ) ) ] ) \mathbf { e } ( t _ { 1 } ) + \mathcal { O } \big ( \| \mathbf { e } ( t _ { 1 } ) \| _ { 2 } ^ { 2 } \big ) .\tag{12}
$$

Because $\Delta t$ is small and $\tilde { \mathbf { G } }$ is continuous, the generalization error is nearly unchanged over one step, $\pmb \epsilon ( t _ { 1 } ) \approx \pmb \epsilon ( t _ { 0 } ) = \mathbf { e } ( t _ { 1 } )$ . Substituting this into the first term gives

$$
\mathbf { e } ( t _ { 2 } ) = ( 2 \mathbf { I } + \Delta t \nabla \mathbf { H } [ \mathcal { N } ( \mathbf { u } _ { T } ( t _ { 1 } ) ) ] ) \mathbf { e } ( t _ { 1 } ) + \mathcal { O } \big ( \| \mathbf { e } ( t _ { 1 } ) \| _ { 2 } ^ { 2 } \big ) .\tag{13}
$$

The matrix acting on $\mathbf { e } ( t _ { 1 } )$ in Eq. (13) is 2I plus a correction of order $\Delta t ,$ so for $\Delta t \ll 1$ its largest eigenvalue is close to 2 and $\lVert \mathbf { e } ( t _ { 2 } ) \rVert \approx 2 \left. \lambda _ { \mathrm { m a x } } \right. \lVert \mathbf { e } ( t _ { 1 } ) \rVert$ This is due to the tight clustering of the spectrum. Since the eigenvalues of J are nearly coincident $( \mathrm { F i g . 2 ( c , d ) } )$ $\lVert \mathbf { M e } \rVert \approx \left| \lambda _ { \mathrm { m a x } } ( \mathbf { M } ) \right| \left\| \mathbf { e } \right\|$ , where M is any square matrix with tightly clustered eigenvalues at unity, holds regardless of the orientation of $\mathbf { e } ,$ and the spectrum of $( 2 \mathbf { I } + \Delta t \nabla \mathbf { H } [ \mathcal { N } ( \mathbf { u } _ { T } ( t _ { 1 } ) ) ] )$ coincides with $2 \left| \lambda _ { \operatorname* { m a x } } \right|$ up to $\mathcal { O } ( \Delta t )$ . The error therefore doubles at the first comparison, which is the $k = 1$ reference line in Fig. $\mathrm { 3 ( a ) }$ . SI, Section 4 shows that this directional insensitivity follows from the near-identity structure of Eq. (10) and not from the clustering of the eigenvalues alone. Section 3.3 examines the cases where this directional insensitivity breaks down.

We now extend this to $p$ steps. Iterating Eq. (11) from the initial condition and substituting the integration-constrained Jacobian gives

$$
\mathbf { e } ( t _ { p } ) = \sum _ { i = 0 } ^ { p - 1 } \epsilon ( t _ { i } ) + \Delta t \sum _ { i = 0 } ^ { p - 2 } \sum _ { j = i + 1 } ^ { p - 1 } \nabla \mathbf { H } [ \mathcal { N } ( \mathbf { u } _ { T } ( t _ { j } ) ) ] \epsilon ( t _ { i } ) + \mathcal { O } \big ( \| \mathbf { e } ( t _ { p - 1 } ) \| _ { 2 } ^ { 2 } \big ) + \mathcal { O } \big ( \Delta t ^ { 2 } \big ) ,\tag{14}
$$

whose leading term is the running sum of generalization errors. The same two approximations apply for as long as the rollout error stays small. The $\mathcal { O } ( \Delta t ^ { 2 } )$ remainder and the explicit $\Delta t$ double sum are subdominant because $p \Delta t \ll 1$ over the step counts considered here, and the continuity of G<sup>˜</sup> keeps the generalization error nearly constant across the early rollout, $\pmb \epsilon ( t _ { i } ) \approx \pmb \epsilon ( t _ { 0 } ) = \mathbf { e } ( t _ { 1 } )$ for $i = 0 , \ldots , p - 1$ The leading sum then collapses to $p \mathbf { e } ( t _ { 1 } )$ , and

$$
\mathbf { e } ( t _ { p } ) = \left( p \mathbf { I } + \Delta t \sum _ { j = 1 } ^ { p - 1 } j \nabla \mathbf { H } [ \mathcal { N } ( \mathbf { u } _ { T } ( t _ { j } ) ) ] \right) \mathbf { e } ( t _ { 1 } ) + \mathcal { O } \big ( \| \mathbf { e } ( t _ { p - 1 } ) \| _ { 2 } ^ { 2 } \big ) + \mathcal { O } \big ( \Delta t ^ { 2 } \big ) .\tag{15}
$$

The bracket is now $p \mathbf { I }$ plus a correction that is $\mathcal { O } ( p \Delta t )$ relative to the leading term, so for $p \Delta t \ll 1$ its largest eigenvalue is close to $p ,$ giving the linear scaling law

$$
\lVert \mathbf { e } ( t _ { p } ) \rVert \approx p \left. \lambda _ { \operatorname* { m a x } } \right. \lVert \mathbf { e } ( t _ { 1 } ) \rVert ,\tag{16}
$$

with $| \lambda _ { \operatorname* { m a x } } | \approx 1$ the largest eigenvalue of the one-step Jacobian J. The 2I result of Eq. (13) is the $p = 2$ case. The slope grows linearly with the step count p, reproducing the slopes 2, 11, 101, and 1001 measured at lead times $k = 1 , 1 0 , 1 0 0 , 1 0 0 0$ (that is, $p = 2 , 1 1 , 1 0 1 , 1 0 0 1 )$ in Fig. 3. Because $\left| \lambda _ { \operatorname* { m a x } } \right| \approx 1$ models separate along these lines only through the small departures of $\lvert \lambda _ { \mathrm { m a x } } \rvert$ from one, so a larger $\lvert \lambda _ { \mathrm { m a x } } \rvert$ corresponds to faster error growth.

The law holds only while both approximations remain valid, that ${ \mathrm { i s } } ,$ while $\Delta t \ll 1$ and while $\| \mathbf { e } ( t _ { p } ) \|$ stays small enough that the $\mathcal { O } ( \| \mathbf { e } \| _ { 2 } ^ { 2 } )$ terms are negligible. Explicitly, the derivation requires $p \Delta t \ll 1$ , i.e., $p \ll 1 / \Delta t = 1 0 0 0$ at $\Delta t = 1 0 ^ { - 3 }$ ; the longer rollouts of Figs. 2(b) and $\mathrm { 3 ( d ) }$ probe beyond this window, and the continued slow accumulation observed there is an empirical observation outside the guaranteed regime rather than a prediction of Eq. (16). The two models trained at $\Delta t = 5 \times 1 0 ^ { - 2 }$ and $\Delta t = 1 0 ^ { - 1 }$ exceed this window by a larger margin at the longest lead times, as discussed in SI, Section 2.4. As the rollout proceeds and the accumulated error grows, these higher-order terms become significant and the error departs from the linear prediction. Within the lead times of Fig. 3 this departure is visible first for the direct-step models, whose large per-step amplification excites the higher-order terms after only a few steps.

## 3 Results

We evaluate the Jacobian eigenanalysis of Section 2 as an a priori diagnostic on the Kuramoto– Sivashinsky (KS) system. The suite studied here comprises 29 models, of which two are direct-step models and 27 are integration-constrained (see Methods). The main-text figures show a representative subset of these models, and the complete set of 29 is reported in Table S1.

## 3.1 Eigenvalues a priori diagnose the amplification of rollout error

Figure 2 contrasts the two model classes using four models, the direct-step MLP and FNO and their Euler-constrained counterparts. The two Euler models stand in for all 27 integration-constrained models examined here (SI, Fig. S1 and SI, Fig. S2), and each model is trained and emulated at $\Delta t = \Delta t _ { \mathrm { D N S } } = 1 0 ^ { - 3 }$ , unless stated otherwise. Throughout, a model whose Jacobian has the form of $\operatorname { E q . } \left( 1 0 \right)$ is called integration-constrained, and every model except the two direct-step models is of this type. For the three implicit models, this holds to first order in $\Delta t$ (SI, Section 2.2).

The central empirical result is in Fig. 2(a). The two direct-step models become unstable and unphysical within a few steps, with rapidly growing error, whereas the integration-constrained models remain stable and physically plausible, with error growing in proportion to the step count and orders of magnitude smaller. Figure $2 ( \mathrm { b } )$ shows the same RMSE on log–log axes over the first $1 0 ^ { 4 }$ steps of the emulation. The direct-step models saturate within a few tens to a few hundreds of steps, whereas the error of the two Euler models accumulates slowly and in proportion to the number of rollout steps, as predicted by Eq. (16), remaining orders of magnitude smaller over most of the window, with the Euler MLP approaching the decorrelation level only at its right edge. Strict proportionality to the step count does eventually give way as the accumulated error grows and the higher-order terms of Section 2.4 become significant, but the error continues to accumulate slowly rather than diverging, in contrast to the direct-step models. Equation (7) separates two budgets: the eigenspectrum sets the rate at which errors are amplified, while the one-step generalization error sets the rate at which they are injected. The integration constraint alone, however, does not guarantee long-term stability. The near-neutral spectrum controls how fast errors are amplified, but the total error also carries the accumulated one-step generalization error ϵ (Eq. (14)). The Euler MLP of Fig. 2(b), whose ϵ is the larger of the two, reaches the climatological decorrelation level by the right edge of that window, while the Euler FNO does not, and the same ordering holds for the remaining MLP and FNO pairs over the full $1 0 ^ { 5 } { \mathrm { - s t e p } }$ emulation (SI, Fig. S1), despite eigenvalues on the unit circle in every case. Long-term stability therefore requires both a near-neutral spectrum and a suficiently small generalization error, with or without the spectral regularizer (Section 6.6), and the fidelity of the predicted spectrum stil depends on architecture (SI, Section 3).

This behavior is set by the eigenvalues of the Jacobian, evaluated at inference rather than during training, and computed once at the initial condition $t _ { 0 }$ . Figure $2 ( \mathrm { c ) }$ plots these eigenvalues on the Argand plane for all four models. The two direct-step models have $| \lambda _ { \operatorname* { m a x } } | > 1$ , while the eigenvalues of the two Euler models collapse onto the unit circle and cluster tightly around 1. The $2 \times 1 0 2 4$ eigenvalues of the two integration-constrained models are nearly coincident there, so they appear as a single dense cluster and are barely visible in (c). Figure $2 ( \mathrm { d } )$ therefore zooms into that cluster and shows a spread confined to the third decimal place, with the two models separated in the fourth, consistent with neutral stability.

The measured values can be compared against the dynamics being emulated. The linear part of the KS operator has symbol $k ^ { 2 } - k ^ { 4 }$ , whose maximum over the resolved wavenumbers is 0.2495 at $k = 0 . 6 9$ so the corresponding time-∆t flow map has $| \lambda _ { \operatorname* { m a x } } | = \exp ( 0 . 2 4 9 5 \Delta t ) = 1 . 0 0 0 2 5$ . The departures from unity measured for the two Euler models, $\phantom { + } 1 . 7 5 \times 1 0 ^ { - 4 }$ and $2 . 0 1 \times 1 0 ^ { - 4 }$ , agree with the value implied by the linear operator, $2 . 5 \times 1 0 ^ { - 4 }$ , to within a factor of 1.5. This estimate neglects the advective contribution to the tangent map and is a reference scale rather than a precise target. It matters because Eq. (10) would place the spectrum within $\mathcal { O } ( \Delta t )$ of unity for any bounded learned operator, including one that has learned the wrong dynamics, so proximity to unity is not by itself evidence that the model has learned anything. Agreement to within a factor of 1.5 with the value implied by the linear operator indicates instead that the trained network has recovered the size of the true tangent map and has not merely inherited the near-identity structure of the constraint. It also gives $\lvert \lambda _ { \mathrm { m a x } } \rvert$ a reference value against which a trained model can be checked before any rollout is run, since a model whose measured $\lvert \lambda _ { \mathrm { m a x } } \rvert$ departs from the value implied by the linear operator is misrepresenting the local growth rate of the system it emulates.

This eigenvalue structure explains the error growth in (a) and (b). The near-unit eigenvalues of the integration-constrained models produce the slow, near-neutral accumulation derived in Section 2.4, while the large $\lvert \lambda _ { \mathrm { m a x } } \rvert$ of the direct-step models $\left( \mathrm { E q . ~ } \left( 9 \right) \right)$ amplifies perturbations strongly and drives the rapid divergence. The collapse onto the unit circle follows directly from $\operatorname { E q . } \ ( 1 0 )$ . The identity term contributes eigenvalues exactly at 1, and the correction $\Delta t \nabla \mathbf { H } [ \mathcal { N } ( \mathbf { u } _ { T } ) ]$ is a small perturbation because $\Delta t = 1 0 ^ { - 3 }$ . The qualitative spectrum, and hence the predicted neutral stability, is therefore insensitive to the architecture, the loss function, the presence or absence of spectral regularization, and the order of the explicit integrator. This insensitivity is why two Euler models can stand in for all 27 integration-constrained models in Fig. 2. SI, Fig. S1 shows the RMSE curves and the eigenspectra of the remaining explicit models, which exhibit the same behavior.

## 3.2 Emergence of a linear scaling law

Figure 3 tests the scaling law of Section 2.4 for all 29 models, plotting $\| \mathbf { e } ( t + \Delta t ) \|$ against $\lvert \lambda _ { \mathrm { m a x } } \rvert \parallel \mathbf { e } ( t ) \rvert$ at successive rollout steps, with $\lvert \lambda _ { \mathrm { m a x } } \rvert$ the largest eigenvalue of the Jacobian J evaluated at the initial condition. The two direct-step models are shown in vermilion and the 27 integration-constrained models—spanning the Euler, RK4, PEC4, and implicit Euler schemes—in blue, with architecture distinguished by marker fill (MLP filled, FNO open). Table S1 lists each model with its integration scheme, architecture, and loss.

The integration-constrained models follow the predicted law $\| \mathbf { e } ( t _ { p } ) \| \approx p \left| \lambda _ { \operatorname* { m a x } } \right| \| \mathbf { e } ( t _ { 1 } ) \|$ of Eq. (16). At rollout step p they lie on a line whose slope is the step index $p$ itself, equal to 2 at the first step and growing to 11, 101, and 1001 at $p = 1 1 , 1 0 1 , 1 0 0 1 ;$ on the log–log axes of the four panels of Fig. 3 these appear as unit-slope reference lines, one per panel, whose vertical ofset grows with $p .$ Because $\lvert \lambda _ { \mathrm { m a x } } \rvert \approx 1$ for these models, they separate along each line only through the small departures of $\lvert \lambda _ { \mathrm { m a x } } \rvert$ from one.

The four panels of Fig. 3 correspond to $p \Delta t = 0 . 0 0 2 , 0 . 0 1 1 , 0 . 1 0 1$ , and 1.001, so they traverse the validity window of Eq. (16) from well inside it to its boundary. The law holds tightly in (a) and (b), where $p \Delta t \ll 1$ . The scatter about the reference line grows in (c), where $p \Delta t \approx 0 . 1$ . In (d), where $p \Delta t \approx 1$ , the condition is no longer met and the models depart from the line in both directions. The breakdown therefore occurs at the step count the derivation predicts, and the agreement in panels (a) to (c) is a test of the law within the regime where it is derived rather than an extrapolation beyond it.

The two direct-step models track the same line for the first few steps and then peel away, growing faster than linearly. This is consistent with the law being a small-error prediction. The eigenvalues of the direct-step models are not clustered near one, so once the accumulated error grows enough to excite the dominant eigenvalue with $| \lambda _ { \operatorname* { m a x } } | > 1$ , the higher-order terms take over. The departure occurs after about 10 steps for the direct-step MLP and about 100 steps for the direct-step FNO, which places them above the reference line at the lead time of panel (b). Beyond that point, their error reaches the climatological decorrelation level and stops growing, so at the longer lead times of panels (c) and (d) they fall below a reference line that continues to grow in proportion to $p .$ This saturation is the same one visible in Fig. 2(a).

The slope p has a simple origin (Eq. (15)). At the first step e(t ) is the generalization error, and the Jacobians of integration-constrained models have $| \lambda _ { \operatorname* { m a x } } | \approx 1$ for $\Delta t \ll 1$ , so the generalization error is nearly unchanged over the early rollout and Eq. (14) reduces to a sum of $p$ near-identical copies of $\mathbf { e } ( t _ { 1 } )$ . As p grows, the accumulated error eventually becomes large enough that the $\mathcal { O } ( \| \mathbf { e } \| _ { 2 } ^ { 2 } )$ terms are no longer negligible, and the points scatter about the line, which is what panel (d) shows.

Enforcing an integration constraint is what makes this linear stability analysis usable. The constraint forces the spectrum onto the unit circle and fixes a single controlled scale, $| \lambda _ { \operatorname* { m a x } } | \approx 1$ , about which the error dynamics linearize cleanly. The direct-step models have no such control, their spectrum spans a wide range with $| \lambda _ { \operatorname* { m a x } } | > 1$ , and no comparable linear law holds for them over the rollout.

The same holds when the constraint is imposed with an implicit rather than an explicit integrator. Three of the models are constrained with an implicit Euler scheme, two of them trained and emulated at $\Delta t = 5 \times 1 0 ^ { - 2 }$ and $\Delta t = 1 0 ^ { - 1 }$ . At every ∆t tested, the implicit constraint gives a $\lvert \lambda _ { \mathrm { m a x } } \rvert$ closer to unity and a lower rate of error growth than the corresponding explicit constraint, and the ordering of $\lvert \lambda _ { \mathrm { m a x } } \rvert$ reproduces the ordering of the measured RMSE. This transfers the classical result that implicit integrators are more stable at large ∆t to the learned setting, and it follows from the inverse structure of the implicit Jacobian derived in SI, Section 2.2. SI, Section 2 reports the full comparison.

The largest eigenvalue is therefore an a priori diagnostic of model behavior, obtained by diferentiating the model output with respect to its input through automatic diferentiation, without running a full rollout. A larger $\lvert \lambda _ { \mathrm { m a x } } \rvert$ implies faster error growth, hence poorer short-term accuracy and weaker long-term stability. This ranking holds both across the two model classes and within the integrationconstrained class, where the small diferences in $\lvert \lambda _ { \mathrm { m a x } } \rvert$ order the models by error growth along the lines of Fig. 3. Section 3.3 quantifies this predictive power with a linearized error law, built from $\lvert \lambda _ { \mathrm { m a x } } \rvert$ and the measured one-step error, that reproduces the measured rollout error across the explicit integration-constrained suite (Fig. 4(c,d)). Because $\lvert \lambda _ { \mathrm { m a x } } \rvert$ is a per-step multiplier, models trained at diferent time steps are compared through the growth rate per unit time $\sigma = \ln { \left| \lambda _ { \operatorname* { m a x } } \right| } / \Delta t$ . Measured this way, every integration-constrained model satisfies $\sigma \leq 0 . 3 4$ , while the two direct-step models give $\sigma = 4 8 . 1$ and $\sigma = 6 0 . 2$ . The two classes are separated by more than two orders of magnitude in the rate at which they amplify perturbations. The leading Lyapunov exponent of the KS attractor at this domain length is of order $1 0 ^ { - 1 }$ , so the integration-constrained models amplify perturbations at a rate of the same order as the dynamics, while the direct-step models exceed it by a factor of roughly 500. Table S1 reports σ for every model and SI, Section 1.1 gives the definition.

For high-dimensional systems the full eigendecomposition of J can be prohibitive, but the clustering of eigenvalues near one gives a cheap estimate of $\lvert \lambda _ { \mathrm { m a x } } \rvert$ . Starting from

$$
\sum _ { i = 1 } ^ { d } \lambda _ { i } = T r ( { \bf J } ) ,\tag{17}
$$

and the triangle inequality

$$
\left| T r ( \mathbf { J } ) \right| = \left| \sum _ { i = 1 } ^ { d } \lambda _ { i } \right| \leq \sum _ { i = 1 } ^ { d } \left| \lambda _ { i } \right| ,\tag{18}
$$

the clustering $\lambda _ { 1 } \approx \lambda _ { 2 } \approx \cdot \cdot \cdot \approx \lambda _ { d } \approx \lambda _ { \mathrm { m a x } }$ for integration-constrained models (Fig. 2(c,d), Eq. (10)) turns Eq. (18) into an approximate equality, so

$$
\left. \lambda _ { \mathrm { m a x } } \right. \approx \frac { 1 } { d } \left. T r ( \mathbf { J } ) \right. ,\tag{19}
$$

with d the dimension of J. The trace can in turn be estimated with computationally tractable random sketching methods [Meyer et al., 2021], avoiding the full spectrum.

## 3.3 Deviation from linear stability theory

We now revisit the linear scaling law of Section 3.2 and examine where the eigenvalue-only view breaks down, that is, where a model with a larger $\lvert \lambda _ { \mathrm { m a x } } \rvert$ nonetheless exhibits lower error growth than a model with a smaller one. Figure $^ \mathrm { 4 ( a , b ) }$ compares two FNO models constrained with the same PEC4 integrator, trained with and without the spectral regularizer $\mu ( \theta )$ . To the precision of the legend the two models share $| \lambda _ { \mathrm { m a x } } | = 1 . 0 0 0 2$ , with Table S1 separating them only in the fifth decimal place, so the eigenvalue magnitude alone cannot account for the diference in their error growth; yet the spectrally regularized model has the visibly smaller error over the early rollout in Fig. 4(a).

Equation (14) identifies two mechanisms behind this deviation. The first is the additive contribution of the generalization error, $\textstyle \sum _ { i } \epsilon ( \mathbf { u } _ { T } ( t _ { i } ) )$ . The spectral regularizer, by construction, suppresses the high-wavenumber component of the one-step error and thereby reduces this sum, shown as the dashed lines in Fig. 4(a). Over the early rollout the running sum tracks the total error for each model, confirming that the accumulated generalization error is the leading term of Eq. (14); at late times the total error saturates while the running sum continues to accumulate, marking where the small-error approximation underlying the linear theory ceases to hold.

The second mechanism is that the amplification depends on the direction of the error and not only on the eigenvalue magnitudes. The Jacobian acts on the vector $\mathbf { e } ( t )$ , so the realized growth $\| \mathbf { J e } \| / \| \mathbf { e } \|$ is a weighted average of the $| \lambda _ { i } |$ over the components of e in the eigenbasis of $\mathbf { J } _ { : }$ , and it equals $\lvert \lambda _ { \mathrm { m a x } } \rvert$ only when e is aligned with the leading eigenvector. When the eigenvalues $\lambda _ { i }$ are tightly clustered around unity, as is characteristic of integration-constrained models, the growth of $\mathbf { e } ( t )$ is determined not only by $\lvert \lambda _ { \mathrm { m a x } } \rvert$ but also by the relative orientation (cosine similarity) between $\mathbf { e } ( t )$ and the eigenvectors $\mathbf { v } _ { i }$ of J. In Fig. 4(b) each eigenvalue is shaded by $| \cos ( \mathbf { e } , \mathbf { v } _ { i } ) |$ : the largest-magnitude eigenvalues of the spectrally regularized model carry very low cosine similarity, so they contribute little to the realized amplification, and the marginally larger $\lvert \lambda _ { \mathrm { m a x } } \rvert$ does not translate into larger $\| \mathbf { e } ( t + \Delta t ) \|$ . SI, Section 4 shows that these direction-dependent efects act within the $\mathcal { O } ( \Delta t )$ residue of the leadingorder eigenvalue approximation, so they are finite- $\Delta t$ corrections to the law rather than a failure of it.

Both mechanisms can be folded into a refined error approximation that remains a priori. Iterating the linearized recursion of Eq. (11) from the initial condition, with the injected one-step error held at its measured norm $\lVert \epsilon \rVert$ and the per-step amplification set to the single controlled scale $a = \left| \lambda _ { \operatorname* { m a x } } \right| ,$ gives the geometric law

$$
{ \hat { e } } ( k ) = \| \epsilon \| { \frac { a ^ { k } - 1 } { a - 1 } } ,\tag{20}
$$

where k counts steps from the initial condition, $t _ { k } = t _ { 0 } + k \Delta t$ , so that $\hat { e } ( k )$ is the predicted error after k rollout steps. Note that this difers from the lead time of ${ \mathrm { F i g . ~ } } 3 ,$ , which is measured from the first step $t _ { 1 } { \mathrm { : } }$ : the reference slope there at lead time k is $p = k + 1$ , whereas here ${ \hat { e } } ( k ) \to k \| \epsilon \|$ as $a  1$ , recovering the linear scaling law of $\operatorname { E q }$ . (16) with $p = k$ , and departing from it as a moves away from one. Both inputs, a and $\lVert \epsilon \rVert$ , are measured at the initial condition, so Eq. (20) predicts the rollout error without running the rollout. Figure $4 ( \mathrm { c } )$ tests this prediction after $k = 1 0$ rollout steps for every explicit integration-constrained model: the suite clusters on the line $y = x ,$ , showing that the two-term law of Eq. (14)—one-step injected error amplified by the Jacobian—predicts error growth across architectures and integration schemes with a single relation. At $k = 1 0$ and $a = 1 . 0 0 0 2$ the geometric factor equals 10.009, within 0.1% of $k ,$ so this panel tests the injected-error term with the amplification factor contributing negligibly. The amplification factor becomes discriminating at longer lead times and for the direct-step models, where a departs from unity by five percent.

The same law also localizes the origin of spectral bias. Restricting the comparison to the highwavenumber band $( \omega \ge \omega _ { c }$ , with $\omega _ { c } = 1 0 0 )$ , replacing the injection term by $\beta ,$ the high-ω content of the one-step error, and keeping the same amplification $a = | \lambda _ { \operatorname* { m a x } } |$ , again places the suite on $y = x$ $\left( \mathrm { F i g . 4 ( d ) } \right)$ ). The high-ω error after $k$ rollout steps is therefore set by the high-ω content of the injected one-step error rather than by any preferential amplification of high wavenumbers, identifying the injected error spectrum $\hat { \epsilon } ( \omega )$ as the origin of spectral-bias growth. The Fourier spectra of the predicted state and of its time derivative are shown in SI, Fig. S3, and SI, Section 3 examines how $\lvert \lambda _ { \mathrm { m a x } } \rvert$ relates to the implicit difusivity of the model.

Deviations from the classical eigenvalue-only prediction therefore emerge from two complementary mechanisms, the direction dependence of error amplification, governed by the eigenvector projections of J, and the cumulative influence of the summed generalization errors, which the spectral regularizer reduces. Once the injected error is measured rather than assumed, the linearized law of Eq. (20) recovers its predictive power across the full explicit integration-constrained suite, and a complete characterization of rollout error growth must account for both the amplification (spectral) and injection

(generalization-error) components of Eq. (7).

## 3.4 Novel stability-promoting loss function

The insight about short-term performance obtained from Section 3.3 allows us to develop a novel loss function (see Eq. (38) in Section 6.7) that uses the linearized model to minimize $\mathbf { e } ( t + \Delta t )$ using the projection of e(t) on J. This constrains both the error as well as the eigenvalues to promote a more stable model. We evaluate the new loss on the FNO-based models constrained with each of the three explicit integrators (Euler, RK4, and PEC4), comparing each model trained with the baseline RMSE loss against its counterpart trained with the proposed Jacobian-based loss. In Fig. 5(a), we verify that training and inference with the new loss follow the scaling obtained in Section 3.2, and that $\lvert \lambda _ { \mathrm { m a x } } \rvert$ diagnoses model performance: for every integrator, the Jacobian-based loss yields a smaller $\lvert \lambda _ { \mathrm { m a x } } \rvert \parallel \mathbf { e } ( t ) \rvert$ and a correspondingly smaller $\| \mathbf { e } ( t + \Delta t ) \|$ than the RMSE baseline.

Figure 5(b) shows the RMSE over the autoregressive rollout for the same six models. To quantify the rate of error growth, the legend reports $\alpha ,$ the slope of a linear fit of $\log _ { 1 0 } ( \mathrm { R M S E } )$ against $\log _ { 1 0 }$ of the timestep over the first 100 rollout steps, i.e., the exponent of the power law RMSE $\propto t ^ { \alpha }$ . For every integrator, the model trained with the Jacobian-based loss has a smaller α than its RMSEtrained counterpart (Euler: 0.916 versus 0.956; PEC4: 0.927 versus 0.946; RK4: 0.965 versus 0.998), demonstrating that the new loss slows the growth of error during rollout relative to the basic RMSE loss. A reduction in the one-step generalization error alone cannot produce this efect: the leading term of Eq. (14) is the running sum of ϵ, so a uniformly smaller ϵ lowers the log-RMSE curve without changing its slope, provided the one-step error evolves similarly along the trajectory, as in the constantinjection law of Eq. (20). The smaller α therefore isolates a reduction in the per-step amplification of the accumulated error, the propagated-error term of Eq. (6). Consistent with Section 3.3, this reduction need not appear in $\lvert \lambda _ { \mathrm { m a x } } \rvert$ , which bounds but does not set the realized amplification: Table S1 shows that the Jacobian-based loss leaves the FNO $\lvert \lambda _ { \mathrm { m a x } } \rvert$ essentially unchanged, so for these models the suppression acts through the orientation of the accumulated error relative to the strongly amplified eigendirections of J, whereas for the MLP models the loss also reduces $\lvert \lambda _ { \mathrm { m a x } } \rvert$ directly. This reduction in the growth rate is an important result: it shows that the Jacobian-driven amplification of error, which the RMSE objective alone leaves uncontrolled, can be regularized directly during training. The reduced growth rate does not by itself guarantee long-term stability however. Several models trained with the Jacobian-based loss (in particular the MLP-based ones) still drift to instability over the longest rollouts because their one-step generalization error ϵ remains large, consistent with Eq. (14), which requires both a near-neutral spectrum and a suficiently small ϵ for long-term stability.

## 4 Conclusion

Autoregressive neural emulators are currently trained on a single time step and then used over many time steps: the training objective focuses on the one-step error, while the quantity of interest at inference is the error after hundreds or thousands of steps; and nothing in the training procedure constrains how a one-step error is carried forward. We have developed a theoretical framework to address this fundamental disconnect. At the core of our approach is a single object, the Jacobian of the learned one-step update map taken with respect to the system state and evaluated after training, Eq. (4). Our approach is distinct from the parameter gradients used during training; and, surprisingly, it is rarely examined when these learned models are evaluated.

Linearizing the error dynamics about the true trajectory separates the two contributions to emulation error, Eq. (7): the one-step generalization error sets the rate at which error is injected, and the error propagator J sets the rate at which injected error is amplified into the propagated error. Training controls the generalization error and leaves the propagated error free, which is why models with comparable one-step accuracy diverge at diferent rates during rollout. The magnitude of the largest eigenvalue of the Jacobian, $\lvert \lambda _ { \mathrm { m a x } } \rvert$ , which governs the growth of a perturbation over one step, is an a priori diagnostic of that amplification. It requires one automatic-diferentiation evaluation at a single state and no rollout. Its definition does not reference the architecture, the integration scheme, or the loss function, and across the suite tested here we observe that its predictive value does not depend on them either.

Emulators fall into two classes according to how the update map is built, and the classes difer in whether the amplification is controlled at all. A direct-step model predicts the next state in one shot, so its Jacobian is that of the network itself, Eq. (9), which carries no dependence on the time step and places no constraint on the spectrum. An integration-constrained model instead uses the network $\mathcal { N }$ to estimate the time derivative and a numerical integrator H to advance it, so the Jacobian takes the form $\mathbf { I } + \Delta t \nabla \mathbf { H }$ , with H acting on the output of N, Eq. (10). Its eigenvalues then lie within $\mathcal { O } ( \Delta t )$ of unity by construction. The implicit form of the same constraint, $( \mathbf { I } - \Delta t \nabla \mathbf { H } ) ^ { - 1 }$ , places an eigenvalue outside the unit circle only along directions in which the learned tangent operator is expanding. Converted to a growth rate per unit time, which is the comparison that is meaningful across models trained at diferent time steps, the direct-step models amplify perturbations by orders of magnitude faster than the integration-constrained models. This separation, rather than the small diference in the per-step eigenvalue, is what the measured rollouts reflect, and it accounts for the divergence of direct-step architectures reported previously [Bi et al., 2023, Chattopadhyay et al., 2023b, 2024, Jiang et al., 2026, Pathak et al., 2022].

The near-neutral spectrum of the integration-constrained models also yields a scaling law, Eq. (16). The amplification per step lies within $\mathcal { O } ( \Delta t )$ of unity by the structure of Eq. (10), and the injected error is nearly unchanged over the early rollout, which follows from the continuity of $\tilde { \mathbf { G } }$ over a small step and which the measured rollouts confirm. The accumulated error after p steps is then p times the first-step error, so the error grows linearly in the step count rather than geometrically in the step count, which is what it does once $\lvert \lambda _ { \mathrm { m a x } } \rvert$ is bounded away from unity. The law holds across architectures, integration schemes, and loss functions, within the window in which it is derived.

A near-neutral spectrum is necessary for slow error growth, but it is not suficient for long-term stability. Several integration-constrained models drift over the longest rollouts because their accumulated one-step error is large, consistent with Eq. (14). Two further efects limit the eigenvalue-only view. The realized amplification is a weighted average of the eigenvalue magnitudes over the components of the accumulated error in the eigenbasis of the Jacobian, so it depends on the orientation of that error relative to the eigenvectors, which the eigenvalues bound but do not set. The high-wavenumber error after k steps is fixed by the high-wavenumber content of the injected one-step error rather than by preferential amplification of high wavenumbers, which locates the origin of spectral-bias growth in the injected error spectrum rather than in the rollout. Measuring the injected error and retaining $\lvert \lambda _ { \mathrm { m a x } } \rvert$ as the amplification scale recovers the prediction, Eq. (20). The same analysis yields a training ob jective, Eq. (38), that regularizes the Jacobian-driven amplification directly and reduces the measured error-growth exponent for every integrator tested.

The eigenspectrum of the integration-constrained operators difers from that of a numerical solver for the same equations. The eigenvalues cluster near unity rather than decaying, which makes $\lvert \lambda _ { \mathrm { m a x } } \rvert$ cheap to estimate and makes the linearization tractable about a single controlled scale. The same clustering is why the eigenvector projections matter, since no direction is strongly preferred by magnitude alone. The direct-step Jacobians do decay, with most of their spectrum massed near the origin and a few eigenvalues beyond the unit circle, which is why neither the single-scale linearization nor the trace estimate applies to them.

Our theory has been developed and tested on the Kuramoto–Sivashinsky equation, a canonical system for multi-scale chaotic dynamics, and one on which a factorial sweep over architecture, integrator, and loss is feasible. The Jacobian is evaluated at the initial condition, and although the reported values are robust over 100 random initial conditions, the diagnostic describes the local tangent map rather than its evolution along the attractor. The scaling law holds only for small time steps and over rollouts short enough that the accumulated error remains small. Extending the analysis to the evolution of the Jacobian during rollout, and from the linear theory to a nonlinear one through random matrix theory [Hodgkinson et al., 2025, Liao and Mahoney, 2021, 2025, Pennington et al., 2018] and iterated discrete recurrence relation theory [Hodgkinson and Mahoney, 2020], are natural next steps, as is testing the diagnostic on higher-dimensional emulators.

Numerical analysis provides established tools for deciding, before a simulation is run, whether a discretization of a diferential equation will be stable, how fast its error will grow, and how that behavior depends on the scheme and the time step. Scientific machine learning has no comparable body of analysis, and stabilization strategies for learned emulators are instead assembled from heuristics and validated by long rollouts [Lippe et al., 2023, Stachenfeld et al., 2021, Wikner et al., 2022]. The framework developed here is a step toward closing that gap. It takes a trained network as a discrete dynamical system, identifies the operator that governs its behavior during inference, and derives from it the quantities that classical stability analysis provides for numerical schemes. Interpretability work for scientific neural networks has instead centered on saliency maps, layer-wise relevance propagation, and related inspection of internal structure [Molnar, 2020]. The diagnostic developed here does not inspect the network, which is what makes it agnostic to architecture and loss, and diagnostics of this kind have been productive outside scientific machine learning [Martin and Mahoney, 2019, 2020, Martin et al., 2021].

## 5 Acknowledgements

AC, PH, and MWM designed the research. CA wrote the computational codes and executed the research. Early versions of these codes were developed by AC. AC and CA wrote the paper, and all the authors analyzed the results and edited the manuscript. AC and CA acknowledge the support from the National Science Foundation (grant no. 2425667), DARPA APaQus program, the Sloan Foundation, and Schmidt Sciences, LLC. PH is grateful to the National Science Foundation (AGS-531264) and Schmidt Sciences, LLC. MWM acknowledges DARPA, NSF, the DOE Competitive Portfolios grant, and the DOE SciGPT grant. Computational resources were provided by NSF ACCESS MTH240019, MTH250006 and NCAR CISL UCSC0008, and UCSC0009. The computational codes can be found on GitHub: https://github.com/cainsliewastaken/Spectral\_Stability.

## 6 Methods and Systems

## 6.1 Kuramoto–Sivashinsky system

We conduct our data-driven experiments on a high-dimensional multi-scale chaotic dynamical system, given by the Kuramoto–Sivashinsky (KS) equations. The governing equations of the KS system are given by:

$$
\frac { \partial u ( x , t ) } { \partial t } + u ( x , t ) \frac { \partial u ( x , t ) } { \partial x } + \frac { \partial ^ { 4 } u ( x , t ) } { \partial x ^ { 4 } } + \frac { \partial ^ { 2 } u ( x , t ) } { \partial x ^ { 2 } } = 0 .\tag{21}
$$

We use a domain length $( L = 1 0 0 ; x \in [ - 5 0 , 5 0 ] )$ to obtain direct numerical simulations (DNS) of the spatio-temporal evolution of this system using an initial condition given by:

$$
u ( x , 0 ) = - \cos \left( \frac { 2 \pi x } { L } \right) \left( 1 - \sin \left( \frac { 2 \pi x } { L } \right) \right) ,\tag{22}
$$

and periodic boundary conditions. We use 1024 spatial grid points and a $\Delta t _ { D N S } = 1 0 ^ { - 3 }$ . We train the suite of autoregressive models on 150000 temporal samples of $u ( x , t )$ with $\Delta t = \Delta t _ { D N S }$ . We then conduct emulation with the autoregressive models using a new initial condition outside the training dataset for another consecutive 100000 time steps. Depending on the model, such emulation will either become unstable, unphysical, or, in certain cases, remain stable and physically consistent.

## 6.2 Autoregressive models

In order to build our suite of autoregressive models, we assume that the underlying dynamical system represented by the KS equations is unknown and the discrete dynamics are represented by:

$$
\mathbf { u } ( x , t + \Delta t ) \approx \underbrace { \mathbf { u } ( x , t ) + \int _ { t } ^ { t + \Delta t } \underbrace { \mathbf { F } \left( \mathbf { u } ( x , t ) \right) d t } _ { \mathbf { M } [ \circ ] } } _ { \mathbf { H } [ \circ ] } .\tag{23}
$$

In Eq. (23), we describe the general formulation of our autoregressive model, where $\mathcal { N }$ is a neural network, and where H is generally a numerical integration method implemented inside a diferentiable layer. Most autoregressive models directly represent $\mathbf { H } [ \circ ]$ with a deep neural network [Bi et al., 2023, Lam et al., 2022, Pathak et al., 2022] (direct step model, from now on); while, based on our previous studies [Chattopadhyay et al., 2023b, 2024, Guan et al., 2025], we introduce a numerical integration-based hard constraint that parameterizes $\mathbf { F } \left( u ( x , t ) \right)$ with a deep neural network and an explicitly-implemented diferentiable layer to perform higher-order integration in $\mathbf { H } [ \mathrm { o } ]$

For either the direct step model or the models with integration-based hard constraint, we train the parameters, θ, of the model by minimizing the diference between the predicted and true value of $\mathbf { u } ( x , t + \Delta t )$ at each time step in the training dataset. The general loss function, without any spectral regularizer (see Section 6.6), is given by the standard root-mean-square error (RMSE):

$$
L ( \theta ) = \frac { 1 } { T } \sum _ { t = 0 } ^ { t = T } \vert \vert \mathbf { u } ( x , t + \Delta t ) - \mathbf { u } ( x , t ) - \Delta t \mathbf { H } \left[ \mathcal { N } \left( \mathbf { u } ( x , t ) , \theta \right) \right] \vert \vert _ { 2 } ,\tag{24}
$$

where $T$ is the total number of time steps (temporal samples) on which the model is trained.

Next, we describe the few diferent explicit schemes and the one implicit time integration scheme that has been used, and how it has been implemented in our data-driven autoregressive models.

## 6.3 Explicit integration-based hard constraints

Here, we describe the few explicit integration-based hard constraints that we have used in this paper. We have used a first-order Euler integrator, an RK4 integrator, and a $4 ^ { t h }$ -order PEC integrator. For the explicit schemes, the $\Delta t$ used to train the models and run inference is equal to $\Delta t _ { D N S }$ , except for the large- $\cdot \Delta t$ comparison against the implicit scheme in SI, Section 2.3.

For the Euler integration scheme, the predicted value of $\mathbf { u } ( x , t + \Delta t )$ from Eq. (23) is written as:

$$
{ \bf { u } } ( x , t + \Delta t ) = { \bf { u } } ( x , t ) + \mathcal { N } [ \circ , \theta ] \Delta t .\tag{25}
$$

The Euler integrator is similar to learning with the residues [Chen and Xiu, 2021], which has been used in several diferent studies prior to this (although, in most cases, the $\Delta t$ term is neglected, which results in a non-convergent integration scheme [Krishnapriyan et al., 2023]). The integration-based constraints used in this study are a generalization of learning the residues using rigorous higher-order integrators.

For the RK4 integrator, we extend Eq. (25) as:

$$
k _ { 1 } = \mathcal { N } \left[ \mathbf { u } ( x , t ) , \theta \right]\tag{26}
$$

$$
k _ { 2 } = \mathcal { N } \left[ { \bf u } ( x , t ) + \frac { \Delta t } { 2 } k _ { 1 } , \theta \right]\tag{27}
$$

$$
k _ { 3 } = \mathcal { N } \left[ { \bf u } ( x , t ) + \frac { \Delta t } { 2 } k _ { 2 } , \theta \right]\tag{28}
$$

$$
k _ { 4 } = \mathcal { N } \left[ \mathbf { u } ( x , t ) + \Delta t k _ { 3 } , \theta \right]\tag{29}
$$

$$
\mathbf { u } ( x , t + \Delta t ) = \mathbf { u } ( x , t ) + \Delta t \underbrace { \frac { 1 } { 6 } \left( k _ { 1 } + 2 k _ { 2 } + 2 k _ { 3 } + k _ { 4 } \right) } _ { \mathbf { H } [ \circ ] } .\tag{30}
$$

For the $4 ^ { t h }$ -order PEC method (PEC4), we use the following formulation:

$$
\tilde { \mathbf { u } } ( x , t + \Delta t ) = \mathbf { u } ( x , t ) + \Delta t \mathcal { N } [ \mathbf { u } ( x , t ) , \theta ]\tag{31}
$$

$$
\hat { \mathbf { u } } ( x , t + \Delta t ) = \mathbf { u } ( x , t ) + \frac { \Delta t } { 2 } \left( \mathcal { N } \left[ \mathbf { u } ( x , t ) , \boldsymbol { \theta } \right] + \mathcal { N } \left[ \tilde { \mathbf { u } } ( x , t + \Delta t ) , \boldsymbol { \theta } \right] \right)\tag{32}
$$

$$
\bar { \mathbf { u } } ( x , t + \Delta t ) = \mathbf { u } ( x , t ) + \frac { \Delta t } { 2 } \left( \mathcal { N } \left[ \mathbf { u } ( x , t ) , \theta \right] + \mathcal { N } \left[ \hat { \mathbf { u } } ( x , t + \Delta t ) , \theta \right] \right)\tag{33}
$$

$$
\mathbf { u } ( x , t + \Delta t ) = \mathbf { u } ( x , t ) + \Delta t \underbrace { \frac { 1 } { 2 } \left( \mathcal { N } \left[ \mathbf { u } ( x , t ) , \boldsymbol { \theta } \right] + \mathcal { N } \left[ \bar { \mathbf { u } } ( x , t + \Delta t ) , \boldsymbol { \theta } \right] \right) } _ { \mathrm { d } \mathrm { } { \nabla \cdot } \mathrm { ~ s ~ i ~ n ~ } } .\tag{34}
$$

For higher-order integrators such as RK4 or PEC4, we represent the predicted $\mathbf { \boldsymbol { \mathsf { \sigma } } } _ { 1 } ( x , t + \Delta t ) = \mathbf { \boldsymbol { \mathsf { u } } } ( x , t ) +$ ∆tH $[ \mathcal { N } \left( \mathbf { u } ( x , t ) , \theta \right) ]$ , where H encapsulates the scheme as a diferentiable operator.

## 6.4 Implicit integration-based hard constraint

Implicit time-integration schemes have improved stability properties for larger $\Delta t ,$ as compared with explicit integration schemes [Frank et al., 1997]. In order to implement implicit time-integration as a hard constraint, we use the theory of implicit layers [Kawaguchi, 2021] in deep learning. In this approach, we can use any higher-order scheme as our H [◦] operator. The equation involving the time stepper is given by:

$$
\underset { \mathbf { y } ^ { * } } { \underbrace { \mathbf { u } ( x , t + \Delta t ) } } = \mathbf { u } ( x , t ) + \Delta t \mathbf { H } \left[ \mathcal { N } \left[ \underset { \mathbf { y } ^ { * } } { \underbrace { \mathbf { u } ( x , t + \Delta t ) } } , \theta \right] \right] ,\tag{35}
$$

where $\mathbf { y } ^ { * }$ is required to be solved via an implicit layer at each epoch (i.e. each value of θ) in the training process. In order to do that, we initialize $\mathbf { y } ^ { * } = \mathbf { u } \left( x , t \right)$ , and we execute a fixed-point iteration to converge to the correct $\mathbf { u } ( x , t + \Delta t )$ . More details about implicit layers and deep equilibrium models can be found in Bai et al. [2019], Kawaguchi [2021]. For the experiments with the implicit method, the chosen $\Delta t$ ranges from $\Delta t _ { D N S }$ up to 100 $\Delta t _ { D N S }$ (SI, Section 2.3). Here, we outline the fixed point iteration in Algorithm 1 used to train (and run inference with) the implicit numerical integrator. Diferentiating Eq. (35) with respect to the state gives $\mathbf { J } = ( \mathbf { I } - \Delta t \bar { \nabla } \mathbf { H } [ \mathcal { N } ( \bar { \mathbf { y } } ^ { * } ; \boldsymbol { \theta } ) ] ) ^ { - 1 }$ , which reduces to Eq. (10) to first order in ∆t. SI, Section 2.2 gives the derivation and the resulting expansion conditions.

Algorithm 1 Implicit time integration scheme   
1: Input: ϵ ▷ Convergence threshold, typically 10<sup>−8</sup>   
2: At each epoch, fix θ   
3: Initialize $\mathbf { y } _ { 0 } ^ { * } = \mathbf { u } ( x , t ) , k = 0$   
4: repeat   
5: $k = k + 1$   
6: $\mathbf { y } _ { k } ^ { * } = \mathbf { u } ( x , t ) + \Delta t \mathbf { H } \left[ \mathcal { N } \left( \mathbf { y } _ { k - 1 } ^ { * } , \boldsymbol { \theta } \right) \right]$   
7: until $| | \mathbf { y } _ { k } ^ { * } - \mathbf { y } _ { k - 1 } ^ { * } | | _ { 2 } \leq \epsilon$   
8: Output: $\mathbf { y } _ { k } ^ { * }$ $\vartriangleright \mathbf { y } _ { k } ^ { * }$ is the predicted value of $\mathbf { u } ( x , t + \Delta t )$

## 6.5 Neural architectures

We have used an MLP and an FNO to represent N. The MLP has 6 layers and 2000 neurons in each layer with a ReLU activation. The FNO that has been used has 6 Fourier blocks and retains 512 modes in each Fourier block along with a width of 32. We have conducted extensive hyperparameter trials before selecting these sets of hyperparameters for our empirical analysis. Each hyperparameter trial optimizes the short-term accuracy of the model.

## 6.6 Spectral bias and spectral regularizer

Generally, to train autoregressive models, previous studies have used a traditional $L _ { 2 }$ based loss function, optimizing the diference between the true $\mathbf { u } ( x , t + \Delta t )$ and its prediction, $\mathbf { H } \left[ \mathcal { N } \left( \mathbf { u } ( x , t ) \right) \right]$ , using Eq. (24). However, for multi-scale dynamical systems, e.g., turbulent flows, a spectral bias [Cao et al., 2019, Chattopadhyay et al., 2023b, Wang and Lai, 2024] prevents the network from learning the highwavenumber component of the dynamics. This often leads to instability and physical inconsistency in the models. Furthermore, spectral bias does not only present itself in the Fourier spectrum of the predicted state, but also in its derivative with respect to time. In this paper, the time-derivative is represented by $\mathcal { N } \left( \mathbf { u } ( x , t ) \right)$ for the explicit methods. We propose a spectral regularizer to the $L _ { 2 }$ loss function presented in Eq. (24) as:

$$
\mu \left( \theta \right) = \frac { 1 } { T } \sum _ { t = 0 } ^ { T } \biggr \| \frac { \partial \hat { \mathbf { u } } ( x , t ) } { \partial t } - \hat { \mathbf { H } } \left[ \mathcal { N } \left[ \mathbf { u } ( x , t ) \right] \right] \biggr \| _ { 2 } ,\tag{36}
$$

where $[ \hat { \textmd o } ]$ represents the 1D Fourier transform in x. To approximate the true and predicted time derivatives from the training and predicted data, we use a first-order finite diference. The total loss,

$$
L _ { t o t a l } = L ( \theta ) + \gamma \mu ( \theta ) ,\tag{37}
$$

where $\gamma$ is the Lagrange multiplier set to 0.10 after significant hyperparameter trials. SI, Section 3 reports the efect of this regularizer on the Fourier spectrum of the emulation.

## 6.7 Stability-promoting loss function

Based on the insight from linear stability analysis as seen in Eq. (10), we propose a new stabilitypromoting loss function for the autoregressive models with integration constraints that minimizes $| | \mathbf { e } ( t + \Delta t ) | | _ { 2 }$ in its linearized form, for each value of t in the training dataset:

$$
\begin{array} { l } { { \displaystyle { \cal L } _ { \mathrm { t o t a l } } = \frac { 1 } { T } \sum _ { t = 0 } ^ { T } \bigg \| { \bf J } \Big ( { \bf u } ( t + \Delta t ) - { \bf u } ( t ) - \Delta t { \bf H } [ { \mathcal N } [ { \bf u } ( t ) , \boldsymbol \theta ] ] \Big ) \bigg \| _ { 2 } } } \\ { ~ + ~ \displaystyle \mu ( \theta ) ~ + ~ \frac { 1 } { T } \sum _ { t = 0 } ^ { T } \bigg \| { \bf u } ( t + \Delta t ) - { \bf u } ( t ) - \Delta t { \bf H } [ { \mathcal N } [ { \bf u } ( t ) , \boldsymbol \theta ] ] \bigg \| _ { 2 } , } \end{array}\tag{38}
$$

where J is defined in Eq. (10). This novel loss function minimizes the error at the next time step, which is approximated by projecting the predicted state at the next time step on the Jacobian of the model. A more aggressive loss function that focuses more on stability could have used the projection of $\mathbf { u } _ { p } ( t + \Delta t )$ on the eigenvector of J corresponding to the largest eigenvalue. Such an approximation would have been accurate for a J with a decaying eigen spectrum; but the spectrum of J for the models with integration constraints is not decaying, instead having eigenvalues that are clustered around 1 on the unit circle (see Fig. $2 ( \mathrm { c } , \mathrm { d } )$ in Section 3). As such, projecting J on the eigenvector corresponding to the largest eigenvalue is often a poor approximation of next time-step error. Models trained with the Jacobian term but without the spectral regularizer $\mu ( \theta )$ are listed as “Jacobian” in Table S1, and those trained with both terms as “Spectral+Jac.”.

a)  
![](images/1b5cb82581319b964fc7ff431b2c162cfc5d0153abb9f28885ce2c56232dcf34.jpg)

b)  
![](images/eb601872529ec8d9972288ea35a627526e8563edf54f19ad85c4fd45a67f97d0.jpg)

c)  
![](images/46f540492ac02f479a0801c978d539572c29424e954125fe686dd5b99f9b2461.jpg)

d)  
![](images/8b1fdf8edce12439a724696998ba6c3a71c5f0f5d0749fd7993bb56834e77948.jpg)  
Figure 2: Short-term error, rollout error accumulation, and eigenanalysis of the models. Throughout, direct-step models are shown in vermilion and integration-constrained models in blue, with MLP as solid lines and FNO as dotted lines. (a) RMSE over the first 100 steps. The two directstep models (Direct MLP, Direct FNO) grow rapidly and saturate near the climatological decorrelation level, where the RMSE is O(1) in these normalized units. The two integration-constrained Euler models (Euler MLP, Euler FNO), whose eigenvalue structure is representative of all 27 integrationconstrained models examined here, remain stable over this window, with error orders of magnitude smaller and indistinguishable from zero on the linear axis. (b) The same RMSE on log–log axes over the first $1 0 ^ { 4 }$ steps. The direct-step models saturate within a few tens of steps (MLP) to a few hundred (FNO), whereas the Euler models grow slowly and in proportion to the step count, remaining orders of magnitude more accurate over most of the window and reaching the decorrelation level only at its right edge; over the full emulation several MLP-based models eventually drift as their accumulated generalization error grows. (c) Eigenvalues of the model Jacobian J at the initial condition $t _ { 0 }$ on the Argand plane for all four models, with $\lvert \lambda _ { \mathrm { m a x } } \rvert$ in the legend. The Direct MLP (◦) and Direct FNO (□) admit eigenvalues with $| \lambda _ { \operatorname* { m a x } } | > 1$ (1.0493 and 1.0620) spread well inside and beyond the unit circle, whereas the Euler MLP (⋆) and Euler FNO (△) eigenvalues collapse onto the unit circle (dotted) and cluster tightly at $\lambda = 1 \left( \left| \lambda _ { \operatorname* { m a x } } \right| = 1 . 0 0 0 2 \right)$ , appearing as a single dense cluster. (d) Zoomed-in view of the integration-constrained eigenvalues from (c), unit circle again dotted. Each spectrum spreads only in the third decimal place, and the two models separate in the fourth, with $\vert \lambda _ { \mathrm { m a x } } \vert = 1 . 0 0 0 1 7 5$ for the Euler MLP and 1.000201 for the Euler FNO. This neutral spectrum, consistent across architecture, explains the slow accumulation in proportion to the step count in (a) and (b) through the Jacobian structure of $\operatorname { E q } .$ . (10). SI, Fig. S1 shows the equivalent curves and spectra for the remaining explicit models, and Table S1 reports $\lvert \lambda _ { \mathrm { m a x } } \rvert$ for every model.

a)  
![](images/ef40bde92e815a64cc3f7e091c3e4c7c25966ad5a6a1509f11dd6bc83d1a397a.jpg)

b)  
![](images/2daaf70f0c7e73fc7bc1045ff1e1ed1cd800047ceee4b46e477046ed44676c6e.jpg)

c)  
![](images/0e840ddbbd716ffa0e0b2e6f12023b943a23538e1f24243885e219f1f3853921.jpg)

d)  
![](images/e55a2aa9bc78dc2574a320f6814ed7c83a9a8a5722fe96179d24d817317c7d19.jpg)  
Figure 3: A single linear scaling law governs error growth across the full model suite. Norm of the accumulated error $\| \mathbf { e } ( t + k \Delta t ) \| _ { 2 }$ plotted against $\vert \lambda _ { \mathrm { m a x } } \vert \Vert \mathbf { e } ( t ) \Vert _ { 2 }$ for all 29 models, where $\lambda _ { \mathrm { m a x } }$ is the largest eigenvalue of the model Jacobian J evaluated at the initial condition and $\| \cdot \| _ { 2 }$ is the Euclidean $\left( L _ { 2 } \right)$ norm of the error vector. Here $t = t _ { 1 } .$ , so ${ \mathbf e } ( t ) = { \mathbf e } ( t _ { 1 } ) = \epsilon ( t _ { 0 } )$ is the first-step (generalization) error and the lead time k is measured from the first step; the reference slope at lead time k is therefore the step count $p = k + 1$ of $\operatorname { E q . }$ (16). Panels $\mathrm { ( a ) - ( d ) }$ show the successive lead times $k = 1 , 1 0 , 1 0 0 , 1 0 0 0$ , one per panel, on log–log axes. Marker color denotes model class (direct-step in vermilion, integration-constrained in blue) and fill denotes architecture (MLP filled, FNO open), while marker shape indicates the integration scheme (Euler, RK4, PEC4, implicit Euler). The gray reference line in each panel is the linear scaling law of Eq. (16) evaluated at that lead time; on log–log axes it is a unit-slope line whose vertical ofset grows with the step count, from slope 2 in (a) to slopes 11, 101, and 1001 in (b)–(d). The panels correspond to $p \Delta t = 0 . 0 0 2 , 0 . 0 1 1 , 0 . 1 0 1$ , and 1.001, so they span the validity window of Eq. (16) from well inside it to its boundary. The integration-constrained models $( | \lambda _ { \operatorname* { m a x } } | \approx 1 )$ lie on the predicted line in (a) and (b), scatter modestly about it in $\mathrm { ( c ) }$ , and depart from it in both directions in (d), where $p \Delta t \approx 1$ and the derivation no longer applies. The two direct-step models sit at much larger $\vert \lambda _ { \mathrm { m a x } } \vert \Vert \mathbf { e } ( t ) \Vert _ { 2 }$ and rise above the reference line at the lead time of (b), reflecting the faster-than-linear growth driven by their $| \lambda _ { \operatorname* { m a x } } | > 1 ;$ by (c) and (d) they have reached the climatological decorrelation level and stopped growing, so they fall below a reference line that continues to grow in proportion to $p .$ This faster-than-linear growth arises from the $\mathcal { O } ( \| \mathbf { e } ( t ) \| _ { 2 } ^ { 2 } )$ higher-order terms in the Taylor expansion of $\tilde { \mathbf { G } }$ , which are neglected in the linear analysis but become non-negligible once the accumulated error grows large. Each model is identified in Table S1. Two of the implicit-Euler models are trained at $\Delta t \gg \Delta t _ { \mathrm { D N S } } ^ { \mathrm { 1 0 } }$ and therefore exceed the $p \Delta t \ll 1$ window by a larger margin at the longest lead time (SI, Section 2.4).

predicted ê(k)  
a)  
b)  
![](images/fa867dd0bc1c8f9bd5b0bc5b6956244a3cbe44d125fa5ec03dbf4c5ef1ab4350.jpg)

![](images/9bdd4abcb73d1c572dc031551fb4d29e63010a043eb35b850c38af7bf5fc5f2f.jpg)

c)  
![](images/4f5c39d9bd629a744b10b44fe274647c24c13f2e3cf254df92d1e24a0b8b41d4.jpg)

d)  
![](images/9c66d8a9796bc31a6dd48603a13be6f78462771913013ea66747656587010e41.jpg)  
Figure 4: Error growth is set by the accumulated generalization error and the error– eigenvector projection, not by $\lvert \lambda _ { \mathrm { m a x } } \rvert$ alone, and is predicted by the linearized error law. $\left( \mathrm { a } , \mathrm { b } \right)$ Two FNO models constrained with the same PEC4 integrator, trained without (FNO, blue) and with (Spectral FNO, green) the spectral regularizer $\mu ( \theta )$ of Eq. (36). (a) RMSE over the autoregressive rollout on log–log axes. For each model the solid line is the total error $\mathbf { e } ( u ( t ) )$ and the black dashed line is the cumulative generalization error $\textstyle \sum _ { i = 1 } ^ { t } \epsilon ( u ( i ) )$ , the leading term of Eq. (14), with the dash length distinguishing the two models. The spectrally regularized model has the smaller error over the early rollout. (b) Eigenvalues of the model Jacobian near $\lambda = 1$ for the FNO (◦) and Spectral FNO (⋆), with the unit circle dotted; to the precision of the legend the two models share $| \lambda _ { \mathrm { m a x } } | = 1 . 0 0 0 2$ Each marker is shaded by the cosine similarity $| \cos ( \mathbf { e } , \mathbf { v } _ { i } ) |$ between its eigenvector $\mathbf { v } _ { i }$ and the error vector $\mathbf { e } ( t )$ (grayscale bar). $( \mathrm { c } , \mathrm { d } )$ The linearized error law of Eq. (20), evaluated with $a = | \lambda _ { \operatorname* { m a x } } |$ from the Jacobian at the initial condition, tested after $k = 1 0$ rollout steps $( t _ { k } = t _ { 0 } + k \Delta t )$ for every explicit integration-constrained model (marker shape: integration scheme, Euler / RK4 $/$ PEC4; fill: MLP filled, FNO open); each point is one model and the gray line is $y = x$ . (c) Total error, with $\lVert \epsilon \rVert$ the measured one-step error. (d) The same law restricted to the high-wavenumber band $( \omega \ge \omega _ { c } ,$ with $\omega _ { c } = 1 0 0 )$ , with the injection term replaced by $\beta ,$ , the high-ω content of the one-step error. See Section 3.3 for the analysis. The two models in (a, b) are models 11 and 12 of Table S1, and the models in (c, d) are models 1 to 24.

a)  
![](images/8e112cc13de8eda0e4333eb9fd6ef2ffa49fa52e31efcccfed3c42fe0739f1fc.jpg)

b)  
![](images/604328d7fc1bc544a8876a4a846bce5a18a9e432c21a9cbb85cb9e0d3b3e85c9.jpg)  
Figure 5: The stability-promoting loss lowers error growth. FNO-based models constrained with the Euler, PEC4, and RK4 integrators, each trained with the baseline RMSE loss and compared against its counterpart trained with the proposed Jacobian-based stability-promoting loss of $\operatorname { E q . }$ (38). (a) The linear scaling law of Fig. 3, $\| \mathbf { e } ( t + \Delta t ) \|$ versus $\lvert \lambda _ { \mathrm { m a x } } \rvert \parallel \mathbf { e } ( t ) \rvert \parallel$ , with the RMSE baselines shown as circles and the stability-promoting loss as stars. All points lie on the predicted unit-slope line of ofset 2 (gray), confirming that the new loss preserves the linear scaling law, and for every integrator the stability-promoting loss yields a smaller $\lvert \lambda _ { \mathrm { m a x } } \rvert \parallel \mathbf { e } ( t ) \rvert$ and correspondingly smaller $\| \mathbf { e } ( t + \Delta t ) \|$ than the baseline. (b) RMSE over the autoregressive rollout for the same models, with the RMSE baselines shown as solid lines and the stability-promoting loss as dotted lines. The legend reports $\alpha .$ the slope of a linear fit of log $_ { \mathrm { { \dot { 1 } 0 } } } \mathrm { { ( R M S E ) } }$ against $\log _ { 1 0 }$ of the timestep over the first 100 rollout steps (the exponent of RMSE $\propto t ^ { \alpha } )$ . For every integrator, the stability-promoting loss produces a smaller $\alpha ,$ and hence lower error growth, than the corresponding RMSE baseline, because it constrains the Jacobian-driven amplification of error, which the RMSE objective alone leaves uncontrolled. The six models shown are models 7, 9, 11, 19, 21, and 23 of Table S1.

# Supporting Information for

Eigenanalysis framework for autoregressive neural emulators of multi-scale chaotic dynamics

Conrad Ainslie, Pedram Hassanzadeh, Michael W. Mahoney, and Ashesh Chattopadhyay

E-mail: aschatto@ucsc.edu

This PDF file includes:

Supporting text, Sections 1 to 4

Figs. S1 to S3

Table S1

Algorithm S1

SI References

## Supporting Information Text

This document contains four sets of supporting results. Section 1 reports the full 29-model suite together with its measured Jacobian spectra, and introduces the time-normalized growth rate used to compare models trained at diferent time steps. Section 2 describes the implicit integration-based hard constraint, derives the Jacobian of the implicit update map, and reports the comparison against explicit schemes over a range of $\Delta t .$ Section 3 presents the Fourier-space analysis of spectral bias. Section 4 establishes the near-normality of the integration-constrained Jacobian and the resulting validity of the eigenvalue approximation used in Section 2.4 of the main text.

## 1 The full model suite

Table S1 lists all 29 models evaluated in this work. Two are direct-step models and 27 are integrationconstrained. Figure S1 demonstrates the RMSE growth and the Jacobian eigenspectra of the 24 explicit integration-constrained models in a single view, providing the complete-suite reference behind the representative subsets shown in Figs. 2 and 3 of the main text.

## 1.1 Time-normalized growth rate

The largest eigenvalue $\lvert \lambda _ { \mathrm { m a x } } \rvert$ is a per-step quantity, so models trained at diferent time steps cannot be ranked by $\lvert \lambda _ { \mathrm { m a x } } \rvert$ alone. Interpreting the learned update map as a time-∆t flow map, $\left| \lambda _ { \operatorname* { m a x } } \right| = \exp ( \sigma \Delta t )$ defines a growth rate per unit time,

$$
\sigma = \frac { \ln | \lambda _ { \mathrm { m a x } } | } { \Delta t } ,\tag{S1}
$$

which is comparable across $\Delta t$ and carries the units of a Lyapunov exponent. Table S1 reports σ alongside $\lvert \lambda _ { \mathrm { m a x } } \rvert$

The normalization separates the two model classes by more than two orders of magnitude. Every integration-constrained model has $\sigma \leq 0 . 3 4$ , and the majority lie in the range 0.01 to 0.25. The two direct-step models have $\sigma = 4 8 . 1$ and $\sigma = 6 0 . 2$ . The direct-step models therefore amplify perturbations at a rate between 140 and 180 times larger than the fastest integration-constrained model, and this separation, rather than the diference in the per-step eigenvalue, is what the rollouts of Fig. 2(a,b) of the main text reflect. The integration-constrained values are of order $1 0 ^ { - 1 }$ per unit time, which is the order of the leading Lyapunov exponent of the KS attractor, so these models amplify perturbations at a rate set by the dynamics rather than by the emulator.

Table S1: The full model suite. All 29 models evaluated in this work, comprising 27 integrationconstrained models and 2 direct-step models. Every explicit model is trained and run at $\Delta t = \Delta t _ { \mathrm { D N S } } =$ $1 0 ^ { - 3 }$ ; the two large-∆t implicit models (26, 27) are the exception. $\lvert \lambda _ { \mathrm { m a x } } \rvert ( t _ { 0 } )$ is the largest-magnitude eigenvalue of the model Jacobian at the initial condition. $\sigma = \ln { \left| \lambda _ { \operatorname* { m a x } } \right| } / \Delta t$ is the corresponding growth rate per unit time of Eq. (S1), which is comparable across $\Delta t ;$ its precision is limited by the five decimal places reported for $\lvert \lambda _ { \mathrm { m a x } } \rvert$ |. RMSE is the short-term error at $k = 1 0$ steps. The spectral-bias metric is $\langle \log _ { 1 0 } ( | \hat { u } _ { \mathrm { p r e d } } | / | \hat { u } _ { \mathrm { t r u e } } | ) \rangle _ { \omega \geq 1 }$ <sub>100</sub> at step 100, with positive values indicating excess high-wavenumber energy relative to truth.
<table><tr><td> $\#$ </td><td>Arch.</td><td>Scheme</td><td>Loss</td><td> $\Delta t$ </td><td> $\left| \lambda _ { \operatorname* { m a x } } \right| ( t _ { 0 } )$ </td><td> $\sigma$ </td><td>RMSE (k = 10)</td><td>Spec. bias</td></tr><tr><td>1</td><td>MLP</td><td>Euler</td><td>RMSE</td><td> $1 0 ^ { - 3 }$ </td><td>1.00018</td><td>0.180</td><td> $1 . 2 8 9 \times 1 0 ^ { - 3 }$ </td><td>12.46</td></tr><tr><td>2</td><td>MLP</td><td>Euler</td><td>Spectral</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 02</td><td>0.020</td><td> $3 . 8 5 0 \times 1 0 ^ { - 4 }$ </td><td>10.46</td></tr><tr><td>3</td><td>MLP</td><td>RK4</td><td>RMSE</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 21</td><td>0.210</td><td> $1 . 3 1 0 \times 1 0 ^ { - 3 }$ </td><td>12.47</td></tr><tr><td>4</td><td>MLP</td><td>RK4</td><td>Spectral</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 07</td><td>0.070</td><td> $3 . 9 7 0 \times 1 0 ^ { - 4 }$ </td><td>10.44</td></tr><tr><td>5</td><td>MLP</td><td>PEC4</td><td>RMSE</td><td> $1 0 ^ { - 3 }$ </td><td>1.00018</td><td>0.180</td><td> $1 . 2 7 3 \times 1 0 ^ { - 3 }$ </td><td>12.49</td></tr><tr><td>6</td><td>MLP</td><td>PEC4</td><td>Spectral</td><td> $1 0 ^ { - 3 }$ </td><td>1.00030</td><td>0.300</td><td> $3 . 5 7 7 \times 1 0 ^ { - 4 }$ </td><td>10.35</td></tr><tr><td>7</td><td>FNO</td><td>Euler</td><td>RMSE</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 20</td><td>0.200</td><td> $1 . 1 4 2 \times 1 0 ^ { - 4 }$ </td><td>9.82</td></tr><tr><td>8</td><td>FNO</td><td>Euler</td><td>Spectral</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 19</td><td>0.190</td><td> $2 . 5 2 6 \times 1 0 ^ { - 5 }$ </td><td>8.79</td></tr><tr><td>9</td><td>FNO</td><td>RK4</td><td>RMSE</td><td> $1 0 ^ { - 3 }$ </td><td>1.00034</td><td>0.340</td><td> $2 . 3 4 3 \times 1 0 ^ { - 4 }$ </td><td>9.87</td></tr><tr><td>10</td><td>FNO</td><td>RK4</td><td>Spectral</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 20</td><td>0.200</td><td> $1 . 0 3 5 \times 1 0 ^ { - 4 }$ </td><td>8.88</td></tr><tr><td>11</td><td>FNO</td><td>PEC4</td><td>RMSE</td><td> $1 0 ^ { - 3 }$ </td><td>1.00019</td><td>0.190</td><td> $1 . 1 5 3 \times 1 0 ^ { - 4 }$ </td><td>9.55</td></tr><tr><td>12</td><td>FNO</td><td>PEC4</td><td>Spectral</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 20</td><td>0.200</td><td> $2 . 2 1 4 \times 1 0 ^ { - 5 }$ </td><td>8.81</td></tr><tr><td>13</td><td>MLP</td><td>Euler</td><td>Jacobian</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 01</td><td>0.010</td><td> $3 . 8 5 0 \times 1 0 ^ { - 4 }$ </td><td>10.86</td></tr><tr><td>14</td><td>MLP</td><td>Euler</td><td>Spectral+Jac.</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 05</td><td>0.050</td><td> $8 . 1 2 1 \times 1 0 ^ { - 5 }$ </td><td>11.01</td></tr><tr><td>15</td><td>MLP</td><td>PEC4</td><td>Jacobian</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 02</td><td>0.020</td><td> $3 . 7 3 9 \times 1 0 ^ { - 4 }$ </td><td>11.23</td></tr><tr><td>16</td><td>MLP</td><td>PEC4</td><td>Spectral+Jac.</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 08</td><td>0.080</td><td> $2 . 3 3 5 \times 1 0 ^ { - 4 }$ </td><td>11.03</td></tr><tr><td>17</td><td>MLP</td><td>RK4</td><td>Jacobian</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 01</td><td>0.010</td><td> $3 . 9 3 5 \times 1 0 ^ { - 4 }$ </td><td>10.96</td></tr><tr><td>18</td><td>MLP</td><td>RK4</td><td>Spectral+Jac.</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 01</td><td>0.010</td><td> $5 . 0 7 1 \times 1 0 ^ { - 5 }$ </td><td>10.95</td></tr><tr><td>19</td><td>FNO</td><td>Euler</td><td>Jacobian</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 21</td><td>0.210</td><td> $1 . 7 9 4 \times 1 0 ^ { - 5 }$ </td><td>9.83</td></tr><tr><td>20</td><td>FNO</td><td>Euler</td><td>Spectral+Jac.</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 21</td><td>0.210</td><td> $9 . 7 1 8 \times 1 0 ^ { - 6 }$ </td><td>9.80</td></tr><tr><td>21</td><td>FNO</td><td>PEC4</td><td>Jacobian</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 22</td><td>0.220</td><td> $2 . 9 7 7 \times 1 0 ^ { - 5 }$ </td><td>9.93</td></tr><tr><td>22</td><td>FNO</td><td>PEC4</td><td>Spectral+Jac.</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 25</td><td>0.250</td><td> $3 . 3 8 1 \times 1 0 ^ { - 5 }$ </td><td>10.27</td></tr><tr><td>23</td><td>FNO</td><td>RK4</td><td>Jacobian</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 23</td><td>0.230</td><td> $2 . 4 8 2 \times 1 0 ^ { - 5 }$ </td><td>10.00</td></tr><tr><td>24</td><td>FNO</td><td>RK4</td><td>Spectral+Jac.</td><td> $1 0 ^ { - 3 }$ </td><td>1.000 21</td><td>0.210</td><td> $4 . 4 5 3 \times 1 0 ^ { - 5 }$ </td><td>10.26</td></tr><tr><td>25</td><td>FNO</td><td>Implicit Euler</td><td>RMSE</td><td> $1 0 ^ { - 3 }$ </td><td>1.00010</td><td>0.100</td><td> $4 . 1 3 1 \times 1 0 ^ { - 4 }$ </td><td>10.21</td></tr><tr><td>26</td><td>FNO</td><td>Implicit Euler</td><td>RMSE</td><td> $5 \times 1 0 ^ { - 2 }$ </td><td>1.000 20</td><td>0.004</td><td> $1 . 8 7 6 \times 1 0 ^ { - 2 }$ </td><td>10.08</td></tr><tr><td>27</td><td>FNO</td><td>Implicit Euler</td><td>RMSE</td><td> $1 0 ^ { - 1 }$ </td><td>1.01890</td><td>0.187</td><td> $3 . 8 4 0 \times 1 0 ^ { - 2 }$ </td><td>10.48</td></tr><tr><td>28</td><td>MLP</td><td>Direct</td><td>RMSE</td><td> $1 0 ^ { - 3 }$ </td><td>1.04930</td><td>48.123</td><td> $5 . 9 0 9 \times 1 0 ^ { - 1 }$ </td><td>14.17</td></tr><tr><td>29</td><td>FNO</td><td>Direct</td><td>RMSE</td><td> $1 0 ^ { - 3 }$ </td><td>1.06200</td><td>60.154</td><td> $3 . 4 3 3 \times 1 0 ^ { - 2 }$ </td><td>11.12</td></tr></table>

The quantity $\sigma$ inherits the precision of $\lvert \lambda _ { \mathrm { m a x } } \rvert$ . At $\Delta t = 1 0 ^ { - 3 }$ the five-decimal values of Table S1 re solve σ to one or two significant figures only, so small diferences in σ within the integration-constrained class should not be over-interpreted. Equation (S1) also uses $\lvert \lambda _ { \mathrm { m a x } } \rvert$ and therefore describes the mod ulus of the dominant multiplier, not the full complex spectrum.

## 2 Implicit integration-based hard constraints

## 2.1 Implementation

Implicit time-integration schemes have improved stability properties for larger $\Delta t ,$ as compared with explicit integration schemes [Frank et al., 1997]. In order to implement implicit time-integration as a hard constraint, we use the theory of implicit layers [Kawaguchi, 2021] in deep learning. In this approach, we can use any higher-order scheme as our $\mathbf { H } \left[ \circ \right]$ operator. The equation involving the time stepper is given by

$$
\underset { \mathbf { y } ^ { * } } { \underbrace { \mathbf { u } ( x , t + \Delta t ) } } = \mathbf { u } ( x , t ) + \Delta t \mathbf { H } \left[ \mathcal { N } \left[ \underset { \mathbf { y } ^ { * } } { \underbrace { \mathbf { u } ( x , t + \Delta t ) } } , \theta \right] \right] ,\tag{S2}
$$

![](images/2b24a6f064fbf8a91f559ea7d4bd783765cc692005d7dc77e2406d4a6e7f2c9b.jpg)  
Figure S1: Short-term error and eigenspectra of the full explicit integration-constrained suite. All 24 explicit integration-constrained models of Table S1, spanning the Euler, RK4, and PEC4 schemes, the $\mathrm { M L P }$ and FNO architectures, and the RMSE, spectral, Jacobian, and spectral+Jacobian losses. Every model is trained and emulated at $\Delta t = \Delta t _ { \mathrm { D N S } } = 1 0 ^ { - 3 }$ , with $\mathrm { M L P }$ models drawn as solid lines and FNO models as dotted lines, and color and marker distinguishing individual models as indicated in the legend. (a) RMSE over the first 100 rollout steps on log–log axes. Every model exhibits the slow, near-linear error accumulation predicted by Eq. (16) of the main text; the curves separate in vertical ofset, set by the one-step generalization error ϵ, rather than in slope, consistent with the near-neutral spectra shared across the suite. The window shown satisfies $p \Delta t \le 0 . 1$ , so it lies inside the validity range of that scaling law. (b) Eigenvalues of the model Jacobian $\mathbf { J } ,$ evaluated at the initial condition $t _ { 0 } .$ on the Argand plane, with the unit circle dotted. For every model the eigenvalues collapse onto the unit circle and cluster tightly at $\lambda = 1$ , with the spread of each spectrum confined to the third decimal place and the models separated in the fourth (Table S1 reports $\lvert \lambda _ { \mathrm { m a x } } \rvert ( t _ { 0 } )$ for each). This confirms that the qualitative spectrum is insensitive to the architecture, the explicit integration scheme, and the loss function, as expected from the Jacobian structure of $\operatorname { E q . }$ (10) of the main text.

where $\mathbf { y } ^ { * }$ is required to be solved via an implicit layer at each epoch, that is, at each value of $\theta ,$ in the training process. In order to do that, we initialize $\mathbf { y } ^ { * } = \mathbf { u } \left( x , t \right)$ , and we execute a fixed-point iteration to converge to the correct $\mathbf { u } ( x , t + \Delta t )$ . More details about implicit layers and deep equilibrium models can be found in Bai et al. [2019], Kawaguchi [2021]. For the experiments with the implicit method, the chosen $\Delta t$ ranges from $\Delta t _ { D N S }$ up to $1 0 0 \Delta t _ { D N S }$ . Algorithm S1 outlines the fixed-point iteration used to train and to run inference with the implicit numerical integrator.

## 2.2 The Jacobian of the implicit update map

The implicit models do not share the Jacobian of the explicitly constrained models, and the diference is what produces their distinct eigenspectrum. Diferentiating $\operatorname { E q . }$ (S2) with respect to the state $\mathbf { u } _ { T }$ and using the fact that $\mathbf { y } ^ { * }$ itself depends on $\mathbf { u } _ { T }$ , gives

$$
\frac { \partial \mathbf { y } ^ { * } } { \partial \mathbf { u } _ { T } } = \mathbf { I } + \Delta t \nabla \mathbf { H } [ \mathcal { N } ( \mathbf { y } ^ { * } ; \boldsymbol { \theta } ) ] \frac { \partial \mathbf { y } ^ { * } } { \partial \mathbf { u } _ { T } } ,\tag{S3}
$$

Algorithm S1 Implicit time integration scheme   
1: Input: ϵ ▷ Convergence threshold, typically $\overline { { 1 0 ^ { - 8 } } }$   
2: At each epoch, fix θ   
3: Initialize $\mathbf { y } _ { 0 } ^ { * } = \mathbf { u } ( x , t ) , k = 0$   
4: repeat   
5: $k = k + 1$   
6: $\mathbf { y } _ { k } ^ { * } = \mathbf { u } ( x , t ) + \Delta t \mathbf { H } \left[ \mathcal { N } \left( \mathbf { y } _ { k - 1 } ^ { * } , \boldsymbol { \theta } \right) \right]$   
7: until $| | \mathbf { y } _ { k } ^ { * } - \mathbf { y } _ { k - 1 } ^ { * } | | _ { 2 } \leq \epsilon$   
8: Output: $\mathbf { y } _ { k } ^ { * }$ $\vartriangleright \mathbf { y } _ { k } ^ { * }$ is the predicted value of $\mathbf { u } ( x , t + \Delta t )$

so that the Jacobian of the implicit update map is

$$
\mathbf { J } = \left( \mathbf { I } - \Delta t \nabla \mathbf { H } [ \mathcal { N } ( \mathbf { y } ^ { * } ; \boldsymbol { \theta } ) ] \right) ^ { - 1 } .\tag{S4}
$$

Expanding Eq. (S4) in a Neumann series gives $\mathbf { J } = \mathbf { I } + \Delta t \nabla \mathbf { H } [ \mathcal { N } ( \mathbf { y } ^ { * } ; \boldsymbol { \theta } ) ] + \mathcal { O } ( \Delta t ^ { 2 } )$ , so the implicit Jacobian agrees with Eq. (10) of the main text to first order in ∆t. The conclusion about explicit schemes carries over to the implicit models, and the implicit models are integration-constrained in that sense. The distinction matters only at $\mathcal { O } ( \Delta t ^ { 2 } )$ and above, which is the regime probed by the large-∆t experiments of Section 2.4.

The inverse structure of Eq. (S4) controls the spectrum. Let $\nu _ { i }$ denote the eigenvalues of $\nabla \mathbf { H } [ \mathcal { N } ( \mathbf { y } ^ { * } ; \boldsymbol { \theta } ) ]$ which carry the units of a rate. The eigenvalues of the implicit Jacobian are

$$
\lambda _ { i } = \frac { 1 } { 1 - \Delta t \nu _ { i } } ,\tag{S5}
$$

whereas the corresponding explicit scheme yields $\lambda _ { i } = 1 + \Delta t \nu _ { i }$ . Writing $\nu _ { i } = a _ { i } + \mathrm { i } b _ { i }$ and evaluating the moduli exactly gives the two conditions for local expansion,

$$
| \lambda _ { i } | > 1 \longleftrightarrow a _ { i } > \frac { \Delta t } { 2 } | \nu _ { i } | ^ { 2 } \quad \mathrm { ( i m p l i c i t ) } , \qquad | \lambda _ { i } | > 1 \longleftrightarrow a _ { i } > - \frac { \Delta t } { 2 } | \nu _ { i } | ^ { 2 } \quad \mathrm { ( e x p l i c i t ) } .\tag{S6}
$$

The implicit scheme places an eigenvalue outside the unit circle only when the learned tangent operator has an expanding direction, $a _ { i } > 0$ . The explicit scheme places an eigenvalue outside the unit circle even for mildly contracting directions with $- \frac { \Delta t } { 2 } | \nu _ { i } | ^ { 2 } < a _ { i } < 0 .$ . At fixed $\Delta t ,$ the implicit spectrum therefore lies closer to and inside the unit disk than the explicit one. This is the origin of the smaller $\lvert \lambda _ { \mathrm { m a x } } \rvert$ and the longer left tail of the implicit eigenspectra in Fig. S2(d), and it is the spectral counterpart of the classical statement that implicit schemes have larger stability regions.

## 2.3 Short-term and long-term performance

From classical numerical analysis, we know that implicit numerical time integrators, while more computationally expensive, have larger stability regions than explicit time integration schemes, and in practice are more accurate at large ∆t for stif problems. We use our proposed a priori diagnostic metric, $| \lambda _ { \mathrm { m a x } } | .$ to investigate whether similar properties emerge for neural autoregressive models. We consider models that are hard-constrained with implicit and explicit integration schemes and are trained and tested with $\Delta t$ ranging from $\Delta t _ { D N S } \mathrm { ~ u p ~ t o ~ } 1 0 0 \Delta t _ { D N S }$ . To do so, we have performed experiments with $\Delta t = \Delta t _ { D N S } = 1 0 ^ { - 3 } , \Delta t = 5 0 \Delta t _ { D N S } = 5 \times 1 0 ^ { - 2 }$ , and $\Delta t = 1 0 0 \Delta t _ { D N S } = 1 0 ^ { - 1 }$ . Unless otherwise stated, all models in this work are trained and emulated at $\Delta t = \Delta t _ { D N S }$ The large-∆t models of this section are the exception, and the explicit models trained at $\Delta t > \Delta t _ { D N S }$ are used only for this comparison and are not part of the 29-model suite of Table S1.

As shown in $\mathrm { F i g . \ S 2 ( a ) }$ , similar to the explicit integration-based constraint with the Euler scheme, the implicit Euler scheme-based model is also long-term stable and physically consistent. However, the RMSE error curves in $\mathrm { F i g . \ S 2 ( c ) }$ clearly demonstrate that for all values of $\Delta t ,$ the implicit time integration constraint-based models have lower error growth, as compared to explicit ones. This is expected from traditional numerical analysis. This is further validated by investigating $\lvert \lambda _ { \mathrm { m a x } } \rvert$ of each of the implicit schemes and confirming that they are smaller, that is, closer to unity, than the explicit schemes. Similar to explicit constraints, for implicit constraints as well, $\lvert \lambda _ { \mathrm { m a x } } \rvert$ acts as an a priori diagnostic of the error-growth rate of the models. This is of particular importance when developing novel autoregressive architectures for complex systems. The integration constraints allow us to translate our knowledge about the performance and stability of diferent integration schemes to neural network-based modeling in scientific computing.

The comparison across $\Delta t$ requires the normalization of Eq. (S1), since $\lvert \lambda _ { \mathrm { m a x } } \rvert$ is a per-step multiplier and the three experiments use diferent step sizes. Expressed as growth rates per unit time, the implicit constraint gives $\sigma = 0 . 1 0 0 , 0 . 0 0 4$ , and $0 . 1 8 7$ at $\Delta t = 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 2 }$ , and $1 0 ^ { - 1 }$ , against $\sigma = 0 . 2 0 0$ 0.155, and 0.243 for the corresponding explicit models. The implicit scheme has the lower rate at every $\Delta t$ tested. The raw $\lvert \lambda _ { \mathrm { m a x } } \rvert$ values reported in the legend of Fig. S2(c) order the implicit and explicit models correctly within each $\Delta t$ pair, but they cannot be compared across pairs, and the normalized rate is what makes the three comparisons commensurate.

## 2.4 Large time steps, expanding eigenvalues, and stable emulation

Model 27 of Table S1 is an implicit Euler FNO trained and emulated at $\Delta t = 1 0 ^ { - 1 }$ . Its Jacobian has $| \lambda _ { \mathrm { m a x } } | = 1 . 0 1 8 9$ , which places an eigenvalue outside the unit circle, and the linear analysis of Section 2.3 of the main text therefore predicts local expansion. The model nonetheless remains stable and physically consistent over the full emulation.

The quantity $\lvert \lambda _ { \mathrm { m a x } } \rvert$ is a multiplier per time step, and models trained at diferent $\Delta t$ take diferent numbers of steps to cover the same physical time, so their per-step values cannot be compared with one another. Equation (S1) converts $\lvert \lambda _ { \mathrm { m a x } } \rvert$ to a growth rate per unit time, which is comparable. Model 27 gives $\sigma = 0 . 1 8 7$ . The two direct-step models give $\sigma = 4 8 . 1$ and $\sigma = 6 0 . 2$ , larger by factors of roughly 260 and 320, even though their per-step eigenvalues, 1.049 and $1 . 0 6 2 .$ are numerically close to 1.019. The closeness of the three raw eigenvalues follows from the diference in $\Delta t$ and carries no information about the models.

A rate of $\sigma \approx 0 . 1 9$ is what an emulator of the KS attractor should give. The attractor has positive Lyapunov exponents, so a tangent operator that reproduces it must have expanding directions, and an emulator whose spectrum lay strictly inside the unit circle would be over-difusive rather than accurate. Equation (S5) converts $| \lambda _ { \mathrm { m a x } } | = 1 . 0 1 8 9$ at $\Delta t = 1 0 ^ { - 1 }$ into an expanding eigenvalue of the learned tangent operator of 0.185 per unit time, which agrees with $\sigma = 0 . 1 8 7$ to within one percent, the residual diference being of order $\Delta t$ and expected from the two definitions. The expansion described by $| \lambda _ { \operatorname* { m a x } } | > 1$ is local and is bounded by nonlinear saturation on the attractor, so it does not imply unbounded divergence, and the emulation confirms this.

Sections 2.3 and 2.4 of the main text assume $\Delta t \ll 1$ . At $\Delta t = 1 0 ^ { - 1 }$ that condition is marginal, the $\mathcal { O } ( \Delta t ^ { 2 } )$ terms neglected in Section 4 are no longer small, and the quantitative predictions of the linear theory carry a correspondingly wider tolerance for this model.

Equation (16) of the main text holds for $p \Delta t \ll 1$ , and this restriction governs where the implicit models sit in Fig. 3 of the main text. At the longest lead time, $p = 1 0 0 1$ , models 26 and 27 reach $p \Delta t = 5 0$ and $p \Delta t = 1 0 0$ , so the condition fails by two orders of magnitude. Their departure from the reference line in that panel is what the stated validity window predicts and is not evidence against the scaling law.

## 3 Spectral bias and the Fourier spectrum of the emulation

This section presents the Fourier-space analysis of spectral bias, connecting the eigenvalues of the model Jacobian to the fidelity of the predicted spectrum and its time derivative.

We analyze whether $\lvert \lambda _ { \mathrm { m a x } } \rvert$ can explain diferent values of spectral bias in neural autoregressive models. As shown previously [Chattopadhyay et al., 2023b, Guan et al., 2025], a spectral regularizer in the form of Eq. (36) of the main text can reduce spectral bias in neural autoregressive models and improve stability. However, for a decaying spectrum like the one in the KS system, the regularizer does not completely eliminate spectral bias either in the state or in its derivative (Fig. S3(b,c)). This is because the amplitude of the modes of the state in a system with a decaying spectrum decreases as a power law. Figure S3(b) shows that after wavenumber 100, the slope of the spectrum increases with diminished amplitude of the high-wavenumber modes. Despite the spectral regularizer, spectra with high slopes are more challenging to capture, an efect mirrored in the spectra of state derivatives (Fig. S3(c)) as well. A detailed explanation of the impact of the slope of the system spectrum on the spectral bias of the autoregressive model can be found in Chattopadhyay et al. [2023b].

Nonetheless, the extent of spectral bias varies across architectures. For instance, neural operators have been shown to better capture the spatial Fourier spectra of chaotic systems [Azizzadenesheli et al., 2024, Oommen et al., 2025]. The magnitude of $\lvert \lambda _ { \mathrm { m a x } } \rvert$ , representing the distance of the dominant Jacobian eigenvalue from the origin, reflects the implicit difusivity of a model. Smaller values indicate higher difusion and often correlate with reduced spectral bias. Figure $\mathrm { S 3 ( c ) }$ shows that the FNO model equipped with the PEC-based integrator and a spectral regularizer has lower spectral bias, and hence more difusion, than the MLP-based model. This is captured by the smaller $\lvert \lambda _ { \mathrm { m a x } } \rvert$ of the FNO model as compared to the MLP model, as shown in $\mathrm { F i g . \ S 3 ( a ) }$

The quantity $\lvert \lambda _ { \mathrm { m a x } } \rvert$ therefore complements the one-step error as an a priori indicator. It sets the amplification scale, while ϵ sets the injection. For the matched pair studied here it also tracks the implicit difusivity and the fidelity of the high-wavenumber spectrum. This correspondence is established for a single matched pair of models and should be read as an empirical association rather than as a general law. Section 3.3 of the main text establishes the more general statement, namely that the high-wavenumber error after k rollout steps is set by the high-ω content of the injected one-step error rather than by preferential amplification of high wavenumbers.

## 4 Near-normality of the integration-constrained Jacobian and validity of the eigenvalue approximation

The scaling arguments of Section 2.4 of the main text replace the action of the Jacobian on the error vector by its largest eigenvalue, $\lVert \mathbf { J e } \rVert \approx \lvert \lambda _ { \mathrm { m a x } } \rvert \rVert \mathbf { e } \rVert$ . For a general non-normal matrix this replacement can fail. Eigenvalues bound only the asymptotic growth of perturbations, while transient, orientationdependent amplification is controlled by the singular values, which for a strongly non-normal matrix can exceed $\lvert \lambda _ { \mathrm { m a x } } \rvert$ substantially. Here we show that the integration-constrained structure, together with the smallness of $\Delta t ,$ renders the constrained Jacobian near-normal, so that the eigenvalue approximation is accurate to the same order as the theory itself. This argument applies only to the integrationconstrained models. No analogous control is available for the direct-step Jacobian.

Write the constrained Jacobian of Eq. (10) of the main text as $\mathbf { J } = \mathbf { I } + \Delta t \mathbf { A }$ , with $\mathbf { A } = \nabla \mathbf { H } [ \mathcal { N } ( \mathbf { u } _ { T } ; \boldsymbol { \theta } ) ]$ Its departure from normality is measured by the self-commutator,

$$
\mathbf { J } \mathbf { J } ^ { * } - \mathbf { J } ^ { * } \mathbf { J } = { \boldsymbol { \Delta } } t ^ { 2 } \left( \mathbf { A } \mathbf { A } ^ { * } - \mathbf { A } ^ { * } \mathbf { A } \right) ,\tag{S7}
$$

since the identity commutes with every matrix and the terms linear in $\Delta t$ cancel. The departure from normality is therefore $\mathcal { O } ( \Delta t ^ { 2 } )$ , one order smaller than the $\mathcal { O } ( \Delta t )$ corrections already retained in the scaling analysis. To first order in $\Delta t$ the constrained Jacobian is normal, regardless of how non-normal the network Jacobian A itself may be.

The same structure controls the singular values directly. The squared singular values of J are the eigenvalues of

$$
{ \bf J } ^ { * } { \bf J } = { \bf I } + \Delta t \left( { \bf A } + { \bf A } ^ { * } \right) + \Delta t ^ { 2 } { \bf A } ^ { * } { \bf A } ,\tag{S8}
$$

so that, for bounded $\| \mathbf { A } \|$ , every singular value satisfies $\sigma _ { i } ( \mathbf { J } ) = 1 + \mathcal { O } ( \Delta t )$ . Consequently, for an error vector of any orientation,

$$
\frac { \| \mathbf { J _ { \theta } } \mathbf { e } \| } { \| \mathbf { e } \| } \in [ \sigma _ { \operatorname* { m i n } } ( \mathbf { J } ) , \sigma _ { \operatorname* { m a x } } ( \mathbf { J } ) ] = 1 + \mathcal { O } ( \Delta t ) , \qquad \mathrm { w h i l e } \qquad | \lambda _ { \operatorname* { m a x } } | = 1 + \mathcal { O } ( \Delta t ) ,\tag{S9}
$$

so that $\left\| \mathbf { J } \mathbf { e } \right\| = { \big | } \lambda _ { \operatorname* { m a x } } { \big | } \left\| \mathbf { e } \right\| + { \mathcal { O } } ( \Delta t ) \left\| \mathbf { e } \right\|$ . Replacing the vectorial action of J by the scalar $\lvert \lambda _ { \mathrm { m a x } } \rvert$ is therefore exact at leading order, and the error incurred is of the same $\mathcal { O } ( \Delta t )$ size as the corrections already neglected in Section 2.4 of the main text. At $\Delta t = \Delta t _ { \mathrm { D N S } } = 1 0 ^ { - 3 }$ this residue is small, consistent with the fourth-decimal spread of the measured spectra in $\mathrm { F i g . 2 ( d ) }$ of the main text and in Table S1.

The symbol $\sigma _ { i }$ in Eqs. (S8) and (S9) denotes a singular value of J and is distinct from the growth rate σ of Eq. (S1).

It is this near-identity structure, rather than the clustering of the eigenvalues per se, that justifies the directional insensitivity invoked in Section 2.4 of the main text. Clustered eigenvalues alone would not constrain the transient amplification of a strongly non-normal matrix. The $\mathcal { O } ( \Delta t )$ residue left open by the leading-order approximation is the range within which the direction-dependent efects of

Section 3.3 of the main text act. The error-eigenvector projections and the accumulated one-step error modulate the realized amplification within the $\mathcal { O } ( \Delta t )$ band around unity. The deviations documented in Section 3.3 of the main text are thus the expected finite-∆t corrections to the leading-order law, rather than a breakdown of the framework. For the direct-step models, by contrast, the Jacobian $\nabla \mathcal { N } ( \mathbf { u } _ { T } ; \boldsymbol { \theta } )$ carries no $\Delta t$ scaling, its departure from normality is uncontrolled, and eigenvalue-based estimates should be read only as bounds on asymptotic growth.

The same $\Delta t$ scaling delimits where the argument applies. At $\Delta t = 1 0 ^ { - 1 }$ , used for model 27 of Table S1, the $\mathcal { O } ( \Delta t ^ { 2 } )$ departure from normality is four orders of magnitude larger than at $\Delta t = \Delta t _ { \mathrm { D N S } }$ and the near-normality guarantee is correspondingly weaker. The implicit Jacobian of Eq. (S4) admits the same expansion, $\mathbf { J } = \mathbf { I } + \Delta t \mathbf { A } + \mathcal { O } ( \Delta t ^ { 2 } )$ with $\mathbf { A } = \nabla \mathbf { H } [ \mathcal { N } ( \mathbf { y } ^ { * } ; \boldsymbol { \theta } ) ]$ , so Eqs. (S7) to (S9) hold for the implicit models at leading order as well.

![](images/68904435c8a4c0195222ff725f9b0d684d0c26c3f4da8162e603661607c5ec3a.jpg)

b)  
![](images/9f8ef24721ab0c0f8e56a2e827eee0261420d69de3913e1b3fb32775b7647f82.jpg)

c)  
![](images/405c9bf1c758948d1a17987aa1cda8050674a6296100895df6b0006abe041a36.jpg)

d)  
![](images/199371a63a05143d1a6fbbbecc027d4e2aa277ccdb3a313137bc301a29af4dd6.jpg)  
Figure S2: Short-term and long-term performance of neural autoregressive models constrained with implicit time integration. Color encodes the time step, $\Delta t = 0 . 1$ in blue, 0.05 in vermilion, 0.001 in black. Implicit schemes are solid lines and filled pentagons, explicit schemes dotted lines and open stars. (a) Long-term emulation of the KS state $u ( x , t )$ by the FNO model constrained with first-order implicit Euler at $\Delta t = 0 . 0 5 = 5 0 \Delta t _ { \mathrm { D N S } }$ , stable and physically plausible over $1 0 ^ { 5 }$ steps of length $\Delta t _ { \mathrm { D N S } }$ . (b) Pointwise diference between the emulation in $\mathrm { ( a ) }$ and the reference numerical solution. The error stays small and develops $\mathrm { s l o w l y , }$ with structure inherited from the flow rather than diverging, which quantifies the physical consistency. (c) RMSE growth on linear axes for the FNO models with implicit and explicit Euler at the three time steps. At each $\Delta t$ the implicit scheme gives lower error growth, and the linear scale makes the separation visible late in the rollout, where accumulated error dominates. The legend reports $\lvert \lambda _ { \mathrm { m a x } } \rvert$ , closer to unity for the implicit scheme in each pair. Within each $\Delta t$ pair the ordering of $\lvert \lambda _ { \mathrm { m a x } } \rvert$ matches the ordering of the RMSE curves, so $\lvert \lambda _ { \mathrm { m a x } } \rvert$ acts as an a priori diagnostic of the error-growth rate. Comparison across pairs requires the normalized rate σ of Eq. (S1), reported in Section 2.3. (d) Eigenvalues on the Argand plane for both schemes at $\Delta t = 0 . 1$ and 0.05, with the unit circle dotted. At each $\Delta t$ the implicit spectrum has a smaller maximum and a more difusive left tail than the explicit one, consistent with its lower error growth. Equation (S6) accounts for both. The implicit map places an eigenvalue outside the unit circle only for expanding directions of the learned tangent operator, the explicit map even for mildly contracting ones.

a)  
![](images/973444517378ad7cc9c8ec8563002000f71d40b8cf8a52edf11965eaebb67976.jpg)

b)  
![](images/75724e1cd83a9cd16fc611534e87af92ce1f7130d7dd5c646f8080859dad0880.jpg)  
ω

c)  
![](images/61301bb8ac63322756562fe521ecd865b87163d56fc5810513f4df7724780d97.jpg)  
ω  
Figure S3: Spectral bias and its connection to the eigenvalues of the model Jacobian. (a) Eigenvalues on the Argand plane for the MLP-based model (vermilion plus signs, $| \lambda _ { \mathrm { m a x } } | = 1 . 0 0 0 3 )$ and the FNO-based model (blue open stars, $| \lambda _ { \mathrm { m a x } } | = 1 . 0 0 0 2 )$ , both constrained with the PEC4 integrator and trained with the spectral regularizer of $\operatorname { E q . }$ (36) of the main text, with the unit circle dotted. The FNO-based model has the smaller $\lvert \lambda _ { \mathrm { m a x } } \rvert$ and a longer left tail, that is, eigenvalues further inside the unit circle, indicating greater implicit difusivity. This is reflected in (b) the spatial Fourier spectrum $| \hat { u } ( \omega ) |$ of the predicted state after 100 time steps and (c) the spectrum of its time derivative $d | { \hat { u } } | / d t$ Up to $\omega \approx 1 0 0$ both models track the reference numerical data (dotted black). In the shaded highwavenumber band $( \omega \gtrsim 1 0 0 )$ , where the true spectrum continues to decay, both learned models instead saturate at a flat noise floor set by the smallest scale each can represent rather than by the true dynamics. The floor is markedly lower for the FNO (blue) than for the MLP (vermilion) in both the state and its derivative, so the FNO more faithfully captures the high-wavenumber tail. For this matched pair, $\lvert \lambda _ { \mathrm { m a x } } \rvert$ thus tracks the implicit difusivity and the high-ω noise floor, with the smaller $\lvert \lambda _ { \mathrm { m a x } } \rvert$ corresponding to greater implicit difusion and more faithful capture of the high-wavenumber tail of the spectrum.

## References

K. Azizzadenesheli, N. Kovachki, Z. Li, M. Liu-Schiafini, J. Kossaifi, and A. Anandkumar. Neura operators for accelerating scientific simulations and design. Nature Reviews Physics, 6(5):320–328, 2024.

S. Bai, J. Z. Kolter, and V. Koltun. Deep equilibrium models. Advances in neural information processing systems, 32, 2019.

K. Bi, L. Xie, H. Zhang, X. Chen, X. Gu, and Q. Tian. Accurate medium-range global weather forecasting with 3D neural networks. Nature, pages 1–6, 2023.

Y. Cao, Z. Fang, Y. Wu, D.-X. Zhou, and Q. Gu. Towards understanding the spectral bias of deep learning. arXiv preprint arXiv:1912.01198, 2019.

A. Chattopadhyay, M. Mustafa, P. Hassanzadeh, E. Bach, and K. Kashinath. Towards physics-inspired data-driven weather forecasting: integrating data assimilation with a deep spatial-transformer-based U-NET in a case study with ERA5. Geoscientific Model Development, 15(5):2221–2237, 2022.

A. Chattopadhyay, J. Pathak, E. Nabizadeh, W. Bhimji, and P. Hassanzadeh. Long-term stability and generalization of observationally-constrained stochastic data-driven models for geophysica turbulence. Environmental Data Science, 2:e1, 2023a.

A. Chattopadhyay, Y. Q. Sun, and P. Hassanzadeh. Challenges of learning multi-scale dynamics with ai weather models: Implications for stability and one solution. arXiv e-prints, pages arXiv–2304, 2023b.

A. Chattopadhyay, M. Gray, T. Wu, A. B. Lowe, and R. He. Oceannet: a principled neural operatorbased digital twin for regional oceans. Scientific Reports, 14(1):21181, 2024.

Z. Chen and D. Xiu. On generalized residual network for deep learning of unknown dynamical systems. Journal of Computational Physics, 438:110362, 2021.

H. Fan, J. Jiang, C. Zhang, X. Wang, and Y.-C. Lai. Long-term prediction of chaotic systems with machine learning. Physical Review Research, 2(1):012080, 2020.

D. Floryan. On instabilities in neural network-based physics simulators. arXiv preprint arXiv:2406.13101, 2024.

J. Frank, W. Hundsdorfer, and J. G. Verwer. On the stability of implicit-explicit linear multistep methods. Applied Numerical Mathematics, 25(2-3):193–205, 1997.

H. Guan, T. Arcomano, A. Chattopadhyay, and R. Maulik. Lucie: A lightweight uncoupled climate emulator with long-term stability and physical consistency. Journal of Advances in Modeling Earth Systems, 17(11):e2025MS005152, 2025.

L. Hodgkinson and M. W. Mahoney. Multiplicative noise and heavy tails in stochastic optimization. Technical Report Preprint: arXiv:2006.06293, 2020.

L. Hodgkinson, Z. Wang, and M. W. Mahoney. Models of heavy-tailed mechanistic universality. Technical Report Preprint: arXiv:2506.03470, 2025.

R. Jiang, X. Zhang, K. Jakhar, P. Y. Lu, P. Hassanzadeh, M. Maire, and R. Willett. Hierarchical implicit neural emulators. Advances in Neural Information Processing Systems, 38:73718–73751, 2026.

K. Kawaguchi. On the theory of implicit deep learning: Global convergence with implicit layers. arXiv preprint arXiv:2102.07346, 2021.

R. Keisler. Forecasting global weather with graph neural networks. arXiv preprint arXiv:2202.07575, 2022.

A. S. Krishnapriyan, A. Gholami, S. Zhe, R. M. Kirby, and M. W. Mahoney. Characterizing possible failure modes in physics-informed neural networks. Technical Report Preprint: arXiv:2109.01050, 2021.

A. S. Krishnapriyan, A. F. Queiruga, N. B. Erichson, and M. W. Mahoney. Learning continuous models for continuous physics. Communications Physics, 6(1):319, 2023.

C.-Y. Lai, P. Hassanzadeh, A. Sheshadri, M. Sonnewald, R. Ferrari, and V. Balaji. Machine learning for climate physics and simulations. Annual Review of Condensed Matter Physics, 16(1):343–365, 2025.

R. Lam, A. Sanchez-Gonzalez, M. Willson, P. Wirnsberger, M. Fortunato, A. Pritzel, S. Ravuri, T. Ewalds, F. Alet, Z. Eaton-Rosen, et al. Graphcast: Learning skillful medium-range global weather forecasting. arXiv preprint arXiv:2212.12794, 2022.

Z. Li, M. Liu-Schiafini, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar. Learning chaotic dynamics in dissipative systems. Advances in Neural Information Processing Systems, 35:16768–16781, 2022.

Z. Liao and M. W. Mahoney. Hessian eigenspectra of more realistic nonlinear models. Advances in Neural Information Processing Systems, 34:20104–20117, 2021.

Z. Liao and M. W. Mahoney. Random matrix theory for deep learning: Beyond eigenvalues of linear models. Technical Report Preprint: arXiv:2506.13139, 2025.

P. Lippe, B. S. Veeling, P. Perdikaris, R. E. Turner, and J. Brandstetter. Pde-refiner: Achieving accurate long rollouts with neural pde solvers. arXiv preprint arXiv:2308.05732, 2023.

L. Lupin-Jimenez, M. Darman, S. Hazarika, T. Wu, M. Gray, R. He, A. Wong, and A. Chattopadhyay. Simultaneous emulation and downscaling with physically-consistent deep learning-based regional ocean emulators. arXiv preprint arXiv:2501.05058, 2025.

C. H. Martin and M. W. Mahoney. Traditional and heavy-tailed self regularization in neural network models. In Proceedings of the 36th International Conference on Machine Learning, pages 4284–4293, 2019.

C. H. Martin and M. W. Mahoney. Heavy-tailed Universality predicts trends in test accuracies for very large pre-trained deep neural networks. In Proceedings of the 20th SIAM International Conference on Data Mining, 2020.

C. H. Martin, T. Peng, and M. W. Mahoney. Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data. Nature Communications, 12(1):4122, 2021.

R. A. Meyer, C. Musco, C. Musco, and D. P. Woodruf. Hutch++: Optimal stochastic trace estimation. In Symposium on Simplicity in Algorithms (SOSA), pages 142–155. SIAM, 2021.

C. Molnar. Interpretable machine learning. Lulu. com, 2020.

V. Oommen, A. Bora, Z. Zhang, and G. E. Karniadakis. Integrating neural operators with difusion models improves spectral representation in turbulence modelling. Proceedings of the Royal Society A, 481(2309):20240819, 2025.

J. Pathak, S. Subramanian, P. Harrington, S. Raja, A. Chattopadhyay, M. Mardani, T. Kurth, D. Hall, Z. Li, K. Azizzadenesheli, et al. FourCastNet: A global data-driven high-resolution weather model using adaptive Fourier neural operators. arXiv preprint arXiv:2202.11214, 2022.

C. Pedersen, L. Zanna, and J. Bruna. Thermalizer: Stable autoregressive neural emulation of spatiotemporal chaos. arXiv preprint arXiv:2503.18731, 2025.

J. Pennington, S. Schoenholz, and S. Ganguli. The emergence of spectral universality in deep networks. In International Conference on Artificial Intelligence and Statistics, pages 1924–1932. PMLR, 2018.

I. Price, A. Sanchez-Gonzalez, F. Alet, T. R. Andersson, A. El-Kadi, D. Masters, T. Ewalds, J. Stott, S. Mohamed, P. Battaglia, et al. Probabilistic weather forecasting with machine learning. Nature, 637(8044):84–90, 2025.

M. Sakarvadia, K. Hegazy, A. Totounferoush, K. Chard, Y. Yang, I. Foster, and M. W. Mahoney. The false promise of zero-shot super-resolution in machine-learned operators. Technical Report Preprint: arXiv:2510.06646, 2025.

A. Sambamurthy and A. Chattopadhyay. Lazy difusion: Mitigating spectral collapse in generative difusion-based stable autoregressive emulation of turbulent flows. arXiv preprint arXiv:2512.09572, 2025.

K. Stachenfeld, D. B. Fielding, D. Kochkov, M. Cranmer, T. Pfaf, J. Godwin, C. Cui, S. Ho, P. Battaglia, and A. Sanchez-Gonzalez. Learned coarse models for eficient turbulence simulation. arXiv preprint arXiv:2112.15275, 2021.

Y. Wang and C.-Y. Lai. Multi-stage neural networks: Function approximator of machine precision. Journal of Computational Physics, 504:112865, 2024.

O. Watt-Meyer, B. Henn, J. McGibbon, S. K. Clark, A. Kwa, W. A. Perkins, E. Wu, L. Harris, and C. S. Bretherton. Ace2: accurately learning subseasonal to decadal atmospheric variability and forced responses. npj Climate and Atmospheric Science, 8(1):205, 2025.

A. Wikner, B. R. Hunt, J. Harvey, M. Girvan, and E. Ott. Stabilizing machine learning prediction of dynamics: Noise and noise-inspired regularization. arXiv preprint arXiv:2211.05262, 2022.

A. Yu, D. Lyu, S. H. Lim, M. W. Mahoney, and N. B. Erichson. Tuning frequency bias of state space models. arXiv preprint arXiv:2410.02035, 2024.

A. Yu, D. C. Maddix, B. Han, X. Zhang, A. F. Ansari, O. Shchur, C. Faloutsos, A. G. Wilson, M. W. Mahoney, and Y. Wang. Understanding the implicit biases of design choices for time series foundation models. Technical Report Preprint: arXiv:2510.19236, 2025.
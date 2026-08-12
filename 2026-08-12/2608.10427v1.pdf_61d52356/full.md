# Causality Sum Rules in Conventional Scattering Matrices

Ning Han<sup>1,2,3,†</sup>, Rui Zhao<sup>1,2,†</sup>, Shuxing Yang<sup>1,2,†</sup>, Mingzhu Li<sup>4</sup>, Hongsheng Chen<sup>1,2,5,\*</sup>, Yihao Yang<sup>1,2,5,\*</sup>

<sup>1</sup> Interdisciplinary Center for Quantum Information, State Key Laboratory of Extreme Photonics and Instrumentation, ZJU-Hangzhou Global Scientific and Technological Innovation Center, Zhejiang University, Hangzhou 310027, China

<sup>2</sup> International Joint Innovation Center, The Electromagnetics Academy at Zhejiang University, Zhejiang University, Haining 314400, China

<sup>3</sup> College of Optical and Electronic Technology, China Jiliang University, Hangzhou 310018, China

<sup>4</sup> School of Information and Electrical Engineering, Hangzhou City University, Hangzhou 310015, China

<sup>5</sup> Key Laboratory of Advanced Micro/Nano Electronic Devices & Smart Systems of Zhejiang, Jinhua Institute of Zhejiang University, Zhejiang University, Jinhua 321099, China

<sup>†</sup>These authors contributed equally to this work.

<sup>\*</sup>Corresponding author. Email:

yangyihao@zju.edu.cn (Y. Y.);

hansomchen@zju.edu.cn (H. C.)

## Abstract

Scattering matrices are the standard experimental and computational description of photonic and electromagnetic devices. Passivity is explicit in the conventional incoming-outgoing matrix, whereas causality sum rules are usually formulated only after transforming the response into auxiliary variables. Here we show that these rules can be written directly in the conventional scattering matrix by removing the time advance introduced by the reference domain. Using the earliest-arrival delay of each channel, we define a domain-delayed matrix that preserves real-frequency passivity while restoring the causal time origin. Under explicit analyticity, transparency, and regularity assumptions, this matrix becomes a Schur function, enabling a Cayley-Herglotz construction. The resulting projected and determinant bounds constrain coherent channel superpositions and aggregate multichannel loss. The framework recovers Rozanov’s absorber limit and spherical-multipole sum rules, while extending causality bounds to measurable quantities including insertion loss, suppressed singularvalue channels, and conditional lossless delay-bandwidth trade-offs. Our work directly connects fundamental causality theory with experimentally accessible scattering data. The initial theoretical route is autonomously explored by Qiushi Engine, an AI research system for open-ended scientific discovery, and subsequently verified, refined, and developed by the authors, demonstrating a hybrid AI-human discovery workflow.

## Introduction

Scattering matrices are the working language of photonic and electromagnetic devices. They are what network analyzers measure, what full-wave solvers return, and what designers use to describe absorbers, antennas, metasurfaces, photonic switches, interferometer meshes, and phased arrays. A conventional incoming-outgoing matrix S( ) ω at the operating frequency ω contains reflection, transmission, insertion loss, mode conversion, and coherent multiport interference in one experimentally accessible object. In this representation, passivity is almost immediate: for a power-normalized passive system, $S ^ { \dagger } ( \omega ) S ( \omega ) \preceq I$ on the real-frequency axis, where † is the conjugate transpose operator and I is the unit matrix. Causality is not so directly visible. It is a statement about the temporal order of response, and in frequency space it appears through analyticity, dispersion relations, and sum rules that connect all frequencies. Thus, the matrix most natural for experiments and simulations makes energy conservation simple, but does not by itself display the full causality constraint.

This separation has a long history. In the 1920s, Kramers and Kronig related causal response to analyticity in the complex-frequency plane<sup>1-3</sup>. Around the same period and into the 1950s, Foster’s reactance theorem and Bode-Fano matching theory showed how passivity and analyticity impose bandwidth limits on passive networks<sup>4-6</sup>. From the 1950s to the 1970s, bounded-real scattering matrices and S-matrix dispersion theory brought these ideas closer to network and scattering language<sup>3,7</sup>. Since the 2000s, electromagnetic sum rules have led to sharper physical limits, including Rozanov’s absorber bound, spherical-multipole scattering bounds, antenna bandwidth limits, and more recent Green-function, local-conservation, and volume-T-operator bounds<sup>8-23</sup>. These developments established a broad principle: passive causal systems carry finite budgets in bandwidth, size, depth, loss, and delay. However, the causal bound formulations are usually obtained in alternative representations, such as Green functions, local conservation formulations, and volume T-operator approaches<sup>21-23</sup>. The missing link is a sum-rule formulation that acts directly on the conventional multichannel scattering matrix S( )ω itself <sup>24</sup>.

Here we show that causality sum rules can be written directly in the conventional scattering matrix once the reference-domain time advance is removed, thereby establishing a direct connection between experimentally accessible scattering measurements and fundamental causality limits. Our approach is based on separating the geometry-dependent reference contribution from the physical scattering response through a domain-delay correction. This transformation converts the conventional scattering matrix into a causal analytic object compatible with Schur-Cayley theory.

From this construction, we derive two universal causality sum rules: a projected bound for coherent channel superpositions and a basis-independent determinant bound for aggregate multichannel attenuation. These results recover established scalar limits, including Rozanov’s absorber bound and spherical-multipole scattering bounds, while extending causality constraints to experimentally accessible multichannel quantities such as coherent suppression, insertion loss, and delay-bandwidth trade-offs. This connection enables causality constraints to be formulated directly in the experimentally accessible scattering representation.

Beyond the theoretical results, this work demonstrates a hybrid AI-human discovery workflow. Human researchers first defined the open-ended scientific question and research objective without prescribing a solution path. Qiushi Engine, an autonomous research system designed for sustained, thousand-step scientific reasoning<sup>25</sup>, then conducted the exploration, developed derivations and numerical checks, and generated a traceable record of data, code, figures, research notes, and reports. The system identified the core theoretical route and produced the initial result, which the authors subsequently verified, refined into a rigorous theory, and physically interpreted, while retaining full responsibility for scientific validation and oversight.

## Main Text

## Domain-delayed Schur construction

We now construct the causal scattering representation required for the sum-rule formulation. The conventional scattering matrix contains a geometry-induced temporal advance arising from the finite scattering domain and the choice of channel reference surfaces. This reference-dependent contribution appears as a phase factor a <sup>i T</sup> e<sup>−</sup> ω , where $T _ { a }$ is the earliest-arrival delay operator, and obscures the causal analytic structure required for direct application of Schur-Cayley theory. To remove this reference contribution, we introduce the earliest-arrival delay operator $T _ { _ a } = \mathrm { d i a g } ( \tau _ { \alpha } ) _ { \alpha } , \ \tau _ { \alpha } \geq 0$ where $\tau _ { \alpha }$ denotes the earliest arrival time of output channel $_ { \alpha . }$ For spherical-wave scattering of an object enclosed by a sphere of radius $^ { a , }$ all channels share the same delay, $\tau _ { \alpha } = 2 a / c$ . The corresponding domain-delay operator is defined as $D _ { a } = e ^ { i \omega T _ { a } }$ ， and the domain-delayed scattering matrix is given by

$$
\tilde { S } ( \omega ) = D _ { a } ( \omega ) S ( \omega ) .\tag{1}
$$

Because $D _ { a } ( \omega )$ is unitary on the real-frequency axis, $D _ { a } ^ { \dagger } ( \omega ) D _ { a } ( \omega ) = I$ , this transformation preserves the real-frequency power balance, $\tilde { S } ^ { \dagger } ( \omega ) \tilde { S } ( \omega ) = S ^ { \dagger } ( \omega ) S ( \omega )$ Therefore, the passivity condition $S ^ { \dagger } ( \omega ) S ( \omega ) \preceq I$ is unchanged by the domain-delay correction. The transformation only removes the reference-dependent temporal advance and restores the causal analytic structure of the scattering response, as shown in Figs. 1a-c.

More precisely, the corrected scattering matrix $\tilde { S } ( \omega )$ belongs to the class of operator Schur functions: it is analytic in the upper half-plane $\mathbb { C } ^ { + }$ and satisfies $\left. \tilde { S } ( \omega ) \right. _ { o p } \leq 1 , z \in \mathbb { C } ^ { + }$ . This is the central mathematical step of the construction. The construction assumes causal response, appropriate high-frequency transparency, and the regularity conditions required for the logarithmic Schur representation. The detailed proof is provided in Lemma S2.1 of the Supplementary Note 2. This result follows from the combination of causality, which provides the Hardy-class analytic continuation after the time-advance correction, and passivity, which preserves the contractive property on the real-frequency axis. The Schur property establishes the missing mathematical connection between experimentally accessible scattering data and causality-based sum rules. Applying the matrix Cayley transform, $W ( \omega ) = i [ 1 + \tilde { S } ( \omega ) ] [ 1 - \tilde { S } ( \omega ) ] ^ { - 1 }$ , maps the contractive scattering representation into an operator Herglotz function satisfying Im $W ( z ) \succeq 0 , z \in \mathbb { C } ^ { + }$ . The Herglotz representation then converts the analyticity and passivity constraints of the corrected scattering matrix into a spectral integral relation, whose low-frequency expansion is determined by the geometric delay operator $T _ { a }$ . This relation provides the universal mechanism for extracting causality sum rules from experimentally accessible scattering responses. Notably, the entire route is accomplished through the cooperation of a hybrid AI-human discovery workflow, and the specific workflow is shown in Fig. 1d.

![](images/f1d2e0ce89d3afdf0bddb3cd038bd673d8fbc0c7f9b0b3d1ec68123af0196deb.jpg)  
C

![](images/fe15449e46a973d8ee92913e4471aca78b4855fa38b1b13b0da927ee46492ec7.jpg)

d  
![](images/2c87f24f2943e99a0d4c717872fcba9c65f6930096cafe92d8c78af07b757e30.jpg)  
Fig. 1 | Physical principle and AI-human hybrid scientific discovery workflow. a, Finite reference domain defined by the scattering geometry and channel reference surfaces. b, Schur-Cayley-Herglotz machinery. c, Schematic of shifting the causal response to remove the time advance. A finite reference domain makes the conventional scattering matrix contain a channeldependent apparent time-advance phase, even though the system is passive on the real-frequency axis. Multiplication by the domain-delay factor $D _ { a } = e ^ { i \omega T _ { a } }$ removes this reference phase without changing real-frequency power balance. The time axis indicates the raw apparent start at $t = - \tau _ { \alpha }$ and the corrected causal start at t = 0 . Under the stated analyticity and transparency assumptions, the relative matrix ${ \tilde { S } } = D _ { a } S$ is the Schur object to which the Cayley-Herglotz and logarithmic sumrule machinery applies. d, Hybrid human-AI scientific discovery workflow. Human scientists define open-ended research questions, while the Qiushi Engine proposes research roadmaps, explores candidate solutions, and generates extensive data, notes, reports, and code scripts. Subsequently, human scientists review, evaluate, and extend the generated outputs, and provide feedback when necessary.

For a delay eigenvector v satisfying $T _ { a } \nu = \tau _ { \nu } \nu _ { : }$ , the scalar projection $\left. \nu , W ( \omega ) \nu \right.$ inherits the Herglotz property. Its low-frequency expansion contains the causal delay contribution $\tau _ { \nu } { : }$ , while the Herglotz integral representation relates this expansion coefficient to a weighted logarithmic integral of the scattering response. This yields the first projected causality sum rule.

## Projected sum rule

The first result establishes a coherent-return suppression limit for channel superpositions within a common-delay subspace. Let v be a unit incident superposition, and measure the component of the scattered field in the same superposition. The resulting amplitude is $S _ { \nu } ( \omega ) = \langle \nu , S ( \omega ) \nu \rangle$ . To remove the domain phase without changing this scalar projection, v must lie in an eigenspace of the delay operator, $T _ { a } \nu = \tau _ { \nu } \nu$ . Then $\tilde { s } _ { \nu } ( \omega ) = \langle \nu , \tilde { S } ( \omega ) \nu \rangle = e ^ { i \omega \tau _ { \nu } } \langle \nu , S ( \omega ) \nu \rangle$ . Consequently, $\mid \widetilde { S } _ { \nu } \mid = \mid s _ { \nu } \mid$ on the real axis. If v combines channels with different values of $\tau _ { \alpha }$ , the correction cannot be factored out as a single phase, and the scalar argument below does not apply. Thus, the positive Herglotz measure is set by the logarithmic suppression of the measured coherent return. Its first inverse-frequency moment is determined by the derivative of $h _ { \nu }$ at the origin. See details in Supplementary Note 3.

Expanding at low frequency separates the geometric delay from the zero and phaseslope terms, convergence of the logarithmic integral and Blaschke product, and the angular-derivative condition, which ensures that the non-geometric terms do not increase the low-frequency coefficient: $h \nu ^ { \prime } \leq \tau _ { \nu } = \left. \nu , T _ { a } \nu \right.$ . The Herglotz moment identity then yields

$$
\int _ { 0 } ^ { \infty } \frac { \ln \mid \langle \nu , S ( \omega ) \nu \rangle \mid } { \omega ^ { 2 } } \mathrm { d } \omega \Bigg \vert \leq \frac { \pi } { 2 } \big \langle \nu , T _ { a } \nu \big \rangle .\tag{2}
$$

Equation (2) limits the bandwidth-integrated logarithmic suppression of return into the selected channel superposition v by its domain-delay time $\tau _ { \nu }$ . Deep suppression over a broad band therefore requires a correspondingly large delay budget. For spherical waves, $T _ { a } = ( 2 a / c ) I$ and the result holds for every superposition in the truncated channel space. For unequal channel delays, it holds within each common-delay eigenspace.

## Determinant sum rule

The projected sum rule characterizes individual coherent scattering channels. To obtain a basis-independent constraint on the total multichannel response, we apply the same Schur-Cayley construction to the determinant of the domain-delayed scattering matrix det $\tilde { S } .$

For a finite-dimensional N-channel system, the determinant magnitude satisfies | det $S \left| = \prod \sigma _ { j } ( S ) \right.$ , where $\sigma _ { j } ( \boldsymbol { S } )$ are the singular values of S. Therefore, ln | det− $\stackrel { j = 1 } { S } \left| = \sum \right.$ ln − $\sigma _ { j }$ quantifies the aggregate logarithmic attenuation of the complete channel map. Unlike individual matrix elements or projected amplitudes, this quantity is invariant under unitary transformations of the input and output channel bases. Taking the determinant of the domain-delayed matrix gives det ( ) detS ω = <sup></sup> $D _ { a } ( \omega )$ det $S ( \omega ) = e ^ { i \omega \mathrm { t r } T _ { a } }$ det $S ( \omega )$ . Because $\tilde { S } ( \omega )$ is an operator Schur function, det $\tilde { S }$ is a scalar Schur function. Applying the determinant form of the logarithmic Herglotz representation under the corresponding convergence, lowfrequency unimodularity, and angular-derivative conditions yields the aggregated causality sum rule (See more details in Supplementary Note 4):

$$
\left| \int _ { 0 } ^ { \infty } \frac { \ln \mid \operatorname* { d e t } S ( \omega ) \mid } { \omega ^ { 2 } } \mathrm { d } \omega \right| { \leq } \frac { \pi } { 2 } \mathrm { t r } T _ { a } .\tag{3}
$$

The determinant bound provides a basis-independent constraint on the aggregate logarithmic attenuation of the multichannel scattering operator. While the projected bound describes suppression along a selected coherent channel, the determinant bound captures the collective contraction of all singular channels, including interchannel coupling and mode conversion. Therefore, it represents a genuinely multichannel causality constraint that cannot be obtained by independently applying scalar bounds to individual matrix elements.

## Physical consequences of the sum rules

The projected and determinant sum rules lead to three directly measurable consequences: a depth-bandwidth constraint for a selected coherent return, an aggregate attenuation bound for the singular-value spectrum, and a bound on the number of independently suppressed singular channels, as shown in Fig. 2.

![](images/ab2428af62ffd9d8c7bed00836ba0d85cebec7457cd61dd0b3a74031c906fffe.jpg)  
Fig. 2 | From the conventional scattering matrix to observable bounds. Removing the referencedomain advance gives the Schur matrix ${ \tilde { S } } = D _ { a } S$ . A projection constrains coherent return, whereas the determinant constrains aggregate singular-value attenuation and suppressed-channel count. The determinant-phase branch is a separate conditional extension for lossless systems.

## From full-frequency constraints to finite-band bounds

The operator-level sum rules constrain the frequency-integrated logarithmic attenuation over the entire spectrum. To connect these universal constraints with experimentally relevant performance metrics, we convert the full-frequency integrals into finite-band depth-bandwidth relations.

Consider a dimensionless scattering observable A( ) ω satisfying $0 < | \ A ( \omega ) | \leq A _ { 1 } < 1$ for $\omega \in [ \omega _ { 0 } , \omega _ { 0 } + B ]$ . The contribution of this frequency interval to the logarithmic integral is bounded by | ln A ${ \prod _ { \omega _ { 0 } } ^ { \omega _ { 0 } + B } \omega ^ { - 2 } d \omega }$ . For a narrow fractional bandwidth $\beta = B / \omega _ { \mathrm { 0 } } \ll 1$ , this contribution can be approximated as | ln $A _ { \mathrm { l } } \mid B / \omega _ { \mathrm { 0 } } ^ { 2 }$

Therefore, the full-frequency sum rules impose finite-band constraints on the achievable attenuation depth, bandwidth, and physical size. For the projected sum rule, the relevant quantity is the coherent return amplitude $S _ { \nu } \left( \omega \right) = \langle \nu , S ( \omega ) \nu \rangle$ . If the coherent response remains below $| s _ { \nu } ( \omega ) | \leq s _ { 0 }$ over a fractional bandwidth $\beta = B / \omega _ { 0 }$ , the integral constraint gives a depth-bandwidth limitation determined by the available delay budget $\langle \nu , T _ { a } \nu \rangle$

For the determinant sum rule, the corresponding quantity is the aggregate channel response | det ( ) | . S ω $\mathrm { I f } \ | \operatorname* { d e t } S ( \omega ) | \leq \Delta _ { 0 }$ within the operating band, the finite-band constraint becomes a limit on the total multichannel attenuation budget determined by tr $T _ { a }$ .

The narrowband approximation $\beta \ll 1$ is used for the expressions and figures below. The exact conversion between the full-frequency integral constraints and finite-band performance metrics, including finite-band corrections beyond the narrowband limit, is provided in Supplementary Note 5.

Consistency checks: scalar limits.

The two operator-level sum rules recover several established scalar causality bounds as limiting cases. These reductions demonstrate that the domain-delayed Schur-Cayley construction provides a unified scattering-matrix formulation of previously known fundamental limits.

First, consider a single-channel passive system, where the scattering matrix reduces to a scalar reflection coefficient, $S ( \omega ) = \Gamma ( \omega )$ . For a planar absorber of thickness d backed by a perfect conductor, the earliest round-trip arrival time is $\tau _ { \nu } = 2 d / c .$ Substituting this into the projected sum rule [see Eq. (2)], gives $\int _ { 0 } ^ { \infty } - \ln \left| \Gamma ( \omega ) \right| / \omega ^ { 2 } \mathrm { d } \omega \leq \pi \tau _ { \nu } / 2 = \pi d / c .$ . This directly recovers the Rozanov absorber bound<sup>8</sup>. In the narrowband approximation, where the reflection magnitude is approximately constant within a bandwidth B centered at $\omega _ { 0 }$ , the above inequality becomes $\left| \ln R _ { 0 } \right| B / \omega _ { 0 } \leq \pi d / \lambda _ { 0 }$ , where $R _ { 0 } = \mid \Gamma ( \omega _ { 0 } ) \mid , \ \lambda _ { 0 } = 2 \pi c / \omega _ { 0 }$ . Thus, the conventional thickness-bandwidth trade-off of absorbers emerges as a special case of the general projected sum rule.

A second recovery follows for spherical-wave scattering. Consider a passive scatterer enclosed by a sphere of radius a . For spherical multipole channels, the earliest arrival delay is identical for all channels, $T _ { a } = ( 2 a / c ) I$ . Therefore, every normalized channel superposition satisfies the delay-eigenvector condition, $T _ { a } \nu = ( 2 a / c ) \nu$ . Choosing a single spherical multipole basis vector, $\textstyle \nu = e _ { n } ^ { ( \sigma ) }$ , the projected sum rule reduces to $\int _ { 0 } ^ { \infty } - \ln \mid s _ { n } ^ { ( \sigma ) } ( \omega ) \mid / \omega ^ { 2 } { \mathrm { d } } \omega \leq \pi a / c .$ , which exactly recovers the Bernland-Gustafsson spherical-multipole scattering bound<sup>9,10</sup>. However, unlike the original diagonal formulation, the present result extends this constraint from individual multipole channels to arbitrary coherent superpositions within a common-delay eigenspace.

These recoveries confirm that the known scalar limits are embedded within the general S-matrix formulation. The remaining advantage of the operator-level theory is its ability to constrain collective scattering behavior beyond individual coefficients. We next explore these multichannel consequences, beginning with coherent return suppression in common-delay channel subspaces.

## Consequence 1: coherent-return suppression.

The projected sum rule establishes a fundamental limit on coherent return suppression for channel superpositions within a common-delay subspace, extending conventional bounds beyond individual diagonal scattering coefficients to coherent combinations of channels.

For a normalized channel superposition v, the coherent return amplitude is $s _ { \nu } = \langle \nu , S ( \omega ) \nu \rangle$ . When v is a delay eigenvector satisfying $T _ { a } \nu = \tau _ { \nu } \nu _ { : }$ , the projected sum rule gives $\left| \int _ { 0 } ^ { \infty } - \ln \left| s _ { \nu } ( \omega ) \right| / \omega ^ { 2 } \mathrm { d } \omega \right| \leq \pi \tau _ { \nu } / 2$ . which directly relates the achievable broadband coherent suppression to the available causal delay budget of the scattering structure.

To illustrate the physical meaning, consider spherical-wave scattering from an object enclosed by a sphere of radius a. In this case, $T _ { a } = ( 2 a / c ) I _ { \mathrm { : } }$ , so every coherent superposition of spherical channels satisfies the delay-eigenvector condition with $\tau _ { \nu } = 2 a / c$ . Therefore, the suppression of any coherent multipole superposition is fundamentally limited by the physical size of the scatterer:

$$
\left| \int _ { 0 } ^ { \infty } \frac { - \ln \left| \left. \nu , S ( \omega ) \nu \right. \right| } { \omega ^ { 2 } } \mathrm { d } \omega \right| \leq \pi a / c .\tag{4}
$$

This result shows that the same causal delay budget constrains not only individual multipole channels but also coherent combinations of channels.

For a coherent return satisfying $| s _ { \nu } ( \omega ) | \leq \rho _ { \nu }$ over a fractional bandwidth $\beta = B / \omega _ { 0 }$ ， define the coherent suppression depth as $D _ { \nu } = 2 0 \log _ { 1 0 } ( 1 / \rho _ { \nu } )$ , Using the narrowband approximation, the bound becomes $D _ { \nu } \leq ( 2 0 \pi / \ln 1 0 ) k _ { \scriptscriptstyle 0 } a / \beta .$

Figure 3a shows this depth-bandwidth-size boundary for different electrical sizes $k _ { 0 } a$ . The suppression depths below each curve satisfy the causal sum-rule constraint. For example, a 60-dB return suppression over a 10% fractional bandwidth corresponds to $\rho _ { _ { \nu } } = 1 0 ^ { - 3 }$ and requires $k _ { 0 } a \gtrsim 0 . 2 2$ . This is a necessary causality condition rather than a sufficient design criterion.

## Consequence 2: aggregate and geometric-mean attenuation.

The coherent-return suppression limit characterizes attenuation along a selected coherent channel superposition. By contrast, the determinant sum rule provides a complementary constraint on the collective attenuation of the full multichannel scattering operator.

For an N-channel scattering system, | det $S \left| = \prod _ { . } ^ { N } \sigma _ { { } _ { j } } [ S ( \omega ) ] \right.$ , where $\sigma _ { j } ( \boldsymbol { S } )$ are the singular values of the scattering matrix. Therefore, ln | det− $S \left| = \sum - \ln \sigma _ { j } \right.$ , measures the aggregate logarithmic attenuation accumulated over all singular scattering channels.

Unlike individual matrix elements or projected coherent amplitudes, the determinant is invariant under unitary transformations of the input and output channel bases. Therefore, it provides a basis-independent measure of the collective contraction of the scattering operator.

The determinant constrains aggregate logarithmic attenuation across the included singular channels, rather than the attenuation of an individual multipole or matrix element. This quantity can be interpreted as absorption only when the retained scattering channels contain all power-carrying output channels. Otherwise, a reduced determinant magnitude may also arise from radiation or coupling into omitted channels.

a  
![](images/f8c531a2f5c7922ec9470ebb9c9d20a479e2f50459a1f67d54dbbd8b216d74b0.jpg)

b  
![](images/a2f54a18ff5aa6ea2f47379f90f07d90cba329893dc9bb13841b05e8b047b38b.jpg)

![](images/ffbd2a0a83d73513e2d2adbe9e7abc3740994294d9c90067ca5d37df46a6d7f3.jpg)  
Fig. 3 | Finite-band consequences of the projected and determinant sum rules. a, Maximum coherent-return depth $D _ { \nu } = 2 0 \log _ { 1 0 } \left( 1 / \rho _ { \nu } \right)$ for a prescribed fractional bandwidth and electrical size. b, Upper bound on the geometric-mean singular-value depth after normalizing the determinant result by the number of channels; this upper bound is independent of N. c, Maximum fraction $m _ { \operatorname* { m a x } } / N$ of singular channels that can remain below the prescribed threshold $\rho _ { \sigma }$ across the operating bandwidth.

For a spherical-wave basis truncated at multipole order $n = N _ { \operatorname* { m a x } }$ , accounting for the $2 n + 1$ azimuthal orders and two polarizations for each multipole order, the finite truncation contains $N = 2 { N _ { \mathrm { m a x } } } ( { N _ { \mathrm { m a x } } + 2 } )$ open scattering channels. For a spherical reference domain, $\begin{array} { r } { T _ { a } = \left( 2 a / c \right) I _ { \mathrm { ~ \normalfont ~ * ~ } } } \end{array}$ , and therefore tr $T a = 2 a N / c$ . Applying the determinant sum rule gives the aggregate attenuation bound

$$
\int _ { 0 } ^ { \infty } \frac { - \ln \mid \operatorname* { d e t } S ( \omega ) \mid } { \omega ^ { 2 } } \mathrm { d } \omega \leq \frac { N \pi a } { c } .\tag{5}
$$

To obtain a finite-band form, define the determinant modulus within an operating band as $\begin{array} { r l } { \Delta _ { 0 } = } & { { } \ \operatorname* { s u p } \quad | \operatorname* { d e t } S ( \omega ) | } \end{array}$ . Using the narrowband approximation, the <sub>0 0</sub>[ , ]B ∈ + determinant constraint becomes $\ln \Delta _ { 0 } \big | B / \omega _ { 0 } \pi \leq N k _ { 0 } a$ . Equivalently, defining the aggregate determinant suppression depth $D _ { \mathrm { d e t } } = 2 0 \log _ { 1 0 } ( 1 / \Delta _ { 0 } )$ , we obtain $D _ { \mathrm { d e t } } B / \omega _ { 0 } \leq 2 7 . 3 N k _ { 0 } a$ (See more details in Supplementary Note 6).

The total determinant budget grows linearly with the number of channels N, but the number of singular values sharing this budget also grows linearly. Therefore, after normalization by the channel number, the geometric-mean attenuation exhibits an $N -$ independent causal limit. Define the geometric-mean singular value $\overline { { \sigma } } _ { g } = \rvert \operatorname* { d e t } S \rvert ^ { 1 / N }$ and its depth $\overline { { D } } _ { g } = - 2 0 \log _ { 1 0 } \overline { { \sigma } } _ { g }$ . Dividing the determinant constraint by N gives an upper bound on the geometric-mean singular-value depth $\overline { { D } } _ { g } B / \omega _ { 0 } \leq ( 2 0 \pi / \ln 1 0 ) k _ { \scriptscriptstyle 0 } a$ , which is independent of the number of channels, as shown in Fig. 3b. This is an average constraint and does not require equal attenuation across all singular channels; different channels may experience different suppression levels while the geometric mean remains bounded by the available causal delay budget. For N = 70 , a determinant depth of 80 dB corresponds to a geometric-mean depth of $8 0 / 7 0 \simeq 1 . 1$ dB. This does not imply equal suppression of every singular channel.

## Consequence 3: suppressed-channel count.

The geometric-mean attenuation bound constrains the average suppression of the singular-value spectrum, but it does not determine how many independent scattering channels can simultaneously achieve a prescribed attenuation level. We therefore derive a bound on the number of suppressed singular channels.

Choose a suppression threshold $0 < \rho _ { \sigma } < 1$ . If at least m singular values satisfy $\sigma _ { { } _ { j } } ( S ) \le \rho _ { { } _ { \sigma } }$ throughout an operating band, then passivity gives | det $S \left. \leq \rho _ { \sigma } ^ { m } \right.$ Combining this inequality with the determinant sum rule and the finite-band approximation yields $m \leq \pi N k _ { 0 } a / ( \left| \ln \rho _ { 0 } \right| B / \omega _ { 0 } )$ , together with $m \leq N$ , this gives the maximum number of singular channels that can simultaneously maintain a target suppression level over a finite bandwidth (See more details in Supplementary Note 7). Here m counts orthogonal incident singular vectors, not physical ports. Figure 3c plots the maximum fraction m N/ for $\rho _ { \sigma } = 0 . 1$

When the scattering matrix contains all power-carrying exterior output channels, the singular values can be related to absorption eigenchannels through $A _ { j } = 1 - \sigma _ { j } ^ { 2 }$ Requiring at least m eigenchannels to satisfy $A _ { j } \geq A _ { 0 }$ throughout the operating band is equivalent to $\rho _ { \sigma } = \sqrt { 1 - A _ { 0 } }$ and gives $m \leq 2 \pi N k _ { 0 } a / \left( \left| \ln ( 1 - A _ { 0 } \right| B / \omega _ { 0 } ) \right.$ . For a channelcomplete $N = 3 0$ model, maintaining $A _ { j } \geq 0 . 9 9$ in all 30 eigenchannels over a 10% fractional bandwidth requires $k _ { 0 } a \ge 7 . 3 \times 1 0 ^ { - 2 }$ . This is a necessary condition from the determinant budget, rather than a sufficient condition for constructing such a device.

If the retained scattering channels do not include all power-carrying output channels, a small singular value indicates only that power has left the measured subspace. It does not distinguish absorption from radiation or coupling into unmeasured channels. Therefore, the absorption interpretation requires a channel-complete scattering description or a closed reducing subspace with no coupling to omitted channels.

## Conditional phase-delay extension for lossless multiport systems

The preceding consequences follow from the projected and determinant logarithmic sum rules and constrain attenuation-related observables. Lossless systems represent a distinct regime: because $| \operatorname* { d e t } S | = 1$ , on the real-frequency axis, the logarithmic attenuation bounds become trivial. In this case, the relevant causal quantity is the accumulated scattering phase rather than the modulus.

For a lossless multiport system, S is unitary and $\left| \operatorname* { d e t } S \right| = 1$ , the determinant can be expressed as det $S ( \omega ) = e ^ { i \Theta ( \omega ) }$ . The Wigner-Smith time-delay operator is defined as $Q ( \omega ) = - i S ^ { \dagger } \partial _ { \omega } S$ , which gives the phase-delay identity $\Theta ^ { \prime } ( \omega ) = \mathrm { t r } Q ( \omega )$ . Therefore, tr ( )Q ω represents the total delay accumulated over all scattering channels.

Define the band-averaged delay trace as $\overline { { \mathrm { t r } \mathcal { Q } } } = \frac { 1 } { B } \int _ { \omega _ { 0 } } ^ { \omega _ { 0 } + B } \mathrm { t r } \mathcal { Q } ( \omega ) \mathrm { d } \omega ,$ Unlike the attenuation bounds, obtaining a universal delay-bandwidth estimate requires an additional spectral-counting assumption. Let  denote the effective propagation length entering the modal-count model, and $\lambda _ { 0 } = 2 \pi c / \omega _ { 0 }$ . If the delay-corrected determinant contains no singular inner factor and its Blaschke phase accumulation satisfies the modal-count hypothesis $H _ { B }$ (see details in Supplementary Note 8), then

$$
\overline { { { \mathrm { t r } } Q } } B \leq 4 \pi ^ { 2 } N \frac { \ell } { \lambda _ { 0 } } .\tag{6}
$$

By using the fractional bandwidth $\beta = B / \omega _ { 0 }$ , the delay-bandwidth relation can be written as $\overline { { \operatorname { t r } \mathscr { Q } } } \omega _ { 0 } \leq 4 \pi ^ { 2 } N ( \ell / \lambda _ { 0 } ) / \beta .$ For the mean delay per channel $q = \overline { { \operatorname { t r } Q } } / { \cal N }$ , one obtains the basis-independent form $q B \leq 4 \pi ^ { 2 } \ell / \lambda _ { 0 }$ (See more details in Supplementary Notes 8 and 9).

For example, consider a target mean normalized delay of $q \omega _ { 0 } = 1 0 ^ { 6 }$ over a fractional bandwidth 1%. The required delay-bandwidth product is then $q B = 1 0 ^ { 4 }$ . Under the modal-count hypothesis, this requires $\ell / \lambda _ { 0 } \gtrsim 2 . 5 { \times } 1 0 ^ { 2 }$ . Thus, achieving extremely large delay over finite bandwidth necessarily requires a correspondingly large effective propagation length. For an N-channel implementation, the aggregate delay trace scales as $\mathrm { t r } Q = N q$

Figure 4a illustrates the linear scaling of the aggregate delay budget with the number of channels N at fixed $\ell / \lambda _ { 0 }$ , while Fig. 4b shows the reciprocal trade-off between delay and fractional bandwidth. These relations are relevant to delay lines, beamforming networks, and interferometric photonic architectures.

a  
![](images/6f7e020514c76aceff42e1bf3c9df6e0a78dfe0b531a1bfbadbe19bc95a9606e.jpg)

b  
![](images/cad568a1290c421004fdeae4aac9cf893a279d323923c5aa50c4e4fa3e2847cf.jpg)  
Fig. 4 | Conditional delay-bandwidth estimates. a, Aggregate delay-bandwidth envelope at $\ell / \lambda _ { 0 } = 1 0 0$ and illustrative network scales. b, Equivalent bound on $\mathrm { t r } Q \omega _ { \mathrm { 0 } }$ as a function of fractional bandwidth. Additional high-Q phase winding is not included.

Importantly, the delay-bandwidth estimate relies on the modal-count hypothesis $H _ { B }$ Sharp resonances can introduce additional Blaschke phase accumulation beyond the assumed spectral count. An additional phase contribution $k _ { \mathrm { e x t r a } }$ modifies the bound by an additive term $2 \pi k _ { \mathrm { e x t r a } }$ in the integrated trace delay. Therefore, Eq. (6) should be interpreted as conditional spectral-counting estimates rather than universal limits applicable to arbitrary high-Q resonant or slow-light systems.

On the use of Qiushi Engine. Because there is growing interest in whether autonomous AI systems can contribute to real scientific discovery, we state explicitly how Qiushi Engine entered this work. The path to the present theory also illustrates a new mode of scientific discovery. Qiushi Engine participated as an autonomous exploration partner within a hybrid human-AI discovery workflow. Given the open problem of formulating causality sum rules directly in the conventional scattering matrix, the system explored candidate mathematical routes, identified the domain-delay correction strategy, generated preliminary derivations, and organized intermediate research artifacts. These outputs were not treated as proofs. The authors subsequently audited the analytic assumptions, corrected the functional-analytic formulation, verified scalar recoveries, examined numerical consequences, and established the final physical interpretation. This division of roles demonstrates a research paradigm in which AI expands the space of candidate scientific pathways, while human researchers retain rigorous validation and scientific responsibility. Supplementary Note 10 records this workflow.

## Discussion

The domain-delayed Schur-Cayley-Herglotz construction provides a direct route to formulating causality sum rules in the conventional scattering matrix. Removing the apparent time advance introduced by the finite reference domain restores the analytic structure required for the sum-rule construction. This leads to two operator-level results: a projected sum rule for coherent channel superpositions and a basis-independent determinant sum rule for aggregate multichannel attenuation. The framework recovers established scalar limits, including Rozanov’s absorber bound and spherical-multipole scattering sum rules, while extending them to measurable multichannel quantities such as coherent suppression, determinant insertion loss, singular-channel suppression, and lossless delay.

Working in the conventional scattering representation keeps these bounds directly connected to the input-output data obtained from experiments, full-wave simulations, and inverse design. The framework complements approaches based on polarizabilities, Green functions, and volume (T)-operators<sup>21-23</sup>, which express causality through material or source variables, by providing an equivalent formulation at the scattering level. The projected rule constrains a selected coherent channel superposition, whereas the determinant rule captures the collective attenuation of the full multichannel response.

More broadly, this work connects fundamental causality theory with the scattering data routinely used in photonics and electromagnetism. Qiushi Engine contributed to the initial exploration of the theoretical route, including candidate formulations, preliminary derivations, numerical checks, and research records. The authors then verified the assumptions, completed the derivation, and established the final physical interpretation. This workflow provides a concrete example of how autonomous research systems can support theoretical discovery while rigorous validation and scientific judgement remain with the researchers.

## Methods

## Scattering convention and channel normalization

We use the time dependence exp( ) −i t ω . Incoming and outgoing channel amplitudes are collected in vectors $a ( \omega )$ and $b ( \omega )$ and related by

$$
b ( \omega ) = S ( \omega ) a ( \omega ) .
$$

The channels are power normalized, so that $\left\| a \right\| ^ { 2 }$ and $\left\| b \right\| ^ { 2 }$ represent incident and outgoing powers. Passivity therefore gives

$$
S ^ { \dagger } ( \omega ) S ( \omega ) \preceq I ,
$$

on the real-frequency axis. Determinants are taken only after restricting the channel map to a finite set of propagating channels. When attenuation is interpreted as absorption, this set must include all propagating output channels into which the chosen inputs can scatter. Otherwise, the missing power may represent radiation or mode conversion into omitted channels rather than dissipation.

## Domain-delay correction

The conventional scattering matrix depends on the reference surfaces and on the finite domain used to define the incoming and outgoing channels. For output channel $\alpha$ , let $\tau _ { \alpha } \geq 0$ denote the magnitude of the earliest apparent time advance introduced by this convention. The correction times are assembled into

$$
\begin{array} { r } { T _ { a } = \mathrm { d i a g } ( \tau _ { \alpha } ) _ { \alpha } , } \end{array}
$$

and the domain-delay operator is

$$
D _ { a } ( \omega ) = \exp ( i \omega T _ { a } ) = \mathrm { d i a g } \big ( \exp ( i \omega \tau _ { \alpha } ) \big ) _ { \alpha } .
$$

We then define the corrected scattering matrix

$$
\tilde { S } ( \omega ) = D _ { a } ( \omega ) S ( \omega ) .
$$

Because $D _ { a }$ is unitary for real $\omega$ , this transformation preserves the real-frequency power balance. In the time domain, if the uncorrected impulse response $s _ { \alpha \beta } ( t )$ can begin at $t = - \tau _ { \alpha }$ , multiplication by $\exp ( i \omega \tau _ { \alpha } )$ gives

$$
\widetilde { s } _ { \alpha \beta } ( t ) = s _ { \alpha \beta } ( t - \tau _ { \alpha } ) ,
$$

whose support begins at $t = 0$ . Thus, $\tau _ { \alpha }$ is a reference-domain correction time. It is not a resonance lifetime, a Wigner-Smith group delay, or the delay of an output-pulse maximum. For a spherical reference surface of radius $^ { a }$ , the open spherical-wave channels have $\tau _ { \alpha } = 2 a / c$

## Acknowledgments

The work at Zhejiang University sponsored by the Key Research and Development Program of the Ministry of Science and Technology under Grants No. 2022YFA1405200 (Y.Y.), No. 2022YFA1404900 (Y.Y.), and No 2022YFA1404704 (H.C.), the Fundamental Research Funds for the Zhejiang Provincial Universities No. 226-2025-00231 (Y.Y.), the Science Challenge Project No. TZ2025015 (Y.Y.), the National Natural Science Foundation of China No. U25D8017 (Y.Y.), the National Natural Science Foundation of China (NSFC) under grant No. 61975176 (H.C.), the Key Research and development Program of Zhejiang Province under grant no. 2022C01036 (H.C.), the National Natural Science Foundation of China (NSFC) under Grant No. 12504497 (N.H.), the National Natural Science Foundation of China (NSFC) under Grant No. 12304472 (M.L.), the Baima Lake Laboratory Joint Fund of the Zhejiang Provincial Natural Science Foundation of China under Grants No. LBMHY25A040002 (M.L.), the Hangzhou Natural Science Foundation under Grants No. 2024SZRYBF050004 (M.L.), and the Natural Science Foundation of Zhejiang Province under Grant No. LQN26A040006 (N.H.).

## Author contributions

Y.Y. conceived the project; N.H., R.Z., S.Y. and M.L. verified the assumptions, limiting cases and numerical examples; H.C. and Y.Y. supervised the project; N.H. and R.Z. wrote the manuscript with input from S.Y., M.L., H.C. and Y.Y. All authors discussed the results and reviewed the manuscript.

## Competing interests

The authors declare no competing interests.

## Data and materials availability

All data that support the findings of this study are present in the paper and the Supplementary Information. Any additional information may be available from the corresponding authors upon request.

## References

1. Kramers, H. A. La diffusion de la lumière par les atomes. Atti Congr. Internazionale Dei Fis. 2, 545-557 (1927).

2. Kronig, R. de L. On the theory of dispersion of X-rays. J. Opt. Soc. Am. 12, 547- 557 (1926).

3. Nussenzveig, H. M. Causality and Dispersion Relations. (Academic Press, New York, 1972).

4. Foster, R. M. A reactance theorem. Bell Syst. Tech. J. 3, 259-267 (1924).

5. Bode, H. W. Network Analysis and Feedback Amplifier Design. (Van Nostrand, New York, 1945).

6. Fano, R. M. Theoretical limitations on the broadband matching of arbitrary impedances. J. Frankl. Inst. 249, 57-83 (1950).

7. Youla, D. C., Castriota, L. J. & Carlin, H. J. Bounded real scattering matrices and the foundations of linear passive network theory. IRE Trans. Circuit Theory 6, 102- 124 (1959).

8. Rozanov, K. N. Ultimate thickness to bandwidth ratio of radar absorbers. IEEE Trans. Antennas Propag. 48, 1230-1234 (2000).

9. Bernland, A., Gustafsson, M. & Nordebo, S. Physical limitations on the scattering of electromagnetic vector spherical waves. J. Phys. Math. Theor. 44, 145401 (2011).

10. Bernland, A. Bandwidth limitations for scattering of higher order electromagnetic spherical waves with implications for the antenna scattering matrix. IEEE Trans. Antennas Propag. 60, 4345-4353 (2012).

11. Bernland, A., Luger, A. & Gustafsson, M. Sum rules and constraints on passive systems. J. Phys. Math. Theor. 44, 145205 (2011).

12. Sohl, C., Gustafsson, M. & Kristensson, G. Physical limitations on broadband scattering by heterogeneous obstacles. J. Phys. Math. Theor. 40, 11165-11182 (2007).

13. Sohl, C., Larsson, C., Gustafsson, M. & Kristensson, G. A scattering and absorption identity for metamaterials: experimental results and comparison with theory. J. Appl. Phys. 103, 054906 (2008).

14. Gustafsson, M., Sohl, C., Larsson, C. & Sjöberg, D. Physical bounds on the allspectrum transmission through periodic arrays. EPL 87, 34002 (2009).

15. Biehs, S.-A., Rousseau, E. & Greffet, J.-J. Mesoscopic description of radiative heat transfer at the nanoscale. Phys. Rev. Lett. 105, 234301 (2010).

16. Gustafsson, M., Vakili, I., Keskin, S. E. B., Sjöberg, D. & Larsson, C. Optical theorem and forward scattering sum rule for periodic structures. IEEE Trans. Antennas Propag. 60, 3818-3826 (2012).

17. Miller, O. D., Johnson, S. G. & Rodriguez, A. W. Shape-independent limits to nearfield radiative heat transfer. Phys. Rev. Lett. 115, 204302 (2015).

18. Shim, H., Fan, L., Johnson, S. G. & Miller, O. D. Fundamental limits to near-field optical response over any bandwidth. Phys. Rev. X 9, 011043 (2019).

19. Molesky, S., Jin, W., Venkataram, P. S. & Rodriguez, A. W. T-operator bounds on angle-integrated absorption and thermal radiation for arbitrary objects. Phys. Rev. Lett. 123, 257401 (2019).

20. Kuang, Z. & Miller, O. D. Computational bounds to light–matter interactions via local conservation laws. Phys. Rev. Lett. 125, 263607 (2020).

21. Molesky, S., Chao, P., Jin, W. & Rodriguez, A. W. Global T operator bounds on electromagnetic scattering: upper bounds on far-field cross sections. Phys. Rev. Res. 2, 033172 (2020).

22. Molesky, S., Chao, P., Mohajan, J., Reinhart, W., Chi, H. & Rodriguez, A. W. Toperator limits on optical communication: metaoptics, computation, and inputoutput transformations. Phys. Rev. Res. 4, 013020 (2022).

23. Zhang, L., Monticone, F. & Miller, O. D. All electromagnetic scattering bodies are matrix-valued oscillators. Nat. Commun. 14, 7724 (2023).

24. O. D. Miller and F. Monticone, Fundamental limits in photonics and electromagnetics: a tutorial, arXiv:2605.24738, 2026.

25. Yang, S., et al. End-to-end autonomous scientific discovery on a real optical platform, arXiv:2604.27092 (2026).
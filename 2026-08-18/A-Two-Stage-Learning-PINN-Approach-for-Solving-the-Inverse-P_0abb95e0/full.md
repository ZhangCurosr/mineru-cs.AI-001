# A Two-Stage Learning PINN Approach for Solving the Inverse Problem of the 1D Porous Medium Equation

Noura Al Helwani \* Sophie Moufawad † Nabil Nassif †

August 18, 2026

## Abstract

The Porous Medium Equation (PME), given by $u _ { t } = \Delta ( u ^ { m } )$ for m > 1, is a degenerate nonlinear parabolic partial differential equation that arises in various physical applications such as fluid flow in porous media, heat transfer in plasmas, and population dynamics. It is known for its nonlinear diffusion and finite propagation speed. In this paper, we study numerical solutions of the one-dimensional direct and inverse PME using Physics-Informed Neural Networks (PINNs), and compare them with classical numerical methods and available analytical and manufactured solutions. While PINNs provide a flexible framework for solving both forward and inverse problems we show that the standard inverse formulation suffers from a strong sensitivity to the initial guess, leading to only local convergence. To address this issue, we propose a novel two-stage PINN training framework for the inverse problem, which significantly improves convergence stability and allows reliable recovery of the unknown parameter even for poor initial guesses. Overall, the proposed approach demonstrates that PINNs are a flexible and accurate alternative to classical methods for the 1D PME, and the introduced two-stage training strategy substantially improves their robustness in inverse problems, providing a solid basis for extensions to more complex geometries and higherdimensional cases.

Keywords: Porous medium equation, Physics-informed neural networks, Inverse problems, Deep learning, Nonlinear partial differential equations, Degenerate parabolic equations, Numerical analysis, Scientific machine learning, Applied mathematics.

## Contents

1 Introduction 2   
2 The Direct Problem of the 1D PME 5   
2.1 Methodologies . 6   
2.1.1 Classical Numerical Scheme 6   
2.1.2 PINN Approach . 7   
2.2 Reference Solutions . 10   
2.2.1 Barenblatt Solution 10   
2.2.2 Manufactured Solution 10   
2.3 Direct PINN Testing without Data 11   
2.4 Direct PINN Testing with Data 16   
3 The 2-stage Learning Inverse PINN 17   
3.1 Classical Approach: fmincon. 17   
3.2 PINN Approach . 18   
3.3 Inverse PINN Design Experiments . 19   
3.3.1 The Proposed Two-Stage Training Inverse PINN Model . 21   
3.4 Testings and results . . 23   
3.4.1 Inverse PINN testing without Noisy Data. 24   
3.4.2 Inverse PINN testing with Noisy Data 32   
4 Conclusion 35   
References 35   
A Visual Comparisons of Direct Solutions 40   
A.1 Barenblatt solution 40   
A.2 Polynomial solution . 44   
A.3 Damped Harmonic Solution 46   
A.4 m-dependent solution . 49   
B Inverse PINN's training loss of the m-dependent solution 51   
C Inverse PINN results before the 2-stage PINN model 54

## 1 Introduction

Inverse problems governed by nonlinear partial differential equations (PDEs) arise in many scientific and engineering applications [1, 5, 3], where unknown parameters or coefficients must be inferred from observed data. In this paper, we solve the inverse Porous Medium Equation (PME) Eq. 1 problem using Physics-Informed Neural Networks (PINNs), which is a mesh-free framework for solving such problems through the incorporation of governing physical laws into the learning. PINNs have several important advantages compared to classical numerical methods. They are mesh-free, which makes them easier and more flexible to extend to higher-dimensional problems, since classical approaches usually require new and more complex discretization schemes and more advanced and expensive numerical methods when moving from 1D to higher dimensions. In addition, inverse problems in classical frameworks are generally more expensive and sometimes ill-conditioned than direct problems, while PINNs can handle inverse problems in a more efficient way within the same framework. Finally, PINNs can also work with sparse data in inverse problems, without the need for full-field or dense measurements. These advantages, along with the growing development of scientific machine learning methods, were the main motivation behind using PINNs as the main approach in this paper.

An elementary attempt to solve the 1D PME using PINNs only for m = 3 was done in [2]. In this paper, we aim to improve the results produced by [2] by improving the PINN model and focusing on the inverse problem. We also generalize the solution for any m, and we manufactured three different solutions to further verify the work and test its robustness. For the inverse problem, we propose a novel two-stage learning approach that achieves convergence even when the initial guess is far from the true solution, which solves the problem of local convergence when only standard PINN techniques were involved. Throughout, we compare the performance of PINNs to traditional numerical solvers in terms of accuracy, stability, and convergence, with the broader aim of understanding the strengths and limitations of PINNs in solving nonlinear diffusion equations like the PME. These classical numerical methods are used in this work as a validation benchmark rather than a competing approach, since the goal is not to outperform them in all cases, but to develop a robust and reliable PINN framework that remains stable and accurate across different settings.

To elaborate, the motivation behind writing this paper and doing this work is to build strong foundations where the 1D PME PINN performs very well, in order to prepare for the main topic and objective: the 2D and higher-dimensional PME in more complex geometries, where classical methods normally struggle, and to explore the advantages of PINNs. The 1D conditions presented in this paper represent the setting where classical methods usually excel. Our aim is to develop a robust PINN that can be as accurate as feasible compared to the classical method in these basic geometries, in order to ensure robustness before moving to higher and more complex geometries, where we expect PINNs to become more advantageous. In particular, establishing a reliable inverse PINN framework in 1D is a necessary step before moving to higher-dimensional cases, because unlike classical numerical methods, which usually need new discretization techniques, new mesh generation, and sometimes even different solvers when going to higher dimensions, PINNs keep the same main structure. Only small changes are needed mainly in the input dimension and the way collocation points are sampled. This is why the paper is divided into two sections: the first section, Section 2, focuses on the direct PME problem, while the second, Section 3 which represents the main contribution of this work, focuses on the inverse problem In brief, the main objectives of this work are:

1. To develop a robust PINN framework for the inverse PME problem in 1D

2. To establish a strong and reliable foundation for extending the methodology to higher-dimensional problems.

Summarizing the main results of this paper, for both the direct and inverse problems, the PINN framework was found to be accurate, with most errors below 5%. It is also more flexible than the classical methods, although it comes at a higher computational cost. Overall, PINNs prove to be a viable and flexible alternative to classical methods for the 1D PME, with clear room for further improvement, and give a solid foundation for extending the framework to higher-dimensional geometries.

The main contributions of this work are summarized as follows:

• We develop an improved PINN framework for the 1D Porous Medium Equation covering both direct and inverse problems.

• We investigate the effect of initialization, training strategy, stability techniques, and loss scheduling on inverse PINN performance (Section 3.3).

• We identify the local convergence limitation of standard inverse PINNs for the PME, where accurate parameter recovery strongly depends on the initial guess (Section 3.2).

• We propose a novel two-stage training PINN framework for the inverse PME problem that improves convergence robustness and stabilizes parameter estimation across different initial guesses (Section 3.3.1).

• We compare the proposed framework against classical numerical inverse solvers across several manufactured solutions and values of m.

• We establish a robust and well-behaved inverse PINN framework for the 1D PME, which provides a solid foundation for extending the methodology to higher-dimensional and more complex geometries.

The proposed two-stage learning strategy is not restricted to the PME and can be naturally extended to other PDE-based inverse problems, where the number of stages and their structure can be adapted depending on the specific problem, the data availability, and the convergence behavior, potentially leading to more general multi-stage learning frameworks tailored to each application.

To better situate this work, we now briefly review relevant literature on machine learning and Physics-Informed Neural Networks for solving PDEs and inverse problems. Machine learning (ML) has become increasingly prominent with the data revolution in the digital age, with huge success in applications such as image classification [9] and natural language processing [10]. These successes are usually driven by large, high-quality data sets. However, a lot of applied science is in the small-data regime with large uncertainty. In such cases, ML-based methods, e.g., PINNs, are particularly valuable [11].

Neural networks in particular possess several properties that make them important for scientific computing. They are universal function approximators [12 and benefit from automatic differentiation, which makes calculating the derivatives that occur in differential equation-solving more straightforward [14]. The use of neural networks to approximate ordinary and partial differential equations started in the 1990s [15, 16, 17] and has continued as an active area of research with deep learning techniques. Early approaches blended neural networks with traditional methods, e.g., weighted residuals, constrained backpropagation, and Galerkin methods, for the solution of initial and boundary value problems [18]. These developments naturally led to physics-informed formulations. More recently, Raissi et al. [11, 8] proposed Physics-Informed Neural Networks, which directly incorporate physical laws into the training procedure. It is a mesh-free approach, avoids interpolation errors, and offers a unified treatment of both forward and inverse problems. PINNs have been applied to diverse areas such as heat transfer [19, 20 and manufacturing [21]. Another advantage is the availability of several Python libraries [13, 22, 23] that facilitate the implementation of PINNs. Most notably, PINNs provide a unified framework for both direct (forward) and inverse problems. These advantages explain the rapid adoption and exploration of PINNs in scientific applications

PINNs do come with some issues, though. Their loss landscapes may be very non-convex, which can induce optimization difficulties and decreased accuracy. More sophisticated physics typically require deeper networks, which increase the computation cost, memory requirements, and training time. Also,

PINNs are confronted with PDEs with parameter discontinuities or domain heterogeneities, e.g., conjugate heat transfer in heterogeneous media. To address these limitations, several enhanced PINN variants have been developed. Domain decomposition techniques have been proposed to overcome this difficulty [24, 25]. These divide the computational domain into subdomains, for which local PINNs are separately defined and coupled together with interface conditions. Adaptive activation functions [26] and even variational principles have been examined to improve accuracy. In order to further improve performance, Transfer Physics-Informed Neural Networks (TPINNs) were introduced. Unlike training separate neural networks for each of the subdomains, TPINNs transfer layers between subdomains but allow certain layers to learn to adjust to local physics. Besides increasing efficiency, it also shows some effectiveness in solving stiff differential equations [27, 28]. Beyond these improvements, several other neural network-based approaches have been proposed to handle complex scientific computing problems where conventional numerical methods struggle, especially in highly nonlinear or high-dimensional systems. Examples include geometry-aware PINNs for more complex domains [30], specific activation functions for high-dimensional nonlinear wave equations [31|, and convolutional PINNs for irregular geometries [32].The demand for inverse PINNs is especially urgent in porous media simulations, as highlighted in recent studies such as [29], where Physics-Informed Neural Networks are used for transport models in porous materials. In this context, correct parameter estimation, such as permeability and dispersivity, is pivotal for accurately modeling large-scale dynamics. Direct measurement of these parameters is typically expensive, time-consuming, and spatially limited. PINNs address this issue by enabling parameter estimation from sparse and noisy measurements of state variables of PDE systems. This makes them particularly valuable not only in porous media applications, but also in agronomy, soil sciences, and hydrology [39, 40, 41], as well as in materials science and biomedical engineering [42].

In summary, PINNs were effective forward and inverse PDE problem solvers, overcoming many of the limitations of conventional numerical methods. In particular, inverse problems are often ill-posed, meaning they can be non-unique or unstable, and are typically more computationally complex than forward problems. While recent studies on PINNs provided improved accuracy and efficiency, issues are still open. This requires further investigation into the application and extension of PINN approaches to problems such as the Porous Medium Equation (PME), where both stable forward solutions and accurate parameter estimation are paramount

The one-dimensional PME has been studied using classical numerical approaches, including finite difference schemes with various sweep and discretization strategies, as well as iterative solvers such as SOR and Newton-based methods [33, 36, 38], and variational approaches [37]. As for the inverse problem, see [34, 35]. However, despite this rich body of work, these approaches do not incorporate machine learning data-driven learning frameworks. This highlights the potential interest in exploring the effectiveness of PINNs for both direct and inverse formulations of the PME.

## 2 The Direct Problem of the 1D PME

The Porous Medium Equation (PME) is a nonlinear degenerate parabolic partial differential equation of the form

$$
\left\{ \begin{array} { l l } { \partial _ { t } u = \Delta ( u ^ { m } ) } & { m > 1 } \\ { u ( t , a ) = 0 , \quad u ( t , b ) = 0 } & { t > t _ { 0 } , } \\ { u ( t _ { 0 } , x ) = u _ { 0 } ( x ) } & { x \in ( a , b ) , } \end{array} \right.\tag{1}
$$

where $u = u ( x , t ) \geq 0$ denotes the density of a diffusing quantity, and m is the nonlinearity (or polytropic) exponent.

The equation is degenerate due to the fact that the diffusion coefficient $D ( u ) = m u ^ { m - 1 }$ becomes zero at $u = 0$ if the equation is equivalently written in the form $\partial _ { t } u = \partial _ { x } ( m u ^ { m - 1 } \partial _ { x } u )$ . The PME unifies a wide range of nonlinear diffusion phenomena occurring in physics, biology, and engineering, including gas transmission in porous media, population growth, and heat transfer in nonlinear materials. The PME can be considered as a nonlinear version of the classical heat equation, which is regained formally in the case $m = 1$

The mathematical theory of the PME developed over the years, beginning with foundational work of Oleinik et al. in the late 1950s, and finally reaching maturity in the 1980s, partly due to developments in functional analysis and nonlinear PDE theory [4].

Finite speed of propagation is also one of the typical features of the PME, meaning that disturbances propagate at finite speed, unlike in the classical heat equation, where infinite speed of propagation occurs. Indeed, if the initial data $u _ { 0 }$ is compactly supported, then so is the solution $u ( t )$ for all $t > 0$ , with a sharp free boundary separating the region where $u > 0$ from the vacuum. That results in the formation of moving interfaces and makes the maximum principle inapplicable. A fundamental exact solution of the PME is the Barenblatt solution. It describes the evolution of a compactly supported density from a Dirac-type initial condition and serves as a paradigm to study spreading rates, self-similarity, and asymptotic behavior. We will utilize such a solution to verify our numerical and PINNs methods. For a complete treatment of the PME and other related nonlinear diffusion equations, we would like to refer the reader to Vázquez's monograph [4], which will be the principal theoretical foundation of this work.

In this section, we solve the one-dimensional direct problem of the PME given in Eq. 1 written in the following form:

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \partial _ { t } u ( t , x ) = \partial _ { x } \left( \beta u ^ { \beta - 1 } \partial _ { x } u ( t , x ) \right) , } & { x \in [ - 1 , 1 ] , \ t \in [ 0 , 1 ] , } \\ { u ( t , - 1 ) = u ( t , 1 ) = 0 , } & { t \in [ 0 , 1 ] , } \\ { u ( 0 , x ) = u _ { 0 } ( x ) , } & { x \in [ - 1 , 1 ] , } \end{array} \right. } \end{array}\tag{2}
$$

We solve this problem using a classical numerical method (Section 2.1.1) and a PINN approach (Section 2.1.2). We then introduce four different reference solutions of the PME (Section 2.2), and finally, we present the results for the 1D direct problem in Section 2.3.

## 2.1 Methodologies

The inverse framework developed in this paper builds upon the direct PME solvers introduced in our earlier work [2]. For completeness, we briefly summarize the classical and PINN formulations used throughout this work. Additional implementation details and algorithmic descriptions can be found in [2].

## 2.1.1 Classical Numerical Scheme

As a reference solution, we employ the finite difference formulation introduced in [2], based on a backward Euler discretization in time and central differences in space. For each interior grid point, the resulting nonlinear residual is

$$
F _ { i } ( u ^ { n + 1 } ) = u _ { i } ^ { n + 1 } - u _ { i } ^ { n } - \frac { m \Delta t } { \Delta x ^ { 2 } } \left[ \left( \frac { u _ { i + 1 } ^ { n + 1 } + u _ { i } ^ { n + 1 } } { 2 } \right) ^ { m - 1 } ( u _ { i + 1 } ^ { n + 1 } - u _ { i } ^ { n + 1 } ) - \left( \frac { u _ { i } ^ { n + 1 } + u _ { i - 1 } ^ { n + 1 } } { 2 } \right) ^ { m - 1 } ( u _ { i } ^ { n + 1 } - u _ { i - 1 } ^ { n + 1 } ) \right] .
$$

with the vector of unknowns at time step $n + 1$ as

$$
\begin{array} { r } { u ^ { n + 1 } = ( u _ { 1 } ^ { n + 1 } , \ldots , u _ { N - 1 } ^ { n + 1 } ) ^ { \top } \in \mathbb { R } ^ { m } , \quad m = N - 1 , } \end{array}
$$

and the residual vector

$$
F ( u ^ { n + 1 } ) = ( F _ { 1 } , \ldots , F _ { N - 1 } ) ^ { \top } .
$$

The nonlinear system is solved at each time step using Newton's method. The update rule is

$$
J ( u ^ { ( k ) } ) \delta u ^ { ( k ) } = - F ( u ^ { ( k ) } ) , \quad u ^ { ( k + 1 ) } = u ^ { ( k ) } + \delta u ^ { ( k ) } ,
$$

where the Jacobian matrix is given by

$$
\left[ J \left( u ^ { ( k ) } \right) \right] _ { i j } = \frac { \partial F _ { i } } { \partial u _ { j } } \Bigg | _ { u = u ^ { ( k ) } } , \qquad i , j = 1 , \dots , s .
$$

and is approximated using finite differences.

Table 1 summarizes the parameters used. The pseudocode of the classical PME direct problem, using the Barenblatt solution 2.2.1 for the initial and boundary conditions, is summarized in Algorithm 11 of [2].

<table><tr><td>Parameter</td><td colspan="2"></td></tr><tr><td>Spatial domain</td><td></td><td> $\overline { { x \in [ - 1 , 1 ] } }$  with N = 100 intervals  $\overline { { ( N + 1 } }$  grid points)</td></tr><tr><td>Time domain</td><td> $t \in [ 0 , 1 ]$ </td><td>with  $\overline { { \Delta t = 0 . 0 1 } }$  (100 time steps)</td></tr><tr><td>Tolerance (Newton)</td><td> $\overline { { 1 0 ^ { - 6 } } }$ </td><td></td></tr><tr><td>Maximum Newton iterations</td><td>20 per time step</td><td></td></tr><tr><td>Jacobian approximation</td><td></td><td>Finite differences with perturbation  $\overline { { h = 1 0 ^ { - 6 } } }$ </td></tr></table>

Table 1: Parameters for Direct Classical Method

## 2.1.2 PINN Approach

Physics-Informed Neural Networks (PINNs) are built upon standard feedforward neural networks, which act as universal function approximators [12] by learning mappings between inputs and outputs through trainable weights and biases. Their training relies on automatic differentiation together with optimization algorithms such as ADAM and L-BFGS [7, 6], to minimize the loss function and accurately approximate PDE solutions. What distinguishes PINNs from traditional neural networks is the fact that they include physical laws, expressed as differential equations, directly in the loss function. To explain. PINNs use the residuals of the governing PDE as part of the loss. This means the network is penalized when it mispredicts known data and also when its predictions violate the underlying physics. For instance, suppose we have a PDE of the form:

$$
\begin{array} { r } { \mathcal { N } [ u ] ( { \bf x } , t ; c ) = 0 , } \end{array}
$$

where $\mathcal { N }$ is a differential operator acting on the unknown function $u ( { \bf x } , t )$ and c is an underlying parameter of the PDE. A PINN seeks to approximate u with a neural network $u _ { \theta } ( \mathbf { x } , t )$ , parameterized by weights θ. During training,

1. In solving the direct problem, the total loss is composed of the PDE, initial value, and boundary conditions residuals. Also, it may include any additional physical constraints.

$$
\mathcal { L } _ { d i r e c t } ( \theta ) = \mathcal { R } _ { \mathrm { P D E } } ( \theta ) + \mathcal { R } _ { \mathrm { i n i t i a l ~ v a l u e } } ( \theta ) + \mathcal { R } _ { \mathrm { b o u n d a r i e s } } ( \theta ) + \lambda \cdot \mathcal { R } _ { \mathrm { c o n s t r a i n t } } ( \theta ) .
$$

where λ is mathematically speaking the Lagrange multiplier, but in our computational context, we refer to it as a hyperparameter. It is worth noting that any other residual in the total loss can also be multiplied by another hyperparameter.

2. In solving the inverse problem, i.e. recovering c, the total loss is composed of the direct problem loss, along with a data loss. In this way, the network will be penalized for mispredicting both the physics and the data.

$$
\mathcal { L } _ { i n v e r s e } ( \theta ) = \mathcal { L } _ { \mathrm { d i r e c t } } ( \theta ) + \mathcal { R } _ { \mathrm { d a t a } } ( \theta )
$$

In this section, we give the details of the PINN constructed to solve the direct PME problem that gave the results in Section 2.3

The direct PINN framework follows the formulation presented in [2] and is summarized in the following section. The direct problem serves as the foundation for the inverse methodology developed later in this paper. A schematic representation of the direct PINN is shown in Figure 1.

![](images/e2d8ac509f2bcad30da44fc2b64d21bdbab54fd43103559385c06847fd99b188.jpg)  
Figure 1: Schematic of the direct PINN

Neural Network Architecture The solution $u ( t , x )$ is approximated using a fully connected neural network $u _ { \theta } ( t , x )$ . The network architecture used throughout this work is summarized in Table 2.

<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Specification</td></tr><tr><td rowspan=1 colspan=1>Input layer</td><td rowspan=1 colspan=1>Size 2 (for t and x)</td></tr><tr><td rowspan=1 colspan=1>Hidden layers</td><td rowspan=1 colspan=1>4 layers, each with 20 neurons</td></tr><tr><td rowspan=1 colspan=1>Activation function</td><td rowspan=1 colspan=1>Hyperbolic tangent tanh</td></tr><tr><td rowspan=1 colspan=1>Output layer</td><td rowspan=1 colspan=1>Single neuron returning $u _ { \theta } ( t , x )$ </td></tr></table>

Table 2: Architecture of the neural network.

weights All weights of the neural network are initialized with the Xavier initialization procedure that works best with tanh activation function. This process initializes the weights by sampling random values from a uniform distribution over the range:

$$
W \sim \mathcal { U } \left( - \sqrt { \frac { 6 } { n _ { \mathrm { i n } } + n _ { \mathrm { o u t } } } } , \sqrt { \frac { 6 } { n _ { \mathrm { i n } } + n _ { \mathrm { o u t } } } } \right) ,
$$

where $n _ { \mathrm { i n } }$ and $n _ { \mathrm { o u t } }$ are the number of input and output units of a layer, respectively [49]. Such initialization helps maintain stable gradients during training, especially when activation functions like tanh are employed since it maintains the signal variance across layers constant.

Loss Function and Optimization The training objective is to minimize a composite loss L that enforces agreement with the initial and boundary conditions as well as the underlying PDE. Let $\mathcal { D } _ { t } , \mathcal { D } _ { b } .$ and $\mathcal { D } _ { \Omega }$ denote sampled points on the initial boundary $t = 0$ , spatial boundaries $x = \pm 1$ , and interior domain Ω, respectively.

The total loss is:

$$
\mathcal { L } _ { \theta } = \log _ { 1 0 } \left( \lambda _ { u } \cdot \left( \mathcal { L } _ { b } + \mathcal { L } _ { t } \right) + \mathcal { L } _ { \mathrm { P D E } } \right)\tag{3}
$$

with

$$
\mathcal { L } _ { b } = \mathbb { E } _ { ( t , x ) \in \mathcal { D } _ { b } } \left[ \left( u _ { \theta } ( t , x ) - u _ { \mathrm { B C } } ( t , x ) \right) ^ { 2 } \right] ,\tag{4}
$$

$$
\mathcal { L } _ { t } = \mathbb { E } _ { ( t , x ) \in \mathcal { D } _ { t } } \left[ \left( u _ { \theta } ( t , x ) - u _ { 0 } ( x ) \right) ^ { 2 } \right] ,\tag{5}
$$

$$
\mathcal { L } _ { \mathrm { P D E } } = \mathbb { E } _ { ( t , x ) \in \mathcal { D } _ { \Omega } } \left[ \left( \partial _ { t } u _ { \theta } - \partial _ { x } \left( m u _ { \theta } ^ { m - 1 } \partial _ { x } u _ { \theta } \right) \right) ^ { 2 } \right] .\tag{6}
$$

where E[·] denotes the empirical mean over the sampled training points, i.e., $\mathbb { E } [ f ( x ) ] = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } f ( x _ { i } )$ which corresponds to the Mean Squared Error (MSE) used in the loss computation. Gradients are computed using automatic differentiation.

Training Procedure The neural network is trained using the following optimization approach:

1. ADAM Optimizer for up to 2000 epochs with a learning rate = 0.001.

2. followed by L-BFGS for up to 500 epochs with a learning rat $\mathrm { { e } = 0 . 0 5 }$

3. Early Stopping: The training halts if the validation loss does not improve by $1 0 ^ { - 6 }$ over a predefined patience window equals to 200 epochs in our case.

The optimizer minimizes the logarithmic loss for stability and to better visualize loss curves, and training points are generated using a Sobol sequence, which ensures a quasi-random coverage of the space-time domain.The network was trained with $n _ { \mathrm { i n t } } ~ = ~ 2 5 6$ interior points, $n _ { \mathrm { s b } } ~ = ~ 6 4$ points on each spatial boundary, and $n _ { \mathrm { t b } } = 6 4$ points for the initial condition with a total of 448 points.

Evaluation and Accuracy The solution $u _ { \theta } ( t , x )$ is evaluated against the analytical form. We compute the relative L2-error over the same grid points used by the classical Newton method which will be put in comparison with our PINN.

$$
\epsilon _ { r e l } = \frac { \| u _ { \theta } - u \| _ { L ^ { 2 } } } { \| u \| _ { L ^ { 2 } } } = \sqrt { \frac { \sum _ { i = 1 } ^ { N } \left( u _ { \theta } ( t _ { i } , x _ { i } ) - u ( t _ { i } , x _ { i } ) \right) ^ { 2 } } { \sum _ { i = 1 } ^ { N } u ( t _ { i } , x _ { i } ) ^ { 2 } } } .
$$

## 2.2 Reference Solutions

In this section, we present the four different PME solutions on which we will test our methods.

## 2.2.1 Barenblatt Solution

The Barenblatt solution, thoroughly explained in [4], is a well-known self-similar solution to the PME and serves as a fundamental benchmark for validating our numerical methods. It represents the evolution of an initial mass concentrated at a point and is particularly useful due to its explicit analytical form. The Barenblatt solution in one spatial dimension is given by:

$$
u ( t , x ) = t ^ { - \alpha } F \left( x t ^ { - \alpha } \right) ,
$$

where the similarity profile F is defined as:

$$
F ( \eta ) = \left( C - \kappa \eta ^ { 2 } \right) _ { + } ^ { \frac { 1 } { m - 1 } } , \quad \mathrm { w i t h } \quad \eta = x t ^ { - \alpha } .
$$

Here, $m > 1$ is the nonlinearity parameter of the PME, and the constants are:

$$
\alpha = \frac { 1 } { m + 1 } , \quad \kappa = \frac { ( m - 1 ) \alpha } { 2 m } = \frac { m - 1 } { 2 m ( m + 1 ) } .
$$

The parameter C is a constant that depends on the total mass of the solution and determines the support of $u ( t , x )$ . The notation (·)+ indicates that the expression is set to zero whenever the argument is negative to ensure compact support. This solution will not involve the homogeneous Dirichlet boundary conditions.

## 2.2.2 Manufactured Solution

To further test the accuracy of our solver, we use two manufactured solutions. These are artificial solutions that we plug into the equation to compute the source term $f ( x , t )$ , so that the equation is exactly satisfied. This helps us evaluate how well the solver can recover known solutions, even when no closed-form analytical solution exists for the original PME.

## Polynomial Solution

We first consider a smooth polynomial function in both time and space:

$$
u ( x , t ) = ( 1 - t ) ^ { 2 } ( 1 - x ^ { 2 } ) ^ { 2 } .
$$

By plugging this into the equation $\partial _ { t } u = \partial _ { x } \left( m u ^ { m - 1 } \partial _ { x } u \right) + f ( x , t )$ , we obtain the forcing term:

$$
f ( x , t ) = - 2 ( 1 - t ) ( 1 - x ^ { 2 } ) ^ { 2 } + 4 m ( 1 - t ) ^ { 2 m } \left[ ( 1 - x ^ { 2 } ) ^ { 2 m - 1 } - 2 ( 2 m - 1 ) x ^ { 2 } ( 1 - x ^ { 2 } ) ^ { 2 m - 2 } \right] .
$$

This solution is zero at the boundaries, so it satisfies the same boundary conditions as our test setup.

## Damped Harmonic Solution

We also test a harmonic function that decays over time:

$$
u ( x , t ) = e ^ { - t } \cos \left( \frac { \pi } { 3 } x \right) ,
$$

with the corresponding source term:

$$
f ( x , t ) = - e ^ { - t } \cos \left( \frac { \pi } { 3 } x \right) + m \left( \frac { \pi } { 3 } \right) ^ { 2 } e ^ { - t m } \left[ - ( m - 1 ) \cos ^ { m - 2 } \left( \frac { \pi } { 3 } x \right) \sin ^ { 2 } \left( \frac { \pi } { 3 } x \right) + \cos ^ { m } \left( \frac { \pi } { 3 } x \right) \right] .
$$

This solution will not involve the homogeneous Dirichlet boundary conditions.

## m-Dependent manufactured solution

As one can notice, all previously manufactured solutions did not depend on m. Here, we suggest an additional m-dependent solution

$$
u ( x , t ) = ( \cos \frac { \pi x } { 2 } ) ^ { 2 m } ( 1 + t ) ^ { m }
$$

with its corresponding source term:

$$
f ( x , t ) = m ( 1 + t ) ^ { m - 1 } \cos ^ { 2 m } \left( \frac { \pi x } { 2 } \right) - \frac { \pi ^ { 2 } } { 2 } m ^ { 2 } ( 1 + t ) ^ { m ^ { 2 } } \left[ ( 2 m ^ { 2 } - 1 ) \cos ^ { 2 m ^ { 2 } - 2 } \left( \frac { \pi x } { 2 } \right) \sin ^ { 2 } \left( \frac { \pi x } { 2 } \right) - \cos ^ { 2 m ^ { 2 } } \left( \frac { \pi x } { 2 } \right) \right]
$$

All these manufactured solutions are used to validate our classical and PINN-based solvers by directly comparing the computed results with the exact ones, as well as to compare the results with those of the classical solver.

## 2.3 Direct PINN Testing without Data

In this section, we validate our classical numerical solver ( 2.1.1) and the PINN approach ( 2.1.2) described against our four reference solutions provided in 2.2. The PINN testing in this section follows the standard approach presented in Section 2.1.2, where no data are incorporated into the loss function. A variation of this standard testing procedure, in which measurement data are added as an additional term in the loss function, is considered later in Section 2.4. First, we test the performance for different values of the polytropic exponent $m \in \{ 2 , 3 , 4 , 5 \}$ . Then, we compute the relative L2-error between the approximated solutions and the exact solution over the 10201 points of the classical mesh in Table 1:

$$
\epsilon _ { \mathrm { r e l } } = \frac { \| u _ { \mathrm { a p p r o x } } - u _ { \mathrm { e x a c t } } \| _ { L ^ { 2 } } } { \| u _ { \mathrm { e x a c t } } \| _ { L ^ { 2 } } } .
$$

and provide a table of the errors for each solution (Tables 3, 6, 9, 12). Other tables are also provided for each solution to compare the time (Tables 5, 8 , 11, 14) and the epochs (Tables 4, 7 , 10, 13) used by each method in each case. The reported epochs are approximate values estimated visually from the loss curves (Appendix A), reflecting the general convergence behavior of each method. Rather than reporting the final training epoch, we report the approximate epoch at which the loss first reached its plateau (i.e., the last meaningful improvement). Consequently, epochs corresponding to a plateau with no significant reduction in the loss, including any patience period preceding early stopping, were not counted. Also, only a visual comparison for the case of $m = 3$ is presented in this section (Figures 2, 4, 5, 6); for the visuals of the other cases, see Appendix A. For this section, all codes were run in Python using Google Colab with an NVIDIA Tesla T4 GPU [50]. Lastly, for each type of solution, we provide some observations and comments when needed

## Barenblatt Solution Results 2.2.1

While both methods are scientifically accurate (relative error below 5% in all cases, as shown in Table 3), the PINN consistently outperformed the classical numerical method in terms of accuracy. On the other hand, the accuracy improved as the nonlinearity parameter m increased for both methods. This is expected, as larger m values lead to more localized Barenblatt profiles, which are easier to approximate numerically and by neural networks. Note that for $m = 2$ , both methods had their largest errors.

Table 3: Relative L2-errors for the Barenblatt Solution
<table><tr><td>m</td><td>PINN vs Exact</td><td>Classical vs Exact</td><td>PINN vs Classial</td></tr><tr><td>2</td><td> $\overline { { 1 . 7 7 4 \times 1 0 ^ { - 2 } } }$ </td><td> $\overline { { 2 . 4 7 5 \times 1 0 ^ { - 2 } } }$ </td><td> $\overline { { 2 . 5 9 1 \times 1 0 ^ { - 2 } } }$ </td></tr><tr><td>3</td><td> $5 . 3 7 8 \times 1 0 ^ { - 4 }$ </td><td> $1 . 5 5 7 \times 1 0 ^ { - 2 }$ </td><td> $1 . 3 1 5 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>4</td><td> $3 . 3 9 4 \times 1 0 ^ { - 4 }$ </td><td> $7 . 2 3 2 \times 1 0 ^ { - 3 }$ </td><td> $6 . 4 0 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>5</td><td> $1 . 5 7 5 \times 1 0 ^ { - 4 }$ </td><td> $3 . 6 5 2 \times 1 0 ^ { - 3 }$ </td><td> $3 . 3 2 2 \times 1 0 ^ { - 3 }$ </td></tr></table>

Table 4: Training epochs for the Barenblatt solution
<table><tr><td rowspan=1 colspan=1>m</td><td rowspan=1 colspan=1>Adam epochs</td><td rowspan=1 colspan=1>LBFGS epochs</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>500</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>500</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>500</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>500</td></tr></table>

Table 5: time (sec) for the Barenblatt solution
<table><tr><td rowspan=1 colspan=1>m</td><td rowspan=1 colspan=1>PINN training time</td><td rowspan=1 colspan=1>Classical time</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>207.37</td><td rowspan=1 colspan=1>4.62</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>194.96</td><td rowspan=1 colspan=1>5.31</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>154.08</td><td rowspan=1 colspan=1>4.95</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>159.39</td><td rowspan=1 colspan=1>5.28</td></tr></table>

![](images/2f2f665486950c6835e3fc920c99f5c6e6f3aa2b3ce46b27140749f17b6e7bb2.jpg)  
(a) PINN solution

![](images/7907a5be84ad690432adaaaed7b014068ad2ae1ac1cf1d3c1a2a345ab1640027.jpg)  
(b) Exact solution

![](images/9c16e3194b9ffa088f944fc569e88862bd436e99e1f3de6f27cdcf153918a916.jpg)  
(c) Classical solution  
Figure 2: visual comparison of Barenblatt numerical solutions for $m = 3$

To better illustrate the evolution of the diffusion process, a zoomed-in view of the solution is presented over a reduced time interval.

![](images/ef0c89ecd1f5ec02da8825075288e871c53c0058f3c870d756d2011856b04f99.jpg)  
Figure 3: Zoomed-in view of the exact Barenblatt solution for m = 3 for $\mathrm { t } \in [ 0 , 0 . 3 ]$

## Polynomial Solution Results 2.2.2

Although the exact solution of the manufactured polynomial case does not depend on $m ,$ we still test the PINN for different values of $m \in \{ 2 , 3 , 4 , 5 \}$ . The goal is not to compare the solution shapes, since they are visually the same, but to observe how the PINN handles increasing nonlinearity as m increases.

![](images/297fec0a76217182eaa44fb92530c05ca256127d8dd14b6ce3a667b2dd451503.jpg)  
(a) PINN solution

![](images/bb80f5b37520261c3c1c1dd5ef6e69692c2a84e0692e9cd7a14fd0068a85e12f.jpg)  
(b) Exact solution

![](images/b747748c6c037f64fab400198e2d6561a9857dd88cd9bb8b76800a4b9e4e81b9.jpg)  
(c) Classical solution  
Figure 4: visual comparison of the Polynomial numerical solutions

The visuals are all the same over different values of m, as expected, since the solution is independent of m. However, from the training loss graphs shown in Appendix A, we can observe that as m increases the number of epochs needed to reach the minimum training loss also increases. This suggests that for higher values of $m ,$ the solution becomes more complex and requires more epochs for the model to converge. Despite this, the convergence behavior remains consistent, with the loss gradually decreasing and eventually stabilizing. This indicates that the PINN is able to learn the solution for all values of $m$ but the learning process takes longer as the nonlinearity strengthens as verified from the Table 7 and 8.

Table 6 shows that the relative $L ^ { 2 } .$ -errors of the PINN and exact solutions are low and accurate for all values of $m$ . While the error for $m = 2$ is slightly higher, all cases are considered accurate, with the PINN performing well and better than the classical method across all other values of $m ,$ except for a negligibly higher error in the case of $m = 2$

Table 6: Relative $L ^ { 2 } .$ -errors for the Polynomial solution
<table><tr><td>m</td><td>PINN vs Exact</td><td>Classical vs Exact</td><td>PINN vs Classical</td></tr><tr><td>2</td><td> $\overline { { 1 . 5 4 8 \times 1 0 ^ { - 2 } } }$ </td><td> $\overline { { 1 . 0 1 2 \times 1 0 ^ { - 2 } } }$ </td><td> $\overline { { 2 . 2 5 5 \times 1 0 ^ { - 2 } } }$ </td></tr><tr><td>3</td><td> $8 . 0 5 0 \times 1 0 ^ { - 4 }$ </td><td> $1 . 1 3 4 \times 1 0 ^ { - 2 }$ </td><td> $1 . 1 2 7 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>4</td><td> $1 . 2 5 5 \times 1 0 ^ { - 3 }$ </td><td> $1 . 1 8 6 \times 1 0 ^ { - 2 }$ </td><td> $1 . 1 5 7 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>5</td><td> $1 . 7 4 0 \times 1 0 ^ { - 3 }$ </td><td> $1 . 2 1 5 { \times } 1 0 ^ { - 2 }$ </td><td> $1 . 1 8 0 \times 1 0 ^ { - 2 }$ </td></tr></table>

Table 7: Training epochs for the Polynomial solution
<table><tr><td rowspan=1 colspan=1>m</td><td rowspan=1 colspan=1>Adam epochs</td><td rowspan=1 colspan=1>LBFGS epochs</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>310</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>410</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>460</td></tr></table>

Table 8: Time (sec) for the Polynomial solution
<table><tr><td rowspan=1 colspan=1>m</td><td rowspan=1 colspan=1>PINN training time</td><td rowspan=1 colspan=1>Classical time</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>38.65</td><td rowspan=1 colspan=1>5.32</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>157.23</td><td rowspan=1 colspan=1>5.82</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>185.16</td><td rowspan=1 colspan=1>5.59</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>194.58</td><td rowspan=1 colspan=1>4.33</td></tr></table>

## Damped Harmonic solution Results 2.2.2

The damped harmonic solution does not depend on m either. Therefore, we perform the same testing as in the polynomial solution case (Section 2.3) for the same reason.

![](images/c72f52e78138a2a02764f64d6084af77eb2ca0b1e3729505f12404b67b030c8f.jpg)  
(a) PINN solution

![](images/893921d6045ce3dd26d0c5386adb86a0225429595cf7db6d3cc8a1743d361a24.jpg)  
(b) Exact solution

![](images/3c0f77315c9e95595c2f042fdb6af50ae47bafa0c44acde0814ee144d5d0714c.jpg)  
(c) Classical solution  
Figure 5: visual comparison of the Harmonic numerical solutions

Table 9 shows that the PINN outperforms the classical method in all cases by two orders of magnitudes

Table 9: Relative $L ^ { 2 } \mathrm { - e r r o r s }$ for the Damped Harmonic solution
<table><tr><td>m</td><td>PINN vs Exact</td><td>Classical vs Exact</td><td>PINN vs Classical</td></tr><tr><td>2</td><td> $\overline { { 2 . 0 8 1 \times 1 0 ^ { - 5 } } }$ </td><td> $\overline { { 1 . 2 9 2 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 1 . 2 8 7 \times 1 0 ^ { - 3 } } }$ </td></tr><tr><td>3</td><td> $2 . 0 9 6 \times 1 0 ^ { - 5 }$ </td><td> $1 . 4 4 9 \times 1 0 ^ { - 3 }$ </td><td> $1 . 4 5 1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>4</td><td> $3 . 6 5 6 \times 1 0 ^ { - 5 }$ </td><td> $1 . 7 2 3 \times 1 0 ^ { - 3 }$ </td><td> $1 . 7 2 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>5</td><td> $5 . 1 5 6 \times 1 0 ^ { - 5 }$ </td><td> $2 . 0 4 9 \times 1 0 ^ { - 3 }$ </td><td> $1 . 5 6 6 \times 1 0 ^ { - 3 }$ </td></tr></table>

Table 10: Training epochs for the Harmonic solution
<table><tr><td rowspan=1 colspan=1>m</td><td rowspan=1 colspan=1>Adam epochs</td><td rowspan=1 colspan=1>LBFGS epochs</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>210</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>140</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>65</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>130</td></tr></table>

Table 11: Time (sec) for the Harmonic solution
<table><tr><td rowspan=1 colspan=1>m</td><td rowspan=1 colspan=1>PINN training time</td><td rowspan=1 colspan=1>Classical time</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>149.95</td><td rowspan=1 colspan=1>5.38</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>112.42</td><td rowspan=1 colspan=1>6.33</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>84.34</td><td rowspan=1 colspan=1>5.29</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>108.06</td><td rowspan=1 colspan=1>6.21</td></tr></table>

## m-dependent solution Results 2.2.2

This solution was harder to approximate using both the classical and the PINN methods. Classically, the higher the m is, the more time it takes to give a solution (Table 14). Meanwhile, the early stopping was triggered in all cases in the PINN case. Moreover, the error increases as m increases in both methods. Table 12 shows that for $m = 2$ , the smallest tested value of $m ,$ both the PINN and the classical method have errors of the same order of magnitude, with the PINN being slightly more accurate. As m increases. the errors of both methods also increase. However, the classical method remains more accurate, while the PINN error grows more rapidly. For values of m greater than 2, the classical method gives errors in an approximate range of 0.8% to 2%, whereas the PINN errors range from about 2% to 15%.

![](images/53ee8cad0d5a13f4e4d80d8cdc2fc6c51c4c9fceada02d2c85f71be42585f606.jpg)  
(a) PINN solution

![](images/91635381dc08f7c21b95b1b839abd644000f5686a1caab8019e17c1ef845e149.jpg)  
(b) Exact solution

![](images/30704b9ce36fa27d7b8040151456855b2cd9dafec2dcae639bd30b54e2120258.jpg)  
(c) Classical solution  
Figure 6: visual comparison of the m-dependent numerical solutions

Note that we can always keep improving the PINN to our problem by tuning more of the hyperparameters, changing architectures, or doing some computational tricks that can reduce the effect of nonlinearity. However, for the tests and comparisons to be adequate and fair across all the test cases and all the different solutions we have, we decided to fix our model to what is detailed in section 2.1.2 and perform the tests.

Table 12: Relative $L ^ { 2 } .$ -errors for the m-dependent solution
<table><tr><td>m</td><td>PINN vs Exact</td><td>Classical vs Exact</td><td>PINN vs Classical</td></tr><tr><td>2</td><td> $\overline { { 1 . 5 8 6 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 2 . 7 1 5 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 1 . 6 3 6 \times 1 0 ^ { - 3 } } }$ </td></tr><tr><td>3</td><td> $1 . 6 5 1 \times 1 0 ^ { - 2 }$ </td><td> $8 . 5 1 4 \times 1 0 ^ { - 3 }$ </td><td> $1 . 3 1 1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>4</td><td> $7 . 6 7 9 \times 1 0 ^ { - 2 }$ </td><td> $1 . 4 3 7 \times 1 0 ^ { - 2 }$ </td><td> $7 . 0 7 4 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>5</td><td> $1 . 5 0 5 { \times } 1 0 ^ { - 1 }$ </td><td> $1 . 8 0 2 \times 1 0 ^ { - 2 }$ </td><td> $1 . 4 2 0 \times 1 0 ^ { - 1 }$ </td></tr></table>

Table 13: Training epochs for the m-dependent sol.
<table><tr><td rowspan=1 colspan=1>m</td><td rowspan=1 colspan=1>Adam epochs</td><td rowspan=1 colspan=1>LBFGS epochs</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>200</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>150</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>120</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>90</td></tr></table>

Table 14: Time (sec) for the m-dependent solution
<table><tr><td rowspan=1 colspan=1>m</td><td rowspan=1 colspan=1>PINN training time</td><td rowspan=1 colspan=1>Classical time</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>62.89</td><td rowspan=1 colspan=1>5.66</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>56.53</td><td rowspan=1 colspan=1>10.09</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>43.45</td><td rowspan=1 colspan=1>11.51</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>36.141</td><td rowspan=1 colspan=1>24.33</td></tr></table>

## 2.4 Direct PINN Testing with Data

A natural extension of the direct PINN is to include additional measurement data in the training process rather than minimizing the loss only over the PDE residual together with the initial and boundary conditions. This setting corresponds to the practical scenario in which the governing PDE is known but a set of observations of the solution is also available. In such a case, the PINN can exploit both the physical constraints imposed by the PDE and the available measurements, potentially improving the quality of the approximation or the time complexity. This experiment is performed only for the polynomial solution. The polynomial test case was selected because, for different values of $m$ the relative errors obtained by the standard direct PINN range from approximately $1 0 ^ { - 2 }$ to $1 0 ^ { - 4 }$ . This wide range is representative of the error magnitudes observed for the other manufactured solutions and therefore provides a suitable benchmark for assessing the influence of incorporating measurement data.

The only modification with respect to the loss function defined in (3) is the addition of a measurement loss term. The total loss becomes

$$
L = \log _ { 1 0 } ( \lambda _ { u } L _ { u } + L _ { \mathrm { i n t } } + \lambda _ { \mathrm { m e a s } } L _ { \mathrm { m e a s } } ) ,
$$

where $\lambda _ { \mathrm { m e a s } }$ is a hyperparameter chosen to be 10, and

$$
L _ { \mathrm { m e a s } } = \frac { 1 } { N _ { \mathrm { m e a s } } } \sum _ { i = 1 } ^ { N _ { \mathrm { m e a s } } } \left( u _ { \theta } ( t _ { i } , x _ { i } ) - u _ { i } ^ { \mathrm { d a t a } } \right) ^ { 2 } .
$$

We choose $\lambda _ { \mathrm { m e a s } } = 1 0$ , equal to the weight assigned to the boundary and initial condition loss $( \lambda _ { u } = 1 0 )$ since both terms enforce agreement with known pointwise values of the solution, whereas the interior loss enforces the governing differential equation. The measurement dataset consists of $N _ { \mathrm { m e a s } } = 2 2 5$ exact solution values sampled on a uniform $1 5 \times 1 5$ grid over the computational domain $( x , t ) \in [ - 1 , 1 ] \times [ 0 , 1 ]$

The results are summarized in Table 15 and compared with the standard direct PINN results reported in Tables 8 and 6. The inclusion of the measurement loss does not lead to a significant improvement in the approximation accuracy. The relative $L ^ { 2 } .$ -errors remain of the same order of magnitude as those obtained with the standard PINN. For $m = 4$ and $m = 5$ , the errors slightly decrease from $1 . 2 5 5 \times 1 0 ^ { - 3 }$ to $1 . 0 0 8 \times 1 0 ^ { - 3 }$ and from $1 . 7 4 0 \times 1 0 ^ { - 3 }$ to $1 . 6 2 2 \times 1 0 ^ { - 3 }$ , respectively. However, for $m = 2$ and $m = 3 .$ the addition of measurement data slightly increases the error from $1 . 5 4 8 \times 1 0 ^ { - 2 }$ to $1 . 7 1 6 \times 1 0 ^ { - 2 }$ and from $8 . 0 5 0 \times 1 0 ^ { - 4 } \mathrm { ~ t o ~ } 8 . 2 1 4 \times 1 0 ^ { - 4 }$ , respectively. Also, the inclusion of measurement data introduces an additional computational cost, as the network must minimize an extra loss term.

<table><tr><td rowspan=1 colspan=1>m</td><td rowspan=1 colspan=1>PINN relative error</td><td rowspan=1 colspan=1>PINN training time</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1> $\overline { { 1 . 7 1 6 \times 1 0 ^ { - 2 } } }$ </td><td rowspan=1 colspan=1>60.74</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1> $\overline { { 8 . 2 1 4 \times 1 0 ^ { - 4 } } }$ </td><td rowspan=1 colspan=1>207.00</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1> $\overline { { 1 . 0 0 8 \times 1 0 ^ { - 3 } } }$ </td><td rowspan=1 colspan=1>193.70</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1> $\overline { { 1 . 6 2 2 \times 1 0 ^ { - 3 } } }$ </td><td rowspan=1 colspan=1>191.69</td></tr></table>

Table 15: Relative $L ^ { 2 } .$ -errors and training time (second) for the direct PINN applied to the polynomial solution using exact data

Therefore, for this test case, where the standard direct PINN already provides accurate solutions, adding additional measurement data is not necessary, as it does not provide a meaningful improvement in accuracy while increasing the training time.

It was also worth performing this test on the m-dependent solution, where the standard direct PINN produced larger errors, exceeding $5 \%$ for $m = 4$ and $m = 5$ . The aim is to check whether adding measurement data helps in the cases where the standard PINN struggles to converge. The results are reported in the following Table 16:

Table 16: Relative $L ^ { 2 } .$ -errors and training time (second) for the direct PINN applied to the m-dependent solution using exact data
<table><tr><td rowspan=1 colspan=1>m</td><td rowspan=1 colspan=1>PINN relative error</td><td rowspan=1 colspan=1>PINN training time</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1> $\overline { { 1 . 1 0 9 \times 1 0 ^ { - 3 } } }$ </td><td rowspan=1 colspan=1>69.66</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1> $\overline { { 1 . 2 8 3 \times 1 0 ^ { - 2 } } }$ </td><td rowspan=1 colspan=1>71.18</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1> $6 . 7 4 7 \times 1 0 ^ { - 2 }$ </td><td rowspan=1 colspan=1>71.32</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1> $\overline { { 2 . 2 8 4 \times 1 0 ^ { - 1 } } }$ </td><td rowspan=1 colspan=1>46.68</td></tr></table>

The addition of measurement data does not resolve the difficulty observed for the m-dependent solution. For $m = 2 , m = 3$ , and $m = 4$ , the relative error decreases slightly, but it remains of the same order of magnitude as before, and the errors for $m = 4$ and $m = 5$ stay well above 5%. For $m = 5$ , the error even increases, from $1 . 5 0 5 \times 1 0 ^ { - 1 }$ to $2 . 2 8 4 \times 1 0 ^ { - 1 }$ . At the same time, the training time increases in all four cases. These observations are consistent with the conclusion drawn from the polynomial test: adding measurement data does not lead to a meaningful improvement in accuracy and does not rescue the cases where the standard PINN fails, while it increases the training time.

In summary, for the direct problem, the PINN achieves accurate solutions close to the classical methods. where the PINN performs better in some cases and the classical method performs better in others, except for the m-dependent solution case, where the difference between the PINN and the classical method is relatively larger than in the other solution cases. Also, in terms of time, the classical method is always faster. However, one should note that the time data presented in the tables for the PINN include the time taken to trigger early stopping, which is equal to the time required to run 200 epochs without notable improvement. Nevertheless, this would still be negligible, and the classical method remains faster.

## 3 The 2-stage Learning Inverse PINN

Since the direct PINN approach provided accurate and stable results, we now extend it to the inverse problem, highlighting the flexibility of PINNs, where only minor modifications are needed to move from solving the direct problem to solving the inverse one. The inverse problem of the PME consists of determining the polytropic parameter m from given measurements. In our study, these measurements are generated from exact reference solutions.

We begin by applying a classical optimization-based method using MATLAB fmincon in Section 3.1. Next, we present a PINN to solve the same inverse problem in Section 3.2, where we propose our novel inverse PINN model 3.3.1. Finally, we compare and analyze the performance of these methods through numerical testing in Section 3.4.

## 3.1 Classical Approach: fmincon

MATLAB's fmincon is an optimization solver that is designed to find the constrained minimum of a multi-variable scalar function from an initial guess. Such a procedure, generally called nonlinear programming or constrained nonlinear optimization, is most appropriate whenever parameters must be

estimated within physically reasonable constraints. In this work, we use it to estimate the parameter m of the PME by minimizing the difference between synthetic reference data and our numerical solution, under the constraint $1 \leq m \leq 1 0$

The implementation can be summarized as follows:

1. We generate synthetic data from solutions 2.2 with a known m.

2. Define an objective function that computes the error between the numerical and exact solutions

$$
J ( \boldsymbol { m } ) = \sum _ { i , k } \big [ u _ { \mathrm { n u m } } ( x _ { i } , t _ { k } ; m ) - u _ { \mathrm { e x a c t } } ( x _ { i } , t _ { k } ) \big ] ^ { 2 } ,
$$

3. For a given m, solve the nonlinear PDE system using Newton's method at each time step to obtain $u _ { \mathrm { n u m } } ( x , t ; m )$

4. Use MATLAB's fmincon to iteratively update m and minimize the objective $J ( m )$ , subject to the bounds $1 \leq m \leq 1 0$

A pseudo-code of the corresponding inverse problem algorithm using the Barenblatt solution is found in Algorithm 8 in [2].

## 3.2 PINN Approach

Physics-informed neural networks are particularly flexible because once a direct PINN is implemented, only minor modifications are needed to adapt it to an inverse problem. Instead of restating the full direct PINN formulation, in this section we focus only on the adjustments required to estimate unknown parameters from data. Below, we highlight the differences in the loss function, network outputs, and training procedure that distinguish the inverse PINN from the direct case.

## Standard Inverse PINN Model

1. Neural Network Architecture: Same architecture and initialization, but we treat m as trainable real scalar parameter.

2. Loss function and optimization: In addition to the previous loss elements, we add the measurement loss

$$
\mathcal { L } _ { \theta } = \log _ { 1 0 } \left( \lambda _ { u } \cdot \left( \mathcal { L } _ { b } + \mathcal { L } _ { t } \right) + \mathcal { L } _ { \mathrm { P D E } } + \lambda _ { m } \cdot \mathcal { L } _ { \mathrm { m e a s } } \right) ,\tag{7}
$$

with $\mathcal { L } _ { \mathrm { m e a s } }$ the data mismatch term:

$$
\mathcal { L } _ { \mathrm { m e a s } } = \mathbb { E } _ { ( t , x ) \in \mathcal { D } _ { \mathrm { m e a s } } } \left[ \left( u _ { \theta } ( t , x ) - u _ { \mathrm { m e a s } } ( t , x ) \right) ^ { 2 } \right] ,
$$

where $\mathcal { D } _ { \mathrm { m e a s } }$ denotes the set of measurement points where we evaluate the exact solution. These measurement points are generated using the corresponding exact solution.

3. Training Procedure Same as in the direct PINN 2.1.2

After these standard modifications for the inverse PINN, the neural network was able to converge only locally, where it provided a good approximation of m only when the initial guess was close to the true value, and large approximation errors otherwise. Therefore, aiming for a global convergence, we started experimenting with several aspects, including different initialization, architecture, optimization techniques, and other details. Among all tested approaches, the most significant improvement came from the introduction of a novel two-stage training PINN framework, which substantially stabilized the inverse optimization process and improved convergence across different initial guesses. Figure 7 shows the results of the inverse problem for the Harmonic solution before and after the two-stage Training Model. For a comparison of the other solutions, see Appendix C. The standard approach is the one in which the only changes over the direct PINN model are the three enumerated points in 3.2, while the two-stage learning model will be detailed in the following section.

![](images/ff373830c31b1362d3515127d42be938ae6dc72df329f32b82c1ea68dd413f76.jpg)  
(a) Standard Inverse PINN

![](images/a7560a8ee7a6461607fc57e609ea811ed19b189de63d594d5a790ef4a4729516.jpg)  
(b) 2-Stage Training PINN  
Figure 7: Relative error of the estimated parameter m with respect to different initial guesses. Each color corresponds to a different true value of m. The x-axis represents the initial guess mo, taken around the true value (±50% and 80% perturbations). The y-axis shows the relative error (%). The left plot shows the standard inverse PINN, where the solution is accurate only when the initial guess is close to the true value, while the right plot shows the results after applying the 2-stage training PINN, where the method achieves significantly lower and more stable errors across different initial guesses.

The comparison in Figure 7 highlights the main contribution of this work. While the standard inverse PINN behaves as a locally convergent method that strongly, the proposed two-stage framework significantly improves robustness and allows accurate parameter recovery across a much wider range of initial guesses. This demonstrates that the training schedule itself plays a critical role in stabilizing inverse PINN optimization for nonlinear PDEs such as the PME.

## 3.3 Inverse PINN Design Experiments

In the following parts of this section, for each PINN component, we explain the different trials we conducted and highlight what worked best and was used in the final proposed two-stage inverse PINN framework 3.3 that produced the results in section 3.4. The following modifications collectively led to the final inverse PINN model 3.3.1 proposed in this work.

## Neural Network Architecture

For the inverse PINN, we first started experimenting by increasing and decreasing the length (i.e. number of neurons per layer) and depth (i.e., the number of layers of the neural network) of our neural network, but no clear improvements were shown. Then, as suggested and discussed in the literature [46] we tried the narrow-wide-narrow and wide-narrow-wide architectures, but again the initial configuration used in the direct problem 2 gave the best results, so we kept it.

## Weights

We experimented with several weight initializations to improve training dynamics and reduce the error in estimating m:

• Xavier normal: Xavier Glorot normal initialization

• Xavier uniform: Xavier Glorot uniform initialization

• kaiming\_normal: He normal initialization

• orthogonal: Orthogonal initialization

• scaled\_normal: Custom scaled normal initialization

Initially, in the direct problem, we used the Xavier uniform scheme, but it was shown that the Xavier normal initialization consistently produced better convergence and more accurate parameter recovery in the inverse setting. Even though both Xavier uniform and normal initializations have the same variance and aim to prevent vanishing or exploding gradients, the normal distribution, due to its unbounded support and variance characteristics [49], proved more effective for our PINN training.

Mathematically, the Glorot initialization selects the variance of each layer's weights such that forward activations and backward gradients remain stable across layers. For a layer with n-in and n-out number of neurons, the target variance is given by

$$
\operatorname { V a r } ( w ) = { \frac { 2 } { \mathrm { n } { \mathrm { . i n } } + \mathrm { n } { \mathrm { . o u t } } } } .
$$

This variance can be realized either through a uniform distribution,

$$
w \sim U \left( - g \sqrt { \frac { 6 } { \mathrm { n } \mathrm { { i n } + \mathrm { n } \mathrm { { - } \mathrm { { o u t } } } } } } , \ : g \sqrt { \frac { 6 } { \mathrm { n } \mathrm { { i n } + \mathrm { n } \mathrm { { - } \mathrm { { o u t } } } } } } \right) ,
$$

or through a normal distribution,

$$
w \sim \mathcal { N } \left( 0 , g ^ { 2 } \frac { 2 } { \mathrm { n . i n } + \mathrm { n . o u t } } \right) ,
$$

with $g = 1$ in our case. The Xavier normal initialization gave us better results in terms of errors.

## Stability

We also did the following modifications to enhance stability:

• Since m is known to be positive, we parameterize it as

$$
m = \exp ( { \widehat { m } } ) ,
$$

This formulation guarantees $m > 0$ by construction and avoids the need for manual clipping.

• we applied L2 gradient clipping to prevent instability caused by large gradients during backpropagation. At each training step, the total gradient norm $\| \mathbf { g } \|$ over all trainable parameters is computed. If

$$
\lVert \mathbf { g } \rVert > \mathrm { m a x . n o r m } ,
$$

the gradients are rescaled according to

$$
\mathbf { g }  \mathbf { g } \cdot \frac { \operatorname* { m a x } _ { - \mathrm { n o r m } } } { \| \mathbf { g } \| } ,
$$

with max\_norm = 1 in our implementation. This limits excessively large parameter updates specifically encountered when m is big, and prevents the occurrence of NaN values during training.

## 3.3.1 The Proposed Two-Stage Training Inverse PINN Model

We increased the number of training points, so the model used 2048 interior points $( n _ { \mathrm { i n t } } )$ , 256 spatial boundary points per side $\left( n _ { \mathrm { s b } } \right)$ , 256 temporal boundary points $\left( n _ { \mathrm { t b } } \right)$ , and 225 measurement points $( n _ { \mathrm { m e a s } } ~ = ~ 1 5 ~ \times ~ 1 5 )$ uniformly spread over the domain. We initially tried to use the same training procedure as in the direct problem (jointly minimizing PDE, initial/boundary, and measurement losses with fixed hyperparameters $( \mathrm { i . e . }$ , loss weights)). However, we observed that the recovered m was highly sensitive to the manual choice of loss weights, and we could not obtain a reliable convergence behavior across different initial guesses $m _ { 0 } \ { \mathrm { ( i . e . } }$ , the model did not consistently converge to the true m for different initial guesses). A natural idea was then to make the loss weights learnable parameters and optimize them during training, but this behaved poorly in practice: it led to unstable training and catastrophic performance even for the direct problem. This suggested that, in our setting, adding more trainable degrees of freedom makes optimization harder rather than easier.

Motivated by these observations, we propose the two-stage training PINN, which is specifically designed to address these problems. The main idea behind the proposed two-stage training PINN is to separate the learning of the solution structure from the learning of the physical parameter $m$ Instead of optimizing both $u _ { \theta }$ and $m$ simultaneously from the beginning, the network is first guided toward learning a physically meaningful solution profile before updating the unknown parameter.

Two-stage strategy Since we already validated in the direct problem that the network can approximate accurate solutions when the PDE parameter is known, we adopted the following training schedule:

1. Stage 1 (Warm-up, 20% of epochs): m is frozen at the initial guess $m _ { 0 }$ . The network $u _ { \theta } ( t , x )$ is trained to satisfy the PDE and initial/boundary conditions as if it is solving the direct problem with $\mathbf { m } = m _ { 0 }$ the initial guess.

2. Stage 2 (Main training): m is unfrozen and we jointly optimize $( \theta , m )$

Intuitively, Stage 1 helps the model learn a reasonable solution structure before it starts adjusting the physical parameter. This reduces the tendency of the optimizer to use m to compensate for a poorly learned $u _ { \theta }$ through incorrect updates of $m$ , which was one of the main causes of unstable convergence in the standard inverse PINN formulation. Also, in this inverse training, we only used ADAM optimizer with 10,000 total epochs.

Curriculum via hyperparameter annealing for measurements Even with two-stage training, the choice of $\lambda _ { m }$ remains important. Instead of selecting a single fixed value, we used a simple curriculum learning (sometimes referred to as loss-weight annealing): during the warm-up, $\lambda _ { m }$ is kept small at 10, and after warm-up it is increased gradually to its final value at 100. Concretely, letting E be the total number of epochs and $E _ { w }$ the warm-up epochs, we use:

![](images/b71e01e3d465edc903ad1c6d5af25b4123de707b0c3cd0cfb54a1cbdac549cf6.jpg)  
Figure 8: Training Diagram

The pseudo-code of the full procedure is summarized in Algorithm 1.

```powershell
Algorithm 1 Two-Stage Training Pseudo-code for the Inverse PINN
Require: Initial guess $m _ { 0 } ;$ total epochs $E ;$ warm-up fraction $f _ { w } ; \lambda _ { u } ; \lambda _ { m } ^ { \mathrm { i n i t } } , \lambda _ { m } ^ { \mathrm { f i n a l } }$
1: $E _ { w } \gets \lfloor f _ { w } \cdot E \rfloor$
Stage 1: Warm-up $( e = 0 , \ldots , E _ { w } - 1 )$
2: Freeze m at $m _ { 0 } \quad ( \nabla _ { m } \gets 0 )$
3: for $e = 0$ to $E _ { w } - 1$ do
4:Update $\theta$ only by minimizing $\log _ { 1 0 } \bigl ( \lambda _ { u } ( \mathcal { L } _ { b } + \mathcal { L } _ { t } ) + \mathcal { L } _ { \mathrm { P D E } } + \lambda _ { m } ^ { \mathrm { i n i t } } \mathcal { L } _ { \mathrm { m e a s } } \bigr )$
5: end for
Stage 2: Main Training $( e = E _ { w } , \ldots , E - 1 )$
6: Unfreeze m
7: for $e = E _ { w }$ to $E - 1$ do
8: $\lambda _ { m } ( e )  \lambda _ { m } ^ { \mathrm { i n i t } } + \frac { e - E _ { w } } { E - E _ { w } } ( \lambda _ { m } ^ { \mathrm { f n a l } } - \lambda _ { m } ^ { \mathrm { i n i t } } )$
9:Update $( \theta , m )$ jointly by minimizing $\log _ { 1 0 } ( \lambda _ { u } ( \mathcal { L } _ { b } + \mathcal { L } _ { t } ) + \mathcal { L } _ { \mathrm { P D E } } + \lambda _ { m } ( e ) \mathcal { L } _ { \mathrm { m e a s } } )$
10: end for
return Trained θ, estimated m
```

Relation to prior work. The idea of splitting training into stages is consistent with existing PINN practices. For example, a two-stage PINN has been used in the direct setting where a second stage adds extra physical structure (e.g., conserved quantities) to refine the solution [43]. In a related direction, a two-stage learning strategy has been proposed where a pre-training phase helps boundary losses converge, followed by a main phase that balances the full objective [44]. Our warm-up (freeze-unfreeze) schedule follows the same general motivation: stabilize training by learning easier constraints first.

Our gradual increase of $\lambda _ { m }$ is also aligned with curriculum learning ideas. Curriculum learning was introduced as training models by presenting easier samples/constraints first and progressively increasing difficulty [45]. Recently, curriculum strategies have been adapted to PINNs and shown to improve convergence and solution quality [47]. Similar curriculum training has also been used in time-dependent physics problems by splitting training along temporal intervals and training progressively over time [48]. In our inverse PME setting, we apply this principle at the loss level (annealing $\lambda _ { m } )$ to smoothly introduce measurement data while maintaining stable PDE/BC learning in our inverse problem

## 3.4 Testings and results

We test our classical and PINN approaches on all the different reference solutions for $m \in \{ 2 , 3 , 4 , 5 \}$ To study the sensitivity of each method to the initial condition, each value of m is tested using four different initial conditions corresponding to ±50% and ±80% of the value. To better analyze the results, they are represented by plots showing the relative error versus the initial guess for every value of m and for all four reference solutions. For this section, the PINN codes were run in Python on Google Colab with an NVIDIA Tesla T4 GPU [50], while the classical codes were run locally in MATLAB R2024a (Version 24.1) using MATLAB's fmincon. When small adjustments are made in either our classical or PINN approaches to improve results or overcome failures and errors, they are noted in their corresponding sections. If no comments are present, this means that the results were obtained using the following configurations for the classical and PINN methods, along with the configuration of the fmincon, in tables 1, 18, and 17 respectively.

Table 17: fmincon Configuration
<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Optimization algorithm</td><td rowspan=1 colspan=1>Interior-point (MATLAB default)</td></tr><tr><td rowspan=1 colspan=1>Maximum iterations</td><td rowspan=1 colspan=1>50</td></tr><tr><td rowspan=1 colspan=1>Optimality tolerance</td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 6 } } }$ </td></tr><tr><td rowspan=1 colspan=1>Lower bound</td><td rowspan=1 colspan=1> $m = 1$ </td></tr><tr><td rowspan=1 colspan=1>Upper bound</td><td rowspan=1 colspan=1> $m = 1 0$ </td></tr></table>

To ensure a fair comparison between the classical inverse solver and the PINN approach, we used the same uniform distribution of measurements for the classical method (Figure 9). Upon testing, some classical objective functions were undefined for some initial guesses, others produced numerical instabilities and sigularities and this resulted in a failed run. Therefore, we wrapped the call inside a try-catch block so that in case something went wrong during the run, the run was tagged as 'failed' in the results file so we don't crash the whole script. This is translated into non-existent points in the following plots. On the contrary, the PINN does not implement the PDE step-by-step but learns a function and adjusts the parameter; therefore, this allows it to handle a very wide range of initial values with better stability and without crashing. In addition to the tests using data generated from the exact reference solutions, the robustness of the inverse PINN is further investigated by testing it with noisy data, where the measurement data is generated from the direct PINN solution and then perturbed with different levels of noise. This analysis is presented in Section 3.4.1 and Section 3.4.2 respectively.

Table 18: Configurations and Parameters for Inverse PINN Method
<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value/Description</td></tr><tr><td rowspan=1 colspan=1>Training Points (see Fig 9)</td><td rowspan=1 colspan=1>n_int (interior points)n_sb (spatial boundary points per side)n_tb (temporal boundary points)n_meas (measurement points)</td><td rowspan=1 colspan=1>204825625615x15 = 225</td></tr><tr><td rowspan=1 colspan=1>Domain</td><td rowspan=1 colspan=1>Time domainSpace domain</td><td rowspan=1 colspan=1>[0, 1][-1, i]</td></tr><tr><td rowspan=1 colspan=1>Loss Weights</td><td rowspan=1 colspan=1>lambda_u (boundary/initial loss weight)lambda_m (measurement loss weight)</td><td rowspan=1 colspan=1>10100</td></tr><tr><td rowspan=1 colspan=1>Neural Network (approximate_solution)</td><td rowspan=1 colspan=1>Input dimensionOutput dimensionHidden layersNeurons per layerActivation functionInitializationRetrain seed</td><td rowspan=1 colspan=1>2 (t, x)1 (u)420nn.Tanh()Xavier normal , bias=042</td></tr><tr><td rowspan=1 colspan=1>Sampling</td><td rowspan=1 colspan=1>Generator</td><td rowspan=1 colspan=1>torch.quasirandom.SobolEngine(dimension=2)</td></tr><tr><td rowspan=1 colspan=1>Optimizer - ADAM</td><td rowspan=1 colspan=1>Learning rate (network)Learning rate (m parameter)</td><td rowspan=1 colspan=1>5e-41e-3</td></tr><tr><td rowspan=1 colspan=1>Training Config</td><td rowspan=1 colspan=1>num_epochsearly_stopping-patienceearly_stopping_min_delta</td><td rowspan=1 colspan=1>100005001e-7</td></tr></table>

Distribution of Training Points  
![](images/8803a10aaa8889dd32c81c1ece00176a0cc3c780341e03569a175a7eabe4ac13.jpg)  
Figure 9: Distribution of Training Points

## 3.4.1 Inverse PINN testing without Noisy Data

In this section, the inverse PINN is first tested using data generated directly from the exact solution, without adding any additional perturbation or considering the direct PINN solution.

Barenblatt Solution The testing results for the Barenblatt solution are presented and visualized in the following Table 19 and Figure 10:
<table><tr><td rowspan=1 colspan=1>true m</td><td rowspan=1 colspan=1>initial guess</td><td rowspan=1 colspan=1>fmincon relative error</td><td rowspan=1 colspan=1>PINN relative error</td><td rowspan=1 colspan=1>fmincontime</td><td rowspan=1 colspan=1>PINN training time</td></tr><tr><td rowspan=4 colspan=1>2</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>1.74%</td><td rowspan=1 colspan=1>2.30</td><td rowspan=1 colspan=1>228.70</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>4.67%</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>238.43</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.28%</td><td rowspan=1 colspan=1>1.07%</td><td rowspan=1 colspan=1>14.50</td><td rowspan=1 colspan=1>229.50</td></tr><tr><td rowspan=1 colspan=1>3.6</td><td rowspan=1 colspan=1>0.28%</td><td rowspan=1 colspan=1>1.71%</td><td rowspan=1 colspan=1>19.88</td><td rowspan=1 colspan=1>124.91</td></tr><tr><td rowspan=4 colspan=1>3</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>0.79%</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>230.82</td></tr><tr><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>0.37%</td><td rowspan=1 colspan=1>0.52%</td><td rowspan=1 colspan=1>12.52</td><td rowspan=1 colspan=1>238.57</td></tr><tr><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>0.37%</td><td rowspan=1 colspan=1>0.11%</td><td rowspan=1 colspan=1>11.16</td><td rowspan=1 colspan=1>216.04</td></tr><tr><td rowspan=1 colspan=1>5.4</td><td rowspan=1 colspan=1>0.37%</td><td rowspan=1 colspan=1>0.19%</td><td rowspan=1 colspan=1>10.21</td><td rowspan=1 colspan=1>212.74</td></tr><tr><td rowspan=4 colspan=1>4</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>3.60%</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>215.14</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.41%</td><td rowspan=1 colspan=1>3.92%</td><td rowspan=1 colspan=1>8.41</td><td rowspan=1 colspan=1>216.10</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0.41%</td><td rowspan=1 colspan=1>2.50%</td><td rowspan=1 colspan=1>8.16</td><td rowspan=1 colspan=1>206.24</td></tr><tr><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>0.41%</td><td rowspan=1 colspan=1>79.00%</td><td rowspan=1 colspan=1>7.00</td><td rowspan=1 colspan=1>58.49</td></tr><tr><td rowspan=4 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>8.73%</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>162.13</td></tr><tr><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>4.11%</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>196.47</td></tr><tr><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>0.19%</td><td rowspan=1 colspan=1>2.76%</td><td rowspan=1 colspan=1>6.92</td><td rowspan=1 colspan=1>58.37</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0.19%</td><td rowspan=1 colspan=1>75.11%</td><td rowspan=1 colspan=1>7.59</td><td rowspan=1 colspan=1>78.01</td></tr></table>

Table 19: Relative error and computational time (seconds) comparison between the classical method and the PINN approach for the Barenblatt solution with different values of m and initial guesses.

![](images/9a7c7ba424f41b6f43a3fd68b2a0dbfc839531a91dbd886ea6720f3ed90e4895.jpg)  
(a) fmincon results

![](images/e3115c93b7d05ef8014d0978a6a809139aacf9b5ca08bf44df81c147fda29e96.jpg)  
(b) PINN results  
Figure 10: Inverse Barenblatt solution results plots

As explained before, many classical cases failed, while the PINN was more stable and produced results for all cases. However, when the classical method did run successfully, it mainly gave more accurate results (smaller relative error). Still, the errors were generally considered accurate, being below 5% for all successful classical and PINN cases, with a few exceptions for the PINN.

After seeing these results, we noticed that the PINN relative error increased with m , and we wanted to understand why the errors for the last cases of $m = 4$ and $m = 5$ , with their corresponding largest initial guesses, gave such off results. So we repeated the runs, but this time we plotted the evolution of both the training loss and the approximation of m versus the epochs.

We noticed that the loss in all cases of m followed a similar trend, where we have a sudden decrease, then a plateau, and then a new decrease, as shown in the example case of $m = 3$ with the initial guess 4.5.

![](images/ba713bfdbc4f42afdcf0a6470a912ac0db94c69c853084c3598354fdafb65cdf.jpg)

![](images/52dbb06fee8295c541c5583fb1845143e1d8c9157d76eebd920c4c6bee218963.jpg)  
Figure 11: Training loss and m evolution for case m=3 with initial guess =4.5

However, we noticed that in the case of m = 4 with an initial guess of 7.5, the training stopped in the plateau region at 4000 epochs due to the early stopping function. We expected it to follow the same loss trend as the other cases. Therefore, when we removed early stopping, the error in this case decreased from 79% to 1.38%.

![](images/689d4e1a4cd7624f1b9cfa1bd721be833195d59017da2c735a65eac578ff698c.jpg)  
(a) with early stopping

![](images/48d00f54b7835bd4253efb361e2a325db6df859360447e4eb91e0b374d2be915.jpg)  
(b) without early stopping  
Figure 12: Training loss and m evolution for case m=4 with initial guess =7.5, before and after removing the early stopping function

m-independent solutions A first try to run the classical solver ended up with numerical errors and singularities and failed for all cases. Some updated values of u may become zero or negative, which causes instability when computing terms such as $u ^ { m - 1 }$ , this is because the bounds fmincon ensures that m-dependent solution The positivity of u in the classical solver was also forced in this case As shown in 22, the PINN model clearly performs differently depending on the nonlinearity parameter m. For $m = 2$ , almost all initial guesses give relative errors approximately within the 5% threshold, except for the case where the initial guess was 50% above the true value. As m increases, the errors grow significantly, reaching between 20% and 80% across the tested initial guesses. For $m = 5$ across all initial guesses, and for $m = 3$ and 4 with initial guesses that are +50% and +80% away from the true value, the relative errors are equal to the initial guess offset itself. This means the PINN did not learn anything and stayed stuck at its initial guess. This confirms that the model struggles as nonlinearity increases, learning best at $m = 2$ and worst at $m = 5$

the parameter m stays between 1.1 and 10, but it does not guarantee that the numerical solution u remains positive during Newton iterations. Therefore, to avoid this issue, we imposed a small lower bound using $u = \operatorname* { m a x } ( u , 1 0 ^ { - 1 2 } )$ , which keeps the solution positive and improves stability. Everything worked perfectly after this change as shown in Tables 20 and 21.
<table><tr><td rowspan=1 colspan=1>true m</td><td rowspan=1 colspan=1>initial guess</td><td rowspan=1 colspan=1>fmincon relative error</td><td rowspan=1 colspan=1>PINN relative error</td><td rowspan=1 colspan=1>fmincon time</td><td rowspan=1 colspan=1>PINN training time</td></tr><tr><td rowspan=4 colspan=1>2</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>1.10%</td><td rowspan=1 colspan=1>0.51%</td><td rowspan=1 colspan=1>30.05</td><td rowspan=1 colspan=1>360.64</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.10%</td><td rowspan=1 colspan=1>0.28%</td><td rowspan=1 colspan=1>29.46</td><td rowspan=1 colspan=1>356.92</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1.10%</td><td rowspan=1 colspan=1>0.15%</td><td rowspan=1 colspan=1>16.38</td><td rowspan=1 colspan=1>298.42</td></tr><tr><td rowspan=1 colspan=1>3.6</td><td rowspan=1 colspan=1>1.10%</td><td rowspan=1 colspan=1>0.22%</td><td rowspan=1 colspan=1>34.53</td><td rowspan=1 colspan=1>354.22</td></tr><tr><td rowspan=4 colspan=1>3</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>0.07%</td><td rowspan=1 colspan=1>0.14%</td><td rowspan=1 colspan=1>33.93</td><td rowspan=1 colspan=1>360.07</td></tr><tr><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>0.07%</td><td rowspan=1 colspan=1>0.20%</td><td rowspan=1 colspan=1>24.96</td><td rowspan=1 colspan=1>362.01</td></tr><tr><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>0.07%</td><td rowspan=1 colspan=1>0.13%</td><td rowspan=1 colspan=1>41.15</td><td rowspan=1 colspan=1>363.28</td></tr><tr><td rowspan=1 colspan=1>5.4</td><td rowspan=1 colspan=1>0.07%</td><td rowspan=1 colspan=1>0.11%</td><td rowspan=1 colspan=1>52.86</td><td rowspan=1 colspan=1>363.28</td></tr><tr><td rowspan=4 colspan=1>4</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>0.22%</td><td rowspan=1 colspan=1>0.54%</td><td rowspan=1 colspan=1>25.95</td><td rowspan=1 colspan=1>358.70</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.22%</td><td rowspan=1 colspan=1>0.25%</td><td rowspan=1 colspan=1>38.87</td><td rowspan=1 colspan=1>349.53</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0.22%</td><td rowspan=1 colspan=1>0.45%</td><td rowspan=1 colspan=1>39.00</td><td rowspan=1 colspan=1>360.74</td></tr><tr><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>0.22%</td><td rowspan=1 colspan=1>0.26%</td><td rowspan=1 colspan=1>35.97</td><td rowspan=1 colspan=1>359.90</td></tr><tr><td rowspan=4 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.14%</td><td rowspan=1 colspan=1>76.74%</td><td rowspan=1 colspan=1>35.50</td><td rowspan=1 colspan=1>143.21</td></tr><tr><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>0.14%</td><td rowspan=1 colspan=1>0.11%</td><td rowspan=1 colspan=1>63.15</td><td rowspan=1 colspan=1>298.59</td></tr><tr><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>0.14%</td><td rowspan=1 colspan=1>0.45%</td><td rowspan=1 colspan=1>35.70</td><td rowspan=1 colspan=1>356.80</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0.14%</td><td rowspan=1 colspan=1>0.31%</td><td rowspan=1 colspan=1>46.35</td><td rowspan=1 colspan=1>356.47</td></tr></table>

Table 20: Relative error and computational time (seconds) comparison between the classical method and the PINN approach for the polynomial solution with different values of m and initial guesses.

![](images/4633e9903ed2c2570190dfc98073c6bd1716c69a78abbef514e316a485c07439.jpg)  
(a) fmincon results

![](images/32b539b221949405c603d4e88989c2c40f9561ce4ddfa0dcaad5efb880017dd9.jpg)  
(b) PINN results  
Figure 13: Inverse Polynomial solution plots

<table><tr><td rowspan=1 colspan=1>true m</td><td rowspan=1 colspan=1>initial1 guess</td><td rowspan=1 colspan=1>fmincon relative error</td><td rowspan=1 colspan=1>PINN relative error</td><td rowspan=1 colspan=1>fmincon time</td><td rowspan=1 colspan=1>PINN training time</td></tr><tr><td rowspan=4 colspan=1>2</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>0.85%</td><td rowspan=1 colspan=1>0.94%</td><td rowspan=1 colspan=1>5.38</td><td rowspan=1 colspan=1>338.86</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.85%</td><td rowspan=1 colspan=1>1.37%</td><td rowspan=1 colspan=1>7.32</td><td rowspan=1 colspan=1>338.78</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.85%</td><td rowspan=1 colspan=1>0.15%</td><td rowspan=1 colspan=1>12.32</td><td rowspan=1 colspan=1>250.15</td></tr><tr><td rowspan=1 colspan=1>3.6</td><td rowspan=1 colspan=1>0.85%</td><td rowspan=1 colspan=1>0.11%</td><td rowspan=1 colspan=1>14.43</td><td rowspan=1 colspan=1>341.01</td></tr><tr><td rowspan=4 colspan=1>3</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>0.70%</td><td rowspan=1 colspan=1>0.31%</td><td rowspan=1 colspan=1>9.31</td><td rowspan=1 colspan=1>352.13</td></tr><tr><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>0.70%</td><td rowspan=1 colspan=1>0.33%</td><td rowspan=1 colspan=1>7.77</td><td rowspan=1 colspan=1>349.41</td></tr><tr><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>0.70%</td><td rowspan=1 colspan=1>0.12%</td><td rowspan=1 colspan=1>7.46</td><td rowspan=1 colspan=1>349.22</td></tr><tr><td rowspan=1 colspan=1>5.4</td><td rowspan=1 colspan=1>0.70%</td><td rowspan=1 colspan=1>0.12%</td><td rowspan=1 colspan=1>9.19</td><td rowspan=1 colspan=1>309.47</td></tr><tr><td rowspan=4 colspan=1>4</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>0.71%</td><td rowspan=1 colspan=1>0.20%</td><td rowspan=1 colspan=1>7.59</td><td rowspan=1 colspan=1>343.31</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.71%</td><td rowspan=1 colspan=1>0.21%</td><td rowspan=1 colspan=1>7.17</td><td rowspan=1 colspan=1>342.84</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0.71%</td><td rowspan=1 colspan=1>0.07%</td><td rowspan=1 colspan=1>7.63</td><td rowspan=1 colspan=1>284.96</td></tr><tr><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>0.71%</td><td rowspan=1 colspan=1>0.01%</td><td rowspan=1 colspan=1>7.61</td><td rowspan=1 colspan=1>343.86</td></tr><tr><td rowspan=4 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.80%</td><td rowspan=1 colspan=1>0.37%</td><td rowspan=1 colspan=1>8.58</td><td rowspan=1 colspan=1>344.33</td></tr><tr><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>0.80%</td><td rowspan=1 colspan=1>0.12%</td><td rowspan=1 colspan=1>7.57</td><td rowspan=1 colspan=1>345.70</td></tr><tr><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>0.80%</td><td rowspan=1 colspan=1>0.08%</td><td rowspan=1 colspan=1>7.58</td><td rowspan=1 colspan=1>343.78</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>0.17%</td><td rowspan=1 colspan=1>1.57</td><td rowspan=1 colspan=1>344.36</td></tr></table>

Table 21: Relative error and computational time (seconds) comparison between the classical method and the inverse PINN approach for the harmonic solution with different values of m and initial guesses

![](images/d5b6ad72982b28fc8dac3230d7f59fab40a7137ff701576c89c224f8e6dbf34e.jpg)  
(a) fmincon results

![](images/7e65f1bd23ec963763e6e32fbcd04ca1c80641ef2d5273e57a7da950fd61e2ce.jpg)  
(b) PINN results  
Figure 14: Inverse Harmonic solution plots

<table><tr><td rowspan=1 colspan=1>true m</td><td rowspan=1 colspan=1>initial1 guess</td><td rowspan=1 colspan=1>fmincon relative error</td><td rowspan=1 colspan=1>PINN relative error</td><td rowspan=1 colspan=1>fmincon time</td><td rowspan=1 colspan=1>PINN training time</td></tr><tr><td rowspan=4 colspan=1>2</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>0.08%</td><td rowspan=1 colspan=1>6.79%</td><td rowspan=1 colspan=1>107.29</td><td rowspan=1 colspan=1>379.69</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.08%</td><td rowspan=1 colspan=1>2.83%</td><td rowspan=1 colspan=1>110.53</td><td rowspan=1 colspan=1>382.94</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.08%</td><td rowspan=1 colspan=1>49.85%</td><td rowspan=1 colspan=1>71.56</td><td rowspan=1 colspan=1>266.39</td></tr><tr><td rowspan=1 colspan=1>3.6</td><td rowspan=1 colspan=1>0.08%</td><td rowspan=1 colspan=1>0.15%</td><td rowspan=1 colspan=1>55.11</td><td rowspan=1 colspan=1>366.76</td></tr><tr><td rowspan=4 colspan=1>3</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>9.49%</td><td rowspan=1 colspan=1>26.84</td><td rowspan=1 colspan=1>370.63</td></tr><tr><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>0.27%</td><td rowspan=1 colspan=1>29.66%</td><td rowspan=1 colspan=1>1248.56</td><td rowspan=1 colspan=1>376.23</td></tr><tr><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>51.31%</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>369.48</td></tr><tr><td rowspan=1 colspan=1>5.4</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>81.89%</td><td rowspan=1 colspan=1>12.34</td><td rowspan=1 colspan=1>369.26</td></tr><tr><td rowspan=4 colspan=1>4</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>22.32%</td><td rowspan=1 colspan=1>39.68</td><td rowspan=1 colspan=1>377.16</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>19.84%</td><td rowspan=1 colspan=1>27.14</td><td rowspan=1 colspan=1>370.12</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>50.46%</td><td rowspan=1 colspan=1>0.28</td><td rowspan=1 colspan=1>364.21</td></tr><tr><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>79.71%</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1>363.13</td></tr><tr><td rowspan=4 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>80.00%</td><td rowspan=1 colspan=1>9.76</td><td rowspan=1 colspan=1>179.46</td></tr><tr><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>50.00%</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>179.96</td></tr><tr><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>50.00%</td><td rowspan=1 colspan=1>4.73</td><td rowspan=1 colspan=1>177.85</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>failed</td><td rowspan=1 colspan=1>80.00%</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>178.40</td></tr></table>

Table 22: Relative error and computational time (seconds) comparison between the classical method and the PINN approach for the m\_dependent solution with different values of m and initial guesses.

![](images/12b5281cc5aa9100b1dd88b25d06dafc05968330362b352fc2bb566a6f0b71ee.jpg)  
(a) fmincon results

![](images/0db7d1310fa45eaeccd4f4059a48803f209cf13f6d8bf6a8390cd512a03da287.jpg)  
(b) PINN results  
Figure 15: Inverse m-dependent solution plots

Looking at the training loss evolution shown in Appendix B, it can be seen that in many cases either early stopping was triggered or the epochs ran out right when the loss was starting to go down. To check this, the number of epochs was increased from 10,000 to 20,000 for the case of $m = 2$ with an initial guess 50% above the true value, and the PINN kept learning, bringing the error down to 2.9% as shown in figure 16.

The loss trends for $m = 3$ and 4 with initial guesses of -50% and -80% suggest that more epochs could give better results in those cases as well. However, for cases where the loss does not improve at all and stays flat, other changes are needed, such as rescaling the problem, using the L-BFGS optimizer. or trying other improved training techniques.

![](images/634262dbb999eb6bafeb1dfccf23985f3bc3c1d73427dea836ac7efb27a906d0.jpg)  
(a) with early stopping  
(b) without early stopping  
Figure 16: Training loss and m evolution for case m=2 with initial guess =3, after increasing the epochs and removing the early stopping, for the m-dependent solution

On the other hand, the classical method was able to accurately estimate m only for $m = 2$ with all initial guesses and $m = 3$ with an initial guess of 1.5 (-50%), while failing to produce results for all other cases due to numerical instabilities and matrix singularities.

Overall, for the inverse problem, the PINN model proposed in this work achieved results comparable to the classical method in terms of accuracy across all tested cases. Even in the m-dependent solution case which was the most challenging for both approaches, the PINN was still able to produce results where the classical method completely failed due to numerical instabilities. While not all PINN results were accurate, the model still has room for improvement, as one can explore different optimization strategies, architectural changes, and rescaling tricks to push the results further. The classical method, on the other hand, offers no such flexibility.

## Time comparison

In this section, the computational times of the PINN approach and the classical method are compared for different test cases. The following plots (Figure 17) illustrate the execution time of each method and highlight their average performance.

Overall, the fmincon method is consistently faster than the PINN approach in all cases. Its average computational time ranges from about 7 s to 36 s depending on the solution, except for the m-dependent solution where it rises to around 107 s, while the PINN method requires roughly 300 s to 340 s on average.

At first glance, the substantial difference in computational time between the two approaches was unexpected. Indeed, the classical inverse solver relies on fmincon, which repeatedly solves the direct problem during the optimization process. Based on the direct problem experiments presented in Section 2.3, solving a single direct problem using the classical Newton method required between approximately 4 s and 24 s in the Python implementation, depending on the considered test case. For instance for the harmonic solution, a single direct solve required on average about 6 s.

Since the inverse optimization typically converges in approximately eight iterations and each iteration requires at least one direct solve, one would expect the overall computational time of the classical inverse method to be at least 48 s for the harmonic solution. However, the measured execution time in MATLAB was only about 7 s, indicating a significant discrepancy.

Time Comparison: Classical Method vs PINN  
![](images/047033cd0c0345ead0003902b674e90b478582d063f92743700e8946bd52a975.jpg)  
Polynomial Solution

Time Comparison: Classical Method vs PINN  
![](images/35e6081ac751f3d8d053c820b8e4dd1f3ffcb67dd4d57d32076ff863ad482cd4.jpg)

Time Comparison: Classical Method vs PINN  
![](images/2e7dfe8ffda49068c4fbaf40cf17a5435022ba3f3d7a4112c34b75331ec9bb99.jpg)

Time Comparison: Classical Method vs PINN  
![](images/b48d30d97b79047cdb9a2e67e1b67bb6d29fa3d140ae25e87923f23e63d59323.jpg)  
Figure 17: Computational time comparison of PINN and the classical method across m-dependent and mindependent solutions. Dashed lines show average times.

To investigate this observation, the direct classical solver was implemented and executed independently in MATLAB using the same numerical method and discretization parameters as in the Python implementation. The comparison revealed that MATLAB executes the direct solver considerably faster. For example, solving the harmonic direct problem for all tested values of m required 26.34 s in Python, whereas the corresponding MATLAB implementation required only 1.21 s. This comparison indicates that the large difference observed in the inverse problem timings is primarily due to the execution speed of the two programming environments rather than the optimization procedure itself. In particular, the Python implementation contains several nested loops that are executed much less efficiently than the corresponding MATLAB implementation, leading to a significantly higher computational cost despite both implementations employing the same numerical algorithm. Table 23 summarizes the computation times (in seconds) of the same classical method presented in 2.1.1 for the four manufactured solutions using MATLAB instead of Python for the direct problem.

Table 23: Computation time (seconds) for solving the direct problem using MATLAB for different manufactured solutions.
<table><tr><td rowspan=1 colspan=1>m</td><td rowspan=1 colspan=1>Barenblatt</td><td rowspan=1 colspan=1>Polynomial</td><td rowspan=1 colspan=1>Harmonic</td><td rowspan=1 colspan=1>m-dependent</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.078</td><td rowspan=1 colspan=1>0.087</td><td rowspan=1 colspan=1>0.182</td><td rowspan=1 colspan=1>0.134</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.040</td><td rowspan=1 colspan=1>0.053</td><td rowspan=1 colspan=1>0.033</td><td rowspan=1 colspan=1>0.054</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>0.388</td><td rowspan=1 colspan=1>0.529</td><td rowspan=1 colspan=1>0.496</td><td rowspan=2 colspan=1>0.9381.133</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>0.378</td><td rowspan=1 colspan=1>0.524</td><td rowspan=1 colspan=1>0.491</td></tr></table>

## 3.4.2 Inverse PINN testing with Noisy Data

Another aspect that should be investigated is the performance of the inverse PINN when the available solution data is not obtained from the exact solution, but rather from a perturbed dataset. In this section, the inverse PINN is tested under three different data scenarios. First, the measurement dataset is generated from the direct PINN solution, where the approximation error of the direct PINN introduces a deviation from the exact solution. Second, an additional 1% noise is added to the direct PINNgenerated dataset, and finally, the same test is performed with 5% noise. These experiments aim to evaluate the robustness of the inverse PINN when the available data becomes increasingly different from the exact solution. As in Section 2.4, this testing is performed only for the polynomial solution due to the range of relative errors obtained for different values of $m$ , which provides a suitable benchmark for testing the effect of data perturbations.

The noise is added to the measurement dataset as follows. Let $u _ { i } ^ { \mathrm { P I N N } }$ denote the value of the direct PINN solution at the measurement point $( t _ { i } , x _ { i } )$ , for $i = 1 , \ldots , N _ { \mathrm { m e a s } } ,$ where $N _ { \mathrm { m e a s } } = 2 2 5$ corresponds to the $1 5 \times 1 5$ grid described in Section 2.4. The perturbed dataset is obtained by adding an independent Gaussian perturbation to each measurement,

$$
u _ { i } ^ { \mathrm { o b s } } = u _ { i } ^ { \mathrm { P I N N } } + \varepsilon _ { i } , \qquad \varepsilon _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , \qquad i = 1 , \dots , N _ { \mathrm { m e a s } } ,\tag{8}
$$

where the standard deviation $\sigma$ is chosen relative to the magnitude of the data itself,

$$
\sigma = \delta \sqrt { \frac { 1 } { N _ { \mathrm { m e a s } } } \sum _ { i = 1 } ^ { N _ { \mathrm { m e a s } } } \left( u _ { i } ^ { \mathrm { P I N N } } \right) ^ { 2 } } ,\tag{9}
$$

and $\delta$ is the prescribed noise level, taken here as $\delta = 0 . 0 1$ and $\delta = 0 . 0 5$ , corresponding to 1% and 5% noise respectively.

Two remarks are in order regarding this choice. First, the perturbation is additive rather than multiplicative. Since the polynomial solution $u ( t , x ) = ( 1 - t ) ^ { 2 } ( 1 - x ^ { 2 } ) ^ { 2 }$ vanishes at $x = \pm 1$ and at t = 1, a perturbation proportional to the local value of the solution would leave these measurements unperturbed, which does not reflect the behaviour of a physical measurement device. Scaling σ by the root mean square of the data instead assigns the same absolute perturbation to every measurement point, independently of the local magnitude of the solution. Second, the noise is applied only to the measurement dataset; the initial and boundary conditions are kept exact, as these are assumed to be known.

The random perturbations are generated with a fixed seed, so that the same realisation of the noise is used across all runs and the results reported below are reproducible. The remaining components of the inverse problem, namely the network architecture, the optimizer, the loss weights and the number of training epochs, are left unchanged from Section 3.2, so that any variation in the recovered value of m can be attributed to the perturbation of the data alone.The parameters used for the data perturbation are summarized in Table 24.

Table 24: Parameters of the data perturbation used in the inverse PINN experiments.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Noise model</td><td>Additive Gaussian,  $\overline { { \varepsilon _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) } }$ </td></tr><tr><td>Noise levels δ</td><td>0%, 1%, 5%</td></tr><tr><td>Standard deviation σ</td><td> $\delta \cdot \mathrm { R M S } \left( u ^ { \mathrm { P I N N } } \right)$ </td></tr><tr><td>Perturbed quantity</td><td>Measurement dataset only (from direct PINN)</td></tr><tr><td>Number of perturbed points Random seed</td><td> $N _ { \mathrm { m e a s } } = 2 2 5$  42</td></tr></table>

The results obtained for the three data scenarios are reported in Tables 25, 26, and 27

Table 25: PINN inverse results for the polynomial solution using measurement data generated from the direct PINN with 0% additional noise
<table><tr><td rowspan=1 colspan=1>true m</td><td rowspan=1 colspan=1>initialguess</td><td rowspan=1 colspan=1>recovered m</td><td rowspan=1 colspan=1>PINN relative error</td><td rowspan=1 colspan=1>PINN training time s</td></tr><tr><td rowspan=4 colspan=1>2</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>2.00</td><td rowspan=1 colspan=1>0.22%</td><td rowspan=1 colspan=1>338.66</td></tr><tr><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>2.00</td><td rowspan=1 colspan=1>0.01%</td><td rowspan=1 colspan=1>285.76</td></tr><tr><td rowspan=1 colspan=1>3.0</td><td rowspan=1 colspan=1>2.00</td><td rowspan=1 colspan=1>0.30%</td><td rowspan=1 colspan=1>224.85</td></tr><tr><td rowspan=1 colspan=1>3.6</td><td rowspan=1 colspan=1>2.00</td><td rowspan=1 colspan=1>0.35%</td><td rowspan=1 colspan=1>244.06</td></tr><tr><td rowspan=4 colspan=1>3</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=1>0.28%</td><td rowspan=1 colspan=1>341.31</td></tr><tr><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=1>0.19%</td><td rowspan=1 colspan=1>337.40</td></tr><tr><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=1>0.16%</td><td rowspan=1 colspan=1>310.95</td></tr><tr><td rowspan=1 colspan=1>5.4</td><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=1>0.09%</td><td rowspan=1 colspan=1>310.50</td></tr><tr><td rowspan=4 colspan=1>4</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>4.02</td><td rowspan=1 colspan=1>0.56%</td><td rowspan=1 colspan=1>343.50</td></tr><tr><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>4.02</td><td rowspan=1 colspan=1>0.53%</td><td rowspan=1 colspan=1>342.98</td></tr><tr><td rowspan=1 colspan=1>6.0</td><td rowspan=1 colspan=1>4.02</td><td rowspan=1 colspan=1>0.53%</td><td rowspan=1 colspan=1>346.10</td></tr><tr><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>4.01</td><td rowspan=1 colspan=1>0.37%</td><td rowspan=1 colspan=1>344.70</td></tr><tr><td rowspan=4 colspan=1>5</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>1.14</td><td rowspan=1 colspan=1>77.17%</td><td rowspan=1 colspan=1>136.88</td></tr><tr><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>4.99</td><td rowspan=1 colspan=1>0.02%</td><td rowspan=1 colspan=1>340.89</td></tr><tr><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>5.02</td><td rowspan=1 colspan=1>0.51%</td><td rowspan=1 colspan=1>343.42</td></tr><tr><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=1>5.02</td><td rowspan=1 colspan=1>0.42%</td><td rowspan=1 colspan=1>343.97</td></tr></table>

Table 26: PINN inverse results for the polynomial solution using measurement data generated from the direct PINN with 1% additional noise
<table><tr><td rowspan=1 colspan=1>true m</td><td rowspan=1 colspan=1>initialguess</td><td rowspan=1 colspan=1>recovered m</td><td rowspan=1 colspan=1>PINN relative error pct</td><td rowspan=1 colspan=1>PINN training time s</td></tr><tr><td rowspan=4 colspan=1>2</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>2.00</td><td rowspan=1 colspan=1>0.04%</td><td rowspan=1 colspan=1>355.85</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.99</td><td rowspan=1 colspan=1>0.22%</td><td rowspan=1 colspan=1>254.72</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1.99</td><td rowspan=1 colspan=1>0.02%</td><td rowspan=1 colspan=1>201.63</td></tr><tr><td rowspan=1 colspan=1>3.6</td><td rowspan=1 colspan=1>2.00</td><td rowspan=1 colspan=1>0.40%</td><td rowspan=1 colspan=1>223.58</td></tr><tr><td rowspan=4 colspan=1>3</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=1>0.04%</td><td rowspan=1 colspan=1>359.60</td></tr><tr><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=1>0.07%</td><td rowspan=1 colspan=1>360.32</td></tr><tr><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=1>0.12%</td><td rowspan=1 colspan=1>227.98</td></tr><tr><td rowspan=1 colspan=1>5.4</td><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=1>0.02%</td><td rowspan=1 colspan=1>233.90</td></tr><tr><td rowspan=4 colspan=1>4</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>4.01</td><td rowspan=1 colspan=1>0.41%</td><td rowspan=1 colspan=1>356.59</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4.01</td><td rowspan=1 colspan=1>0.30%</td><td rowspan=1 colspan=1>329.19</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>4.01</td><td rowspan=1 colspan=1>0.40%</td><td rowspan=1 colspan=1>246.92</td></tr><tr><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>4.00</td><td rowspan=1 colspan=1>0.24%</td><td rowspan=1 colspan=1>326.11</td></tr><tr><td rowspan=4 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.15</td><td rowspan=1 colspan=1>76.96%</td><td rowspan=1 colspan=1>141.84</td></tr><tr><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>4.99</td><td rowspan=1 colspan=1>0.13%</td><td rowspan=1 colspan=1>354.21</td></tr><tr><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>5.01</td><td rowspan=1 colspan=1>0.39%</td><td rowspan=1 colspan=1>324.37</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>5.01</td><td rowspan=1 colspan=1>0.34%</td><td rowspan=1 colspan=1>324.97</td></tr></table>

Table 27: PINN inverse results for the polynomial solution using measurement data generated from the direct PINN with 5% additional noise
<table><tr><td rowspan=1 colspan=1>true m</td><td rowspan=1 colspan=1>initialguess</td><td rowspan=1 colspan=1>recovered m</td><td rowspan=1 colspan=1>PINN relative error</td><td rowspan=1 colspan=1>PINN training time s</td></tr><tr><td rowspan=4 colspan=1>2</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>1.98</td><td rowspan=1 colspan=1>0.58%</td><td rowspan=1 colspan=1>199.68</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.99</td><td rowspan=1 colspan=1>0.38%</td><td rowspan=1 colspan=1>178.59</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1.99</td><td rowspan=1 colspan=1>0.39%</td><td rowspan=1 colspan=1>165.93</td></tr><tr><td rowspan=1 colspan=1>3.6</td><td rowspan=1 colspan=1>1.99</td><td rowspan=1 colspan=1>0.16%</td><td rowspan=1 colspan=1>176.49</td></tr><tr><td rowspan=4 colspan=1>3</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>2.96</td><td rowspan=1 colspan=1>1.27%</td><td rowspan=1 colspan=1>210.94</td></tr><tr><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>2.96</td><td rowspan=1 colspan=1>1.20%</td><td rowspan=1 colspan=1>197.06</td></tr><tr><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>2.99</td><td rowspan=1 colspan=1>0.27%</td><td rowspan=1 colspan=1>172.51</td></tr><tr><td rowspan=1 colspan=1>5.4</td><td rowspan=1 colspan=1>2.99</td><td rowspan=1 colspan=1>0.25%</td><td rowspan=1 colspan=1>176.41</td></tr><tr><td rowspan=4 colspan=1>4</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>3.98</td><td rowspan=1 colspan=1>0.49%</td><td rowspan=1 colspan=1>222.55</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3.99</td><td rowspan=1 colspan=1>0.12%</td><td rowspan=1 colspan=1>191.97</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>4.00</td><td rowspan=1 colspan=1>0.14%</td><td rowspan=1 colspan=1>171.97</td></tr><tr><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>4.00</td><td rowspan=1 colspan=1>0.12%</td><td rowspan=1 colspan=1>177.12</td></tr><tr><td rowspan=4 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.17</td><td rowspan=1 colspan=1>76.48%</td><td rowspan=1 colspan=1>142.86</td></tr><tr><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>4.98</td><td rowspan=1 colspan=1>0.37%</td><td rowspan=1 colspan=1>205.75</td></tr><tr><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>5.03</td><td rowspan=1 colspan=1>0.61%</td><td rowspan=1 colspan=1>164.99</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>5.02</td><td rowspan=1 colspan=1>0.56%</td><td rowspan=1 colspan=1>173.50</td></tr></table>

Overall, across all three data scenarios, the recovered value of m remains accurate, with the relative error not exceeding 1.5% in every case. The only exception is the case where the true m is 5 and the initial guess is 1, for which the error stays between 76% and 77% across all perturbation levels. This case, however, already fails in the noise-free scenario, where the error is 76.74%; and since the noisy datasets are obtained from the same direct PINN solution, this behaviour is carried over to the 1% and

5% cases as well. Apart from this case, the recovered m remains stable as the noise level increases, which indicates that the inverse PINN is indeed robust to perturbations in the measurement data.

## 4 Conclusion

In this paper, we developed a PINN framework for solving the inverse one-dimensional Porous Medium Equation. The main objective was to establish a robust inverse PINN methodology and provide a strong foundation for future extensions to higher-dimensional PME problem. To achieve this, we investigated the factors affecting inverse PINN performance and proposed a novel two-stage learning framework to improve convergence robustness and parameter recovery.

The results showed that the proposed framework successfully solves both the direct and inverse PME problems and produces accurate solutions across different manufactured solutions and values of m . In particular, the proposed two-stage strategy improves the robustness of the inverse problem and reduces the dependence on the initial parameter guess.

Future work will focus on extending the framework to two-dimensional and higher-dimensional PME problems, including Barenblatt solutions and solutions generated using finite difference or other numerical methods for validation, comparison, and data collection. At this point, one clear advantage of the proposed PINN framework is that these high-dimensional extensions require only minor modifications to the existing code, mainly through adding extra spatial coordinates to the network inputs, updating the PDE residual, and sampling collocation points in the new domains. Overall, the developed framework provides a solid foundation for future studies of more complex PME problems, where the advantages of PINNs are expected to become more significant.

## References

[1] Moufawad, S. M., Nassif, N. R., and Triki, F. Direct and Inverse Problem for Gas Diffusion in Polar Firn. Communications on Analysis and Computation, 2024, 2(1): 71-109. doi: 10.3934/cac.2024005

[2] Al Helwani, N. M., Moufawad, S. M., Sakr, G. C. . "Solving Inverse PDE Problems using Minimization Methods and AI". arXiv preprint, arXiv:2603.01731, 2023. https://arxiv.org/abs/ 2603.01731

[3] Caro, P., and Ruiz, A. “An inverse problem for data-driven prediction in quantum mechanics". arXiv preprint, arXiv:2302.10553, 2023. https://arxiv.org/abs/2302.10553

[4] J. L. Vázquez, The Porous Medium Equation: Mathematical Theory, Oxford Mathematical Monographs, Clarendon Press / Oxford University Press, 2007. Available: https://doi.org/10.1093/ acprof:oso/9780198569039.001.0001

[5] A. Gholami, A. Mang, and G. Biros, An inverse problem formulation for parameter estimation of a reaction-diffusion model of low grade gliomas, Journal of Mathematical Biology, vol. 72, pp. 409-433, 2016. Available: https://doi.org/10.1007/s00285-015-0888-x

[6] D. P. Kingma and J. Ba, Adam: A Method for Stochastic Optimization, International Conference on Learning Representations (ICLR), 2015. Available: https://arxiv.org/abs/1412.6980

[7] D. C. Liu and J. Nocedal, On the limited memory BFGS method for large scale optimization, Mathematical Programming, vol. 45, no. 1, pp. 503–528, 1989. doi: 10.1007/BF01589116

[8] M. Raissi, P. Perdikaris, and G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics, vol. 378, pp. 686–707, 2019. doi: 10.1016/j.jcp.2018.10.045

[9] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," Advances in Neural Information Processing Systems, vol. 25, pp. 1097–1105, 2012.

[10] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," Nature, vol. 521, no. 7553, pp. 436–444, 2015. doi:10.1038/nature14539.

[11] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics Informed Deep Learning (Part I): Data-Driven Solutions of Nonlinear Partial Differential Equations," arXiv preprint arXiv:1711.10561, 2017. https://arxiv.org/abs/1711.10561.

[12] K. Hornik, Approximation capabilities of multilayer feedforward networks, Neural Networks, vol. 4, no. 2, pp. 251–257, 1991. doi: 10.1016/0893-6080(91)90009-T

[13] Paszke, A., Gross, S., Massa, F., et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library". Advances in Neural Information Processing Systems, Vol. 32, 2019. https: //arxiv.org/abs/1912.01703

[14] A. G. Baydin, B. A. Pearlmutter, A. A. Radul, and J. M. Siskind, “Automatic differentiation in machine learning: a survey," Journal of Machine Learning Research, vol. 18, pp. 1–43, 2018. https://www.jmlr.org/papers/v18/17-468.html.

[15] A. J. Meade, Jr. and A. A. Fernandez, The numerical solution of linear ordinary differential equations by feedforward neural networks, Mathematical and Computer Modelling, vol. 19, no. 12, pp. 1– 25, 1994. Available: https://doi.org/10.1016/0895-7177(94)90095-7

[16] B. Ph. van Milligen, V. Tribaldos, and J. A. Jiménez, Neural network differential equation and plasma equilibrium solver, Physical Review Letters, vol. 75, pp. 3594–3597, 1995. Available: https: //doi.org/10.1103/PhysRevLett.75.3594

[17] I. E. Lagaris, A. Likas, and D. I. Fotiadis, Artificial neural networks for solving ordinary and partial differential equations, IEEE Transactions on Neural Networks, vol. 9, no. 5, pp. 987–1000, Sept. 1998. Available: https://doi.org/10.1109/72.712178

[18] I. E. Lagaris, A. C. Likas, and D. G. Papageorgiou, Neural-network methods for boundary value problems with irregular boundaries, IEEE Transactions on Neural Networks, vol. 11, no. 5, pp. 1041– 1049, 2000. Available: https://doi.org/10.1109/72.870037

[19] N. Zobeiry and K. D. Humfeld, A physics-informed machine learning approach for solving heat transfer equation in advanced manufacturing and engineering applications, Engineering Applications of Artificial Intelligence, vol. 101, p. 104232, 2021. Available: https://doi.org/10.1016/j. engappai.2021.104232

[20] S. Mishra and R. Molinaro, Physics informed neural networks for simulating radiative transfer, Journal of Quantitative Spectroscopy and Radiative Transfer, vol. 270, p. 107705, 2021. Available: https://doi.org/10.1016/j.jqsrt.2021.107705

[21] Q. Zhu, Z. Liu, and J. Yan, Machine learning for metal additive manufacturing: predicting temperature and melt pool fluid dynamics using physics-informed neural networks, Computational Mechanics, vol. 67, pp. 619–635, 2021. Available: https://doi.org/10.1007/s00466-020-01952-9

[22] L. Lu, X. Meng, Z. Mao, and G. E. Karniadakis, DeepXDE: A deep learning library for solving differential equations, SIAM Review, vol. 63, no. 1, pp. 208–228, 2021. Available: https://doi. org/10.1137/19M1274067

[23] F. Chen, D. Sondak, P. Protopapas, M. Mattheakis, S. Liu, D. Agarwal, and M. D. Giovanni, NeuroDiffEq: A Python package for solving differential equations with neural networks, Journal of Open Source Software, vol. 5, no. 46, p. 1931, 2020. Available: https://doi.org/10.21105/joss. 01931

[24] K. Shukla, A. K. Dwivedi, P. B. Nair, and S. T. Wereley, Parallel physics-informed neural networks via domain decomposition, Journal of Computational Physics, vol. 447, no. C, Sep. 2021, 110683. Available: https://doi.org/10.1016/j.jcp.2021.110683

[25] B. Moseley, A. Markham, and T. Nissen-Meyer, Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations, Advances in Computational Mathematics, vol. 49, p. 62, 2023. Available: https://doi.org/10.1007/ s10444-023-10065-9

[26] A. D. Jagtap, K. Kawaguchi, and G. E. Karniadakis, Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks, Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, vol. 476, p. 20200334, 2020. Available: https: //doi.org/10.1098/rspa.2020.0334

[27] E. Seiler, W. Lei, and P. Protopapas, Stiff Transfer Learning for Physics-Informed Neural Networks, arXiv preprint arXiv:2501.17281, 2025. Available: https://arxiv.org/abs/2501.17281

[28] Y. Wang, J. Bai, M. S. Eshaghi, C. Anitescu, X. Zhuang, T. Rabczuk, and Y. Liu, Transfer Learning in Physics-Informed Neural Networks: Full Fine-Tuning, Lightweight Fine-Tuning, and Low-Rank Adaptation, arXiv preprint arXiv:2502.00782, 2025. Available: https://arxiv.org/abs/2502. 00782

[29] M. Berardi et al., Inverse Physics-Informed Neural Networks for transport models in porous materials, Computer Methods in Applied Mechanics and Engineering, vol. 435, p. 117628, 2025. Available: https://doi.org/10.1016/j.cma.2024.117628

[30] N. Sukumar and A. Srivastava, Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks, Computer Methods in Applied Mechanics and Engineering, vol. 389, Article 114333, 2022.

[31] Z. Zhou, L. Wang, and Z. Yan, Deep neural networks learning forward and inverse problems of two-dimensional nonlinear wave equations with rational solitons, Computers & Mathematics with Applications, vol. 151, pp. 164–171, 2023.

[32] H. Gao, L. Sun, and J.-X. Wang, PhyGeoNet: physics-informed geometry-adaptive convolutional neural networks for solving parameterized steady-state PDEs on irregular domain, Journal of Computational Physics, vol. 428, p. 110079, 2021. Available: https://doi.org/10.1016/j.jcp.2020. 110079

[33] Chew, J. V. L., Sulaiman, J., Sunarto, A. "Solving One-Dimensional Porous Medium Equation Using Unconditionally Stable Half-Sweep Finite Difference and SOR Method". Mathematics and Statistics, Vol. 9, No. 2, pp. 166–171, 2021. DOI: 10.13189/ms.2021.090211. http://www .hrpub.org

[34] Blessing, L., Barth, A. “An Inverse Boundary Value Problem for the Porous Medium Equation: Numerical Methods". GAMM Archive for Students, Vol. 7, No. 1, 2025. DOI: 10.14464/gammas.v7i1.907. https://doi.org/10.14464/gammas.v7i1.907

[35] Cârstea, C. I., Ghosh, T., Uhlmann, G. “An Inverse Problem for the Porous Medium Equation with Partial Data and a Possibly Singular Absorption Term". arXiv preprint, arXiv:2108.12970v2, 2021. https://arxiv.org/abs/2108.12970

[36] Chew, J. V. L., Sulaiman, J. “On Quarter-Sweep Finite Difference Scheme for One-Dimensional Porous Medium Equations". International Journal of Applied Mathematics, Vol. 33, No. 3, pp. 439–450, 2020. DOI: 10.12732/ijam.v33i3.6.

[37] Duan, C., Liu, C., Wang, C., Yue, X. "Numerical Methods for Porous Medium Equation by an Energetic Variational Approach". arXiv preprint, arXiv:1806.10775v2, 2018. https://arxiv.org/ abs/1806.10775

[38] Chew, J. V. L., Sulaiman, J. "Implicit Solution of 1D Nonlinear Porous Medium Equation Using the Four-Point Newton-EGMSOR Iterative Method". Journal of Applied Mathematics and Computational Mechanics, Vol. 15, No. 2, pp. 11–21, 2016. DOI: 10.17512/jamcm.2016.2.02. http://www.amcm.pcz.pl

[39] M. Berardi, F. V. Difonzo, and R. Guglielmi, A preliminary model for optimal control of moisture content in unsaturated soils, Computers & Geosciences, vol. 27, no. 6, pp. 1133–1144, 2023.

[40] M. A. Celia, E. T. Bouloutas, and R. L. Zarba, A general mass-conservative numerical solution for the unsaturated flow equation, Water Resources Research, vol. 26, no. 7, pp. 1483–1496, 1990.

[41] G. Marinoschi, Functional Approach to Nonlinear Models of Water Flow in Soils, Springer, Dordrecht, The Netherlands, 2006. ISBN: 1-4020-4879-3.

[42] F. Wein, N. Chen, N. Iqbal, M. Stingl, and M. Avila, Topology optimization of unsaturated flows in multi-material porous media: Application to a simple diaper model, Communications in Nonlinear Science and Numerical Simulation, vol. 78, Article 104871, 2019.

[43] S. Lin and Y. Chen, “A two-stage physics-informed neural network method based on conserved quantities and applications in localized wave solutions," Journal of Computational Physics, vol. 457, p. 111053, 2022. doi: 10.1016/j.jcp.2022.111053.

[44] S. Shi, D. Liu, R. Ji, and Y. Han, “An Adaptive Physics-Informed Neural Network with Two-Stage Learning Strategy to Solve Partial Differential Equations," Numerical Mathematics: Theory, Methods and Applications, vol. 16, no. 2, pp. 298–322, 2023. doi: 10.4208/nmtma.OA-2022-0063

[45] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, "Curriculum learning," in Proceedings of the 26th Annual International Conference on Machine Learning (ICML '09), ACM, 2009, pp. 41–48. doi:10.1145/1553374.1553380.

[46] M.-H. Chen, Y.-A. Lee, F.-T. Liao, and D.-S. Shiu, "Rethinking the shape convention of an MLP," arXiv preprint arXiv:2510.01796, 2025. Available: https://arxiv.org/abs/2510.01796.

[47] C. Duffy and G. V. Velikova, “Dynamic Curriculum Regularization for Enhanced Training of Physics-Informed Neural Networks," in NeurIPS 2024 Workshop on Machine Learning and the Physical Sciences (ML4PS), poster presentation, Vancouver, Canada, Dec. 2024. Available: https://neurips.cc/virtual/2024/100067.

[48] Y. W. Bekele, "Physics-informed neural networks with curriculum training for poroelastic flow and deformation processes,"arXiv:2404.13909, 2024. Available: https://arxiv.org/abs/2404.13909.

[49] X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks," in Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS 2010), PMLR, vol. 9, 2010, pp. 249–256. Available: https://proceedings.mlr.press/v9/glorot10a.html.

[50] Google Research, Google Colaboratory, Available: https://colab.research.google.com

## A Visual Comparisons of Direct Solutions

For each Solution and m, we visualize the heatmap of the PINN, exact, and classical solution, along with the evolution of the PINN training loss

## A.1 Barenblatt solution

![](images/12a978fa27d3e755155595e1389606669ed781e8629f4eae35fae23f9adc621e.jpg)  
(a) PINN solution

![](images/3192488969247ce8efbe044cc5a7e81111b48221891014a3adc0c4b9186633c4.jpg)  
(b) Exact solution

![](images/2ecd614ec02b0ecacd14bc683e011191cbdf749e59b1b96e4e4931e50d884aac.jpg)  
(c) Classical solution

![](images/80879206a4d9e0c5dbc1cdbb4b526f80ce50811c3dd4a0e64bb8d5d25301a7a3.jpg)  
Figure 19: Zoomed-in view of the exact Barenblatt solution for m = 2 for $\mathrm { t } \in [ 0 , 0 . 3 ]$

![](images/13b014430dfcb2799146c8aba4c4e8d7de89b7c50038d3440c3382f3ae4543cb.jpg)  
(a) PINN ADAM loss

![](images/ff61b063906b5030371a899e7cafdbba9e55c2bb060156d9c50f0dc69d019724.jpg)  
(b) PINN LBFGS loss

$$
\mathbf { m } = 3
$$

![](images/e050de01ad842edc815002cca739fc24e397b6d4fe33d59df4f38114fd6446bf.jpg)  
(a) PINN solution

![](images/682ab6a590e1da0613b4263b2a449b9018743d4590703589a0b9b1c91c492f78.jpg)  
(b) Exact solution

![](images/627e22ef42457f560bb13dbf844ab13d23eaa92cb00f57963d60396af2a230aa.jpg)  
(c) Classical solution

![](images/5c934ed6922fa3a645a1f37f746f48fca59a5fe3fc969f641f7ed38da424f1c0.jpg)  
Figure 22: Zoomed-in view of the exact Barenblatt solution for $m = 3$ for $\mathrm { t } \in [ 0 , 0 . 3 ]$

![](images/77813c10725363e4fae5f89b9d5ed3868915da7692d19b820360224fb008aa01.jpg)  
(a) PINN ADAM loss

![](images/7304341f72305424559687a704f37e4d7e912a5ba5bbde712c034d2093a78dfe.jpg)  
(b) PINN LBFGS loss

$$
\mathbf { m } = 4
$$

![](images/8b9543fa6622b8c7369f9e06d1fac9e3700ab80e48511bbe825eddee36b29336.jpg)  
(a) PINN solution

![](images/8a027af44e4b3f6be3814cf1ba8cbcc910cde0e0e08db91a0f611de0b01f56b4.jpg)  
(b) Exact solution

![](images/5e831b697a88fed9a770ca9e7a9c990307fa47db4f4ece923227a3f77bff4195.jpg)  
(c) Classical solution

![](images/605900119b55ad67395ca68c79dbfe317a8480ac7354f1598e7ee35e9986a97d.jpg)  
Figure 25: Zoomed-in view of the exact Barenblatt solution for $m = 4$ for $\mathrm { t } \in [ 0 , 0 . 3 ]$

![](images/680467beb117b1cf64128e485757f8f7abba3eb63751b3c6a220c83647541b63.jpg)  
(a) PINN ADAM loss

![](images/db9dda97f75de7d36b335cf235a24245b5e3cf0ee1223ad55e32c9a63a5ae763.jpg)  
(b) PINN LBFGS loss

$$
\mathbf { m } = 5
$$

![](images/27048c036f4fe2b01f7ca7ef739e9dcda7373ccc139f50ebec63316fccde6746.jpg)  
(a) PINN solution

![](images/78ba3993be72d3c395a78dba0c82c1728fae309b0465af46f0e2d37d9d7b6cd0.jpg)  
(b) Exact solution

![](images/4839aaeff70f1a52643180ce86cc2ca020ef4f029dd44124837f38381e63a835.jpg)  
(c) Classical solution

![](images/1e3f0c9008a47a672a676c20a7c7bb0b67c1152cdf3f10dda7e0a32eec94d2fb.jpg)  
Figure 28: Zoomed-in view of the exact Barenblatt solution for $m = 5$ for $\mathrm { t } \in [ 0 , 0 . 3 ]$

![](images/f60d118d7685391dabfa3a8399ab1342cbf83b32086777f36611409a8a3c50ac.jpg)  
(a) PINN ADAM loss

![](images/302159c1b617acb70bda3ba1ef564edac27293b1667740ed30bdc0cc192a727f.jpg)  
(b) PINN LBFGS loss

## A.2 Polynomial solution

m = 2  
![](images/c9ab2d5d17cfc92f58d400a080250e8f3cc326029e8865487d6f80008e8c9f6d.jpg)  
(a) PINN solution

![](images/1db06eb32b2670f1070b01753ecd7e798046d260994e66aff892191d744dba3d.jpg)  
(b) Exact solution

![](images/177b548ceec814cc92b91588e05812df40069544f88fcb62499ef27db8fe06dd.jpg)  
(c) Classical solution

![](images/4e3c6c9eeb3c345a5ec77d469cba66c549ad981ea9d2ec266a045d4cfe9a4ce4.jpg)  
(a) PINN ADAM loss

![](images/3ace2c1945b8d9e1fb7362cf517908d40847c77e570e2f7e2af423a15555746a.jpg)  
(b) PINN LBFGS loss

m = 3  
![](images/258bc743dfcdbe4c84e43d006ab045e144591bcb0a102e5b58380f51c2227a6a.jpg)  
(a) PINN solution

![](images/4b688f7b565f036c01005fe6afd52b0117b0929bd84859a9de03dfd029ee1581.jpg)  
(b) Exact solution

![](images/46c7689b00a4ffea4f13376aba56eae4d7d78c484e4f10794a2cb7dabba861f6.jpg)  
(c) Classical solution

![](images/120bdc61d367d7530a9ea39afa61464401afd78686674cd54bf2925234c73c5f.jpg)  
(a) PINN ADAM loss

![](images/067d270a489b292edfa0c1f3147dbeb7343ce399805faeaa0c2cc8669f82c528.jpg)  
(b) PINN LBFGS loss

m = 4  
![](images/2cd96c46d91551bab24a4e90722318322dcc0d2695fc41aee27181eda850ad46.jpg)  
(a) PINN solution

![](images/91fb1d3f41f45b0e4b72b331d13c664b4b893acfa40d20ad526ad20cd1c5ddcc.jpg)  
(b) Exact solution

![](images/7339fb021739419b38d1e1ecb4dd981d509a1c3b2bea28fcdda314694dc5960c.jpg)  
(c) Classical solution

![](images/d6ab23e9b248ec226baa2d162494a217eaff0456ee8398b33632e2ff5ff7e000.jpg)  
(a) PINN ADAM loss

![](images/e3dbcefd1c283a5615c372d9352d0eba3cd16d0774eac3c1b061fb8bfe3171b1.jpg)  
(b) PINN LBFGS loss

![](images/1dd0d77f02511362e34d0d6264d44fe713f04d5b320ad6fbbe8b30d7c0102bcf.jpg)  
(a) PINN solution

![](images/4bd163b35fbc3e6796f573b9ac7511fbca9eb565afa63419789adeb1e802d7f4.jpg)  
(b) Exact solution

![](images/76c81985c27e97a37cf069e1023a26dcbba1cb5cc4abf30938b7df3bbac9f1a4.jpg)  
(c) Classical solution

![](images/0803bef6a5b19c3a3c6b21b2e5a90e7bc8447759a0a8980b324690bc8dabb1ee.jpg)  
(a) PINN ADAM loss

![](images/feb4f6e629bd9a61ec4dfe74e406406f7f9a8f39f98d17e0f8fb73810abe9e44.jpg)  
(b) PINN LBFGS loss

A.3 Damped Harmonic Solution  
m = 2  
![](images/63f146361f8a077d281a553b70d4405ee5a238298c9c8c67eaebe96f3669c4c5.jpg)  
(a) PINN solution

![](images/16bb731ccb53cc650f54d18161f690e63e192be01e15faf5109e4bbec7b7ef4e.jpg)  
(b) Exact solution

![](images/d1a917270715b40ad8962bf424e58f3ac9d41d97f773a91d1220db6dcb4cae55.jpg)  
(c) Classical solution

![](images/7175133819bc181e39cc69e2c43309995e290c638e8a98f99137ae06d6a6425d.jpg)  
(a) PINN ADAM loss

![](images/2292299f47e8848ae9be22c169d3a890cbbbf2b20b45ae5b7543d4bbebb2f31f.jpg)  
(b) PINN LBFGS loss

m = 3  
![](images/29b467223c9a1989eeb476104c0afe3cecd1dc0e2d6417befaf1da428d9af550.jpg)  
(a) PINN solution

![](images/7499a07b9763e463daabf674e739a65466aaded1283d00c04bc5fde50b87a570.jpg)  
(b) Exact solution

![](images/df91f121f21f34d89e3e5a5e70051700b61dd36a8763f102ee05f066d9006488.jpg)  
(c) Classical solution

![](images/6bcc6012aa85f4587b09a5529258c2b5d725c2d27cd93117f4ff725577754d1c.jpg)  
(a) PINN ADAM loss

![](images/69a2ca317f61448e8e206ff31a17bd20802037af5101652b6929d0c00c2d993d.jpg)  
(b) PINN LBFGS loss

![](images/1755112aacf12cbeed9c7b6e0243675eea72f8857431256d609ff014c68cf349.jpg)  
(a) PINN solution

![](images/414468c85389fd1187d0583c7c4ba5bc4a9a9ef87c4819776ef2f6241c7be15f.jpg)  
(b) Exact solution

![](images/60998454d118cc1bb9f183b090bcbeb12c52cd8b3ea58124e3ac3169c07e0500.jpg)  
(c) Classical solution

![](images/fce3e80f551016960b2f06865b7a540f9573638c073c3a3e315b0ac06cce2a2f.jpg)  
(a) PINN ADAM loss

![](images/fcf3b0d9526cf39ae048f9d230f3e3f444ecdab229c81a91f8adec04cad1c114.jpg)  
(b) PINN LBFGS loss

m = 5  
![](images/57b2d39c3860923206160da5775769e10f6789035c39fc347db0a1f6a52e99c4.jpg)  
(a) PINN solution

![](images/198b115bad51cd39a173d748929e98049d46f433e6ccd5df9c3e3126d86477fa.jpg)  
(b) Exact solution

![](images/ab60294d473ec8bbf15b2eac5e8d786613123873a4d38b4dcea6a59dd7ae28e3.jpg)  
(c) Classical solution

![](images/8c804bf33a8e158aabae4795c27d7e89612326cf0c196570e9752f7c23bb5217.jpg)  
(a) PINN ADAM loss

![](images/4e65a6d08ccc84780ecb68d81b04603a1f6cccd7f15c62c848326cf86c5f6490.jpg)  
(b) PINN LBFGS loss

## A.4 m-dependent solution

m = 2  
![](images/b7671056ad81f0c1f606a0d7c4b89da8d42a71aafcdd041f92f1ea8f9135fa00.jpg)  
(a) PINN solution

![](images/276432f3ad2bf78586e1e7e6c425088fb2674311ae0e506226b760d6bd791699.jpg)  
(b) Exact solution

![](images/0c32a649c909c04a8f610470c2725b9dac09bc9683ffadbf350eb2c32c194891.jpg)  
(c) Classical solution

![](images/868206f6a4440cb8933f78c95e271dfae92b59442262e98a8ba10510c5e311ec.jpg)  
(a) PINN ADAM loss

![](images/3c207a84cec7a2faf1b8008950d02f9b27c6fb607b288d40d59f55a7628722e4.jpg)  
(b) PINN LBFGS loss

$$
\mathbf { m } = 3
$$

![](images/984624b567fb63c3352c3d679afab0cc8dd1f1847d066a2502156dfaca5ba28c.jpg)  
(a) PINN solution

![](images/6d46a2c01e2f1aa93105caff62c6d17cef9b7277f88686bf89483a6b6b1dceae.jpg)  
(b) Exact solution

![](images/21ff6a1150a1f8fc19e472938c7197635bb381b39cd59b189a2f2ab06c3ab393.jpg)  
(c) Classical solution

![](images/c34271d46a6ce1f6a297396f75c0caf14278f92e634e910431f2c3222b5ad3d7.jpg)  
(a) PINN ADAM loss

![](images/fd45b66d5337b4575f544d6e02ef091095693711fd8e74bbd1d645c5a3597cff.jpg)  
(b) PINN LBFGS loss

m = 4  
![](images/bcd4dd653371f3db6b7db936b428a6cc57a5f55ee0a7933590ea0a1470d8ddc1.jpg)  
(a) PINN solution

![](images/8e517847a91fa01073ce255f3cb1560800583b21ade0a165c1218b8ffe18f40f.jpg)  
(b) Exact solution

![](images/e18539bb36cedf949851ed3a4ca23bca3e16c2ae163839bf32a5a6d7c3bfd8ff.jpg)  
(c) Classical solution

![](images/a79b96b520372881fa4a7e28a5def24628cb2dbf9960b6d8ca6f7661bc9d9341.jpg)  
(a) PINN ADAM loss

![](images/de1487fbe7bdc1af6b69861cbc7a069370642b6e43e312ecb2bb834e4b7526ee.jpg)  
(b) PINN LBFGS loss

![](images/7b4674a059e6966bd02abe041faf29e64e034e6f5dea8c19f7eaf9aba68eef03.jpg)  
(a) PINN solution

![](images/9003d835b3630fcbe0aac7b54d4e02062da59c6001fe346a52280a6f987553fd.jpg)  
(b) Exact solution

![](images/fe4c3b18db4dfae90b41257ad436d0b577e9a406c03b5645e97e11931fcbbc97.jpg)  
(c) Classical solution

![](images/14d3edc04d392fdc1b28bca5b954ace42d572fa3508805241dfc1476d48a80c3.jpg)  
(a) PINN ADAM loss

![](images/533675d35383773f0400185becb84a0afd4131e7cf6fe2b811df0eaf260f2824.jpg)  
(b) PINN LBFGS loss

## B Inverse PINN's training loss of the m-dependent solution

For the inverse PINN of the m-dependent solution, for each m and Initial Guess (IG), we show the evolution of the m parameter and the training loss over the epochs.

![](images/d65ecaf38fb907d50d2f7da2335220b07ff3ee8162fd8a11151d23370ebf14bd.jpg)  
m = 3 and IG = 4.50  
m = 3 and IG = 5.40

![](images/46387e88c2664645e695aded8c9006ae2a61d7a00c7d0a0aecf9871bd7694fa2.jpg)

## C Inverse PINN results before the 2-stage PINN model

In this section, the results of the inverse PINN problem after the standard modifications (i.e., before the 2-stage PINN model) are shown here for the different reference solutions from section 2.2 following the same plot layout of figure 7.

![](images/5e813878c17b0b8aa904114eccb3920001021545e2ad6bef66e4dbb6b014ca6e.jpg)

![](images/9eac6fde95757cf5bb31c555249239ff5ab27549f58f66b9053e426724142ea9.jpg)

![](images/4d501e0a77b5894ceec127ee13196c0d8d4bed675882e623dd8be03d466aaf81.jpg)

![](images/1332beb52cd17b6f40b7201897e646b18f591ca10562f864f5948c123a57e5d8.jpg)
# Improving the matrix multiplication exponent with modern optimization and AlphaEvolve

Emilien Dupont<sup>\*,1</sup>, Marvin Eisenberger<sup>\*,1</sup>, Borislav Kozlovskii<sup>\*,1</sup>, Abbas Mehrabian<sup>\*,1</sup>, Francisco J. R. Ruiz<sup>\*,1</sup>, Abigail See<sup>\*,1</sup>, Renfei Zhou<sup>\*,2</sup>, Josh Alman<sup>3</sup>, Virginia Vassilevska Williams<sup>4</sup> and Matej Balog <sup>\*</sup>Equal contribution in alphabetical order, <sup>1</sup>Google DeepMind, <sup>2</sup>Carnegie Mellon University, <sup>3</sup>Columbia University, <sup>4</sup>MIT

The current best bounds on the matrix multiplication exponent � are obtained through a refinement of the laser method called combination loss analysis (Duan et al., 2023; Williams et al., 2024; Alman et al., 2025). In this note, we address the optimization problem at the core of this approach and propose several improvements. First, we reformulate the optimization problem allowing us to solve it in a larger setting than was previously possible. Second, we leverage recent advances in machine learning to design a new optimization algorithm for this problem. Finally, we refine the resulting optimization algorithm with AlphaEvolve. Our combined approach yields an upper bound of � < 2.371177, improving the previous best bound of 2.371339.

## 1. Introduction

From accelerating machine learning computations to enabling realistic computer graphics, matrix multiplication is a fundamental operation underpinning critical applications in computer science. Despite its prominence, the computational complexity of matrix multiplication—the number of arithmetic operations needed to multiply large matrices—is unknown, and determining it is a major open question in theoretical computer science (Bläser, 2013).

The pioneering work of Strassen (Strassen, 1969) showed that two �×� matrices can be multiplied in sub-cubic time—specifically, $O ( n ^ { \omega + o ( 1 ) } ) _ { }$ ) operations for � < 2.81—spurring a line of work attempting to further reduce the complexity exponent � (Pan, 1978; Bini et al., 1979; Schönhage, 1981; Romani, 1982; Coppersmith and Winograd, 1981; Strassen, 1986; Coppersmith and Winograd, 1990; Stothers, 2010; Williams, 2012; Le Gall, 2014; Alman and Williams, 2024; Duan et al., 2023; Williams et al., 2024; Alman et al., 2025). All improvements in the past 40 years rely on the laser method, a mathematical technique to indirectly design matrix multiplication algorithms. The current best bound is achieved by a refinement of the laser method called combination loss analysis (Duan et al., 2023), a technique that requires solving a non-convex optimization problem as part of the computer-assisted proof, and yields � < 2.371339 (Alman et al., 2025).

Here, we improve the bound to � < 2.371177 using a two-step approach. First, we leverage recent advances in machine learning and adjacent areas to address the non-convex optimization problem of combination loss analysis using a gradient descent approach; this alone improves the previous state-of-the-art (SOTA) bound by $\approx 0 . 9 7 \times 1 0 ^ { - 4 }$ . Second, we use AlphaEvolve (Novikov et al., 2025) to improve our optimization algorithm; this raises the improvement over the SOTA $\mathrm { t o } \approx 1 . 6 2 \times 1 0 ^ { - 4 }$

<table><tr><td rowspan=1 colspan=1>Duan et al. (2023)</td><td rowspan=1 colspan=1>2.371866</td></tr><tr><td rowspan=1 colspan=1>Williams et al. (2024)</td><td rowspan=1 colspan=1>2.371552</td></tr><tr><td rowspan=1 colspan=1>Alman et al. (2025)</td><td rowspan=1 colspan=1>2.371339</td></tr><tr><td rowspan=1 colspan=1>This note</td><td rowspan=1 colspan=1>2.371177</td></tr></table>

Table 1 | Recent improvements to �.

The optimization problem at the core of combination loss analysis is formulated in Alman et al. (2025), where it was also shown that any feasible solution provides an upper bound on �. To achieve our new bound, we target a slightly diferent optimization problem. Specifically, combination loss analysis has a parameter—the maximum recursion level, denoted by $\ell ^ { * }$ —that introduces a trade-of between the complexity of the optimization (which grows doubly exponentially in ℓ<sup>∗</sup>) and the best possible bound on � it can achieve. The previous SOTA bound was found with $\ell ^ { * } = 3$ (Alman et al., 2025). In contrast, our gradient-based optimization, implemented in Jax (Bradbury et al., 2018), allows for hardware parallelization and can handle $\ell ^ { * } = 4$ (the number of optimizable parameters increases from approximately 25k to 7 million when moving from $\ell ^ { * } = 3 \mathrm { t o } \ell ^ { * } = 4 )$

In this note, we describe the full optimization problem from Alman et al. (2025) in detail, explain how we numerically solved it and applied AlphaEvolve to it, and finally show how we rigorously certified the resulting omega bound.

## 2. Optimization problem

The high-level structure of the optimization problem can be captured by a rooted tree, where every node is associated with a collection of optimizable parameters. Integers $q \geq 1$ and $\ell ^ { * } \geq 2$ are fixed as hyperparameters. Intuitively, the tree structure describes a recursive way to decompose the tensor $C W _ { q } ^ { \otimes 2 ^ { \ell ^ { * } } }$ into smaller tensors, where $C W _ { q }$ is the Coppersmith-Winograd tensor (Coppersmith and Winograd, 1990); see Alman et al. (2025) for more details about this correspondence. We start by defining the tree structure, and will then define the optimizable parameters associated to its nodes.

## 2.1. Tree structure

Denote $[ k ] : = \{ 1 , \ldots , k \}$ . Each non-root node � in the tree is associated with a level, a shape $s _ { T } .$ , and a region � , defined as follows. $r _ { T }$

• The level of a non-root node is a positive integer $\ell \in \{ 2 , 3 , \ldots , \ell ^ { * } \}$ , describing its depth in the tree; higher is closer to the root. Direct children of the root node have level $\ell ^ { * }$ , which is a hyperparameter fixed in advance.

• A level-ℓ shape is a triple $\boldsymbol { s } = ( s _ { \scriptscriptstyle X } , s _ { \scriptscriptstyle Y } , s _ { \scriptscriptstyle Z } )$ , where $s _ { X } , s _ { Y } , s _ { Z }$ are non-negative integers summing to $2 ^ { \ell }$ . We use

$$
S _ { \ell } : = \left\{ ( i , j , k ) \in \mathbb { Z } _ { \geq 0 } ^ { 3 } \left| \ i + j + k = 2 ^ { \ell } \right. \right\}
$$

to denote the set of all level-ℓ shapes. We use dimensions $X , Y , Z$ to refer to the indices of the three coordinates in a shape.

• There are six regions, indexed by an integer $r \in [ 6 ]$ . Each region is associated with $\pi _ { r } ,$ the �-th permutation over symbols $\{ X , Y , Z \}$ in lexicographic order.

We call a non-root node � a positive-shape node if all coordinates in its shape $s _ { T }$ are positive. Otherwise it is a zero-shape node. The root node is denoted by � and does not have the aforementioned attributes—level, shape, and region.

Next, we define the tree structure by describing the child nodes for diferent types of nodes.

• Root: For every region $r \in [ 6 ]$ and level-ℓ<sup>∗</sup> shape $s \in S _ { \ell ^ { * } }$ , the root � has a child at level $\ell ^ { * }$ with shape � and region �, denoted as $G [ s , r ]$

• Positive-shape node: For every level-ℓ shape $\boldsymbol { s } = ( s _ { \scriptscriptstyle X } , s _ { \scriptscriptstyle Y } , s _ { \scriptscriptstyle Z } )$ , we define

$$
\mathrm { S p l i t } ( s ) : = \left\{ u \in S _ { \ell - 1 } ~ \Big | ~ 0 \leq u _ { X } \leq s _ { X } , ~ 0 \leq u _ { Y } \leq s _ { Y } , ~ 0 \leq u _ { Z } \leq s _ { Z } \right\} .
$$

Fixing a positive-shape node � at level $\ell \geq 3$ , for every region $r \in [ 6 ]$ and level- $( \ell - 1 )$ shape $u \in S \mathrm { p l i t } ( s _ { \scriptscriptstyle T } )$ , the node � has a child at level ℓ − 1 with shape � and region $r ,$ denoted as $T [ u , r ]$ Note that the region index � of the child can be diferent from that of � itself.

• Zero-shape nodes and level-2 positive-shape nodes do not have child nodes. We call them the leaves.

## 2.2. Optimizable parameters

Next we list the free variables of the optimization problem, grouped by the associated node on the tree. For any finite set $D ,$ we use $\Delta ( D )$ to denote the simplex of probability distributions on �:

$$
\Delta ( D ) : = \{ p : D  [ 0 , 1 ]  \sum _ { x \in D } p ( x ) = 1 \} 
$$

Root node. The optimizable parameters associated with the root node � are:

• A distribution $A _ { G } = ( A _ { G } ^ { ( 1 ) } , \dots , A _ { G } ^ { ( 6 ) } ) \in \Delta ( [ 6 ] )$ over the six regions;

• For $r \in [ 6 ]$ , a distribution $\alpha _ { G } ^ { ( r ) } \in \Delta ( S _ { \ell ^ { * } } )$ over all level- $\cdot \ell ^ { * }$ shapes.

Positive-shape node. Each level-ℓ positive-shape node � with $\ell \geq 3$ is associated with the following parameters:

• A distribution $A _ { T } = ( A _ { T } ^ { ( 1 ) } , \dots , A _ { T } ^ { ( 6 ) } ) \in \Delta ( [ 6 ] )$ over the six regions;

• For $r \in [ 6 ]$ , a distribution $\alpha _ { T } ^ { ( r ) } \in \Delta ( \mathrm { S p l i t } ( s _ { T } ) )$ , where $s _ { T }$ is the shape of $T _ { \mathbf { \delta } }$

Zero-shape node. For a level ℓ and an integer � with $0 \leq a \leq 2 ^ { \ell }$ , we define

$$
C _ { \ell , a } : = \left\{ L \in \{ 0 , 1 , 2 \} ^ { 2 ^ { \ell - 1 } } \left| \sum _ { p = 1 } ^ { 2 ^ { \ell - 1 } } L _ { p } = a \right. \right\} .
$$

This is the set of $\mathrm { l e n g t h } { - 2 ^ { \ell - 1 } }$ vectors over {0, 1, 2} whose entries sum to �. A distribution over $C _ { \ell , a }$ is called a level-ℓ complete split distribution of �.

For every level-ℓ zero-shape node $T ,$ let $W \in \{ X , Y , Z \}$ be the first dimension where $s _ { T , W }$ is nonzero. � is then associated with a complete split distribution $\beta _ { T , W } \in \Delta ( C _ { \ell , s _ { T , W } } )$

Level-2 node. The only remaining nodes are level-2 nodes with shapes (1, 1, 2), (1, 2, 1), or (2, 1, 1), which are the only valid strictly positive shapes for level $^ { 2 , }$ since their coordinates must sum to $2 ^ { 2 } = 4$ . Each such node is associated with a scalar $\mu _ { { \scriptscriptstyle T } } \in [ 0 , 1 / 2 ]$

## 2.3. Derived quantities

With the free variables fixed, all remaining quantities can be computed deterministically. In the following, all logarithms and entropies are in base two. For a distribution $\rho$ with finite support, we denote its entropy by

$$
H ( \rho ) : = - \sum _ { x \in \mathrm { s u p p } ( \rho ) } \rho ( x ) \log \rho ( x ) .
$$

We also use $H ( p _ { 1 } , p _ { 2 } , \ldots , p _ { k } )$ to denote the entropy of a distribution over [�] with probability masses $p _ { 1 } , \ldots , p _ { k }$ . If � is a set of shapes and $\rho \in \Delta ( D )$ , let $\rho _ { w }$ denote the marginal of $\rho$ in coordinate $W _ { i }$ namely, $\rho _ { W } ( w ) : = \sum _ { a \in D : a _ { W } = w } \rho ( a )$ . Define

$$
H _ { D } ^ { \operatorname* { m a x } } ( \rho ) : = \operatorname* { s u p } _ { \rho ^ { \prime } \in \Delta ( D ) } H ( \rho ^ { \prime } ) ,\tag{1}
$$

as well as the penalty notion

$$
\begin{array} { r } { P _ { D } ( \rho ) : = H _ { D } ^ { \operatorname* { m a x } } ( \rho ) - H ( \rho ) . } \end{array}
$$

Masses. Each non-zero-shape node � of the tree has a real number $m _ { T } \in [ 0 , 1 ]$ associated with it, called its mass. The masses of non-root nodes are computed top-down as follows:

• The root � has $m _ { G } = 1$

• For the children of the root, we set

$$
m _ { G [ s , r ] } : = A _ { G } ^ { ( r ) } \cdot \alpha _ { G } ^ { ( r ) } ( s ) , \qquad \forall s \in S _ { \ell ^ { * } } , r \in [ 6 ] .
$$

• For any positive-shape node $T$ of level $\ell \geq 3$ , we set

$$
m _ { T [ u , r ] } = m _ { T } \cdot A _ { T } ^ { ( r ) } \cdot \left( \alpha _ { T } ^ { ( r ) } ( u ) + \alpha _ { T } ^ { ( r ) } ( s _ { T } - u ) \right) , \qquad \forall u \in \mathrm { S p l i t } ( s _ { T } ) , \ r \in [ 6 ] .
$$

Complete split distributions. Every level-ℓ non-root node � carries, for each dimension $W \in$ $\{ X , Y , Z \}$ , a complete split distribution $\beta _ { T , W } \in \Delta \big ( C _ { \ell , s _ { T , W } } \big )$ . These distributions are calculated as follows.

Let $\vec { 0 } : = ( 0 , 0 , \ldots , 0 )$ and $\vec { 2 } : = ( 2 , 2 , \ldots , 2 )$ denote vectors of length $2 ^ { \ell - 1 }$ . For a zero-shape node $T ,$ , let $W _ { 0 } \in \{ X , Y , Z \}$ be the first zero coordinate of $s _ { T } , W _ { 1 }$ be the first nonzero coordinate of $s _ { T . }$ , and $W _ { 2 }$ be the other coordinate. $\beta _ { T , W _ { 0 } }$ is the point mass distribution at the length- $- 2 ^ { \ell - 1 }$ vector $\vec { 0 } ; \beta _ { { } _ { T , W _ { 1 } } }$ was defined as optimizable parameters in section 2.2; $\beta _ { T , W _ { 2 } } : = \beta _ { T , W _ { 1 } } ^ { \vee }$ , where for any complete split distribution $\beta _ { z }$ , we define

$$
\beta ^ { \vee } ( \vec { 2 } - L ) : = \beta ( L ) , \qquad \forall L \in \operatorname { s u p p } ( \beta ) .
$$

(Notice, in particular, that if the components of � sum to $s _ { T , W _ { 1 } }$ , then the components of ${ \vec { 2 } } - L$ sum to $2 \cdot 2 ^ { \ell - 1 } - s _ { T , W _ { 1 } } = s _ { T , W _ { 2 } }$ , keeping the mapping validly within the correct domain.)

For a positive-shape node � at level $\ell \geq 3$ , we define

$$
\beta _ { T , W } ^ { ( r ) } : = \sum _ { u \in \mathrm { S p l i t } ( s _ { T } ) } \alpha _ { T } ^ { ( r ) } ( u ) \cdot \big ( \beta _ { T [ u , r ] , W } \times \beta _ { T [ s _ { T } - u , r ] , W } \big ) , \qquad \forall r \in [ 6 ] ,
$$

where $\times$ denotes the Cartesian product of complete split distributions; that is, if $L _ { \mathrm { l e f t } }$ and $L _ { \mathrm { r i g h t } }$ are two sequences of length $2 ^ { \ell - 2 }$ in supports of $\beta _ { \mathrm { l e f t } } : = \beta _ { T [ u , r ] , W }$ and $\beta _ { \mathrm { r i g h t } } : = \beta _ { T [ s _ { T } - u , r ] , W }$ , respectively, then we form a new sequence $L ,$ of length $2 ^ { \ell - 1 }$ , by concatenating $\boldsymbol { L } _ { \mathrm { l e f t } }$ and ${ \cal L } _ { \mathrm { r i g h t } } ,$ and letting

$$
( \beta _ { \mathrm { l e f t } } \times \beta _ { \mathrm { r i g h t } } ) ( L ) : = \beta _ { \mathrm { l e f t } } ( L _ { \mathrm { l e f t } } ) \cdot \beta _ { \mathrm { r i g h t } } ( L _ { \mathrm { r i g h t } } ) .
$$

Then, the complete split distributions of � are calculated by

$$
\beta _ { T , W } : = \sum _ { r = 1 } ^ { 6 } A _ { T } ^ { ( r ) } \beta _ { T , W } ^ { ( r ) } .
$$

The only remaining case is a level-2 positive-shape node, which must have shape $( 1 , 1 , 2 ) , ( 2 , 1 , 1 )$ , or (1, 2, 1). For a node � with shape (1, 1, 2), recall that $\mu _ { \ / T } \in [ 0 , 1 / 2 ]$ is the only optimizable parameter associated with �. We let $\delta _ { a , b }$ denote the point mass at $\dot { ( a , b ) } \in \{ 0 , 1 , 2 \} ^ { 2 }$ and set

$$
\beta _ { T , X } = \beta _ { T , Y } = \frac { 1 } { 2 } \delta _ { 0 , 1 } + \frac { 1 } { 2 } \delta _ { 1 , 0 } , \qquad \beta _ { T , Z } = \mu _ { T } \delta _ { 0 , 2 } + \mu _ { T } \delta _ { 2 , 0 } + ( 1 - 2 \mu _ { T } ) \delta _ { 1 , 1 } .\tag{2}
$$

Similarly, for a node � with shape (2, 1, 1), we set

$$
\beta _ { T , Y } = \beta _ { T , Z } = \frac { 1 } { 2 } \delta _ { 0 , 1 } + \frac { 1 } { 2 } \delta _ { 1 , 0 } , \qquad \beta _ { T , X } = \mu _ { T } \delta _ { 0 , 2 } + \mu _ { T } \delta _ { 2 , 0 } + ( 1 - 2 \mu _ { T } ) \delta _ { 1 , 1 } ,
$$

and for a node � with shape (1, 2, 1), we set

$$
\beta _ { T , X } = \beta _ { T , Z } = \frac { 1 } { 2 } \delta _ { 0 , 1 } + \frac { 1 } { 2 } \delta _ { 1 , 0 } , \qquad \beta _ { T , Y } = \mu _ { T } \delta _ { 0 , 2 } + \mu _ { T } \delta _ { 2 , 0 } + ( 1 - 2 \mu _ { T } ) \delta _ { 1 , 1 } .
$$

Retained exponent at root node. Next, we define how to calculate the retained exponent $E _ { G }$ associated with the root $G ,$ which will be obtained from intermediate quantities $E _ { G } ^ { ( r ) }$ for regions $r \in [ 6 ]$ We start by introducing how to compute $E _ { G } ^ { ( 1 ) }$ for region $r = 1$ , where $\pi _ { 1 }$ is the identity permutation. In all summations below, � ranges over $S _ { \ell ^ { * } }$ . We will use the symbols $* , + ,$ and < in our notation below as mnemonic placeholders corresponding to the dimensions, indicating the meanings “unconstrained / any value”, “strictly positive”, and “strictly less than the parent’s coordinate”, respectively. We define

$$
\eta _ { G , Y } ^ { ( 1 ) } : = \sum _ { s : s _ { z } = 0 } \alpha _ { G } ^ { ( 1 ) } ( s ) \cdot H \Bigl ( \beta _ { G [ s , 1 ] , Y } \Bigr ) + \sum _ { j = 0 } ^ { 2 ^ { \ell ^ { * } } } \alpha _ { G } ^ { ( 1 ) } ( * , j , + ) \cdot H \Bigl ( \bar { \beta } _ { G , Y , * , j , + } ^ { ( 1 ) } \Bigr ) ,\tag{3}
$$

where

$$
\alpha _ { G } ^ { ( 1 ) } ( \ast , j , + ) : = \sum _ { \substack { s : s _ { \gamma } = j , s _ { z } > 0 } } \alpha _ { G } ^ { ( 1 ) } ( s ) , \qquad \bar { \beta } _ { G , Y , \ast , j , + } ^ { ( 1 ) } : = \frac { 1 } { \alpha _ { G } ^ { ( 1 ) } ( \ast , j , + ) } \sum _ { \substack { s : s _ { \gamma } = j , s _ { z } > 0 } } \alpha _ { G } ^ { ( 1 ) } ( s ) \cdot \beta _ { G [ s , 1 ] , Y } .
$$

We then define

$$
\eta _ { G , Z } ^ { ( 1 ) } : = \sum _ { \substack { s : s _ { x } = 0 \mathrm { ~ o r ~ } s _ { \gamma } = 0 } } \alpha _ { G } ^ { ( 1 ) } ( s ) \cdot H \Bigl ( \beta _ { G [ s , 1 ] , Z } \Bigr ) + \sum _ { k = 0 } ^ { 2 ^ { \ell ^ { * } } } \alpha _ { G } ^ { ( 1 ) } ( + , + , k ) \cdot H \Bigl ( \bar { \beta } _ { G , Z , + , k } ^ { ( 1 ) } \Bigr ) ,\tag{4}
$$

where

$$
\alpha _ { G } ^ { ( 1 ) } ( + , + , k ) : = \sum _ { \substack { s : s _ { X } > 0 , s _ { Y } > 0 , s _ { Z } = k } } \alpha _ { G } ^ { ( 1 ) } ( s ) , \qquad \bar { \beta } _ { G , Z , + , + , k } ^ { ( 1 ) } : = \frac { 1 } { \alpha _ { G } ^ { ( 1 ) } ( + , + , k ) } \sum _ { \substack { s : s _ { X } > 0 , s _ { Y } > 0 , s _ { Z } = k } } \alpha _ { G } ^ { ( 1 ) } ( s ) \cdot \beta _ { G | s , 1 ] , Z } .
$$

To avoid the issue of denominators $\alpha _ { G } ^ { ( 1 ) } ( \cdot )$ being zero, we regard 0 · undefined $: = 0$ , so that Equation (3) and Equation (4) are always well-defined.

The quantity $E _ { G } ^ { ( 1 ) }$ is then given by

$$
\begin{array} { r } { E _ { G } ^ { ( 1 ) } : = \operatorname* { m i n } \Big \{ H \Big ( \big ( \alpha _ { G } ^ { ( 1 ) } \big ) _ { X } \Big ) - P _ { S _ { \ell ^ { * } } } \Big ( \alpha _ { G } ^ { ( 1 ) } \Big ) , \ H \Big ( \bar { \beta } _ { G , Y , * , * , * } ^ { ( 1 ) } \Big ) - \eta _ { G , Y } ^ { ( 1 ) } , \ H \Big ( \bar { \beta } _ { G , Z , * , * } ^ { ( 1 ) } \Big ) - \eta _ { G , Z } ^ { ( 1 ) } \Big \} , } \end{array}\tag{5}
$$

where

$$
\bar { \beta } _ { G , W , * , * , * } ^ { ( 1 ) } : = \sum _ { s \in S _ { \ell ^ { * } } } \alpha _ { G } ^ { ( 1 ) } ( s ) \cdot \beta _ { G [ s , 1 ] , W } , \qquad \forall W \in \{ X , Y , Z \} .
$$

For $r = 2 , \ldots , 6 .$ , we define $E _ { G } ^ { ( r ) }$ by applying the same formula Equation (5) after relabeling the coordinates �, �, � as $\pi _ { r } ( X ) , \pi _ { r } ( Y ) , \pi _ { r } ( Z )$ , and replacing the region index 1 with �. Finally, we set

$$
E _ { G } : = \sum _ { r = 1 } ^ { 6 } A _ { G } ^ { ( r ) } \cdot E _ { G } ^ { ( r ) } .
$$

Retained exponents at level $\ell \geq 3$ . For each positive-shape node � at level $\ell \geq 3$ , we will calculate intermediate quantities $E _ { T , W } ^ { ( r ) }$ for each region $r \in [ 6 ]$ and $W \in \{ X , Y , Z \}$ . These quantities will later be aggregated to form the retained exponent of level ℓ. Note that the region index � in the calculation can be diferent from the region index $r _ { T }$ of node � itself. As in the above, we start by defining the quantities for region $r = 1$ where $\pi _ { 1 }$ is the identity permutation. In the summations below, � ranges over $\mathrm { { S p l i t } } ( s _ { T } )$

We first define

$$
\begin{array} { r l } & { \eta _ { T , Y } ^ { ( 1 ) } : = \displaystyle \sum _ { u : u _ { Z } = 0 } \Big ( \alpha _ { T } ^ { ( 1 ) } ( u ) + \alpha _ { T } ^ { ( 1 ) } ( s _ { T } - u ) \Big ) \cdot H \Big ( \beta _ { T [ u , 1 ] , Y } \Big ) } \\ & { \quad \quad \quad + \displaystyle \operatorname* { m i n } \{ s _ { T , Y } , 2 ^ { \ell - 1 } \} } \\ & { \quad \quad \quad + \quad \displaystyle \sum _ { j = 0 } \Big ( \alpha _ { T } ^ { ( 1 ) } ( * , j , + ) + \alpha _ { T } ^ { ( 1 ) } ( * , s _ { T , Y } - j , < ) \Big ) \cdot H \Big ( \bar { \beta } _ { T , Y , * , j , + } ^ { ( 1 ) } \Big ) , } \end{array}\tag{6}
$$

where

$$
\alpha _ { T } ^ { ( 1 ) } ( \ast , j , + ) : = \sum _ { \substack { u : u _ { Y } = j , u _ { Z } > 0 } } \alpha _ { T } ^ { ( 1 ) } ( u ) , \qquad \alpha _ { T } ^ { ( 1 ) } ( \ast , s _ { T , Y } - j , < ) : = \sum _ { \substack { u : u _ { Y } = s _ { T , Y } - j , } \atop u _ { Z } < s _ { T , Z } } \alpha _ { T } ^ { ( 1 ) } ( u ) ,
$$

$$
\bar { \beta } _ { T , Y , * , j , + } ^ { ( 1 ) } : = \frac { 1 } { \alpha _ { T } ^ { ( 1 ) } ( * , j , + ) + \alpha _ { T } ^ { ( 1 ) } ( * , s _ { T , Y } - j , < ) } \cdot \sum _ { u : u _ { Y } = j , u _ { Z } > 0 } \left( \alpha _ { T } ^ { ( 1 ) } ( u ) + \alpha _ { T } ^ { ( 1 ) } ( s _ { T } - u ) \right) \cdot \beta _ { T [ u , 1 ] , Y } .
$$

We also define

$$
\begin{array} { r l } & { \eta _ { T , Z } ^ { ( 1 ) } : = \displaystyle \sum _ { u : u _ { X } = 0 \mathrm { ~ o r ~ } u _ { Y } = 0 } \Big ( \alpha _ { T } ^ { ( 1 ) } ( u ) + \alpha _ { T } ^ { ( 1 ) } ( s _ { T } - u ) \Big ) \cdot H \Big ( \beta _ { T [ u , 1 ] , Z } \Big ) } \\ & { \quad \quad \quad \quad \quad \operatorname* { m i n } \{ s _ { T , Z } , 2 ^ { \ell - 1 } \} } \\ & { + \quad \displaystyle \sum _ { k = 0 } \quad \quad \Big ( \alpha _ { T } ^ { ( 1 ) } ( + , + , k ) + \alpha _ { T } ^ { ( 1 ) } ( < , < , s _ { T , Z } - k ) \Big ) \cdot H \Big ( \bar { \beta } _ { T , Z , + , + , k } ^ { ( 1 ) } \Big ) , } \end{array}\tag{7}
$$

where

$$
\alpha _ { T } ^ { ( 1 ) } ( + , + , k ) : = \sum _ { \substack { u : u _ { X } > 0 , u _ { Y } > 0 , u _ { Z } = k } } \alpha _ { T } ^ { ( 1 ) } ( u ) , \qquad \alpha _ { T } ^ { ( 1 ) } ( < , < , s _ { T , Z } - k ) : = \sum _ { \substack { u : u _ { X } < s _ { T , X } , u _ { Y } < s _ { T , Y } , } \atop u _ { Z } = s _ { T , Z } - k } \alpha _ { T } ^ { ( 1 ) } ( u ) ,
$$

$$
\bar { \beta } _ { T , z , + , + , k } ^ { ( 1 ) } : = \frac { 1 } { \alpha _ { T } ^ { ( 1 ) } ( + , + , k ) + \alpha _ { T } ^ { ( 1 ) } ( < , < , s _ { T , z } - k ) } \cdot \sum _ { u : u _ { x } > 0 , u _ { y } > 0 , u _ { z } = k } \left( \alpha _ { T } ^ { ( 1 ) } ( u ) + \alpha _ { T } ^ { ( 1 ) } ( s _ { T } - u ) \right) \cdot \beta _ { T [ u , 1 ] , z } .
$$

Again, as we define 0 · undefined $: = 0$ , Equation (6) and Equation (7) are well-defined even when some denominators $\alpha _ { T } ^ { ( 1 ) }$ are zeros.

Then, $E _ { T , W } ^ { ( 1 ) }$ are given by

$$
\begin{array} { r l } & { E _ { T , X } ^ { ( 1 ) } : = m _ { T } A _ { T } ^ { ( 1 ) } \cdot \Bigl ( H \Bigl ( \bigl ( \alpha _ { T } ^ { ( 1 ) } \bigr ) _ { X } \Bigr ) - P _ { \mathsf { S p l i t } ( s _ { T } ) } \Bigl ( \alpha _ { T } ^ { ( 1 ) } \Bigr ) \Bigr ) , } \\ & { E _ { T , Y } ^ { ( 1 ) } : = m _ { T } A _ { T } ^ { ( 1 ) } \cdot \Bigl ( H \Bigl ( \beta _ { T , Y } ^ { ( 1 ) } \Bigr ) - \eta _ { T , Y } ^ { ( 1 ) } \Bigr ) , } \\ & { E _ { T , Z } ^ { ( 1 ) } : = m _ { T } A _ { T } ^ { ( 1 ) } \cdot \Bigl ( H \Bigl ( \beta _ { T , Z } ^ { ( 1 ) } \Bigr ) - \eta _ { T , Z } ^ { ( 1 ) } \Bigr ) . } \end{array}\tag{8}
$$

For regions $r = 2 , \ldots , 6 .$ , we define $E _ { T , \pi _ { r } ( X ) } ^ { ( r ) } , E _ { T , \pi _ { r } ( Y ) } ^ { ( r ) } :$ , and $E _ { T , \pi _ { r } ( Z ) } ^ { ( r ) }$ by applying Equation (6), Equation (7), and Equation (8) after relabeling coordinates $X , Y , Z$ as $\pi _ { r } ( X ) , \pi _ { r } ( Y ) , \pi _ { r } ( Z )$ and replacing region index 1 by �.

To calculate the retained exponent for level $\ell \in [ 3 , \ell ^ { * } ]$ , we let $\mathcal { T } _ { \ell } ^ { + }$ denote the set of positive-shape nodes at level ℓ. Then, the retained exponent at level ℓ is given by

$$
E _ { \ell } : = \sum _ { r = 1 } ^ { 6 } \operatorname* { m i n } \left\{ \sum _ { \substack { T \in \mathcal { T } _ { \ell } ^ { + } } } E _ { T , X } ^ { ( r ) } , \sum _ { T \in \mathcal { T } _ { \ell } ^ { + } } E _ { T , Y } ^ { ( r ) } , \sum _ { T \in \mathcal { T } _ { \ell } ^ { + } } E _ { T , Z } ^ { ( r ) } \right\} .
$$

Retained exponent at level 2. For a positive level-2 node � of shape (1, 1, 2), we define

$$
\left( E _ { T , X } , E _ { T , Y } , E _ { T , Z } \right) : = m _ { T } \cdot \left( 1 , \ 1 , \ H \big ( \mu _ { T } , \mu _ { T } , 1 - 2 \mu _ { T } \big ) \right) .\tag{9}
$$

Similarly, for a node � of shape (2, 1, 1), we define

$$
\left( E _ { T , X } , E _ { T , Y } , E _ { T , Z } \right) : = m _ { T } \cdot ( H ( \mu _ { T } , \mu _ { T } , 1 - 2 \mu _ { T } ) , \ 1 , \ 1 ) ,
$$

and for a node � of shape (1, 2, 1), we define

$$
\Big ( E _ { T , X } , E _ { T , Y } , E _ { T , Z } \Big ) : = m _ { T } \cdot \big ( 1 , ~ H \big ( \mu _ { T } , \mu _ { T } , 1 - 2 \mu _ { T } \big ) , ~ 1 \big ) ,
$$

Then, letting $\mathcal { T } _ { 2 } ^ { + }$ denote the set of positive-shape nodes at level 2, the retained exponent at level 2 is given by

$$
E _ { 2 } : = \operatorname* { m i n } \left\{ \sum _ { T \in \mathcal { T } _ { 2 } ^ { + } } E _ { T , X } , \ \sum _ { T \in \mathcal { T } _ { 2 } ^ { + } } E _ { T , Y } , \ \sum _ { T \in \mathcal { T } _ { 2 } ^ { + } } E _ { T , Z } \right\} .
$$

Local matrix size for zero-shape nodes. For every zero-shape node �, we let $W _ { 0 } \in \{ X , Y , Z \}$ be the first zero coordinate of $s _ { T }$ and $W _ { \ u { 1 } }$ be the first nonzero coordinate of $s _ { T }$ . Then, we define the local matrix size of $T ,$ written $( M _ { T , X } , M _ { T , Y } , M _ { T , Z } )$ , as

$$
M _ { T , W _ { 0 } } : = m _ { T } \cdot \Bigg ( H \Big ( \beta _ { T , W _ { 1 } } \Big ) + \sum _ { L \in \mathrm { s u p p } ( \beta _ { T , W _ { 1 } } ) } \beta _ { T , W _ { 1 } } ( L ) \cdot \Big | \big \{ p \in [ 2 ^ { \ell - 1 } ] \ \Big | \ L _ { p } = 1 \big \} \Big | \cdot \log q \Bigg ) ;
$$

the other two coordinates in the local matrix size are zeros.

Local matrix size for (1, 1, 2)-nodes. For a node � with shape (1, 1, 2), we define its local matrix size as

$$
\left( M _ { T , X } , M _ { T , Y } , M _ { T , Z } \right) : = m _ { T } \cdot \left( ( 1 - 2 \mu _ { T } ) \log q , \ ( 1 - 2 \mu _ { T } ) \log q , \ 2 \mu _ { T } \log q \right) .\tag{10}
$$

Similarly, for a node � with shape (2, 1, 1), we define its local matrix size as

$$
\left( M _ { T , X } , M _ { T , Y } , M _ { T , Z } \right) : = m _ { T } \cdot \left( 2 \mu _ { T } \log q , \left( 1 - 2 \mu _ { T } \right) \log q , \left( 1 - 2 \mu _ { T } \right) \log q \right) ,
$$

and for a node � with shape (1, 2, 1), we define its local matrix size as

$$
\left( M _ { T , X } , M _ { T , Y } , M _ { T , Z } \right) : = m _ { T } \cdot \left( ( 1 - 2 \mu _ { T } ) \log q , \ 2 \mu _ { T } \log q , \ ( 1 - 2 \mu _ { T } ) \log q \right) .
$$

## 2.4. Final assembly

Finally, we define the total retained exponent as

$$
E _ { \mathrm { t o t a l } } : = E _ { G } + E _ { 2 } + \sum _ { \ell = 3 } ^ { \ell ^ { * } } E _ { \ell } ,
$$

and define the total matrix size as

$$
M _ { \mathrm { t o t a l } } : = \operatorname* { m i n } \Biggl \{ \sum _ { T \in \mathcal { L } } M _ { T , X } , \ \sum _ { T \in \mathcal { L } } M _ { T , Y } , \ \sum _ { T \in \mathcal { L } } M _ { T , Z } \Biggr \} ,
$$

where $\mathcal { L }$ is the set of leaves. We can now write the full optimization problem.

```tcl
minimize Ω
subject to $E _ { \mathrm { t o t a l } } + M _ { \mathrm { t o t a l } } \cdot \Omega \geq 2 ^ { \ell ^ { * } - 1 } \log ( q + 2 ) ,$ (11)
all free variables lie in the domains stated in section 2.2.
```

Theorem 1 (Alman et al. (2025)). Anyfeasible solution of Equation (11) implies $\omega \leq \Omega ,$ where � is the asymptotic matrix multiplication exponent.

## 2.5. Dealing with maximum entropies

Among the quantities introduced above, $H _ { D } ^ { \mathrm { m a x } } ( \rho )$ is the only type that cannot be computed analytically. However, it is easy to see that replacing each occurrence of $H _ { D } ^ { \mathrm { m a x } }$ in the computation of $E _ { \mathrm { t o t a l } }$ with its upper bound can only decrease $E _ { \mathrm { t o t a l } } ,$ so any solution that satisfies the constraints in Equation (11) after this replacement is feasible for Equation (11) itself, and still yields a valid upper bound $\omega \leq \Omega$ In this section, we describe how to derive an upper bound of $H _ { D } ^ { \mathrm { m a x } } ( \rho )$

For a set � of shapes and a probability distribution $\rho \in \Delta ( D )$ , a valid certificate for $H _ { D } ^ { \mathrm { m a x } } ( \rho )$ consists of:

1. a distribution $y \in \Delta ( D )$ satisfying $y ( a ) > 0$ for every $a \in D$ and $y _ { W } = \rho _ { W }$ for every $W \in \{ X , Y , Z \}$

2. a real number $\lambda _ { 0 }$ and, for each $W \in \{ X , Y , Z \}$ and each marginal value � $\in \{ a _ { \scriptscriptstyle W } \mid a \in D \}$ , a real number $\lambda _ { w } ( w )$ (these values play the role of Lagrange multipliers for the maximization defining $H _ { D } ^ { \operatorname* { m a x } } ( \rho ) )$ ; and

3. a real number $\varepsilon \geq 0$ satisfying, for every $a = ( a _ { _ X } , a _ { { Y } } , a _ { { Z } } ) \in { \cal D } _ { \Sigma }$

$$
\left| \log y ( a ) - \left( \lambda _ { 0 } + \lambda _ { X } ( a _ { X } ) + \lambda _ { Y } ( a _ { Y } ) + \lambda _ { Z } ( a _ { Z } ) \right) \right| \le \varepsilon .\tag{12}
$$

Lemma 1. For any valid certificate, we have $H ( y ) \leq H _ { D } ^ { \operatorname* { m a x } } ( \rho ) \leq H ( y ) + 2 \varepsilon .$

Proof. The left inequality follows from the definition: � has the same marginals as $\rho ,$ so it is feasible for the maximization defining $H _ { D } ^ { \mathrm { m a x } } ( \rho )$ (see Equation (1)).

For the right inequality, fix any $\rho ^ { \prime } \in \Delta ( D )$ with $\rho _ { W } ^ { \prime } = \rho _ { W }$ for all $W \in \{ X , Y , Z \}$ ; to prove the right inequality we need only show $H ( \rho ^ { \prime } ) \leq H ( y ) + 2 \varepsilon$

The entropy function � is concave on $\Delta ( D )$ , and since � is strictly positive, � is diferentiable at � with $\nabla H ( y ) _ { a } = - \log y ( a ) - \log e$ (recall that all logarithms are in base 2). Therefore

$$
H ( \rho ^ { \prime } ) \leq H ( y ) + \big \langle \nabla H ( y ) , \rho ^ { \prime } - y \big \rangle .
$$

Write $g ( a ) : = \lambda _ { 0 } + \lambda _ { X } ( a _ { X } ) + \lambda _ { Y } ( a _ { Y } ) + \lambda _ { Z } ( a _ { Z } )$ , and view �, log �, and the constant log � as vectors indexed by $a \in D$ . Then,

$$
\begin{array} { r l } & { \big \langle g + \log e , \rho ^ { \prime } - y \big \rangle = \big ( \lambda _ { 0 } + \log e \big ) \cdot \displaystyle \sum _ { a \in D } \big ( \rho ^ { \prime } ( a ) - y ( a ) \big ) + \displaystyle \sum _ { W \in \{ X , Y , Z \} } \displaystyle \sum _ { a \in D } \lambda _ { W } ( a _ { W } ) \cdot \big ( \rho ^ { \prime } ( a ) - y ( a ) \big ) } \\ & { \qquad = \big ( \lambda _ { 0 } + \log e \big ) \cdot ( 1 - 1 ) + \displaystyle \sum _ { W \in \{ X , Y , Z \} } \displaystyle \sum _ { w \in \{ a _ { W } | a \in D \} } \lambda _ { W } ( w ) \cdot \big ( \rho _ { W } ^ { \prime } ( w ) - y _ { W } ( w ) \big ) = 0 , } \end{array}
$$

where the last step uses $\rho _ { W } ^ { \prime } = \rho _ { W } = y _ { W }$ for each $W \in \{ X , Y , Z \}$ . Hence,

$$
\left. \nabla H ( y ) , \rho ^ { \prime } - y \right. = \left. g - \log y , \rho ^ { \prime } - y \right. \leq \varepsilon \cdot \| \rho ^ { \prime } - y \| _ { 1 } \leq 2 \varepsilon ,
$$

where the first inequality uses Equation (12) and the second one uses the fact that the $\ell _ { 1 } { \mathrm { - } } \mathrm { d i s t a n c e }$ between two probability distributions is at most 2. Thus, $H ( \rho ^ { \prime } ) \leq H ( y ) + 2 \varepsilon$ , proving the lemma. □

## 3. Solving the optimization problem numerically

We wish to numerically minimize Ω from Equation (11) while satisfying all the constraints, with $q = 5$ and $\ell ^ { * } = 4$ . In Alman et al. (2025), this was done (for $\ell ^ { * } = 3 )$ via a sequential quadratic programming (SQP) algorithm using the software package SNOPT (Gill et al., 2002). In contrast, we take a gradient-based approach to tackle the non-convex minimization. We develop an algorithm applying several techniques from machine learning and adjacent areas, such as optimal transport.

## 3.1. Diferentiable objective

To apply a gradient-based optimization algorithm, we need a diferentiable objective function to be optimized. However, obtaining a diferentiable objective from Equation (11) is challenging due to the problem constraints. Since most of the parameters to be optimized are distributions, this creates the constraints that the probabilities must be non-negative and must add up to one. To sidestep that issue and obtain unconstrained optimization, we parameterize distributions in terms of their logits, and obtain the distribution by applying the softmax operation on the free parameters. We initialize the algorithm with random (logit) parameters.

The other relevant set of constraints comes from the maximum entropy distributions. Given an input probability distribution, we must find the distribution with maximum entropy that shares the same marginals as the input. This is reminiscent of a problem that typically arises in optimal transport, where the Sinkhorn-Knopp algorithm (Sinkhorn and Knopp, 1967) is used to obtain a stable and diferentiable objective (Cuturi, 2013). Thus, unlike Alman et al. (2025), we do not treat the maximum entropy distributions (and the corresponding Lagrange multipliers) as free parameters to be optimized together with the rest of distributions; rather, we obtain the maximum entropy distributions (and the Lagrange multipliers) using the Sinkhorn-Knopp algorithm.

We compute the gradients of the objective using automatic diferentiation. To improve stability, we employ implicit diferentiation for backpropagating through the Sinkhorn-Knopp algorithm (Cuturi et al., 2020; Eisenberger et al., 2022). We update the parameters using Adam (Kingma and Ba, 2015).

## 3.2. Software implementation

To obtain a fast algorithm that can leverage hardware platforms (such as GPUs) and scale up to level $\ell ^ { * } = 4$ (with nearly 7 million parameters to optimize) we implement our algorithm in Jax (Bradbury et al., 2018), applying some techniques to improve eficiency. The key to an efective speed-up is finding a new representation for the problem, switching from a loop over graph nodes (as done in the SQP implementation from Alman et al. (2025)) to a naturally parallelizable computation over tensors. In the implementation from Alman et al. (2025), the free parameters are stored in the nodes of a graph where updates happen through message-passing (both from child nodes to parent nodes and vice versa). The graph is non-uniform enough (e.g., diferent nodes have diferent number of children and diferent types of them) to make parallelization hard, but we solve this via two techniques. Firstly, we introduce phantom nodes in the graph with masking; as a trade-of, we pay the cost of increasing the number of optimization parameters by up to a factor of 3. Secondly, we cluster the nodes into a limited number of highly specialized groups (“stages”). With these techniques, we are able to represent the full graph with multi-dimensional tensors, enabling parallel processing over a number of axes (up to 10), challenging the limits of tensor-processing backends.

## 3.3. Applying AlphaEvolve

We use AlphaEvolve (Novikov et al., 2025) to further improve the optimization algorithm. Specifically, we let AlphaEvolve modify the optimization program, which is then executed (taking approximately 5 hours on a single GPU) to output a bound on omega. AlphaEvolve then evolves the code to minimize omega. We found improved results by using AlphaEvolve’s “evolving constructions” feature, where the optimization algorithm at each generation starts at the best solution point found by the parent algorithm.

## 4. Rigorous verification of the omega upper bound

To rigorously certify our omega bound, we run a separate verification step computing all quantities in rational arithmetic to guard against floating point errors. More specifically, we round the floating point solution obtained at the end of the optimization to rational numbers, ensuring the maximum-entropy certificates stay valid. We then evaluate all derived quantities in exact rational arithmetic, and replace each logarithm with a rational bound rounded in the proper direction that ensures each constraint in Equation (11) is satisfied, so the certified bounds are free from numerical errors. We are preparing a repository in which we will release the verification code and our discovered solution.

## 5. Discussion

In this note, we improve the SOTA upper bound on � to 2.371177. Our improvement is comparable in magnitude to most improvements in the last 40 years since � < 2.376 was attained by Coppersmith and Winograd (1990). We achieve this by leveraging modern optimization techniques and AlphaEvolve to design a better optimization algorithm for the problem described by Alman et al. (2025). While further modest improvements may be obtained in this manner, achieving larger improvements to � likely requires new mathematical ideas and is an exciting area of research.

Acknowledgments. The work presented in this note was motivated by initial conversations during the Complexity and Linear Algebra program at Simons Institute for the Theory of Computing in Fall 2025.

## References

J. Alman and V. V. Williams. A refined laser method and faster matrix multiplication. TheoretiCS, 3 (11261), 2024.

J. Alman, R. Duan, V. V. Williams, Y. Xu, Z. Xu, and R. Zhou. More Asymmetry Yields Faster Matrix Multiplication, pages 2005–2039. Society for Industrial and Applied Mathematics, 2025. doi: 10.1137/1.9781611978322.63. URL https://epubs.siam.org/doi/abs/10.1137/1. 9781611978322.63.

D. Bini, M. Capovani, F. Romani, and G. Lotti. O (�<sup>2.7799</sup>) complexity for � × � approximate matrix multiplication. Information Processing Letters, 8(5):234–235, 1979. ISSN 0020-0190.

M. Bläser. Fast matrix multiplication. Theory of Computing, pages 1–60, 2013.

J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/jax-ml/jax.

D. Coppersmith and S. Winograd. On the asymptotic complexity of matrix multiplication. In 22nd Annual Symposium on Foundations of Computer Science (sfcs 1981), pages 82–90, 1981.

D. Coppersmith and S. Winograd. Matrix multiplication via arithmetic progressions. Journal of Symbolic Computation, 9(3):251–280, 1990. ISSN 0747-7171. Computational algebraic complexity editorial.

M. Cuturi. Sinkhorn distances: lightspeed computation of optimal transport. In International Conference on Neural Information Processing Systems, pages 2292—-2300, 2013.

M. Cuturi, O. Teboul, J. Niles-Weed, and J.-P. Vert. Supervised quantile normalization for low-rank matrix approximation. In International Conference on Machine Learning, 2020.

R. Duan, H. Wu, and R. Zhou. Faster matrix multiplication via asymmetric hashing. In 2023 IEEE 64th Annual Symposium on Foundations of Computer Science (FOCS), pages 2129–2138, 2023. doi: 10.1109/FOCS57990.2023.00130.

M. Eisenberger, A. Toker, L. Leal-Taixé, F. Bernard, and D. Cremers. A unified framework for implicit Sinkhorn diferentiation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 499–508, 2022.

P. E. Gill, W. Murray, and M. A. Saunders. SNOPT: An SQP algorithm for large-scale constrained optimization. SIAM Journal on Optimization, 12(4):979–1006, 2002.

D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

F. Le Gall. Algebraic complexity theory and matrix multiplication. In Proceedings of the 39th International Symposium on Symbolic and Algebraic Computation, ISSAC ’14, page 23, New York, NY, USA, 2014. Association for Computing Machinery. ISBN 9781450325011.

A. Novikov, N. Vu, M. Eisenberger, E. Dupont, P.-S. Huang, A. Z. Wagner, S. Shirobokov, B. Kozlovskii,˜ F. J. R. Ruiz, A. Mehrabian, M. P. Kumar, A. See, S. Chaudhuri, G. Holland, A. Davies, S. Nowozin, P. Kohli, and M. Balog. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv, 2025. URL https://arxiv.org/abs/2506.13131.

V. Y. Pan. Strassen’s algorithm is not optimal trilinear technique of aggregating, uniting and canceling for constructing fast algorithms for matrix operations. In 19th Annual Symposium on Foundations of Computer Science, pages 166–176, 1978.

F. Romani. Some properties of disjoint sums of tensors related to matrix multiplication. SIAM Journal on Computing, 11(2):263–267, 1982.

A. Schönhage. Partial and total matrix multiplication. SIAM Journal on Computing, 10(3):434–455, 1981.

R. Sinkhorn and P. Knopp. Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of Mathematics, 21:343–348, 1967.

A. J. Stothers. On the complexity of matrix multiplication. Phd thesis, University of Edinburgh, Edinburgh, United Kingdom, 2010.

V. Strassen. Gaussian elimination is not optimal. Numerische mathematik, 13(4):354–356, 1969.

V. Strassen. The asymptotic spectrum of tensors and the exponent of matrix multiplication. In 27th Annual Symposium on Foundations of Computer Science, pages 49–54, 1986.

V. V. Williams. Multiplying matrices faster than Coppersmith-Winograd. In Proceedings of the Forty-Fourth Annual ACM Symposium on Theory of Computing, STOC ’12, page 887–898, New York, NY, USA, 2012. Association for Computing Machinery. ISBN 9781450312455.

V. V. Williams, Y. Xu, Z. Xu, and R. Zhou. New Bounds for Matrix Multiplication: from Al pha to Omega, pages 3792–3835. Society for Industrial and Applied Mathematics, 2024. doi: 10.1137/1.9781611977912.134. URL https://epubs.siam.org/doi/abs/10.1137/ 1.9781611977912.134.
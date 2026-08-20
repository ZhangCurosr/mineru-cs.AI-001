# The Role of Grid Cells in Reducing Spatial Aliasing in Hippocampal Place Representations

Alexander Johnson<sup>∗</sup>   
Dept. of Electrical   
and Computer Engineering   
University of Cincinnati   
Cincinnati, USA   
johns9a4@mail.uc.edu   
Obadah Ghizawi<sup>∗</sup>   
Dept. of Electrical   
and Computer Engineering   
University of Cincinnati   
Cincinnati, USA   
ghizawon@mail.uc.edu

Ali A. Minai, Senior Member IEEE

Dept. of Electrical

and Computer Engineering

University of Cincinnati

Cincinnati, USA

minaiaa@ucmail.uc.edu

Abstract—Spatial aliasing occurs when two or more distinct locations produce highly similar place-cell representations, primarily due to environmental symmetry or repetitive structures. This issue is most pronounced when place representations are constructed solely from boundary vector cell (BVC) inputs, because symmetric or repetitive structures can yield indistinguishable sensory patterns across multiple locations in an environment. This work introduces grid cell signals to mitigate spatial aliasing in such settings. Because grid cells contribute periodic, internally generated spatial signals that vary independently of environmental geometry, they play a key role in disambiguating perceptually identical locations. We integrate multiple modules of analytically constructed grid cells with BVC-driven place cells and show that this leads to a 94–99% reduction in spatial aliasing relative to a BVC-only baseline across three environments: an open environment without obstacles; an environment with a cross-shaped central obstacle creating high visual symmetry; and a maze environment. The greatest improvement occurs in the environment with the highest visual symmetry. These results indicate that grid cells provide information complementary to boundary-based inputs, yielding more reliable place representations in geometrically ambiguous environments.

Index Terms—grid cells, place cells, spatial navigation, spatial cognition

## I. INTRODUCTION

Biologically-inspired computational models of spatial cognition aim to replicate the navigational abilities observed in animals. These models are often derived from the hippocampal region of the brain, which is known to be central for spatial mapping and navigation in mammals [1]. Place cells are specialized neurons in the hippocampus that exhibit spatially localized regions of activity, known as place fields [2]–[4]. Place field firing patterns have been shown to depend on distal visual cues [5], and especially cues about the distance and location of walls and other objects mediated through boundary vector cells (BVCs) [6], [7]. However, maps built using only these cues are susceptible to spatial aliasing, in which the same place cells are activated at multiple spatial locations due to visual symmetries in the environment [8].

Grid cells are neurons in the medial entorhinal cortex (MEC), whose periodic spatial firing fields at different orienta-

tions and spatial frequencies form triangular lattices that cover any environment visited by an animal [9], [10]. According to Bush et al. [9], grid cells provide a context-dependent spatial metric for path integration and vector navigation, and these complementary signals interact with place cells to support reliable coding of large environments. In this paper, we show that one way that grid cells improve the accuracy of spatial representations is by reducing aliasing. They do this by introducing a secondary source of information, thereby disambiguating visually similar locations.

Note: The simulator used for this work is part of a larger hippocampal system simulation platform being developed for use with the Webots environment. It will be shared publicly once it is complete and described fully in a future report.

## II. BACKGROUND

## A. Place Cells

The first report of place-specific activity in hippocampal cells was by O’Keefe et al. in 1971 [2]. Place cells (PCs) are primarily located in the CA3 and CA1 regions of the hippocampus. PCs exhibit location-specific and boundaryaware firing, meaning that a given cell fires only when the animal is in a particular part of the environment (i.e., the place field) and maintains a tunable distance (and direction) from boundaries. For example, when a place cell is recorded in a number of rectangular environments, the peak response location of the PC often maintains a fixed distance from the two nearest walls [11]. A PC’s firing rate is modeled as a function of the animal’s location through the thresholded sum of firing rates of several inputs (traditionally, BVCs) [6]. Place cells are a critical component of the system that enable mammals to represent their location within a given environment.

## B. Head Direction Cells

Head direction cells (HDCs) were first recorded and characterized by Taube et al. in 1990. These cells are located in the postsubiculum (Brodmann area 48), a key region in the hippocampal formation. Each cell fires rapidly only when the animal’s head is pointing in a restricted range of angles, with the maximum activation occurring in approximately the middle of this range. This angle is called the preferred direction of the cell [12]. Together, activated head direction cells provide a population-coded signal indicating the way an animal is facing at any given time. Head direction cells are an important component of the hippocampus’s navigational circuitry.

## C. Boundary Vector Cells

Boundary vector cells (BVCs) were first predicted to exist through computational models using hippocampal place cells in 1996 [7], and later in 2000 [6], [13]. BVCs were then confirmed as a cell type in 2009 by Lever et al. [14]. A BVC fires whenever an environmental boundary intersects a receptive field located at a specific distance (i.e., preferred distance) from an animal in a specific allocentric direction. The firing of a BVC only depends on the animal’s location, rather than the animal’s heading [14]. BVCs are a key input to place cells (PCs), as the firing of a PC is a thresholded sum of the firing of the BVCs connected (via synapses) to it [6]. This means that BVCs provide critical information about surrounding boundaries and their directions to PCs, allowing PCs to activate at specific BVC configurations and, thus, for each PC to have an independent role in the navigational system.

## D. Grid Cells

Entorhinal grid firing patterns were first discovered by Hafting et al. in 2005 [10]. The dorsocaudal medial entorhinal cortex (dMEC) contains a topographically organized neural representation of space, with the key units being grid cells. A grid cell exhibits multiple spatial firing fields that form a periodic hexagonal lattice across the environment. It has been shown that grids of neighboring cells differ in terms of their vertex locations (i.e., their phases), with the spacing and size of individual fields increasing from dorsal to ventral dMEC [10]. Grid cells are a core component of the MEC, supporting path integration and enabling navigation even in the absence of external landmarks.

1) Grid Cell Modeling: A variety of computational models have been proposed to explain GC firing patterns. These can be broadly categorized as either dynamical or learning-based models. Dynamical models propose that grid patterns emerge from the temporal evolution of neural activity. For example, oscillatory inference models [15], [16] rely on interference patterns of theta-frequency oscillations, which are modulated by velocity. Another dynamical approach is to use continuous attractor networks (CANs) [17], where grid patterns emerge as stable activity “bumps” that shift in response to velocity input. Alternatively, learning-based approaches have demonstrated that grid-like representations can naturally emerge in deep neural networks trained on vector-based navigation [18].

As this work focuses on the functional utility of GCs rather than their biological origin, we adopt a direct analytical approach to their implementation. By explicitly constructing grid-like firing patterns using cosine interference, we achieve precise control over grid scale and orientation while maximizing computational efficiency. The goal is not to reproduce the physiology of grid cells, but to leverage their functional properties—namely, periodic, metrically consistent spatial information—to reduce place-field aliasing in complex, symmetric environments.

![](images/da989f3625412d7eb868f53c873bb387f1d1bdc7f3c77348f1536a370149c90c.jpg)  
Fig. 1: Example of spatial aliasing in a single place cell. In the Cross environment, this neuron exhibits multiple firing-field peaks (four distinct activation maxima), indicating ambiguous spatial encoding where separated locations elicit similar place cell responses.

## E. Spatial Aliasing

Spatial aliasing is a phenomenon where similar place rep resentations arise in visually similar locations. This problem is especially serious in models in which place cell activity is driven mainly by sensory inputs, such as BVC activity. While some spatial aliasing has been observed in animals, and even humans may occasionally be confused by similar-looking locations, hippocampal representations of different locations are usually quite distinct and can be used to localize the animal’s position in experimental setups [19]–[21]. Recent work in our group has shown that, when available, the 3- dimensional structure of the environment may help mitigate the problem [8]

In this paper, we present a more robust and universally applicable solution to spatial aliasing using grid cells.

## III. MODEL ARCHITECTURE

The model consists of four core layers, each modeled after its biological counterpart: Head Direction Cell (HDC), Boundary Vector Cell (BVC), Grid Cell (GC), and Place Cell (PC) layers. The HDC layer encodes the agent’s heading, with each cell tuned to a specific allocentric direction. The BVC layer encodes obstacle and boundary information, with each cell exhibiting preferred angular and distance tuning. The GC layer serves as an internal positional signal for the agent and is organized into discrete modules of grid cell populations. All modules share the same spatial scale but differ in phase offset. The PC layer serves as the model’s core spatial representation and receives weighted input from both the BVC and GC layers.

The HDC, BVC, and PC layers are adapted from previous models [22], [23].

![](images/3e44d454c61649701db3d4b723b2ffd1e3f7b25aeabed2458b6366a9a1b6426f.jpg)  
Fig. 2: Model architecture for spatial representation. Black arrows indicate excitatory pathways, while red arrows indicate inhibitory pathways. Positional input drives the grid cell network, compass input drives head direction cells, and lidar input drives boundary vector cells (BVCs). Grid cell and BVC signals provide afferent excitation to the place cell network, while afferent and recurrent inhibition regulate competition and sparsity. Head direction signals provide a shared orientation reference between grid and boundary representations.

The main contribution of this work is the addition of the GC layer, which provides a metric spatial coordinate system that complements the boundary-based representations and enables stable place field formation in environments with geometric ambiguity.

## A. Notation & Overview

For each of the four cell types—Head Direction (h), Boundary Vector (b), Grid $( g ) _ { ; }$ and Place (p)—we use the superscript symbols h, b, g, and $p ,$ respectively. The time-varying firing rate of the $i ^ { \mathrm { { t h } } }$ cell of type k is denoted by $v _ { i } ^ { k } ( t )$ (or simply $v _ { i } ^ { k } )$ . Similarly, the time-varying synaptic weight from the $j ^ { \mathrm { t h } }$ cell of type l to the $i ^ { \mathrm { { t h } } }$ cell of type k is denoted as $W _ { i j } ^ { k l } ( t )$ (or $W _ { i j } ^ { k l }$ when time indices are omitted). These definitions apply throughout all network modules and environments unless otherwise specified.

## B. Head Direction Cell Layer

The Head Direction Cell (HDC) layer provides a global orientation signal to the model, where each cell has a preferred allocentric direction. For simplicity, this model uses $N _ { h } = 8$ cells, whose preferred directions are fixed at $0 ^ { \circ } , 4 5 ^ { \circ } , \ldots , 3 1 5 ^ { \circ }$

Each HDC i has a preferred direction $\theta _ { i } ^ { h }$ and activates according to:

$$
v _ { i } ^ { h } ( t ) = \cos \bigl ( \theta ( t ) - \theta _ { i } ^ { h } \bigr ) ,\tag{1}
$$

where $\theta ( t )$ is the agent’s current global heading. This cosine tuning, derived from the dot product [cos $\theta ( t )$ , sin $\theta ( t ) ]$ [cos $\theta _ { i } ^ { h }$ , sin $\theta _ { i } ^ { h } ]$ , produces maximal response when the agent faces the cell’s preferred direction, decreasing smoothly with angular deviation. This formulation is adapted from Erdem and Hasselmo [24].

## C. Boundary Vector Cell Layer

The Boundary Vector Cell (BVC) layer provides geometric spatial information by encoding the agent’s relationship to environmental boundaries. This implementation follows the boundary vector formulation of Barry et al. [25].

Each BVC i is characterized by a preferred boundary displacement, specified by a radial distance $d _ { i }$ and an allocentric bearing $\phi _ { i }$ relative to the agent. The layer receives sensory input from LiDAR in the form of per-beam range measurements $\mathbf { r } ( t ) ~ = ~ \{ r _ { 1 } , r _ { 2 } , . . . , r _ { n _ { \mathrm { r e s } } } \}$ at fixed allocentric angles $\{ \theta _ { 1 } , \theta _ { 2 } , \ldots , \theta _ { n _ { \mathrm { r e s } } } \}$ , where $n _ { \mathrm { r e s } }$ denotes the angular resolution of the sensor (720 in this work).

Each BVC integrates evidence across all sensor beams by combining radial and angular Gaussian tuning functions. With tuning widths $\sigma _ { r }$ (distance) and $\sigma _ { \theta }$ (angle), the firing rate of BVC i is:

$$
v _ { i } ^ { b } ( t ) ~ = ~ \frac { 1 } { N _ { \mathrm { B V C } } } \sum _ { j = 1 } ^ { n _ { \mathrm { r e s } } } \Biggl ( \frac { \exp \bigl [ - \frac { \left( r _ { j } - d _ { i } \right) ^ { 2 } } { 2 \sigma _ { r } ^ { 2 } } \bigr ] } { \sqrt { 2 \pi } \sigma _ { r } } ~ \times ~ \frac { \exp \bigl [ - \frac { \left( \theta _ { j } - \phi _ { i } \right) ^ { 2 } } { 2 \sigma _ { \theta } ^ { 2 } } \bigr ] } { \sqrt { 2 \pi } \sigma _ { \theta } } \Biggr ) ,\tag{2}
$$

where $N _ { \mathrm { B V C } }$ is the total number of boundary vector cells. The normalization by $N _ { \mathrm { B V C } }$ ensures activations remain in a consistent range for different BVC population sizes. Preferred displacements $\left( d _ { i } , \phi _ { i } \right)$ are uniformly distributed to tile the boundary coding space: distances $d _ { i }$ span $[ 0 , r _ { \mathrm { m a x } } ]$ (the maximum LiDAR range), while bearings $\phi _ { i }$ cover $[ 0 , 2 \pi )$ This arrangement ensures comprehensive coverage of potential boundary configurations surrounding the agent.

## D. Grid Cell Layer

The introduction of grid cells into the model was motivated by the desire to address a key limitation of prior models: severe place-field aliasing in environments with symmetry or repeated geometry. Grid cells provide an internally generated representation of space by combining their hexagonal firing patterns. This periodic structure provides a unique spatial signature at every location. When observing population activity within a grid cell module—where cells share a common scale and orientation but differ in spatial phase—the combined pattern becomes highly distinctive. Importantly, this spatial code is not affected by environmental geometry in the same way as boundary-based input. Even in the presence of repeated geometry or symmetrical layouts, the relative phase of grid cell activity differs across locations, providing the information needed to disambiguate them.

This work uses an analytical approach to grid cell modeling: we directly construct the hexagonal periodic structure mathematically rather than simulating underlying neural mechanisms [26]. This provides computational efficiency and precise control over spatial properties while maintaining biological plausibility through post-processing and obstacleaware masking. The model implements $M \ = \ 8$ grid cell modules aligned with head direction cells at orientations $\alpha _ { m } \in \{ 0 ^ { \circ } , 4 5 ^ { \circ } , \ldots , 3 1 5 ^ { \circ } \}$ , each containing $N _ { m } = 5 0$ cells sharing a common spatial scale λ but differing in spatial phase offset $\phi _ { j }$

1) Analytical Grid Cell Construction: For a grid cell with orientation α, spatial scale λ, and phase offset $\phi = ( \phi _ { x } , \phi _ { y } )$ we define the spatial frequency $\omega ~ = ~ 2 \pi / \lambda$ . The position $\mathbf { x } = ( x , y )$ is rotated into the grid cell’s reference frame and translated by its phase offset:

$$
{ \binom { r _ { x } } { r _ { y } } } = { \binom { \cos \alpha } { - \sin \alpha } } \ { \sin \alpha } ) { \binom { x - \phi _ { x } } { y - \phi _ { y } } } .\tag{3}
$$

The canonical grid pattern is constructed as the mean of three cosine gratings oriented $6 0 ^ { \circ }$ apart:

$$
\begin{array} { l } { { \displaystyle { G ( { \bf x } ) = \frac { 1 } { 3 } \Big [ \cos ( \omega r _ { x } ) } } \ ~ } \\ { { \displaystyle ~ + \cos \Big ( \omega \big ( \frac { r _ { x } } { 2 } + \frac { \sqrt { 3 } } { 2 } r _ { y } \big ) \Big ) } \ ~ } \\ { { \displaystyle ~ + \cos \Big ( \omega \big ( \frac { r _ { x } } { 2 } - \frac { \sqrt { 3 } } { 2 } r _ { y } \big ) \Big ) \Big ] } . } \end{array}\tag{4}
$$

To achieve the sharply tuned firing fields observed in biological recordings [10], three post-processing steps are applied. First, a power-law transformation adjusts the activation profile:

$$
G ^ { \prime } ( \mathbf { x } ) = \operatorname { s g n } ( G ) \cdot | G | ^ { 1 / \beta } ,\tag{5}
$$

where $\beta \in [ 1 . 2 , 1 . 8 ]$ controls field sharpness. Second, percell min-max normalization ensures consistent dynamic range. Third, soft thresholding creates sparse activations:

$$
v ^ { g } ( \mathbf { x } ) = \operatorname* { m a x } \Big ( 0 , \frac { G ^ { \prime \prime } ( \mathbf { x } ) - \theta _ { \mathrm { g c } } } { 1 - \theta _ { \mathrm { g c } } } \Big ) ,\tag{6}
$$

where $\theta _ { \mathrm { g c } }$ controls sparsity.

2) Obstacle-Aware Masking: Analytical construction based on position alone produces activations that extend across walls, contradicting biological observations where grid cells fragment at barriers while maintaining phase coherence [27]. We address this through a spatial masking procedure applied independently to each grid cell.

The procedure identifies connected activation components (“blobs”) and determines which intersect obstacles. For blobs where obstacles cover $\ge ~ 2 0 \%$ of their diameter, we apply obstacle masks to split or suppress the activation. When a blob splits into multiple components, only the largest is retained. Blobs that remain connected after initial masking are processed with dilated obstacle boundaries to force fragmentation. Finally, Gaussian smoothing $( \sigma \approx 0 . 5 )$ restores biologically plausible curved edges that taper near walls rather than exhibiting sharp cutoffs.

Figure 3 illustrates this effect: raw activations extend continuously across walls (left), while masked activations fragment at boundaries while preserving local phase coherence (right).

![](images/5f79f8de62268074ddbf2c442a1c3cae7f475e94bcbafebab6f42d02ba6039aa.jpg)

![](images/f24854a49b42efebcf5c1eea9827e9d61857b08f3fc5e64a5efa585fcec6ac47.jpg)  
(a) Raw (no masking)  
(b) Masked (boundary-aware)  
Fig. 3: Boundary-aware masking for grid cell activations in an obstacle-filled environment. Without masking (left), activations extend across walls. With masking (right), activations fragment at barriers while preserving local phase coherence within each compartment.

## E. Place Cell Layer

The Place Cell (PC) layer forms the core spatial representation of the model. Place cells develop spatially localized receptive fields (place fields) through competitive learning, with each cell encoding a specific location in the environment. This layer integrates information from both BVC and GC layers: BVCs provide geometric constraints from environmental boundaries, while GCs contribute metric phase information that disambiguates perceptually similar locations. The relative contribution of these two input streams is controlled by a weighting parameter $\eta ~ \in ~ [ 0 , 1 ]$ , where BVC inputs are weighted by $( 1 - \eta )$ and GC inputs by η.

1) Membrane Dynamics and Activation: The membrane potential $s _ { i } ^ { p }$ of place cell i evolves as a leaky integrator combining excitatory input from BVCs and GCs with global inhibition:

$$
\tau _ { p } \frac { d s _ { i } ^ { p } } { d t } = - s _ { i } ^ { p } + E _ { i } - I _ { i } ,\tag{7}
$$

where the excitatory drive is

$$
E _ { i } = ( 1 - \eta ) \sum _ { j } W _ { i j } ^ { p b } { v } _ { j } ^ { b } + \eta \sum _ { k } W _ { i k } ^ { p g } { v } _ { k } ^ { g } ,\tag{8}
$$

and the inhibitory drive is

$$
I _ { i } = ( 1 - \eta ) \Gamma ^ { p b } \sum _ { j } v _ { j } ^ { b } + \eta \Gamma ^ { p g } \sum _ { k } v _ { k } ^ { g } + \Gamma ^ { p p } \sum _ { l } v _ { l } ^ { p } .\tag{9}
$$

Here, $W _ { i j } ^ { p b }$ and $W _ { i k } ^ { p g }$ are synaptic weights from BVCs and GCs to place cell $i ; v _ { j } ^ { b } , v _ { k } ^ { g } .$ , and $v _ { l } ^ { p }$ are the firing rates of BVCs, ${ \mathrm { G C s } } ,$ and PCs respectively; and $\Gamma ^ { p b }$ , Γ<sup>pg</sup>, Γ<sup>pp</sup> are inhibitory gain parameters that control the strength of afferent (boundary and grid) and recurrent (place-to-place) inhibition. The firing rate is obtained through rectification and saturation:

$$
v _ { i } ^ { p } = \operatorname { t a n h } \bigl ( [ \psi s _ { i } ^ { p } ] _ { + } \bigr ) ,\tag{10}
$$

where $\psi$ is a gain factor and $[ \cdot ] _ { + } = \operatorname* { m a x } ( 0 , \cdot )$ . This nonlinearity produces sparse, localized place fields through the combined effects of global inhibition and thresholding.

![](images/852e0adbd2946aa8adb5ecdd13ed59854ee92c81117276bd046c86213c506c0f.jpg)  
(a) Open

![](images/840c3d7e0155d5be58ac58800f5817f3090883d28489d8a498d7a32530175dbb.jpg)  
(b) Cross

![](images/960a8df8c4fd06c64ccb64947f477a72ca3ccebe44170349720c773adee40f72.jpg)  
(c) Maze  
Fig. 4: Evaluation environments used in experiments: (a) Open, (b) Cross, and (c) Maze. All environments are 20m x 20m in size.

2) Self-Organization via Competitive Learning: Place fields emerge through competitive learning during exploration. Synaptic weights $W _ { i j } ^ { p b }$ and $W _ { i k } ^ { p g }$ are initialized sparsely with connection probabilities of approximately 0.25 and 0.30, respectively, ensuring each place cell receives input from a distinct subset of BVCs and GCs. During learning, weights are updated according to Oja’s rule [22], [23], [28]:

$$
{ \tau _ { w ^ { p b } } } \frac { d W _ { i j } ^ { p b } } { d t } = \left( 1 - \eta \right) { v _ { i } ^ { p } } \left( { v _ { j } ^ { b } } - \frac { 1 } { { \alpha _ { p b } } } { v _ { i } ^ { p } } W _ { i j } ^ { p b } \right) ,\tag{11}
$$

$$
\tau _ { w ^ { p g } } \frac { d W _ { i k } ^ { p g } } { d t } = \eta v _ { i } ^ { p } \biggl ( v _ { k } ^ { g } - \frac { 1 } { \alpha _ { p g } } v _ { i } ^ { p } W _ { i k } ^ { p g } \biggr ) ,\tag{12}
$$

where $\tau _ { w ^ { p b } } , \tau _ { w ^ { p g } }$ are learning time constants and $\alpha _ { p b } , \alpha _ { p g }$ are normalization factors that control the strength of weight decay. The first term in each equation strengthens synapses when pre- and postsynaptic cells are co-active (Hebbian learning), while the second term normalizes total synaptic input and enforces competition among place cells. Combined with global inhibition $( I _ { i } )$ , this produces winner-take-all dynamics where each place cell specializes to represent a unique combination of BVC and GC activity patterns, forming a spatially localized place field.

## IV. EXPERIMENTAL SETUP

All experiments were conducted using an RTX 3090 GPU and a Ryzen 9 9900X CPU. The simulation software used to run the experiments was Webots R2025a. The agent used across the experiments was a Roomba bot with a compass for head-direction updates and a rangefinder with 720 beams covering 360 degrees, which provided information directly to the BVC layer. The rangefinder has a maximum distance of 25m, to account for the maximum distance that would need to be read in a 20 m × 20 m environment. All environments used are 20 m × 20 m in size.

## A. Data Collection

The agent explored three environments of varying complexities. The first is an open environment without obstacles, the second features cross-shaped walls that divide it into four regions, and the last is modeled as a maze, as shown in Fig. 4. To demonstrate the effects of grid cells in mitigating aliasing, 5 trials were conducted for each environment with grid cells, and another set of 5 trials was conducted without grid cells. The results across these trials were averaged for each case tested. Consequently, 10 trials were conducted for each environment, yielding a total of 30 trials. In a given trial, there were training and evaluation phases. During the training phase, the agent randomly explored the environment until it had covered at least 95 percent of it. This was calculated by dividing the environment into a square grid with a bin size of 0.5m and tracking the areas visited until the grid’s coverage reached 95 percent. Bins that overlap with obstacles were not included in this calculation. During the evaluation phase, the agent explored until it reached 99 percent coverage, with a bin size of 0.2m, matching the bin size used to calculate the Mean Spatial Aliasing Index (MSAI) metric (see below).

## B. Model Parameters

TABLE I: Neural Cell Populations Used in the Model
<table><tr><td>Cell Type</td><td>Number of Cells</td></tr><tr><td>Place Cells</td><td>1000</td></tr><tr><td>Boundary Vector Cells (BVCs)</td><td>400</td></tr><tr><td>Head Direction Cells</td><td>8</td></tr><tr><td>Grid Cells</td><td>400</td></tr></table>

Table I summarizes the number of cells used for each of the respective cell types within the experiments.

TABLE II: Boundary Vector Cell Parameters
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Radial Tuning Width  $( \sigma _ { r } )$ </td><td> $1 . 0 \mathrm { m }$ </td></tr><tr><td>Angular Tuning Width  $\left( \sigma _ { \theta } \right)$ </td><td> $3 . 0 ^ { \circ }$ </td></tr><tr><td>BVCs per Direction</td><td>50</td></tr><tr><td>Total Number of BVCs</td><td>400</td></tr></table>

Table II summarizes the BVC parameters used within the experiments. Along each head direction, 50 BVCs were used. As summarized in Table I, there are 8 head directions, meaning a total of $5 0 \times 8 = 4 0 0 ~ \mathrm { B V C s }$ were used in the model.

TABLE III: Grid Cell Parameters
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Number of Modules</td><td>8</td></tr><tr><td>Cells per Module Total Number of Grid Cells</td><td>50 400</td></tr><tr><td>Spatial Scale (λ)</td><td>5.5</td></tr><tr><td>Grid Influence Weight (η)</td><td>0.35</td></tr></table>

Table III summarizes the grid cell parameters used within the experiments. Grid cells were divided into 8 modules, with 50 cells per module, resulting in a total of $5 0 ~ \times ~ 8 ~ = ~ 4 0 0$ grid cells in the model. A spatial scale of $\lambda = 5 . 5$ yields an activation diameter of roughly 2.7m. It is important to note that during the ”no grid cell” experiments, η is set to 0 to eliminate any effect of grid cells on the model.

TABLE IV: Place Cell Learning & Inhibition Parameters
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td rowspan="3">BVC Normalization Factor  $( \alpha _ { p b } )$  GC Normalization Factor  $( \alpha _ { p g } )$  Recurrent Inhibition Gain (ΓpP)</td><td> $\overline { { { \sqrt { 0 . 4 } } \approx 0 . 6 3 2 } }$ </td></tr><tr><td> $\sqrt { 0 . 4 } \approx 0 . 6 3 2$ </td></tr><tr><td>0.70</td></tr><tr><td>BVC Afferent Inhibition Gain  $( \Gamma ^ { p b } )$ </td><td>0.35</td></tr><tr><td colspan="2">GC Afferent Inhibition Gain (ΓP9) Learning Time Constant  $( \tau _ { w } )$  10 timesteps (960 ms)</td></tr></table>

Table IV summarizes the place cell learning and inhibition parameters. Both normalization factors were set to $\sqrt { 0 . 4 }$ , and the afferent inhibitory gains for BVC and GC inputs were each set to half the recurrent inhibition gain.

## C. Metrics for Spatial Aliasing

Spatial aliasing was measured using the Spatial Aliasing Index (SAI) and Mean Spatial Aliasing Index (MSAI) metrics to examine how place cells responded in different regions of an environment. For a bin i located at $( x _ { i } , y _ { i } )$ , the SAI is given by:

$$
\mathrm { S A I } ^ { ( i ) } = \frac { 1 } { N } \sum _ { \stackrel { j = 1 } { j \neq i } } ^ { N } \mathbf { 1 } \{ \| ( x _ { i } , y _ { i } ) - ( x _ { j } , y _ { j } ) \| > d _ { \mathrm { t h } } \}\tag{13}
$$

where:

• $N$ is the total number of bins.

$d _ { \mathrm { t h } }$ is the distance threshold to exclude nearby bins.

$\mathbf { 1 } \{ \cdot \}$ is the indicator function (equal to 1 if the distance between bins exceeds $d _ { \mathrm { t h } } .$ , and 0 otherwise).

$\mathrm { C o s S i m } ( \mathbf { a } ^ { ( i ) } , \mathbf { a } ^ { ( j ) } )$ is the cosine similarity between the place cell activation vectors in bins i and $j .$

To evaluate the model’s overall aliasing performance, the MSAI is computed as the average of the SAI across all bins:

$$
\mathrm { M S A I } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { S A I } ^ { ( i ) }\tag{14}
$$

Larger values of $\mathrm { S A I } ^ { ( i ) }$ indicate greater spatial aliasing. This means that spatially distant bins exhibit similar place-cell activation patterns. The MSAI aggregates this effect across the environment, with higher MSAI values suggesting more aliasing. Conversely, lower $\mathrm { S A I } ^ { ( i ) }$ and MSAI values indicate stronger place-cell localization and improved spatial discrimination.

## V. RESULTS & DISCUSSION

TABLE V: Mean Spatial Aliasing Index $( \mathrm { M S A I } , \ \times 1 0 ^ { - 4 } )$ across environments with and without grid cells (GC). Values report mean ± standard deviation over five independent trials.
<table><tr><td>Environment</td><td>GC</td><td>No GC</td><td>Improvement</td></tr><tr><td>Open</td><td> $0 . 3 \pm 0 . 1$ </td><td> $5 . 7 \pm 0 . 3$ </td><td>94.73%</td></tr><tr><td>Cross</td><td> $0 . 3 \pm 0 . 1$ </td><td> $4 2 . 4 \pm 1 . 0$ </td><td>99.29%</td></tr><tr><td>Maze</td><td> $1 . 3 \pm 2 . 2$ </td><td> $3 8 . 9 \pm 2 . 7$ </td><td>96.65%</td></tr></table>

![](images/09adb69fdca102cd46ab1b6f8814fae840e755624fb277f4ab5f1c141f8c8523.jpg)  
Fig. 5: Spatial Aliasing Index (SAI) heatmaps with and without grid cells across environments. Rows show Open, Cross, and Maze (top to bottom); columns show with grid cells (left) and without grid cells (right). Color indicates aliasing magnitude on a logarithmic scale (lower is better). Reported Mean Spatial Aliasing Index (MSAI) values are shown beneath each heatmap.

This section summarizes the results of the model’s performance with and without grid cells in three different environments: Open, Cross, and Maze, as shown in Fig. 4. A total of five trials were conducted for each configuration (with and without grid cells) in each environment, and the MSAI scores were averaged for the five trials. The results are shown in Table V.

Examples of MSAI heatmaps from one trial for all three environments are shown in Fig. 5. Consistent with the averaged MSAI values in Table V, the heatmaps show lower SAI across most spatial bins when grid-cell input is included, with the most pronounced reduction in the Cross environment. These visualizations complement the results shown in Table V by localizing where aliasing is mitigated within each environment.

An important detail to consider is how prone a particular environment is to aliasing when using place representations based purely on boundary vector cells (BVC). The “Cross” environment, for example, divides the map into four rotationally symmetric quadrants, as shown in Fig. 4. This symmetry creates sensory similarity within the quadrants, increasing the potential for place-field aliasing. In models where place representations are constructed primarily from BVC inputs, such environments are therefore especially prone to aliasing [29].

Adding a source of path integration signals to the model, namely grid cells, substantially improves the disambiguation of these perceptually similar regions, as evidenced by a reduction of approximately 99.3% in the averaged Mean Spatial Aliasing Index (MSAI) in five trials in the Cross environment.

The Open environment exhibits more moderate baseline aliasing due to fewer repeated boundary configurations. The Maze environment shows substantial but more variable improvement: although grid cells mitigate much of the BVCinduced aliasing, corridors introduce repeated spatial patterns, which can cause distant locations to retain moderate similarity in their place-cell representations, contributing to higher and more variable MSAI values.

## VI. CONCLUSION & FUTURE WORK

Overall, our findings demonstrate that adding a source of path integration signals, namely grid cells, helps mitigate the degree of aliasing exhibited in place-cell representations across environments with varying degrees of structural symmetry. Future work will investigate the use of an attractor-based grid cell model and whether its biological plausibility introduces different trade-offs in aliasing mitigation and stability. In particular, it would be valuable to implement 3D attractorbased grid cells (accompanied by other relevant 3D cells, such as BVCs, place cells, and head direction cells) [8], and examine the unique challenges that may arise by testing this implementation in varied, real-world environments. Such extensions would help clarify the challenges associated with scalable, biologically grounded spatial representations and move the model toward greater robustness in realistic navigation settings.

## ACKNOWLEDGMENTS

The authors thank Andrew Gerstenslager and Bekarys Dukenbaev for helpful discussions and explanations of previous models. All results and analyses presented in this paper are solely the work of the authors.

## REFERENCES

[1] Edvard I. Moser, Emilio Kropff, and May-Britt Moser. Place cells, grid cells, and the brain’s spatial representation system. Annu. Rev. Neurosci., 31(1):69–89, 2008.

[2] John O’Keefe and Jonathan Dostrovsky. The hippocampus as a spatial map. preliminary evidence from unit activity in the freely-moving rat. Brain Research, 34(1):171–175, 1971.

[3] John O’keefe and Lynn Nadel. The hippocampus as a cognitive map. Oxford university press, 1978.

[4] Phillip J. Best, Aaron M. White, and Ali Minai. Spatial processing in the brain: the activity of hippocampal place cells. Annual review of neuroscience, 24(1):459–486, 2001.

[5] Robert U. Muller and John L. Kubie. The effects of changes in the environment on the spatial firing of hippocampal complex-spike cells. Journal of Neuroscience, 7(7):1951–1968, 1987.

[6] Tom Hartley, Neil Burgess, Colin Lever, Francesca Cacucci, and John O’Keefe. Modeling place fields in terms of the cortical inputs to the hippocampus. Hippocampus, 10(4):369–379, 2000.

[7] John O’Keefe and Neil Burgess. Geometric determinants of the place fields of hippocampal neurons. Nature, 381(6581):425–428, 1996.

[8] Andrew Gerstenslager, Bekarys Dukenbaev, and Ali A. Minai. Improved accuracy of robot localization using 3-d lidar in a hippocampus-inspired model, 2025. Presented at the 2025 International Joint Conference on Neural Networks, Rome, July 2025.

[9] Daniel Bush, Caswell Barry, and Neil Burgess. What do grid cells contribute to place cell firing? Trends in Neurosciences, 37(3):136–145, 2014.

[10] Torkel Hafting, Marianne Fyhn, Sturla Molden, May-Britt Moser, and Edvard I. Moser. Microstructure of a spatial map in the entorhinal cortex. Nature, 436(7052):801–806, 2005.

[11] Neil Burgess and John O’Keefe. Neuronal computations underlying the firing of place cells and their role in navigation. Hippocampus, 6(6):749–762, 1996.

[12] Jeffrey S. Taube, Robert U. Muller, and James B. Ranck. Head-direction cells recorded from the postsubiculum in freely moving rats. i. description and quantitative analysis. Journal of Neuroscience, 10(2):420–435, 1990.

[13] Neil Burgess, Andrew Jackson, Tom Hartley, and John O’Keefe. Predictions derived from modelling the hippocampal role in navigation. Biological cybernetics, 83(3):301–312, 2000.

[14] Colin Lever, Stephen Burton, Ali Jeewajee, John O’Keefe, and Neil Burgess. Boundary vector cells in the subiculum of the hippocampal formation. Journal of Neuroscience, 29(31):9771–9777, 2009.

[15] Neil Burgess, Caswell Barry, and John O’Keefe. An oscillatory interference model of grid cell firing. Hippocampus, 17(9):801–812, 2007.

[16] Michael E. Hasselmo, Lisa M. Giocomo, and Eric A. Zilli. Grid cell firing may arise from interference of theta frequency membrane potential oscillations in single neurons. Hippocampus, 17(12):1252–1271, 2007.

[17] Yoram Burak and Ila R Fiete. Accurate path integration in continuous attractor networks of grid cells. PLoS Computational Biology, 5(2):e1000291, 2009.

[18] Andrea Banino, Caswell Barry, Benigno Uria, Charles Blundell, Timothy Lillicrap, Piotr Mirowski, Alexander Pritzel, Martin J Chadwick, Thomas Degris, Joseph Modayil, et al. Vector-based navigation using grid-like representations in artificial agents. Nature, 557(7705):429–433, 2018.

[19] Thomas J. Davidson, Fabian Kloosterman, and Matthew A. Wilson. Hippocampal replay of extended experience. Neuron, 63(4):497–507, 2009.

[20] Margaret F. Carr, Shantanu P. Jadhav, and Loren M. Frank. Hippocampal replay in the awake state: a potential physiological substrate of memory consolidation and retrieval. Nature Neuroscience, 14(2):147–153, 2011.

[21] Georg Dragoi and Susumu Tonegawa. Preplay of future place cell sequences by hippocampal cellular assemblies. Nature, 469(7330):397– 401, 2011.

[22] Adedapo Alabi, Ali A. Minai, and Dieter Vanderelst. One shot spatial learning through replay in a hippocampus-inspired reinforcement learning model. In 2020 International Joint Conference on Neural Networks, pages 1–8. IEEE, 2020.

[23] Adedapo Alabi, Dieter Vanderelst, and Ali A. Minai. Rapid learning of spatial representations for goal-directed navigation based on a novel model of hippocampal place fields. Neural Networks, 161:116–128, 2023.

[24] Ugur M. Erdem and Michael E. Hasselmo. A goal-directed spatial˘ navigation model using forward trajectory planning based on grid cells. European Journal of Neuroscience, 35(6):916–931, 2012.

[25] Caswell Barry, Colin Lever, Robin Hayman, Tom Hartley, Sarah Burton, John O’Keefe, Kathryn Jeffery, and Neil Burgess. The boundary vector cell model of place cell firing and spatial memory. Hippocampus, 16(9):765–784, 2006.

[26] Hugh T. Blair, Adam C. Welday, and Kechen Zhang. Scale-invariant memory representations emerge from moire interference between grid´ fields that produce theta oscillations: a computational model. Journal of Neuroscience, 27(12):3211–3229, 2007.

[27] Dori Derdikman, Jonathan R. Whitlock, Albert Tsao, Marianne Fyhn, Torkel Hafting, May-Britt Moser, and Edvard I. Moser. Fragmentation of grid cell maps in a multicompartment environment. Nature Neuroscience, 12(10):1325–1332, 2009.

[28] E. Oja. Simplified neuron model as a principal component analyzer. Journal of Mathematical Biology, 15(3):267–273, 1982.

[29] William E. Skaggs and Bruce L. McNaughton. Spatial firing properties of hippocampal ca1 populations in an environment containing two visually identical regions. Journal of Neuroscience, 18(20):8455–8466, 1998.
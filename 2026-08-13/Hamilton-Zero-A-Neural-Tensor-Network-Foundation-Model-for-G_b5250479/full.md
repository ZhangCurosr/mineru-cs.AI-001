# Hamilton-Zero: A Neural Tensor-Network Foundation Model for Ground States of Arbitrary Quadratic Qubit Hamiltonians

Timothy Heightman<sup>1,2,∗</sup> Elena Orlova<sup>2</sup> Philip Mantrov<sup>2</sup> Aleksei Ustimenko<sup>2,∗</sup>

<sup>1</sup>ICFO – Institut de Ci\`encies Fot\`oniques, The Barcelona Institute of Science and Technology,

08860 Castelldefels, Barcelona, Spain

<sup>2</sup>Simulacra Research Inc., London, UK & Chicago, USA

timothyheightman@simulacra-ai.com aleksei@simulacra-ai.com

## Abstract

A central promise of useful quantum advantage is the ability to compute ground states of Hamiltonian systems beyond the reach of classical simulation methods. Here we demonstrate that this problem can be efectively amortized across an arbitrary and universal set of Hamiltonians by a foundation model with ∼ 0.5B variational parameters, trained with contemporary techniques from large language models and deep reinforcement learning. To do this, we formulate spin-1/2 quantum ground-state learning as manifold variational optimisation over centrally odd scalar functions on SU(2)<sup>N</sup> . This replaces explicit Hilbert-space vector amplitudes with manifold functions on which the Hamiltonian acts through Lie derivatives, evaluated by custom automatic diferentiation primitives. We prove that the resulting variational principle on this manifold preserves the spin-1/2 sector’s ground-state upper bound using the Peter-Weyl theorem, then pre-train our foundation model on a dataset of hundreds of thousands of diferent Hamiltonian systems, varying the connection topology, system size, interaction types and strengths, bringing together a century of many-body literature. Using a novel SU(2) replica-exchange Langevin sampler and sharded natural-gradient optimisation, we train our model with our own extension of the Kronecker-Factored Approximate Curvature (KFAC) optimiser on system sizes up to 64 qubits. On a held-out generalisation dataset, we fine-tune our model on system sizes of up to 1024 qubits, and evaluate on systems up to 8100 qubits.

## Contents

1 Introduction 2   
2 Theory 4   
3 Hamilton-Zero’s Architecture 6   
4 Pretraining 8   
5 Results 9   
6 Discussion and outlook 23

## Supplementary Material

S1 Variational principles on Spinor Manifolds 26   
S2 Foundation Ansatz and Entanglement Structure via Reinforcement Learning 32   
S3 Datasets and further detail on training 66   
S4 Optimization and Engineering 70   
A The Casimir regulariser for nonlinear ans¨atze 79   
B Proofs from Supp. Mat. S1 81

## 1 Introduction

A central problem in quantum many-body systems is computing the ground state of a given spin-1/2 Hamiltonian. Through a fermion-to-qubit mapping such as Jordan-Wigner, it encodes the electronicstructure Hamiltonians of quantum chemistry, and with them this ground state problem is relevant from materials design through to drug discovery [1–3]. Ground states across a Hamiltonian family also trace out its phase diagram, as well as encode the solutions of combinatorial optimisation problems relevant to supply-chain [4], portfolio optimisation [5], electric vehical charging [6, 7] or flight routing [8] to name some examples [9, 10].

For a large many-body system, however, computing such ground states has proven famously dificult due to the curse of dimensionality. This curse refers to the fact that a spin- <sup>1</sup> Hamiltonian on N sites has a ground state in $2 ^ { N }$ complex dimensions. Decades of research have uncovered many complementary methods to solve this problem as generally as possible. Of note are tensor-network methods [11–13], variational quantum algorithms (VQAs) [14, 15], and neural quantum states (NQS) [16, 17]. The first is the natural tool in one dimension, where the ground states of gapped local Hamiltonians obey an area law and a fixed bond dimension captures them. In two dimensions, that same area law turns against the method, since the entanglement across a region’s boundary, and with it the bond dimension required, grows with that boundary, confining eficient tensor networks to a low-entanglement corner of Hilbert space. The second, VQAs, has proved initially successful for smaller many-body systems, but has trainability and expressivity tradeofs due to the so-called barren plateau phenomenon [18–20]. As such, the third method, NQS, has taken of in popularity over the last years, solving for variational upper-bounds to ground states on large systems , with the largest systems approaching 1500 − 2000 qubits [21]. This accuracy on single systems alone however has not, until recently, proven transferable across diferent Hamiltonians [22].

Indeed, recent works aiming for this transferrability constitute early foundation neural quantum states (FNQS), in which a single neural quantum state is trained on many diferent hamiltonians [22]. This amortises the cost of training a new network from scratch at the per-system level. Thus far however, the community has gone in a direction where they fix the Hamiltonian’s interaction graph (i.e. the topology of the connections between spin sites), varying only the coeficients of a given Hamiltonian structure, and keeping both said structure fixed as well as the number of spins [22, 23], see Figure 1.

This lineage is short but fast moving and highly active area of investigation. The transformer quantum state of Zhang and Di Ventra [24] was the first to carry a single network across many instances of one model class. Rende and collaborators [25] then pretrained at one point of a phase diagram and fine-tuned outward from it. The same group made the idea systematic in their foundation neural quantum states [26], a single network conditioned on the couplings of one lattice and swept across its phase diagram and disorder. The same line later settled a two-dimensional spin glass [27], and the construction carried over to lattice fermions [23]. The cost these models amortise is that of deep variational Monte-Carlo. This per-system paradigm exemplified in the electronic setting by FermiNet, PauliNet and DeepErwin [28–30], each of which solves a single system from scratch to high accuracy, at a price paid anew for every Hamiltonian. Several of the second-order optimisation primitives we build on below originate there. Within the spin-FNQS lineage reviewed above however, prior models condition primarily on the coeficients of a fixed interaction graph, with the only reconfigurable such variational models being the variational architectures on quantum computers, which are themselves also trained from scratch for every new instance of a problem.

Two neighbouring programmes do vary the structure, though outside the qubit paradigm considered here. The Large Electron Model [31] and its successor QERNEL [32] are foundation models for interacting electrons which generalise across particle number. QERNEL further conditions on the depth of the moir´e superlattice potential, the single parameter that tunes a semiconductor heterobilayer between its liquid and crystalline electronic states. The molecular wavefunction programme reaches furthest of all on structure. It runs from weight sharing across geometries [33, 34] to the joint-molecule ansatz of Gao and G¨unnemann [35], and on to Orbformer [36], pretrained across 22,000 molecular structures. These genuinely change the structure they condition on. Yet the type interaction remains fixed even as the geometric structure varies owing to the nature of the systems modelled. That is, in these models, we see only the Coulomb for the electrons, and a single symmetry class for the spins. Indeed, this obstruction is mechanical. A Hamiltonian read in as a fixed-length vector of couplings has fixed its interaction graph in the length and the meanings of the encoding of that vector. It cannot then be conditioned on a diferent type of interaction, nor a diferent number of spins. To cross from the coeficients to the structure, the Hamiltonian must enter as a variable and typed interaction graph. To-date, no such model has proven successful with such an architectural choice. Two final threads sit just of this map, neither a transfer model in itself. The first is the per-system transformer wavefunction of Roca-Jerat and collaborators [37]. The second is the neural scaling laws of Jiang et al. [38], our empirical reason to expect pretraining at the scale of hundreds of thousands of diferent Hamiltonians to amortize, also more recently reported in [39].

![](images/88f5961757243f9f3ba74a1f7b70d3546bc9be6aacdfdd2f66a8ba1f5a3b02ee.jpg)  
Figure 1: The generalisation surface of foundation wavefunction models. Each bar is one pretrainedmodel family. Its length is how the model varies its coupling across its training and evaluation corpus, with the axes ordered by structural depth. Solid bars mark demonstrated generalisation, hatched bars partial or model-dependent reach, with each reviewed in the main text.

In this contribution, we introduce Hamilton-Zero, a novel foundation model that generalises across both multiple system sizes, diferent interaction topologies, and diferent interaction types, trained on a corpus of Hamiltonian systems spanning a century of quantum many-body literature. This pretraining corpus has two levels of structure. At the top level sit thousands of distinct quadratic interaction graphs. Within each graph family, we then vary both the interaction parameters, so that a single family’s data is comparable in scope to the entire corpus of an existing FNQS model, and the number of spins, up to N = 64. Combined with a hot-swapped perturbation scheme in pretraining, these expansions yield the hundreds of thousands of distinct training Hamiltonian topologies, system sizes and interaction types.

For larger systems up to 8000 qubits, in many instances our results ofer energy bounds that are not accessible by tensor network methods, or DMRG due to the nature of the non-local interaction topologies made accessible by this model. Indeed, the only other architecture that could currently find these bounds is neural quantum states, trained from scratch on each instance of Hamiltonian interaction topology. However, this is amortised by the open-source release of this model’s weights.

To do so, we work in a diferent configuration space for many-body spin systems, akin to the many-body phase space formalism of [40] with two notable improvements. First, we represent each factor $\mathrm { S U } ( 2 ) \cong S ^ { 3 }$ by four global embedding coordinates subject to one unit-norm constraint. Thus the configuration manifold has intrinsic dimension 3N and is supplied to the network through 4N constrained coordinates. This representation results in a smooth, site-local diferential structure, on which the Hamiltonian acts through Lie derivatives. These, in turn, enable the use of automatic diferentiation primitives, scalable to 8000+ qubits, well beyond the early proposals of [40]. Second, our model genuinely respects the variational principle in the Hilbert space of spin-1/2 systems, something which was acknowledged to be an open issue of phase-space-based quantum machine learning [40].

Our model’s wavefunction hence constitutes a centrally odd scalar function on SU(2)<sup>N</sup> , a function class properly containing all possible neural quantum states, as we prove in Theorem S1.2. To do so, when optimising, we constrain the model to the physical sector identified by the Peter-Weyl theorem. At 547,521,152 variational parameters, Hamilton-Zero is, to our knowledge, the largest wavefunction ansatz yet trained for spin systems. At this scale, pretraining practices developed for large language models and reinforcement learning become directly relevant. On the fixed pretraining workload defined in SM § S4, an of-the-shelf JAX ecosystem [41] with folx [42] required more than one year of wall time. The forked versions we release with this work reduce that estimate to approximately four days.

The rest of this work is structured as follows. In Sec. 2 we describe briefly the key mathematical steps that underpin our automatic diferentiation primitives and our choice of manifold function. In Sec. 3 we briefly describe Hamilton-Zero’s structure and similarities to Large Language Models, including a brief overview of our deep reinforcement learning pipeline for inferring entanglement structure. Next in Sec. 4 we describe the key aspects of pretraining and data augmentation. Sec. 5 then presents the results, in which we show (i) a single shared optimisation trajectory improves the energy of a heterogeneous distribution of hundreds of thousands of Hamiltonians, (ii) that it can solve for energies of unseen systems when evaluated on frozen weights, and fine-tune to within 1% of ED energies on such systems with relatively little compute, (iii) that Hamilton-Zero can be taken far outside the system-size distribution on which it was trained on three case-studies, (iv) that it can witness a phase transition from a single checkpoint of its weights, (v) that it becomes approximately equivariant with respect to gauge groups in spinor-physics and (vi) that the entanglement structure of both pretraining and larger out-of-distribution systems are efectively learned by the router. Of note here is our scaling laws, which empirically suggest our model has 5× better scaling than today’s LLMs.

The supplementary material develops each component of Hamilton-Zero in full. In SM § S1, we derive our variational principle for the function class $S U ( 2 ) ^ { N } \to \mathbb { C } .$ , describing the necessary theory to facilitate Hamiltonian (and other observable) action via automatic diferentiation. Here we also resolve an open problem around harmonic leakage for phase space quantum machine learning that enables future generations of the model to gain even more expressiveness. In SM § S2 we describe the foundation ansatz, which contains a featuriser for the Hamiltonian dataset’s difering interaction topologies, weights and system sizes, an LLM style trunk of attention and FFN layers, and a readout mechanism trained with deep reinforcement learning to enable generalisation across unseen system topologies and system sizes. In SM § S3, we lay out the training routines we employed for pretraining in full, as well as detail for the training, validation and evaluation datasets we created for pre-training, fine-tuning and zero-shot evaluation. SM § S4 details our research into optimisation and engineering that made this model’s training feasible. Of note here is our novel sampling scheme for MCMC on Riemannian manifolds which facilitates high-quality, decorrelated samples from our model’s quaternionic density at unprecedented speeds. Here we also detail an extension of the KFAC optimiser of Martens and Grosse [43], which allows it to work with higher-order Fisher-information tensors with a provably faster convergence rate than its rank-two counterpart.

Everything above can be found in the open-source github repository for this paper, along with usage instructions for installing and using the model. It is also available on hugging-face.

## 2 Theory

We consider the ground-state problem for the universal [44] spin-1/2 family of Hamiltonians,

$$
\hat { H } = \frac { 1 } { 4 } \sum _ { i < j } \sum _ { a , b } J _ { i j } ^ { a b } \hat { \sigma } _ { i } ^ { a } \hat { \sigma } _ { j } ^ { b } + \frac { 1 } { 2 } \sum _ { i } \sum _ { a } h _ { i } ^ { a } \hat { \sigma } _ { i } ^ { a } ,\tag{2.1}
$$

where $J _ { i j } ^ { a b }$ encodes arbitrary quadratic Pauli interactions and $h _ { i } ^ { a }$ encodes local fields.

It is useful to think of J as the data of a weighted graph on N sites, where each unordered pair $( i , j )$ with $i \neq j$ carries a $3 \times 3$ Pauli-pair coupling matrix $J _ { i j } \in \mathbb { R } ^ { 3 \times 3 }$ , which we read as nine flavours of edge, one per Pauli-pair $( a , b ) \in \{ x , y , z \} ^ { 2 }$ giving the operator $\sigma _ { i } ^ { a } \sigma _ { j } ^ { b }$ , see Fig. 2.

For example, the diagonal entries a = b (XX, YY, ZZ) are the Heisenberg exchanges and the of-diagonal entries (XY, XZ, YX, YZ, ZX, ZY) encode Kitaev-Γ [45, 46], Dzyaloshinskii–Moriya [47], compass, and other cross-channel couplings. We will informally refer to the $N \times N$ edges as bonds. Meanwhile the diagonal (i, i) self-bonds do not correspond to any quadratic interaction in this graph.

Indeed, standard Neural Quantum States (NQS) [16, 24, 37] learn amplitudes for these interaction graphs on the discrete spin basis $\{ \uparrow , \downarrow \} ^ { N }$ . This makes each model tied to one system size and one Hamil tonian instance, so transfer across Hamiltonian structure is dificult [25–27]. Only recently researchers have tried to start generalising within a given Hamiltonian interaction topology [23–27]. Hamilton-Zero instead represents the state as a smooth function on the product Lie group $\dot { \psi } _ { \boldsymbol { \theta } } : \dot { \mathrm { S U } } ( 2 ) ^ { N }  \mathbb { C }$ Each spin is parametrised by a unit quaternion $q _ { i } \in S U ( 2 ) \cong S ^ { 3 } \hookrightarrow \mathbb { R } ^ { 4 }$ , so the network consumes

![](images/e86d870da528709a650fabbf57388d65b34be97578761f8bcd433146344d19b3.jpg)

![](images/642d27f11740a253a77d4753c51b49dbd2650b0787a5634a3398eb6e32c76ceb.jpg)  
(b) Cross-channel coupling (Kitaev-Γ) on a 1D chain. Each bond carries a distinct of-diagonal Pauli pair, 1–2 is YZ, 2–3 is ZX, 3–4 is XY. Source and destination Pauli difer, so each edge changes colour at its midpoint.

(a) (a) Heisenberg XXX model on a 1D spin-chain. Each nearest neighbour pair of sites carries XX, YY and ZZ channels.  
![](images/2647d15e9dac82118ec7443866dfdd46f575adb85f8d283a0428c6cdb6c1a739.jpg)

![](images/29c7b49efa9c910cd35d97ddff423ab15a2aaeddca7bb79e4e43c4195912ebf2.jpg)  
(c) Compass model on a $\mathrm { 2 D 2 \times 3 }$ grid, where horizontal bonds carry XX; vertical bonds carry YY. Bond orientation determines the Pauli channel.  
(d) A J1J1 Heisenberg chain with nearestneighbour $( J _ { 1 } ,$ straight) and next-nearestneighbour $( J _ { 2 } , \mathrm { a r c s } )$ couplings, both ZZ.  
Figure 2: Four interaction-graph examples drawn from the training data. The colour map $\mathrm { X } , \Pi \mathrm { Y }$ Z shows diferent types of edges.

4N continuous coordinates rather than the discrete basis of NQS [16]. Thus spin operators can act on this manifold through left-invariant vector fields $L _ { i } ^ { a } , \hat { \sigma } _ { i } ^ { a } \longleftrightarrow - i L _ { i } ^ { a }$ for first order and for second, $\hat { \sigma } _ { i } ^ { a } \hat { \sigma } _ { j } ^ { b } \longleftrightarrow - L _ { i } ^ { a } L _ { j } ^ { b }$ . The Hamiltonian becomes a second-order diferential operator on $\mathrm { S U } ( 2 ) ^ { N }$ , which we can compute with automatic diferentiation of the model with respect to its input coordinates,

$$
\hat { H } \longleftrightarrow - \frac { 1 } { 4 } \sum _ { i < j , a , b } J _ { i j } ^ { a b } L _ { i } ^ { a } L _ { j } ^ { b } - \frac { i } { 2 } \sum _ { i , a } h _ { i } ^ { a } L _ { i } ^ { a } ,\tag{2.2}
$$

extending recent works on moment generating functions [40] to the manifold function directly via custom JAX-primitives of these Lie derivatives. Hence, the Hamiltonian’s interaction data is simply a bare coupling tensor $( J , h ) \in \mathbb { R } ^ { N \times N \times 9 } \times \mathbb { R } ^ { 3 N }$ , while its action on states is evaluated by autograd of the manifold wavefunction’s inputs. The quaternionic frame realising each $L _ { i } ^ { a }$ and the log-derivative identities behind the local energy are Supplimentary Material (SM) §S1; the kernels evaluating these operators at scale are SM §S4.2.

By moving from amplitudes to functions on $\mathrm { S U } ( 2 ) ^ { N }$ , we have a space $L ^ { 2 } ( \mathrm { S U } ( 2 ) ^ { N } )$ which is larger than the physical spin- $_ { - } / 2$ Hilbert space. However, the Peter–Weyl theorem [48, 49] gives us a spectrum for this function space,

$$
L ^ { 2 } ( \mathrm { S U ( 2 ) } ) = \bigoplus _ { j = 0 , \frac { 1 } { 2 } , 1 , \dots } V _ { j } ^ { * } \otimes V _ { j } ^ { }\tag{2.3}
$$

decomposing it site-wise into irreducible spin sectors, $V _ { j }$ . The eigenfunctions in the physical subspace satisfy $\begin{array} { r } { - \Delta _ { i } \psi = \frac { 3 } { 4 } \psi } \end{array}$ for every site i, where $\Delta _ { i }$ is the Laplace–Beltrami operator on each local set of $S U ( 2 )$ coordinates and $- \Delta _ { i }$ realises the per-site Casimir $\hat { S } _ { i } ^ { 2 } \ [ 5 0 , 5 1 ]$ . All physical spin-1/2 functions on this manifold satisfy this Casimir relation, however an unconstrained neural function on $\mathrm { S U } ( 2 ) ^ { N }$ may leak into higher $j$ sectors giving unphysical energies below the ground state.

Our ansatz avoids the leak by construction, as we impose per-site oddness $\psi ( . ~ . ~ . , - q _ { i } , . ~ . ~ . ) ~ =$ $- \psi ( \cdot \cdot \cdot , q _ { i } , \cdot \cdot \cdot )$ , and we keep the wavefunction linear in each site’s quaternion. This is because oddness removes all integer-j sectors, and linear functions of $q _ { i }$ are exactly the span of the four spin-1/2 Wigner D-matrix elements $D _ { m n } ^ { 1 / 2 } ( q _ { i } )$ , so a centrally odd wavefunction that is linear in every quaternion lies analytically in the spin-1/2 sector,

$$
\psi _ { \boldsymbol \theta } ( q _ { 1 } , \dots , q _ { N } ) = \sum _ { m , n } T _ { m , n } ( \boldsymbol \theta ) \prod _ { k = 1 } ^ { N } D _ { m _ { k } n _ { k } } ^ { 1 / 2 } ( q _ { k } ) ,\tag{2.4}
$$

On this sector, the Lie-derivative operator is unitarily equivalent to the physical Hamiltonian, so we may therefore define a variational principle,

$$
\frac { \langle \psi _ { \theta } | \hat { H } | \psi _ { \theta } \rangle } { \langle \psi _ { \theta } | \psi _ { \theta } \rangle } \geq E _ { 0 } ,\tag{2.5}
$$

and every energy our optimiser reports is a rigorous upper bound on the physical ground-state energy. See SM § S1 for a full exposition and derivation of this variational principle. We emphasise that despite this linearity, nothing prevents a non-linear conditioning on the Hamiltonian’s interaction data, which is precisely where Hamilton-Zero’s expressivity lies. We defer details of its architecture to Sec. 3 and SM § S2.

Had the ansatz been nonlinear in the quaternions however, oddness alone would leave the higher half-integer sectors open, meaning leakage could occur in principle. However, we can resolve this open problem of leakage to higher-spin sectors recorded in [40] with a regulariser that pushes the model towards the physical sector, and an energy penalty that analytically restores the variational principle. For a full discussion on these cases, we refer to Sup. Mat. S1 and Appendix A for the proofs. For this generation of model however, the regular variational principle is suficient when we have centrally odd functions that are multi-linear in the manifolds quaternionic coordinates.

We finish this section with a remark on the expressivity of our model’s ambient function class in comparison to NQS. In brief, NQS span a variational sub-manifold of their full Hilbert space unless they have control over $\mathcal { O } ( 2 ^ { N } )$ independent amplitudes through variational amplitudes, which is dimensionally cursed (and the reason why practical NQS architectures span a smaller variationa sub-manifold). Hamilton-Zero does not inherit this restriction. Since the model is centrally odd and multilinear in the quaternion coordinates, it has the capacity to act in

$$
{ \mathcal F } _ { \mathrm { l i n } } \cong \left( V _ { 1 / 2 } ^ { * } \otimes V _ { 1 / 2 } \right) ^ { \otimes N } , \qquad \mathrm { d i m } { \mathcal F } _ { \mathrm { l i n } } = 4 ^ { N } ,\tag{2.6}
$$

which strictly contains the canonical lift of the entire NQS Hilbert space, and therefore also contains every practical NQS variational manifold. The Hamiltonian acts on the one Peter–Weyl index and leaves the additional index untouched, so this larger function class still respects the physical variational principle. If we allow the model to depend non-linearly on each quaternion (but remain centrally odd), its capacity grows again to include all higher half-integer sectors. This however requires an energy penalty to remain a variational principle on the original Hilbert space, as well as regularisation. We discuss these aspects further in SM § S1 and prove an expressivity hierarchy in Theorem S1.2. The theorem compares ambient function classes; the expressivity of a finite parameterisation remains architecture-dependent.

## 3 Hamilton-Zero’s Architecture

In this section, we give a brief account of the architecture of our model, deferring its technicalities to SM § S2.

The architecture has two types of inputs. First is a sampled spin configuration $q \in \mathrm { S U } ( 2 ) ^ { N }$ , and second is a Hamiltonian’s interaction tensor (J, h). The two are routed through separate pathways and meet at a single, controlled point, so that the central oddness and per-site linearity of Sec. 2 are preserved exactly by construction (see Fig. 3). We give a brief account of each component here, deferring technicalities to SM § S2.

The Hamiltonian data enter the model through a fixed cache followed by a learnable featuriser. The cache bond-symmetrises J and forms a temporary Hermitian matrix from J and h solely to determine one shared spectral scale. Once that scale has been computed, the diagonal field block is discarded and the normalised bond tensor and local field are supplied to the model as separate channels. This places every system in a common numerical regime without needing the model to relearn an overall energy scale.

Next, a featuriser then reads the spectrally normalised Hamiltonian data and produces embeddings for it for the downstream blocks of our model. The featuriser normalises in learned groups by root-mean-square (RMS) maps, and aggregates by column and row attention to produce a per-bond embedding, a per-site embedding, and a global context vector. Bonds and fields absent from a system substitute learned absent tokens rather than zeros. This is so that the absence of a connection, which changes the system’s topology, is a representable direction of the feature basis. Meanwhile padding, which carries no structural meaning to the system’s interaction topology, is done with masks.

Once the Hamiltonian’s data is embedded by the featuriser, it is passed through L pre-norm residual blocks. Each block updates the bonds first through an edge map, biases the per-site attention logits with the fresh bonds so heads can attend along the interaction graph, and closes with a feed-forward update. Every sublayer normalised by the same root-mean-square rule and is damped by $\mathrm { ~ a ~ } 1 / \sqrt { L }$ residual gain. A global data-stream runs alongside this so that downstream components can also act directly on the feature basis. This global stream lives on the sphere of unit root-mean-square norm, and is refreshed once per block by descriptor attention over the states the block has just written. It moves by the normalised-interpolation update of nGPT, so near-identical readings from symmetric systems saturate instead of accumulating with depth.

![](images/be42c41936cd1255089cbeb7ac4d86dd4486cfccd9775b615522b6a4d30a83f1.jpg)  
Figure 3: Hamilton-Zero information flow. The Hamiltonian (J, h) is spectrally normalised, featurised in polar form, and propagated through the trunk; a route policy conditioned on the trunk’s outputs permutes sites and padding slots before the leaf contextualiser; the spin configuration q enters only at the odd leaf builder enforcing per-site oddness before the shared rank-four merge tree returns log ψ<sub>θ</sub>(q | p). The rail beneath the pipeline is the global stream $^ { g , }$ refreshed at every stage on the unit-RMS sphere. Component diagrams are SM §S2.

We emphasise that the spin configuration never enters this trunk, and that everything up to the leaves is a function of $( J , h )$ alone. This means the bulk of the model’s parameters do not need any symmetry constraints that would limit its expressivity. The symmetry constraints are enforced at the single point where quaternionic coordinates enter.

For the coordinate dependence, a bias-free map first lifts each q<sub>i</sub> into a linear odd data-stream. Hamiltonian context supplies only the coeficients of the map applied to that stream. The resulting raw leaf carrier is therefore linear in $q _ { i }$ . We normalise the carrier, $u ,$ for numerical stability, record the divided-out leaf magnitude in a log-scale, s, and restore it at the root together with all later merge scales. The normalised carrier alone is odd but nonlinear, meaning only the represented pair $( u , s )$ reconstructs the linearity property needed from Sec. 2 carrier exactly. Bilinear tree merges then make the final amplitude multilinear in the site coordinates, while the Hamiltonian-only stream retains the model’s nonlinear capacity.

Finally, a readout mechanism contracts the N carriers through a balanced binary tree of merges, in the spirit of tree tensor networks [11–13], with leaves in a bit-reversed canonical frame and one merge shared across every level and position. The shared merge is a rank-four tensor contracting blockwise, bilinear in its two children so a sign flip at any leaf passes to the root and thus preserves our symmetry constraints from Sec. 2. The leaves and every internal node renormalise their carriers to unit root-mean-square while storing the divided-out magnitudes in an additive log-scale, all of which is again restored at the root readout. A Hamiltonian context stream is computed up the tree beside the carriers by the same nGPT-style normalised interpolation as the global context data-stream. This context stream reads the coarse-grained bond field between the two subtrees being merged, and a leve attention with a tree-distance bias lets subtrees exchange information laterally, deleting the depth penalty on long-range correlations. Thus our merge tree is conditioned by a coarse-graining of the even data-stream, which we think of as a neural augmentation of tree tensor networks. Note that padding slots carry learned contexts of their own, so one tree serves many system sizes.

A learned site-to-slot map adapts the contraction tree to diferent interaction topologies and system sizes. To learn entanglement structures, we make a map from physical sites to the leaf slots of the merge contraction itself learnable. This map is a discrete latent variable between physical site indices and leaf slots, which we learn with deep reinforcement learning [52, 53]. A pointer-network policy conditioned on the trunk’s outputs decodes a permutation slot by slot. At every row it composes every remaining candidate afresh from its site state, the fixed Hamiltonian global, the decode clock, bidirectional prefix and sufix bond summaries, an order-aware prefix, and hole-placement statistics. The selected row states become the leaves of a partial dyadic merge tree. Causal same-level refinement builds its complete subtrees, their mixed-scale dyadic cover produces the next query, and the remaining candidates attend the cover, earlier queries, and one another before the pointer chooses. A conditional quotient then removes indistinguishable choices before sampling. The sampled route relabels sites, bonds, and walkers before the merge mechanism and its context construction runs, so that the corresponding context around each site and its bonds is carried through the learned reordering.

From a physical perspective, the model represents the incoherent mixture of states over sampled routes, a relaxation that costs nothing because the energy is linear in the density matrix and is minimised at a pure state. The policy trains by a score-function gradient with a beam approximation to the policy mode and a self-normalised importance-sampling baseline evaluated on the routed walkers. The walker channel is centred at its route-conditional mean. Here π is the routing policy, m is the slot mask, and $\operatorname { A u t } ( J , h , m )$ is the group of slot permutations preserving the bond tensor $^ { J , }$ site fields $h ,$ and mask m. For a route $p ,$ its orbit and stabiliser are

$$
{ \mathcal { O } } ( p ) : = \{ g \cdot p : g \in { \mathrm { A u t } } ( J , h , m ) \} , \qquad { \mathrm { S t a b } } ( p ) : = \{ g \in { \mathrm { A u t } } ( J , h , m ) : g \cdot p = p \} .\tag{3.1}
$$

For every $g \in \mathrm { A u t } ( J , h , m )$ , a route p and $g \cdot p$ have equal energy. A symmetrised policy is uniform within each occupied orbit. Writing $\begin{array} { r } { H [ \pi ] : = - \sum _ { p } \pi ( p ) \log \pi ( p ) } \end{array}$ for the policy’s Shannon entropy, a policy supported on one orbit $\mathcal { O } ( p )$ has

$$
H [ \pi ] = \log \left| \mathcal O ( p ) \right| = \log \frac { | \operatorname { A u t } ( J , h , m ) | } { | \operatorname { S t a b } ( p ) | } .\tag{3.2}
$$

Thus the measured entropy of a learned policy and its analytic correspondence to automorphisms of the interaction graph provide us with a heuristic tool to measure whether the router’s training has converge to the true optimum, see SM § §S2.5.4.

Indeed in SM §S2 we give a full exposition of this component along with all the others, including a proof of optimality when the router learns automorphisms. We now proceed to Hamilton-Zero’s training data and pretraining.

## 4 Pretraining

To train Hamilton-Zero, we collated a curated dataset of 5000 diferent interaction topologies, spanning a century of quantum many-body and quantum computing literature [54–99]. Because the corpus also includes Hamiltonians native to superconducting, trapped-ion, Rydberg, spin-qubit, and molecular platforms, Hamilton-Zero can be evaluated as a classical foundation surrogate for hardware-relevant ground-state problems. The dataset is stratified into physics cells and coverage tiers, from canonical anchors through frustrated, disordered, gauge and volume-law regimes, each system built deterministically, see SM §S3 for a full account of the contents of the training dataset. By the standards of large language models, however, such a dataset is small, and trained on it as-is, the model would memorise couplings, fields, and frames alike.

As such, here we briefly discuss the important aspects of the augmentations of our training datasets that prevent overfitting to specific edge-values and field-values, or overfitting to the topologies seen in pre-training. To that end, our training routine involves a series of perturbations of the training data. The first tier perturbs the physics itself, new bonds, new fields, new reference energies to compute; the second applies exact symmetry transformations, which generate unlimited new coupling tensors whose physics, and whose references, carry over unchanged. Together the two tiers expand the dataset into the hundreds of thousands of distinct training Hamiltonians of our corpus, which is again a relatively small dataset by today’s standard for large-language models.

During training, every new epoch receives a perturbed version of the pretraining data, which is made in parallel to the training loop’s live progress. This perturbed data is then hot-swapped at every epoch end. To keep the training loop running continuously, this perturbation routine stays 2 rounds ahead of the loop, precomputing exact-diagonalisation references for every perturbed system to $N \leq 2 2$

Each structured system receives one of five changes at equal probability. The changes can be one of the following: (i) leave the system untouched, (ii) add one new bond, (iii) add noise to one existing bond, (iv) remove one bond, or (v) adds one random orbit-symmetric field, with sites, bonds, and orbits sampled stratified over the Weisfeiler–Leman orbit partition of $( J , h )$ . Note that for (v) the magnitudes are set relative to the system’s own coupling scale. The orbit-symmetric field draws one random three-vector of strength $\eta \sim \mathcal { U } ( 0 . 2 5 , 1 )$ times the coupling scale and adds it to every site of one orbit. For small random systems $N \leq 8$ , we additionally densify towards all-to-all coupling, with a field-free such system gaining a substantial random dense field half the time, and occasional bond drops.

On top of each round’s perturbation the worker applies an exact symmetry change. It draws one Haar-random u $\in \mathrm { S U } ( 2 )$ per Weisfeiler–Leman orbit of the interaction graph, shared by the sites of that orbit. Write u for the draw assigned to the orbit containing site i. Let $R _ { \ker } ( u _ { i } )$ denote the adjoint rotation expressed in the kernel’s ordered $( x , y , z ) = ( \hat { k } , \hat { \jmath } , \hat { \imath } )$ frame. The exact gauge augmentation is

$$
J _ { i j } \mapsto R _ { \mathrm { k e r } } ( u _ { i } ) J _ { i j } R _ { \mathrm { k e r } } ( u _ { j } ) ^ { \top } , \qquad h _ { i } \mapsto R _ { \mathrm { k e r } } ( u _ { i } ) h _ { i } , \qquad q _ { i } \mapsto q _ { i } \bar { u } _ { i } ,\tag{4.1}
$$

where $\bar { u } _ { i } = u _ { i } ^ { - 1 }$ . Under this right multiplication the left-invariant fields transform by the adjoint action, so the simultaneous rewrite of $( J , h , q )$ leaves the local energy and spectrum unchanged. Sharing u within a class preserves the interaction graph’s symmetry structure and leaves the routing algorithm unchanged.

Physically, this means the model cannot overfit to a choice of frame, and the physics of the interaction graph is identical under this transformation. Hence, training across gauged rounds penalises whatever gauge violation the model carries. What this encourages is approximate ${ \mathrm { S U } } ( 2 )$ covariance resolved per orbit. Single-site fields transform by one rotation per orbit; a bond between two diferent orbits sees independent left and right factors, hence approximate $\mathrm { S U } ( 2 ) \ifmmode \times \else \texttimes \fi { } \mathrm { S U } ( 2 )$ covariance across inter-orbit bonds, and in the random small-N band the rotations are fully per-site, so every bond there trains under independent factors. The walkers transform on SU(2) itself, the double cover of the SO(3) rotations acting on the couplings, and the prediction delta across a gauge transformation is a direct probe of learned covariance (see SM §S3.2).

## 5 Results

In this section, we show that one shared optimisation trajectory improves the energy of a heterogeneous distribution of Hamiltonians in a quantitatively regular way in 5.1. We opt for the V-score [100] as our measure of quality here, as it provides a fair comparison across system types and sizes. In 5.2, we show the results for zero-shot inference and fine-tuning. Here, zero-shot refers to taking Hamilton-Zero’s frozen model weights and evaluating them on the three datasets above, and hence zero training steps were taken for those shots. We applied both to three diferent datasets detailed in SM § S3.3. In brief, these datasets cover (A) combinatorial optimisation, (B) unseen interaction and channel topologies, and (C) systems we observed to be hardest when initially developing Hamilton-Zero that we subsequently held out for the model weights released in this work.

We next show how the same pretrained wavefunction can be taken far outside the system-size distribution on which it was trained in 5.3. We study three rather diferent problems: weighted MaxCut instances [101, 102], the Pariser–Parr–Pople (PPP) model of a conjugated carbon chain [103– 105], and frustrated Heisenberg antiferromagnets on square and triangular lattices [106]. Together, these cases test non-local weighted graphs, long-range fermionic interactions, and geometrically local frustration. This allows us to see the maximum system sizes Hamilton-Zero can compile and sample from, and whether the resulting zero-shot and fine-tuned energies remain close to reference values when available at such scales.

Next in 5.4 we show that Hamilton-Zero can witness a phase transitions via the fidelity susceptibility method [107]. We will apply this to the Ising model and the J1J2 Heisenberg model. We then show that Hamilton-Zero recovers approximate equivariance with respect to the gauge transformations in the perturbation scheme of its pretraining data in 5.5.

Finally, in 5.6 we show how the router learned the entanglement structure of our pretraining dataset, how its commitment to a certain entangling contraction path is indicative of the quality of the energy bounds, and that it generalises to unseen interaction topologies and interaction types on the scale of the large-system case studies. Here we will also show the overall model has learned approximate equivariance with respect to gauge groups actions on the interaction graph (J, h).

## 5.1 Pretrain and scaling laws

After the initial transient, Fig. 4 shows the V-score falls approximately as $C ^ { - 0 . 5 3 }$ , while the ED-relative gap falls as $C ^ { - 0 . 6 1 }$ . Their similar compute dependence shows that local-energy fluctuations remain informative about energy accuracy across the training distribution, including where ED is unavailable. Moreover, the convergence of $V ^ { 1 / 2 }$ across the system-size bands shows that its normalisation removes most of the extensive dependence on N. These exponents describe this pretraining run and its evolving Hamiltonian mixture. We note that C is measured in cumulative eight-H200 hours rather than exact FLOPs, so this result is hardware dependent. By the standards of neural scaling laws in other domains like LLMs, this exponent shows highly favourable scaling with compute, sitting around $5 \times$ better than the values reported for LLMs [108].

![](images/e5ae4e557c207a3148222b389a8a679d35a0ba392c5e40273de21472aed1b0cb.jpg)

![](images/6400592534a7ac448137a5641dc0796ef793adec5050803ac6d957a631091909.jpg)

![](images/edca983a1fdc8102299d420fa59cc59035211245e896b0fe02fc6fb5ae6d0754.jpg)

![](images/b24197946b0c2b92f79f11713c330ec13952e10baf65616ac8dbda55f042aa60.jpg)  
Figure 4: Compute scaling of variance and variational energy error through step 280 000. (a) V-score decreases approximately as $V \propto C ^ { - 0 . 5 3 }$ after the initial transient. (b) The observed relative gap for ED-covered systems decreases approximately as $C ^ { - 0 . 6 1 }$ . (c) $V ^ { 1 / 2 }$ approaches a common late-time scale across the six system-size bands, showing that the normalization removes most of the extensive size dependence. (d) $V ^ { 1 / 2 }$ and the predicted relative gap decrease together over the full 5,000-system dataset, connecting variance reduction to improved inferred accuracy. Compute is cumulative execution time across eight NVIDIA H200 GPUs. The V-score is $V = N \bar { \mathrm { V a r } } ( E _ { L } ) \bar { / } ( E - E _ { \infty } ) ^ { 2 }$ , with $E _ { \infty } = 0$ here, and is therefore a normalized variational-accuracy metric rather than raw variance. The relative gap denotes $r = ( E - E _ { \mathrm { E D } } ) / | E _ { \mathrm { E D } } |$ . Solid curves show medians and shaded regions show interquartile ranges.

Next we turn our attention to the zero-variance principle [106]. As shown in Fig. 5, the bounded zero-variance model follows the calibration data without a systematic residual drift with $N ,$ and a calibration restricted to $N \leq 1 8$ predicts the unseen $N = 1 9 { - } 2 2$ systems with relative ease. This is supporting evidence for transfer across size within the ED-accessible regime.

![](images/ad3a868520944f1a2af78b7868366c4d38371770261713c28270f5c69adc0975.jpg)

![](images/3171ed701c2da6d625b987e41260068289a5bc8e97d613db81f5af224499817f.jpg)

![](images/655272f195a3582f03fc466379728efc8d09131ba86ee1a200841dfe62275675.jpg)  
Figure 5: Zero-variance calibration and held-out size validation. (a) Relative gap against V-score, both defined as in Fig. 4. Here we show the median and IQR as a blue line and shaded blue region, with a bounded zero-variance principle in dashed orange, obtained from $q = ( E - E _ { \mathrm { E D } } ) / | E | = \kappa V$ with $\kappa = 0 . 2 8 2 5$ and $r = q / ( 1 + q )$ ; a free robust fit gives $\bar { r } \propto V ^ { 0 . 8 9 }$ . (b) The median calibration residual shows no monotonic drift with N, supporting transfer across system size. (c) Fit of gap prediction showing sizes $N \ge 1 8$ follow the same relationship as $1 9 \leq N \leq 2 2$ . Blue points belong to the fitted $N \leq 1 8$ population and orange points to the unseen $N = 1 9 { - } 2 2$ population. Their overlap around the one-to-one line demonstrates held-out size generalization within the range where ED is still applicable. See also Fig. 6 for further size extrapolation.

Subject to that qualification within the ED regime, Fig. 6 shows continuity at the $N = 2 2$ ED boundary (beyond which we no longer computed ED reference values). For $N > 2 2$ , the predicted median relative gap is 3.45%, with an interquartile range of $1 . 8 6 \% \mathrm { - } 1 0 . 2 1 \%$ . The median therefore remains at the few-percent scale, while the widening distribution shows that larger systems are increasingly heterogeneous rather than uniformly harder. Finally, the exchange discrepancy tracks the total-energy error, whereas the field discrepancy is smaller. Since the component energies are not separately variational and may cancel, this identifies where the remaining energy discrepancy lies.

![](images/c6f6e3984df94a47c7f32858662fe8dc6976534a0b04c8faa3cdf9276d923aa6.jpg)

![](images/3cc41637ff026497c66d837e312d857b441b943c78154a1fed7eb060c57a9f96.jpg)

![](images/57ae4c93faded504499fcb2830f5e9a55f9875e8aa2004893a6086c3b5a33fd0.jpg)  
Figure 6: Energy gap extrapolation across full (a) Orange points show observed median relative gaps where ED is available; the blue curve and band show the median and interquartile range of the zerovariance predictions of Fig. 5. The dashed line marks the $N = 2 2$ ED boundary. The extrapolation remains continuous across this boundary indicating the pretraining run’s relative gap is consistently non-monotonic with $N .$ , rather than some cuttof past the ED regime in accuracy. (b) Total, exchange, and field discrepancies are normalized by $| E _ { \mathrm { E D } } |$ of the total Hamiltonian. Only total energy obeys the variational bound; component discrepancies are displayed in absolute value. Exchange tracks the total error, while the field contribution is smaller. (c) Predicted relative-gap distributions for $N > 2 2$ show modest median variation but broader system-to-system tails with size. Across all $N > 2 2$ systems, the median prediction is 3.45%, with interquartile range $1 . 8 6 \% \mathrm { - } 1 0 . 2 1 \%$

## 5.2 Zero-shot Inference and Fine-tuning

In order to evaluate the model, we let the pretrained router select the tree ordering, and compile the wavefunction on that ordering, and sample the fixed model for 512 measurement steps with 256 walkers across eight tempered replicas and 24 adaptive Langevin moves per step (see SM § S4 for full details on our sampling scheme), retaining $5 1 2 \times 2 5 6 \times 8 \simeq 1 . 0 5 \times 1 0 ^ { 6 }$ samples per system. Energies are reported as $E = E _ { \mathrm { k e r n e l } } / 2$ with the pooled standard deviation of per-walker means as the uncertainty.

Zero shot, the pretrained model’s energies against ED values (where available) are reported in the top row of Fig. 7, with the cumulative frequency of systems below a given percentage threshold on the bottom row. We see from the top row an overall agreement of the systems energies against ED values, with median signed gaps of 4.02%, 4.21% and 12.63% on set (A) , (B) and (C). There are 28%, 21% and 3% of systems within 1% of ED for these three sets, and the degradation across them is structured. Indeed, the gap’s rank correlation with system size grows from 0.14 (A) through 0.25 (B) to 0.47 (C). Within each evaluation set, the hardest systems are on a distinct family: the maximum-independentset instances on (a) (gaps up to 31%), the t-fractals on B (35–39%), and the Schwinger chains at nonzero θ on C (40–50%), consistent with Axis C sitting farthest from the pretraining distribution (see also SM § S3).

(a)  
![](images/29d9e7d5262a62eb0e05960f82d38bb8a191a2caab3d1a8b87fbdb2862b80f17.jpg)

(b)  
![](images/b3c4cd35b93af4d79e4eafa3c4b65e66bff855e1c6cf15b0ee66f6203abde7cb.jpg)

(c)  
![](images/83b09b638a1e9df6cefbd6a1c9ed53fce2569882ed4ee0975f1b84c9ae0177df.jpg)

![](images/6e47e8b92848d5d782dd812f4c14d24dca77870c0729799f73c43f61ee3b1628.jpg)

![](images/2fe0a757a6495ac6581902c7c6bf506a819e1b86007215cb07e80b1cb5e37cfb.jpg)

![](images/9c17ef89b4d93519e009954fc6c5b69aeec63a71737296b12558d3723a9fcbd2.jpg)  
Figure 7: Held-out energy accuracy for the 512-step evaluation. Panels (a)–(c) show the magnitudes of the variational Monte Carlo and exact-diagonalization energies per site for Axes A, B, and $\mathrm { C } ,$ respectively, on logarithmic axes. Energy magnitudes are plotted because the reported energies are negative. Every point carries its corresponding walker-mean tail standard-deviation bar; where no bar extends beyond its marker, the uncertainty is smaller than the marker diameter. Panels (d)–(f) show the empirical cumulative percentage of systems whose relative energy gap is at most the threshold for Axes A, B, and C, respectively, with a logarithmic gap-threshold axis. Colors identify the same axis in both rows. All energies are additionally divided by N for per-site values. The empirical-CDF denominators are the systems with exact-diagonalization references: 89 for Axis A, 66 for Axis B, and 79 for Axis C.

Next, we turn our attention to fine-tuning. For fine-tuning, we restart from the same checkpoint as evaluation, let the pretrained router pick the tree ordering, and compile the wavefunction on that ordering. We then only train the merge-tree on that fixed ordering, leaving a compiled ansatz with ∼4.7M parameters, against 547M for the full model as the object being trained. Our rationale for this is that the trunk, leaf contextualiser and other upstream parts of the Hamiltonian context need not be retrained on a single system since they are amortizing over many diferent Hamiltonian families, so it is counter-productive to train them to fit the key features of a single system. This is also justified by the fact that once compiled, we can then fine-tune on a single A100 GPU at a considerably faster stepweight owing to the fact that the number of variational parameters is $< 1 \%$ of the overall model. Each system is optimised independently on a single A100 for $1 0 ^ { 4 }$ KFAC steps with 256 walkers, four MCMC moves per step and a 128-step burn-in, with learning rate $0 . 0 1 / ( 1 { + } t / 1 0 ^ { 4 } )$ (ending $\mathrm { a t } \approx 0 . 0 0 5 )$ , damping $1 0 ^ { - 3 }$ , and curvature EMA 0.9. After training, every system is re-evaluated with its new tree weights. The final checkpoint is resumed at an exact zero learning rate for an additional 512 measurement steps with the sampler state carried over. ED references are never supplied to evaluation or fine-tuning and enter only in the analysis below.

Fig. 8 shows the results for fine-tuning against ED values (where available). We see the remaining errors from zero-shot decrease by two to three orders of magnitude on every set of the $( \mathrm { A } ) \mathrm { - } ( \mathrm { C } )$ evaluation datasets. The median signed gap falls to $3 . 1 \times 1 0 ^ { - 4 } \% , 1 . 2 \times 1 0 ^ { - 3 } \%$ and $6 . 0 \times 1 0 ^ { - 3 0 } \%$ respectively for $( \mathrm { A } ) \mathrm { - } ( \mathrm { C } )$ , with 90%, 94% and 100% of systems landing within 1% of ED, and 75%, 49% and 35% already at the $1 0 ^ { - 3 0 \% }$ error. Because every fine-tuned value comes from the frozen-weight re-evaluation rather than from the training tail, and the two agree within combined uncertainties on all 256 systems $( | z | \leq 2 . 7 )$ , the comparison against zero shot is like for like, and we see the failure modes do not persist through fine-tuning. Hamilton-Zero was thus able to find these ground states too.

The cost of this accuracy is modest, with convergence heavily front-loaded in the number of training steps as shown in Fig. 9. The mean gap drops below 1% within a few hundred steps on B and $\mathrm { C } ,$ and Axis A plateaus at 3.1% by roughly 2000 steps. In terms of compute cost, we took 234 GPU h to fine-tune all 256 systems of sets $( \mathrm { A } ) \mathrm { - } ( \mathrm { C } )$ , plus 24 GPU h to re-evaluate them, against 47 GPU h for the zero-shot evaluation alone, roughly 5.5× the zero-shot cost for the two to three orders of magnitude in accuracy. Wall time is set by the padded tree width rather than by physical N, so cost is determined by the power of two a system falls into, with the largest instances $( N = 4 0 – 5 4 )$ taking about 3.5 h each.

## 5.3 Large System Case Studies

We begin by recalling some important properties of the three types of systems we study here: MaxCut, a PPP carbon chain, and frustrated 2D magnets. We then show the results for zero-shot and fine-tuning for each system.

For a weighted graph $G = ( V , \mathcal { E } , w )$ , MaxCut asks for a bipartition of the vertices which maximises the total weight of edges crossing the partition. If $z _ { i } \in \{ - 1 , + 1 \}$ labels the side containing vertex i, its objective is

$$
C ( z ) = \frac { 1 } { 2 } \sum _ { ( i , j ) \in \mathcal E } w _ { i j } ( 1 - z _ { i } z _ { j } ) .\tag{5.1}
$$

We drop the additive constant and give Hamilton-Zero the diagonal Ising Hamiltonian

$$
H = \frac { 1 } { 2 } \sum _ { ( i , j ) \in \mathcal { E } } w _ { i j } Z _ { i } Z _ { j } , \qquad \langle C \rangle = \frac { 1 } { 2 } \sum _ { ( i , j ) \in \mathcal { E } } w _ { i j } - \langle H \rangle .\tag{5.2}
$$

The ground state is consequently a computational-basis state encoding an optimal cut. The variational energy gives an expected cut weight, whilst sampling the wavefunction also returns ordinary feasible cuts. We report both quantities below, where the first tests the full distribution represented by the wavefunction, whereas the second is the relevant output if the model is used as a combinatorial optimiser.

Next, the PPP Hamiltonian is an interacting π-electron model for conjugated carbon systems. For a chain of L carbon sites, at half filling, we use the particle–hole-centred form

$$
\begin{array} { l } { { \displaystyle { H _ { \mathrm { P P P } } = - \sum _ { \langle i , j \rangle , \sigma } t _ { i j } \left( c _ { i \sigma } ^ { \dagger } c _ { j \sigma } + c _ { j \sigma } ^ { \dagger } c _ { i \sigma } \right) + U \sum _ { i } \left( n _ { i \uparrow } - \frac { 1 } { 2 } \right) \left( n _ { i \downarrow } - \frac { 1 } { 2 } \right) } } } \\ { { \displaystyle { \qquad + \sum _ { i < j } V _ { i j } ( n _ { i } - 1 ) ( n _ { j } - 1 ) , \qquad V _ { i j } = \frac { U } { \sqrt { 1 + \left( U r _ { i j } / e ^ { 2 } \right) ^ { 2 } } } , \qquad e ^ { 2 } = 1 4 . 3 9 7 \mathrm { e V } \bar { \mathrm { A } } } . } } \end{array}\tag{5.3}
$$

![](images/6d91d0f0ddbbdd62189b0f1809fab31c6ef6a19a31b6de96b5183a184edb8b84.jpg)

![](images/430f160e312aa9fb83fa046ca1db64a6a881eb754d43525048fbc8d4063bdc8e.jpg)

![](images/e92eff97ac16f96449afea94f4edc3fc6a2744b6811b8594592216321876cbd2.jpg)

![](images/33aa0001526b4b6affaea00553adfe69b28b95cbb1bf157b47ba320c01e0db19.jpg)  
|signed ED gap| (%)

![](images/3ff7aed4add97e4e79a1cc81c5963fa09dda22764e78b5f15da3a38b7eed218b.jpg)  
|signed ED gap| (%)

![](images/ba97614b474d9798296d40c169e0bc09369f897760e2faa19d7d59e9e3ac1e79.jpg)  
Figure 8: Held-out energy accuracy for the 512-step evaluation. Panels (a)–(c) show the magnitudes of the variational Monte Carlo and exact-diagonalization energies per site for Axes A, B, and $\mathrm { C } ,$ respectively, on logarithmic axes. Energy magnitudes are plotted because the reported energies are negative. Every point carries its corresponding walker-mean tail standard-deviation bar; where no bar extends beyond its marker, the uncertainty is smaller than the marker diameter. Panels (d)–(f) show the empirical cumulative percentage of systems whose relative energy gap is at most the threshold for Axes A, B, and C, respectively, with a logarithmic gap-threshold axis. Colors identify the same axis in both rows. All energies are additionally divided by N for per-site values. The empirical-CDF denominators are the systems with exact-diagonalization references: 89 for Axis A, 66 for Axis B, and 79 for Axis C.

Here $t _ { i j }$ is the nearest-neighbour hopping, U the on-site repulsion and $V _ { i j }$ the Ohno interpolation for the long-range Coulomb interaction [109]. Its Jordan–Wigner image contains $N = 2 L$ qubits, one for each spin orbital. We report the centred energy per carbon, $E _ { \mathrm { { c e n t } } } / N _ { \mathrm { { C } } } ;$ any comparison must use this convention. In particular, the infinite-chain value quoted below is a literature scale for the corresponding PPP convention rather than a rigorous finite-chain bound, since Coulomb geometry and end corrections difer between calculations [110].

Finally, the square and triangular spin systems can be written without committing to a drawing of the lattice. Let $\mathcal { E } _ { 1 }$ and ${ \mathcal { E } } _ { 2 }$ denote the sets of nearest- and next-nearest-neighbour bonds. We study

$$
H _ { J _ { 1 } J _ { 2 } } = J _ { 1 } \sum _ { ( i , j ) \in \mathcal { E } _ { 1 } } \mathbf { S } _ { i } \cdot \mathbf { S } _ { j } + J _ { 2 } \sum _ { ( i , j ) \in \mathcal { E } _ { 2 } } \mathbf { S } _ { i } \cdot \mathbf { S } _ { j } , \qquad \mathbf { S } _ { i } = \frac { 1 } { 2 } ( X _ { i } , Y _ { i } , Z _ { i } ) .\tag{5.4}
$$

On the square lattice, $\mathcal { E } _ { 1 }$ contains the horizontal and vertical bonds and ${ \mathcal { E } } _ { 2 }$ the plaquette diagonals. The case studied here has $J _ { 2 } / J _ { 1 } = 1 / 2$ , near the maximally frustrated region of the model [111]. The triangular calculation instead sets $J _ { 2 } = 0$ and places the $J _ { 1 }$ bonds on a triangular lattice, for which every bulk spin has six neighbours and the antiferromagnetic interactions cannot all be minimised simultaneously. The four constructions are shown in Fig. 10.

![](images/dbea0427dec2ec223791fad3895007af2bd1ec56438d84e7d2fa533ddb7aa4de.jpg)  
Figure 9: Convergence of the fine-tuning runs on the ED-referenced systems of datasets A, B, and C, respectively. Lines show the per-axis median of the absolute signed relative energy gap to exact diagonalization as a function of fine-tuning step, and shaded bands span the interquartile range across systems. Gaps are computed from the per-step walker-mean energies recorded every 50 steps, so the late-time level reflects single-step Monte Carlo noise of order $1 0 ^ { - 2 0 } \%$ . The converged medians quoted in the text are lower because the frozen-weight evaluation averages 512 such steps.

![](images/028ffe7f8b33a0801305f3398da2d47ef34c85363b8a60ff8f2f124e725292a2.jpg)  
Figure 10: Large-system case studies. (a) A weighted graph for MaxCut. Blue and white vertices show a candidate bipartition; orange edges cross the cut, while the labels illustrate that the instances are weighted. (b) The PPP model on a conjugate carbon backbone, drawn in standard skeletal notation. Alternating bonds support the π system, with nearest-neighbour hopping $t _ { i , i + 1 }$ and the long-range Ohno interaction $V _ { i j } ;$ two spin orbitals per carbon map to $N = 2 L$ qubits. (c) The square $J _ { 1 } { - } J _ { 2 }$ model, with solid nearest-neighbour and dashed diagonal bonds. (d) The nearest-neighbour triangular Heisenberg antiferromagnet. The drawings illustrate the interaction graphs, and the evaluated systems are larger than those drawn.

On these systems, Tab 1 shows how Hamilton-Zero’s performance scales. We see relative agreement up to the 2000 qubit scale, with the model degrading at 4000 and deteriorating at 8k. This shows us where its current size-extrapolation limits lie in the zero-shot setting, when trained on system sizes up

Table 1: Zero-shot and fine-tuned large-system evaluation. Energies are per qubit in the model’s coupling units, except the PPP rows, which are particle–hole-centred energies in eV per carbon with the number of carbons $N / 2 .$ The square lattices use open boundary conditions and the triangular lattice periodic boundary conditions. MaxCut comparisons are $E _ { \mathrm { f e a s } } / N = ( W / 2 - C _ { \mathrm { r e f } } ) / N$ from the archived feasible cuts $C _ { \mathrm { r e f } } .$ , so they are feasible reference energies not true ground states. Reference energy values are also in the table. PPP and lattice comparisons are thermodynamic-limit literature scales $e _ { \infty }$ rather than exact finite-size energies [103–105]. $\mathrm { N } / \mathrm { A }$ marks systems with no fine-tuning run. Recovery is the fraction of the comparison value reached; marked with an asterisk <sup>∗</sup> are where no fine-tuning happened.
<table><tr><td>System</td><td>N</td><td>Zero-shot E/N</td><td>Fine-tuned E/N</td><td>Comparison</td><td>Recovery</td></tr><tr><td>MaxCut-12</td><td>256</td><td> $- 0 . 3 2 1 0 6 8 \pm 0 . 0 0 0 5 9 4$ </td><td>-1.636595 (20k)</td><td> $- 1 . 9 4 3 3 5 9 \ ( C _ { \mathrm { r e f } } = 2 4 5 6 )$ </td><td>96.8%</td></tr><tr><td>MaxCut-12</td><td>512</td><td> $+ 0 . 0 1 0 2 4 6 \pm 0 . 0 0 1 8 4 5$ </td><td>-2.250606 (20k)</td><td> $- 2 . 7 8 5 1 5 6 \ ( C _ { \mathrm { r e f } } = 9 2 7 5 )$ </td><td>97.1%</td></tr><tr><td>MaxCut-12</td><td>1024</td><td> $+ 0 . 5 3 9 9 2 6 \pm 0 . 0 0 3 3 4 7$ </td><td>-2.516722 (20k)</td><td> $- 3 . 9 4 7 7 5 4 \ ( C _ { \mathrm { r e f } } = 3 5 4 6 9 )$ </td><td>95.9%</td></tr><tr><td>PPP-Ohno</td><td>256</td><td> $- 3 . 1 8 7 4 8 3 \pm 0 . 0 0 4 1 2 7$ </td><td>-4.951437 (10k)</td><td> $- 5 . 0 1 5 ( 2 0 )$ </td><td>98.7%</td></tr><tr><td>PPP-Ohno</td><td>512</td><td> $- 3 . 0 1 3 3 3 4 \pm 0 . 0 0 2 4 6 0$ </td><td>-4.935446 (10k)</td><td> $- 5 . 0 1 5 ( 2 0 )$ </td><td>98.4%</td></tr><tr><td>PPP-Ohno</td><td>1024</td><td> $- 2 . 7 2 5 3 0 9 \pm 0 . 0 0 3 9 9 2$ </td><td>-4.657041 (10k)</td><td> $- 5 . 0 1 5 ( 2 0 )$ </td><td>92.9%</td></tr><tr><td>Square  $J _ { 1 } - J _ { 2 }$ </td><td>484</td><td> $- 0 . 4 4 6 6 6 1$ </td><td>-0.462856 (100k)</td><td> $\simeq - 0 . 4 9 6 8$ </td><td>93.2%</td></tr><tr><td>Square  $J _ { 1 } { - } J _ { 2 }$ </td><td>1024</td><td> $- 0 . 4 4 5 6 7 4$ </td><td>-0.451456 (30k)</td><td> $\simeq - 0 . 4 9 6 8$ </td><td>90.9%</td></tr><tr><td>Square  $J _ { 1 } { - } J _ { 2 }$ </td><td>2025</td><td> $- 0 . 4 2 1 7 6 3 \pm 0 . 0 0 0 0 4 6$ </td><td>N/A</td><td> $\simeq - 0 . 4 9 6 8$ </td><td>84.9%*</td></tr><tr><td>Square  $J _ { 1 } { - } J _ { 2 }$ </td><td>4096</td><td> $- 0 . 3 5 0 0 0 9 \pm 0 . 0 0 0 0 4 8$ </td><td> $\mathrm { { N / A } }$ </td><td> $\simeq - 0 . 4 9 6 8$ </td><td>70.5%*</td></tr><tr><td>Square  $J _ { 1 } { - } J _ { 2 }$ </td><td>8100</td><td> $+ 0 . 1 2 8 1 9 6 \pm 0 . 0 0 0 1 0 1$ </td><td> $\mathrm { { N / A } }$ </td><td> $\simeq - 0 . 4 9 6 8$ </td><td> $- 2 5 . 8 \% ^ { * }$ </td></tr><tr><td>Triangular J1</td><td>1764</td><td> $- 0 . 3 9 8 3 6 9 \pm 0 . 0 0 0 0 5 0$ </td><td> $\mathrm { { N / A } }$ </td><td> $- 0 . 5 5 0 3 ( 8 )$ </td><td> $7 2 . 4 \% ^ { * }$ </td></tr></table>

We next restart from the corresponding zero-shot checkpoint, retain the route chosen by the pretrained router, and train only the compiled merge tree. As before, this reduces the optimised object from the 547M-parameter foundation model to approximately 4.7M parameters and permits each system to be trained on a single A100 GPU. The MaxCut and PPP runs use 256 walkers, eight replicas, two MCMC moves per KFAC step, and a burn-in of N steps (see SM § S4). Their learning rate is $0 . 0 0 2 / ( 1 + t / 1 0 ^ { 4 } )$ , with damping $1 0 ^ { - 3 }$ and curvature exponential moving average 0.99. The 22×22 and 32×32 square-lattice runs use the same walker, replica and update counts, a 128-step burn-in and $0 . 0 1 / ( 1 + t / 1 0 ^ { 4 } )$ with curvature moving average 0.9.

Figure 11 shows the energy intensiveness in the site count of each problem: qubits for MaxCut, carbons for PPP and physical spins for $J _ { 1 } { - } J _ { 2 }$ . The optimisation traces are strongly front-loaded, as we saw before on smaller systems. At the final logged steps, the $N = 2 5 6$ and 512 MaxCut energies correspond to expected cut weights 2377.5 and 8917.1, or 96.8% and 96.1% of the feasible references. The PPP chains reach −4.9514 and −4.9359 eV/C after $1 0 ^ { 4 }$ steps, within 1.3% and 1.6% of the infinite chain scale under the centered convention. For square-lattice $J _ { 1 } { - } J _ { 2 }$ , the 22×22 case has 484 physical spins in the padded $N = 5 1 2$ routing bucket and improves from $E / N = - 0 . 4 4 6 6 6$ at initialisation to −0.46316 after $1 0 ^ { 5 }$ steps. The 32×32 case has $N = 1 0 2 4$ physical spins and reaches $E / N = - 0 . 4 5 1 4 5 6$ after $3 \times 1 0 ^ { 4 }$ steps, essentially its initial value of −0.44567 after an early transient.

![](images/0b1dacabaf69af18c477188a6a354ff25075becf9fadae00e877eb7de5f00cf6.jpg)

![](images/5ad8d015a11a6e6b6e6448ba2891af131e02787b0a1c9ec8698235f29e691015.jpg)

![](images/0e6deb76b0937fc38ae2801e2478c73bef9e180c9f35e1778d0b2d0ef7f11ca8.jpg)  
Figure 11: Fine-tuning on the large-system case studies. The faint curves are decimated per-step training estimates and the solid curves are non-overlapping block means (250 steps for MaxCut and $\mathrm { P P P ; }$ 500 for $J _ { 1 } { - } J _ { 2 } )$ . Open markers show representative block means. Diamonds at step zero are the independent Axis-A zero-shot evaluations in $^ { ( \mathrm { a } , \mathrm { b } ) }$ and the logged training initialisations in (c). The dashed grey lines in $^ { ( \mathrm { b , c } ) }$ are literature scales, not optimisation targets. (a) MaxCut energy per qubit; more negative energy corresponds to a larger expected cut through Eq. (5.2). (b) Particle–hole-centred PPP energy per carbon. (c) Textbook $J _ { 1 } { - } J _ { 2 }$ energy per physical spin: 22×22 means 484 physical spins in the padded $N = 5 1 2$ routing bucket, while $3 2 \times 3 2$ means $N = 1 0 2 4$ physical spins.

## 5.4 Witnessing a Phase Transition with Zero-Shot Weights

Consider the periodic transverse-field Ising Hamiltonian

$$
H ( g ) = - J \sum _ { i = 1 } ^ { N } \sigma _ { i } ^ { z } \sigma _ { i + 1 } ^ { z } - g J \sum _ { i = 1 } ^ { N } \sigma _ { i } ^ { x } , \qquad \sigma _ { N + 1 } \equiv \sigma _ { 1 } .\tag{5.5}
$$

Here, $g$ is the dimensionless transverse-field strength relative to the nearest-neighbour Ising coupling $J .$ Varying this parameter drives the model through its thermodynamic quantum critical point at $g _ { c } = 1$

For the normalized ground state $| \psi ( g ) \rangle$ of $H ( g )$ , the fidelity between two nearby Hamiltonians is

$$
F ( g , g + \delta g ) = \left| \langle \psi ( g ) \mid \psi ( g + \delta g ) \rangle \right| .\tag{5.6}
$$

The leading response defines the fidelity susceptibility,

$$
\chi _ { F } \left( g + \frac { \delta g } { 2 } \right) \simeq - \frac { 2 \ln F ( g , g + \delta g ) } { ( \delta g ) ^ { 2 } } .\tag{5.7}
$$

Near the transition, the ground-state wavefunction changes rapidly with $^ { g , }$ decreasing the fidelity and producing a peak in $\chi _ { F }$ . The finite-size quantity $\chi _ { F } / N$ therefore provides a practical witness of the transition.

In Fig. 12, we see Hamilton-Zero witnesses this phase-transition with zero-shot inference. This size-consistent maximum near $g _ { c } = 1$ provides a proof of principle that foundation models could in future serve as search engines for phase transitions, see Sec. 6 for further discussion.

![](images/b12e92bbcfa458d1684fffb58dac6074ccf68312286b24895210ff777309d3b6.jpg)  
Figure 12: Fidelity susceptibility per site, $\chi _ { F } / N _ { ; }$ , of the periodic transverse-field Ising chain is shown for $N = 8 .$ , 16, and 32 spins. Every curve is evaluated directly from the same pretrained Hamilton-Zero checkpoint without fine-tuning; markers denote adjacent-field overlap estimates at midpoint $g + \Delta g / 2$ with $\Delta g = 0 . 0 5$ , and shaded regions give 95% confidence intervals. The vertical dashed line marks the thermodynamic phase-transition point, $g _ { c } ~ = ~ 1$ The zero-shot response develops a finite-size maximum close to $g _ { c }$ for all three sizes, providing a phase-transition witness from the fixed pretrained wavefunction.

## 5.5 Approximate Equivariance

We tested the invariance of the predicted mean energy under a group action on the complete physical triplet $( J , h , q )$ corresponding to the interaction graph $( J , h )$ and quaternionic coordinates $q .$ Recall for site-dependent rotations $u _ { i } \in \mathrm { S U } ( 2 )$ , the action is

$$
J _ { i j } \mapsto R ( u _ { i } ) J _ { i j } R ( u _ { j } ) ^ { \mathsf { T } } , \qquad h _ { i } \mapsto R ( u _ { i } ) h _ { i } , \qquad q _ { i } \mapsto q _ { i } { \bar { u } } _ { i } ,\tag{5.8}
$$

where $R ( u _ { i } ) \in \mathrm { S O } ( 3 )$ is the adjoint rotation and $\bar { u } _ { i } = u _ { i } ^ { - 1 }$ . The global test is the diagonal subgroup obtained by setting $u _ { i } = u$ at every site; it is therefore not a rotation of the fields alone. The site-local test instead draws the $u _ { i }$ independently, inducing independent left and right ${ \mathrm { S U } } ( 2 )$ actions at the two ends of each bond.

For each transformed system, we measured the relative mean-energy residual

$$
r _ { E } = \frac { \left| \bar { E } _ { g } - \bar { E } \right| } { \left| \bar { E } \right| } ,\tag{5.9}
$$

using paired baseline and group-transformed samples. To distinguish symmetry violation from Monte Carlo noise, we also estimated a paired-sample standard error. For walker $w ,$ let

$$
\overline { { \Delta E } } _ { w } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \Big ( E _ { w , t } ^ { ( g ) } - E _ { w , t } \Big ) .\tag{5.10}
$$

The relative standard error plotted in Fig. 13 is

$$
\mathrm { S E } _ { \mathrm { r e l } } = \frac { \mathrm { s d } _ { w } \left( \overline { { \Delta E } } _ { w } \right) } { \sqrt { W } \left| \bar { E } \right| } ,\tag{5.11}
$$

with $W = 2 5 6$ walkers and $T = 1 2 8$ steps. This quantity estimates the Monte Carlo uncertainty of each paired residual.

At initialization, the median residual was on the order of unity for both actions: 1.30 for the global action and 1.39 for the site-local action. By step 70 000, these values had fallen to $1 . 4 1 \times 1 0 ^ { - 2 }$ and $3 . 2 8 \times 1 0 ^ { - 2 }$ , respectively. Thus, most of the equivariant behaviour was acquired during the first quarter of training. Subsequent checkpoints show a non-monotonic plateau rather than a steady improvement:

over steps 70 000–280 000, the global median lies between $1 . 1 4 \times 1 0 ^ { - 2 }$ and $2 . 0 1 \times 1 0 ^ { - 2 }$ , whereas the site-local median lies between $2 . 5 7 \times 1 0 ^ { - 2 }$ and $3 . 9 5 \times 1 0 ^ { - 2 }$

At step 280 000, the median residuals are $1 . 1 4 \times 1 0 ^ { - 2 }$ for the global action and $2 . 9 1 \times 1 0 ^ { - 2 }$ for the site-local action. The corresponding median paired-sample standard errors are $2 . 1 2 \times 1 0 ^ { - 3 }$ and $6 . 5 3 \times 1 0 ^ { - 3 }$ . The reduction from order-one residuals at initialization to percent-level residuals after training demonstrates that the model has learned approximate equivariance. The consistently larger site-local residual shows that the full $\mathrm { S U } ( 2 ) ^ { N }$ action is learned less accurately than its global diagonal subgroup.

![](images/463747cd50f7a8d368a1568a07e1c7b7f5f02c181e739bb754b507d4641b35ad.jpg)  
Figure 13: Learning of mean-energy equivariance under joint transformations of $( J , h , q )$ on 20 held-out systems from the (B) evaluation set (see SM § S3). Blue denotes the global diagonal SU(2) action and orange the site-local $\mathrm { S U } ( 2 ) ^ { N }$ action. Solid curves and shaded bands show the median and interquartile range of the relative energy residual, respectively; dashed curves show the median paired-sample Monte Carlo standard error estimated from 256 walkerwise averages of 128 steps.

## 5.6 Router Commitment and Generalisation

Recall from Sec. 3 (see also SM § S2) that the router converts the Hamiltonian-conditioned trunk embeddings into a permutation of the physical sites and padded leaves, thereby selecting a binary contraction tree used by the wavefunction. In this section, we visualise the map that the router picks, showing that it recovers the entanglement structure of the case studies of 5.3. For visualisation, we map the saved route from its padded, bit-reversed compiler labels back to canonical physical coordinates. The resulting diagrams therefore show the contraction actually selected by the model.

At zero-shot evaluation, the router decodes eight candidate trees. Each candidate is subjected to a short burn-in and measurement contest, after which the winning ordering is compiled and used for the full evaluation. Owing to the fact that our algorithm selects from a contest, we can quantify the concentration of the route policy by its commitment,

$$
C _ { \mathrm { r o u t e } } : = 1 - { \frac { H _ { \mathrm { p o l i c y } } } { H _ { \mathrm { m a x } } } } ,\tag{5.12}
$$

where $C _ { \mathrm { r o u t e } } \simeq 0$ denotes a difuse policy and $C _ { \mathrm { r o u t e } } \simeq 1$ denotes concentration on a small set of preferred, potentially symmetry-related routes. Figure 14(a) shows the median commitment across the 13 suficiently populated system categories of the pretraining dataset (SM § S3), with the grey band giving the interquartile range, while Fig. 14(b) resolves the individual category trajectories. Commitment rises from approximately 0.5 to at least 0.96 during the first eight augmentation rounds and remains close to saturation thereafter. The agreement across categories is indicative of the router learning a stable structural preference early in pretraining, rather than acquiring a diferent arbitrary ordering for each family late in the run. It is thus able to train efectively over a heterogeneous set of Hamiltonians in the pretraining set.

![](images/98f89b0c7f2d50bba46b68e45e033722035f7408521d73ac35b2801482fac869.jpg)  
Figure 14: Entropy-realized commitment, $1 - H _ { \mathrm { p o l i c y } } / H _ { \mathrm { m a x } } .$ , by augmentation round for the 13 system categories with $S \geq 8 0$ . (a) Round-wise median across categories, with the grey band denoting the interquartile range. (b) Stratification by system-category, see Fig. S13 of SM § S3. All categories commit rapidly during the first ∼8 rounds and subsequently remain near saturation.

Indeed, figure 15 shows there is a direct correlation between commitment and the energy quality of Hamilton-Zero’s predictions; a higher level of commitment generally signals a lower gap to ED. Panel (a) follows the round-wise median in the plane of router commitment and relative gap to exact diagonalisation for systems in the pretraining set where ED data is available. Panel (b) shows the corresponding trajectories for the individual system categories, showing this trend is consistent between diferent types of system. Commitment is therefore a correlational diagnostic rather than an energy certificate: a concentrated policy may still select a poor ordering.

![](images/98546658f4f8e1c9e74479474a09e63e445e92f230369afdcb3d325444cdba5a.jpg)  
Figure 15: Router commitment versus relative gap to ED over augmentation rounds $0  5 5$ for the 13 system categories of our pretraining dataset (SM § S3). (a) The round-wise median trajectory; grey shading denotes the interquartile range in the commitment and relative-gap coordinates. (b) System-category trajectories, with the enlarged marker indicating the final round.

The router’s selected routes also generalise to system sizes and topologies beyond those seen in training. On the $1 0 \times 1 0 ~ J _ { 1 }$ –J<sub>2</sub> torus (Fig. 16), the first merge pairs nearest-neighbour spins and the second produces exact $2 \times 2$ plaquettes for every fully occupied four-site cell. Larger cells then remain spatially local as they are combined towards the root. No lattice geometry is supplied to the contraction mechanism, this plaquette hierarchy is inferred from the Hamiltonian-conditioned representations and selected because it supports a lower variational energy. This is supporting evidence that the router is able to recover the locally entangling geometry of the J1J2 model, at a scale above its pretraining.

More strikingly, the same behaviour survives far beyond the pretraining sizes. The $4 5 \times 4 5$ square lattice in Fig. 17 contains 2,025 physical sites, more than thirty times the largest physical size used in pretraining. Nevertheless, the zero-shot route first forms local cells and subsequently organises them into extended diagonal bands consistent with the frustrated $J _ { 1 }$ – $J _ { 2 }$ interaction structure. At Step 6, 75.0% of nearest-neighbour bonds lie within the same 64-leaf cell, compared with 3.1% under random ordering; at the final root split, 91.4% lie within one branch, compared with the random baseline of 50.0%. The router has therefore preserved interaction locality across eleven merge levels without having encountered comparable system sizes during training.

![](images/264b089a071a3d3eb529db74b55dfdb1df560937e666d49d18edcb70e3d355ce.jpg)

![](images/d37f771675464db9171e30deeeb582828bf575e5eb6bbf00b4cb031745b17152.jpg)

![](images/61980e1e79c37b3967cd92f2572f14ed95b4f9f76cfb4c5f990cdb141444a486.jpg)

![](images/a1b7fda533a5b094185d5bc2f109c33689f2d92f843554d4032139cfd87b1a97.jpg)

![](images/a5016bfdc0e10807f403e4ed827752c118aacaccc0bbc444e4fcd14e2a29981f.jpg)

![](images/940106501c730cf3f91e52870df926e0648d3d034a5901d801136b4bf81d910a.jpg)

![](images/105b4be36e6adad10e039c1a0c6b289f7141a669351887c9e1265c95deb8a0b4.jpg)

![](images/d98124c5de7c29cdad99953283a8a449089621e71a62540c7ea6d0f11f87041b.jpg)  
Figure 16: Learned coarse-graining of the $1 0 \times 1 0 ~ J _ { 1 } – J _ { 2 }$ torus. The panels are read along the arrows, with successive rows traversed in opposite directions. Step 0 shows the 100 physical lattice sites. At Step k, the black boundaries enclose the efective cells remaining after k levels of binary merging; a boundary disappears when the two cells on either side are merged. Colours are reassigned independently at each step solely to distinguish adjacent cells and therefore do not track cell identity between panels. For $J _ { 2 } / J _ { 1 } = 0 . 3$ , the first merge pairs only nearest-neighbour sites. At Step 2, all 24 fully occupied four-site cells are exact $2 \times 2$ plaquettes. The few incompletely occupied cells arise because the 100 physical sites are embedded in a 128-leaf binary tree, with 28 virtual leaves. Subsequent levels combine the plaquettes into progressively larger spatially local regions, culminating in two root branches at Step 6 and a single scalar at Step 7 of the merge-tree.

![](images/b7d72f8e9839ec1c8a9ecc7f8c2ae19285c33434d5974a4ac3fe380085c330ed.jpg)  
Figure 17: Zero-shot coarse-graining of the $4 5 \times 4 5$ square $J _ { 1 } { - } J _ { 2 }$ lattice. The panels follow the arrows from the 2,025 physical sites at Step 0 to the scalar output at Step 11. At each step, black boundaries delimit the current efective cells and disappear when those cells are merged. Colours are reassigned independently within each panel only to separate adjacent cells visually. At $J _ { 2 } / J _ { 1 } = 0 . 5$ , the early merges are predominantly local. The larger efective cells subsequently organize into extended diagona bands, showing that the learned hierarchy preserves spatial structure far beyond the training sizes. At Step 6, 75.0% of nearest-neighbour bonds lie within the same 64-leaf cell, compared with a randomorder mean of 3.1%. At the root split in Step 10, 91.4% lie within one branch, compared with 50.0% for a random ordering. Some cells contain fewer than $2 ^ { k }$ physical sites because the lattice is embedded in a 2,048-leaf tree containing 23 virtual leaves.

The PPP–Ohno case of 5.3 provides a contrasting route. Fig. 18 shows the router first pairs the two spin orbitals belonging to each carbon, then combines neighbouring carbons into four-orbital cells, and thereafter doubles the length of the contiguous carbon blocks at each level. It recovers the chemically natural nested factorisation of a long-range interacting fermionic chain despite operating on qubitlabelled Hamiltonian data [103–105]. This shows the router demonstrates transfer across interaction type as well as scale.

Together, these examples show that the router has not merely memorised a site ordering or learned a generic preference for balanced trees. It extracts a topology-dependent, multiscale contraction hierarchy from the Hamiltonian representations and transfers that hierarchy across periodic and open lattices, long-range chains, interaction families, and system sizes. The result is evidence that pretraining has produced a reusable structural prior for organising entanglement, one that can be converted directly into the compiled contraction path used for inference.

![](images/614a6846f0de7706a58bc7589e1fbae6853b6c02595cda2ea13480aff57bc669.jpg)  
Figure 18: Learned coarse-graining of the 128-carbon PPP–Ohno chain. The 256 spin orbitals are folded into a serpentine arrangement for visualization only, so that consecutive carbons remain adjacent when the chain passes between rows. The two spin orbitals belonging to each carbon are placed vertically. Panels are read along the arrows. Black boundaries enclose the efective cells present at each merge level and disappear when cells are combined; colours are reassigned independently at every step and serve only to distinguish neighbouring cells. At Step 1, every efective cell contains the two spin orbitals of one carbon. Step 2 combines adjacent carbons into four-orbital cells, and every subsequent level doubles the length of the contiguous carbon block. The hierarchy therefore proceeds through blocks of 1, 2, 4, 8, . . . , 64 carbons before the final scalar merge at Step 8. This exact nested structure provides direct evidence that the router has learned the interaction-local factorisation of the PPP Hamiltonian.

## 6 Discussion and outlook

Quantum computing has attracted great interest in both academic and industrial settings on the premise that classical cost will continue rising until useful quantum many-body calculations have to move onto quantum hardware. Hamilton-Zero takes the other side of this premise for ground state problems specifically. As a foundation model, pay once for learning ground states across a distribution of Hamiltonians and store what the model learns in its weights. As a result, we have a classical model that improves with pretraining, transfers across interaction graphs and system sizes, and returns an explicit, diferentiable wavefunction on hardware that exists today.

We believe this changes the economics of quantum ground-state computation. Most classical ground-state solvers restart, or substantially re-optimise, for each new Hamiltonian. Their cost scales with the number of problems solved. Hamilton-Zero replaces repeated training from scratch with one pretraining run whose cost is shared across every subsequent use of the checkpoint. If we let M denote the number of target Hamiltonians evaluated using the same pretrained checkpoint, and let R denote the number of additional fine-tuning runs performed across those targets. So $R = 0$ for a completely zero-shot workload, $R = M$ if every target is fine-tuned separately, and $0 < R < M$ if one fine-tuned checkpoint is reused across several related Hamiltonians. Then we denote $C _ { \mathrm { p r e } }$ for the dollar cost of producing the pretrained checkpoint, $C _ { \mathrm { e v a l } } ^ { ( m ) }$ for the dollar cost of applying a fixed checkpoint to target Hamiltonian $m ,$ , including inference, MCMC sampling and estimation of the reported observables, and $C _ { \mathrm { f t } } ^ { ( r ) }$ for the dollar cost of fine-tuning run r. The average cost per target Hamiltonian is then

$$
\frac { C _ { \mathrm { H Z } } ( M ) } { M } = \frac { C _ { \mathrm { p r e } } } { M } + \frac { 1 } { M } \sum _ { m = 1 } ^ { M } C _ { \mathrm { e v a l } } ^ { ( m ) } + \frac { 1 } { M } \sum _ { r = 1 } ^ { R } C _ { \mathrm { f t } } ^ { ( r ) } .\tag{6.1}
$$

The first term falls as $1 / M$ , while the evaluation term remains the marginal cost of using the model. Meanwhile the fine-tuning term depends on how often fine-tuning is required and how widely each fine-tuned checkpoint can subsequently be reused. In the zero-shot setting the final sum vanishes.

At the measured throughput of our inference stack, drawing one sample from Hamilton-Zero costs below $1 0 ^ { - 6 }$ USD. Public quantum hardware is already priced in the same unit, with services like Amazon Braket listing on-demand QPU prices from $4 . 2 5 \times 1 0 ^ { - 4 }$ to $8 . 0 \times 1 0 ^ { - 2 }$ USD per shot in addition to a 0.30 USD charge for every submitted job [112]. Even if we discard the job charge entirely, one quantum shot therefore costs between 425 and 80,000 times one Hamilton-Zero sample at current public prices. Amazon’s own error-mitigation example prices an IonQ task of 2,500 shots at 200.30 USD [112]; the same number of Hamilton-Zero samples costs approximately 0.0025 USD. This is still the comparison most favourable to quantum hardware, one shot returns one measurement outcome, whereas estimating a ground-state energy requires repeated state preparation and measurements across many operator groups.

Indeed, when we compare this more broadly with the larger research and engineering costs or the continued development and improvement of quantum hardware, the diference is also stark. Improving Hamilton-Zero requires more pretraining compute, which distributes immediately with the release of new model weights, whereas hardware expenditure depreciates. Furthermore, our scaling laws show Hamilton-Zero’s compute costs scale around 5 times more favourably than today’s LLMs [108], making this a relatively resource-eficient compute demand by today’s standards. For reference, the checkpoint we release in this work, including its previous iterations, cost around \$30k to pretrain from scratch. Evaluation and fine-tuning cost another \$30k using on-demand resources. Since a foundation model can improve, absorb new problem classes, and serve more users from the same weights, the economic question of useful quantum advantage is now whether a quantum device can produce a ground-state estimate at lower total cost than a pretrained classical model whose marginal cost will continue to fall with every subsequent release of such models.

Hence, with Hamilton-Zero, we believe the baseline of useful quantum advantage in ground state computation has moved. Comparisons against a classical model trained from scratch establish only an advantage over cold-start optimization, but should, in fact, compare against amortized computation now that a model like Hamilton-Zero is known to be workable. This is especially crucial in the knowledge that to date, no quantum computer has solved quantum chemistry problems beyond a few dozens of qubits at a time for example, yet Hamilton-Zero is able to compile wave-functions to the 8000 qubits scale. From this point, a ground-state quantum-advantage claim should report the same Hamiltonians, the same observables, the same target error and the same confidence level, while counting state preparation, sampling, mitigation, classical outer-loop optimisation and verification. It should then compare total wall time, dollar-cost and energy-cost against both zero-shot Hamilton-Zero and Hamilton-Zero fine-tuned at the same budget if such a claim is to hold up against this model. Until a quantum device wins that comparison, it beats only a now-obsolete classical baseline.

We also believe in future Hamilton-Zero could change how we can search for new physics through phase diagrams. The fact that the fixed weights could witness a phase transition in the Ising mode is promising evidence that this is a realistic goal for such foundation models. Indeed, a conventiona phase-diagram study solves a sparse collection of coupling points in the Hamiltonian’s parameter-space and spends most of its compute rediscovering nearby ground states. Whereas Hamilton-Zero swept the coupling space of the Ising model in zero-shot inference. This avenue of identifying sharp changes in fidelity, correlations, or order parameters, can let us concentrate fine-tuning where the phase structure changes in future. It could also be used to supply synthetic samples for moment-matrices that underpin recent advances in semi-definite programming techniques for mapping phase diagrams [113]. As such in this role the model nominates regions for exact methods or experiment.

Hamilton-Zero is also not without limitations however. First, the cost of compute is $\mathrm { p o l y } ( 2 ^ { \mathrm { c e i l } ( l o g _ { 2 } N ) } )$ due to padding that aims to construct a perfectly balanced tree. So, a 129-qubit system costs the same as a 256-qubit one, as they share the next power of 2 available. Second, the automatic diferentiation primitives to evaluate energies have an order of diferentiation that scales with the weight of the Pauli string present in the Hamiltonian. As such, we restricted ourselves to quadratic Hamiltonians. However, we do note that such a set is universal in the sense of quantum computation [44, 114] as it corresponds to two-qubit gates. Hence, higher-order terms can be compiled onto a system of physica and ancilla qubits at quadratic order. Finally, even though QMC is variational by design, finite-sample Monte-Carlo estimate may violate the variational inequality by either of two mechanisms: estimator noise and/or mixing bias, which is dependent on the spectral gap [115] and how long the burn-in is [106] , which can always be alleviated with larger efective sample size and longer burn-in. We report confidence intervals, which also do not guarantee that its upper bound would be above the ground state energy, as confidence intervals only have a guarantee to contain true value within repeated sampling. To empirically minimize bias efects, we conducted 3 diferent stationarity tests on energy values to detect distribution drifts during estimation, and chose burn-in duration empirically based on this; see SM § S4.

Three extensions follow naturally from this work. First, we can create versions of Hamilton-zero that were fine-tuned on larger datasets to boost its performance in sector-specific problem families such as quantum chemistry or combinatorial optimisation. This would likely further amortize subsequent fine-tuning of individual systems and lead to even stronger zero-shot inference. Second, we can extend Hamilton-Zero’s capabilities from static ground state problems to dynamics via a time encoding. For a pretraining dataset of time-dependent Hamiltonians H(t), we could train against the Schr¨odinger residual i∂ ψ −H(t)ψ together with an initial-condition loss, as was done in deep stochastic mechanics of bosonic systems [116]. Finally, a per-site-odd but nonlinear successor could use higher half-integer Peter–Weyl sectors. Unlike the multilinear architecture used here, it would require the certified Casimir penalty of Appendix A to retain a physical upper bound. We leave both constructions and their validation to future work.

## Acknowledgements

We thank the NVIDIA Inception Program for hardware support and technical engagement, the Google for Startups Program for cloud infrastructure support, and Nebius for providing the GPU compute that supported this work’s training runs. We would also like to thank Jose Ramon Martinez for his help proof-reading the manuscript.

## Supplementary Material

In the supplementary material, we describe the theory, architecture, datasets, optimisation, and engineering that underpin Hamilton-Zero. SS1 covers variational principles on spinor manifolds, with a full exposition on the theory behind our automatic-diferentiation primitives of Lie derivatives, the Peter-Weyl theorem that shows why our model has support on a larger manifold than NQS, as well as how our variational principle manages the case of non-linear quaternionic functions through a Casimiroperator-based regulariser and energy penalty that preserves the spin-1/2 variational principle. Next SS2 describes Hamilton-Zero’s architecture, drawing parallels and design choices from large-scale pretraining runs of large-language models. Here we also describe the router we use to learn entanglement structure through deep reinforcement learning, characterise its automorphism-orbit entropy, as well as the tree tensor-network-style contraction mechanism that is conditioned on the model’s embeddings of interaction data. Next, we describe in full detail the pretraining routines and datasets we used in SS3. We then describe in SS4 our state-of-the-art sampling scheme for MCMC on the configuration manifold $S U ( 2 ) ^ { N }$ , our energy kernels, and our extension of the KFAC optimiser, all of which we made shard-mappable and GEMM-optimised for training on Nvidia H200 GPUs.

## S1 Variational principles on Spinor Manifolds

## S1.1 From basis amplitudes to wavefunctions on $\mathrm { S U } ( 2 ) ^ { N }$

The standard $\operatorname { s p i n } - { \frac { 1 } { 2 } }$ wavefunction is a complex-valued function on $\{ + , - \} ^ { N }$ , equivalently a vector $\Psi \in \mathbb { C } ^ { 2 ^ { N } }$ with one amplitude per computational-basis configuration. Two features of this picture constrain every method built on it. The dimension is exponential in N, and the domain is discrete. Thus a Hamiltonian acts through explicit sums over connected configurations, enumerated anew for each interaction structure. Tensor networks and NQS address the first constraint by replacing the $2 ^ { N }$ -dimensional space with a lower-dimensional variational sub-manifold. The second constraint they inherit unchanged however, and it is what ties each trained model to one Hamiltonian’s connectivity. The central objective of this section is to remove both at once, by replacing the discrete domain with the continuous group manifold $\mathrm { S U } ( 2 ) ^ { N }$ , on which Hamiltonians act by diferentiation.

Tensor networks and neural quantum states replace the exponentially large Hilbert space by a lower-dimensional variational family [106, 117], but the resulting optimisation can still be limited by stability and trainability. On the quantum-circuit side, barren plateaus provide a distinct failure mode in which gradient variance can vanish exponentially with system size [20].

The central objective of this section is to expose the methods that allow us to sidestep both the curse of dimensionality and the curse of linearity, by replacing the discrete domain of the spin-1/2 basis vectors with a continuous Lie group domain. The wavefunctions we employ,

$$
\psi : \left( \mathrm { S U } ( 2 ) \right) ^ { N } \to \mathbb { C } ,\tag{S1.1}
$$

are smooth functions on the Lie group manifold. Here, each spin is parametrised by a unit quaternion $q _ { i } \in S ^ { 3 }$ via the standard identification $\mathrm { S U } ( 2 ) \cong S ^ { 3 }$ embedded in $\mathbb { R } ^ { 4 }$ . Hence any neural network model in our framework thus consumes 4N continuous inputs for N bodies which is linear in N.

As usual with representation theory, there is no free lunch. In the following, we will see the signature of the Hilbert-space curse of dimensionality in this picture, and explain how to side-step it by relaxing into a larger domain, $L ^ { 2 } ( \mathrm { S U } ( 2 ) ^ { N } )$ , than the spin- <sup>1</sup> sector, while maintaining the variational principle, so that we can find ground-state energies of many-body systems.

First, though, let us build some intuition for the quaternionic representation, so that examples later in this part can be written directly in $( q _ { 0 } , q _ { 1 } , q _ { 2 } , q _ { 3 } )$ . Recall a unit quaternion $q = ( q _ { 0 } , q _ { 1 } , q _ { 2 } , q _ { 3 } ) \in S ^ { 3 }$ $( q _ { 0 } ^ { 2 } + q _ { 1 } ^ { 2 } + q _ { 2 } ^ { 2 } + q _ { 3 } ^ { 2 } = 1 )$ is the standard coordinate on SU(2) via

$$
g ( q ) = q _ { 0 } I + q _ { a } \left( i \sigma ^ { a } \right) = \binom { q _ { 0 } + i q _ { 3 } q _ { 2 } + i q _ { 1 } } { i q _ { 1 } - q _ { 2 } } , \qquad g ( q ) \in \mathrm { S U } ( 2 ) ,\tag{S1.2}
$$

where det $g ( q ) = q _ { 0 } ^ { 2 } + q _ { 1 } ^ { 2 } + q _ { 2 } ^ { 2 } + q _ { 3 } ^ { 2 } = 1 , \mathrm { a n d } g ( q ) g ( q ) ^ { \dag } = I ,$

The four entries of $g ( q )$ are the spin- <sup>1</sup><sub>2</sub> Wigner D-matrix elements. That is, for any spin $j , D _ { m n } ^ { j } ( q )$ is the matrix element of the rotation representation on the spin-j irrep. For $\begin{array} { r } { j = \frac { 1 } { 2 } } \end{array}$ with row/column indices m, $n \in \{ 0 , 1 \}$ ,

$$
D _ { m n } ^ { 1 / 2 } ( q ) \ = \ g _ { m n } ( q ) , \qquad D _ { 0 0 } ^ { 1 / 2 } ( q ) = q _ { 0 } + i q _ { 3 } ,\tag{S1.3}
$$

To identify physical single-site spin states with SU(2)-functions, we fix the row $m = 0$ , because the column index carries the physical spin. Thus

$$
| \uparrow \rangle \longleftrightarrow D _ { 0 0 } ^ { 1 / 2 } ( q ) = q _ { 0 } + i q _ { 3 } , \qquad | \downarrow \rangle \longleftrightarrow D _ { 0 1 } ^ { 1 / 2 } ( q ) = q _ { 2 } + i q _ { 1 } .\tag{S1.4}
$$

A general single-site state $\alpha | \uparrow \rangle + \beta | \downarrow \rangle$ becomes

$$
\psi ( q ) = \alpha ( q _ { 0 } + i q _ { 3 } ) + \beta ( q _ { 2 } + i q _ { 1 } ) .\tag{S1.5}
$$

For N sites, let $s _ { k } \in \{ 0 , 1 \}$ with $0 \equiv \uparrow$ and $1 \equiv \downarrow$ . Then

$$
{ \left. s _ { 1 } \cdot \cdot \cdot s _ { N } \right. } \longleftrightarrow \prod _ { k = 1 } ^ { N } D _ { 0 , s _ { k } } ^ { 1 / 2 } ( q _ { k } ) ,\tag{S1.6}
$$

and a general $\mathrm { s p i n } { - } \frac { 1 } { 2 }$ state is a linear combination of these products; see $\mathrm { S M \ \ S \ S 1 . 3 }$

Recall a variational ansatz $\Psi _ { \theta }$ over the physical Hilbert space $( \mathbb { C } ^ { 2 } ) ^ { \otimes N }$ gives an upper bound

$$
\frac {  \Psi _ { \theta } | H | \Psi _ { \theta }  } {  \Psi _ { \theta } | \Psi _ { \theta }  } \ge E _ { 0 }\tag{S1.7}
$$

for any normalised $\Psi _ { \theta } .$ . Minimising this over $\theta$ converges to (or above) $E _ { 0 }$ . This is the variational principle in its standard form, the rest of SM § S1 is about preserving this inequality under our non-standard quaternionic parameterisation.

We instantiate the manifold amplitude as a neural network returning

$$
\log \psi _ { \theta } : \mathrm { S U ( 2 ) } ^ { N } \longrightarrow \mathbb { C } .\tag{S1.8}
$$

For the multilinear architecture, its Peter–Weyl coeficients carry a row-index multiplicity and a column-index physical-spin factor. Fixing a row multiplicity vector recovers the state-vector lift described above; a general model state is interpreted through the decomposition of Sec. S1.3.

We defer the details of the architecture to SM § S2, but the signature is a smooth non-linear function on the manifold, to be optimised by gradient descent on a variational loss. As is standard practise in Quantum Variational Monte Carlo, we use log ψ<sub>θ</sub> to prevent underflow, and to allow for stable natural gradient flow [106].

Having now fixed our model to be a smooth 4N-input, 1-output complex-valued network, predicting log ψ on the Lie-group manifold, we can now turn to the question of how the Hamiltonian acts on this object, and how to project this object into the physical spin- $\cdot \frac { 1 } { 2 }$ sector of this function family.

## S1.2 Operators with automatic diferentiation

In this subsection we explain how the shift away from Hilbert space lets us realise the action of spin operators as Lie derivatives, executable directly through automatic diferentiation. We restrict attention to quadratic, that is two-body, Hamiltonians throughout, since two-body interactions are universal for quantum [118], and higher-order interactions can be built from them [119, 120].

Recall from the standard theory that at each site i, the spin is carried by three Hermitian operators $\hat { S } _ { i } ^ { x } , \hat { S } _ { i } ^ { y } , \hat { S } _ { i } ^ { z }$ , closing the algebra $[ \hat { S } _ { i } ^ { a } , \hat { S } _ { i } ^ { b } ] = i \varepsilon _ { a b c } \hat { S } _ { i } ^ { c }$ Their squared magnitude is the Casimir $\hat { S } _ { i } ^ { 2 } : =$ $( \hat { S } _ { i } ^ { x } ) ^ { 2 } + ( \hat { S } _ { i } ^ { y } ) ^ { 2 } + ( \hat { S } _ { i } ^ { z } ) ^ { 2 }$ , which on a single spin- <sup>1</sup> is the scalar $\textstyle { \frac { 3 } { 4 } }$ , and we write $\hat { \sigma } _ { i } ^ { a } : = 2 \hat { S } _ { i } ^ { a }$ for the associated Pauli matrices. The most general quadratic Hamiltonian we target couples these Paulis pairwise, together with an on-site field,

$$
\hat { H } \ = \ \frac { 1 } { 4 } { \sum _ { i < j } \sum _ { a , b } } J _ { i j } ^ { a b } \hat { \sigma } _ { i } ^ { a } \hat { \sigma } _ { j } ^ { b } \ + \ \frac { 1 } { 2 } { \sum _ { i , a } } h _ { i } ^ { a } \hat { \sigma } _ { i } ^ { a } ,\tag{S1.9}
$$

with exchange tensor $J \in \mathbb { R } ^ { N \times N \times 3 \times 3 }$ and field $\boldsymbol { h } \in \mathbb { R } ^ { N \times 3 }$ . As stated in the main text, it is useful to think of J as the data of a weighted interaction graph on N sites, where each unordered pair $( i , j )$ with $i \neq j$ carries a $3 \times 3$ Pauli-pair coupling matrix $J _ { i j } \in \mathbb { R } ^ { 3 \times 3 }$

To move this operator onto the manifold, we use on the ith copy of ${ \mathrm { S U } } ( 2 )$ the left-invariant vector fields

$$
( L _ { i } ^ { a } f ) ( q _ { i } ) : = \left. { \frac { \mathrm { d } } { \mathrm { d } t } } f ( q _ { i } \exp ( t X _ { a } ) ) \right| _ { t = 0 } ,\tag{S1.10}
$$

the anti-Hermitian infinitesimal generators of right translations. For $D _ { m n } ^ { j } ( q ) = \langle j , m | D ^ { j } ( q ) | j , n \rangle$ , they act on the column index:

$$
L _ { i } ^ { a } D _ { m n } ^ { j } ( q _ { i } ) = \sum _ { r } D _ { m r } ^ { j } ( q _ { i } ) \left( \mathrm { d } D ^ { j } ( X _ { a } ) \right) _ { r n } .\tag{S1.11}
$$

We therefore use the column index n as the physical spin index and the row index m as the Peter– Weyl multiplicity. The fields close $[ L _ { i } ^ { a } , L _ { i } ^ { b } ] = \kappa \varepsilon _ { a b c } L _ { i } ^ { c }$ , with $\kappa = - 2$ in the unit-length (half-Lie) normalisation we adopt. Setting $\hat { S } _ { i } ^ { a } \ = \ c L _ { i } ^ { a }$ and matching the two algebras forces $c \kappa = i ,$ hence $c = i / \kappa = - \textstyle { \frac { i } { 2 } }$ and $\hat { \sigma } _ { i } ^ { a } = 2 c L _ { i } ^ { a } = - i L _ { i } ^ { a }$

Under this identification each Pauli pair becomes $\hat { \sigma } _ { i } ^ { a } \hat { \sigma } _ { j } ^ { b } = - L _ { i } ^ { a } L _ { j } ^ { b }$ and each field term becomes $\hat { \sigma } _ { i } ^ { a } = - i L _ { i } ^ { a }$ , so the Hamiltonian (S1.9) turns into the second-order diferential operator

$$
\hat { H } \  \ - \frac 1 4 \sum _ { i < j , a , b } J _ { i j } ^ { a b } L _ { i } ^ { a } L _ { j } ^ { b } \ - \ \frac i 2 \sum _ { i , a } h _ { i } ^ { a } L _ { i } ^ { a }\tag{S1.12}
$$

on $L ^ { 2 } ( \mathrm { S U } ( 2 ) ^ { N } )$ . The same identification sends the spin magnitude to a Laplacian: $\begin{array} { r } { \hat { S } _ { i } ^ { 2 } = c ^ { 2 } \sum _ { a } ( L _ { i } ^ { a } ) ^ { 2 } = } \end{array}$ $- \Delta _ { i } .$ , where $\begin{array} { r } { \Delta _ { i } : = \frac { 1 } { 4 } \sum _ { a } ( L _ { i } ^ { a } ) ^ { 2 } } \end{array}$ is the Laplace–Beltrami operator on the i-th sphere.

What makes $\mathrm { ( S 1 . 1 2 ) }$ executable is that each $L _ { i } ^ { a }$ is a known linear combination of the four partial derivatives $\partial / \partial q _ { i , \mu }$ at site i,

$$
{ \cal L } _ { i } ^ { a } = \sum _ { \mu = 0 } ^ { 3 } A _ { \mu } ^ { a } ( q _ { i } ) \frac { \partial } { \partial q _ { i , \mu } } ,\tag{S1.13}
$$

with coeficients $A _ { \mu } ^ { a }$ the left-invariant frame on $S ^ { 3 } \cong \operatorname { S U } ( 2 )$ , obtained by right-multiplying $q _ { i }$ by the three imaginary quaternion units $\hat { \imath } , \hat { \jmath } , \hat { k }$

$$
\begin{array} { r l r } & { } & { \left[ A _ { \mu } ^ { x } ( q ) \right] _ { \mu = 0 , 1 , 2 , 3 } = ( - q _ { 3 } , \ q _ { 2 } , \ - q _ { 1 } , \ q _ { 0 } ) , } \\ & { } & { \left[ A _ { \mu } ^ { y } ( q ) \right] = ( - q _ { 2 } , \ - q _ { 3 } , \ q _ { 0 } , \ q _ { 1 } ) , } \\ & { } & { \left[ A _ { \mu } ^ { z } ( q ) \right] = ( - q _ { 1 } , \ q _ { 0 } , \ q _ { 3 } , \ - q _ { 2 } ) . } \end{array}\tag{S1.14}
$$

A direct expansion using $| q | ^ { 2 } = 1$ gives the identity we will lean on,

$$
\sum _ { a \in \{ x , y , z \} } A _ { \mu } ^ { a } ( q ) A _ { \nu } ^ { a } ( q ) = \delta _ { \mu \nu } - q _ { \mu } q _ { \nu } = \left[ { \cal P } _ { T _ { q } S ^ { 3 } } \right] _ { \mu \nu } ,\tag{S1.15}
$$

so the three tangents $A ^ { a } ( q )$ are orthonormal and their outer product is the ambient projector onto $T _ { q } S ^ { 3 }$ . This is what makes $\{ L ^ { a } \}$ a genuine orthonormal frame of the round metric, and hence $\Delta _ { i }$ the Laplace–Beltrami operator named above. When two Lie derivatives act on the same site the channel sum (S1.15) applies; across two distinct sites the two frames carry diferent arguments and are simply transported along.

Equation (S1.13) makes the diferential action computationally tractable. The custom forward-Laplacian propagates the value, the required first derivatives, and their contracted second derivatives together through one structured program traversal, never needing to materialise a dense Hessian. We can also avoid running an independent second-order pass for every pair. For a Hamiltonian with edge set $\mathcal { E } ,$ the pair contraction has leading combinatorial cost $\mathcal { O } ( | \mathcal { E } | )$ , which is $\mathcal { O } ( N ^ { 2 } )$ in the dense case. Here $\mathcal { E }$ is the Hamiltonian’s interacting edge set. In the balanced readout, whose padded width is $\hat { N } : = 2 ^ { \lceil \log _ { 2 } N \rceil }$ , the cross-derivative work summed over levels is likewise $\mathcal { O } ( \hat { N } ^ { 2 } )$ with model-width factors suppressed. .

Applying the frame (S1.13) to the Lie form (S1.12) now gives the action of $\hat { H }$ on $\psi$ in quaternionic coordinates. In a pair term the outer derivative $L _ { i } ^ { a }$ meets the frame $A ^ { b } ( q _ { j } )$ of the other site as a constant, so it contributes a pure second derivative with no first-order remainder, while the field term stays first order. Hence, for any $J _ { i j } ^ { a b }$

$$
\hat { \cal H } \psi ( q ) \ = \ - \ \frac 1 4 \sum _ { i < j , a , b } J _ { i j } ^ { a b } \sum _ { \mu , \nu = 0 } ^ { 3 } A _ { \mu } ^ { a } ( q _ { i } ) A _ { \nu } ^ { b } ( q _ { j } ) \frac { \partial ^ { 2 } \psi } { \partial q _ { i , \mu } \partial q _ { j , \nu } } ( q ) \ - \ \frac { i } { 2 } \sum _ { i , a } h _ { i } ^ { a } \sum _ { \mu } A _ { \mu } ^ { a } ( q _ { i } ) \frac { \partial \psi } { \partial q _ { i , \mu } } ( q ) .\tag{S1.16}
$$

One step remains to make this usable. The network returns log $\psi _ { \theta }$ , not $\psi _ { \boldsymbol { \theta } }$ , whereas the local energy needs ratios $L _ { i } ^ { a } L _ { j } ^ { b } \psi _ { \theta } / \psi _ { \theta }$ . Writing $\psi = e ^ { \log \psi }$ and applying the two first-order operators in turn bridges the two: for any twice-diferentiable $\psi \neq 0$ , we have that,

$$
\frac { L _ { i } ^ { a } L _ { j } ^ { b } \psi } { \psi } \ = \ L _ { i } ^ { a } L _ { j } ^ { b } \log \psi \ + \ \big ( L _ { i } ^ { a } \log \psi \big ) \big ( L _ { j } ^ { b } \log \psi \big ) .\tag{S1.17}
$$

On a single site, summing the diagonal channels means the spin magnitude $\hat { S } _ { l } ^ { 2 } = - \Delta _ { l }$ reads,

$$
\frac { - \Delta _ { l } \psi } { \psi } = - \Delta _ { l } \log \psi - { \textstyle \frac { 1 } { 4 } } \sum _ { a } \bigl ( L _ { l } ^ { a } \log \psi \bigr ) ^ { 2 } .\tag{S1.18}
$$

Both (S1.17) and (S1.18) depend only on log $\psi _ { \theta }$ and its first two derivatives at the sampled point. These same quantities determine local estimators for observables containing at most two spin operators, including all two-point correlation functions. In (S1.16), the Hamiltonian enters through the coeficients $J _ { i j } ^ { a b }$ and $h _ { i } ^ { a }$ that contract these derivatives. In particular, the nonzero pattern of J carries the interaction graph. The graph is therefore supplied as data to the same diferential expression, rather than encoded in a lattice-specific rule. Changing the topology or system size changes the tensor entries and index ranges, and the form of the calculation remains identical aside from that.

## S1.2.1 Three examples from the many-body literature

To make (S1.16) concrete, we read of three couplings that recur across the spin-model literature. The isotropic Heisenberg model $\begin{array} { r } { \hat { H } _ { X X X } = J \sum _ { \langle i j \rangle } \hat { \sum _ { a } } \hat { S _ { i } ^ { a } } \hat { S _ { j } ^ { a } } } \end{array}$ has $J _ { i j } ^ { a b } = J \delta ^ { a b }$ on each bond and no field; its Lie form is $\begin{array} { r } { - \frac { J } { 4 } \sum _ { \langle i j \rangle , a } L _ { i } ^ { a } L _ { j } ^ { a } } \end{array}$ , since $\hat { S } _ { i } ^ { a } \hat { S } _ { j } ^ { a } = c ^ { 2 } L _ { i } ^ { a } L _ { j } ^ { a } = - { \textstyle \frac { 1 } { 4 } } L _ { i } ^ { a } L _ { j } ^ { a }$ , and (S1.16) collapses to a diagonal channel sum,

$$
\hat { H } _ { X X X } \psi ( q ) \ = \ - \ \frac { \cal J } { 4 } { \sum _ { \langle i j \rangle } } \sum _ { a \in \{ x , y , z \} } \ \sum _ { \mu , \nu = 0 } ^ { 3 } { A } _ { \mu } ^ { a } ( q _ { i } ) \ : A _ { \nu } ^ { a } ( q _ { j } ) \ : \frac { \partial ^ { 2 } \psi } { \partial q _ { i , \mu } \partial q _ { j , \nu } } ( q ) ,\tag{S1.19}
$$

three mixed-site Hessian sandwiches per bond, one per Pauli channel. The Kitaev-Γ coupling $\hat { H } _ { \Gamma } =$ $\begin{array} { r l } { \sum _ { \langle i j \rangle } \Gamma _ { i j } \big ( \hat { S } _ { i } ^ { y } \hat { S } _ { j } ^ { z } + \hat { S } _ { i } ^ { z } \hat { S } _ { j } ^ { y } \big ) } & { { } } \end{array}$ is symmetric of-diagonal, $J _ { i j } ^ { y z } = J _ { i j } ^ { z y } = \Gamma _ { i j }$ , and pairs the two cross-channels,

$$
\hat { \cal H } _ { \Gamma } \psi ( q ) \ = \ - \frac { 1 } { 4 } \sum _ { \langle i j \rangle } \Gamma _ { i j } \sum _ { \mu , \nu = 0 } ^ { 3 } \big [ A _ { \mu } ^ { y } ( q _ { i } ) A _ { \nu } ^ { z } ( q _ { j } ) + A _ { \mu } ^ { z } ( q _ { i } ) A _ { \nu } ^ { y } ( q _ { j } ) \big ] \frac { \partial ^ { 2 } \psi } { \partial q _ { i , \mu } \partial q _ { j , \nu } } ( q ) .\tag{S1.20}
$$

The Dzyaloshinskii–Moriya coupling $\begin{array} { r } { \hat { H } _ { \mathrm { D M } } = \sum _ { \langle i j \rangle } \vec { D } _ { i j } \cdot ( \hat { \vec { S } } _ { i } \times \hat { \vec { S } } _ { j } ) } \end{array}$ is antisymmetric, $\begin{array} { r } { J _ { i j } ^ { b c } = \sum _ { a } D _ { i j } ^ { a } \varepsilon ^ { a b c } } \end{array}$ and collecting its three channels into a cross product (well defined because $L _ { i } ^ { a }$ and $L _ { j } ^ { b }$ commute for $i \neq j )$ gives

$$
\hat { H } _ { \mathrm { D M } } \psi ( q ) \ = \ - \frac 1 4 \sum _ { \langle i j \rangle } \sum _ { \mu , \nu = 0 } ^ { 3 } \vec { D } _ { i j } \cdot \left[ \vec { A } _ { \mu } ( q _ { i } ) \times \vec { A } _ { \nu } ( q _ { j } ) \right] \frac { \partial ^ { 2 } \psi } { \partial q _ { i , \mu } \partial q _ { j , \nu } } ( q ) ,\tag{S1.21}
$$

with $\vec { A } _ { \mu } ( q ) : = ( A _ { \mu } ^ { x } , A _ { \mu } ^ { y } , A _ { \mu } ^ { z } ) ( q )$ the frame sliced at fixed $\mu .$ Here the bracket is antisymmetric under the joint swap $( i , \dot { \mu } ) \stackrel { \cdot } {  } ( j , \nu )$ while the mixed Hessian is symmetric, so a Dzyaloshinskii–Moriya term contributes nothing when both legs sit on one site: there is no on-site DMI, exactly as the antisymmetry $J ^ { b c } = - J ^ { c b }$ demands.

## S1.3 Peter–Weyl decomposition and the spin-1/2 sector

Here we use the Peter–Weyl decomposition to identify the physical spin- $L ^ { 2 } ( \mathrm { S U } ( 2 ) ^ { N } )$ We will show that the multilinear manifold ansatz occupies a strictly larger ambient function class than a fixed lift of any NQS family, while the physical Hilbert space itself remains $2 ^ { \dot { N } }$ -dimensional. The comparison concerns ambient function classes; finite-parameter expressivity depends on the chosen architecture. We then establish the variational principle on the target spin- $\cdot \frac { 1 } { 2 }$ sector.

The Peter–Weyl theorem decomposes the single-site space into the matrix coeficients of the irreducible representations,

$$
L ^ { 2 } ( \mathrm { S U ( 2 ) } ) \cong \bigoplus _ { j = 0 , \frac { 1 } { 2 } , 1 , \frac { 3 } { 2 } , \ldots } V _ { j } ^ { * } \otimes V _ { j } .\tag{S1.22}
$$

Here $D _ { m n } ^ { j } ( q ) = \langle j , m | D ^ { j } ( q ) | j , n \rangle$ carries the row index m in the first, multiplicity factor $V _ { j } ^ { * }$ and the column index n in the second, physical factor $V _ { j }$ . The coeficients $\{ D _ { m n } ^ { j } \} _ { m , n = - j , . . . , j }$ form a complete orthogonal basis under the Haar measure, with $\| D _ { m n } ^ { j } \| _ { L ^ { 2 } } ^ { 2 } = 1 / ( 2 j + 1 )$ . The $\mathrm { s p i n } { - } \frac { 1 } { 2 }$ coeficients $D _ { m n } ^ { 1 / 2 }$ are the four entries of $g ( q )$ from $\ S \mathrm { S 1 . 1 }$ ; the higher-j coeficients are the degree-2j monomials in those entries.

On N sites the basis is the product

$$
\Phi _ { J , m , n } ( q _ { 1 } , \ldots , q _ { N } ) = \prod _ { k = 1 } ^ { N } { \cal D } _ { m _ { k } , n _ { k } } ^ { j _ { k } } ( q _ { k } ) ,\tag{S1.23}
$$

labelled by $\pmb { J } = ( j _ { 1 } , \dots , j _ { N } )$ , and the full space splits as

$$
L ^ { 2 } ( \operatorname { S U } ( 2 ) ^ { N } ) \cong \bigoplus _ { J } \bigotimes _ { k = 1 } ^ { N } ( V _ { j _ { k } } ^ { * } \otimes V _ { j _ { k } } \big ) .\tag{S1.24}
$$

with $J _ { k } \in \{ 0 , { \frac { 1 } { 2 } } , 1 , \dots \}$ , and any ψ has a unique expansion,

$$
\psi = \sum _ { J , m , n } c _ { J , m , n } \Phi _ { J , m , n } .\tag{S1.25}
$$

The per-site Casimir of § S1.2 acts diagonally on this basis. Each $D _ { m n } ^ { j }$ restricts to a degree-2j harmonic polynomial in the quaternion coordinates, on which $- \Delta _ { i }$ carries the eigenvalue $j ( j + 1 )$ 2

$$
- \Delta _ { i } D _ { m _ { i } , n _ { i } } ^ { j _ { i } } ( q _ { i } ) \ = \ j _ { i } ( j _ { i } + 1 ) D _ { m _ { i } , n _ { i } } ^ { j _ { i } } ( q _ { i } ) ,\tag{S1.26}
$$

and in particular $\begin{array} { r } { - \Delta _ { i } D _ { m n } ^ { 1 / 2 } = \frac { 3 } { 4 } D _ { m n } ^ { 1 / 2 } } \end{array}$ , the $\mathrm { s p i n } { - } \frac { 1 } { 2 }$ value $\begin{array} { r } { \hat { S } _ { i } ^ { 2 } = \frac { 3 } { 4 } } \end{array}$ recorded in $\ S \ S 1 . 2$

The consequence for the network is immediate. An unconstrained $\psi _ { \theta } \in L ^ { 2 } ( \mathrm { S U } ( 2 ) ^ { N } )$ generically has $c { \pmb J } , m , n \not = 0$ for every multi-index J in the expansion, integer and half-integer alike. It is not a spin- <sup>1</sup> wavefunction, nor even a single-spin-j wavefunction; its image lives across the full direct sum.

Indeed, the physical content occupies a single slice of that sum. A wavefunction ψ describes N genuine spin- <sup>1</sup> degrees of freedom, one per site, precisely when it is an eigenfunction of every per-site - <sup>1</sup>

$$
- \Delta _ { k } \psi = { \textstyle \frac { 3 } { 4 } } \psi \qquad \mathrm { f o r ~ e v e r y ~ s i t e } \ k = 1 , \dots , N .\tag{∗}
$$

Condition $( * )$ selects the Peter–Weyl sector $J = \left( { \scriptstyle { \frac { 1 } { 2 } } , \ldots , { \frac { 1 } { 2 } } } \right)$ , and projecting a generic manifold function onto it is what turns it into a physical state. This leads us to the following proposition.

Proposition S1.1 (The $\mathrm { s p i n } { - } \frac { 1 } { 2 }$ sector). A function $\psi \in L ^ { 2 } ( \mathrm { S U } ( 2 ) ^ { N } )$ satisfies (∗) if and only if

$$
\psi ( q _ { 1 } , \dots , q _ { N } ) = \sum _ { m , n } T _ { m , n } \prod _ { k = 1 } ^ { N } D _ { m _ { k } , n _ { k } } ^ { 1 / 2 } ( q _ { k } ) , \qquad T _ { m , n } \in \mathbb { C } .\tag{S1.27}
$$

The coeficient tensor $T \in \mathbb { C } ^ { 4 ^ { N } }$ parameterises the entire sector. Fixing the fiducial row $m _ { k } = 0$ at every site fixes the Peter–Weyl multiplicity and leaves the $2 ^ { N }$ coeficients indexed by $_ { n ; }$ these are the amplitudes of $\Psi \in ( \mathbb { C } ^ { 2 } ) ^ { \otimes N }$ under the identification of § S1.1.

The proof, a term-by-term application of the Casimir eigen-equation (S1.26), is given in $\mathrm { A p \mathrm { - } }$ pendix B.1.

To see how Prop. S1.1 works, it is instructive to write down some entangled states of interest to the quantum information community directly in the quaternionic coordinates of § S1.1.

The Bell state $\begin{array} { r } { | \Phi ^ { + } \rangle = \frac { 1 } { \sqrt { 2 } } ( | \uparrow \uparrow \rangle + | \downarrow \downarrow \rangle ) } \end{array}$ and singlet $\begin{array} { r } { | \Psi ^ { - } \rangle = \frac { 1 } { \sqrt { 2 } } ( | \uparrow \downarrow \rangle - | \downarrow \uparrow \rangle ) } \end{array}$ read

$$
\begin{array} { r l } & { \psi _ { \Phi ^ { + } } ( q _ { 1 } , q _ { 2 } ) = \frac { 1 } { \sqrt { 2 } } \Big [ ( q _ { 1 , 0 } + i q _ { 1 , 3 } ) ( q _ { 2 , 0 } + i q _ { 2 , 3 } ) + ( q _ { 1 , 2 } + i q _ { 1 , 1 } ) ( q _ { 2 , 2 } + i q _ { 2 , 1 } ) \Big ] , } \\ & { \psi _ { \Psi ^ { - } } ( q _ { 1 } , q _ { 2 } ) = \frac { 1 } { \sqrt { 2 } } \Big [ ( q _ { 1 , 0 } + i q _ { 1 , 3 } ) ( q _ { 2 , 2 } + i q _ { 2 , 1 } ) - ( q _ { 1 , 2 } + i q _ { 1 , 1 } ) ( q _ { 2 , 0 } + i q _ { 2 , 3 } ) \Big ] . } \end{array}
$$

The three-site GHZ and W states, which read,

$$
\begin{array} { r } { | \mathrm { G H Z } \rangle = \frac { 1 } { \sqrt { 2 } } ( | \uparrow \uparrow \uparrow \rangle + | \downarrow \downarrow \downarrow \rangle ) , ~ } \\ { | \mathrm { W } \rangle = \frac { 1 } { \sqrt { 3 } } ( | \uparrow \uparrow \downarrow \rangle + | \uparrow \downarrow \uparrow \rangle + | \downarrow \uparrow \uparrow \rangle ) , } \end{array}
$$

have manifold representatives,

$$
\begin{array} { r l } & { \psi _ { \mathrm { G H Z } } ( q _ { 1 } , q _ { 2 } , q _ { 3 } ) = \frac { 1 } { \sqrt { 2 } } \Big [ \prod _ { k = 1 } ^ { 3 } ( q _ { k , 0 } + i q _ { k , 3 } ) + \prod _ { k = 1 } ^ { 3 } ( q _ { k , 2 } + i q _ { k , 1 } ) \Big ] , } \\ & { \psi _ { \mathrm { W } } ( q _ { 1 } , q _ { 2 } , q _ { 3 } ) = \frac { 1 } { \sqrt { 3 } } \Big [ ( q _ { 1 , 0 } + i q _ { 1 , 3 } ) ( q _ { 2 , 0 } + i q _ { 2 , 3 } ) ( q _ { 3 , 2 } + i q _ { 3 , 1 } ) + ( q _ { 1 , 0 } + i q _ { 1 , 3 } ) ( q _ { 2 , 2 } + i q _ { 2 , 1 } ) ( q _ { 3 , 0 } + i q _ { 3 , 3 } ) } \\ & { \qquad + ( q _ { 1 , 2 } + i q _ { 1 , 1 } ) ( q _ { 2 , 0 } + i q _ { 2 , 3 } ) ( q _ { 3 , 0 } + i q _ { 3 , 3 } ) \Big ] . } \end{array}
$$

Each state is a linear combination of products with the single fiducial row $m _ { k } = 0$ , exactly the physical lift described above and the form allowed by Proposition S1.1.

We can now state the comparison with neural quantum states precisely. Let $G = { \mathrm { S U } } ( 2 )$ and define

$$
\begin{array} { r } { \mathcal { K } _ { N } : = ( { V } _ { 1 / 2 } ^ { * } ) ^ { \otimes N } , \qquad \mathcal { H } _ { N } : = { V } _ { 1 / 2 } ^ { \otimes N } , } \end{array}\tag{S1.28}
$$

where $\kappa _ { N }$ carries the row/multiplicity indices m and $\mathcal { H } _ { N }$ carries the column/physical indices n. The multilinear and per-site-odd function spaces are

$$
\begin{array} { r l } & { \mathcal { F } _ { \mathrm { l i n } } : = \mathrm { s p a n } \Bigg \{ \displaystyle \prod _ { i = 1 } ^ { N } D _ { m _ { i } n _ { i } } ^ { 1 / 2 } ( q _ { i } ) \Bigg \} \cong K _ { N } \otimes \mathcal { H } _ { N } , } \\ & { \mathcal { F } _ { \mathrm { o d d } } : = \big \{ f \in L ^ { 2 } ( G ^ { N } ) : f ( \dots , - q _ { i } , \dots ) = - f ( \dots , q _ { i } , \dots ) \ \forall i \big \} } \\ & { \qquad = \displaystyle \bigoplus _ { j _ { 1 } , \dots , j _ { N } \in \{ \frac 1 2 , \frac 3 2 , \dots \} } \bigotimes ( V _ { j _ { i } } ^ { * } \otimes V _ { j _ { i } } ) . } \end{array}\tag{S1.29}
$$

(S1.30)

Fix any unit vector $\chi \in { \mathcal { K } } _ { N }$ . Its canonical lift is

$$
\iota _ { \chi } ( \Psi ) ( q _ { 1 } , \dots , q _ { N } ) : = 2 ^ { N / 2 } \sum _ { m , n } \chi _ { m } \Psi _ { n } \prod _ { i = 1 } ^ { N } D _ { m _ { i } n _ { i } } ^ { 1 / 2 } ( q _ { i } ) .\tag{S1.31}
$$

Theorem S1.2 (Ambient expressivity hierarchy). For $N \geq 1$ , the $l i f t$ (S1.31) is an isometry and

$$
\iota _ { \chi } ( { \mathcal { H } } _ { N } ) \subsetneq { \mathcal { F } } _ { \mathrm { l i n } } \subsetneq { \mathcal { F } } _ { \mathrm { o d d } } .\tag{S1.32}
$$

Consequently, for every NQS variational family $\mathcal { M } _ { \mathrm { N Q S } } \subseteq \mathcal { H } _ { N }$ ，

$$
\iota _ { \chi } ( { \mathcal M } _ { \mathrm { N Q S } } ) \subseteq \iota _ { \chi } ( { \mathcal H } _ { N } ) \subsetneq { \mathcal F } _ { \mathrm { l i n } } \subsetneq { \mathcal F } _ { \mathrm { o d d } } .\tag{S1.33}
$$

Let $\widetilde { H }$ be the right-regular diferential representation of the physical Hamiltonian H, generated by the $l e f t .$ -invariant fields. On the linear class,

$$
\widetilde { \cal H } \big | _ { \mathcal { F } _ { \mathrm { l i n } } } \cong I _ { \mathcal { K } _ { N } } \otimes H _ { \mathcal { H } _ { N } } ,\tag{S1.34}
$$

where $H _ { \mathcal { H } _ { N } }$ denotes H acting on $\mathcal { H } _ { N }$ . Consequently, every nonzero $f \in \mathcal { F } _ { \mathrm { l i n } }$ obeys

$$
\frac { \langle f , \widetilde { H } f \rangle } { \langle f , f \rangle } \geq E _ { 0 } , \qquad \operatorname* { i n f } _ { f \in \mathcal { F } _ { \mathrm { l i n } } \setminus \{ 0 \} } \frac { \langle f , \widetilde { H } f \rangle } { \langle f , f \rangle } = E _ { 0 } .\tag{S1.35}
$$

On the nonlinear odd class, define

$$
\widehat { C } : = \sum _ { i = 1 } ^ { N } \left( \widehat { S } _ { i } ^ { 2 } - \frac { 3 } { 4 } I \right) .\tag{S1.36}
$$

Then $\widehat { C } \succeq 0$ on $\mathcal { F } _ { \mathrm { o d d } }$ and ker $\widehat { C } = \mathcal { F } _ { \mathrm { l i n } }$ . For every certified $\lambda \geq \mu _ { \star } ( J , h )$ from Theorem A.3, proved below,

$$
\frac { \langle f , ( \widetilde { H } + \lambda \widehat { C } ) f \rangle } { \langle f , f \rangle } \geq E _ { 0 } \quad f o r \ a l l \ f \in \mathcal { F } _ { \mathrm { o d d } } \setminus \{ 0 \} ,\tag{S1.37}
$$

and the infimum of the regularised quotient over $\mathcal { F } _ { \mathrm { o d d } } \backslash \{ 0 \}$ is exactly $E _ { 0 }$

The strict containments in Theorem S1.2 concern functions on the group manifold. The additional row Peter–Weyl index is a multiplicity, or purification, degree of freedom on which the physical Hamiltonian acts trivially; the column index carries the physical spin. This does not enlarge the physical $2 ^ { N }$ -dimensional Hilbert space. Likewise, the nonlinear class adds higher-half-integer sectors that become variationally consistent only after the Casimir penalty. The theorem proves the exact ambient hierarchy and the variational principle for both the linear and nonlinear cases.

## S1.4 Central oddness and variational principles

Notice first that the Peter–Weyl expansion splits into two parity classes under the per-site reflection $q _ { i } \to - q _ { i }$ . The point −q represents the central element $- I \in \mathrm { S U } ( 2 )$ , which acts on the spin-j irrep as the scalar $( - 1 ) ^ { \hat { 2 } j }$ , so the Wigner coeficients carry that same sign,

$$
D _ { m n } ^ { j } ( - q ) \ = \ ( - 1 ) ^ { 2 j } D _ { m n } ^ { j } ( q ) ,\tag{S1.38}
$$

even (+1) on the integer sectors and odd (−1) on the half-integer ones. The integer-j sectors are therefore exactly the even subspace of $L ^ { 2 } ( \mathrm { S U } ( 2 ) )$ , and the half-integer sectors the odd subspace.

Hence, per-site oddness,

$$
\psi ( \dots , - q _ { i } , \dots ) ~ = ~ - \psi ( \dots , q _ { i } , \dots ) ~ \mathrm { f o r ~ e v e r y ~ s i t e } ~ i ,\tag{S1.39}
$$

annihilates by (S1.38) every component with integer $j _ { i } ,$ , and the support collapses onto the half-integer sectors $j _ { i } \in \{ { \textstyle { \frac { 1 } { 2 } } } , { \textstyle { \frac { 3 } { 2 } } } , { \textstyle { \frac { 5 } { 2 } } } , \dots \}$ , with the physical target $\begin{array} { r } { j _ { i } = \frac { 1 } { 2 } } \end{array}$ now the lowest of them.

The architecture of SM § S2 enforces (S1.39) exactly, so we take it as given here. Although each normalised carrier is a nonlinear function of its quaternion, the leaf and merge log-scales cancel those normalisations exactly in the reconstructed amplitude. Lemma S2.1 therefore shows that the final $\psi _ { \theta }$ is degree one in every $q _ { i }$ . A degree-one function on SU(2) is a linear combination of the four coeficients $D _ { m n } ^ { 1 / 2 }$ , so Proposition S1.1 places the ansatz in $\mathcal { F } _ { \mathrm { l i n } }$ identically.

Two distinct cases follow as a consequence of Theorem S1.2. First, the foundation architecture we describe next is linear in each quaternion, so lies in $\mathcal { F } _ { \mathrm { l i n } }$ and directly obeys the variational prin ciple $E [ \psi _ { \theta } ] \geq E _ { 0 }$ . Second, any future generation of model which is per-site odd but non-linear in the quaternioncs lies in the larger $\mathcal { F } _ { \mathrm { o d d } }$ . Such a model’s higher-half-integeter sectors invalidate the variational principle as it stands, however it can be restored with an energy penalty restores the upper bound while retaining that larger ambient function class, see Appendix A.

## S2 Foundation Ansatz and Entanglement Structure via Reinforcement Learning

In this section, we describe the foundation model’s architecture. The model has two diferent types of input. First is the Hamiltonian’s interaction tensor $( J , h )$ , and second is a spin configuration $q \in$ $S U ( 2 ) ^ { N }$ sampled from the model’s own density. In this section, we will assume eficient access to such samples, deferring the details of our novel sampling scheme to SM § S4.1.

These two sources of data naturally divide the architecture into two components that together allow us to simultaneously condition the foundation model wavefunction on the Hamiltonian (allowing for generalisation) as well as preserving the necessary exchange anti-symmetry of our quaternionic representation from SM § S1.

The first component, which we call the trunk, is a function of the Hamiltonian’s interaction tensor. In (§ S2.1) we describe how a given input $( J , h )$ is cached into two tensors $J ^ { \prime \prime }$ and $h ^ { \prime \prime }$ . In (§ S2.2) we then describe our learnable featurizer, which converts the cached inputs $J ^ { \prime \prime }$ and $h ^ { \prime \prime }$ into embeddings of the Hamiltonian’s system context for the trunk. We then describe the architecture of the trunk itself in (§ S2.3), which consists of L residual attention blocks nested between feed-forward networks, as well as a per-site leaf builder where the spin configuration q enters (§ S2.3).

The second component is a readout contraction mechanism of the trunk that incorporates spin coordinates. This component is akin to a tree-tensor network, augmented by three routines that enable generalisation to arbitrary quadratic Hamiltonians and system sizes. First, we use edge-level attention in a tree to enable it to condition on the Hamiltonian’s interaction tensor. Second, we use a learnable routing policy to allow the tree to entangle arbitrary sites in the system. Finally, we use a contextualizer to allow the tree to condition on Hamiltonian data while preserving the necessary exchange anti-symmetry of our quaternionic representation.

The contraction itself is detailed in (§ S2.4), where we describe a rank-4 quadrilinear merge with level attention on each contraction layer, as well as the contextualizer. Section (§ S2.5) then describes the learnable routing policy to optimise over contraction paths with deep reinforcement learning.

Together, the merged contraction with a learnable path enables arbitrary entanglement between sites, with the entanglement conditioned on the Hamiltonian’s interaction tensor (J, h) enabling a Hamiltonian-dependent contraction pathway.

Figure S1 shows the information flow of our ansatz over a forward pass. Recall for the spinconfiguration input $q \in ( \mathrm { S U } ( 2 ) ) ^ { N }$ , we use the quaternion representation embedded in $\mathbb { R } ^ { 4 }$ , so $q _ { i } =$ $( q _ { i , 0 } , q _ { i , 1 } , q _ { i , 2 } , q _ { i , 3 } )$ with $q _ { i , 0 } ^ { 2 } + q _ { i , 1 } ^ { 2 } + q _ { i , 2 } ^ { 2 } + q _ { i , 3 } ^ { 2 } = 1$ . This makes the input space a product of spheres $\mathrm { S U } ( 2 ) \cong S ^ { 3 } \hookrightarrow \mathbb { R } ^ { 4 }$

## S2.1 Hamiltonian encoding and spectral normalisation

Recall from §S1.2 that the Hamiltonian family we target is

$$
\hat { \cal H } \ = \ \frac 1 4 \sum _ { i < j } \sum _ { a , k } J _ { i j } ^ { a k } \sigma _ { i } ^ { a } \sigma _ { j } ^ { k } \ + \ \frac 1 2 \sum _ { i } \sum _ { a } h _ { i } ^ { a } \sigma _ { i } ^ { a } ,\tag{S2.1}
$$

with exchange tensor $J \in \mathbb { R } ^ { N \times N \times 3 \times 3 }$ (cross-channel Pauli-pair couplings on of-diagonal $( i , j ) )$ and transverse field $\boldsymbol { h } \in \mathbb { R } ^ { N \times 3 }$ (per-site three-vector).

As stated in the main text, it is useful to think of J as the data of a weighted graph on N sites, where each unordered pair $( i , j )$ with $i \neq j$ carries a $3 \times 3$ Pauli-pair coupling matrix $J _ { i j } \in \mathbb { R } ^ { 3 \times 3 }$ , which we read as nine flavours of edge, one per Pauli-pair $( a , b ) \in \{ x , y , z \} ^ { 2 }$ giving the operator $\sigma _ { i } ^ { a } \sigma _ { j } ^ { b }$ , see Fig. 2.

We keep exchange and field as separate input channels for the model. Before feeding them into the model, we normalize them in the following way.

Step 1 (of-diagonal — bond-symmetrise). On Hermitian-conjugate pairs $( i , j )  ( j , i )$ , average to enforce the swap symmetry:

$$
J _ { i j , a b } ^ { \mathrm { s y m } } ~ = ~ \textstyle { \frac { 1 } { 2 } } \left( J _ { i j , a b } + J _ { j i , b a } ^ { \ast } \right) ~ ( i \ne j ) .\tag{S2.2}
$$

We do this because Pauli interactions are hermitian, so at the level of the interaction tensor and its relation to an interaction graph, it must be bi-directionally symmetric.

Step 2 (form the common Hermitian scale matrix). For the scale computation only, the cache builder forms

$$
M _ { i j , a b } = J _ { i j , a b } ^ { \mathrm { s y m } } + \delta _ { i j } i \varepsilon _ { a b c } h _ { i } ^ { c } ,\tag{S2.3}
$$

and reshapes it as

$$
\widetilde M : = \mathrm { r e s h a p e } ( M , 3 N \times 3 N ) .\tag{S2.4}
$$

For real $h _ { i } .$ , the block $A _ { i , a b } : = \varepsilon _ { a b c } h _ { i } ^ { c }$ is real and antisymmetric, so

$$
( i A _ { i } ) ^ { \dagger } = i A _ { i } .\tag{S2.5}
$$

![](images/633b7bba48eea0929c89998fc701871f9da30fcc31486aadd64414c2f4755735.jpg)  
Figure S1: Information flow through the foundation ansatz. The Hamiltonian $( J , h )$ is cached into the two tensors $J ^ { \prime \prime }$ and $h ^ { \prime \prime } \ ( \ S \mathrm { S 2 . 1 } )$ and consumed by the SystemFeaturizer (§ S2.2), which emits the per-bond embedding $e ^ { ( 0 ) }$ , the per-site local embedding $\ell ^ { ( 0 ) }$ and the seed $g ^ { ( 0 ) }$ of the global stream. The trunk (§ S2.3) then propagates all three streams through L residual attention blocks, refreshing $g$ once per block (Figure S3 traces the stream on its own). Notice the spin configuration $q$ does not enter the trunk, it is a function of the Hamiltonian interaction alone. After the trunk, the route policy (§ S2.5) samples a permutation $p \in S _ { \hat { N } }$ from $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { p } \mid h , \ell ^ { ( L ) } , e ^ { ( L ) } , m )$ , conditioned by its own fork of the global stream, and that permutation relabels both the trunk outputs $( \ell ^ { ( L ) } , e ^ { ( L ) } )$ and the spin configuration $q$ before downstream consumption. In particular before the contextualizer, whose slot clocks and tree-distance decays encode position and must therefore act in the routed frame. A twoblock leaf contextualizer (§ S2.4.3) then refines the routed trunk outputs over the padded slot frame, manufacturing contexts for the holes, and emits the per-slot states $\hat { \ell }$ together with the rewritten edges that seed the tree’s level-0 edge field $E ^ { ( 0 ) }$ (the latter not drawn). The per-site leaf builder (§ S2.3.1) is the unique entry point for $q$ and consumes $( \hat { \ell } _ { i } , g , q _ { i } )$ per site, with $g$ the stream’s refreshed postcontext value (it does not see the per-bond edges). Its output $u _ { i } ^ { ( 0 ) }$ feeds the tree readout (§ S2.4), which contracts the $\hat { N }$ leaf carriers down to a single root carrier; the per-bond $e ^ { ( L ) }$ re-enters the tree internally at the level-edge attention in (§ S2.4.4) (not drawn here). The root readout is the routed log-amplitude log $\psi _ { \theta } ( q \mid p ) \in \mathbb { C }$

Together with the Hermitian swap symmetrisation of Step 1, this makes $\widetilde { M }$ Hermitian. Its eigenvalues are therefore real, and the shared scale is

$$
s : = \| \widetilde { M } \| _ { 2 } = \operatorname* { m a x } _ { k } \bigl | \lambda _ { k } ( \widetilde { M } ) \bigr | ,\tag{S2.6}
$$

computed once per system with jnp.linalg.eigvalsh. The spectral radius max $_ { \left| \cdot k \right| } \lambda _ { k } \mathinner { | { } | }$ is defined for every square matrix. The role of the factor i is instead to make $\widetilde { M }$ Hermitian, so that the Hermitian eigensolver is appropriate and the spectral radius equals the spectral norm. The diagonal field block

participates only in this scale computation and is discarded afterwards.

Step 3 (normalise and flatten). We divide the separate exchange and field channels by the same scale s. The exchange keeps its of-diagonal bonds and an identity-marker channel, with the diagonal Pauli block left empty since the field no longer lives there,

$$
J _ { i j , c } ^ { \prime \prime } = \left\{ \begin{array} { l l } { ( J ^ { \mathrm { s y m } } / s ) _ { i j , a b ( c ) } } & { c = 0 , \ldots , 8 \mathrm { ~ a n d ~ } i \ne j , } \\ { 0 } & { c = 0 , \ldots , 8 \mathrm { ~ a n d ~ } i = j , } \\ { \delta _ { i j } } & { c = 9 , } \end{array} \right.\tag{S2.7}
$$

where $c = 0 , \ldots , 8$ enumerates the nine $( a , b )$ pairs in row-major order, and the field is cached as its own per-site tensor,

$$
h _ { i } ^ { \prime \prime } \ = \ h _ { i } / s .\tag{S2.8}
$$

Building these inputs costs one call to eigvalsh per system, paid once per system when a panel or an augmentation round is loaded, and once per system at evaluation or fine-tuning. We emphasise here that the size of the interaction tensor, and the graph that follows, is polynomial in the number of bodies, since our foundation model handles quadratic interactions which are a small sub-group of the overall Pauli group on N spins.

By construction, this normalisation makes both cached inputs invariant under simultaneous rescaling $( J , h ) \to \alpha ( J , h )$ for $\alpha > 0 .$ , since the spectral radius s scales linearly with $\alpha .$ . This avoids the attention layers needing to learn the invariance from our datasets. Under $\alpha < 0$ the inputs are signflip-equivariant: a global sign passes through to the nine Pauli channels of $J ^ { \prime \prime }$ and to $h ^ { \prime \prime }$ , while the identity channel is unafected. Finally they are smooth at $h = 0$ , since $h ^ { \prime \prime }$ is linear in h and the scale s stays bounded away from zero. These three properties are exactly the ones our featurizer (§ S2.2) relies on for equivariance under rescaling and for the $h  0$ limit.

This encoding is a bijection between the original $( J , h )$ and the cached pair $( J ^ { \prime \prime } , h ^ { \prime \prime } )$ , and both representations are in the datasets we have open-sourced.

The diagonal field block in M is solely a common-normalisation device. It is neither a cached model input nor a replacement of the physical first-order field operator. Including both exchange and field in Mf lets one scale normalise the two channels together. Once s has been computed, the diagonal block is discarded, and the model receives the bond tensor $J ^ { \prime \prime }$ and the per-site field $h ^ { \prime \prime }$ separately, while the local-energy expression retains the field as the first-order term in Eq. (S1.16).

The per-system rescaling-invariance is then the standard ML input-normalisation trick applied at the level of the Hamiltonian itself. Dividing through by the spectral radius lands every system in a unit-norm regime regardless of physical units, so the training dataset’s overall energy scale stops being a feature the network has to re-learn for every batch. Indeed this has direct analogy to dividing pixel values by 255 or attention keys by ${ \sqrt { d } } ,$ removing a known-uninformative scale so the network spends capacity on what actually distinguishes systems. We can simply rescale the output by the renormalisation factor s to get back to physical units, and optimise in the normalised space where the network is more stable and generalises better.

## S2.2 Learnable featurizer

The featurizer is a single fusion block that consumes the two cached inputs and emits three downstreamready tensors:

$$
{ \mathrm { S y s t e m F e a t u r i z e r : ~ } } \left( J ^ { \prime \prime } \in { \mathbb R } ^ { N \times N \times 1 0 } , \ h ^ { \prime \prime } \in { \mathbb R } ^ { N \times 3 } \right) \ \longmapsto \ \left( e \in { \mathbb R } ^ { N \times N \times 9 6 } , \ \ell \in { \mathbb R } ^ { N \times 1 2 8 } , \ g \in { \mathbb R } ^ { 1 0 2 4 } \right) .\tag{S2.9}
$$

These three objects correspond to a per-bond embedding e for the trunk’s attention layers, a per-site local embedding ℓ for the trunk’s attention layers, and a global context vector $g$ that seeds the global stream the trunk maintains (§ S2.3). The featurizer is a single-pass architecture with no iteration or recurrence; it is a single function of the Hamiltonian that produces these three tensors in one go. Figure S2 shows the data-flow and we detail each step below.

Both cached tensors enter in polar form. We read the nine Pauli-pair entries of each bond block as a vector $x _ { i j } ^ { J } = \mathrm { v e c } ( J _ { i j , 1 : 9 } ^ { \prime \prime } ) \in \mathbb { R } ^ { 9 }$ and split it into a radius and a softened direction,

$$
r _ { i j } ^ { J } = \| x _ { i j } ^ { J } \| _ { 2 } , \qquad u _ { i j } ^ { J } = \frac { x _ { i j } ^ { J } } { \sqrt { \| x _ { i j } ^ { J } \| _ { 2 } ^ { 2 } + \tau ^ { 2 } } } ,\tag{S2.10}
$$

![](images/125b8a9d193df6e8a9d7025a41edf5edba44559539707fdaef1c7ea4053ef92e.jpg)  
Figure S2: Data flow inside the system featurizer (§S2.2). The two cached inputs enter on separate spines, each first mapped to its polar form (raw, radial, and angular coordinates): the exchange $J ^ { \prime \prime }$ is embedded per bond (B) by a grouped-RMS map and aggregated by column attention into the bond descriptor $\ell ^ { \mathrm { { r a w } } }$ , while the field $h ^ { \prime \prime }$ is lifted into the Zeeman base $\ell ^ { \mathbf { z } }$ by the same grouped-RMS pattern. Two row-attention layers pool the two views $( g ^ { \mathrm { r a w } } , g ^ { \mathrm { z } } )$ , which a fusion map, fed also by the spectrum block $\phi _ { \mathrm { s p e c } }$ and the relative-scale channels, combines into the global vector $^ { g , }$ the seed of the global stream. Solid arrows are the main pipeline; dashed magenta arrows are skip-connections: the Zeeman base $\ell ^ { \mathbf { z } }$ is the residual backbone of the local feature $\ell ,$ conditioned by $\ell ^ { \mathrm { { r a w } } }$ and $^ { g , }$ and the edges are conditioned by B and, through a global FiLM, by $g .$ The outputs $( e , \ell , g )$ are the three tensors consumed by the trunk.

where $\tau = 1 0 ^ { - 3 }$ is a fixed feature-scale floor, and $| | \cdot | | _ { 2 }$ is the Euclidean norm. The radius $r ^ { J }$ shows whether a bond is present, whilst the direction $u ^ { J }$ reports its angular type and stays bounded as the bond vanishes. Notice $\tau$ is a fixed feature-scale floor, not a numerical ε. The bond input stacks the raw, radial, angular, and self-bond coordinates, and the field is lifted the same way,

$$
\begin{array} { r } { \hat { x } _ { i j } ^ { J } = \left[ x _ { i j } ^ { J } , ~ r _ { i j } ^ { J } , ~ u _ { i j } ^ { J } , ~ \mathbf { 1 } _ { i = j } \right] \in \mathbb { R } ^ { 2 0 } , \qquad \hat { x } _ { i } ^ { h } = \left[ h _ { i } ^ { \prime \prime } , ~ \lVert h _ { i } ^ { \prime \prime } \rVert _ { 2 } , ~ \frac { h _ { i } ^ { \prime \prime } } { \sqrt { \lVert h _ { i } ^ { \prime \prime } \rVert _ { 2 } ^ { 2 } + \tau ^ { 2 } } } \right] \in \mathbb { R } ^ { 7 } . } \end{array}\tag{S2.11}
$$

They are functions of the cached inputs alone and are computed once per system alongside them $( \ S \ S 2 . 1 )$ . We hand the model the coordinate, its magnitude, and its direction separately rather than make a dense layer recover all three: the loci that matter physically, vanishing bonds or fields, isotropic blocks, vanishing skew or symmetric-traceless parts, rank-deficient couplings, are exactly where a plain projection smears presence into direction.

The two embedding paths normalise in groups, and the normalisation is the root-mean-square kind

throughout: for a vector z with learned per-channel scale $\gamma .$

$$
\begin{array} { r } { \mathrm { R M S } ( z ) = \gamma \odot \frac { z } { \sqrt { z ^ { 2 } } + \varepsilon } , \qquad \overline { { z ^ { 2 } } } : = \frac { 1 } { d } \sum _ { a } z _ { a } ^ { 2 } , } \end{array}\tag{S2.12}
$$

with no mean subtraction and no shift. The mean subtraction of LayerNorm is left out on purpose: it couples every channel through a shared ofset and worsens the conditioning of the curvature the optimiser preconditions against, and dropping it is the same trade that moved the large-languagemodel community from LayerNorm to RMSNorm [121]. Every normalisation layer on the even path, here and in every later stage, is of this form. Reshaping a vector into K groups of width $d _ { \mathrm { g r p } } ,$ , the grouped variant normalises each group on its own with its own scale,

$$
\mathrm { G R M S } ( z ) _ { k } = \gamma _ { k } \frac { z _ { k } } { \sqrt { \overline { { z _ { k } ^ { 2 } } } + \varepsilon } } , \qquad \gamma _ { k } = 1 \mathrm { a t } \mathrm { i n i t } ,\tag{S2.13}
$$

with $\overline { { z _ { k } ^ { 2 } } }$ the mean square within group k. Each group is then free to become a learned filter over the raw, radial, and angular coordinates, a data-driven stand-in for the named decomposition $J = t I + Q + A$ into isotropic, symmetric-traceless, and skew parts, without hard-coding it. We use $K = 1 6$ groups of width $d _ { \mathrm { g r p } } = 1 6$ on both paths.

The per-bond embedding is then a two-layer map, a grouped norm, and two more layers,

$$
\begin{array} { r } { z _ { i j } ^ { J } = \mathrm { G R M S } \big ( \mathrm { S i L U } ( \hat { x } _ { i j } ^ { J } W _ { 1 } ^ { J } ) W _ { 2 } ^ { J } \big ) , \qquad B _ { i j } = m _ { i } m _ { j } \cdot \mathrm { S i L U } \big ( \mathrm { S i L U } ( z _ { i j } ^ { J } W _ { 3 } ^ { J } ) W _ { 4 } ^ { J } \big ) , } \end{array}\tag{S2.14}
$$

with $d _ { \mathrm { b o n d } } = 1 2 8$ , every bias zero at initialisation, and $m _ { i } m _ { j }$ the site-padding pair mask: it zeroes pairs with a padded endpoint and carries no bond information.

Bonds absent from the system’s adjacency pattern are not masked to zero. Wherever a bond feature is consumed, absence substitutes a learned token in its place,

$$
z _ { i j } \ \longleftarrow \ \{ { z _ { i j } , } \mathrm { { p r e s e n t } } _ { i j } \mathrm { { } } \qquad \mathrm { { p r e s e n t } } _ { i j } \ = \ [ \exists c \leq 8 : J _ { i j , c } ^ { \prime \prime } \neq 0 ] \lor \ [ i = j ] { } , \\tag{S2.15}
$$

with one trained token t per consumption site. Six sites carry a token: the bond embedding, the bond attention’s key and value input, the Zeeman base, the field’s row-attention view, the field branch of the global fusion, and the field’s entry in the local cross-conditioning. On the field path a zero field substitutes its tokens the same way, per site everywhere except the fusion branch, which gates per system. The physical reading is that “no bond” becomes a representable direction of the basis rather than the numeric zero, while the substitution still cuts the degenerate-input gradients that a hard zero would poison. The site-padding masks $m _ { i }$ and $M _ { j } ^ { \mathrm { k e y } }$ are a diferent object: they mark sites absent from the padded system, not bonds absent from a present system’s graph. Nothing downstream carries a bond mask; bond presence reaches the trunk only through these tokens.

Column attention then attends over the $j$ index for each (i, h) using per-row learnable queries $Q _ { \mathsf { h } } ^ { \mathrm { c o l } } \in \mathbb { R } ^ { d _ { \mathsf { h } } }$ ,

$$
\begin{array} { r } { K _ { i j , \mathrm { h } } = W _ { K } \mathrm { R M S } ( B _ { i j } ) , \quad V _ { i j , \mathrm { h } } = W _ { V } \mathrm { R M S } ( B _ { i j } ) , \quad \alpha _ { i j , \mathrm { h } } ^ { \mathrm { c o l } } = \mathrm { s o f t m a x } _ { j } \Big ( \frac { Q _ { \mathrm { h } } ^ { \mathrm { c o l } } \cdot K _ { i j , \mathrm { h } } } { \sqrt { d _ { \mathrm { h } } } } + M _ { j } ^ { \mathrm { k e p } } \Big ) , } \end{array}\tag{S2.16}
$$

where h $\in \{ 1 , \ldots , n _ { \mathsf { h } } \}$ is the attention-head index (we use sans-serif to disinguish from the transverse field $h ) , d _ { \mathsf { h } }$ is the per-head width with $n _ { \mathsf { h } } \cdot d _ { \mathsf { h } } = d _ { \mathsf { b o n d } }$ , and $W _ { K } , W _ { V } \in \mathbb { R } ^ { d _ { \mathrm { b o n d } } \times d _ { \mathrm { b o n d } } }$ are linear projectors whose outputs split into $n _ { \mathrm { h } }$ head blocks of width $d _ { \mathsf { h } }$ indexed by h. Here RMS is the map of Eq. (S2.12), applied along the bond-feature axis, softmax normalises over the second site index, and $M _ { j } ^ { \mathrm { k e j } }$ is the key-padding mask which is 0 when site $j$ is present in the system, and −∞ when it is absent. The kernel computes twice the output head count and gates the heads in bisected pairs,

$$
o _ { i , \mathfrak { h } } = \sum _ { j } \alpha _ { i j , \mathfrak { h } } ^ { \mathrm { c o l } } V _ { i j , \mathfrak { h } } , \qquad \ell _ { i } ^ { \mathrm { r a w } } = m _ { i } \cdot \Bigl [ \mathrm { s i g m o i d } \bigl ( o _ { i } ^ { ( 1 : n _ { \mathfrak { h } } ) } \bigr ) \odot o _ { i } ^ { ( n _ { \mathfrak { h } } + 1 : 2 n _ { \mathfrak { h } } ) } \Bigr ] ,\tag{S2.17}
$$

a GLU-style multiplicative gate at the head level [122, 123]. Every attention layer of SM $\ S \ S 2$ shares this micro-structure. The descriptor $\ell _ { i } ^ { \mathrm { { r a w } } }$ carries the exchange structure seen from site i.

The field enters on its own per-site channel. Because each site carries its own $h _ { i } ^ { \prime \prime } .$ , the field is a genuinely local quantity, and we lift its polar input $\hat { x } _ { i } ^ { h }$ into a Zeeman base through the same two-layer / grouped-norm / two-layer pattern,

$$
z _ { i } ^ { h } = \mathrm { G R M S } \big ( \mathrm { S i L U } ( \hat { x } _ { i } ^ { h } W _ { 1 } ^ { h } ) W _ { 2 } ^ { h } \big ) , \qquad \ell _ { i } ^ { z } = m _ { i } \cdot \mathrm { S i L U } ( z _ { i } ^ { h } W _ { 3 } ^ { h } ) W _ { 4 } ^ { h } ,\tag{S2.18}
$$

with $d _ { \ell } = 1 2 8$ and, again, every bias zero at initialisation. The field groups learn their own thresholded detectors — field nearly zero, field isotropic, a single dominant axis — in the same way the bond groups do.

The global context pools both views of the system. A first row-attention layer with $n _ { q }$ learnable global queries $Q _ { \boldsymbol { q } , \boldsymbol { h } } ^ { \mathrm { r o w } } \in \bar { \mathbb { R } } ^ { n _ { \boldsymbol { q } } \times n _ { \boldsymbol { \mathsf { h } } } \times d _ { \boldsymbol { \mathsf { h } } } }$ attends over the bond descriptors $\ell ^ { \mathrm { { r a w } } }$ to give $g ^ { \mathrm { r a w } }$ , and a second, structurally identical layer attends over the Zeeman bases $\ell ^ { \mathbf { z } }$ to give $g ^ { \mathrm { z } }$

Meanwhile the relative weight is already implicit in $( J ^ { \prime \prime } , h ^ { \prime \prime } )$ , and $\phi _ { J h }$ surfaces it as an explicit bounded feature:

$$
\phi _ { J h } ~ = ~ \left[ \log ( 1 { + } r _ { J } ) , ~ \log ( 1 { + } r _ { h } ) \right] .\tag{S2.19}
$$

The branch is guarded so its gradient signal remains clean on the $h = 0$ majority of the panel. The fusion normalises each pooled view on its own, takes the two derived blocks raw (both are bounded by construction), and closes with an output normalisation,

$$
g \ = \ \mathrm { R M S } \Big ( \mathrm { F F N } _ { g } \big ( \big [ \mathrm { R M S } ( g ^ { \mathrm { r a w } } ) , \mathrm { R M S } ( g ^ { \mathrm { z } } ) , \phi _ { J h } \big ] \big ) \Big ) .\tag{S2.20}
$$

On a system that carries no field at all, $\mathrm { R M S } ( g ^ { \mathrm { z } } )$ is replaced by the fusion’s own absent-field token. The per-site local feature then takes the Zeeman base as its backbone and folds in the bond descriptor and the global context through a pre-norm residual map, each input normalised on its own,

$$
\ell _ { i } \ = \ m _ { i } \cdot \mathrm { R M S } \Big ( \ell _ { i } ^ { \mathrm { z } } + \mathrm { F F N } _ { \ell } \big ( \big [ \mathrm { R M S } ( \ell _ { i } ^ { \mathrm { z } } ) , \mathrm { R M S } ( \ell _ { i } ^ { \mathrm { r a w } } ) , \mathrm { R M S } ( g ) \big ] \big ) \Big ) ,\tag{S2.21}
$$

and at a zero-field site the first entry is this map’s own absent-field token. The field is the residual base because it is the dominant per-site signal, while the exchange reaches a site only by aggregation over its bonds. Edge cross-conditioning then refines each bond with the new local features, and the global vector enters only as a gentle modulation. The core mixes the bond with its two endpoints, while a FiLM scale and shift read of the global context,

$$
c _ { i j } = \mathrm { F F N } _ { e } \big ( \mathrm { R M S } \big [ B _ { i j } , \ell _ { i } , \ell _ { j } \big ] \big ) , \qquad \big [ \gamma , \beta \big ] = W _ { F } \mathrm { R M S } ( g ) + b _ { F } , \quad \gamma  0 . 1 \operatorname { t a n h } \gamma , \ \beta  0 . 1 \beta ,\tag{S2.22}
$$

$$
e _ { i j } \ = \ m _ { i } m _ { j } \cdot \operatorname { R M S } \bigl ( P B _ { i j } + ( 1 + \gamma ) c _ { i j } + \beta \bigr ) .\tag{S2.23}
$$

The FiLM factors start near zero, P is a bias-free residual projector applied when $d _ { \mathrm { b o n d } } \ne d _ { \mathrm { e d g e } }$ , and the pair mask is, again, padding alone.

Given the graph structure of the Hamiltonian input, we note here that the featurizer is a GNN in the sense that it performs neighbor aggregation and global pooling. There is, however, no separate message-passing module since attention subsumes the GNN role, framed in a form compatible with our GPU kernel pipeline and natural-gradient optimisers see SM § S4 for details of our engineering.

## S2.3 Trunk and per-site leaf builder

Recall that the featurizer of § S2.2 emits three tensors: the per-bond embedding $e \in \mathbb { R } ^ { N \times N \times d _ { e } }$ , the per-site local feature $\ell \in \mathbb { R } ^ { \hat { N } \times d _ { \ell } }$ , and the global context $\textit { g } \in \mathbb { R } ^ { d _ { g } }$ . Both e and ℓ depend on the Hamiltonian $( J , h )$ only through the cached inputs $J ^ { \prime \prime }$ and $h ^ { \prime \prime }$ of § S2.1, and every system-context quantity downstream of this point is, by construction, a function of $( J ^ { \prime \prime } , h ^ { \prime \prime } )$ and therefore a function of $( J , h )$ . The trunk’s job is to propagate $( e , \ell , g )$ through L residual blocks: it maintains all three streams, mixing information across sites and bonds and keeping the global summary current as it does so. We emphasise that this is completely independent functions of spin configuration $q ,$ which enter only once, at a per-site leaf builder that sits between the trunk and the merge tree. As such the per-site antisymmetry constraint of § S1.4,

$$
\psi ( \dots , - q _ { i } , \dots ) ~ = ~ - \psi ( \dots , q _ { i } , \dots ) ~ \mathrm { f o r ~ e v e r y ~ s i t e } ~ i ,\tag{S2.24}
$$

becomes a property of the leaf builder alone, and the expressivity of the trunk is not limited by any symmetry requirement and thus we may use any standard transformer recipe [124].

The trunk is structured by blocks, indexed by $\beta \in \{ 0 , 1 , \ldots , L - 1 \}$ . Let $\overline { { \ell ^ { ( \beta ) } } } \in \mathbb { R } ^ { N \times d _ { \ell } }$ and $e ^ { ( \beta ) } \in \mathbb { R } ^ { N \times N \times d _ { \epsilon } }$ be the per-site and per-bond states at block $\beta ,$ with $\ell ^ { ( 0 ) } = \ell$ and $e ^ { ( 0 ) } = e ^ { ( 0 ) }$ from the featurizer. We write $\ell _ { i } \in \mathbb { R } ^ { d _ { \ell } }$ for the row of $\ell ^ { ( \beta ) }$ at site i and $e _ { i j } \in \mathbb { R } ^ { d _ { e } }$ for the bond slot at $( i , j )$ dropping the (β) superscript when the block is unambiguous.

The global context is the trunk’s third stream, and it is a maintained one. The natural alternative is to broadcast the featurizer’s $g$ unchanged, as a fixed conditioning signal for every block. But a fixed global summarises the couplings as the featurizer saw them, and that summary grows stale exactly as the trunk’s refinement makes the site and bond states worth summarising. The trunk therefore refreshes $g$ once per block, from the states the block has just updated. The stream lives on the sphere of unit root-mean-square norm: writing

$$
\operatorname { N o r m } ( x ) : = { \frac { x } { \sqrt { \operatorname* { m a x } ( { \overline { { x ^ { 2 } } } } , \varepsilon ) } } } , \qquad { \overline { { x ^ { 2 } } } } : = { \frac { 1 } { d } } \sum _ { a } x _ { a } ^ { 2 } ,\tag{S2.25}
$$

for the parameter-free projection $\mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ onto that sphere (the floor $\varepsilon = 1 0 ^ { - 4 }$ replaces the usual added constant, so the map is exact whenever ${ \overline { { x ^ { 2 } } } } \geq \varepsilon$ and sends the origin to itself), the trunk receives the featurizer’s global as $g ^ { ( 0 ) } = \mathrm { N o r m } ( g )$ and returns every update to the same sphere. Norm carries no parameters; the RMS layers of $\ S \ S 2 . 2$ normalise the same way but carry a learned per-channel scale, and the two should not be conflated.

A block applies three residual sublayers in sequence: an edge-update sublayer on the per-bond stream, then an edge-biased multi-head self-attention (MHA) sublayer and a feed-forward network (FFN) sublayer on the per-site stream; a refresh of the global closes the block. The edge stream moves first,

$$
\begin{array} { r } { e ^ { ( \beta + 1 ) } \ = \ e ^ { ( \beta ) } \ + \ \frac { 1 } { \sqrt { L } } \operatorname { E d g e U p d a t e } \big ( \operatorname { R M S } ( e ^ { ( \beta ) } ) , \ \ell ^ { ( \beta ) } \big ) , } \end{array}\tag{S2.26}
$$

so that the attention which follows reads fresh bonds: the just-updated $e ^ { ( \beta + 1 ) }$ enters the attention logits as a per-head additive bias, defined below. The per-site stream then takes two residual steps,

$$
\begin{array} { r } { \ell ^ { \prime } = \ell ^ { ( \beta ) } + \frac { 1 } { \sqrt { L } } \mathrm { M H A } \big ( \mathrm { R M S } ( \ell ^ { ( \beta ) } ) ; e ^ { ( \beta + 1 ) } \big ) , } \end{array}\tag{S2.27}
$$

$$
\begin{array} { r } { \ell ^ { ( \beta + 1 ) } = \ell ^ { \prime } + \frac { 1 } { \sqrt { L } } \mathrm { F F N } \big ( \mathrm { R M S } ( \ell ^ { \prime } ) + W _ { g } g ^ { ( \beta ) } \big ) , } \end{array}\tag{S2.28}
$$

and the global closes the block by reading the sites the block has just written,

$$
g ^ { ( \beta + 1 ) } = \mathrm { U p d a t e } \big ( g ^ { ( \beta ) } , \mathrm { p o o l } \big ( g ^ { ( \beta ) } , \ell ^ { ( \beta + 1 ) } \big ) \big ) ,\tag{S2.29}
$$

with pool and Update defined below. In Eq. (S2.27) the semicolon indicates that the map is conditioned on the argument after it; RMS is the normalisation of § S2.2, applied along the feature axis; and $W _ { g } \in \mathbb { R } ^ { d _ { \ell } \times d _ { g } }$ is a learned projection whose output is broadcast to every site. The global enters the per-site path exactly once, additively, at the FFN’s input. We emphasise that the two per-site steps are sequential residuals (the FFN acts on the attention’s output $\ell ^ { \prime } .$ not on $\ell ^ { ( \beta ) } )$ and that all three sublayers are pre-norm: each reads a normalised copy of its stream and adds its result to the raw stream. The factor $1 / \sqrt { L }$ applies inverse-square-root depth scaling to each residual branch, matching the depth dependence induced by batch normalisation at initialisation [125]. Under the usual independence approximation at initialisation, each additive update contributes variance $\mathcal { O } ( 1 / L )$ to the running stream, so the cumulative variance across L blocks stays $\mathcal { O } ( 1 )$

The two maps that refresh the global recur at every later stage of the architecture; the trunk’s instance is the first of many. The pool is a descriptor attention: its query is produced from the current global state and determines the attention weights over the input set. For a set $\{ x _ { i } \} _ { i = 1 } ^ { n }$ of states $x _ { i } \in \mathbb { R } ^ { d _ { x } }$ and the current global $g ,$

$$
\begin{array} { r } { q _ { \mathrm { \scriptsize { h } } } = W _ { \mathrm { \scriptsize { h } } } ^ { Q } g , \qquad k _ { \mathrm { \scriptsize { h } } , i } = W _ { \mathrm { \scriptsize { h } } } ^ { K } \mathrm { R M S } ( x _ { i } ) , \qquad v _ { \mathrm { \scriptsize { h } } , i } = W _ { \mathrm { \scriptsize { h } } } ^ { V } \mathrm { R M S } ( x _ { i } ) , } \end{array}\tag{S2.30}
$$

$$
\begin{array} { r } { a _ { \mathbf { h } , i } = \mathrm { s o f t m a x } _ { i } \Big ( \frac { q _ { \mathrm { h } } \cdot k _ { \mathrm { h } , i } } { \sqrt { d _ { k } } } + M _ { i } \Big ) , \qquad \mathrm { p o o l } \big ( g , \{ x _ { i } \} \big ) = \Big [ \sum _ { i } a _ { \mathbf { h } , i } v _ { \mathbf { h } , i } \Big ] _ { \mathrm { h = 1 } } ^ { n _ { \mathrm { h } } } \ \in \ \mathbb { R } ^ { n _ { \mathrm { h } } d _ { v } } , } \end{array}\tag{S2.31}
$$

where h is the sans-serif head index of § S2.2, $W _ { \mathsf { h } } ^ { Q } \in \mathbb { R } ^ { d _ { k } \times d _ { g } }$ $W _ { \mathsf { h } } ^ { K } \in \mathbb { R } ^ { d _ { k } \times d _ { x } }$ and $W _ { \mathsf { h } } ^ { V } \in \mathbb { R } ^ { d _ { v } \times d _ { x } }$ are per-head projectors, $[ \cdot ] _ { \mathsf { h } }$ concatenates the head outputs, and $M _ { i }$ is the key-padding mask of § S2.2 (0 where the element is present, −∞ where it is absent). This is the featurizer’s row attention with two changes: the learnable queries are replaced by projections of $^ { g , }$ and the head outputs concatenate plainly, without the gate of Eq. (S2.17). We use ${ n } _ { \mathrm { h } } = 4$ heads of width $d _ { k } = d _ { v } = 6 4$ , so the pooled reading has width $n _ { \mathrm { h } } d _ { v } = 2 5 6$ , a quarter of the stream’s own width $d _ { g } = 1 0 2 4$

The Hamiltonian’s data streams use the normalised interpolation of nGPT [126]. For a current state x and candidate $\Delta _ { x } .$ , let $a _ { x }$ be a learned per-channel gate, write 1 for the all-ones vector and ⊙ for elementwise multiplication, and define $\mathrm { l o g i t } ( r ) : = \log ( r / ( 1 - r ) )$ . Then

$$
\tilde { \alpha } _ { x } : = 0 . 8 \ \mathrm { s i g m o i d } ( a _ { x } ) , \qquad a _ { x } \big | _ { \mathrm { i n i t } } = \log \mathrm { i t } \bigg ( \frac { 0 . 2 } { 0 . 8 } \bigg ) \ 1 , \qquad \mathcal { N } _ { x } ( x , \Delta _ { x } ) : = \mathrm { N o r m } \Big ( x + \tilde { \alpha } _ { x } \odot \big ( \mathrm { N o r m } ( \Delta _ { x } ) - x \big ) \Big ) .\tag{S2.32}
$$

Thus every channel starts with interpolation weight 0.2 and remains in the open interval (0, 0.8). Throughout the architecture, $\mathcal { N } _ { \bullet }$ denotes this same normalised-interpolation map with an independently learned gate for the named stream. Given a pooled reading $p ,$ the global update reads,

$$
\Delta _ { g } = \mathrm { F F N } _ { g } \big ( \mathrm { R M S } [ W _ { \mathrm { t a p } } g , p ] \big ) , \qquad \mathrm { U p d a t e } ( g , p ) = \mathcal { N } _ { g } ( g , \Delta _ { g } ) .\tag{S2.33}
$$

The bottleneck $W _ { \mathrm { t a p } }$ balances the global and pooled entries before the joint RMS normalisation.   
Figure S3 traces the stream’s stations through the architecture.

![](images/1af4bd61b83fd303ef1ae395683303af21ce12fb2c5d72aaa9ba2d874628ff70.jpg)

Figure S3: The global stream. From the featurizer’s seed onward, $g$ lives on the unit-RMS sphere, and every station applies the same two maps: a descriptor pool (Eq. (S2.31)), then the normalised interpolation of Eq. (S2.33). The trunk refreshes g once per block from the updated site states and closes with the factored row-and-column edge reading (Eq. (S2.37)). The stream then forks: the physics leg and the router leg each refresh their own copy through a leaf-contextualizer stack and its closing edge reading (§ S2.4, § S2.5), and the copies never exchange information again. The physics copy is refreshed once per merge level and conditions the root readout; the router copy advances per decode position by reading the dyadic cover, and conditions the pointer.

The stream is wide, $d _ { g } = 1 0 2 4$ , and stays wide along the whole ladder; what varies is how the rest of the network reads it. The attention pools and the additive taps $( W _ { g }$ above, and their analogues downstream) are learned projections by construction. The remaining consumers, the leaf gate, the merge context leg and the root readout hypernet, each read the stream through their own RMSnormalised linear projection to a narrower interface of width 256; where a later display writes g as an input to one of these three, it denotes that consumer’s projected copy.

The attention itself is not the vanilla recipe, and both of its deviations recur across the architecture. First, the logits carry the bond stream:

$$
\mathrm { l o g i t } _ { { \bf h } , i j } ~ = ~ \frac { q _ { { \bf h } , i } \cdot k _ { { \bf h } , j } } { \sqrt { d _ { \bf h } } } ~ \mathrm { M L P } ( \mathrm { R M S } ( e _ { i j } ^ { \beta + 1 } ) )\tag{S2.34}
$$

with $\beta _ { \mathsf { h } } : \mathbb { R } ^ { d _ { e } }  \mathbb { R } \mathrm { ~ a ~ }$ small per-head scalar projector, so a head can attend along the interaction graph rather than by content alone. Second, the heads are gated in bisected pairs, the GLU micro-structure the featurizer’s attention layers already carry (Eq. (S2.17)). The level attention of § S2.4.4 and the contextualizer’s slot attention [127] reuse it in turn. Figure S4 draws one block.

The edge-update map itself combines a direct-pair contribution with a three-part aggregate that pools over every site k acting as a third vertex. Concretely, its increment at bond (i, j) is

$$
\mathrm { E d g e U p d a t e } \big ( \mathrm { R M S } ( e ) , \ell \big ) _ { i j } \ = \ \mathrm { F F N } \big ( \big [ \mathrm { p a i r } _ { i j } , P _ { i j } \big ] \big ) ,\tag{S2.35}
$$

![](images/bfc734763d6bed1311cd819517a3d390eea20645a50d587fb60e7e81b90cda5e.jpg)  
Figure S4: One block of the trunk (§S2.3); sublayers run in the order drawn. The per-bond stream $e ^ { ( \breve { \beta } ) } \in \mathbb { R } ^ { N \times N \times d _ { e } }$ (deeper teal) moves first through the pre-norm edge update of Eqs. (S2.35)–(S2.36), consuming the endpoint sites $\ell _ { i } , \ell _ { j }$ . The freshly updated bonds $e ^ { ( \beta + \bar { 1 } ) }$ then bias the per-site attention’s logits (Eq. (S2.34)), whose output is head-gated, and a pre-norm FFN closes the per-site path. Every sublayer normalises with RMS and carries the residual gain $1 / \sqrt { L }$ . The global stream $g ^ { ( \bar { \beta ) } }$ (magenta) enters the node FFN through th additive projection $\bar { W _ { g } } g ^ { ( \beta ) }$ and is refreshed once per block from the updated site states (Eq. (S2.29)). L identical blocks are stacked.

where pai $\mathrm { r } _ { i j } : = [ e _ { i j } , \ell _ { i } , \ell _ { j } ]$ is the direct-pair feature, i.e. the bond together with its two endpoints, and $P _ { i j } \in \mathbb { R } ^ { d _ { c } }$ is the three-part aggregate,

$$
P _ { i j , c } = \frac { 1 } { \sqrt { N } } \sum _ { k = 1 } ^ { N } \Psi _ { c } ^ { \mathrm { L } } ( \mathrm { p a i r } _ { i k } ) \Psi _ { c } ^ { \mathrm { R } } ( \mathrm { p a i r } _ { k j } ) , \qquad c \in \{ 1 , \ldots , d _ { c } \} .\tag{S2.36}
$$

The two embedders $\Psi ^ { \mathrm { L } } , \Psi ^ { \mathrm { R } } : \mathbb { R } ^ { d _ { e } + 2 d _ { \ell } }  \mathbb { R } ^ { d _ { c } }$ are independent SiLU MLPs that act on the left leg $i  k$ and the right leg $k  j$ of the triangle $( i , k , j )$ . The $1 / \sqrt { N }$ prefactor keeps the variance of $P _ { i j }$ at O(1) as N grows. Sites k that do not complete a physical triangle with $( i , j )$ enter through legs carrying the absent-bond tokens of $\operatorname { E q } .$ . (S2.15), which the embedders learn to discount; absence is a representable input here, not a hard zero. The update costs $\mathcal { O } ( N ^ { 2 } d _ { e } d _ { c } )$ flops per layer, which is the same asymptotic order as the pair-feature attention already in the block.

This is the triangular update pattern of AlphaFold’s Evoformer [128], adapted from residue-pair representations to coupling graphs. We opt for this three-part update because it is well suited to handle frustrated systems. On a triangular antiferromagnet, a kagome lattice, or any $J _ { 1 } { - } J _ { 2 }$ chain whose nearest- and next-nearest-neighbour bonds close into triangles, two bonds can share endpoint features and still sit in physically distinct environments. Thus the bond’s efective behaviour is set by the triangle of competing couplings it belongs to, not by its endpoints alone. Without $P _ { i j }$ , Eq. (S2.35) reduces to the pair-only form standard in message-passing GNNs, which collapses such bonds into a single representation. With $P _ { i j }$ , the FFN sees the full multiset $\{ ( \mathrm { p a i r } _ { i k } , \mathrm { p a i r } _ { k j } ) : k \}$ of triangle completions and separates them.

In graph-theoretic language, the pair-only limit of Eq. (S2.35) is 1-Weisfeiler–Lehman in distinguishing power on the line graph, which is the level at which most of-the-shelf message-passing GNNs operate. The triangular aggregate $P _ { i j }$ lifts the update strictly to the 2-WL refinement on pairs [129, 130] which distinguishes any two bonds whose triangle structures difer.

After the L blocks the global context stream takes one further update, this time reading the bonds rather than the sites. Placing all $N ^ { 2 }$ bond states under a single softmax would be wasteful, so the reading is factored through row and column descriptors,

$$
r _ { i } = \mathrm { p o o l } \bigl ( g , e _ { i \cdot } ^ { ( L ) } \bigr ) , \qquad c _ { j } = \mathrm { p o o l } \bigl ( g , e _ { \cdot j } ^ { ( L ) } \bigr ) , \qquad p = \mathrm { p o o l } \bigl ( g , \{ r _ { 1 } , \ldots , r _ { N } , c _ { 1 } , \ldots , c _ { N } \} \bigr ) ,\tag{S2.37}
$$

followed by the update $g \mapsto { \mathrm { U p d a t e } } ( g , p )$ of $\operatorname { E q . }$ (S2.33). The row and column descriptors compress the $N ^ { 2 }$ bonds to 2N summaries, and the second-stage pool reads those; every pool here runs under the physical-site mask. The same factored reading recurs whenever the global reads an edge field downstream.

The trunk thus emits the triple $( e ^ { ( L ) } , \ell ^ { ( L ) } , g )$ , where g now denotes the stream’s post-trunk value. Each of these is, by construction, a deterministic function of the featurizer’s outputs $( e ^ { ( 0 ) } , \ell ^ { ( 0 ) } , g ^ { ( 0 ) } )$ , which are themselves deterministic functions of the cached inputs $J ^ { \prime \prime }$ and $h ^ { \prime \prime }$ of the raw Hamiltonian $( J , h )$

Hence the whole pipeline up to here is a deterministic function of $( J , h )$ alone, and the spin configuration q coordinates have not yet entered the model. The per-site $\ell ^ { ( L ) }$ feeds the leaf builder defined immediately below, refined on the way by a leaf contextualizer that the tree readout introduces (§ S2.4); the per-bond $e ^ { ( L ) }$ re-enters at the level-edge attention variant of the merge tree (§ S2.4); and the global stream continues through every later stage, refreshed at each in the same pool-and-update pattern.

## S2.3.1 The per-site leaf builder

The leaf builder is the unique entry point for the spin configuration $q \colon$ everything upstream of it is a function of the Hamiltonian alone, and everything downstream of it is a function of $q$ as well. It acts one site at a time. Given the contextualized per-slot state $\hat { \ell } _ { i } \in \mathbb { R } ^ { d _ { \ell } }$ — the trunk’s per-site outputs, refined over the readout’s slot frame by the leaf contextualizer whose construction we give in § S2.4.3, the global context $g ~ \in \mathbb { R } ^ { 2 5 6 }$ , the stream’s refreshed post-context value read through the leaf’s own projection (§ S2.3), and the quaternion-embedded spin $q _ { i } \in \mathrm { S U } ( 2 ) \hookrightarrow \mathbb { R } ^ { 4 }$ with $\textstyle \sum _ { a } ( q _ { i } ^ { a } ) ^ { 2 } = 1$ (§ S1.1), it produces a per-site carrier

$$
u _ { i } ^ { ( 0 ) } \in \mathbb { R } ^ { d _ { u } }
$$

that encodes everything site i contributes to the wavefunction amplitude. The N carriers $\{ u _ { i } ^ { ( 0 ) } \} _ { i = 1 } ^ { N }$ 1 are the leaves of the tree readout of $\ S \ S 2 . 4$ , which contracts them into the single scalar log $\psi _ { \theta } ( q ) \in \mathbb { C } ;$ the width $d _ { u }$ is set by that readout.

Recall from the opening of $\ S \ S 2 . 3$ that, because q enters nowhere else, the per-site antisymmetry constraint (S2.24) becomes a property of this one map. The design problem is therefore: build $u _ { i } ^ { ( 0 ) }$ as a function of $( \hat { \ell } _ { i } , g , q _ { i } )$ that is expressive in all three arguments yet exactly odd under $q _ { i } \to - q _ { i }$ Our solution keeps the two parities on separate legs and couples them one way only, which keeps the necessary properties of the sector argument we made in § S1.4, namely that the model is odd and linear in each quaternion $q _ { i }$ . Indeed, the q-leg is linear in $q _ { i }$ because it is bias-free (since a constant would be even), and the other leg is an unconstrained function of $( \hat { \ell } _ { i } , g )$ . The two legs meet in a product, never in a sum, thus preserving the central oddness of the model. Figure S5 draws one site’s unit.

Let BiasFreeLinear $( d _ { \mathrm { i n } } , d _ { \mathrm { o u t } } )$ denote a linear map with no additive bias. The q-leg is a single bias-free lift of the quaternion into what we call the odd stream,

$$
\begin{array} { r l r } { z _ { i } : = W _ { q \to z } q _ { i } \in \mathbb { R } ^ { d _ { o } } , } & { } & { W _ { q \to z } \in \mathrm { B i a s F r e e L i n e a r } ( 4 , d _ { o } ) , } \end{array}\tag{S2.38}
$$

with $d _ { o }$ the odd-stream width. The absence of a bias is not a stylistic choice: an additive constant is even under $q _ { i } \to - q _ { i }$ , and (S2.38) is the first link in a chain whose oddness must be exact — this is the bias-free constraint of § S1.4 in its simplest form.

The even leg maps the per-slot and global context to a gate vector,

$$
m _ { i } : = W _ { h } \left[ \widehat { \ell } _ { i } , g \right] \in \mathbb { R } ^ { R } ,\tag{S2.39}
$$

where here $\left[ \cdot , \cdot \right]$ denotes concatenation, $W _ { h } \in \mathbb { R } ^ { R \times ( d _ { \ell } + 2 5 6 ) }$ , and R is a rank hyperparameter we return to in a moment. Since $\boldsymbol { \hat { \ell } } _ { i }$ and $g$ are functions of the Hamiltonian alone (§ S2.3), carrying no q-dependence whatsoever, the gate $m _ { i }$ is q-independent by construction — in particular invariant under $q _ { i } \to - q _ { i }$

The two legs meet in a Hadamard (elementwise) product inside an R-dimensional bottleneck, followed by a bias-free output projection,

$$
u _ { i } ^ { ( 0 ) } = U \big ( m _ { i } \odot V z _ { i } \big ) \ \in \ \mathbb { R } ^ { d _ { u } } , \qquad V \in \mathbb { R } ^ { R \times d _ { o } } , \quad U \in \mathbb { R } ^ { d _ { u } \times R } ,\tag{S2.40}
$$

with V and U bias-free for the same parity reason as $W _ { q  z }$ . Collecting the linear maps shows what (S2.40) really is:

$$
u _ { i } ^ { ( 0 ) } = M ( \hat { \ell } _ { i } , g ) \ z _ { i } , \qquad M ( \hat { \ell } _ { i } , g ) \ : = U \mathrm { d i a g } ( m _ { i } ) V \ \in \ \mathbb { R } ^ { d _ { u } \times d _ { o } } .\tag{S2.41}
$$

![](images/8fe2f431411a1875001c8a814d3d548648ee770ccc0cace548cdc55e5edd15b6.jpg)  
Figure S5: The per-site leaf builder (§ S2.3.1). Two input streams, one output carrier. The Hamiltonian context of per-slot $\hat { \ell } _ { i }$ and global $g$ (teal) sets the gate $m _ { i } = { \cal W } _ { h } [ \widehat \ell _ { i } , g ]$ of $\operatorname { E q . }$ (S2.39), which is $q -$ independent by construction. The spin configuration $q _ { i }$ (from $\operatorname { M C M C } )$ enters through the bias-free lift $z _ { i } = W _ { q  z } q _ { i }$ (orange) and meets the gate in the rank-R Hadamard product of $\operatorname { E q }$ . (S2.40) (violet), followed by the bias-free projection $U$ and the parity-odd scale normalisation of $\operatorname { E q } .$ . (S2.43). The RMS normalisation is nonlinear in $q _ { i } ,$ but its divided-out scale is stored as $s _ { i } ^ { ( 0 ) } = \log r _ { i } ^ { ( 0 ) }$ and restored exactly at the root. Consequently $u _ { i } ^ { ( 0 ) }$ is odd, while the reconstructed carrier $e ^ { s _ { i } ^ { ( 0 ) } } u _ { i } ^ { ( 0 ) } = B _ { i } q _ { i }$ is exactly linear in $q _ { i }$

The even leg emits the weights of a linear map of rank at most $R ,$ and the spin is read out through that map. In ML terms this is a small hypernetwork [131], one network outputs the parameters of the map applied by another, with the Hadamard gate playing the role of FiLM-style multiplicative conditioning [132] in a rank-R bottleneck. In physics terms, the even sector dresses the odd sector without ever mixing into it. The same three-matrix shape $U ( \cdot \odot V \cdot )$ reappears as the residual hypernet at every merge node of the tree (§ S2.4.3); the leaf builder is its first and simplest instance.

Finally, we normalise the carrier to unit root-mean-square while banking the divided-out magnitude in the log-scale stream. To that end, we may write

$$
B _ { i } : = M ( \ell _ { i } , g ) W _ { q \to z } , \qquad u _ { i , \mathrm { r a w } } ^ { ( 0 ) } : = B _ { i } q _ { i } ,\tag{S2.42}
$$

to define

$$
r _ { i } ^ { ( 0 ) } : = \sqrt { \overline { { \big ( u _ { i , \mathrm { r a w } } ^ { ( 0 ) } \big ) ^ { 2 } } } + \varepsilon } , \qquad u _ { i } ^ { ( 0 ) } : = \frac { u _ { i , \mathrm { r a w } } ^ { ( 0 ) } } { r _ { i } ^ { ( 0 ) } } , \qquad s _ { i } ^ { ( 0 ) } : = \log r _ { i } ^ { ( 0 ) } ,\tag{S2.43}
$$

where,

$$
\left( r _ { i } ^ { ( 0 ) } \right) ^ { 2 } \ = \ \frac { 1 } { d _ { u } } q _ { i } ^ { \top } B _ { i } ^ { \top } B _ { i } q _ { i } + \varepsilon , \qquad \overline { { { x ^ { 2 } } } } : = \frac { 1 } { d _ { u } } \sum _ { a } x _ { a } ^ { 2 } .\tag{S2.44}
$$

The normalised carrier $u _ { i } ^ { ( 0 ) }$ is odd but, by itself, nonlinear in $q _ { i }$ because its denominator is the square root of the quadratic form in Eq. (S2.44). The pair $( u _ { i } ^ { ( 0 ) } , s _ { i } ^ { ( 0 ) } )$ ), however, preserves the raw linear carrier exactly,

$$
e ^ { s _ { i } ^ { ( 0 ) } } u _ { i } ^ { ( 0 ) } \ = \ u _ { i , \mathrm { r a w } } ^ { ( 0 ) } \ = \ B _ { i } q _ { i } .\tag{S2.45}
$$

Thus no magnitude degree of freedom is deleted at the leaf. The normalisation is a numerical gauge choice whose scale is banked at the beginning of the merge, propagated additively through the tree, and restored at the root readout. This means the non-linearity introduced by normalising cancels with the global scale added through each step of the merge, and the overall model remains multi-linear in each of the manifold’s quaternion coordinates.

Next we highlight the the reason for using a product here and not a standard architectural choice of a FFN. A generic FFN on concat $\mathbf { \xi } ( \hat { \ell } _ { i } , g , z _ { i } )$ would mix even and odd inputs nonlinearly, generate terms of even degree in $q _ { i } .$ , and (S2.24) would hold only to the extent that training happened to learn it. The gated form (S2.41) is the simplest combiner that makes oddness structural. The q-dependence stays linear in $z _ { i } ,$ with coeficients the Hamiltonian context may set freely. Hence, the even side of the model is completely unconstrained when it comes to representational capacity.

Indeed, every map on the q-leg is linear and bias-free, so $z _ { i } ( - q _ { i } ) = - z _ { i } ( q _ { i } )$ , the matrix $M ( \hat { \ell } _ { i } , g )$ does not move when $q _ { i }$ flips, and the normalisation (S2.43) has an odd numerator and an even denominator. The composition is therefore odd,

$$
u _ { i } ^ { ( 0 ) } ( \hat { \ell } _ { i } , g , - q _ { i } ) = - u _ { i } ^ { ( 0 ) } ( \hat { \ell } _ { i } , g , q _ { i } ) .\tag{S2.46}
$$

This parity requirement also propagates through the rest of the network. Each merge node of $\ S \ S 2$ .4 is linear in each of its two children separately, and each scale normalisation along the way flips its output whenever its input flips, so a sign flip at leaf i propagates to a sign flip of the root carrier and changes nothing else. The complex readout at the root converts that flip into log $\psi _ { \theta } \mapsto \log \psi _ { \theta } + i \pi$ . That is, $\psi _ { \theta } \mapsto - \psi _ { \theta }$ with $| \psi _ { \theta } |$ untouched in line with (S2.24).

The distinction between the normalised carrier and the represented amplitude is essential. The normalised carrier $u _ { i } ^ { ( 0 ) }$ contains the even denominator of Eq. (S2.44) and is therefore not a degree-one function of $q _ { i }$ . The tree never discards that denominator however. Instead it carries $s _ { i } ^ { ( 0 ) } = \log r _ { i } ^ { ( 0 ) }$ adds every later merge scale, and restores the complete scale at the root. Hence the represented leaf amplitude is $e ^ { s _ { i } ^ { ( 0 ) } } u _ { i } ^ { ( 0 ) } = B _ { i } q _ { i }$ , a pure $\mathrm { s p i n } { - } \frac { 1 } { 2 }$ object. Thus the scale normalisations do not intro duce higher per-site Peter–Weyl harmonics into the final wavefunction, since their apparent nonlinear dependence cancels algebraically in the reconstructed amplitude.

## S2.4 Neural tensor network readout from leaves

The leaf builder from § S2.3.1 produces, at every site $i \in \{ 1 , \ldots , N \}$ , a carrier $u _ { i } ^ { ( 0 ) } \in \mathbb { R } ^ { d _ { u } }$ that is linear in its own site’s $q _ { i }$ and independent of every other site’s. To evaluate a wavefunction, we need to collapse these $N$ carriers into a single complex scalar log $\psi _ { \theta } ( q ) \in \mathbb { C } .$ , and we need the collapse to stay linear in each site separately, as multilinearity is the function class the spin- <sup>1</sup> sector demands, see also (§ S2.3.1). A natural way to do so with multilinear pieces is a balanced binary tree of merges, which pairs the carriers up, merges each pair into a parent carrier through a bilinear map, then pairs the parents up, merges them in turn, and continue for $K : = \log _ { 2 } \hat { N }$ levels until one root carrier survives, where $\hat { N }$ is the closest power of 2 larger than N. A linear readout on the root produces the scalar.

Indeed, the tensor-network community has used this concept in the contraction schedule of a tree tensor network [11]. The comparison is worth dwelling on for a moment because it reveals limitations of a binary-tree merge. Indeed, a contraction geometry is an entanglement prior, since the tree structure encodes in advance which sites can correlate cheaply (through a shallow common ancestor) and which only expensively (through many intermediate tensors), so choosing the geometry well classically requires knowing the correlation structure of the ground state one is trying to find. For a single Hamiltonian, we are able to make such a prior contraction geometry by hand, and both the Tensor Network and the Neural Quantum State literatures [11, 16] have found that a carefully chosen architecture as a training prior can outperform a generic one. However, a foundation model spanning chains, frustrated lattices, disordered ensembles, and long-range couplings cannot commit to one geometry.

In this section, we detail how to circumvent the contraction flexibility issue with a single, shared tree architecture that can be cheaply reconditioned to fit the correlation structure of any Hamiltonian that is quadratic weight in the Pauli group.

One merge module is shared across the entire tree, and it is modulated by Hamiltonian context; the levels exchange information laterally through attention, and the assignment of physical sites to leaves is promoted in § S2.5 to a learned, per-Hamiltonian contraction.

Every node of the tree carries a state of three objects, and a fourth object lives between the nodes

of each level:

$$
\underbrace { c \in \mathbb { R } ^ { d _ { c } } } _ { \mathrm { c o n t e x t ~ ( e v e n ) } } , \qquad \underbrace { u \in \mathbb { R } ^ { d _ { u } } } _ { \mathrm { c a r r i e r ~ ( o d d ) } } , \qquad \underbrace { s \in \mathbb { R } } _ { \mathrm { l o g - s c a l e } } , \qquad \underbrace { E ^ { ( \kappa ) } \in \mathbb { R } ^ { N _ { \kappa } \times N _ { \kappa } \times d _ { \mathrm { e d g e } } } } _ { \mathrm { l e v e l ~ e d g e ~ f i e l d } } ,\tag{S2.47}
$$

where $N _ { \kappa } : = \hat { N } / 2 ^ { \kappa }$ is the number of nodes at level $\kappa \in \{ 0 , \ldots , K \}$ . The context c is the q-independent stream, and it tells a merge node what physical Hamiltonian it is contracting through its encoding in the trunk, see § S2.4.3.

The normalised carrier u is the odd stream of § S2.3.1, together with the log-scale s it represents the reconstructed carrier

$$
a : = e ^ { s } u .\tag{S2.48}
$$

At an active leaf, $a _ { i } ^ { ( 0 ) } = B _ { i } q _ { i }$ is exactly linear in its own quaternion. The scalar s starts at the leaf with $s _ { i } ^ { ( 0 ) } = \log r _ { i } ^ { ( 0 ) }$ and then accumulates every magnitude divided out by the K levels of merge normalisation. The individual u stream is kept at unit RMS for numerical stability, and multilinearity is an invariant of the reconstructed stream $a = e ^ { s } u$

The edge field $E ^ { ( \kappa ) }$ is a coarse-grained descendant of the trunk’s per-bond data $e ^ { ( L ) }$ , with one feature vector per pair of level-κ subtrees. It plays the role of an efective coupling between blocks, in the spirit of Wilson renormalisation [133].

From these objects in the tree, the readout happens in five steps, one per subsection below, with the routing of §S2.5 having to its own dedicated section. We start by defining the balanced frame of slots and padding holes, then state the bilinear merge and identify the two bottlenecks it imposes (§ S2.4.1). The first bottleneck, a hidden locality prior on which leaves entangle cheaply, is loosened by the bit-reversed canonical frame (§ S2.4.2) and then we fully remove any locality priors by the learned routes of § S2.5. The second bottleneck, limited per-merge expressivity, is fixed by widening the merge to a blockwise rank-4 contraction tensor with a context-conditioned residual gate, after which the leaf contextualizer prepares the per-slot contexts and the level-0 edge field that the tree consumes at its leaves (§ S2.4.3). Level attention with a tree-aware position bias then deletes the $\log _ { 2 } \hat { N }$ depth penalty on long-range pair correlations (§ S2.4.4), and the root readout closes the construction and states the structural invariant that the energy kernels of SM § S4 exploit (§ S2.4.5), see Figure S6.

![](images/77f5b5cf842f31b13b5f45d7a089bdd953a1f0f0dda7c5cc59a3f371125d37e8.jpg)  
Figure S6: The tree readout at $\hat { N } = 8$ (§ S2.4). Orange slots carry the per-site carriers $u _ { i } ^ { ( 0 ) } \in \mathbb { R } ^ { d _ { u } }$ in the bit-reversed canonical frame; per-slot contexts and the level-0 edge field come from the leaf contextualizer (§ S2.4.3). Violet merge nodes apply the shared three-stream merge of Figure S8 to the node states $( c , u , s )$ . Lavender bands between merge levels are the level-attention insert of Eq. (S2.81) with the edge-biased LCA-ALiBi logits of $\operatorname { E q . }$ (S2.84). The teal lane on the right is the edge field, coarsened level by level (Eq. (S2.77), with 2-FWL refinement); it biases the bands and, at the top, conditions the root readout of Eq. (S2.86). The magenta lane on the left is the global stream, refreshed once per level from the freshly merged contexts; its final value enters the root readout’s hypernet alongside $e _ { \mathrm { r o o t } }$ and $c _ { \mathrm { r o o t } }$ (Eq. (S2.85)).

## S2.4.1 The balanced frame: slots, holes, and the bilinear merge

Let K be the smallest non-negative integer with $2 ^ { K } \geq N$ , and let $\hat { N } : = 2 ^ { K }$ , which we refer to as the N<sup>ˆ</sup> slots for leaf positions. The physical mask $m \in \{ 0 , 1 \} ^ { \hat { N } }$ has $m _ { t } = 1$ on the N slots occupied by physical sites and $m _ { t } = 0$ on the $\hat { N } - N$ padded slots, which we call holes. Holes carry zero couplings $( J _ { t u } = 0$ whenever $m _ { t } m _ { u } = 0 , h _ { t } = 0 )$ and the physical mask gates every carrier:

$$
u _ { t } ^ { ( 0 ) } \mapsto m _ { t } u _ { t } ^ { ( 0 ) } ,\tag{S2.49}
$$

so a hole can never inject amplitude into the wavefunction.

The carrier mask does not send a zero hole through an ordinary bilinear contraction however. For a parent node $P$ with child subtrees A and B, let $n _ { X }$ be the physical-leaf count of subtree X. The odd carrier merge is the masked pass-through

$$
\mathcal { M } _ { P } ( u _ { A } , u _ { B } ) : = \left\{ \begin{array} { l l } { M _ { P } T [ u _ { A } , u _ { B } ] , } & { n _ { A } > 0 , \ n _ { B } > 0 , } \\ { u _ { B } , } & { n _ { A } = 0 , \ n _ { B } > 0 , } \\ { u _ { A } , } & { n _ { A } > 0 , \ n _ { B } = 0 , } \\ { 0 , } & { n _ { A } = n _ { B } = 0 , } \end{array} \right.\tag{S2.50}
$$

Here T is the shared bilinear carrier contraction introduced in Eq. (S2.53) and implemented blockwise in Eq. (S2.61), and $M _ { P }$ is the parent-specific, context-dependent linear refinement defined in Eq. (S2.64). The log-scale is passed through with the live child in the one-live-child cases. Hole contexts and edge features still follow the even merge, so a hole can condition later gates without injecting an odd amplitude. A hole therefore does not annihilate a nonempty sibling subtree.

Holes are not erased from the even side either. The tree treats them as structural sites in their own right. A hole slot carries a context like any other slot, with two holes still merging their contexts, and the tree is always the perfect binary tree over N<sup>ˆ</sup> slots. This buys two things. A system’s tree is the same regardless of how far the batch later pads it, so evaluation is independent of batch composition, and a single foundation model can therefore serve many sizes of physical system with this mechanism. And since the network learns that a subtree contains holes only through the contexts prepared for it, where the holes sit becomes a structural decision the routing of § S2.5 can learn, not a convention fixed in advance. The implementation accordingly keeps two masks: the physical mask above for everything that touches amplitude, and a structural typing (a flag to mark the context features as coming from a hole) for everything that does not. These design choices are what allows the model to condition properly on arbitrary interaction topologies, including any topological defects in an otherwise idealised lattice tensor with block-diagonal J tensor.

Before the trunk’s outputs can be loaded into this frame, it is worth noting that the frame already comes equipped with a natural notion of distance between slots. Every piece of machinery in this section measures position with it. For two distinct slots $a , b \in \{ 0 , \ldots , \hat { N } - 1 \}$ , let lca $( a , b )$ denote the depth of their lowest common ancestor, counted from the root at depth 0. Slots a and b lie in the same level-κ subtree exactly when their binary expansions agree on all bits above position $\kappa ,$ since the subtrees of the balanced tree are the dyadic blocks of the slot index. The level at which two slots first meet is therefore set by their highest difering bit, which gives the common-ancestor depth a closed form via bitwise XOR,

$$
\operatorname { l c a } ( a , b ) = K - 1 - \lfloor \log _ { 2 } ( a \oplus b ) \rfloor , \qquad a \neq b .\tag{S2.51}
$$

For example, at $K = 3$ , slots 0 and 1 have a ⊕ $b = 1$ , so lca = 2 and they are siblings, while slots 3 and 4 have $a \oplus b = 7 .$ , so $\mathrm { { l c a } = 0 . }$ , and they only meet at the root, despite being index-adjacent. The example is worth pausing on: the slot metric is dyadic rather than translation-invariant. What matters is therefore the largest aligned block two slots share, counter to our intuition of how far apart their indices sit. From the common-ancestor depth we define the tree distance,

$$
d _ { \mathrm { t r e e } } ( a , b ) : = 2 \big ( { \cal K } - \mathrm { l c a } ( a , b ) \big ) ,\tag{S2.52}
$$

which counts edges from a up to the common ancestor and back down to b. Computationally, lca(a, b) is one XOR plus one bit-scan per pair, cheaper than any learned position table and trivially fused into whatever kernel consumes it. And (S2.51) is a statement about slots, not about sites, so nothing that reassigns sites to slots, including the learned routes of § S2.5, touches it.

A merge node at level κ takes two children, $( c _ { A } , u _ { A } , s _ { A } )$ and $\left( c _ { B } , u _ { B } , s _ { B } \right)$ , and produces a parent, on all three per-node objects, $\left( c _ { P } , u _ { P } , s _ { P } \right)$ , of (S2.47). The carrier half is the part that must stay multilinear, and its simplest form is one bilinear contraction through a learnable merge tensor,

$$
u _ { P } ^ { \alpha } = T \bigl [ u _ { A } , u _ { B } \bigr ] ^ { \alpha } = \sum _ { \beta , \gamma = 1 } ^ { d _ { u } } T _ { \alpha \beta \gamma } u _ { A } ^ { \beta } u _ { B } ^ { \gamma } , \qquad \alpha \in \{ 1 , \ldots , d _ { u } \} ,\tag{S2.53}
$$

which is linear in each child separately. Hence a sign flip can pass through (§ S2.3.1), and it is also bilinear in the pair. The context meanwhile has no parity constraint and hence we merge bottom-up, computing a candidate update

$$
\Delta _ { P } = \mathrm { F F N } _ { c } \big ( \big [ c _ { A } , c _ { B } , E _ { A B } ^ { ( \kappa ) } , E _ { B A } ^ { ( \kappa ) } , \tau _ { P } , \Phi _ { P } ; g \big ] \big )\tag{S2.54}
$$

and folding it into the averaged children by the normalised interpolation of Eq. (S2.33),

$$
\bar { c } = \mathrm { N o r m } \big ( \frac { 1 } { 2 } ( c _ { A } + c _ { B } ) \big ) , \qquad c _ { P } = \mathcal { N } _ { c } ( \bar { c } , \Delta _ { P } ) , \qquad \widetilde { \alpha } _ { c } = 0 . 8 \mathrm { ~ s i g m o i d } ( a _ { c } ) , \quad a _ { c } \big | _ { \mathrm { i n i t } } = \log \mathrm { i t } \bigg ( \frac { 0 . 2 } { 0 . 8 } \bigg ) \textbf { 1 } .\tag{S2.55}
$$

Here [ · ] is concatenation as in § S2.3.1. $E _ { A B } ^ { ( \kappa ) }$ and $E _ { B A } ^ { ( \kappa ) }$ are the two directed edge-field entries between exactly the two subtrees being merged via the renormalised coupling of (S2.47), and encode the interaction this contraction is about to absorb, and $\tau _ { P }$ is a fixed sinusoidal encoding [124] of the node’s level and of its dyadic position counted from the root, which we can think of as a root-centred clock. Centering the clock at the root rather than the leaves means trees of diferent depth agree near the top, which is one of the small choices that lets a model trained at one $\hat { N }$ evaluate at a larger one. The retraction guarantees a context of bounded magnitude at every level, and $\widetilde { \alpha } _ { c }$ sets the bounded per-channel interpolation from the averaged children toward the update. The same rule, with its own gate, reappears at the router’s tree-prefix merge (§ S2.5).

The input $\Phi _ { P } \in \mathbb { R } ^ { 3 2 }$ is a sinusoidal encoding of the merge’s counts and of its level. With $n _ { A }$ and n<sub>B</sub> the physical-leaf counts of the two children, $n _ { \mathrm { r e m } }$ the physical sites the subtree under P has not yet absorbed, κ the level and K the tree depth,

$$
\begin{array} { r l } & { \Phi _ { P } = \Big [ \sin ( \omega _ { f } \lambda ) , \cos ( \omega _ { f } \lambda ) \Big | } \\ & { \qquad \lambda \in \big \{ \log _ { 2 } ( 1 { + } n _ { A } ) , \log _ { 2 } ( 1 { + } n _ { B } ) , \log _ { 2 } ( 1 { + } n _ { \mathrm { r e m } } ) \big \} , } \\ & { \qquad \omega _ { f } = \pi / 2 ^ { f } , \ f = 0 , \ldots , 3 \Big ] . } \end{array}\tag{S2.56}
$$

together with the same sine–cosine encoding of the pair $( \kappa , K { - } 1 { - } \kappa )$ at $f = 0 , 1$ , for $2 4 + 8 = 3 2$ channels. The counts are deliberately ordered: $n _ { A }$ enters before $n _ { B }$ because the reduction path breaks the child-swap symmetry. They are also absolute rather than fractional, so identical physical content encodes identically across the sampled tree topologies of § S2.5. The level pair reports the node’s distance from the root and from the leaves.

The global g in Eq. (S2.54) is the stream’s value at the level being merged, read through the tree’s projection (§ S2.3): after each level’s merges, the tree refreshes the stream once over the contexts they produced, $g \mapsto \mathrm { U p d a t e } \big ( g , \mathrm { p o o l } ( g , \{ c _ { P } ^ { ( \kappa ) } \} ) \big )$ , pooling under the structural mask, so virtual siblings participate, and re-projects it for the next level. The final value feeds the root readout of § S2.4.5. We defer the log-scale half $s _ { P }$ to § S2.4.3, where its purpose becomes visible.

Using (S2.54) allows us to separate top-down conditioning pathway from the trunk of §S2.3 into the tree itself. The contexts are computed up the tree alongside the carriers, rooted in per-slot seeds prepared from the trunk’s outputs. This is the subject of § S2.4.3, which we visit before understanding two necessary constraints that a bilinear cascade imposes in terms of expressivity of the model and a locality prior.

Before that, we clarify two limitations of the bare bilinear cascade, since the next three subsections and the design choices that led to them aim to address them. First, the slot-to-site map is a hidden locality prior. Any two leaves at slots a and b can only exchange information through their lowest common ancestor in the tree, and the depth of that ancestor depends entirely on the chosen map. To couple two physical sites through a short contraction path, the map must place them in slots with a shallow common ancestor. Second, the bilinear contraction (S2.53) is either too expensive or too narrow in practise. Materialising T at the full carrier width costs $d _ { u } ^ { 3 }$ parameters and flops per merge, giving over $3 \times 1 0 ^ { 9 }$ at our reference width $d _ { u } = 1 5 3 6$ , which is infeasible on every merge node of the tree. On the other hand, shrinking $d _ { u }$ until the merge is cheap enough introduces a bottleneck that limits the model’s capacity to pass information. And at any width, the merge at the common ancestor remains the only channel between two leaves, lca(a, b) levels up. § S2.4.2 loosens the locality prior by re-indexing the frame; § S2.4.3 fixes the width; § S2.4.4 removes the depth penalty.

## S2.4.2 The bit-reversed canonical frame

The slot metric determines the path lengths induced by the tree, while the site-to-slot assignment determines which physical pairs receive those path lengths. The naive assignment of site i at slot i − 1 to start at zero, inherits the dyadic metric of (S2.51) directly. This is because site pairs meet early when their zero-based indices agree on high bits, so chain neighbours that straddle an aligned block boundary (e.g. sites 4 and 5, sitting at the slots 3 and 4 of the worked example in § S2.4.1) are maximally separated, while neighbours inside a block are siblings.

The prior that this encodes is therefore biasing towards aligned dyadic blocks of the site index, which is at best a rough proxy for chain locality, and meaningless for frustrated, disordered, or longrange systems.

To get around this, we opt for a diferent canonical frame, which instead places site i at slot brev $\boldsymbol { \kappa } ( i - 1 )$ , where brev is the K-bit reversal. To that end, let $b _ { r } ( t ) \in \{ 0 , 1 \}$ denote the bit of t at position $r \in \{ 0 , \ldots , K - 1 \}$ (least-significant at $r = 0 )$ , so that $\begin{array} { r } { t = \sum _ { r = 0 } ^ { K - 1 } b _ { r } ( t ) \cdot 2 ^ { r } } \end{array}$ ; the reversal writes those bits in the opposite order:

$$
\mathrm { b r e v } _ { K } ( t ) : = \sum _ { r = 0 } ^ { K - 1 } b _ { r } ( t ) \cdot 2 ^ { K - 1 - r } .\tag{S2.57}
$$

At $K = 3$ for example, this gives the map,

$$
0 \mapsto 0 , ~ 1 \mapsto 4 , ~ 2 \mapsto 2 , ~ 3 \mapsto 6 , ~ 4 \mapsto 1 , ~ 5 \mapsto 5 , ~ 6 \mapsto 3 , ~ 7 \mapsto 7 .\tag{S2.58}
$$

Figure S7 compares the two placements at $K = 3$ on the same example pair.

![](images/4f16508054d2340f9adc288fe28e1d401c6dcfe95b4d7ebd235a767201f3e708.jpg)  
Figure S7: Identity placement (left) versus the bit-reversed canonical frame $( \mathrm { r i g h t } )$ at $K = 3 \ ( { \hat { N } } = 8 )$ Orange squares are slots carrying physical site indices; violet squares are merge nodes. The bold path traces the common-ancestor route for one representative site pair. Under the identity placement, sites 1 and 5 (index distance 4) only meet at the root; in the bit-reversed canonical frame, the same sites are siblings. The slot metric (S2.51) is identical in both panels; only the occupancy changes.

Because brev $\displaystyle { } ^ { \prime } { \cal K } ( x ) \oplus \mathrm { b r e v } _ { K } ( y ) = \mathrm { b r e v } _ { K } ( x \oplus y )$ , reversal swaps the roles of high and low bits in (S2.51). Hence, under the bit-reversed placement, two sites meet at the level set by the lowest difering bit of their indices. Two sites whose indices difer by 1 (say sites 1 and 2) land at slots 0 and 4, on opposite halves of the tree; two sites whose indices difer by $\hat { N } / 2 = 4$ (say sites 1 and 5) land at slots 0 and 1, which are siblings. The canonical frame is therefore deliberately anti-local in the site index, in the sense that long aligned strides merge first, and index neighbours meet last.

Two further considerations make the anti-local frame the right canonical choice. First, a foundation model’s canonical frame should not bake the nearest-neighbor-chain prior into every system. the bitreversed frame commits the architecture to no particular correlation structure a priori; whatever locality a system needs must come through the learned routes, rather than some locality prior. The learned routes of § S2.5 condition on $( J , h )$ and can recover a contiguous layout when favoured by the couplings, up to automorphisms of the interaction graph. The second is bookkeeping. The packing is arranged so that a system’s N sites plus its $\hat { N } - N$ holes occupy the contiguous slot block $[ 0 , \hat { N } )$ bit-reversed within that block, however far the batch later pads beyond $\hat { N }$

This is what makes a system’s tree the same object at every batch shape, so that evaluation across mixed sizes and size extrapolations beyond the pre-training set are well-defined.

## S2.4.3 Rank-4 quadrilinear merge and the log-scale chain

We now fix the width problem that comes with a naive bi-linear merge (see § S2.4.1). The full-width merge tensor costs $d _ { u } ^ { 3 }$ per merge, and we seek the $d _ { u }$ -wide carrier. The resolution is to split the carrier into sub-blocks, contiguous slices of the carrier indexed by a counter $i \in \{ 1 , \ldots , B \}$ . Let

$$
B : = d _ { u } / d _ { r }\tag{S2.59}
$$

denote the sub-block count (config-time constraint: $d _ { r }$ divides $d _ { u } )$ , and for each carrier $u \in \mathbb { R } ^ { d _ { u } }$ define the 2-D reshape

$$
u ^ { ( \mathrm { 2 D } ) } \in \mathbb { R } ^ { B \times d _ { r } } , \qquad u _ { i , k } ^ { ( \mathrm { 2 D } ) } : = u _ { ( i - 1 ) d _ { r } + k } ,\tag{S2.60}
$$

with first index $i \in \{ 1 , \ldots , B \}$ enumerating sub-blocks and second index $k \in \{ 1 , \ldots , d _ { r } \}$ enumerating channels within a sub-block. The merge tensor can then keep a narrow contraction width $d _ { r }$ but increases in expressivity thanks to the sub-block axis, $T \in \mathbb { R } ^ { B \times \bar { d } _ { r } \times d _ { r } \times d _ { r } }$ . Contracting it with the two reshaped children produces the bilinear output, which we write $\tilde { u } ;$ the rank-4 quadrilinear contraction then reads,

$$
\tilde { u } _ { i , j } \ = \ \sum _ { k = 1 } ^ { d _ { r } } \sum _ { l = 1 } ^ { d _ { r } } T _ { i , j , k , l } u _ { A } ^ { ( 2 \mathrm { D } ) } { } _ { i , k } u _ { B } ^ { ( 2 \mathrm { D } ) } { } _ { i , l } , \qquad i \in \{ 1 , \ldots , B \} , \ j \in \{ 1 , \ldots , d _ { r } \} ,\tag{S2.61}
$$

flattened back to $\tilde { u } \in \mathbb { R } ^ { d _ { u } }$ for the next contraction level.

Each sub-block i runs an independent rank-3 trilinear sub-merge on its own slice of the children via its own slab $T _ { i , : , : , : }$ of the merge tensor, and the parent carrier is the concatenation of the B sub-block outputs. The cost arithmetic is what makes this work. With $B \cdot d _ { r } ^ { 3 }$ parameters instead of $d _ { u } ^ { 3 }$ , at the reference configuration $( d _ { r } = 3 2 , B = 4 8 , d _ { u } = 1 5 3 6 )$ that is $\sim 1 0 ^ { 6 }$ instead of $\sim 3 \times 1 0 ^ { 9 }$ , cheap enough that T is stored as a learnable tensor during pre-training.

We emphasise this because it marks a deliberate design choice; the merge tensor itself is not produced by a hypernet and does not depend on the context. Instead one $T ,$ shared across every level and every position of the tree, contracts pairs of leaves to the final pair below the root. Hamiltonian dependence enters the carrier path only through the multiplicative gate we add next. Sharing one merge across the tree is the TTN analogue of weight tying [11], and it is a key step for generalisation. The parameter count is independent of $\hat { N } ,$ and a deeper tree at evaluation time reuses the same merge it trained at every level. The sub-block axis i is also not a batch dimension to sum over.

A context-conditioned residual gate augments the quadrilinear merge without breaking parity or multilinearity. The bilinear output ˜u of Eq. (S2.61) can be refined by a context-conditioned residual hypernet, without projecting out of the carrier width. Let $R \leq d _ { u }$ be a hypernet bottleneck, and let $\bar { V } \in \mathbb R ^ { R \times d _ { u } } , U \in \bar { \mathbb R } ^ { d _ { u } \times R } , \bar { W } _ { h } \in$ BiasFreeLinear $\cdot ( d _ { c } , R )$ be three learned linear maps. The hypernet output reads

$$
H ( \tilde { u } , c _ { P } ) : = U \big ( W _ { h } ( c _ { P } ) \odot V \tilde { u } \big ) ,\tag{S2.62}
$$

with $\odot$ the elementwise (Hadamard) product. This is the same three-matrix gate as the leaf builder’s Eq. (S2.40), meaning it still preserves parity and multilinearity because the even sector modulates the odd sector multiplicatively. The map is dimension-preserving $( \tilde { u } \mapsto H ( \tilde { u } , c _ { P } ) \in \mathbb { R } ^ { d _ { u } } )$ and linear in ˜u

with q-independent coeficients, so the parent carrier below stays bilinear in $( u _ { A } , u _ { B } )$ . Hence the gate refines the quadrilinear merge with a parity-preserving signal about the context.

For numerical stability, the output is renormalised to unit root-mean-square, with the divided-out magnitude banked in the log-scale stream ,

$$
u _ { P } = \frac { u _ { \mathrm { o u t } } } { \widetilde { s } } , \qquad \widetilde { s } : = \sqrt { \overline { { u _ { \mathrm { o u t } } ^ { 2 } } } + \varepsilon } , \qquad s _ { P } = s _ { A } + s _ { B } + \log \widetilde { s } ,\tag{S2.63}
$$

where $\begin{array} { r } { u _ { \mathrm { o u t } } : = \tilde { u } + H ( \tilde { u } , c _ { P } ) , \overline { { x ^ { 2 } } } : = \frac { 1 } { d _ { \mathfrak { n } } } \sum _ { a } x _ { a } ^ { 2 } } \end{array}$ as in Eq. (S2.43), and $\varepsilon > 0$ is a small numerical-stability constant, see Fig. S8.

With the level-zero initialisation of $\operatorname { E q } .$ . (S2.43), define the reconstructed carrier $a _ { X } : = e ^ { s _ { X } } u _ { X }$ and the context-dependent linear map

$$
M _ { P } : = I _ { d _ { u } } + U ~ \mathrm { d i a g } ( W _ { h } ( c _ { P } ) ) V .\tag{S2.64}
$$

Since $H ( \widetilde { u } , c _ { P } ) = U \mathrm { d i a g } ( W _ { h } ( c _ { P } ) ) V \widetilde { u } .$ , one merge satisfies

$$
a _ { P } = M _ { P } T [ a _ { A } , a _ { B } ] .\tag{S2.65}
$$

The coeficients of $M _ { P }$ are independent of $q ,$ so the map preserves multilinearity while the recorded log-scale cancels the numerical normalisation exactly.

The factor ˜s and every $s _ { X }$ are even under a single-site sign flip, so the normalised carriers remain odd, and the reconstructed carriers remain multilinear. The accumulated scale is reattached at the root readout (§ S2.4.5). Figure S8 shows one merge node, with all three streams.

![](images/c5cadb1311032e9564eea021737647dce807f37a285c8785118261c5992a5683.jpg)  
Figure S8: One merge node, blown up (§ S2.4.3); the same module is applied at every level and position of the tree. Left (magenta): the even context leg, Eqs. (S2.54)–(S2.55): an FFN over both child contexts, the two directed sibling entries of the level edge field (teal), the root-centred dyadic clock $\tau _ { P } .$ , the count features $\Phi _ { P }$ and the per-level-refreshed global $^ { g , }$ folded into the averaged children by the normalised interpolation with gate $\widetilde { \alpha } _ { c }$ . Centre (violet): the odd carrier leg, children reshaped to $B \times d _ { r }$ , contracted by the literal shared merge tensor T (Eq. (S2.61)), gated by the residual hypernet $H ( \tilde { u } , c _ { P } )$ (Eq. (S2.62); the only place context touches the carrier), and renormalised to unit RMS. Right (grey): the log-scale leg banks log ˜s, Eq. (S2.63), so magnitudes accumulate additively instead of multiplicatively.

We now have a mechanism to merge a system of N spin-1/2 coordinates that formulates our manifold function’s map to a single complex scalar. Our map uses one merge tensor $T$ shared across the tree, and the same three-stream module at every level and position of the binary tree. It also has a conditioning pathway through the contexts, which are computed up the tree and can therefore carry information about the Hamiltonian and the system’s structure.

With the merge machinery in hand, what remains is to feed it context from the trunk that carries the (J, h) signal. This corresponds to formulating the bottom row of Figure S6 from the trunk’s outputs in Figure S4. Notice the frames of the trunk and readout mechanism are diferent. The trunk’s outputs $( \ell ^ { ( L ) } , e ^ { ( L ) } , g )$ are indexed by physical sites and know nothing about slots. Meanwhile the holes have no trunk data at all, yet we know from § S2.4.1 that holes must carry contexts of their own in order for the model to carry meaningful information across diferent sized systems. So before the first merge runs, every slot, hole or not, needs a structural description written for it that is a function of the trunk’s output data. We construct these descriptions with a leaf-contextualizer module.

As a module, the contextualizer is a single-pass bridge between the two frames,

$$
\mathtt { L e a f C o n t e x t u a l i z e r : \ } \big ( \ell ^ { ( L ) } , \ e ^ { ( L ) } , \ g \big ) \ \longmapsto \ \big ( \hat { \ell } \in \mathbb { R } ^ { \hat { N } \times d _ { \xi } } , \ E ^ { ( 0 ) } \in \mathbb { R } ^ { \hat { N } \times \hat { N } \times d _ { \mathrm { e q g e } } } , \ g \big ) ,\tag{S2.66}
$$

realised as two blocks in sequence. Two copies of this stack exist, the physics leg described here and the router leg of § S2.5; each forks its own global from the same post-trunk value, and the two copies never exchange information afterwards. The per-slot output does double duty: <sup>ˆ</sup>ℓ is the state that gates the leaf builder of § S2.3.1, and, projected once where $d _ { c } \neq d _ { \ell }$ , it seeds the level-0 context array $c ^ { ( 0 ) }$ of the merge recursion. Each block makes four updates on the running triple $( c , E , g ) \colon$ a context refresh, then a refresh of the global that reads the new contexts, then an edge rewrite that uses both, then a slot attention that reads the rewritten edges. The trunk seeds the state directly in the slot frame, with $c _ { t } = \ell _ { t } ^ { ( L ) }$ and $E _ { t t ^ { \prime } } = e _ { t t ^ { \prime } } ^ { ( L ) }$ . A hole inherits whatever the masked trunk left in its rows, and the first update discards exactly this. Two further inputs appear in every step. The physical mask $m _ { t }$ of § S2.4.1 decides who keeps their seed. A fixed sinusoidal clock $\tau _ { t } \in \mathbb { R } ^ { d _ { \ell } }$ on the slot index gives the module its only notion of slot identity, every learned map here is indiferent to slot order, so position enters only through the clock and the tree-distance weights and biases below.

Step 1 (context refresh — tree-weighted bond summary). Every slot first summarises its incident bonds, assigning larger weights to partners that it merges with earlier. Define the normalised tree-decay weights

$$
\omega _ { t t ^ { \prime } } = \frac { w _ { t t ^ { \prime } } } { \sum _ { t ^ { \prime \prime } \ne t } w _ { t t ^ { \prime \prime } } } , \qquad w _ { t t ^ { \prime } } = e ^ { - \frac { 1 } { 2 } \big ( K - \mathrm { l c a } ( t , t ^ { \prime } ) \big ) } ,\tag{S2.67}
$$

recalling that $K - \mathrm { l c a } ( t , t ^ { \prime } )$ is the level at which t and $t ^ { \prime }$ merge, so siblings $\left( K - \operatorname { l c a } = 1 \right)$ dominate while opposite halves of the tree $( K - \operatorname { l c a } = K )$ barely register. The bond summary and the context refresh are then

$$
\bar { e } _ { t } = \sum _ { t ^ { \prime } \neq t } \omega _ { t t ^ { \prime } } \Psi ^ { \mathrm { s } } \big ( E _ { t t ^ { \prime } } , E _ { t ^ { \prime } t } , \mathrm { s g n } ( t ^ { \prime } - t ) \big ) ,\tag{S2.68}
$$

$$
\Delta _ { t } ^ { \mathrm { c t x } } = \mathrm { F F N } _ { \mathrm { c t x } } \Big ( \mathrm { R M S } [ c _ { t } + \tau _ { t } , \bar { e } _ { t } , 1 - m _ { t } ] \Big ) ,\tag{S2.69}
$$

$$
c _ { t } \gets \mathcal { N } _ { \mathrm { c t x } } ( c _ { t } , \Delta _ { t } ^ { \mathrm { c t x } } )\tag{S2.70}
$$

where $\Psi ^ { \mathrm { s } } : \mathbb { R } ^ { 2 d _ { \mathrm { e d g e } } + 1 }  \mathbb { R } ^ { d _ { \ell } }$ is a small MLP on the two directed edges of the pair and their index order, and $\mathrm { F F N } _ { \mathrm { c t x } } : \mathbb { R } ^ { 2 d _ { \ell } + 1 } \to \mathbb { R } ^ { d _ { \ell } }$ proposes the update.

Step $\mathcal { Q } ~ ( g l o b a l ~ r e f r e s h )$ . The global then takes one update of Eq. (S2.33), pooling the just-refreshed contexts over the full slot set, $g \longleftarrow \mathrm { U p d a t e } \big ( g , \mathrm { p o o l } ( g , \{ c _ { t } \} ) \big )$ , holes included: after Step 1 a hole’s context carries the tree geometry it was rebuilt from, and that structural information is exactly what the global should read.

Step 3 (edge rewrite — synthesise hole-incident bonds). The refreshed contexts and global then rewrite the edge tensor,

$$
\Delta _ { t t ^ { \prime } } ^ { E } = \mathrm { F F N } _ { \mathrm { c t x - e d g e } } \Big ( [ m _ { t } m _ { t ^ { \prime } } \mathrm { R M S } ( E _ { t t ^ { \prime } } ) , \widetilde { c } _ { t } , \widetilde { c } _ { t ^ { \prime } } , W _ { E } g ] \Big ) , \qquad E _ { t t ^ { \prime } } \longleftarrow \mathcal { N } _ { \mathrm { c t x - e d g e } } ( E _ { t t ^ { \prime } } , \Delta _ { t t ^ { \prime } } ^ { E } ) .\tag{S2.71}
$$

where $\tilde { c } _ { t }$ is a bias-free projection of the refreshed context’s RMS copy, $W _ { E } \in \mathbb { R } ^ { 6 4 \times d _ { g } }$ reads the refreshed global into 64 further input channels, broadcast to every pair, and $\mathrm { F F N _ { c t x - e d g e } }$ maps the concatenation back to $\mathbb { R } ^ { d _ { \mathrm { e d g e } } }$ . Unlike the trunk’s additive tap of Eq. (S2.28), the global enters here as extra channels of the input. The pair mask $m _ { t } m _ { t ^ { \prime } }$ zeroes the edge input whenever either endpoint is a hole, so a hole-incident edge is synthesised purely from its two endpoint contexts. By the time the tree runs, an edge into a hole is as meaningful as an edge into a site.

Step 4 (slot attention). Steps 1 and 3 are local, in the sense that they are fixed pairwise maps with a fixed decay profile, so each block closes with the one operation that moves information across the whole slot set in a content-dependent way: a pre-norm self-attention layer. Queries, keys and values are projected from $\mathrm { R M S } ( c _ { t } + \tau _ { t } )$ and split into $n _ { \mathrm { h } }$ heads of width $d _ { \mathfrak { h } }$ , with the sans-serif head index of § S2.2, and the logits carry two additive biases,

$$
\begin{array} { r } { \mathrm { l o g i t } _ { \mathrm { h } , t t ^ { \prime } } = \frac { q _ { \mathrm { h } , t } \cdot k _ { \mathrm { h } , t ^ { \prime } } } { \sqrt { d _ { \mathrm { h } } } } - \alpha _ { \mathrm { h } } \left( K - \mathrm { l c a } ( t , t ^ { \prime } ) \right) + \beta _ { \mathrm { h } } \left( E _ { t t ^ { \prime } } \right) , } \end{array}\tag{S2.72}
$$

$$
\Delta _ { t } ^ { \mathrm { s l o t } } = \mathrm { A t t n } _ { \mathrm { s l o t } } ( \{ c _ { t ^ { \prime } } \} ; E ) _ { t } , \qquad c _ { t } \longleftarrow \mathcal { N } _ { \mathrm { s l o t - a t t n } } ( c _ { t } , \Delta _ { t } ^ { \mathrm { s l o t } } ) ,\tag{S2.73}
$$

$$
\Delta _ { t } ^ { \mathrm { s l o t - f i n } } = \mathrm { F F N } _ { \mathrm { s l o t } } ( \mathrm { R M S } ( c _ { t } ) ) , \qquad c _ { t } \gets \mathcal { N } _ { \mathrm { s l o t - f i n } } ( c _ { t } , \Delta _ { t } ^ { \mathrm { s l o t - f i n } } ) .\tag{S2.74}
$$

where $\alpha _ { \mathsf { h } } \geq 0$ is a fixed per-head slope (a scalar per head, not to be confused with the learned perchannel gates $\widetilde { \alpha } _ { g }$ and $\widetilde { \alpha } _ { c }$ of the global stream and the context leg), so that the heads see the tree’s geometry, and $\bar { \beta } _ { \mathfrak { h } } : \mathbb { R } ^ { d _ { \mathrm { e d g e } } }  \mathbb { R }$ is a per-head scalar projector of the freshly rewritten edge, so that the heads see the renormalised physics, again akin to Wilson’s renormalisation group. Both biases reappear in full in the level attention of § S2.4.4. The attention and FFN candidates are incorporated through the bounded updates above.

After the second block, the global takes one closing update: the factored row-and-column edge reading of Eq. (S2.37), now over the rewritten edges and under the full slot mask. Holes participate here precisely because the contextualizer has just written their states, where the post-trunk reading of § S2.3 saw physical sites alone. The outputs after the second block are therefore the per-slot contexts $c _ { t } ^ { ( 0 ) }$ that seed the level-0 context array, the rewritten edge tensor, which becomes $E ^ { ( 0 ) }$ , and the refreshed global. As such, this contextualizer manufactures the structural information for the holes that balance the binary tree, which is what makes hole placement a meaningful learned decision rather than dead padding and allows the model to generalise over diferent sized systems. With the frame loaded, the merge recursion has everything it needs at $\kappa = 0$ . For $\kappa \geq 1$ , the recursion additionally requires a coarse-grained edge field $E ^ { ( \bar { \kappa } ) }$ and the attention that consumes $\mathrm { i t } ;$ these are defined in the next subsection.

## S2.4.4 The edge field and level attention

Even with bit-reversed leaves and a quadrilinear merge, the cascade still imposes a $\log _ { 2 } \hat { N }$ depth penalty on long-range pair correlations. This is because information between two leaves at common-ancestor depth $K - \operatorname { l c a } ( a , b )$ has to travel through as many merge nodes as there are between it and its lca before they meet. We remove this penalty by interweaving lateral self-attention between the levels edge data during the tree contraction.

Recall from (S2.47) that level $\kappa$ carries an edge field $E ^ { ( \kappa ) } \in \mathbb { R } ^ { N _ { \kappa } \times N _ { \kappa } \times d _ { \mathrm { e d g e } } }$ , one feature vector per ordered pair of level-κ subtrees, seeded at $\kappa = 0$ by the leaf contextualizer’s rewritten edge tensor (§ S2.4.3). As the tree contracts, the size of the edge field must be coarse-grained so that it has the correct shape for a given level’s attention layer.

Thus, when a level’s nodes are merged pairwise, the edge field is coarse-grained alongside them. The four child entries connecting the two children of $P$ to the two children of $Q ,$ , together with the four child contexts, are combined into the single parent entry,

$$
\bar { E } _ { P Q } ^ { ( \kappa + 1 ) } : = \mathrm { N o r m } \left( \frac { 1 } { 4 } \sum _ { a , b \in \{ 0 , 1 \} } E _ { 2 P + a , 2 Q + b } ^ { ( \kappa ) } \right) ,\tag{S2.75}
$$

$$
\Delta _ { P Q } ^ { \mathrm { c o u r s e } } : = \mathrm { F F N } _ { \mathrm { c o a r s e } } \Big ( \big [ E _ { 2 P , 2 Q } ^ { ( \kappa ) } , E _ { 2 P , 2 Q + 1 } ^ { ( \kappa ) } , E _ { 2 P + 1 , 2 Q } ^ { ( \kappa ) } , E _ { 2 P + 1 , 2 Q + 1 } ^ { ( \kappa ) } , c _ { 2 P } ^ { ( \kappa ) } , c _ { 2 P + 1 } ^ { ( \kappa ) } , c _ { 2 Q } ^ { ( \kappa ) } , c _ { 2 Q + 1 } ^ { ( \kappa ) } \big ] \Big ) ,\tag{S2.76}
$$

$$
E _ { P Q } ^ { ( \kappa + 1 ) } : = \mathcal { N } _ { \mathrm { c o a r s e } } \Big ( \bar { E } _ { P Q } ^ { ( \kappa + 1 ) } , \Delta _ { P Q } ^ { \mathrm { c o a r s e } } \Big ) .\tag{S2.77}
$$

This update is applied at every parent pair $( P , Q )$ , with one $\mathrm { F F N _ { c o a r s e } }$ shared across all levels, like the merge tensor itself. This again has a direct analogy to Wilson’s real space renormalisation group at the level of a block-spin step [134, 135]. Each level halves the system and rewrites the efective couplings between blocks. Despite needing no spectral rank truncation, the four child edge cells and their contexts are nevertheless compressed into one fixed-width parent cell, and the information retained is exactly what the learned $\mathrm { F F N _ { c o a r s e } }$ and nGPT interpolation preserve. The result is a learned, rather than spectral, coarse-graining of the edge field across levels.

The diagonal is included deliberately, since the self-edge $E _ { P P } ^ { ( \kappa ) }$ , built from the on-diagonal $2 \times 2$ child block, carries the internal structure of block P (such as the frustration its subtree encloses) while the of-diagonal entries carry the renormalised couplings between blocks.

After this coarsening step, the edge field is refined by the same 2-FWL path-composition feature already used in the trunk’s edge update, applying Eq. S2.36 to the merge-node indices of level κ. This is so triangle-closure information survives the blocking. Figure S9 draws one coarsening step.

![](images/0e5251d61844b187af5e36fc190135261e287be3484cf828a9e59a75e9d5b238.jpg)  
Figure S9: One edge-coarsening step (§ S2.4.4). $\textup { A 2 } \times 2$ block of child edge cells (teal, highlighted) and the four child contexts (magenta) produce one parent edge cell via Eq. (S2.77) — a learned blockspin step. Diagonal self-edges (shaded) summarise the internal structure of their own subtree. The coarsened field is refined by the tree-level 2-FWL block and then biases the level-attention logits of Eq. (S2.84).

The attention layer between contraction levels is defined as follows. Let $c ^ { ( \kappa ) } \in \mathbb { R } ^ { N _ { \kappa } \times d _ { c } }$ be the context array at level $\kappa \in \{ 1 , \ldots , K \}$ with one row per merge node, and recall $N _ { \kappa } = \hat { N } / 2 ^ { \kappa }$ . Let $m ^ { ( \kappa ) } \in \{ 0 , 1 \} ^ { N _ { \kappa } }$ be the level mask that marks live merge nodes. The level-attention insert applies a pre-norm multi-head self-attention to block to $c ^ { ( \kappa ) }$ and applies two bounded updates,

$$
\Delta ^ { \mathrm { a t t n } , ( \kappa ) } : = \mathrm { A t t n } \big ( \mathrm { R M S } ( c ^ { ( \kappa ) } ) ; E ^ { ( \kappa ) } \big ) ,\tag{S2.78}
$$

$$
c ^ { ( \kappa ) } \longleftarrow m ^ { ( \kappa ) } \odot \mathcal { N } _ { \mathrm { l e v e l - a t t n } } \Big ( c ^ { ( \kappa ) } , \Delta ^ { \mathrm { a t t n } , ( \kappa ) } \Big ) + \big ( 1 - m ^ { ( \kappa ) } \big ) \odot c ^ { ( \kappa ) } ,\tag{S2.79}
$$

$$
\Delta ^ { \mathrm { H n } , ( \kappa ) } : = \mathrm { F F N } _ { \mathrm { l e v e l } } ( \mathrm { R M S } ( c ^ { ( \kappa ) } ) ) ,\tag{S2.80}
$$

$$
c ^ { ( \kappa ) } \longleftarrow m ^ { ( \kappa ) } \odot \mathcal { N } _ { \mathrm { l e v e l - f i n } } \Big ( c ^ { ( \kappa ) } , \Delta ^ { \mathrm { f i n } , ( \kappa ) } \Big ) + \left( 1 - m ^ { ( \kappa ) } \right) \odot c ^ { ( \kappa ) } .\tag{S2.81}
$$

As per the rank-4 contraction tensor and the FFN coarse graining network, the block’s parameters are shared across all K levels. This is again to allow us to act on arbitrary system sizes, since the tree metric is self-similar in that siblings are at distance one at every level. Hence one attention layer can serve every level, and at evaluation time, levels deeper than any seen in training can be constructed.

This block is our functional stand-in for the disentanglers of the multiscale entanglement renormalisation ansatz (MERA) [136]. Like them, it communicates across block boundaries at every scale, but it acts on the q-independent conditioning stream rather than on the state itself. The carrier path receives that signal only through the context-conditioned gates that modulate its multilinear coeficients. Consequently the block can alter cross-subtree correlations represented by the wavefunction without introducing nonlinear dependence on any quaternion.

Whilst we want the initial attention layers here to act as identity functions, we still need to encode the tree geometry with a positional encoding. Otherwise the attention layer will treat the merge nodes at each level as an unordered set. ALiBi [137] is the simplest construction that does this for a one-dimensional sequence. For a sequence of length L and an attention head $h \in \{ 1 , \ldots , H \}$ , the sequence-ALiBi logit at query position i and key position j is

$$
\mathrm { l o g i t } _ { h , i , j } ^ { \mathrm { s e q } } \ = \ \frac { q _ { h , i } \cdot k _ { h , j } } { \sqrt { d } } \ - \ \alpha _ { h } | i - j | ,\tag{S2.82}
$$

with $\alpha _ { h } \geq 0$ a per-head slope drawn from a geometric schedule (Press et al. [137, 138] use $\alpha _ { h } ~ =$ $2 ^ { - 8 h / H } )$ and d the per-head dimension. Here the content term $q _ { h , i } \cdot k _ { h , j } / \sqrt { d }$ scores how compatible the query at position i is with the key at position $j$ in content space, with no positional information mixed into $Q$ or $K$ . Positions enter the attention map only through the additive bias term, and the projection matrices are free to encode content purely. Notice also that the geometric slope schedule spans orders of magnitude, so the heads partition naturally by attention range; steep-slope heads see a short neighbourhood, while shallow-slope heads attend nearly globally, without any learned alignment. Finally, the bias is a function of the distance $| i - j |$ , so longer sequences at inference slot into the same linear penalty, and no out-of-distribution positional embeddings are needed when generalising to larger systems.

We adapt the ALiBi encoding [137] to a tree by replacing the sequence distance $| i - j |$ with the tree distance $d _ { \mathrm { t r e e } } = 2 ( K - \mathrm { l c a } )$ of Eq. (S2.52) from § S2.4.1. This means the LCA-ALiBi logit absorbs the factor of 2 into the head slope,

$$
\begin{array} { r } { \log \mathrm { i t } _ { h , a , b } ^ { \mathrm { t r e e } } \ = \ \frac { q _ { h , a } \cdot k _ { h , b } } { \sqrt { d } } \ - \ \alpha _ { h } \left( K - \mathrm { l c a } ( a , b ) \right) , \qquad \alpha _ { h } \ = \ \frac { 3 } { 2 } \cdot 2 ^ { - 4 h / ( H - 1 ) } , } \end{array}\tag{S2.83}
$$

where $h \in \{ 0 , \ldots , H - 1 \}$ . We set the slopes to fixed constants spanning the geometric ladder from 1.5 down to ≈ 0.09 (tuned empirically), so the heads partition by tree-distance band exactly as in the sequence case, with zero learnable positional parameters. The bias depends on the node pair only through $a \oplus b$ (via lca, $\mathrm { E q . \ S 2 . 5 1 } )$ . This means we have only one XOR plus one bit-scan per pair to compute, and is the same function at every level, meaning the computation is self-similar across the tree and we can share the block across diferent layers. Hence the same position encoding drives the route policy of § S2.5, and we will frequently refer back to it in § S2.5.

Finally, the coarsened edge field enters the logits as a learned additive bias alongside the treedistance term,

$$
\mathrm { l o g i t } _ { h , a , b } ^ { \mathrm { t r e e , ~ e d g e } } \ = \ \frac { q _ { h , a } \cdot k _ { h , b } } { \sqrt { d } } \ - \ \alpha _ { h } \left( K - \mathrm { l c a } ( a , b ) \right) \ + \ \beta _ { h } \big ( E _ { a b } ^ { ( \kappa ) } \big ) ,\tag{S2.84}
$$

with $\beta _ { h } : \mathbb { R } ^ { d _ { \mathrm { e d g e } } }  \mathbb { R }$ a small per-head scalar projector, scaled by the same $1 / \sqrt { d }$ as the content term so that changing the head width does not reweight geometry against content. Thus the heads see where two blocks sit in the tree and how strongly the renormalised Hamiltonian couples them.

## S2.4.5 Root readout and the structural invariant

After K levels of merges and level attention, one node remains carrying the root state $( c _ { \mathrm { r o o t } } , u _ { \mathrm { r o o t } } , s _ { \mathrm { r o o t } } )$ together with the top of the edge stream $e _ { \mathrm { r o o t } } : = E _ { 0 0 } ^ { ( K ) } \in \mathbb { R } ^ { d _ { \mathrm { e d g e } } }$ as a fully renormalised state of the whole system’s coupling structure. The readout maps the root carrier to a pair of real numbers through a hypernet-generated readout matrix $W \in \mathbb { R } ^ { 2 \times d _ { u } }$ , produced from the root’s full even state,

$$
W \ = \ W \big ( \big [ \mathrm { R M S } ( e _ { \mathrm { r o o t } } ) , \ c _ { \mathrm { r o o t } } , \ g \big ] \big ) ,\tag{S2.85}
$$

by the same low-rank $\mathrm { g a t e ^ { 2 } }$ as Eqs. (S2.40) and (S2.62): the renormalised coupling structure $e _ { \mathrm { r o o t } } .$ the root context $c _ { \mathrm { r o o t } }$ , and the global stream’s final value, refreshed through every level of the tree and read through the root’s own projection (§ S2.4.1). This allows us to assemble a complex log-amplitude from that pair along with the banked log-scale,

$$
\begin{array} { r } { ( \psi _ { 1 } , \psi _ { 2 } ) = W \boldsymbol { u } _ { \mathrm { r o o t } } \in \mathbb { R } ^ { 2 } , \qquad \log \psi _ { \boldsymbol { \theta } } ( q ) = \underbrace { \frac { 1 } { 2 } \log \bigl ( \psi _ { 1 } ^ { 2 } + \psi _ { 2 } ^ { 2 } \bigr ) + s _ { \mathrm { r o o t } } } _ { \log | \psi _ { \boldsymbol { \theta } } | } + i \arctan 2 ( \psi _ { 2 } , \psi _ { 1 } ) . } \end{array}\tag{S2.86}
$$

Here atan2 is the two-argument arctangent, giving the usual phase of the complex scalar $\psi _ { 1 } + i \psi _ { 2 }$ in $( - \pi , \pi ]$ . Hence, decomposition is exactly $\psi _ { \theta } = e ^ { s _ { \mathrm { r o o t } } } ( \psi _ { 1 } + i \psi _ { 2 } )$ , and the pair is the amplitude in Cartesian form at unit scale, with $s _ { \mathrm { r o o t } }$ re-attaching every magnitude the scale normalisations banked on the way up (§ S2.4.3). We can formalise this by the following lemma.

Lemma S2.1 (Restored-scale multilinearity). For every active site $i ,$ let $a _ { i } ^ { ( 0 ) } : = e ^ { s _ { i } ^ { ( 0 ) } } u _ { i } ^ { ( 0 ) }$ . Under the leaf initialisation (S2.43), the bilinear merge (S2.65), and the root readout (S2.86), the represented amplitude $\psi _ { \theta } ( q _ { 1 } , \dots , q _ { N } )$ is degree one in each quaternion $q _ { i }$ separately.

Proof. At a leaf, Eq. (S2.45) gives $a _ { i } ^ { ( 0 ) } = B _ { i } q _ { i }$ , with $B _ { i }$ independent of every quaternion. Suppose inductively that $a _ { A }$ and $a _ { B }$ are multilinear in the active leaves of two disjoint subtrees A and B. When both children contain physical leaves, the map $M _ { P } T$ is bilinear; when exactly one is empty, Eq. (S2.50) passes the other child through linearly. Induction over the nonempty leaves therefore preserves degree one in every physical quaternion. Induction up the balanced tree gives a multilinear $a _ { \mathrm { r o o t } }$ . Finally, if $W _ { 1 } , W _ { 2 }$ are the two rows of the root readout, then

$$
\psi _ { \theta } ~ = ~ e ^ { s _ { \mathrm { r o o t } } } ( \psi _ { 1 } + i \psi _ { 2 } ) ~ = ~ ( W _ { 1 } + i W _ { 2 } ) a _ { \mathrm { r o o t } } ,\tag{S2.87}
$$

which is a linear functional of a multilinear carrier. Therefore $\psi _ { \boldsymbol { \theta } }$ is degree one in each $q _ { i }$ □

One might ask why the readout does not simply emit log ψ as a linear function of $u _ { \mathrm { r o o t } }$ , which looks more direct. However we emphasise that this would violate the central oddness of our phase space since (S2.24) demands that a single-site flip send $\psi _ { \theta }  - \psi _ { \theta }$ , i.e. log $\psi _ { \theta }  \log \psi _ { \theta } + i \pi$ . Notice that this is an afine shift at the level of the logarithm, not a sign flip. Eq. (S2.86) is built so that the linearity sits exactly there, and thus,

$$
q _ { i } \to - q _ { i } \implies u _ { i } ^ { ( 0 ) } \to - u _ { i } ^ { ( 0 ) } \implies u _ { \mathrm { r o o t } } \to - u _ { \mathrm { r o o t } } \implies ( \psi _ { 1 } , \psi _ { 2 } ) \to - ( \psi _ { 1 } , \psi _ { 2 } ) \implies \psi _ { \theta } \to - \psi _ { \theta } .\tag{S2.88}
$$

The first arrow is the leaf oddness (S2.46). The second holds because each merge is linear in each child and its normalisation is parity-odd, so the sign flip propagates up the unique root-ward path through exactly one node per level. Every log-scale ˜s is even and stays put, and nothing else can change. The third arrow holds because W is a function of $( e _ { \mathrm { r o o t } } , c _ { \mathrm { r o o t } } , g )$ , all even, q-independent quantities. And the fourth is (S2.86): $\psi _ { 1 } ^ { 2 } + \psi _ { 2 } ^ { 2 }$ is invariant, so log|ψ<sub>θ</sub>| does not move, while the atan2 picks up exactly π. Per-site oddness therefore holds at every parameter value of our model, and the symmetry is a property of the architecture itself.

Figure S6 draws the full readout of this section, from the initial bit-reversed canonical frame to the root readout on top. Notice that many diferent shapes and tensor ranks flow through it $- \ d _ { u ^ { - } }$ wide carriers, scalar log-scales, the complex pair at the root — yet every q-dependent intermediate, irrespective of its overall dimension, shares one structure on its two leading axes: at level κ it splits into $\hat { N } / 2 ^ { \kappa }$ independent chunks, one per node, and the chunk at node P depends on q only through the $3 \cdot 2 ^ { \kappa }$ Lie coordinates (§ S1.2) of $P { ^ { \circ } \mathrm { s } }$ own subtree. The merge is the only operation that grows this dependence set, uniting two subtrees into one; everything else in the readout leaves it untouched. This is what will allow the energy kernel of § S4.2 to propagate each chunk’s derivatives along its own $3 \cdot 2 ^ { \kappa }$ coordinates and no others, bringing the forward-mode Laplacian to $\mathcal { O } ( \hat { N } ^ { 2 } )$ instead of the $\mathcal { O } ( \hat { N } ^ { 3 } )$ a generic implementation would pay.

## S2.5 Deep reinforcement learning of leaf permutations

In this section we detail our deep reinforcement learning strategy to learn the leaf-to-site map, which we call the router. We will not introduce deep reinforcement learning here from first principles; see [52] for a comprehensive introduction.

The bit-reversed leaf assignment of § S2.4.2 fixes the address geometry, but the map from physical sites to leaf positions is currently the same irrespective of the Hamiltonian interaction data. Indeed, site i goes to leaf bre $\tau _ { K } ( i - 1 )$ , regardless of $( J , h )$ . On systems whose correlation structure aligns with the bit-reversed prior (mainly the short-range and translation-invariant ones), a static assignment is fine. However, for systems whose correlation structure aligns with no fixed assignment, such as frustrated couplings, quenched disorder, topologically non-trivial geometry, or long-range tails, we want the leaf-to-site map to be learnable and to condition on the $( J , h )$ tensors of the Hamiltonian.

We let the leaf-to-site map be a discrete latent variable, and define route as a permutation of the whole slot block,

$$
p \in S _ { \hat { N } } ,\tag{S2.89}
$$

where $S _ { \hat { N } }$ is the symmetric group on the $\hat { N }$ slots, containing physical sites and holes on an equal footing. We emphasise that the holes move, per our earlier discussion on scaling to unseen system sizes. The natural alternative is to constrain routes to be mask-preserving, $m _ { p _ { t } } = m _ { t } ,$ so that padding stays frozen at its canonical slots. But § S2.4.1 made hole placement informative, since a hole carries a context, and where the holes sit shapes every context merge above them to the root readout. The router therefore frees the holes and decodes their positions with the same care as the sites’. At decode step $t = 0$ we restrict the candidate set to physical sites. This is a policy-support convention, it anchors the autoregressive prefix in a physical token before the hole-placement statistics are updated. Freed holes do cost an $( \hat { N } { - } N ) ! { - } \mathrm { f o l d }$ exchange redundancy, as routes that difer only by which hole went where produce the identical routed system. However, our sampler detailed below removes this redundancy exactly at every decode step, by collapsing the interchangeable holes into a single candidate class.

Given a route $p ,$ every per-site and per-bond tensor is relabelled into the route’s frame (including masks for the holes),

$$
\begin{array} { r } { q _ { t } ^ { \prime } : = q _ { p _ { t } } , \quad h _ { t } ^ { \prime } : = h _ { p _ { t } } , \quad J _ { t u } ^ { \prime } : = J _ { p _ { t } , p _ { u } } , \quad \ell _ { t } ^ { \prime } : = \ell _ { p _ { t } } ^ { ( L ) } , \quad e _ { t u } ^ { \prime } : = e _ { p _ { t } , p _ { u } } ^ { ( L ) } , \quad m _ { t } ^ { \prime } : = m _ { p _ { t } } . } \end{array}\tag{S2.90}
$$

Indeed, under mask-preserving routes the mask was a constant of the system, but once real and virtual positions mix, which slots are real is itself route-dependent, and the energy kernel, the sampler, and the carrier gating of $\operatorname { E q . }$ (S2.49) all read the routed mask $m ^ { \prime }$ . Inside the routed frame everything downstream (the leaf contextualizer, leaf builder, merge tree, and root readout) operates on an ordinary already-routed system. Thus each kernel sees $( q ^ { \prime } , h ^ { \prime } , \ell ^ { \prime } , e ^ { \prime } , m ^ { \prime } )$ exactly as if the permutation had been applied at the input, and the route is invisible to the conditional ansatz beyond this relabelling.

Let $\pi _ { \boldsymbol { \theta } } ( { p } \mid h , \ell ^ { ( L ) } , e ^ { ( L ) } , m )$ denote a Hamiltonian-conditioned route policy (parametrised below), and let $\psi _ { \boldsymbol \theta } ( \cdot \mid p )$ be the routed ansatz of $\ S \ S 2 . 3$ and $\ S \ S 2 . 4$ evaluated in the p-frame. The policy returns probabilities, not amplitudes, and a route is drawn afresh at each evaluation, so the object the model represents is not a single wavefunction but an incoherent convex mixture of the routed states, that is, the density matrix

$$
\rho _ { \theta } ( q , q ^ { \prime } ) = \sum _ { p \in S _ { \hat { N } } } \pi _ { \theta } ( p \mid h , \ell ^ { ( L ) } , e ^ { ( L ) } , m ) \frac { \psi _ { \theta } ( q \mid p ) \psi _ { \theta } ^ { * } ( q ^ { \prime } \mid p ) } { Z _ { p } } ,\tag{S2.91}
$$

with $\begin{array} { r } { Z _ { p } = \int | \psi _ { \theta } ( q | p ) | ^ { 2 } } \end{array}$ dq the squared norm of the routed state. The parameters θ are shared between the policy and the conditional ansatz. Its energy is the matching convex combination of the routed states’ Rayleigh quotients, a variational principle over mixed states, but a benign one: the energy is linear in $\rho _ { \theta }$ , so its minimum over the convex set of density matrices is attained at an extreme point, a pure state, namely the ground state. The relaxation therefore costs nothing, and at the optimum $\rho _ { \theta }$ is pure even where the policy keeps its weight spread over routes that realise one and the same physical state. Conditioning only on the trunk’s outputs $( h , \ell ^ { ( L ) } , e ^ { ( L ) } )$ and the slot mask m keeps the learned raw scoring map permutation-equivariant under relabelling of the slot block. The minimumindex representative retained after quotienting fixes a bookkeeping gauge inside each WL class, so the labelled representative route itself need not transform equivariantly; class probabilities and the routed physical state remain invariant when the classes are genuine symmetry orbits. Since the policy never sees $q ,$ each term of the mixture preserves the per-site oddness of § S2.3.1.

The rest of this section sets up the architecture of the policy, and the machinery to train it with policy gradients via a discrete sampling step, a score-function estimator, and some variance-control analysis.

We conclude by characterising optimal symmetrised policies through the automorphism orbits of the routed problem. A policy supported uniformly on one energy-minimising orbit O has entropy log |O|, which is generally far below $\log ( \hat { N } ! )$ . This orbit construction, establishes that a non-uniform optimum exists beyond the guarantees of having a lower bound below the maximum.

The sum in $\operatorname { E q . }$ (S2.91) contains N<sup>ˆ</sup> ! routes and is therefore estimated by sampling. Tractable ancestral sampling and exact log-probabilities come from factorising the policy autoregressively into $\hat { N }$ masked slot picks. Diferentiation through the discrete draw is handled separately by the scorefunction estimator of § S2.5.3.

To that end, let $p _ { < t } : = ( p _ { 0 } , p _ { 1 } , . . . , p _ { t - 1 } )$ denote the prefix of picks taken before step $t ,$ and let $p _ { \le t } : = ( p _ { 0 } , p _ { 1 } , . . . , p _ { t } )$ . Then

$$
\pi _ { \theta } ( p \mid h , \ell ^ { ( L ) } , e ^ { ( L ) } , m ) \ = \ \pi _ { \theta } ( p _ { 0 } \mid h , \ell ^ { ( L ) } , e ^ { ( L ) } , m ) \prod _ { t = 1 } ^ { \tilde { N } - 1 } \pi _ { \theta } ( p _ { t } \mid p _ { < t } , h , \ell ^ { ( L ) } , e ^ { ( L ) } , m ) ,\tag{S2.92}
$$

where this runs over the full slot block, and holes are placed by the same factors that place sites. The first pick anchors the tree, with a physical site at slot $p _ { 0 }$ . If $\mathcal { Q } _ { \mathrm { 0 } } ^ { \mathrm { r e a l } }$ denotes the WL classes after

restricting the empty-prefix candidate set to physical sites and $r ( C )$ is the retained representative of class C, then

$$
\pi _ { \theta } ( p _ { 0 } = r ( C ) \mid h , \ell ^ { ( L ) } , e ^ { ( L ) } , m ) \ = \ \frac { \exp \bigl ( \tilde { \eta } _ { 0 , C } \bigr ) } { \sum _ { D \in \mathcal { Q } _ { 0 } ^ { \mathrm { r e a l } } } \exp \bigl ( \tilde { \eta } _ { 0 , D } \bigr ) } ,\tag{S2.93}
$$

where $\tilde { \eta }$ is the class logit of Eq. (S2.109). We emphasise that the anchor is learned, not uniform. This is because which site the contraction is built around is itself part of the structure the policy should discover. The raw logits at all $\hat { N }$ positions are Hamiltonian-conditioned; the final factor has one available class and therefore contributes the identically zero log-probability and gradient.

## S2.5.1 Hamiltonian-conditioned scoring row

Before quotienting, each factor in Eq. (S2.92) has raw pointer logits $\eta _ { t } \in \mathbb { R } ^ { \hat { N } }$ over the candidate slots, with the empty prefix at $t = 0 ;$ the actual factor is the softmax over the resulting WL-class logits $\tilde { \eta } _ { t }$ The decoder is a pointer network in the sense of Vinyals et al. [139], with one structural diference: a standard pointer decoder represents the placed prefix as a flat sequence, whereas ours materialises the partial merge tree implied by the picks made so far. The logits are produced by a candidate composer, a tree-prefix encoder, a tied dot-product pointer, and a conditional symmetry quotient. We describe them in turn.

(a) Candidate composer. Let $P _ { \ell } : \mathbb { R } ^ { d _ { \ell } }  \mathbb { R } ^ { d _ { h } }$ be a small bias-free linear projector on per-site features from the trunk, as in § S2.2, and let $\phi _ { \mathrm { p r e } } , \phi _ { \mathrm { s u f } } : \mathbb { R } ^ { 2 d _ { e } } \to \mathbb { R } ^ { d _ { h } }$ be independent per-channel SiLU MLPs. Both read the two directed bonds of a pair through $E _ { i u } : = [ e _ { i , u } ^ { ( L ) } , e _ { u , i } ^ { ( L ) } ]$ . Define the projected node embedding at slot $i ,$

$$
\begin{array} { r } { \mathrm { n o d e } _ { i } : = { \cal P } _ { \ell } ( \ell _ { i } ^ { ( L ) } ) . } \end{array}\tag{S2.94}
$$

Let $U _ { t } : = \{ 0 , \ldots , \hat { N } - 1 \} \setminus p _ { < t }$ be the available candidate set, and write $\bar { g } = W _ { \mathrm { g l o b a l } } g + b _ { \mathrm { g l o b a l } }$ for the projection of the fixed forked router global. Define

$$
P _ { t , i } : = \frac { 1 } { \sqrt { \operatorname* { m a x } ( t , 1 ) } } \sum _ { u \in p _ { < \ell } } \phi _ { \mathrm { p r e } } ( E _ { i u } ) , \qquad S _ { t , i } : = \frac { 1 } { \sqrt { \operatorname* { m a x } ( | U _ { t } | - 1 , 1 ) } } \sum _ { v \in U _ { t } \backslash \{ i \} } \phi _ { \mathrm { s u f } } ( E _ { i v } ) .\tag{S2.95}
$$

The implemented composer does not first make a static vocabulary token. Let $T _ { q } : \mathbb { R } ^ { d _ { g } }  \mathbb { R } ^ { 2 5 6 }$ denote the active bottleneck on the raw-global channel, and write the router’s stabilized RMS map as

$$
\mathrm { R M S } _ { \mathrm { r } } ( z ) : = \gamma _ { \mathrm { r m s } } \odot \frac z { \sqrt { \operatorname* { m a x } \{ \overline { { z ^ { 2 } } } , 1 0 ^ { - 2 } \} } } .\tag{S2.96}
$$

Here every occurrence has its own learned scale $\gamma _ { \mathrm { r m s } }$ . The composer RMS-normalises each learned vector stream, leaves the three hole ratios raw, and adds learned afine projections into one row– candidate accumulator,

$$
\begin{array} { r l } & { a _ { i } ^ { ( t ) } = b + W _ { \ell } \mathrm { R M S } _ { \mathrm { r } } ( \ell _ { i } ^ { ( L ) } ) + W _ { g } \mathrm { R M S } _ { \mathrm { r } } ( T _ { g } g ) + W _ { \bar { g } } \mathrm { R M S } _ { \mathrm { r } } ( \bar { g } + \gamma ( t ) ) } \\ & { \qquad + W _ { P } \mathrm { R M S } _ { \mathrm { r } } ( P _ { t , i } ) + W _ { O } \mathrm { R M S } _ { \mathrm { r } } ( O _ { t , i } ) + W _ { S } \mathrm { R M S } _ { \mathrm { r } } ( S _ { t , i } ) } \\ & { \qquad + W _ { H } \mathrm { R M S } _ { \mathrm { r } } ( H _ { t , i } ) + W _ { \rho } \rho _ { t , i } . } \end{array}\tag{S2.97}
$$

where $O _ { t , i }$ is the order-aware prefix stream described below, $H _ { t , i }$ is the ordered hole-prefix stream, and $\rho _ { t , i }$ collects the three remaining-hole ratios. In the reported model one residual candidate FFN acts on $a _ { i } ^ { ( t ) }$ ; an output projection is then added to node<sub>i</sub> to give $x _ { i } ^ { ( t ) } \in \mathbb { R } ^ { d _ { h } }$ . Thus $g$ and $\bar { g }$ are fixed across decode rows, while the root-centred clock $\gamma ( t )$ tells the composer how deep into the sequence it is. The prefix summary $P _ { t , i }$ runs over already-picked slots and the candidate-specific sufix $S _ { t , i }$ runs over every other available slot. Both can be maintained as cumulative scans, so each append at step t costs $\mathcal { O } ( \hat { N } d _ { h } )$ work to update the prefix and sufix summaries at every candidate. Across a complete route this bookkeeping costs $\mathcal { O } ( \hat { N } ^ { 2 } d _ { h } )$ The dense candidate refinement below contributes a cubic core; sampled tree recomputation and the conditional quotient are accounted for separately.

The order-aware stream $O _ { t , i }$ is a second read of the prefix: the candidate-specific edge message from position $s < t$ is weighted by a learned Gaussian filterbank over the dyadic LCA distance between decode positions t and s, so that when a neighbour was placed matters as well as whether. The hole streams track both how many holes remain and where they have been going. Every one of these quantities depends on the current row $t ;$ there is no candidate embedding that is reused unchanged throughout a route.

(b) Tree-prefix encoder. The structural heart of the decoder is that the placed prefix is not summarised as a flat sequence; it is materialised as the thing it really is, a partial merge tree. A learned dyadic merge, whose inputs and combination rule (children, directed sibling edges, dyadic clocks, and the normalised interpolation of Eq. (S2.55) with its own gate $\alpha _ { r } )$ deliberately mirror the physical merge of § S2.4.3, scans the placed tokens bottom-up, with causal per-level 2-FWL refinement and causal level attention, the machinery of § S2.4.4 restricted to the prefix. The prefix [0, t) is then represented by its dyadic cover: the at most $\log _ { 2 } \hat { N }$ complete subtrees given by the binary decomposition of t. Cover segments self-attend, and the candidate tokens cross-attend to the cover through their own pooled candidate-to-segment edges. Four read-only cover-query blocks produce the decode state $y _ { t } \in \mathbb { R } ^ { d _ { h } }$ ; two candidate-to-cover blocks and four post-prefix blocks refine the candidate states before the pointer is applied. The decode state is, in other words, a learned surrogate of the partial wavefunction contraction that the next choice is about to extend. Positional information enters only through structure-aware signals, decode-position clocks, dyadic segment clocks, and the tree-distance attention biases of Eq. (S2.83), and never through absolute slot identities, which is what keeps the raw scoring map equivariant under relabelling of the slot block before the representative gauge is fixed.

The router’s global has two distinct roles. It owns the second copy of the contextualizer stack of § S2.4.3: this copy forks from the same post-trunk value as the physics leg, refreshes through that stack’s per-block updates, and closes with the factored edge reading of $\operatorname { E q } .$ (S2.37) over the route-refined edges. The resulting fixed fork $g$ supplies the two composer channels in Eq. (S2.97). A projection of the same fixed $g$ enters every selected-leaf merge and every cover-query FFN; the same-level 2-FWL and attention refinements receive no additional global tap.

Only after the prefix cover has been formed does the stream acquire a row-specific value,

$$
g ^ { ( t ) } : = \{ { \begin{array} { l l } { g , } & { t = 0 , } \\ { { \mathrm { U p d a t e } } \vert g , { \mathrm { p o o l } } ( g , \mathcal { C } _ { t } ) ) , } & { t > 0 , } \end{array} }\tag{S2.98}
$$

one update of $\operatorname { E q . }$ . (S2.33) reading the complete-subtree roots in $\begin{array} { r } { \mathcal { C } _ { t } ; } \end{array}$ at an empty cover the learned update is masked out and the global remains unchanged. This refreshed value enters only the FFNs of the two candidate-to-cover blocks. It is not fed back into the composer, the query seed, the tree merge, the cover-query blocks, or the four post-prefix blocks. Each position computes $\operatorname { E q }$ . (S2.98) independently from the same fixed $^ { g , }$ so teacher scoring of a stored route and sequential sampling of a fresh one evaluate the same row map (up to floating-point evaluation order).

The attention pattern is defined explicitly as follows. For a query sequence $A = \left( a _ { i } \right)$ , a key–value sequence $B = ( b _ { j } )$ , write $\hat { a } _ { i } = \mathrm { R M S } _ { \mathrm { r } } ( a _ { i } )$ and $\hat { b } _ { j } = \mathrm { R M S } _ { \mathrm { r } } ( b _ { j } )$ for the block’s tokenwise pre-norms. For an additive structural bias $\beta _ { h }$ , a per-head key dimension $d _ { k }$ , and a mask $M \in \{ 0 , - \infty \bar  \} ^ { | A | \times | B | }$ , write

$$
\mathrm { A t t n } _ { h } ( A , B ; \beta _ { h } , M ) : = \mathrm { s o f t m a x } \left( \frac { ( \widehat { A } W _ { Q } ^ { h } ) ( \widehat { B } W _ { K } ^ { h } ) ^ { \top } } { \sqrt { d _ { k } } } + \beta _ { h } + M \right) \widehat { B } W _ { V } ^ { h } .\tag{S2.99}
$$

Equation (S2.99) displays one head; in the residual equations below, Attn denotes the implemented gated multihead collapse followed by the learned bias-free output projection $W _ { O }$ . All projections are learned separately in each block. The important object retained at decode position s is not a static embedding of the selected site, but the raw row-conditioned candidate state $\tilde { v } _ { s }$ . The active tree normalisation maps it through the floored RMS sphere transform before the first merge,

$$
\tilde { v } _ { s } : = x _ { p _ { s } } ^ { ( s ) } , \qquad v _ { s } : = \mathrm { S p h e r e } ( \tilde { v } _ { s } ) : = \frac { \tilde { v } _ { s } } { \sqrt { \operatorname* { m a x } \{ \overline { { \tilde { v } _ { s } ^ { 2 } } , 1 0 ^ { - 4 } \} } } } .\tag{S2.100}
$$

Thus a tree leaf already records the Hamiltonian, the prefix and sufix seen when it was selected, the hole statistics, and the decode clock.

The states $v _ { 0 } , \ldots , v _ { t - 1 }$ are merged bottom-up over aligned dyadic intervals. After every merge level, ordered subtree-pair features undergo a row-causal 2-FWL update, and roots of the same dyadic size attend one another using $\operatorname { E q . }$ (S2.99) with the refined pair features as $\beta$ and the mask $M _ { a b } = 0$ for

$b \leq a$ and −∞ otherwise. This is a same-scale causal refinement, since information can move between complete subtrees without flattening their internal contractions.

Let the binary expansion of t decompose the prefix into its unique set of maximal aligned intervals,

$$
[ 0 , t ) = \bigcup _ { a = 1 } ^ { c _ { t } } I _ { t , a } , \qquad \mathcal { C } _ { t } : = \{ z ( I _ { t , a } ) \} _ { a = 1 } ^ { c _ { t } } , \qquad c _ { t } = \mathrm { p o p c o u n t } ( t ) \leq \lceil \log _ { 2 } \hat { N } \rceil ,\tag{S2.101}
$$

where $z ( I )$ is the root state of the complete subtree on I. The fresh query seed $r _ { t } ^ { ( 0 ) } = \bar { g } + \gamma ( t )$ , with $\gamma ( t )$ the root-centred decode clock, is appended to this cover. Four edge-biased cover blocks then apply Eq. (S2.99) with a deliberately asymmetric mask: every live cover root may read all live cover roots, and the appended query may read the roots and itself, but no root may read the appended query. Padding is masked in both directions. The cover roots generally have diferent dyadic sizes; their all-to-all interaction is causal because every member of $\mathcal { C } _ { t }$ lies strictly inside the already selected prefix. The resulting query

$$
y _ { t } = \mathrm { C o v e r R e a d } _ { \theta } \big ( r _ { t } ^ { ( 0 ) } , \mathcal { C } _ { t } , E _ { t } ^ { \mathrm { c o v e r } } \big )\tag{S2.102}
$$

where $E _ { t } ^ { \mathrm { c o v e r } }$ is the directed root-pair message tensor formed by pooling the physical pair features over the leaves of each ordered pair of cover segments. Thus $y _ { t }$ is a read-only observation of every complete subtree that exactly tiles the current prefix. $\mathrm { A t } ~ t = 0$ the cover is empty and $y _ { 0 }$ is the learned four-block transform of the global–clock seed, which attends itself. In particular, $y _ { t }$ is rebuilt for every t from the complete-subtree roots in that row’s cover.

First, two blocks let each $x _ { i } ^ { ( t ) } , i \in U _ { t }$ , cross-attend to $\mathcal { C } _ { t } ;$ their biases pool the directed candidate– subtree edges over the leaves of each cover segment, and their FFNs alone receive the refreshed $g ^ { ( t ) }$ of Eq. (S2.98). Four post-prefix blocks then repeat the ordered update. Let $X _ { t } ^ { ( 0 ) }$ be the output of the second candidate-to-cover block, set $L _ { \mathrm { p o s t } } = 4 _ { \cdot }$ , and write $\lambda _ { \mathrm { p o s t } } : = L _ { \mathrm { p o s t } } ^ { - 1 / 2 }$ . Then

$$
H _ { t } ^ { ( r ) } = X _ { t } ^ { ( r ) } + \lambda _ { \mathrm { p o s t } } \mathrm { A t t n } _ { \mathrm { h i s t } } ^ { ( r ) } \big ( X _ { t } ^ { ( r ) } , Y _ { < t } ; \beta _ { t } ^ { \mathrm { h i s t } } , M _ { t } ^ { \mathrm { h i s t } } \big ) ,\tag{S2.103}
$$

$$
M _ { t , s } ^ { \mathrm { h i s t } } = \left\{ \begin{array} { l l } { 0 , } & { s < t , } \\ { - \infty , } & { s \geq t , } \end{array} \right.\tag{S2.104}
$$

$$
\begin{array} { r } { \boldsymbol { S } _ { t } ^ { ( r ) } = \boldsymbol { H } _ { t } ^ { ( r ) } + \lambda _ { \mathrm { p o s t } } \mathrm { A t t n } _ { \mathrm { s e l f } } ^ { ( r ) } \big ( \boldsymbol { H } _ { t } ^ { ( r ) } , \boldsymbol { H } _ { t } ^ { ( r ) } ; \beta _ { t } ^ { \mathrm { c a n d } } , \boldsymbol { M } ^ { U _ { t } } \big ) , } \end{array}\tag{S2.105}
$$

$$
X _ { t } ^ { ( r + 1 ) } = S _ { t } ^ { ( r ) } + \lambda _ { \mathrm { p o s t } } \mathrm { F F N } _ { r } \bigl ( \mathrm { R M S } _ { \mathrm { r } } ( S _ { t } ^ { ( r ) } ) \bigr ) , \qquad r = 0 , \dots , L _ { \mathrm { p o s t } } - 1 .\tag{S2.106}
$$

We set $\operatorname { A t t n } ( A , \varnothing ; \beta , M ) : = 0$ . Hence at $t = 0$ the candidate-to-cover and history-attention residual sublayers are identities, while their FFNs and the remaining-candidate self-attention still run. Here $Y _ { < t } = \left( y _ { 0 } , \ldots , y _ { t - 1 } \right)$ ; the strict history mask excludes the current and future queries, while $M ^ { U _ { t } }$ permits all-to-all attention among the still-available candidates and masks every placed slot. The history bias contains the direct bidirectional edge pair between candidate i and the site selected at history position $s ,$ together with a dyadic-LCA term; $\beta _ { t } ^ { \mathrm { c a n d } }$ contains the candidate-pair edge features. Consequently, before the pointer chooses one site, the candidates may compare themselves conditional on the same prefix and on one another. Writing $\bar { x } _ { i } ^ { ( t ) } : = X _ { t , i } ^ { ( L _ { \mathrm { p o s t } } ) }$ , the pointer of Eq. (S2.107) pairs $y _ { t }$ with these refined candidate states.

For Hamilton-Zero, the active depths are four cover-query blocks, two candidate-to-cover blocks, and four post-prefix blocks, at width $d _ { h } = 5 1 2$ with 16 logical heads and a 512-dimensional pointer score. Each repeated stack uses its own depth gain: $\lambda _ { \mathrm { c o v e r } } = 4 ^ { - 1 / 2 } = 1 / 2 , \lambda _ { \mathrm { c a n d } } = 2 ^ { - 1 / 2 } = 1 / \sqrt { 2 } $ and $\lambda _ { \mathrm { p o s t } } = 4 ^ { - 1 / 2 } = 1 / 2$ . There is no tokenwise causal backbone and no key–value attention cache in this model: previous $y _ { s }$ are retained as states and their keys and values are projected afresh in each post block. Only the support of the prefix query is logarithmic in $\hat { N } ;$ the candidate-to-cover and remaining-candidate stages cost $\mathcal { O } ( \hat { N } \log \hat { N } )$ and $\mathcal { O } ( \hat { N } ^ { 2 } )$ per dense row, respectively. Across all $\hat { N }$ rows, the latter makes the dense attention core $\mathcal { O } ( \hat { N } ^ { 3 } ) ;$ ; the conditional WL quotient is accounted for separately. Figure S10 shows the resulting computation.

The two router refinement residuals retain their ordinary pre-norm skip connections. Every named update in the tree and global streams uses Eq. (S2.32), including the route-global pool update of Eq. (S2.98).

(c) Pointer. [139] Let $W _ { q } , W _ { k } \in \mathbb { R } ^ { d _ { h } \times d _ { h } }$ be a single pair of bias-free maps shared across all decode steps, and let $\tau > 0$ be a temperature parameter that sets the exploration scale of the policy. The per-step logit at candidate slot i is the tied dot product of the projected decode state against the projected candidate token,

![](images/33c778d2af5f91dce895bbb1c96f9e19d331b8099696ca86b4ea791fef9fa049.jpg)  
Figure S10: The multiscale prefix query at $t = 7 .$ The selected prefix $p _ { < 7 }$ is decomposed into its canonical dyadic cover $\mathcal { C } _ { 7 } = \{ z [ 0 , 4 ) , z [ 4 , 6 ) , v _ { 6 } \}$ , comprising complete subtrees of four, two, and one leaves. Same-level causal 2-FWL and attention refine the subtree roots during their construction. Four read-only query blocks attend from the conditioned seed $\bar { g } + \gamma ( 7 )$ to the three cover roots, without updating those roots, and produce the query state $y _ { 7 } ,$ , which is retained for later decoding positions. In parallel, each remaining candidate receives a fresh prefix-dependent embedding $x _ { i } ^ { ( 7 ) }$ . Two candidate– cover cross-attention blocks condition these embeddings on $\mathcal { C } _ { 7 } .$ , using the pooled edge bias and the routed global state $g ^ { ( 7 ) }$ . Four post-prefix blocks then apply history cross-attention to $y _ { < 7 } ,$ sufix self attention among the remaining candidates, and a conditioned FFN, producing $\bar { x } _ { i } ^ { ( 7 ) }$ . The tied pointer scores $y _ { 7 }$ against every refined candidate; the conditional symmetry quotient collapses equivalent candidates into Weisfeiler–Leman classes before the Gumbel pick. Here $| \mathcal { C } _ { 7 } | = \mathrm { p o p c o u n t } ( 7 ) = 3$ , so only the support of the read-only prefix query grows logarithmically with prefix length.

$$
\eta _ { t , i } ~ = ~ \frac { 1 } { \tau } \frac { \left( W _ { q } y _ { t } \right) \cdot \left( W _ { k } \bar { x } _ { i } ^ { ( t ) } \right) } { \sqrt { d _ { h } } } ~ + ~ \mathrm { m a s k } _ { t , i } ,\tag{S2.107}
$$

with the availability mask

$$
\mathrm { m a s k } _ { t , i } ~ = ~ \left\{ { \begin{array} { l l } { - \infty } & { { \mathrm { i f ~ } } i \in p _ { < t } , { \mathrm { ~ o r ~ i f ~ } } t = 0 { \mathrm { ~ a n d ~ } } m _ { i } = 0 , } \\ { 0 } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.\tag{S2.108}
$$

The availability mask first removes placed slots, holes remain eligible, except at the physical anchor step, and Eq. (S2.109) then converts the surviving raw pointer logits into the class logits whose softmax is the actual pick distribution. The query projection $W _ { q }$ is initialised at zero, so the raw pointer logits start equal. After the mean quotient of Eq. (S2.109), routing therefore starts uniform over valid WL classes (not necessarily over individual slots), and every early preference is learned rather than inherited from the initialisation.

Figure S11 shows the complete pipeline.

(d) Symmetry quotient. Before any sampling, each step’s logits are collapsed over WL colour classes of the remaining candidates. The pair refinement is the second-order update the trunk’s edge block already realises (Eqs. (S2.35)–(S2.36)): colours live on ordered pairs of vertices of the coloured coupling graph and update from the multiset of two-leg paths through every third vertex, run here with the placed prefix individualised, and vertex colours are read of the diagonal. Prefix-preserving automorphisms leave these colours invariant, so candidates in the same exact orbit cannot be separated; the converse is not a theorem for 2-WL on arbitrary graphs. At the empty prefix, the converged colour classes matched the exact orbits on all 247 verifiable systems in the checked panel; prefix-conditioned checks additionally matched on a 20-system chain-and-ladder spot sample. The per-system dispatch uses the cheaper first-order refinement when its empty-prefix partition agrees with the pair refinement and the system metadata permits ${ \mathrm { i t } } ,$ and uses the pair version elsewhere. Let $\mathcal { Q } _ { t }$ be the resulting partition of $U _ { t } ,$ and let $C _ { t } ( i ) \in \mathcal { Q } _ { t }$ contain candidate i. Each class keeps a single representative whose logit averages its members,

$$
\tilde { \eta } _ { t , C } \ = \ \log \frac { 1 } { | C | } \sum _ { i \in C } e ^ { \eta _ { t , i } } ,\tag{S2.109}
$$

and the step samples among representatives only (a Gumbel-max draw [140], which is exact ancestral sampling). The mean, rather than a sum, is what stops a class’s probability from scaling with its class size. All hole slots share one class at every step, so the $( { \hat { N } } - N ) !$ hole-exchange redundancy of Eq. (S2.89) is removed here. The same collapse is applied when scoring, so the factors of $\operatorname { E q . }$ (S2.113) are understood over the collapsed logits.

We refresh the route at every training step. The refresh is cheap because nothing downstream is redrawn: sampling the new permutation costs one decoder pass, and the walker population is simply relabelled into the new frame, which is a measure-preserving coordinate change. After this, the route re-equilibrates in a few sampler sweeps (SM § S4, § S4.1) with no dedicated burn-in.

![](images/3d7022b7cd9fc9d87ecab0b15aab9b7853be92fa1d2f754c2d4a1b11dbfd9400.jpg)  
Figure S11: The route decoder (§ S2.5), read from top to bottom. The candidate composer RMSnormalises and adds the learned vector streams with the raw hole ratios, then applies one residual candidate block to obtain the row-dependent tokens $x _ { i } ^ { ( t ) } ( \mathrm { E q s . ~ S 2 . 9 4  – S 2 . 9 7 } )$ . In parallel, the prefix branch materialises the selected leaves as a dyadic tree, extracts its cover, and forms the read-only multiscale query $y _ { t } .$ . Candidates cross-attend to the cover, then each post-prefix block applies history cross-attention, sufix self-attention, and an FFN in that order. The fixed router global conditions the composer, tree, and cover query; the cover-refreshed $g ^ { ( t ) } \ ( \mathrm { E q . }$ (S2.98)) enters only the two candidateto-cover FFNs. Finally, the tied pointer scores $y _ { t }$ against every refined candidate (Eq. S2.107), and the conditional symmetry quotient collapses WL-indistinguishable choices before sampling (Eq. S2.109).

## S2.5.2 Sequential sampling and vectorised teacher scoring

The policy has two execution schedules for one mathematical scoring row. Denote that row map by

$$
\mathcal { D } _ { \theta } \bigl ( p _ { < t } ; h , \ell ^ { ( L ) } , e ^ { ( L ) } , m \bigr ) = \bigl ( y _ { t } , \bar { X } _ { t } , \eta _ { t } \bigr ) ,\tag{S2.110}
$$

where ${ \bar { X } } _ { t }$ contains the fully refined states of the available candidates and $\eta _ { t }$ is obtained by the tied pointer; the symmetry quotient is applied immediately afterwards.

In sequential sampling, the Hamiltonian-dependent projections, directed message tables, and structural attention biases that do not depend on the sampled prefix are computed once. At step $t ,$ the prefix, sufix, order and hole scans form every fresh $\bar { x _ { i } ^ { ( t ) } }$ . The already selected states $\tilde { v } _ { s } = x _ { p _ { s } } ^ { ( s ) }$ are retained and their spherical images $v _ { s } = \mathrm { S p h e r e } ( \tilde { v } _ { s } )$ form the partial tree, its cover gives $y _ { t }$ , and the candidate-to-cover and post-prefix blocks produce ${ \bar { X } } _ { t }$ before the tied pointer produces $\eta _ { t }$ . A quotientaware Gumbel-max draw supplies $p _ { t } .$ , after which the summary scans, raw selected-state history, and query-state history are updated. In Hamilton-Zero the selected tree is recomputed from the retained $\tilde { v } _ { s }$ at the next row. The retained history consists of states, not projected attention keys and values, and thus there is (counter-intuitively) no tokenwise key–value recurrence behind $\operatorname { E q } .$ . (S2.110).

In teacher-forced scoring, a complete stored route makes every prefix known. All step–candidate states are therefore constructed together,

$$
\mathbf { X } : = \big ( x _ { i } ^ { ( t ) } \big ) _ { t , i } \in \mathbb { R } ^ { \hat { N } \times \hat { N } \times d _ { h } } , \qquad \tilde { v } _ { t } = \mathbf { X } _ { t , p _ { t } } , \qquad v _ { t } = \mathrm { S p h e r e } ( \tilde { v } _ { t } ) .\tag{S2.111}
$$

The prefix and sufix terms are vector cumulative sums; the order-aware terms are triangular masked contractions. A single bottom-up tree scan builds every complete-subtree state at every level. For each row t, dynamic indexing gathers precisely the roots in $\mathcal { C } _ { t }$ from Eq. (S2.101); the cover-query, candidate-to-cover, strict-history, remaining-candidate, pointer and quotient operations then run over all rows in parallel with their row-specific masks. In particular, the raw leaf gathered at position t is $\mathbf { X } _ { t , p _ { t } }$ and is then sphered for the tree; it is not a site embedding shared across decode positions.

The two schedules consequently satisfy, in exact arithmetic,

$$
\eta _ { t , i } ^ { \mathrm { s a m p l e } } = \left[ { \mathcal { D } } _ { \theta } ( p _ { < t } ; h , \ell ^ { ( L ) } , e ^ { ( L ) } , m ) \right] _ { \eta , i } = \eta _ { t , i } ^ { \mathrm { t e a c h e r } } , \qquad i \in U _ { t } .\tag{S2.112}
$$

The distinction is computational, sampling evaluates one changing row after another, whereas teacher forcing materialises every changing row and every dyadic cover in one masked batch. Figure S12 displays the two schedules.

![](images/73584bafef7c32e21b5b2335fe2a846374ab4c759a079598ba4e3fc12d920140.jpg)  
Figure S12: Two schedules for the same route scoring row. Left: sequential sampling constructs the dynamic candidates and prefix tree, samples one quotient class, and stores the selected row-conditioned leaf and query state before the next row. Right: teacher forcing constructs all $\hat { N } ^ { 2 }$ dynamic candidate states, gathers and spheres the selected leaf in every row, performs one multilevel tree scan, dynamically gathers every row’s dyadic cover, and evaluates all masked scoring rows together.

For teacher-forced scoring, the route log-probability therefore reads

$$
\log \pi _ { \theta } ( p \mid h , \ell ^ { ( L ) } , e ^ { ( L ) } , m ) = \sum _ { t = 0 } ^ { \hat { N } - 1 } \Bigl [ \tilde { \eta } _ { t , C _ { t } ( p _ { t } ) } - \log \sum _ { C \in \mathcal { Q } _ { t } } e ^ { \tilde { \eta } _ { t , C } } \Bigr ] .\tag{S2.113}
$$

Before detailing the training algorithm, we emphasise here that each route row contributes separately to the summed log-probability and to the Fisher statistics. The second-order tooling of § S4.3 aggregates those contributions over the route-position repeat axis into the shared policy-parameter blocks. This lets the model learn to route $b y$ the same second-order natural gradient that trains the wavefunction itself, without any extra added optimization machinery to calculate such gradients.

## S2.5.3 Training the route policy: the score-function gradient

The route is discrete, so its derivative is estimated with the score-function identity [141], whereas the conditional wavefunction retains the standard VMC derivative. The sampling rule reads hierarchically as,

$$
p \sim \pi _ { \theta } , \qquad q \mid p \sim \rho _ { \theta , p } , \qquad \rho _ { \theta , p } ( q ) : = \frac { | \psi _ { \theta } ( q \mid p ) | ^ { 2 } } { Z _ { \theta , p } } .\tag{S2.114}
$$

Here the fixed Hamiltonian-conditioning arguments of $\pi _ { \theta }$ are suppressed, and $\textstyle Z _ { \theta , p } : = \int | \psi _ { \theta } ( q \mid p ) | ^ { 2 }$ dq is the squared norm that makes $\rho _ { \theta , p }$ a probability density. Thus the route and walker are not statistically independent. They are sampled by separate procedures: the policy first draws $p ,$ after which the MCMC kernel targets the routed density $\rho _ { \boldsymbol { \theta } , \boldsymbol { p } } .$ . The procedures do not share transition steps, but the walker’s target does however depend on the route.

For a fixed route, we can define

$$
E _ { \mathrm { l o c } } ^ { ( p ) } ( q ) : = \frac { ( \hat { H } \psi _ { \boldsymbol \theta } ( \cdot  { | } p ) ) ( q ) } { \psi _ { \boldsymbol \theta } ( q  { | } p ) } , \qquad \bar { E } ^ { ( p ) } : = \mathbb { E } _ { q \sim \rho _ { \boldsymbol \theta , p } } [ E _ { \mathrm { l o c } } ^ { ( p ) } ( q ) ] , \qquad E : = \mathbb { E } _ { p \sim \pi _ { \boldsymbol \theta } } [ \bar { E } ^ { ( p ) } ] .\tag{S2.115}
$$

Diferentiating the two factors gives

$$
\begin{array} { r l } & { \nabla _ { \theta } E = \mathbb { E } _ { p \sim \pi _ { \theta } } \mathbb { E } _ { q \sim \rho _ { \theta , p } } \left[ \left( E _ { \mathrm { l o c } } ^ { ( p ) } ( q ) - \bar { E } ^ { ( p ) } \right) 2 \mathrm { R e } \nabla _ { \theta } \log \psi _ { \theta } ( q \mid p ) \right] } \\ & { \quad \quad \quad \quad + \mathbb { E } _ { p \sim \pi _ { \theta } } \left[ \left( \bar { E } ^ { ( p ) } - E \right) \nabla _ { \theta } \log \pi _ { \theta } ( p ) \right] . } \end{array}\tag{S2.116}
$$

The first line is the ordinary VMC gradient for each routed state and must be centred at that route’s own mean $\bar { E } ^ { ( p ) }$ . Replacing it by the global mean $E$ changes the finite-sample estimator and introduces a systematic cross-route term. The second line is the REINFORCE gradient from the policy-gradient theorem [52]. The following construction supplies robust energy scales and self-normalised importance sampling for this estimator.

Let $e _ { s }$ collect the local energies used by one policy update, over its route and walker indices, and let $B$ be their total count. We first form the batch mean and mean (absolute) total variation (TV),

$$
\mu _ { 1 } : = \frac { 1 } { B } \sum _ { s } e _ { s } , \qquad \mathrm { T V } _ { 1 } : = \frac { 1 } { B } \sum _ { s } | e _ { s } - \mu _ { 1 } | .\tag{S2.117}
$$

We then clip before estimating the standard deviation,

$$
\begin{array} { l } { { \displaystyle e _ { s } ^ { \mathrm { c l i p } } : = \mathrm { c l i p } ( e _ { s } , \mu _ { 1 } - 5 \mathrm { T V } _ { 1 } , \mu _ { 1 } + 5 \mathrm { T V } _ { 1 } ) } , \ ~ } \\ { { \displaystyle { \bar { e } ^ { \mathrm { c l i p } } } : = \frac { 1 } { B } \sum _ { s } e _ { s } ^ { \mathrm { c l i p } } , \qquad \sigma _ { 1 } : = \sqrt { \frac { 1 } { B } \sum _ { s } \bigl ( e _ { s } ^ { \mathrm { c l i p } } - { \bar { e } ^ { \mathrm { c l i p } } } \bigr ) ^ { 2 } + \varepsilon _ { \sigma } } . } } \end{array}\tag{S2.118}
$$

Here $\mathrm { c l i p } ( x , a , b )$ clamps $x$ to $[ a , b ]$ , and $\varepsilon _ { \sigma } > 0$ is the numerical floor under the variance. We use the shared normalised reward $R _ { s } : = - ( e _ { s } ^ { \mathrm { c l i p } } - \bar { e } ^ { \mathrm { c l i p } } ) / \sigma _ { 1 }$ . We write $R ( p , q )$ for this same normalised reward evaluated at walker $q$ under route p. The clipping therefore protects the scale estimate itself (since $\sigma _ { 1 }$ is not computed from the unclipped tail).

Beam search is used to approximate the policy mode,

$$
p _ { 0 } : = \arg \operatorname* { m a x } _ { p \in \mathcal { B } _ { \mathrm { b e a m } } } \log \pi _ { \theta } ( p ) .\tag{S2.119}
$$

Here $B _ { \mathrm { b e a m } }$ is the set of complete routes retained by the beam search. For a sampled action $p _ { i }$ with $B _ { w }$ walkers $q _ { i b } \sim \rho _ { \theta , p _ { i } }$ , indexed by $b = 1 , \dots , B _ { w }$ , the mode expectation is estimated on those walkers with self-normalised weights,

$$
w _ { i b } : = \frac { | \psi _ { \boldsymbol \theta } ( q _ { i b } \mid p _ { 0 } ) | ^ { 2 } } { | \psi _ { \boldsymbol \theta } ( q _ { i b } \mid p _ { i } ) | ^ { 2 } } , \qquad \bar { w } _ { i b } : = \frac { w _ { i b } } { \sum _ { b ^ { \prime } = 1 } ^ { B _ { w } } w _ { i b ^ { \prime } } } , \qquad \widehat b _ { i } ^ { \mathrm { m o d e } } : = \sum _ { b = 1 } ^ { B _ { w } } \bar { w } _ { i b } R ( p _ { 0 } , q _ { i b } ) .\tag{S2.120}
$$

The unknown normalising constants cancel under self-normalisation. The estimator depends on the sampled action through the proposal density in the denominator. It is biased at finite $B _ { w }$ , but it is consistent and asymptotically unbiased as $B _ { w } \ \to \ \infty$ under the usual support and finite-weight conditions.

With K the number of sampled routes in the policy update, $\begin{array} { r } { \bar { R } _ { i } : = { B _ { w } ^ { - 1 } } \sum _ { b } R ( p _ { i } , q _ { i b } ) } \end{array}$ , and $J _ { \mathrm { r o u t e } } : =$ $\mathbb { E } _ { p \sim \pi _ { \theta } } \mathbb { E } _ { q \sim \rho _ { \theta , p } } [ R ( p , q ) ]$ the expected routed reward, the policy channel is estimated by

$$
\widehat { \nabla _ { \theta } J } _ { \mathrm { r o u t e } } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \Bigl ( \bar { R } _ { i } - \widehat { b } _ { i } ^ { \mathrm { m o d e } } \Bigr ) \nabla _ { \theta } \log \pi _ { \theta } ( p _ { i } ) .\tag{S2.121}
$$

All log $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { p } _ { i } )$ values are evaluated by the teacher-forced autoregressive scoring pass of $\operatorname { E q } .$ (S2.113). Cost. Table S1 gives the compute distribution per system per training step.
<table><tr><td>Operation</td><td>When</td><td>Cost</td></tr><tr><td>ancestral dense candidate core</td><td>K routes</td><td> $\overline { { \mathcal { O } ( K \hat { N } ^ { 3 } d _ { h } ) } }$  arithmetic</td></tr><tr><td>sampled prefix-tree recomputa- K routes tion</td><td></td><td> $\mathcal { O } ( K \hat { N } ^ { 4 } d _ { e } )$  in slot count</td></tr><tr><td>beam baseline, width  $K _ { \mathrm { b e a m } }$ </td><td>every step</td><td> $K _ { \mathrm { b e a m } }$  times the sampled-route costs</td></tr><tr><td>teacher scoring, dense  $_ \mathrm { c o r e } ~ +$  one tree</td><td>K routes</td><td> $\mathcal { O } ( K \hat { N } ^ { 3 } ( d _ { h } + d _ { e } ) )$  arithmetic, batched</td></tr><tr><td>conditional quotient, one route every row row</td><td></td><td> $\mathcal { O } ( R \hat { N } ^ { 3 } ) ~ ( \mathrm { 1 - W L } ) ; \mathcal { O } ( R \hat { N } ^ { 4 } ) ~ ( \mathrm { 2 - F W L } )$ </td></tr><tr><td>advantage + policy KFAC fac- every step tor</td><td></td><td> $\mathcal { O } ( | \mathrm { p o l i c y ~ p a r a m s } | )$ </td></tr></table>

Table S1: Per-step cost of the route policy machinery. $\hat { N }$ is the routed slot count; $d _ { h }$ and $d _ { e }$ are fixed feature widths; $K = 8$ is the per-physical-system route count, and $K _ { \mathrm { b e a m } } = 1 6$ the beam width in the reported checkpoint. The repeated sampled-tree bound comes from rerunning a cubic same-level $2 { \cdot } \mathrm { F W L }$ tree scan at each route row. In the quotient row, R is the refinement-round limit $( R = \hat { N }$ by default); its per-row cost must additionally be multiplied by the route rows and routes actually evaluated. The compiled checkpoint maps the eight sampled routes to separate lanes, which changes elapsed critical path but not total arithmetic.

The teacher schedule does not reduce the arithmetic order: it exposes the $K \hat { N }$ scoring rows to masked batched kernels. Sampling and beam search remain sequential in route position, whereas the teacher-forced scoring that carries the policy gradient is vectorised over rows and routes.

## S2.5.4 An entropy floor for the optimal route policy

We close the routing story with an aside. Nothing in this subsubsection is needed to build or train the architecture. We include it because it is a sharp theoretical observation about what routing can achieve, and because it doubles as a powerful sanity check on the implementation.

We characterise the invariant policies through automorphism orbits and identify the minimum entropy among invariant policies supported on energy-minimising routes. Here we will show that the result depends on both route-orbit sizes, their stabilisers, and the group size.

Define the Hamiltonian’s automorphism group as the permutations of the slot block that preserve $J , h .$ and m simultaneously:

$$
\mathrm { A u t } ( J , h , m ) \ : = \ \Big \{ g \in S _ { \hat { N } } \ : \ J _ { g \cdot t , g \cdot u } = J _ { t , u } , \ h _ { g \cdot t } = h _ { t } , \ m _ { g \cdot t } = m _ { t } \ \forall t , u \Big \} .\tag{S2.122}
$$

$\operatorname { A u t } ( J , h , m )$ is finite and always contains the identity. Notice that it also always contains the $( \hat { N } - N ) !$ permutations of the holes among themselves: hole rows of J and h are zero and the mask is preserved, so exchanging holes changes nothing the network can see. On top of this padding symmetry sit the physical ones. For a Heisenberg ring, $\operatorname { A u t } ( J , h , m )$ contains the lattice translations and reflections; for an irregular disordered system, it contains nothing more than the hole exchanges.

Gauge equivalence under automorphism. For every $g \in \mathrm { A u t } ( J , h , m )$ the routed wavefunction satisfies

$$
\psi _ { \boldsymbol \theta } ( q \mid p ) = \psi _ { \boldsymbol \theta } ( g \cdot q \mid g \cdot p ) \cdot e ^ { i \varphi ( g ) } ,\tag{S2.123}
$$

up to a global phase $\varphi ( g )$ that depends only on $g .$ The proof is by induction up the readout of $\ S \ S 2 . 4 \colon$ every Hamiltonian-side input, contexts, edges, clocks, masks, is built equivariantly from $( J , h , m )$ , which g preserves, so the carrier path evaluates the same contraction on relabelled inputs. Consequently the routes p and $g \cdot p$ yield the same observable energy at every walker: $E _ { \mathrm { l o c } } ^ { ( p ) } ( q ) = E _ { \mathrm { l o c } } ^ { ( g \cdot p ) } ( g \cdot q )$

Optimal policy invariance. Among unbiased policies, the variance-minimising one is invariant under the $\operatorname { A u t } ( J , h , m )$ action:

$$
\pi _ { \theta } ^ { * } ( p ) = \pi _ { \theta } ^ { * } ( g \cdot p ) \quad \forall g \in \mathrm { A u t } ( J , h , m ) .\tag{S2.124}
$$

The argument is symmetrisation. Take any non-invariant optimum ν and average it over the group, $\begin{array} { r } { \bar { \nu } ( p ) : = | \mathrm { A u t } ( J , h , \dot { m } ) | ^ { - 1 } \sum _ { a } \nu ( g ^ { - 1 } \cdot p ) } \end{array}$ . By (S2.123) the average has the same expected reward, and by Jensen’s inequality on the variance functional it has weakly smaller gradient variance.

The policy entropy decomposes over route orbits as follows. Let $G : = \operatorname { A u t } ( J , h , m )$ act on the route set, and let $\{ \mathcal { O } _ { k } \}$ be its route orbits. Write $o _ { k } : = | \mathcal { O } _ { k } |$ and let $\omega _ { k }$ be the total policy mass assigned to orbit k. A G-invariant policy is uniform within each occupied orbit, so, with $\begin{array} { r } { H [ \pi ] : = - \sum _ { p } \pi ( p ) } \end{array}$ log π(p) and $\begin{array} { r } { H ( \omega ) : = - \sum _ { k } } \end{array}$ ω<sub>k</sub> log ω<sub>k</sub> denoting the corresponding Shannon entropies,

$$
H [ \pi ] = - \sum _ { k } \omega _ { k } \log { \frac { \omega _ { k } } { o _ { k } } } = H ( \omega ) + \sum _ { k } \omega _ { k } \log o _ { k } .\tag{S2.125}
$$

By orbit–stabiliser,

$$
o _ { k } = \frac { | G | } { | \operatorname { S t a b } _ { G } ( p _ { k } ) | } , \qquad p _ { k } \in \mathcal { O } _ { k } .\tag{S2.126}
$$

where ${ \mathrm { S t a b } } _ { G } ( p _ { k } ) : = \{ g \in G : g \cdot p _ { k } = p _ { k } \}$ is the stabiliser subgroup of the representative route $p _ { k }$ . Let $\mathfrak { D } _ { \tau }$ be the set of orbits containing energy-minimising routes. Because the energy objective is linear in the policy, there exists an optimal G-invariant policy supported uniformly on one orbi $\mathcal { O } _ { \star } \in \mathcal { D } _ { \star }$ Among such optimal invariant policies, the minimum entropy is

$$
H _ { \star } ( J , h , m ) = \operatorname* { m i n } _ { \mathcal { O } \in \mathfrak { D } _ { \star } } \log | \mathcal { O } | ,\tag{S2.127}
$$

attained on a smallest energy-minimising orbit. Spreading mass across several optimal orbits adds the nonnegative mixing entropy $H ( \omega )$

Equation (S2.127) is an orbit-specific optimum, not the universal quantity log |G| unless the action on the relevant route is free, meaning that its stabiliser is the identity subgroup. Whenever $| \mathcal { O } _ { \star } | < \hat { N } !$ the explicit single-orbit construction yields a non-uniform optimum; other optima may be uniform.

## S3 Datasets and further detail on training

We pretrain the foundation ansatz on a corpus of quadratic spin Hamiltonians, spanning a century of many body literature [54–99]. In this section, we detail the core composition of this dataset in Sec. S3.1, and then walk through in detail the augmented training and per-round perturbation and held-out evaluation datasets that were mentioned in the main text.

The pretraining data uses a quality-balanced mixture rather than empirical family frequencies. Rare regimes are over-represented, and each system’s neighbourhood is densified by the perturbative cover as training proceeds. During pretraining, curvature damping is set as low as the natural-gradient norm allows (SM § S4). Model selection uses a held-out out-of-distribution dataset, followed by evaluation on a separate unseen dataset.

## S3.1 Pretraining data core composition

Our pretraining data’s core is a finite set of quadratic spin Hamiltonians $\{ ( J ^ { ( s ) } , h ^ { ( s ) } ) \} _ { s = 1 } ^ { S }$ in the Paulipair representation of SM § S2. There J is the $N \times N \times 3 \times 3$ bond tensor carrying every Pauli-pair channel, and h is the per-site field. We group the systems first by where they commonly arise in the physics literature and refer to each group as a cell. Within a cell the Hamiltonians range over size, interaction graph, coupling channel, field and anisotropy.

Construction is deterministic. A parametrised builder emits every system from a seed committed to the repository, and the exact systems can be rebuilt from source.

The systems are grouped into physics cells, with three further coverage tiers<sup>3</sup>. The defining property of each cell’s Hamiltonians are briefly given below.

• Canonical: Heisenberg, XXZ, transverse-field Ising and Majumdar–Ghosh on regular lattices, through their exactly solvable points.

<table><tr><td>Cell</td><td>Systems</td></tr><tr><td></td><td>Canonical 1051</td></tr><tr><td></td><td>SYK &amp; random graphs 396</td></tr><tr><td></td><td>Frustration &amp; spin liquids 394</td></tr><tr><td></td><td>Topological &amp; SPT 322</td></tr><tr><td></td><td>Cross-channel J 297</td></tr><tr><td></td><td>Hardware-platform-native 391</td></tr><tr><td></td><td>Disorder &amp; MBL 215</td></tr><tr><td></td><td>Quantum many-body scars 509</td></tr><tr><td>Critical-point fans</td><td>153</td></tr><tr><td>Extremal entanglement</td><td>86</td></tr><tr><td></td><td>Quantum chemistry &amp; Hubbard 79</td></tr><tr><td>Lattice gauge</td><td>42</td></tr><tr><td>Continuous symmetry &amp; Goldstone</td><td>17</td></tr><tr><td colspan="2">Coverage tiers</td></tr><tr><td></td><td>Small-N random cover 664</td></tr><tr><td>Simple systems</td><td>287</td></tr><tr><td></td><td>WL-1-breaking 97 5000</td></tr><tr><td colspan="2">Total</td></tr></table>

![](images/73b94980a0e30817880610bda360b93ec8898617475b15996cafbf4fff1566ef.jpg)  
Figure S13: Composition and system-size distribution of the 5,000-system pretraining dataset Left: the number of systems contributed by each physics cell and coverage tier. Right: the number of systems at each spin count N, stacked by cell. Colours match the table colours and follow the table order from the bottom of each bar upwards.

• Topological / SPT: SSH, Rice–Mele, the Kitaev chain and Haldane, cluster and CZX proxies across topological transitions.

• Critical-point fans: non-Ising quantum critical points, the 2D XXZ, J–Q and Ising–Dzyaloshinskii– Moriya families.

• Hardware-native: Hamiltonians native to superconducting, trapped-ion, Rydberg, spin-qubit and molecular platforms.

• Frustration / spin liquids: kagome, triangular, Kitaev-honeycomb, $J _ { 1 } { - } J _ { 2 }$ and Shastry–Sutherland magnets.

• Disorder / MBL: random-field and random-bond chains across the many-body-localisation transition.

• Extremal entanglement: volume-law ground states, from random Hermitians to the Lipkin– Meshkov–Glick, Page and GHZ parents.

• Lattice gauge: $\mathbb { Z } _ { 2 }$ and U(1) gauge theories in two-body form.

• Quantum chemistry / Hubbard: Fermi–Hubbard and molecular Hamiltonians through the Jordan– Wigner map.

• Goldstone: gapless symmetry-broken magnets with long-wavelength modes.

• Many-body scars: athermal eigenstate towers, the PXP, AKLT and η-pairing families.

• SYK / random graphs: Hamiltonians with no lattice structure, dense and sparse SYK and random-graph couplings.

• Cross-channel J: of-diagonal Pauli couplings, the Dzyaloshinskii–Moriya, compass and perbond-random families.

• Coverage tiers: an easy low-correlation band, a space-filling random-Hamiltonian cover of the small-N regime, and graphs that defeat the routing quotient of SM § S2.

Fig. S13 reports how many systems each cell contributes to the dataset.

## S3.2 Augmented training by per-round perturbation

Every datapoint in the pretrain set is a single set of edge values on one interaction graph $( J , h )$ . Trained on that point alone, the model can overfit to the exact edge values rather than the physics they carry. Therefore in our pretraining, we augment the training data with a perturbation routine that we detail in this subsection.

Each round, we perturb the edge values of every system independently, so the optimiser sees a neighbourhood of values around each datapoint and averages over it instead of fitting one point. A round is one such perturbed version of the whole training dataset, swapped in at an epoch end. Round 0 is an unperturbed base, and the run settles back on it after having trained for a few epochs on the perturbed data.

For each structured system and round, a subroutine draws one of five mutations with equal probability: (i) leave the system unchanged, (ii) add one bond, (iii) perturb one existing bond, (iv) remove one bond, or (v) add one orbit-symmetric random field. Sites, bonds, and orbits are sampled stratified over the Weisfeiler–Leman orbit partition of $( J , h )$ [142, 143]. In case (v), one random three-vector of magnitude $\eta s ,$ , with $\eta \sim \mathcal { U } ( 0 . 2 5 , 1 )$ and s the system’s coupling scale, is added to every site in the selected orbit. For small random systems with $N \leq 8$ , the worker additionally densifies towards all-to-all coupling; a field-free system receives a substantial random dense field with probability 1/2, together with occasional bond drops.

Speculative re-gauging is evaluated after the MCMC update, when the routed walker population is already available. It applies

$$
J _ { i j } \mapsto R _ { i } J _ { i j } R _ { j } ^ { \top } , \qquad h _ { i } \mapsto R _ { i } h _ { i } , \qquad q _ { i } \mapsto q _ { i } \bar { u } _ { i } ,\tag{S3.1}
$$

with one $u _ { i }$ per Weisfeiler–Leman orbit, $\bar { u } _ { i } : = u _ { i } ^ { - 1 }$ , and $R _ { i } : = R _ { \mathrm { k e r } } ( u _ { i } ) \in \mathrm { S O } ( 3 )$ the adjoint rotation expressed in the kernel frame. The transformed Hamiltonian is unitarily equivalent to the original and the transformed walkers target its routed density exactly.

We emphasise that a round difers from the base in the values of (J, h) and nothing else. As such, the perturbation is cheap, because it changes only the values in (J, h), meaning shapes remain static, and thus we avoid any recompiling.

To make sure none of this stalls training, we make the routine CPU-based such that it executes the perturbation run asynchronously. The routine stays 2 rounds in front of the inner training loop. For each upcoming round it perturbs the pretrain dataset and precomputes the reference data, the exact-diagonalisation energies to $N \leq 2 2$ and the per-system routing fields, so the next round is ready before the loop reaches it. The loop consumes one round per epoch and, at each epoch boundary, swaps in the next round if the routine has produced it. The look-ahead is the margin that absorbs the time the perturbation and the diagonalisations take, so the training loop has zero latency with respect to this routine. Figure S14 shows how the two processes run asynchronously with zero latency.

![](images/3c33b41a52960b87a49da052343c04989b0d8575c2f85f597012a44279f22180.jpg)  
Figure S14: Augmented training and asynchronous hot-swap. The CPU samples one of five system perturbations, builds its references and routing fields 2 rounds ahead, and caches the result. At each epoch end the GPU swaps in the next cached round. Post-MCMC speculative re-gauging reuses the current walkers and adds no measured overhead to the inner loop.

The cover is local. Each round perturbs a system within a neighbourhood of its base, so augmentation fills in the surroundings of the curated physics rather than reaching for new phases. The critical points the model must reproduce are placed in the pretraining data itself, in the fans of § S3.1, since they are not reached by the perturbation in general.

The exact diagonalisations the worker computes are a second payof. Training is self-supervised and needs no reference energies, so they serve only as a convergence diagnostic, and because each is valid only within its own round, the gap to exact is read per round. Yet each round contributes 2882 of them at $N \leq 2 2$ , and across a run they accumulate into a corpus of tens of thousands of exactly solved systems, a byproduct of the cover rather than a cost of it.

## S3.3 Evaluation and fine-tuning datasets

In this section, we describe the three datasets we used to investigate how well Hamilton-Zero generalises to unseen interaction topologies and interaction types. To do this, we kept several types Hamiltonian families out of training, using them only here for evaluation and fine-tuning.

First in the set of three is Hamiltonian’s corresponding to combinatorial optimisation problems. These are Hamiltonians whose bonding structure stays inside the training distribution (graph Ising, $\begin{array} { r } { H = \sum _ { ( i , j ) \in E } J _ { i j } \sigma _ { i } ^ { z } \sigma _ { j } ^ { z } + \sum _ { i } h _ { i } \sigma _ { i } ^ { z } ) } \end{array}$ but whose problem semantics and interaction topologies are absent from training. Every Hamiltonian in this set encodes an NP-hard combinatorial optimisation problem whose ground state is the problem’s optimal solution. We emphasise that unlike many systems in training and the other evaluation sets, such solutins correspond to separable ground states, so the dificulty here not in the entanglement of ground states, but rather their single-direction support in an otherwise exponentially large Hilbert space. The sub-families are listed in Table S2.

<table><tr><td>Sub-family</td><td>#cells</td><td>Graph / structure</td></tr><tr><td>MaxCut on random regular graphs</td><td>20</td><td>k-regular  $\overline { { ( k \in \{ 3 , 5 \} ) } }$  , BA scale-free, planted-cut</td></tr><tr><td>MaxCut on Erdős-Rényi</td><td>8</td><td>Density  $p \in \{ 0 . 4 , 0 . 6 \}$ </td></tr><tr><td>QUBO (dense + sparse)</td><td>16</td><td> $J _ { i j } \in \{ \pm 1 \} , h _ { i } \sim { \mathcal { N } } ( 0 , 1 ) ;$  dense  $K _ { N }$  or sparse-ER</td></tr><tr><td>Number partitioning</td><td>10</td><td>Rank-1 J = v vT, vi ∼ Uniform or power-law</td></tr><tr><td>Max-2-SAT</td><td>10</td><td>Random clauses at clause-to-variable ratio α ∈ {0.8, 2}</td></tr><tr><td>Max independent set / min vertex cover</td><td>14</td><td>Penalty form  $\begin{array} { r } { H = \sum _ { v } - \sigma _ { v } ^ { z } + \lambda \sum _ { ( i , j ) \in E } ( \sigma _ { i } ^ { z } + 1 ) ( \sigma _ { j } ^ { z } + 1 ) / 4 } \end{array}$ </td></tr><tr><td>Graph 2-colouring (adversarial)</td><td>8</td><td>5-regular graphs with many odd cycles</td></tr><tr><td>Hamiltonian-path proxy</td><td>6</td><td>2-body approximation of the 4-body Lucas encoding</td></tr></table>

Table S2: Table of systems for the first evaluation set

The second evaluation set covers unseen topologies and interaction types, including fractal, quasiperiodic, and higher-dimensional interaction geometries such as hypercubes. Its subfamilies are listed in Table S3.

<table><tr><td>Sub-family</td><td>#cells Graph</td><td></td></tr><tr><td>Fractal: Sierpinski + T-fractal</td><td>14</td><td>Hausdorff dim log 3/log 2 ≈ 1.585 (Sierpinski); branching tree (T-fractal)</td></tr><tr><td>Penrose / Fibonacci quasi-periodic</td><td>10</td><td>Fibonacci chain ± pentagon ring</td></tr><tr><td>Decorated lattices</td><td>16</td><td>Lieb (flat band), dice (T3, AB caging), square-octagon (4.8.8)</td></tr><tr><td>3D high-coordination</td><td>10</td><td>FCC (frustrated), BCC (bipartite), diamond</td></tr><tr><td>4D hypercube N=16</td><td>8  $K _ { 2 } ^ { 4 }$ </td><td> AFM / FM / XXZ / TFIM</td></tr><tr><td>5D hypercube N=32</td><td>4  $K _ { 2 } ^ { 5 }$ </td><td>, all out of scope for ED; the headline extrapolation cell</td></tr><tr><td>Hyperbolic {7, 3}</td><td>10 ED)</td><td>Heptagon  $r = 1 + r = 2$  patches (2 systems out of scope for</td></tr><tr><td>Möbius / complete-bipartite</td><td>7</td><td>Möbius-twisted chain,  $K _ { a , b }$ </td></tr></table>

Table S3: The second evaluation set containing unseen interaction topology and unseen interaction channels. The training dataset has no fractal, no quasi-periodic-2D, no decorated, no 4D/5D hypercube, and no hyperbolic lattices.

The third evaluation set contains Hamiltonians with physical structures absent from training: multi-spin-1/2-per-physical-site encodings, complex hoppings, and a gauge theory. Its subfamilies are listed in Table S4.

<table><tr><td>Sub-family</td><td>#cells Encoding</td><td></td></tr><tr><td>SU(3) ULS chain (PROXY)</td><td>14</td><td>2 spin-1/2 per physical site; FM intra-bond → S=1; Heisenberg + ZZ-boost approximation of the bilinear-biquadratic spin-1 chain.</td></tr><tr><td>Kugel-Khomskii spin- orbital (PROXY)</td><td>14</td><td>2 spin-1/2 per site (spin + orbital); factorised as 2-body S·S + τ·τ+ cross-Ising (PROXY for the native 4-body  $( \mathbf { S } \cdot \mathbf { S } ) ( \pmb { \tau } \cdot \pmb { \tau } ) )$ </td></tr><tr><td>Hofstadter strip</td><td>24</td><td>2/3/4-leg ladders with complex hop  $t \to t e ^ { i \phi } \Rightarrow \mathrm { D M } _ { z }$  terms in real spin language; flux α per plaquette (incl. golden-mean quasi- periodic). Native 2-body.</td></tr><tr><td>Spin-1 XXZ chain (encoded)</td><td>12</td><td>Bilinear-only spin-1 chain via 2 spin-1/2; ∆ sweep covers Haldane / AKLT-like / easy-axis. Bethe-ansatz integrable reference points.</td></tr><tr><td>1D Hubbard JW</td><td>10</td><td>Lieb-Wu integrable line U = 4, Mott  $U \ = \ 8 .$  weakly correlated U = 2 (truncated-JW PROXY).</td></tr><tr><td>Schwinger θ-vacuum</td><td>11</td><td>1D U(1) gauge after staggered + JW + Gauss-law;  $\theta \in \{ 0 , \pi / 2 , \pi \}$  sweep; Coleman transition at θ = π. Native 2-body.</td></tr></table>

Table S4: The third and hardest evaluation dataset.

## S3.3.1 Evaluation and fine-tuning configuration

For evaluation, we let the pretrained router select the tree ordering (eight candidate trees decoded at temperature τ = 1 from a width-64 beam, each raced over a short burn-in and measurement window before collapsing to the winner), compile the wavefunction once on that ordering, and sample the fixed model for 512 measurement steps with 256 walkers across eight tempered replicas and 24 adaptive Langevin moves per step, retaining $5 1 2 \times 2 5 6 \times 8 \simeq 1 . 0 5 \times 1 0 ^ { 6 }$ samples per system. Energies are reported with the pooled standard deviation of per-walker means as the uncertainty.

fine-tuning restarts from the same checkpoint but commits the route once, at initialisation, with a width-16 beam search: the wavefunction is compiled on that fixed ordering and the eager model and router are discarded, leaving a compiled ansatz of one shared kernel plus per-spin tree weights (∼4.7M parameters, against 547M for the full model) as the object being trained. Each system is optimised independently on a single A100 for $1 0 ^ { 4 }$ KFAC steps with 256 walkers, four MCMC moves per step and a 128-step burn-in, with learning rate $0 . 0 1 / ( 1 + t / 1 0 ^ { 4 } )$ (ending at $\approx 0 . 0 0 5 )$ , damping $1 0 ^ { - 3 }$ , curvature EMA 0.9 and curvature/inverse updates every $2 / 4$ steps. After training, every system is re-evaluated with frozen weights: the final checkpoint is resumed at exactly zero learning rate for a further 512 measurement steps with the sampler state carried over, the freeze is certified per system by a byte-identical model checkpoint written inside the evaluation window, and the standard error carries an AR(1) autocorrelation correction. ED references are never supplied to evaluation or fine-tuning and enter only in the analysis of the main text.

## S4 Optimization and Engineering

In this section, we detail our state-of-the-art sampling scheme for MCMC on the configuration manifold $S U ( 2 ) ^ { N }$ , as well as our energy kernels, our extension of the KFAC optimizer to the trunk’s stacked weights and the rank-4 merge tensor, all of which we made shard-mappable and GEMM-optimised. We take them in the order the data flows; the sampler produces walker configurations, then energy kernels evaluate the energy and its gradient on them, and finally the KFAC extensions consume both.

## S4.1 MCMC on the SU(2) manifold

Improving the energy and updating the weights both require samples drawn from the model’s own density $| \psi _ { \theta } ( q ) | ^ { 2 }$ on $( \mathrm { S U } ( 2 ) ) ^ { N }$ . The variational energy is the loss, and its gradient drives the optimiser, so the variance of these Monte Carlo estimates sets how noisy each training step is. That variance grows with the chain’s integrated autocorrelation time and shrinks with the number of samples, so a sampler that decorrelates faster reaches a given accuracy from fewer samples. The sampler is therefore worth as much engineering as the energy kernel. Ours goes well beyond the Metropolis–Hastings (MH) algorithm [144, 145] that variational Monte Carlo conventionally uses, whose limitations for electronic-structure sampling are analysed in [146].

MH algorithm proposes a move $q  q ^ { \prime }$ from a kernel $g ( q ^ { \prime } \mid q )$ and accepts it with probability min 1, $| \psi _ { \theta } ( q ^ { \prime } ) | ^ { 2 } g ( \bar { q } \mid \bar { q ^ { \prime } } ) / ( | \psi _ { \theta } ( q ) | ^ { 2 } g ( \bar { q ^ { \prime } } \mid q ) ) \bar { ) }$ . The usual random-walk proposal forces a bad trade-of in high dimensions. Large steps land in low-density regions and are almost all rejected; small steps are accepted but barely move. Either way, the chain decorrelates slowly, the autocorrelation time grows, and the efective sample size collapses [147].

Langevin dynamics breaks the trade-of by drifting the proposal up the density with gradient ascent. The Langevin difusion

$$
\mathrm { d } q _ { t } = \nabla \log | \psi _ { \boldsymbol { \theta } } ( q _ { t } ) | ^ { 2 } \mathrm { d } t + \sqrt { 2 } \mathrm { d } W _ { t }\tag{S4.1}
$$

has $| \psi _ { \theta } | ^ { 2 }$ as its stationary density, but a finite discretisation of it does not. A single Euler–Maruyama step of size σ gives the unadjusted Langevin proposal, and because it only approximates the difusion, it samples a perturbed density whose bias away from $| \psi _ { \theta } | ^ { 2 }$ vanishes only as $\sigma  0$ . The Metropolisadjusted Langevin algorithm [148] removes this bias exactly by passing the Langevin proposal through the MH accept step, and so keeps the drift’s fast mixing while restoring $| \psi _ { \theta } | ^ { 2 }$ as the stationary density. Its optimal step size scales as $\sigma \sim D ^ { - 1 / 3 }$ rather than the random walk’s $\dot { D } ^ { - 1 / 2 }$ [149], a large gain at our dimension.

MALA alone, however, is still not enough. Each rejected move discards a gradient evaluation, which is the most expensive quantity in the step, so reducing σ to cut rejections also shortens the steps and slows exploration. And a single local chain with a single scalar step size mixes slowly across the rugged, multimodal landscape of a frustrated or disordered Hamiltonian, where it can remain in one basin for many steps. The subsections below close these gaps in turn. First, a manifold-aware MALA proposal that respects the group geometry, then a ladder of tempered replicas [150, 151] that exchange to carry walkers between basins, followed by a global Haar-random refresh that erases stuck sites outright, and finally online adaptations that hold each piece at its optimal operating point [152]. Assembled, the scheme cuts the chain’s integrated autocorrelation time, and with it the variance of every energy and gradient estimate, so a target accuracy needs far fewer samples. And it returns batches diferentiable in both the manifold coordinates and the model weights, the two derivatives the energy estimator and the natural-gradient step consume.

Concretely, the sampler runs once per training step, in three stages. A setup stage runs first and once: it hoists the per-system quantities that do not depend on the configuration $q ,$ the featurizer and trunk outputs, and refreshes the cached log-density log $| \psi _ { \theta } | ^ { 2 }$ and its gradient at the current q against the freshly updated model. A single global Haar refresh of m sites then fires, before any local move. Finally, a loop of inter-steps runs, each one a manifold MALA move followed by a cross-chain swap whose edge set alternates with the step parity. Three adaptations run on a slower cadence and condition the next pass: a multiplicative bang-bang rule drives the MALA step size to its optimal acceptance, a proportional rule tunes the number of Haar-refreshed sites to its own optimal acceptance, and a monotone-cubic interpolation re-spaces the temperature ladder to equalise swap rejection. Figure S15 shows this pipeline and where each subsection below fits.

## S4.1.1 Proposal density on the Lie group

A proposal $q  q ^ { \prime }$ on ${ \mathrm { S U } } ( 2 )$ is parameterised by a Lie algebra increment $\xi \in \mathbb { R } ^ { 3 }$ via

$$
q ^ { \prime } = q \cdot \exp \bigl ( \sigma \xi ^ { a } e _ { a } ( q ) \bigr ) ,\tag{S4.2}
$$

where $\{ e _ { a } ( q ) \}$ is the left-invariant frame. The Gaussian density of $\xi$ on the algebra is closed-form, but the induced density on the group is not: it must be wrapped around the periodicity of the exponential map. We use the wrap sum

$$
p _ { \mathrm { w r a p } } ( \xi ; \sigma ) = \sum _ { n \in \mathbb { Z } ^ { 3 } , \| n \| \leq N _ { \mathrm { w r a p } } } \mathcal { N } \big ( \xi + 2 \pi n ; 0 , \sigma ^ { 2 } I _ { 3 } \big ) ,\tag{S4.3}
$$

truncated to $N _ { \mathrm { w r a p } } = 8$ shells in the experiments reported here. For $\sigma < 7 \pi$ , vastly larger than any practical step, the truncation error falls below double-precision noise. The wrap sum costs about 3× a single Gaussian and takes under 0.3% of MCMC wall time.

## S4.1.2 Manifold MALA on $\mathrm { S U } ( 2 ) ^ { N }$

Langevin and Hamiltonian proposals on manifolds have a developed literature [153, 154]; the construction below specialises to the product group $\mathrm { S U } ( 2 ) ^ { N }$ , where the exponential map and its wrapped proposal density are available in closed form. The MALA proposal of the introduction needs two changes on the Lie group. Vanilla addition $q + \xi$ is not a group operation, so we exponentiate a Liealgebra increment back onto the group, $q ^ { \prime } = q \cdot \exp ( \sigma \xi ^ { a } e _ { a } ( q ) )$ , as in the proposal density above. And the Euclidean gradient becomes the left-invariant gradient $\nabla _ { a } \log | \psi _ { \theta } | ^ { 2 } = e _ { a } ( q ) \cdot \nabla$ log $| \psi _ { \theta } | ^ { 2 }$ along the frame $\{ e _ { a } ( q ) \}$ . The manifold-MALA proposal is then

![](images/1c975007d0e183b4e436e2c54a21185dd2eee150e3e6daa85c272c4d5b8417c1.jpg)  
Figure S15: The per-chain MCMC pipeline at inverse temperature $\beta _ { r } ~ \mathrm { ( 8 S 4 . 1 ) }$ , run once per training step. A setup stage runs first and once: it hoists the per-system, q-independent featurizer and trunk outputs, then refreshes the cached log-density and its gradient at the current $q$ (§ S4.1.7). A single global Haar refresh of m sites (§ S4.1.5, dashed border for its stochastic accept) fires before any local move. The inner loop then runs a fixed number of inter-steps (the loop-back on the left), each a manifold MALA move (§ S4.1.2) followed by a deterministic even/odd cross-chain swap (§ S4.1.3) whose edge set alternates with the step parity, producing $q _ { t + 1 } ^ { ( r ) }$ . Three adaptations (magenta, dashed) run on a slower cadence and condition the next pass: the refresh count m is tuned to the Haar acceptance rate (§ S4.1.5), the multiplicative rule of § S4.1.6 drives the MALA step size σ to its acceptance, and the Hyman monotone cubic (§ S4.1.4) re-spaces the inverse-temperature grid $\{ \beta _ { r } \}$ to the pair-rejection rates. All boxes are shades of the leaf-green hue used for the spin-configuration input in Figure S1, darkening from the setup stage along the loop; the colour match marks this figure as the subroutine that produces q.

$$
\begin{array} { r } { \xi \ = \ \frac { \sigma ^ { 2 } } { 2 } \nabla _ { a } \log | \psi _ { \theta } ( q ) | ^ { 2 } \ + \ \sigma \eta , \qquad \eta \sim { \mathcal N } ( 0 , I _ { 3 } ) . } \end{array}\tag{S4.4}
$$

The MH correction now needs both the wrap-sum proposal density (S4.3) and the volume Jacobian of the exponential map. Either correction alone breaks detailed balance, because the wrapped Gaussian is the proposal density on the group while the Jacobian accounts for the change of measure between the algebra and the group.

## S4.1.3 Replica exchange with deterministic even-odd swaps

A single chain, however well tuned, still struggles to cross between well-separated modes, so we run a ladder of them. R chains run in parallel at decreasing inverse temperatures $\beta _ { 1 } > \beta _ { 2 } > \cdot \cdot \cdot > \beta _ { R } ;$ the coldest replica, at $\beta _ { 1 }$ , sits near the modes of $| \psi _ { \theta } | ^ { 2 }$ , while the hottest replica, at $\beta _ { R } = 0$ , explores broadly.

After each local MALA move the chain proposes swaps between adjacent replicas along the deterministic even/odd (DEO) schedule of Syed et al. [155], whose eligible edges alternate with the step parity. At even steps the even edges swap, $( 1 , 2 ) , ( 3 , 4 ) , \ldots ;$ at odd steps the odd edges swap, $( 2 , 3 ) , ( 4 , 5 ) , \ldots ;$ so every adjacent pair is ofered a swap once in every two steps. Each pair accepts with the standard MH ratio,

$$
\begin{array} { r } { \alpha _ { r , r + 1 } = \mathrm { m i n } \Big ( 1 , \mathrm { e x p } \big ( ( \beta _ { r } - \beta _ { r + 1 } ) ( \log | \psi _ { \theta } ( x _ { r + 1 } ) | ^ { 2 } - \log | \psi _ { \theta } ( x _ { r } ) | ^ { 2 } ) \big ) \Big ) . } \end{array}\tag{S4.5}
$$

The alternation is non-reversible. A pair swapped at an even step is not ofered again until the next even step, two steps later; in between, the odd-step swap carries the walker on to its new neighbour rather than back. This directed transport along the ladder is what maximises information flow between the hot and cold chains [155]. Figure S16 shows the swap pattern over a few steps.

![](images/3c15bd7001b1c95ec29d7e5e72d0d6be0bfcba696508401eebe065f9dcef1bfa.jpg)  
Figure S16: Deterministic even/odd (DEO) replica-exchange schedule $( \ S \ S 4 . 1 . 3 )$ . R chains run in parallel at decreasing inverse temperatures $\beta _ { 1 } > \beta _ { 2 } > \cdots > \beta _ { R }$ (rows, cold → hot colour gradient). At each MCMC time step (columns), pairs of adjacent chains are proposed for Metropolis swaps: evenedge pairs (1, 2) and $( 3 , 4 )$ at even t, odd-edge pair (2, 3) at odd t. The alternation is deterministic and exhaustive, every pair firing once in every two steps, and non-reversible in the sense that consecutive swaps cannot trivially undo each other, which maximises information flow along the temperature ladder.

## S4.1.4 Online β-adaptation via monotone cubic interpolation

The ladder only carries walkers if adjacent rungs overlap enough to swap: too coarse a spacing rejects every swap, too fine a spacing wastes replicas. At stationarity the optimal schedule equalises the adjacent-pair rejection rates across the ladder [156]: for a target rate $\bar { \lambda } ,$ every adjacent pair $( r , r + 1 )$ should reject with probability λ<sup>¯</sup>. This is the schedule-optimisation problem that Syed et al. [155] solve through the cumulative rejection (their communication barrier) $\begin{array} { r } { \Lambda ( \beta ) = \int _ { \beta _ { 1 } } ^ { \beta } \lambda ( \beta ^ { \prime } ) | \mathrm { d } \beta ^ { \prime } | } \end{array}$ : the optimal grid divides Λ into equal increments, $\beta _ { r } = \Lambda ^ { - 1 } \big ( ( r - 1 ) \bar { \lambda } \big )$ with $\bar { \lambda } : = \Lambda ( \beta _ { R } ) / ( R - 1 )$ . What we add is the online version: the running chain gives only the R samples $\{ ( \beta _ { r } , \hat { \Lambda } ( \beta _ { r } ) ) \} _ { r } ,$ , so recovering the grid means inverting a monotone interpolant of Λ, refreshed as the model, and with it the rejection profile, drifts during training.

Piecewise cubic Hermite interpolation fixes both the value $f ( \beta _ { r } )$ and the derivative $f ^ { \prime } ( \beta _ { r } )$ at each node and fits each segment with a cubic. The choice of node derivative is what separates the methods. PCHIP, the piecewise cubic Hermite interpolating polynomial of Fritsch and Carlson [157], sets the derivative to zero wherever the local secant slope $\delta _ { r } = ( f ( \beta _ { r + 1 } ) - f ( \beta _ { r } ) ) / ( \beta _ { r + 1 } - \beta _ { r } )$ changes sign and to the harmonic mean of $\delta _ { r - 1 } , \delta _ { r }$ otherwise; it is monotone by construction, but the harmonic mean loses accuracy through a smooth inflection. Hyman’s Hyman [158] derivative is instead the centred secant $f ^ { \prime } ( \beta _ { r } ) = ( f ( \beta _ { r + 1 } ) - f ( \beta _ { r - 1 } ) ) / ( \beta _ { r + 1 } - \beta _ { r - 1 } )$ , which is exact for quadratic data, and applies the Fritsch–Carlson [157] monotonicity clamp only as a safety net at a detected sign change. On the quadratic test $f ( \beta ) \stackrel { \cdot } { = } \beta ^ { 2 }$ Hyman recovers $f ^ { \prime }$ to double precision $( \sim 1 0 ^ { - 1 2 }$ relative error) where PCHIP errs at $\sim 1 0 ^ { - 3 }$ near an inflection. Spin-manifold rejection curves carry smooth inflection regions where that harmonic-mean error, not monotonicity, is the binding constraint, so we use Hyman.

Each adaptation interval then runs four steps. It updates $\hat { \Lambda } ( \beta _ { r } )$ by an exponential moving average over the last $T$ swap attempts at each pair, fits the Hyman cubic to $\{ ( \beta _ { r } , \hat { \Lambda } ( \beta _ { r } ) ) \} _ { i }$ , bisects it to recover the equal-rejection grid $\{ \tilde { \beta } _ { r } \}$ , and blends that grid into the old one, $\beta _ { r }  ( 1 - \mu ) \beta _ { r } + \mu \tilde { \beta } _ { r }$ , for stability.

On synthetic cold-pair-compression benchmarks the end-to-end gain is $+ 1 5 \mathrm { ~ t o ~ } + 1 9 \%$ over a one-sided backward-Hermite baseline.

## S4.1.5 Global m-site Haar refresh and the 0.234 optimum

Local moves and swaps still change the configuration only gradually, so a basin that no replica has entered stays unvisited. A global move sidesteps that. Once per training step, before the local-move loop, the chain refreshes m randomly chosen sites to Haar-uniform values on ${ \mathrm { S U } } ( 2 )$ . This non-local move breaks the within-chain correlation that MALA builds up and reaches disconnected modes that the local Langevin proposal cannot bridge.

The refresh acceptance rate falls as m grows, since a larger refresh lands further from the current state and is accepted less often. We tune m on the same slow cadence as the step size, by a proportional rule: increment it by one when the running Haar acceptance exceeds the target, decrement it by one when it falls short, clipped to $[ 1 , n _ { \mathrm { a c t i v e } } ]$ . A unit step is the right granularity because acceptance is discrete in m and the search space has only $n _ { \mathrm { a c t i v e } }$ candidates, so the optimum is reached quickly. The hottest replica, at $\beta = 0$ , is held at full erasure, $m = n _ { \mathrm { a c t i v e } } ,$ , where every refresh is accepted and the mixing is free.

The target $\alpha ^ { \star }$ is the acceptance that maximises the expected squared-jump distance per move. For a k-site refresh on a compact group the expected squared-jump distance (ESJD) grows linearly in $m ,$ and so does the variance V of the log-density increment, which is a sum of m per-site contributions; the two are therefore proportional, and $\mathbb { E } [ \mathrm { E S J D } ] \propto \alpha V$ . Because the increment is a m-term sum, the central limit theorem makes it approximately Gaussian whenever the site contributions are weakly correlated, and exactly Gaussian when the target factorises over sites, as for the rank-one ferromagnetic ground state. We therefore take

$$
\Delta \log | \psi _ { \boldsymbol \theta } | ^ { 2 } = M + Z , \qquad Z \sim { \mathcal { N } } ( 0 , V ) ,\tag{S4.6}
$$

with mean $M < 0 .$ . The refresh draws its sites independently of the current state, so the proposal is symmetric; detailed balance at stationarity then gives $\mathbb { E } [ e ^ { \Delta } ] = 1$ , which for a Gaussian increment implies

$$
M \ = \ - V / 2 , \mathrm { e q u i v a l e n t l y } V \ = \ - 2 M .\tag{S4.7}
$$

The expected MH acceptance probability is then the truncated Gaussian-tail integral

$$
\alpha ( V ) ~ = ~ \mathbb { E } \big [ \operatorname* { m i n } ( 1 , e ^ { \Delta \log | \psi _ { \theta } | ^ { 2 } } ) \big ] ~ = ~ 2 \Phi \big ( - \sqrt { V } / 2 \big ) ,\tag{S4.8}
$$

where Φ is the standard-normal CDF. The expected squared-jump distance (ESJD) is proportional to $\alpha ( V ) \cdot V$ . Setting $\mathrm { d } / \mathrm { d } V [ \alpha ( V ) V ] = 0$ and solving numerically:

$$
\sqrt { V ^ { \star } } / 2 \approx 1 . 1 9 , \qquad V ^ { \star } \approx 5 . 6 6 , \qquad \alpha ^ { \star } \approx 0 . 2 3 4 .\tag{S4.9}
$$

The resulting optimum, $\alpha ^ { \star } \approx 0 . 2 3 4$ , matches the random-walk Metropolis limit of Roberts et al. [147]. Both derivations reduce to maximising the expected squared-jump distance when the log-density increment is asymptotically Gaussian and stationarity fixes its mean $\mathrm { t o } - V / 2$ . The present derivation concerns an m-site Haar refresh on a compact group and uses neither a random walk, Euclidean geometry, nor a small-step limit.

## S4.1.6 Multiplicative step-size adaptation

Every piece above assumes its proposal sits at the right acceptance rate. The step size σ of the manifold MALA proposal (§ S4.1.2) is the component that holds the local-move acceptance at its optimum. For MALA that optimum is 0.574 in the high-dimensional limit [149]; a plain random-walk proposal would instead target 0.234. The Haar refresh has no step size and reaches its own 0.234 optimum through the site count m of § S4.1.5, not through σ.

Each replica drives its own σ to the target with a multiplicative rule. After every adaptation window it measures the local acceptance A and moves one fixed factor toward the target,

$$
\sigma \longleftarrow \mathrm { c l i p } \Big ( \sigma f ^ { \mathrm { s i g n } ( A - A ^ { \star } ) } , 1 0 ^ { - 4 } , \pi \Big ) , \quad \quad f = 1 . 1 , \quad A ^ { \star } = 0 . 5 7 4 .\tag{S4.10}
$$

The rule is bang-bang: σ rises by the factor $f$ when acceptance runs hot and falls by it when acceptance runs cold, so at equilibrium it oscillates within a factor f of the value that holds $A = A ^ { \star }$ . That is all the precision the sampler needs, because the MALA eficiency curve is flat near its optimum, and the rule carries no schedule and no state. The clip keeps σ inside the range where the proposal is meaningful on the group: steps below $1 0 ^ { - 4 }$ are numerically idle, and π is the diameter-scale move on SU(2).

The natural alternative is a Robbins–Monro stochastic-approximation schedule on log σ [159], whose diminishing gains converge almost surely to the exact target. But a diminishing gain is built for a stationary target, and ours moves: training reshapes $| \psi _ { \theta } | ^ { 2 }$ continuously, so an adapter that never stops moving is the safer default.

## S4.1.7 Gradient cache between MCMC inter-steps

A second saving falls out of the chain’s own structure. The gradient the chain computes when it accepts a move at $q$ is the same gradient the next inter-step needs before it proposes from q. Threading this cached gradient through the scan saves one value and grad per inter-step and halves the per-step gradient cost. At the start of each training step the cache is refreshed against the freshly updated model, which is the caching step the figure marks before the Haar refresh.

## S4.2 Energy kernels

The bottleneck of variational Monte Carlo on a Hamiltonian quadratic in the Pauli operators is the energy estimator. For a sampled configuration $q \in ( \mathrm { S U } ( 2 ) ) ^ { N }$ the local energy splits into a two-body exchange term and a linear field term. The field is the first order in the derivatives of log ψ and is computed as by-product of computing second order terms. The whole cost sits in the exchange term, which contracts the coupling tensor against the second derivatives of log ψ,

$$
{ \cal E } _ { \mathrm { e x c h } } ( q ) \ = \ - \textstyle { \frac { 1 } { 4 } } \sum _ { i , j } \sum _ { a , b } J _ { i j } ^ { a b } \left[ L _ { i } ^ { a } L _ { j } ^ { b } \log \psi ( q ) + \left( L _ { i } ^ { a } \log \psi ( q ) \right) \left( L _ { j } ^ { b } \log \psi ( q ) \right) \right] ,\tag{S4.11}
$$

where $L _ { i } ^ { a }$ is the left-invariant derivative along Pauli direction a at site i (§S1.3) and the prefactor $\textstyle { \frac { 1 } { 4 } }$ comes from $S = { \textstyle { \frac { 1 } { 2 } } } \sigma$ . The first bracket is a Hessian of log $\psi$ contracted against $J ;$ the second is an outer product of its gradient. The Hessian term is the expensive one. Our custom JAXPR interpreter evaluates it, together with every other term the estimator needs, in a single forward pass: a custom forward-mode second-order operator written for the tree, described next.

## S4.2.1 Custom forward-mode Second-order operator

One JAXPR interpreter serves the whole local energy. Its function signature reads,

$$
\begin{array} { r } { \mathtt { c u s t o m \_ l a p } : \ q \ \longmapsto \ \Big ( \log \psi _ { \theta } ( q ) , \ \nabla \log \psi _ { \theta } ( q ) , \ \mathrm { T r } \big ( W \ H _ { \log \psi _ { \theta } } ( q ) \big ) \Big ) , } \end{array}\tag{S4.12}
$$

one forward pass per walker, from which the score, the linear field term and the exchange term of Eq. (S4.11) are all read of: no Hessian is ever materialised and no Hessian-vector product is ever formed on the energy path. The kernel propagates a jet through the network, so each intermediate carries its value, its Jacobian along the Lie-frame input directions, and a running weighted trace, and each primitive advances the three together. The trace advances as a quadratic form of the Jacobian it already carries,

$$
\mathrm { t r } ~ \longleftarrow ~ \mathrm { t r } + \sum _ { \mu , \nu , c } { \mathcal I } _ { \mu c } \Lambda _ { c \mu \nu } { \mathcal I } _ { \nu c } , ,\tag{S4.13}
$$

where ${ \mathcal { I } } _ { \mu c }$ is the intermediate’s Jacobian $( \mu ,$ ν over the Lie-frame input directions, c over channels),not to be confused with the coupling tensor $J _ { \textrm { : } }$ , and $\Lambda _ { c \mu \nu }$ is the second-order weight the chain rule assigns to the primitive at hand; for every linear primitive Λ vanishes, so the sum is paid only where genuine curvature enters. (S4.13) is the update for element-wise primitives, whose curvature is diagonal in the channel index. The merge tensor is the one primitive with cross-channel curvature: there the update couples the two child direction blocks through $T$ itself, and it is the only point in the network where a cross-block entry enters the running trace.

The natural alternative is a generic forward-mode Laplacian[160], as implemented in $\mathbf { f o l x } \ [ 4 2 ]$ . Its tracer treats the second-derivative tangent as densely coupled across sites, because the Pauli tangents hide the structure of $J { \boldsymbol { : } }$ it sees a dense $4 N \times 3 N$ Jacobian fan-out and pays for all of it. The tree of S2.4 makes that coupling sparse. Every q-dependent intermediate at merge level κ carries a Jacobian of the chunked shape $\left[ 3 k , N / k , * S \right]$ , where $k = 2 ^ { \kappa }$ is the number of sites per chunk, $N / k$ the number of chunks, the factor 3 the Lie-frame directions per site, and $\ast S$ the trailing feature axes; the only new cross-block coupling in the Jacobian is the one the merge itself creates between its two children. custom lap is a forward-mode Second order tracer written for exactly this layout. Its tracer treats the second-derivative tangent as densely coupled across sites, because the Pauli tangents hide the structure of $\mathcal { I } \colon$ it sees a dense $4 N \times 3 N$ Jacobian fan-out and pays for all of it. custom lap is our extension of this technique: a weighted-trace generalisation $( \operatorname { T r } ( W H )$ rather than $\operatorname { T r } ( H )$ , which is what lets one pass serve an arbitrary coupling tensor $J ^ { a b } )$ , specialised to the tree’s layout, where chunk structure sets the cost. At level κ the $\hat { N } / 2 ^ { \kappa }$ chunks each carry $3 \cdot 2 ^ { \kappa }$ Lie-frame directions, so the quadratic-form updates at that level cost $\mathcal { O } \big ( ( \hat { N } / 2 ^ { \kappa } ) \cdot ( 3 \cdot 2 ^ { \kappa } ) ^ { 2 } \cdot S \big ) = \mathcal { O } \big ( \hat { N } 2 ^ { \kappa } S \big )$ with S the feature width. Summing the geometric series over $\kappa \leq K$ gives $\mathcal { O } ( \hat { N } ^ { 2 } S )$ for the full pass, against the $\mathcal { O } ( \hat { N } ^ { 3 } )$ of a dense forward-mode tracer. This claim is anticipated in § S2.4.5.

We emphasize that our extension here cannot be used as a universal diferentiator since it assumes the balanced binary tree, and a symmetric two-body weight $W : \mathbb { R } ^ { 3 N } \xrightarrow { } \mathbb { R } ^ { 3 N }$ . Thus, the custom second order operator that we built for our model, can produce correct second order diferential operator values only on the tree-like ansatz. This is the core reason we aren’t bound by the generic algorithmic complexity of evaluating the Hessian of a black box function.

## S4.3 Kronecker-factored curvature for the foundation ansatz

KFAC (Kronecker-factored approximate curvature [43]) is the natural-gradient method we use for pretraining. Natural gradient multiplies the raw gradient by the inverse of the Fisher information matrix, the curvature of the model’s distribution; the resulting step is invariant to how the model is parameterised and reaches a given loss in far fewer iterations than plain gradient descent. The Fisher information is a $P \times P$ matrix for P parameters, far too large to form or invert at our scale, and KFAC’s contribution is to approximate it cheaply.

For a single dense layer with $\mathrm { w e i g h t } W \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ , input activation $a \in \mathbb { R } ^ { d _ { \mathrm { i n } } }$ and back-propagated output gradient $\delta \in \mathbb { R } ^ { d _ { \mathrm { o u t } } }$ , the block of the Fisher information belonging to W is the covariance of the per-example gradient vec $( \delta \boldsymbol { a } ^ { \intercal } )$ ),

$$
F _ { W } ~ = ~ \mathbb { E } \big [ \mathrm { v e c } ( \delta a ^ { \top } ) \mathrm { v e c } ( \delta a ^ { \top } ) ^ { \top } \big ] ~ = ~ \mathbb { E } \big [ a a ^ { \top } \otimes \delta \delta ^ { \top } \big ] .\tag{S4.14}
$$

KFAC replaces the expectation of this Kronecker product by the product of expectations, which is exact when a and δ are independent,

$$
F _ { W } \approx A \otimes G , \qquad A : = \mathbb { E } [ a a ^ { \top } ] , \ : G : = \mathbb { E } [ \delta \delta ^ { \top } ] .\tag{S4.15}
$$

The factorisation is what makes the method afordable. Inverting the two factors separately, $( A \otimes$ $G ) ^ { - 1 } = A ^ { - 1 } \otimes G ^ { - 1 }$ , costs $\mathcal { O } ( d ^ { 3 } )$ per layer in place of the $\mathcal { O } ( d ^ { 6 } )$ of the full block. Before inversion, the KFAC implementation adds damping $\gamma , \widetilde { F } ^ { - 1 } \approx ( \widehat { F } + \gamma I ) ^ { - 1 }$ , which trades fidelity for stability: a smaller γ keeps the preconditioner closer to the true Fisher information and each step closer to the natural-gradient direction, so fewer steps reach a given loss.

Our model breaks three assumptions that KFAC’s standard blocks build in. The trunk’s L repeated blocks stack their weights through a loop, so the Fisher information of those weights couples across the loop iterations. The attention projections couple across the query, key, value and head channels. And the merge tensor of § S2.4 is cubic in its width, so its full Fisher information is far too large to form. The patches below extend the kfac jax library [161] to each case. Together they remove from the model every dense Fisher-information block, and leave a curvature estimate faithful enough to run near the edge of stability. The Fisher information itself is registered on both output channels, the log-amplitude and the phase, as two normal-predictive registrations: the Fubini–Study metric of the complex output.

In the variational Monte Carlo literature this natural gradient is stochastic reconfiguration [162], with the Fubini-Study metric playing the role of the Fisher information [163]. KFAC-preconditioned wavefunction optimisation was introduced for deep ansatze by [28], whose registration strategy ours extends to a complex output.

## S4.3.1 Fusing the query, key and value projections

Each trunk block’s attention sublayer from § S2.3 projects the per-site stream $\ell \in \mathbb { R } ^ { d _ { \ell } }$ into queries $q _ { \mathsf { h } }$ keys $k _ { \mathrm { h } }$ and values $v _ { \mathrm { h } }$ for each of its $n _ { \mathrm { h } }$ heads. We compute all three projections with one weight matrix, $W _ { \mathsf { q k v } } \in \mathbb { R } ^ { ( 3 n _ { \mathsf { h } } d _ { \mathsf { h } } ) \times d _ { \ell } }$ , rather than three separate ones. The fusion needs no custom curvature class: the fused weight registers with KFAC as one ordinary dense layer. The standard factorisation hands KFAC a single gradient-covariance factor G of size $( 3 n _ { \mathsf { h } } d _ { \mathsf { h } } ) ^ { 2 }$ , which carries the covariance between the query, key, value and head channels; three separate registrations would set that cross-covariance to zero by construction.

## S4.3.2 The stacked trunk and a third Kronecker factor

A standard KFAC block factors a dense layer’s Fisher information as $A \otimes G$ . The trunk’s dense layers, however, are stacked: the same layer appears in all L trunk blocks, carried on an extra leading axis of length L. Sharing one $A \otimes G$ across the L copies treats them as independent and throws away any correlation between the blocks. Our block keeps that correlation, factoring the Fisher information as a three-way Kronecker product

$$
F _ { \mathrm { s c a n } } \approx M \otimes A \otimes G ,\tag{S4.16}
$$

where M is the $L \times L$ covariance across the L blocks and A, G are the activation and gradient covariances as before. Here and below $\mathbb { S } _ { + + } ^ { n }$ denotes the $n \times n$ symmetric positive-definite matrices, so $M \in \mathbb { S } _ { + + } ^ { L } , A \in \mathbb { S } _ { + + } ^ { d _ { \mathrm { i n } } }$ and $G \in \mathbb { S } _ { + + } ^ { d _ { \mathrm { o u t } } }$ . Two-factor KFAC is the special case $M \propto I _ { L }$ , which discards the cross-block correlation entirely. The third factor is cheap: $L ,$ the number of trunk blocks, is far smaller than the layer widths, so M adds only an $L ^ { 2 }$ term to factors that already cost $d ^ { 2 }$ . Kronecker structure across a loop axis has precedent in the recurrent setting [164], where the shared weight’s Fisher couples across time steps. Carrying an explicit learned $L \times L$ factor across stacked, distinct block weights, capturing the cross-block correlation that layer-wise block-diagonal KFAC discards, is, as far as we are aware, novel.

The block estimates M online, as a running exponentially-weighted average of the gradient correlations between the L blocks. This is the only statistic it keeps beyond what two-factor KFAC already tracks, and at $\mathcal { O } ( L ^ { 2 } )$ per block its cost is negligible against the $O ( d ^ { 2 } )$ of the activation and gradient factors.

## S4.3.3 Stacked RMS scale parameters

The RMS normalisation layers of SM $\ S \ S 2$ carry a per-channel scale and no shift, and the scales are stacked across the L blocks in the same way. Their Fisher information takes the same form, with the gradient factor replaced by a diagonal: a block-coupling factor across the L copies, times a diagonal per-feature term D. With a single block it reduces, exactly, to the standard diagonal treatment.

## S4.3.4 The merge tensor

The rank-4 merge tensor T of $\ S 2 . 4$ is the hardest block. With B sub-blocks of contraction width $d _ { r }$ it carries $B d _ { r } ^ { 3 }$ parameters, of order $1 0 ^ { 6 }$ at the widths used here, so its full Fisher information would have $( B d _ { r } ^ { 3 } ) ^ { 2 }$ entries, of order $1 0 ^ { 1 3 }$ , roughly ten terabytes, which cannot be formed. The custom block instead matricises the four axes $( i , j , k , l )$ into a balanced pair and factorises the Fisher information as the two-factor Kronecker product $\Sigma _ { L } \otimes \Sigma _ { R }$ over that split, storing and inverting two Gram factors of a few megabytes together; the factored state is about six orders of magnitude smaller than the dense block. The sub-block axis rides inside one of the two factors, a genuine Kronecker leg rather than a batch dimension. Without it the merge tensor would fall back to a dense block and, at this width, would not train; with it the tensor receives a genuine natural-gradient preconditioner.

## S4.3.5 Damping

The faithful three-way Fisher information changes what limits the damping. With the approximation tightened, the floor on how small $\gamma$ can go is the noise in the curvature estimate itself, not the approximation error. We hold $\gamma$ at that noise floor: a constant identity-weight damping of $1 0 ^ { - 3 }$ , with a hard floor of $1 0 ^ { - 4 }$ and a norm constraint of $1 0 ^ { - 3 }$ on the preconditioned step.

Large-model pretraining has shown [165] that deep networks converge in an edge-of-stability regime, where the largest curvature eigenvalue sits just above $2 / \eta$ for learning rate η and the loss oscillates without diverging. Our schedule rides the analogous edge. A cumulative-sum change detector (CUSUM) on the natural-gradient norm classifies the run as stable, marginal or divergent; γ is raised only when the run crosses into the divergent regime and drifted back down afterwards, so the optimiser sits at the smallest stable damping rather than behind a conservative margin.

## S4.3.6 Benchmarking the third factor versus per-layer curvature

The natural alternative to the third factor of Eq. (S4.16) is per-layer curvature: give each of the L stacked copies its own two-factor block and precondition with the block diagonal $\widetilde { F } _ { \mathrm { b d } } : = \bigoplus _ { k = 1 } ^ { L } \widehat { A } _ { k } \otimes \widehat { G } _ { k }$ Per diagonal block this is the more expressive family — every diagonal block of ${ \widetilde { F } } _ { 3 } = { \widehat { M } } \otimes { \widehat { A } } \otimes { \widehat { G } }$ is the same ${ \widehat { A } } \otimes { \widehat { G } } ~ { \mathrm { u p } }$ to the scalar $\widehat { M } _ { k k }$ , whereas $\tilde { F } _ { \mathrm { b d } }$ fits all $L$ blocks independently — and L times more expensive in factor state and inverse work; for the fused $W _ { \mathsf { q k v } }$ projection alone the factor state is 8× larger. The extra freedom buys accuracy only at initialisation, and only on the fused QKV block, where per-layer curvature wins decisively, 0.956 against 0.343 in the damped-action cosine (S4.17); the three-factor block leads on the other three trunk modules already at initialisation, and at the trained point it wins on all four.

We evaluate both families against the exact sampled Fisher of Hamilton-Zero. Curvature capture runs through the estimator’s own one-hot seeded vector–Jacobian products, so the per-layer activations $a _ { k }$ and cotangents $\delta _ { k }$ are precisely the tensors the curvature blocks consume; the exact block $F _ { \mathrm { s c a n } }$ is assembled from the per-walker gradients and inverted exactly through the Woodbury identity at damping $\gamma = 1 0 ^ { - 3 }$ . A candidate $\widetilde { F }$ is scored by its damped-action alignment on the model’s true gradient g,

$$
\cos \angle \big ( ( \widetilde { F } + \gamma I ) ^ { - 1 } g , ~ ( F _ { \mathrm { s c a n } } + \gamma I ) ^ { - 1 } g \big ) ,\tag{S4.17}
$$

evaluated at two points: random initialisation, on a 64-site open-boundary Heisenberg chain, and a trained checkpoint (step 30 000) with equilibrated MCMC walkers, aggregated across the 13 training Hamiltonians categories of §S3 as the estimator aggregates, i.e. factors averaged across systems as the exponential moving average averages them, gradients and damped actions per system.

<table><tr><td colspan="3">Initialisation</td><td rowspan="2"> $\widetilde { F } _ { \mathrm { b d } }$ </td><td rowspan="2">Trained (step 30 000)  ${ \widetilde { F } } _ { 3 }$ </td><td rowspan="2"> $\operatorname { c o n d } ( \widehat { M } )$ </td></tr><tr><td>Trunk module</td><td> $\widetilde { F } _ { \mathrm { b d } }$ </td><td> ${ \widetilde { F } } _ { 3 }$ </td></tr><tr><td>Fused QKV  $\overline { { ( W _ { \mathfrak { q k v } } ) } }$ </td><td>0.956</td><td>0.343</td><td>0.112</td><td>0.153</td><td>4.08</td></tr><tr><td>FFN in-projection</td><td>0.0717</td><td>0.699</td><td>0.0657</td><td>0.224</td><td>3.23</td></tr><tr><td>FFN out-projection</td><td>0.0309</td><td>0.422</td><td>0.0468</td><td>0.160</td><td>9.21</td></tr><tr><td>Attention output</td><td>0.676</td><td>0.833</td><td>0.0761</td><td>0.215</td><td>4.81</td></tr></table>

Table S5: Damped-action alignment of the two stacked-trunk curvature families: the cosine (S4.17) between the approximate damped natural gradient $( \widetilde F + \gamma I ) ^ { - 1 } g$ and the exact $( F _ { \mathrm { s c a n } } + \gamma I ) ^ { - 1 } g$ on the model’s true gradient, at damping $\gamma = 1 0 ^ { - 3 }$

; larger is better. $\widetilde { F } _ { \mathrm { b d } }$ is the per-layer block diagonal $\begin{array} { r } { \bigoplus _ { k } \widehat { A } _ { k } \otimes \widehat { G } _ { k } ; \widetilde { F } _ { 3 } } \end{array}$ is the deployed three-factor block ${ \widehat { M } } \otimes { \widehat { A } } \otimes { \widehat { G } }$ of (S4.16). Initialisation: random parameters on a 64-site open-boundary Heisenberg chain. Trained: checkpoint step 30 000 with equilibrated MCMC walkers; entries are medians of per-system cosines across the 11 training Hamiltonians, with factors aggregated across systems using the same exponential moving average as in training. Last column: median cond(Mc) at the trained point over the three systems with retained per-walker captures.

Table S5 fixes why the trained point belongs to the third factor. First, $F _ { \mathrm { s c a n } }$ is dominated by crosslayer coupling: 0.86–0.87 of its squared Frobenius mass lies of the layer-diagonal, at initialisation and at the trained point alike. $\widetilde { F } _ { \mathrm { b d } }$ is layer-diagonal by construction, so no choice of per-layer factors touches this mass, while the of-diagonal blocks ${ \widehat { M } } _ { k l } { \widehat { A } } \otimes { \widehat { G } }$ of ${ \widetilde { F } } _ { 3 }$ model it directly. Second, the trained $\widehat { M }$ is far from $c I _ { L } \mathbf { : }$ : across the four trunk modules cond $( \hat { M } )$ has medians 3.23–9.21.

The trained regime is three-factor shaped because the residual stream makes both ends of every block gradient shared across the stack. Each stacked copy reads the same normalised stream state, so the activations $a _ { k }$ are nearly independent of $k ;$ the stream update is $r \mapsto r + f _ { k } ( r )$ , so the cotangents propagate through near-identity Jacobians, $\delta _ { k } = J _ { k } ^ { \top } \delta _ { k + 1 }$ with $J _ { k } = I + \partial f _ { k } / \partial r$ , and stay near-collinear in turn. The per-sample gradient of the k-th copy, $g _ { k } = \delta _ { k } a _ { k } ^ { \top }$ , therefore shares a rank-one direction with layer-dependent amplitude, and every block of $F _ { \mathrm { s c a n } }$ factorises accordingly,

$$
g _ { k } \approx \alpha _ { k } \delta a ^ { \top } + \varepsilon _ { k } \qquad \Longrightarrow \qquad ( F _ { \mathrm { s c a n } } ) _ { k l } \approx \mathbb { E } \big [ \alpha _ { k } \alpha _ { l } \big ] A \otimes G ,\tag{S4.18}
$$

with $\delta , \ a$ shared within each sample and $\varepsilon _ { k }$ the idiosyncratic remainder — exactly (S4.16), with $M = \mathbb { E } [ \alpha \alpha ^ { \top } ]$ far from a multiple of the identity. Trained attention concentrates the tangents onto a few specialised low-rank directions, strengthening the shared component; the per-layer family spends its L-fold extra freedom fitting the remainders $\varepsilon _ { k } .$ , i.e. batch noise.

Initialisation favours per-layer curvature on the fused QKV block for an equally structural reason. At random parameters the blocks are weakly input-sensitive, since attention is near-uniform and $\partial f _ { k } / \partial r$ is small, so per-block gradient statistics are near-Gaussian and second moments sufice: on precisely this block, $\widehat { A } _ { k } \otimes \widehat { G } _ { k }$ reproduces its diagonal Fisher block to a relative error of $1 . 1 9 \times 1 0 ^ { - 3 }$ and winning the layer-diagonal is everything a layer-diagonal preconditioner can win. Training grows the input sensitivity and installs the shared low-rank stream of (S4.18), which simultaneously breaks per-block Gaussianity, so second moments stop being suficient, and organises the cross-layer mass into the coherent correlation that only the M factor models. The trained regime is where the optimiser spends the run; there the deployed block is both the better preconditioner and the cheaper one, by a factor of L.

## A The Casimir regulariser for nonlinear ans¨atze

This appendix constructs a variational principle for ans¨atze that satisfy the per-site oddness (S1.39) but are nonlinear in the quaternions, so their support spreads across the half-integer sectors. This is the setting of the phase-space programme of [40], where maintaining a variational principle was left open; the closed-form couplings below resolve it. What oddness does not do is single out the lowest half-integer. The network may still place weight on $\begin{array} { r } { j _ { i } = \frac { 3 } { 2 } } \end{array}$ and above, or on superpositions across them. Pinning it to $\mathrm { ~ a ~ } \frac { 1 } { 2 }$ variational principle requires a second ingredient.

The second ingredient leans on the shape of the Casimir spectrum. By (S1.26) the per-site Casimir $\hat { S } _ { l } ^ { 2 } = - \Delta _ { l }$ reads of the sector: it acts as $j _ { l } ( j _ { l } + 1 )$ , which oddness restricts to the values $\textstyle { \frac { 3 } { 4 } } , { \frac { 1 5 } { 4 } } , { \frac { 3 5 } { 4 } } , \dotsc .$ at $j _ { l } = { \textstyle { \frac { 1 } { 2 } } } , { \textstyle { \frac { 3 } { 2 } } } , { \textstyle { \frac { 5 } { 2 } } } , . . . .$ . The target value $\frac 3 4$ is the floor, and every step away from it grows quadratically in $j _ { l } .$ Oddness is also what lets us charge that gap linearly: on the odd subspace every site sits at $\hat { S } _ { l } ^ { 2 } \geq \frac { 3 } { 4 }$ , so the bare per-site deviation is non-negative, and so is its sum

$$
\hat { C } = \sum _ { l = 1 } ^ { N } \bigl ( \hat { S } _ { l } ^ { 2 } - { \textstyle \frac { 3 } { 4 } } \bigr ) , \qquad C [ \psi ] = \frac { \langle \psi | \hat { C } | \psi \rangle } { \langle \psi | \psi \rangle } = \sum _ { l = 1 } ^ { N } \mathbb { E } _ { x \sim | \psi | ^ { 2 } } \Bigl [ \frac { - \Delta _ { l } \psi } { \psi } \bigl ( x \bigr ) - { \textstyle \frac { 3 } { 4 } } \Bigr ] ,\tag{A.1}
$$

which vanishes exactly on the target sector (Proposition S1.1). Were the integer sectors still present, the $j = 0$ site at $\hat { S } ^ { 2 } = 0$ would sit below the floor and this penalty would reward $\mathrm { i t } ;$ removing them is precisely what makes the bare deviation a sound penalty, and is why we did not need to square it.

The reason this rescues the variational principle is a race between two rates. An excursion into a higher sector lowers the energy only through the spin magnitudes $\sqrt { j _ { l } ( j _ { l } + 1 ) }$ , which grow linearly in $j _ { l }$ , whereas the penalty charges the Casimir gap $\begin{array} { r } { j _ { l } ( j _ { l } + 1 ) - \frac { 3 } { 4 } } \end{array}$ , which grows quadratically. The gap outpaces the gain, so there is a finite multiplier λ above which every unit of energy bought by drifting of-target is overpaid by the penalty, and the regularised objective

$$
E _ { \lambda } [ \psi ] = \frac { \langle \psi | \hat { H } | \psi \rangle } { \langle \psi | \psi \rangle } + \lambda C [ \psi ]\tag{A.2}
$$

stays at or above the physical ground-state energy $E _ { 0 }$ . Because oddness has already removed every integer sector, the nearest a state can sit to the target without lying in it is $\begin{array} { r } { j _ { l } = \frac { 3 } { 2 } } \end{array}$ , the tightest case and the one that fixes how large λ must be.

Evaluating $C [ \psi ]$ costs nothing beyond the energy autodif scheme above, since the log-derivative identity (S1.18) of § S1.2 turns each per-site Casimir ratio $- \Delta _ { l } \psi / \psi$ into log ψ and its first two automatic derivatives. Indeed these are the same quantities the local energy already uses. What remains is to compute, from the couplings (J, h), a λ large enough that $E _ { \lambda } [ \psi ] \ge E _ { 0 }$ for every per-site-odd $\psi .$

In this appendix, we prove there is a λ such that $E _ { \lambda } [ \psi ] \ge E _ { 0 }$ for every per-site-odd ψ for any given Hamiltonian. We then find a tighter bound in which the scalar λ is promoted to a per-edge tensor $\mu _ { \star }$ that saturates the variational principle, i.e. our variational space contains the ground state energy on the tigher bound only. Both are a function of a given Hamiltonian’s interaction data, $J , h$

We use two quantities of interest that can be eficiently computed. First, at each edge (i, k) the relevant size of the $3 \times 3$ interaction block $J _ { i k } \subset J$ is its spectral norm, i.e. the top singular value,

$$
A _ { i k } : = \| J _ { i k } \| _ { 2 } = \sigma _ { 1 } ( J _ { i k } ) ,\tag{A.3}
$$

From it we form the per-site operator-norm weighted degree and the field magnitude,

$$
\mathrm { w d } _ { i } : = \sum _ { k \neq i } A _ { i k } , \qquad b _ { i } : = \big \| h _ { i } \big \| _ { 2 } .\tag{A.4}
$$

The weighted degree wd<sub>i</sub> collects how strongly site i is coupled to its neighbours through any quadratic Pauli channel, and $b _ { i }$ is the size of its transverse field. Both are closed-form functions of $( J , h )$ , and can be eficiently computed once.

Theorem A.1 (Suficient coupling). Let $\begin{array} { r } { \hat { C } = \sum _ { l } ( \hat { S } _ { l } ^ { 2 } - \frac { 3 } { 4 } ) } \end{array}$ and $H ( \lambda ) = \hat { H } + \lambda \hat { C }$ , and write $\varphi =$ $( 1 + { \sqrt { 5 } } ) / 2$ for the golden ratio. For every $\lambda \geq \lambda _ { \operatorname* { m i n } } ( J , h )$ , where

$$
\begin{array} { r } { \lambda _ { \operatorname* { m i n } } ( J , h ) ~ = ~ ( 1 + \epsilon ) \displaystyle \operatorname* { m a x } _ { i \in [ N ] } \Bigl [ \frac { \varphi } { 2 } \mathrm { w d } _ { i } ~ + ~ \frac { 2 } { 3 } b _ { i } \Bigr ] , } \end{array}\tag{A.5}
$$

and every per-site-odd $\psi \in L ^ { 2 } ( ( \mathrm { S U } ( 2 ) ) ^ { N } )$ ),

$$
\langle \psi | H ( \lambda ) | \psi \rangle \geq E _ { 0 } \Vert \psi \Vert ^ { 2 } ,
$$

with $E _ { 0 }$ the physical spin- <sup>1</sup><sub>2</sub> ground-state energy.

The constants $\textstyle { \frac { \varphi } { 2 } }$ and $\frac { 2 } { 3 }$ are the maxima of three gain-to-penalty ratios over the allowed sectors. The slack $\epsilon > 0$ absorbs numerical safety holes (normalisation conventions, fp32 drift in $\operatorname { w d } _ { i } ,$ the isotropy test at the $1 0 ^ { - 9 }$ threshold); we set $\epsilon = 0 . 1$ in all reported runs.

The proof is given in Appendix B.3.

For orientation, the 1D Heisenberg chain at unit coupling has $\mathrm { w d } _ { i } = 2$ in the bulk and no field, so (A.5) reads $\lambda _ { \operatorname* { m i n } } = \left( 1 + \epsilon \right) \varphi \approx 1 . 6 1 8 \left( 1 + \epsilon \right)$ . The remaining gap to the true physical threshold is over-conservatism, not a safety hole, and the next refinement closes most of it.

The coeficient $\textstyle { \frac { \varphi } { 2 } }$ in (A.5) is the universal one: it rests on the spectral norm (A.3) alone and holds for every $J _ { i k }$ . It is loose for structured couplings, and the implementation tightens it edge by edge. The only place the proof touched $J _ { i k }$ was its Cauchy–Schwarz step (Appendix B.3, 3b), through the top singular value, so using more of the matrix sharpens the per-edge coeficient. Call $\alpha \geq 0$ a certificate for a pair tensor J if

$$
\begin{array} { r } { \mathopen { } \mathclose \bgroup \left\| \vec { S } _ { i } ^ { \top } J \vec { S } _ { k } \aftergroup \egroup \right\| _ { \mathrm { o p } , ( s _ { i } , s _ { j } ) } + \mathopen { } \mathclose \bgroup \left\| \vec { S } _ { i } ^ { \top } J \vec { S } _ { k } \aftergroup \egroup \right\| _ { \mathrm { o p } , ( \frac { 1 } { 2 } , \frac { 1 } { 2 } ) } \leq \alpha \left[ q ( s _ { i } ) + q ( s _ { j } ) \right] } \end{array}\tag{A.6}
$$

for every non-trivial half-integer pair $( s _ { i } , s _ { j } ) \neq ( \frac { 1 } { 2 } , \frac { 1 } { 2 } )$ . This is exactly the per-edge condition the proof of Theorem A.1 verified for $\begin{array} { r } { \alpha = \frac { \varphi } { 2 } \| J \| _ { \mathrm { o p } } ; } \end{array}$ any certificate may take its place, and three are worth recording.

Lemma A.2 (Per-edge certificates). Write $\| J \| _ { o p } = \sigma _ { 1 } ( J )$ and $\begin{array} { r } { \| J \| _ { * } = \sum _ { a } \sigma _ { a } ( J ) } \end{array}$ for the spectral and nuclear norms. Each of

$$
\begin{array} { r } { \alpha _ { o p } = \frac { \varphi } { 2 } \parallel J \parallel _ { o p } , \qquad \alpha _ { n u c } = \frac { 1 } { 2 } \parallel J \parallel _ { * } , \qquad \alpha _ { o r t h } = \frac { 3 } { 4 } \mid \lambda \mid { \mathrm { \small ~ ( f o r ~ } J = \lambda Q , ~ Q \in \mathcal { O } ( 3 ) ) } , } \end{array}
$$

is a certificate in the sense of $\left( \mathrm { A . 6 } \right)$ , and certificates add: $\alpha ( J ^ { ( 1 ) } + J ^ { ( 2 ) } ) \le \alpha ( J ^ { ( 1 ) } ) + \alpha ( J ^ { ( 2 ) } )$

The three certificates and their additivity are proved in Appendix B.4.

Additivity turns the three primitives into a family. Peeling of an isotropic part $J = \tau I + K$ with $\tau = { \textstyle \frac { 1 } { 3 } } \mathrm { t r } ( J )$ (so τ I takes the orthogonal certificate with $Q = I )$ and bounding the remainder K by the operator or nuclear certificate, or peeling of the nearest orthogonal part $J = \lambda Q + K _ { \mathrm { p o l } }$ with

$Q = U V ^ { \top }$ from the SVD and $\begin{array} { r } { \lambda = \frac { 1 } { 3 } \mathrm { t r } ( Q ^ { \top } J ) } \end{array}$ , gives six certificates whose minimum is used for each edge,

$$
\begin{array} { r } { \alpha _ { \mathrm { f u l l } } ( J ) = \operatorname* { m i n } \left\{ \frac { \varphi } { 2 } \| J \| _ { \infty } , \ \frac { 1 } { 2 } \| J \| _ { * } , \ \frac { 3 } { 4 } | \tau | + \frac { \varphi } { 2 } \| K \| _ { \infty } , \ \frac { 3 } { 4 } | \tau | + \frac { 1 } { 2 } \| K \| _ { * } , \ \frac { 3 } { 4 } | \lambda | + \frac { \varphi } { 2 } \| K _ { \mathrm { p o l } } \| _ { \infty } , \ \frac { 3 } { 4 } | \lambda | + \frac { 1 } { 2 } \| K _ { \mathrm { p o l } } \| _ { * } \right\} } \end{array}\tag{A.7}
$$

Theorem A.3 (Tight per-edge threshold). With $\alpha _ { f u l l }$ from (A.7), the threshold

$$
\mu _ { \star } ( J , h ) \ = \ ( 1 + \epsilon ) \ \operatorname* { m a x } _ { i \in [ N ] } \Bigl [ \sum _ { k \neq i } \alpha _ { f u l l } ( J _ { i k } ) + \textstyle \frac { 2 } { 3 } b _ { i } \Bigr ]\tag{A.8}
$$

obeys $\mu _ { \star } \le \lambda _ { \operatorname* { m i n } }$ and the conclusion ofTheorem A.1: for every $\lambda \geq \mu _ { \star }$ and per-site-odd ψ, $\langle \psi | H ( \lambda ) | \psi \rangle \geq$ $E _ { 0 } \| \psi \| ^ { 2 }$

The proof is given in Appendix B.5.

On the canonical couplings the minimum lands on the physically right primitive. Heisenberg $J = J _ { 0 } I$ gives $\alpha _ { \mathrm { f u l l } } = \frac { 3 } { 4 } | J _ { 0 } |$ from the isotropic entry, recovering the textbook $\begin{array} { r } { \mu _ { \star } = \frac { 3 } { 4 } \mathrm { w d } _ { i } , } \end{array}$ that is 1.5 on the 1D unit chain rather than the universal $\varphi \approx 1 . 6 1 8 ;$ pure Ising or XY (J rank one) drops to $\scriptstyle { \frac { 1 } { 2 } } \left| J \right|$ through the nuclear entry; pure Dzyaloshinskii–Moriya stays at ${ \scriptstyle { \frac { \varphi } { 2 } } } | D | .$ , the operator entry, since its singular values are dense.

## B Proofs from Supp. Mat. S1

The five results stated in SM § S1 and Appendix A are proved here: Proposition S1.1, Theorem S1.2, Theorems A.1 and A.3, and Lemma A.2. Proposition S1.1 is elementary and can be checked by hand.

## B.1 Proof of Proposition S1.1

Proof. By the Peter–Weyl expansion above, $\begin{array} { r } { \psi = \sum _ { J , m , n } c _ { J , m , n } \Phi _ { J , m , n } } \end{array}$ , with $- \Delta _ { k }$ diagonal of eigenvalue $j _ { k } ( j _ { k } + 1 )$ by (S1.26). For the forward direction, (∗) subtracts to

$$
0 = \sum _ { J , m , n } c _ { J , m , n } \left( j _ { k } ( j _ { k } + 1 ) - { \textstyle \frac { 3 } { 4 } } \right) \Phi _ { J , m , n } ,\tag{B.1}
$$

and orthogonality of the basis forces $c _ { J , m , n } ~ = ~ 0$ whenever $j _ { k } \ \ne \ \frac 1 2$ . Holding at every site, only $\begin{array} { r } { \pmb { J } = \big ( \frac { 1 } { 2 } , \dots , \frac { 1 } { 2 } \big ) } \end{array}$ survives, and $T _ { m , n } : = c _ { ( { \frac { 1 } { 2 } } , \dots , { \frac { 1 } { 2 } } ) , m , n }$ gives the stated form. Conversely, applying $- \Delta _ { k }$ to the tensor form, the factors at sites $r \neq k$ pass through and the site-k factor is a spin- <sup>1</sup> Wigner matrix on which $- \Delta _ { k }$ acts as $\begin{array} { r } { \frac { 3 } { 4 } , \mathrm { { s o } } - \Delta _ { k } \psi = \frac { 3 } { 4 } \psi } \end{array}$ at every site. □

## B.2 Proof of Theorem S1.2

Proof. Schur orthogonality gives, on one copy of G,

$$
\int _ { G } \overline { { { D _ { m n } ^ { 1 / 2 } ( q ) } } } D _ { m ^ { \prime } n ^ { \prime } } ^ { 1 / 2 } ( q ) \mathrm { d } q = \frac { 1 } { 2 } \delta _ { m m ^ { \prime } } \delta _ { n n ^ { \prime } } .\tag{B.2}
$$

Taking the product over N sites and using $\| \chi \| = 1$ shows $\| \iota _ { \chi } ( \Psi ) \| _ { L ^ { 2 } ( G ^ { N } ) } = \| \Psi \| _ { \mathcal { H } _ { N } } ;$ hence $\iota _ { \chi }$ is an isometry and in particular is injective.

Peter–Weyl identifies the spin- <sup>1</sup><sub>2</sub> sector with $\mathcal { K } _ { N } \otimes \mathcal { H } _ { N }$ , which is exactly $\mathcal { F } _ { \mathrm { l i n } }$ . Since

$$
\dim \iota _ { \chi } ( \mathcal { H } _ { N } ) = 2 ^ { N } \quad \mathrm { a n d } \quad \dim \mathcal { F } _ { \mathrm { l i n } } = 4 ^ { N } ,\tag{B.3}
$$

the first inclusion in $\operatorname { E q . }$ (S1.32) is strict for $N \geq 1$

The central element $- I \in G$ acts on $V _ { j }$ as $( - 1 ) ^ { 2 j } I .$ Consequently a Peter–Weyl coeficient is odd under $q _ { i } \mapsto - q _ { i }$ if and only if its site-i spin $j _ { i }$ is half-integer. This proves the direct sum in Eq. (S1.30) and gives $\mathcal { F } _ { \mathrm { l i n } } \subseteq \mathcal { F } _ { \mathrm { o d d } }$ . The inclusion is strict because, for example,

$$
D _ { 3 / 2 , 3 / 2 } ^ { 3 / 2 } ( q _ { 1 } ) \prod _ { i = 2 } ^ { N } D _ { 1 / 2 , 1 / 2 } ^ { 1 / 2 } ( q _ { i } ) \in \mathcal { F } _ { \mathrm { o d d } } \setminus \mathcal { F } _ { \mathrm { l i n } } ,\tag{B.4}
$$

with the product over $i \geq 2$ omitted when $N = 1$

On $V _ { 1 / 2 } ^ { * } \otimes V _ { 1 / 2 }$ , the right-regular diferential action generated by the left-invariant fields acts on the second, column/physical factor and leaves the first, row/multiplicity factor invariant. Tensoring over sites therefore gives

$$
\widetilde { \cal H } \big | _ { \mathcal { F } _ { \mathrm { l i n } } } \cong I _ { \mathcal { K } _ { N } } \otimes H _ { \mathcal { H } _ { N } } .\tag{B.5}
$$

Its lowest eigenvalue is $E _ { 0 }$ , which proves Eq. (S1.35); equality is attained by $\iota _ { \chi } ( \Psi _ { 0 } )$ for any physical ground state $\Psi _ { 0 }$

Finally, on a per-site-odd Peter–Weyl block labelled by ${ \pmb j } = ( j _ { 1 } , \dots , j _ { N } )$ 2

$$
\widehat { C } = \sum _ { i = 1 } ^ { N } \left( j _ { i } ( j _ { i } + 1 ) - \frac { 3 } { 4 } \right) I .\tag{B.6}
$$

Every summand is nonnegative for half-integer $\begin{array} { r } { j _ { i } \geq \frac { 1 } { 2 } } \end{array}$ , and the sum vanishes if and only if every $\begin{array} { r } { j _ { i } = \frac { 1 } { 2 } } \end{array}$ Thus $\widehat { C } \succeq 0$ on $\mathcal { F } _ { \mathrm { o d d } }$ and ker $\widehat { C } = \mathcal { F } _ { \mathrm { l i n } }$ . Theorem A.3 gives Eq. (S1.37) for $\lambda \geq \mu _ { \star } ( J , h )$ . Conversely the lifted ground state $\iota _ { \chi } ( \Psi _ { 0 } )$ lies in ker $\widehat { C }$ and has regularised quotient $E _ { 0 }$ , so the infimum over $\mathcal { F } _ { \mathrm { o d d } }$ is exactly $E _ { 0 }$ □

## B.3 Proof of Theorem A.1 (suficient coupling)

Proof. Write $q ( j ) : = j ( j + 1 ) - \frac { 3 } { 4 }$ for the Casimir gap and $r ( j ) : = \sqrt { j ( j + 1 ) }$ for the Casimir root, and recall the Peter–Weyl decomposition of the per-site-odd subspace

$$
\mathcal { H } _ { \mathrm { o d d } } = \bigoplus _ { s : j _ { i } \in \{ \frac { 1 } { 2 } , \frac { 3 } { 2 } , \frac { 5 } { 2 } , \dots \} } \mathcal { H } _ { s } , \qquad \mathcal { H } _ { s } : = \bigotimes _ { i = 1 } ^ { N } \bigl ( V _ { j _ { i } } ^ { * } \otimes V _ { j _ { i } } \bigr ) .
$$

(1) $\hat { H }$ and $\hat { C }$ are block-diagonal in s. The spin operators $\hat { S } _ { i } ^ { a }$ act through the right-regular diferential representation generated by the left-invariant fields. This action preserves every Peter–Weyl isotypic component $V _ { j _ { i } } ^ { * } \otimes V _ { j _ { i } }$ and acts on its second factor. Hence each $\hat { S } _ { i } ^ { a }$ acts within $\mathcal { H } _ { s }$ , and $\begin{array} { r } { \hat { H } = \sum J _ { i j } ^ { a b } \hat { S } _ { i } ^ { a } \hat { S } _ { j } ^ { b } + \sum h _ { i } ^ { a } \hat { S } _ { i } ^ { a } } \end{array}$ preserves $\mathcal { H } _ { s }$ . The penalty $\begin{array} { r } { \hat { C } = \sum _ { l } ( \hat { S } _ { l } ^ { 2 } - \frac { 3 } { 4 } ) } \end{array}$ is a function of the per-site Casimirs, so it acts on $\mathcal { H } _ { s }$ as the scalar $\hat { C } \equiv Q ( s ) I$ with $\begin{array} { r } { Q ( s ) : = \sum _ { l } q ( j _ { l } ) } \end{array}$

(2) Sector-gap upper bound (variational + purification, derived inline). Fix a sector s with raised set $R : = { \bar { R } } ( s ) = \{ i : j _ { i } \neq { \frac { 1 } { 2 } } \}$ non-empty, and $R ^ { c } : = [ N ] \setminus R$ . Split the Hamiltonian by which edges/fields touch R:

$$
\hat { H } \ = \ \hat { H } _ { \mathrm { o f f } } \ + \ \hat { H } _ { \mathrm { i n c } } \ + \ \hat { H } _ { \mathrm { f i e l d } } ^ { R } ,
$$

with $\hat { H } _ { \mathrm { o f f } }$ collecting pair terms with both endpoints in $R ^ { c }$ and the field on $R ^ { c } ; \hat { H } _ { \mathrm { i n c } }$ collecting pair terms with at least one endpoint in $R ;$ and $\hat { H } _ { \mathrm { f i e l d } } ^ { R }$ the field on R. By construction $\hat { H } _ { \mathrm { o f f } }$ acts trivially on the R factor and lives entirely on $R ^ { c }$

Upper bound on $E _ { 0 } ( \mathcal { H } _ { \frac { 1 } { 2 } } )$ (variational). Let $| \phi _ { 0 } \rangle _ { R ^ { c } }$ be the spin- <sup>1</sup> ground state of $\hat { H } _ { \mathrm { o f f } }$ on

$$
\bigotimes _ { i \in R ^ { c } } ( V _ { 1 / 2 } ^ { * } \otimes V _ { 1 / 2 } ) ,
$$

with energy $E _ { 0 } ^ { \mathrm { o f f } , 1 / 2 }$ . Let $| \xi \rangle _ { R }$ be any unit vector in

$$
\bigotimes _ { i \in R } ( V _ { 1 / 2 } ^ { * } \otimes V _ { 1 / 2 } ) .
$$

The product $| \psi _ { \frac { 1 } { 2 } } ^ { \mathrm { v a r } } \rangle : = | \phi _ { 0 } \rangle _ { R ^ { c } } \otimes | \xi \rangle _ { R }$ lies in $\mathcal { H } _ { \frac { 1 } { - } }$ , so 2

$$
\begin{array} { r l } { E _ { 0 } ( \mathcal { H } _ { \frac { 1 } { 2 } } ) \ \leq \ \langle \psi _ { \frac { 1 } { 2 } } ^ { \mathrm { v a r } } | \hat { H } | \psi _ { \frac { 1 } { 2 } } ^ { \mathrm { v a r } } \rangle } \\ { \ } & { = \ \langle \phi _ { 0 } | \hat { H } _ { \mathrm { o f f } } | \phi _ { 0 } \rangle + \langle \psi _ { \frac { 1 } { 2 } } ^ { \mathrm { v a r } } | ( \hat { H } _ { \mathrm { i n c } } + \hat { H } _ { \mathrm { f i e l d } } ^ { R } ) | \psi _ { \frac { 1 } { 2 } } ^ { \mathrm { v a r } } \rangle } \\ { \ } & { \leq \ E _ { 0 } ^ { \mathrm { o f f } , 1 / 2 } \ + \ \left. \hat { H } _ { \mathrm { i n c } } + \hat { H } _ { \mathrm { f i e l d } } ^ { R } \right. _ { \mathrm { o p } , \frac { 1 } { 2 } } . } \end{array}
$$

The choice of $| \xi \rangle _ { R }$ is free; the inequality holds for any choice and hence for the supremum, the operator-norm upper bound is independent of |ξ⟩.

Lower bound on $E _ { 0 } ( \mathcal { H } _ { s } )$ (purification $/$ partial trace). Let $| \psi _ { s } \rangle$ be the ground state of $\hat { H }$ on $\mathcal { H } _ { s } .$ energy $E _ { 0 } ( \mathcal { H } _ { s } )$ . Define the reduced density matrix $\rho _ { R ^ { c } } : = \mathrm { t r } _ { R } | \psi _ { s } \rangle \langle \psi _ { s } |$ on $\otimes _ { i \in R ^ { c } } ( V _ { 1 / 2 } \otimes V _ { 1 / 2 } ^ { * } )$ (since $\begin{array} { r } { j _ { i } = \frac { 1 } { 2 } } \end{array}$ for $i \in R ^ { c }$ by definition of R). Then

$$
\begin{array} { r l } & { E _ { 0 } ( \mathcal { H } _ { s } ) = \langle \psi _ { s } | \hat { H } _ { \mathrm { o f f } } | \psi _ { s } \rangle ~ + ~ \langle \psi _ { s } | ( \hat { H } _ { \mathrm { i n c } } + \hat { H } _ { \mathrm { f i e l d } } ^ { R } ) | \psi _ { s } \rangle } \\ & { ~ = ~ \mathrm { t r } \big ( \hat { H } _ { \mathrm { o f f } } \rho _ { R ^ { c } } \big ) ~ + ~ \langle \psi _ { s } | ( \hat { H } _ { \mathrm { i n c } } + \hat { H } _ { \mathrm { f i e l d } } ^ { R } ) | \psi _ { s } \rangle } \\ & { ~ \geq ~ E _ { 0 } ^ { \mathrm { o f f } , 1 / 2 } ~ - ~ \big \| \hat { H } _ { \mathrm { i n c } } + \hat { H } _ { \mathrm { f i e l d } } ^ { R } \big \| _ { \mathrm { o p } , s } . } \end{array}
$$

The first equality uses that $\hat { H } _ { \mathrm { o f f } }$ acts on the $R ^ { c }$ factor only, so $\langle \hat { H } _ { \mathrm { o f f } } \rangle = \mathrm { t r } ( \hat { H } _ { \mathrm { o f f } } \rho _ { R ^ { c } } )$ . The inequality uses (i) tr $( \hat { H } _ { \mathrm { o f f } } \rho _ { R ^ { c } } ) \geq E _ { 0 } ^ { \mathrm { o f f } , 1 / 2 } \cdot \mathrm { t r } ( \rho _ { R ^ { c } } ) = E _ { 0 } ^ { \mathrm { o f f } , 1 / 2 }$ (spectral lower bound on $\hat { H } _ { \mathrm { o f f } } .$ , which has spectrum $\ge E _ { 0 } ^ { \mathrm { o f f } , 1 / 2 }$ on the $\mathrm { s p i n } { - } \frac { 1 } { 2 }$ sector where $\rho _ { R ^ { c } }$ is supported); and (ii) $| \langle \psi _ { s } | M | \psi _ { s } \rangle | \leq$ $\| M \| _ { \mathrm { o p } , s }$ for any operator M acting on $\mathcal { H } _ { s }$

Subtract:

$$
\begin{array} { r } { E _ { 0 } ( \mathcal { H } _ { \frac { 1 } { 2 } } ) - E _ { 0 } ( \mathcal { H } _ { s } ) \ : \leq \ : \underbrace { { \left. \hat { H } _ { \mathrm { i n c } } + \hat { H } _ { \mathrm { f i e l d } } ^ { R } \right. } _ { \mathrm { o p } , \frac { 1 } { 2 } } + { \left. \hat { H } _ { \mathrm { i n c } } + \hat { H } _ { \mathrm { f i e l d } } ^ { R } \right. } _ { \mathrm { o p } , s } } _ { = : B ( s ) } . } \end{array}
$$

The $E _ { 0 } ^ { \mathrm { o f f } , 1 / 2 }$ pieces cancel exactly.

(3) Per-edge bounds, derived inline. By triangle inequality on the sum defining $\hat { H } _ { \mathrm { i n c } } + \hat { H } _ { \mathrm { f i e l d } } ^ { R }$ , each side $t \in \{ \textstyle { \frac { 1 } { 2 } } , s \}$ satisfies

$$
\big \| \hat { H } _ { \mathrm { i n c } } + \hat { H } _ { \mathrm { f i e l d } } ^ { R } \big \| _ { \mathrm { o p } , t } \leq \sum _ { ( i , k ) \mathrm { ~ i n c . ~ t o ~ } R } \big \| \vec { S } _ { i } ^ { \top } J _ { i k } \vec { S } _ { k } \big \| _ { \mathrm { o p } , t } + \sum _ { i \in R } \big \| \vec { h } _ { i } ^ { \top } \vec { S } _ { i } \big \| _ { \mathrm { o p } , t } .
$$

Two factor bounds are needed; both follow from one-line spin-algebra calculations.

(3a) Field op-norm: $\| \vec { h } _ { i } ^ { \top } \vec { S } _ { i } \| _ { o p , V _ { j } } = b _ { i } \cdot j$ . Pick the orthogonal frame in which $\vec { h } _ { i }$ points along $\hat { z } ,$ so $\vec { h } _ { i } ^ { \top } \vec { S } _ { i } = b _ { i } S _ { i } ^ { z }$ . On $V _ { j } , \ S _ { i } ^ { z }$ is diagonal with eigenvalues $\{ - j , - j + 1 , \ldots , j \}$ ; its operator norm equals the maximum absolute eigenvalue, $j .$ . Rotational invariance of the operator norm on the spin representation gives the result for any direction of $\vec { h } _ { i }$

(3b) Cauchy–Schwarz pair op-norm: $\| \vec { S } _ { i } ^ { \top } J _ { i k } \vec { S } _ { k } \| _ { o p , V _ { j _ { i } } \otimes V _ { j _ { k } } } \leq A _ { i k } r ( j _ { i } ) r ( j _ { k } )$ . Take the real SVD $J _ { i k } = U \Sigma V ^ { \top }$ with $\Sigma = \mathrm { d i a g } ( \sigma _ { 1 } , \sigma _ { 2 } , \sigma _ { 3 } )$ and $\sigma _ { 1 } = A _ { i k }$ ; rewrite

$$
\vec { S } _ { i } ^ { \top } J _ { i k } \vec { S } _ { k } \ = \ \sum _ { a = 1 } ^ { 3 } { \sigma } _ { a } \left( \vec { u } _ { a } \cdot \vec { S } _ { i } \right) \left( \vec { v } _ { a } \cdot \vec { S } _ { k } \right) \ = \ \sum _ { a } { \sigma } _ { a } A _ { a } B _ { a } ,
$$

with $A _ { a } : = \vec { u } _ { a } \cdot \vec { S } _ { i } , B _ { a } : = \vec { v } _ { a } \cdot \vec { S } _ { k }$ . For any unit $\vert \psi \rangle \in V _ { j _ { i } } \otimes V _ { j _ { k } }$ , two successive Cauchy–Schwarz steps give

$$
\begin{array} { r } { \left| \left. \psi \right| \sum _ { a } \sigma _ { a } A _ { a } B _ { a } | \psi \rangle \right| \leq \sum _ { a } \sigma _ { a } \left\| A _ { a } | \psi \rangle \right\| \| B _ { a } | \psi \rangle \| \leq \sqrt { \sum _ { a } \sigma _ { a } ^ { 2 } \| A _ { a } | \psi \rangle \| ^ { 2 } } \sqrt { \sum _ { a } \| B _ { a } | \psi \rangle \| ^ { 2 } } . } \end{array}
$$

For the first factor: $\begin{array} { r } { \sum _ { a } \sigma _ { a } ^ { 2 } \| A _ { a } | \psi \rangle \| ^ { 2 } \leq \sigma _ { 1 } ^ { 2 } \sum _ { a } \langle \psi | A _ { a } ^ { \dagger } A _ { a } | \psi \rangle = A _ { i k } ^ { 2 } \langle \psi | \vec { S } _ { i } ^ { \top } ( U U ^ { \top } ) \vec { S } _ { i } | \psi \rangle = A _ { i k } ^ { 2 } j _ { i } ( j _ { i } + 1 ) , } \end{array}$ 1), where $U U ^ { \top } = I _ { 3 } \ ( \mathrm { o r t h o g o n a l } )$ and ${ \vec { S } } _ { i } \cdot { \vec { S } } _ { i } = j _ { i } ( j _ { i } + 1 ) I$ on $V _ { j _ { i } }$ (Casimir). For the second factor: $\begin{array} { r } { \sum _ { a } \| B _ { a } | \psi \rangle \| ^ { 2 } = \langle \psi | \vec { S } _ { k } \cdot \vec { S } _ { k } | \psi \rangle = j _ { k } \big ( j _ { k } + 1 \big ) } \end{array}$ by the same argument with $V V ^ { \top } = I _ { 3 }$ . Product: $| \langle \psi | \vec { S } _ { i } ^ { \top } J _ { i k } \vec { S } _ { k } | \psi \rangle | \leq A _ { i k } r ( j _ { i } ) r ( j _ { k } )$ , which is the operator-norm bound by taking the supremum over unit |ψ⟩.

Packaging. Using $j _ { i } ^ { e } : = j _ { i }$ for $i \in R$ and $\frac { 1 } { 2 }$ for $i \in R ^ { c }$ , and adding the $\begin{array} { r } { t = \frac { 1 } { 2 } } \end{array}$ and $t = s$ sides:

$$
B ( s ) \ \leq \ \sum _ { ( i , k ) \ \mathrm { i n c . } } A _ { i k } \big [ r \big ( j _ { i } ^ { e } \big ) r \big ( j _ { k } ^ { e } \big ) + r \big ( \textstyle { \frac { 1 } { 2 } } \big ) ^ { 2 } \big ] \ + \ \sum _ { i \in R } b _ { i } \big [ j _ { i } ^ { e } + \textstyle { \frac { 1 } { 2 } } \big ] .
$$

(4) Per-site accounting. Assign each contribution in the packaged $B ( s )$ to a raised endpoint, measured against the gap $q ( j )$ that the linear penalty collects there. The reader can read each ratio straight of the packaging.

Boundary edges. One endpoint i raised, the other k at $\textstyle { \frac { 1 } { 2 } }$ , so the contribution is $A _ { i k } \left[ r ( j _ { i } ) r ( \textstyle { \frac { 1 } { 2 } } ) + \right.$ $r \big ( { \textstyle { \frac { 1 } { 2 } } } \big ) ^ { 2 } \big ]$ . Assigning it to i and dividing by $q ( j _ { i } )$ defines

$$
\gamma _ { \mathrm { b d y } } ( j ) : = \frac { r ( \frac { 1 } { 2 } ) \left( r ( j ) + r ( \frac { 1 } { 2 } ) \right) } { q ( j ) } .
$$

Internal edges. Both endpoints raised, contribution $A _ { i k } \left[ r ( j _ { i } ) r ( j _ { k } ) + r ( \textstyle { \frac { 1 } { 2 } } ) ^ { 2 } \right]$ . Apply AM–GM, $\begin{array} { r } { r ( j _ { i } ) r ( j _ { k } ) \le \frac { 1 } { 2 } ( j _ { i } ( j _ { i } + 1 ) + j _ { k } ( j _ { k } + 1 ) ) , } \end{array}$ ), split the constant $r (  { { \frac { 1 } { 2 } } } ) ^ { 2 } =  { { \frac { 1 } { 2 } } } r (  { { \frac { 1 } { 2 } } } ) ^ { 2 } +  { { \frac { 1 } { 2 } } } r (  { { \frac { 1 } { 2 } } } ) ^ { 2 }$ symmetrically, and split half to each endpoint; the i-side share, over $q ( j _ { i } )$ , defines

$$
\gamma _ { \mathrm { i n t } } ( j ) : = \frac { j ( j + 1 ) + \frac { 3 } { 4 } } { 2 q ( j ) } .
$$

Field at $i \in R$ . Assigning $b _ { i } ( j _ { i } + \textstyle { \frac { 1 } { 2 } } )$ to i and dividing by $q ( j _ { i } )$ defines

$$
\gamma _ { h } ( j ) : = \frac { j + \frac { 1 } { 2 } } { q ( j ) } .
$$

With $\gamma _ { J } ( j ) : = \operatorname* { m a x } \bigl ( \gamma _ { \mathrm { b d y } } ( j ) , \gamma _ { \mathrm { i n t } } ( j ) \bigr )$ , every incident edge contributes at most $A _ { i k } \gamma _ { J } ( j _ { i } ) q ( j _ { i } )$ to the i-side share, and summing the incident edges into $\mathrm { w d } _ { i }$

$$
B ( s ) \ \leq \ \sum _ { i \in R } \left[ \gamma _ { J } ( j _ { i } ) \mathrm { w d } _ { i } + \gamma _ { h } ( j _ { i } ) b _ { i } \right] q ( j _ { i } ) .
$$

(5) Monotonicity: the worst harmonic component sits at $\begin{array} { r } { j = \frac { 3 } { 2 } } \end{array}$ . Treat $j$ as a continuous variable on $\textstyle \left( { \frac { 1 } { 2 } } , \infty \right)$ and write $u : = j ( j + 1 )$ , so $\begin{array} { r } { \bar { q ( j ) } = u - \frac { 3 } { 4 } } \end{array}$ and $u ^ { \prime } \bar { = } 2 j + 1 > 0$ . Each ratio is strictly decreasing. For $\begin{array} { r } { \gamma _ { h } ( j ) = ( j + \frac { 1 } { 2 } ) / q ( j ) } \end{array}$ , the quotient rule with $q ^ { \prime } ( j ) = 2 j + 1$ and $2 ( j + { \textstyle { \frac { 1 } { 2 } } } ) = 2 j + 1$ gives

$$
\partial _ { j } \gamma _ { h } = \frac { q ( j ) - ( j + \frac { 1 } { 2 } ) ( 2 j + 1 ) } { q ( j ) ^ { 2 } } = \frac { q ( j ) - \frac { 1 } { 2 } ( 2 j + 1 ) ^ { 2 } } { q ( j ) ^ { 2 } } = \frac { - j ^ { 2 } - j - \frac { 5 } { 4 } } { q ( j ) ^ { 2 } } < 0 .
$$

For $\begin{array} { r } { \gamma _ { \mathrm { i n t } } ( j ) = ( u + \frac { 3 } { 4 } ) / \big ( 2 ( u - \frac { 3 } { 4 } ) \big ) } \end{array}$ , diferentiating in u (sign-preserving since $u ^ { \prime } > 0 )$ ),

$$
\partial _ { u } \gamma _ { \mathrm { i n t } } ~ = ~ { \frac { ( u - { \frac { 3 } { 4 } } ) - ( u + { \frac { 3 } { 4 } } ) } { 2 ( u - { \frac { 3 } { 4 } } ) ^ { 2 } } } ~ = ~ { \frac { - { \frac { 3 } { 2 } } } { 2 ( u - { \frac { 3 } { 4 } } ) ^ { 2 } } } ~ < ~ 0 .
$$

For $\begin{array} { r } { \gamma _ { \mathrm { b d y } } ( j ) = \frac { \sqrt { 3 } } { 2 } \big ( \sqrt { u } + \frac { \sqrt { 3 } } { 2 } \big ) / ( u - \frac { 3 } { 4 } ) } \end{array}$ , the quotient rule on u (with $\partial _ { u } \sqrt { u } = 1 / ( 2 \sqrt { u } ) )$ carries the numerator $\begin{array} { r } { \frac { u - 3 \bar { / 4 } } { 2 \sqrt { u } } - \sqrt { u } - \frac { \bar { \sqrt { 3 } } } { 2 } = - \frac { \bar { u + 3 / 4 } } { 2 \sqrt { u } } - \frac { \sqrt { 3 } } { 2 } < 0 . } \end{array}$ , so $\partial _ { u } \gamma _ { \mathrm { b d y } } < 0$ . All three decrease on $\textstyle \left( { \frac { 1 } { 2 } } , \infty \right)$ , so $\gamma _ { J } = \operatorname* { m a x } ( \gamma _ { \mathrm { b d y } } , \gamma _ { \mathrm { i n t } } )$ does too. Per-site oddness $( \ S \ S 1 . 4 )$ leaves $\begin{array} { r } { j = \frac { 3 } { 2 } } \end{array}$ as the smallest allowed value above $\frac { 1 } { 2 }$ , where the ratios reach their worst case,

$$
\gamma _ { \mathrm { b d y } } \bigl ( { \textstyle { \frac { 3 } { 2 } } } \bigr ) = \frac { 1 + \sqrt { 5 } } { 4 } = \frac { \varphi } { 2 } , \qquad \gamma _ { \mathrm { i n t } } \bigl ( { \textstyle { \frac { 3 } { 2 } } } \bigr ) = \frac { 3 } { 4 } , \qquad \gamma _ { h } \bigl ( { \textstyle { \frac { 3 } { 2 } } } \bigr ) = \frac { 2 } { 3 } ,
$$

so $\begin{array} { r } { \gamma _ { J } ( j _ { i } ) \le \frac { \varphi } { 2 } } \end{array}$ (the boundary ratio wins, $\textstyle { \frac { \varphi } { 2 } } > { \frac { 3 } { 4 } } )$ and $\begin{array} { r } { \gamma _ { h } ( j _ { i } ) \le \frac { 2 } { 3 } } \end{array}$ at every raised site. Substituting into the per-site bound of step (4),

$$
\begin{array} { r l l } { \displaystyle B ( s ) \ \le \ \sum _ { i \in R } \left[ \frac { \varphi } { 2 } \operatorname { w d } _ { i } + \frac { 2 } { 3 } b _ { i } \right] q ( j _ { i } ) } \\ { \displaystyle } & { \le \ \left( \operatorname* { m a x } _ { i } \left[ \frac { \varphi } { 2 } \operatorname { w d } _ { i } + \frac { 2 } { 3 } b _ { i } \right] \right) \sum _ { i \in R } q ( j _ { i } ) } \\ { \displaystyle } & { = \ \frac { \lambda _ { \operatorname* { m i n } } ( J , h ) } { 1 + \epsilon } Q ( s ) \ \le \ \lambda _ { \operatorname* { m i n } } ( J , h ) Q ( s ) . } \end{array}
$$

(6) Assemble. Decompose any per-site-odd $\psi = \textstyle \sum _ { s } \psi _ { s }$ over the orthogonal sectors. Using blockdiagonality (1) on H<sup>ˆ</sup> and the scalar action of $\hat { C }$ from (1),

$$
\begin{array} { r c l } { \langle \psi | H ( \lambda ) | \psi \rangle } & { = } & { \displaystyle \sum _ { s } \left[ \langle \psi _ { s } | \hat { H } | \psi _ { s } \rangle + \lambda Q ( s ) \| \psi _ { s } \| ^ { 2 } \right] \quad \quad \mathrm { ( b l o c k - d i a g o n a l ~ i n ~ } s ) } \\ & { \geq } & { \displaystyle \sum _ { s } \left[ E _ { 0 } ( \mathcal { H } _ { s } ) + \lambda Q ( s ) \right] \| \psi _ { s } \| ^ { 2 } \quad \quad \mathrm { ( w i t h i n - s e c t o r ~ v a r i a t i o n a l ~ p r i n c i p l e ) } } \\ & { \geq \displaystyle \sum _ { s } \left[ E _ { 0 } - B ( s ) + \lambda Q ( s ) \right] \| \psi _ { s } \| ^ { 2 } \quad \quad \mathrm { ( s t e p ~ 2 : ~ } E _ { 0 } ( \mathcal { H } _ { s } ) \geq E _ { 0 } - B ( s ) ) } \\ & { \geq \displaystyle \sum _ { s } E _ { 0 } \| \psi _ { s } \| ^ { 2 } \quad \quad \quad \quad \quad \quad \quad \quad \mathrm { ( s t e p ~ 5 ~ w i t h ~ } \lambda \geq \lambda _ { \operatorname* { m i n } } ) } \\ & { = \displaystyle E _ { 0 } \| \psi \| ^ { 2 } . } \end{array}
$$

For the all- <sup>1</sup> sector $s = ( \textstyle { \frac { 1 } { 2 } } , \dots , \textstyle { \frac { 1 } { 2 } } )$ the raised set is empty, $B ( s ) = 0$ and $Q ( s ) = 0$ , and the chain collapses to the physical variational bound $\langle \psi _ { \frac { 1 } { 2 } } | \hat { H } | \psi _ { \frac { 1 } { 2 } } \rangle \geq E _ { 0 } \Vert \psi _ { \frac { 1 } { 2 } } \Vert ^ { 2 }$

## B.4 Proof of Lemma A.2 (per-edge certificates)

Proof. Operator. Step (3b) of Appendix B.3 gives $\| \vec { S } _ { i } ^ { \top } J \vec { S } _ { k } \| _ { ( s _ { i } , s _ { i } ) } \leq \| J \| _ { \mathrm { o p } } r ( s _ { i } ) r ( s _ { j } )$ , and $\leq \| J \| _ { \mathrm { o p } } \frac { 3 } { 4 }$ on $\left( { \frac { 1 } { 2 } } , { \frac { 1 } { 2 } } \right)$ , so the left side of (A.6) is at most $\| J \| _ { \mathrm { o p } } [ r ( s _ { i } ) r ( s _ { j } ) + \textstyle { \frac { 3 } { 4 } } ]$ . The ratio $[ r ( s _ { i } ) r ( s _ { j } ) + { \textstyle \frac { 3 } { 4 } } ] / [ q ( s _ { i } ) + q ( s _ { j } ) ]$ over non-trivial pairs is largest at $\bigl ( \frac { 3 } { 2 } , \frac { 1 } { 2 } \bigr )$ , where it equals $( { \textstyle { \frac { \sqrt { 1 5 } } } } { \frac { \sqrt { 3 } } { 2 } } + { \frac { 3 } { 4 } } ) / 3 = ( { \textstyle { \sqrt { 5 } } } + 1 ) / 4 = { \frac { \varphi } { 2 } }$

Nuclear. For rank one $J = \sigma u v ^ { \top } , \vec { S } _ { i } ^ { \top } J \vec { S } _ { k } = \sigma ( u \cdot \vec { S } _ { i } ) ( v \cdot \vec { S } _ { k } )$ ; each factor has operator norm s (eigenvalues $- s , \ldots , s )$ , so the pair norm is $\leq \left| \sigma \right| s _ { i } s _ { j }$ , while on $\bigl ( { \textstyle { \frac { 1 } { 2 } } } , { \textstyle { \frac { 1 } { 2 } } } \bigr )$ ) the two ± <sup>1</sup> factors give $| \sigma | / 4 .$ Then $\begin{array} { r } { [ s _ { i } s _ { j } + \frac { 1 } { 4 } ] / [ q ( s _ { i } ) + q ( s _ { j } ) ] \le \frac { 1 } { 2 } } \end{array}$ , because $q ( s _ { i } ) + q ( s _ { j } ) - 2 ( s _ { i } s _ { j } + { \frac { \top } { 4 } } ) = ( s _ { i } - s _ { j } ) ^ { \tilde { 2 } } + ( s _ { i } + s _ { j } ) - 2 \ge 0$ for every non-trivial half-integer pair $\left( s _ { i } + s _ { j } \ge 2 \right)$ . Summing the rank-one pieces of the SVD gives ${ \frac { 1 } { 2 } } \parallel J \parallel ,$ .

Orthogonal. For $J = \lambda Q , Q \in { \mathcal { O } } ( 3 )$ , the rotated operators $\tilde { S } _ { k } = Q \vec { S } _ { k }$ keep the Casimir $\begin{array} { r } { \sum _ { a } ( \tilde { S } _ { k } ^ { a } ) ^ { 2 } = } \end{array}$ $s _ { k } ( s _ { k } + 1 )$ , so $\vec { S } _ { i } \cdot \tilde { S } _ { k }$ is unitarily (det $Q = + 1 )$ or anti-unitarily (det $Q = - 1 )$ equivalent to $\vec { S } _ { i } \cdot \vec { S } _ { k }$ whose pair operator norm is $s _ { i } s _ { j } + \operatorname* { m i n } ( s _ { i } , s _ { j } )$ (the extreme total-spin multiplets). The ratio $[ s _ { i } s _ { j } +$ min $+ \textstyle { \frac { 3 } { 4 } } ] / [ q ( s _ { i } ) + q ( s _ { j } ) ]$ ] peaks at $\left( { \frac { 3 } { 2 } } , { \frac { 3 } { 2 } } \right)$ , value $( { \textstyle { \frac { 9 } { 4 } } } + { \frac { 3 } { 2 } } + { \frac { 3 } { 4 } } ) / 6 = { \frac { 3 } { 4 } }$

Additivity is the triangle inequality applied to each operator-norm term in (A.6).

## B.5 Proof of Theorem A.3 (tight per-edge threshold)

Proof. By Lemma A.2 and its additivity each entry of (A.7) is a certificate (the isotropic and polar entries split J and sum a primitive on each part), and the minimum of certificates is a certificate, so $\alpha _ { \mathrm { f u l l } } ( J _ { i k } )$ satisfies (A.6). Assigning it to the endpoints (a boundary edge has $\begin{array} { r } { q ( \frac { 1 } { 2 } ) = 0 } \end{array}$ , so its whole weight lands on the raised site) and keeping the field bound $\begin{array} { r } { j + \frac { 1 } { 2 } \stackrel {  } { \le } \bar { \frac { 2 } { 3 } } \bar { q ( j ) } } \end{array}$ of Appendix B.3 (its step 5), steps (2) and (6) there go through with $\textstyle \sum _ { k } \alpha _ { \mathrm { f u l l } } ( J _ { i k } )$ in place of $\frac { \varphi } { 2 } \mathrm { w d } _ { i }$ . Since every entry of the minimum is $\begin{array} { r } { \leq \frac { \varphi } { 2 } \| J _ { i k } \| _ { \mathrm { o p } } , } \end{array}$ , we have $\mu _ { \star } \le \lambda _ { \operatorname* { m i n } }$ □

## References

[1] Bela Bauer, Sergey Bravyi, Mario Motta, and Garnet Kin-Lic Chan. Quantum algorithms for quantum chemistry and quantum materials science. Chemical reviews, 120(22):12685–12717, 2020.

[2] Sam McArdle, Suguru Endo, Al´an Aspuru-Guzik, Simon C Benjamin, and Xiao Yuan. Quantum computational chemistry. Reviews of Modern Physics, 92(1):015003, 2020.

[3] Riley W Chien, Mitchell Chiew, Brent Harrison, Jason Necaise, Weishi Wang, Maryam Mudassar, Campbell McLauchlan, Thomas M Henderson, Gustavo E Scuseria, Sergii Strelchuk, et al. Simulating fermions with a digital quantum computer. Nature Reviews Physics, 8(3):131–145, 2026.

[4] Guman Singh and Mohammad Rizwanullah. Combinatorial optimization of supply chain networks: A retrospective & literature review. Materials today: proceedings, 62:1636–1642, 2022.

[5] Fang He and Rong Qu. A two-stage stochastic mixed-integer program modelling and hybrid solution approach to portfolio selection problems. Information Sciences, 289:190–205, 2014.

[6] Nora Touati-Moungla, Vincent Jost, et al. Combinatorial optimization for electric vehicles management. Journal of Energy and Power Engineering, 6(5):738–743, 2012.

[7] JQ James, Wen Yu, and Jiatao Gu. Online vehicle routing with neural combinatorial optimization and deep reinforcement learning. IEEE Transactions on Intelligent Transportation Systems, 20 (10):3806–3817, 2019.

[8] David Garc´ıa-Heredia, Antonio Alonso-Ayuso, and Elisenda Molina. A combinatorial model to optimize air trafic flow management problems. Computers & operations research, 112:104768, 2019.

[9] Gang Yu. Industrial applications of combinatorial optimization. Springer Science & Business Media, 2013.

[10] Ding-Zhu Du and Panos M Pardalos. Handbook of combinatorial optimization. Springer Science & Business Media, 2013.

[11] Yao-Yun Shi, Lu-Ming Duan, and Guifr´e Vidal. Classical simulation of quantum many-body systems with a tree tensor network. Physical Review A, 74(2):022320, 2006. doi: 10.1103/ PhysRevA.74.022320.

[12] Valentin Murg, Ors Legeza, Reinhard M. Noack, and Frank Verstraete. Simulating strongly <sup>¨</sup> correlated quantum systems with tree tensor networks. Physical Review B, 82(20):205105, 2010. doi: 10.1103/PhysRevB.82.205105.

[13] Naoki Nakatani and Garnet Kin-Lic Chan. Eficient tree tensor network states (TTNS) for quantum chemistry: Generalizations of the density matrix renormalization group algorithm. Journal of Chemical Physics, 138(13):134113, 2013. doi: 10.1063/1.4798639.

[14] Alberto Peruzzo, Jarrod McClean, Peter Shadbolt, Man-Hong Yung, Xiao-Qi Zhou, Peter J. Love, Al´an Aspuru-Guzik, and Jeremy L. O’Brien. A variational eigenvalue solver on a photonic quantum processor. Nature Communications, 5:4213, 2014. doi: 10.1038/ncomms5213.

[15] M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, J. R. McClean, K. Mitarai, X. Yuan, L. Cincio, and P. J. Coles. Variational quantum algorithms. Nature Reviews Physics, 3:625–644, 2021. doi: 10.1038/s42254-021-00348-9.

[16] Giuseppe Carleo and Matthias Troyer. Solving the quantum many-body problem with artificial neural networks. Science, 355:602–606, 2017. arXiv:1606.02318.

[17] Hannah Lange, Anka Van de Walle, Atiye Abedinnia, and Annabelle Bohrdt. From architectures to applications: A review of neural quantum states. Quantum Science and Technology, 9(4): 040501, 2024.

[18] Martin Larocca, Supanut Thanasilp, Samson Wang, Kunal Sharma, Jacob Biamonte, Patrick J Coles, Lukasz Cincio, Jarrod R McClean, Zo¨e Holmes, and Marco Cerezo. Barren plateaus in variational quantum computing. Nature Reviews Physics, 7(4):174–189, 2025.

[19] Michael Ragone, Bojko N Bakalov, Fr´ed´eric Sauvage, Alexander F Kemper, Carlos Ortiz Marrero, Mart´ın Larocca, and Marco Cerezo. A lie algebraic theory of barren plateaus for deep parameterized quantum circuits. Nature Communications, 15(1):7172, 2024.

[20] Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, and Hartmut Neven. Barren plateaus in quantum neural network training landscapes. Nature Communications, 9: 4812, 2018. doi: 10.1038/s41467-018-07090-4.

[21] Luciano Loris Viteritti, Riccardo Rende, Subir Sachdev, and Giuseppe Carleo. Approaching the thermodynamic limit with neural-network quantum states. arXiv preprint arXiv:2602.02665, 2026.

[22] R. Rende, L. L. Viteritti, F. Becca, A. Scardicchio, A. Laio, and G. Carleo. Foundation neuralnetwork quantum states as a unified ansatz for multiple Hamiltonians. Nature Communications, 16:7213, 2025. doi: 10.1038/s41467-025-62098-x.

[23] Timothy Zaklama, Daniele Guerci, and Liang Fu. Attention-based foundation model for quantum states. arXiv preprint arXiv:2512.11962, 2025.

[24] Yuan-Hang Zhang and Massimiliano Di Ventra. Transformer quantum state: A multi-purpose model for quantum many-body problems. Phys. Rev. B, 107:075147, 2023. arXiv:2208.01758.

[25] Riccardo Rende, Sebastian Goldt, Federico Becca, and Luciano Loris Viteritti. Fine-tuning neural network quantum states. Phys. Rev. Research, 6:043280, 2024. arXiv:2403.07795.

[26] Riccardo Rende, Luciano Loris Viteritti, Federico Becca, Antonello Scardicchio, Alessandro Laio, and Giuseppe Carleo. Foundation neural-network quantum states as a unified ansatz for multiple hamiltonians. Nature Communications, 16:7213, 2025. arXiv:2502.09488.

[27] Luciano Loris Viteritti, Riccardo Rende, Giacomo Bracci-Testasecca, Jacopo Niedda, Roderich Moessner, Giuseppe Carleo, and Antonello Scardicchio. Quantum spin glass in the twodimensional disordered heisenberg model via foundation neural-network quantum states. arXiv preprint arXiv:2507.05073, 2025.

[28] David Pfau, James S. Spencer, Alexander G. de G. Matthews, and W. M. C. Foulkes. Abinitio solution of the many-electron Schr¨odinger equation with deep neural networks. Phys. Rev. Research, 2:033429, 2020. arXiv:1909.02487.

[29] Jan Hermann, Zeno Sch¨atzle, and Frank No´e. Deep-neural-network solution of the electronic Schr¨odinger equation. Nature Chemistry, 12:891–897, 2020. arXiv:1909.08423.

[30] Leon Gerard, Michael Scherbela, Philipp Marquetand, and Philipp Grohs. Gold-standard solutions to the Schr¨odinger equation using deep learning: How much physics do we need? Advances in Neural Information Processing Systems (NeurIPS), 35, 2022. arXiv:2205.09438.

[31] Timothy Zaklama, Max Geier, and Liang Fu. Large electron model: A universal ground state predictor. arXiv preprint arXiv:2603.02346, 2026.

[32] Khachatur Nazaryan and Liang Fu. QERNEL: A scalable large electron model. arXiv preprint arXiv:2604.26018, 2026.

[33] Michael Scherbela, Rafael Reisenhofer, Leon Gerard, Philipp Marquetand, and Philipp Grohs. Solving the electronic Schr¨odinger equation for multiple nuclear geometries with weight-sharing deep neural networks. Nature Computational Science, 2:331–341, 2022. arXiv:2105.08351.

[34] Michael Scherbela, Leon Gerard, and Philipp Grohs. Towards a foundation model for neural network wavefunctions. arXiv preprint arXiv:2303.09949, 2023.

[35] Nicholas Gao and Stephan G¨unnemann. Generalizing neural wave functions. International Conference on Machine Learning (ICML), 2023. arXiv:2302.04168.

[36] Adam Foster, Zeno Sch¨atzle, P. Bern´at Szab´o, Lixue Cheng, Jonas K¨ohler, Gino Cassella, Nicholas Gao, Jiawei Li, Frank No´e, and Jan Hermann. An ab initio foundation model of wavefunctions that accurately describes chemical bond breaking. arXiv preprint arXiv:2506.19960, 2025.

[37] Sebasti´an Roca-Jerat, Manuel Gallego, Fernando Luis, Jes´us Carrete, and David Zueco. Transformer wave function for quantum long-range models. Phys. Rev. B, 110:205147, 2024. arXiv:2407.04773.

[38] Du Jiang, Xuelan Wen, Yixiao Chen, Ruichen Li, Weizhong Fu, Hung Q. Pham, Ji Chen, Di He, William A. Goddard, Liwei Wang, and Weiluo Ren. Neural scaling laws surpass chemical accuracy for the many-electron Schr¨odinger equation. arXiv preprint arXiv:2508.02570, 2025.

[39] Riccardo Rende, Alessandro Sinibaldi, Luciano Loris Viteritti, Roeland Wiersema, Antoine Georges, and Giuseppe Carleo. Scaling laws for neural-network quantum states. arXiv preprint arXiv:2606.02794, 2026.

[40] Timothy Heightman, Edward Jiang, Ruth Mora-Soto, Maciej Lewenstein, and Marcin P lodzie´n. Quantum machine learning in multi-qubit phase-space Part I: Foundations. arXiv preprint arXiv:2507.12117, 2025.

[41] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Yash Katariya, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/jax-ml/jax.

[42] Nicholas Gao, Jonas K¨ohler, and Adam Foster. folx – forward laplacian for JAX. https: //github.com/microsoft/folx, 2023. Version 0.2.5.

[43] James Martens and Roger Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In International conference on machine learning, pages 2408–2417. PMLR, 2015.

[44] Michael A Nielsen, Isaac L Chuang, et al. Quantum computation and quantum information, volume 1. Cambridge university press Cambridge, 2000.

[45] Erik S Sørensen, Andrei Catuneanu, Jacob S Gordon, and Hae-Young Kee. Heart of entanglement: Chiral, nematic, and incommensurate phases in the kitaev-gamma ladder in a field. Physical Review X, 11(1):011013, 2021.

[46] Shang-Shun Zhang, G´abor B Hal´asz, Wei Zhu, and Cristian D Batista. Variational study of the kitaev-heisenberg-gamma model. Physical Review B, 104(1):014411, 2021.

[47] Hongxin Yang, Jinghua Liang, and Qirui Cui. First-principles calculations for dzyaloshinskii– moriya interaction. Nature Reviews Physics, 5(1):43–61, 2023.

[48] Brian C. Hall. Lie Groups, Lie Algebras, and Representations: An Elementary Introduction, volume 222 of Graduate Texts in Mathematics. Springer, Cham, 2 edition, 2015. doi: 10.1007/ 978-3-319-13467-3.

[49] F. Peter and H. Weyl. Die vollst¨andigkeit der primitiven darstellungen einer geschlossenen kontinuierlichen gruppe. Mathematische Annalen, 97:737–755, 1927. URL https://eudml.org/ doc/182662.

[50] Gerald B. Folland. A Course in Abstract Harmonic Analysis. CRC Press, Boca Raton, 2 edition, 2016. ISBN 9781498727136. doi: 10.1201/b19172.

[51] Anthony W. Knapp. Representation Theory of Semisimple Groups: An Overview Based on Examples, volume 36 of Princeton Mathematical Series. Princeton University Press, Princeton, NJ, 2001. ISBN 9780691090894. Paperback reprint with a new preface; originally published in 1986.

[52] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. Adaptive Computation and Machine Learning. MIT Press, Cambridge, MA, 1998. ISBN 9780262193986.

[53] Vincent Fran¸cois-Lavet, Peter Henderson, Riashat Islam, Marc G. Bellemare, and Joelle Pineau. An introduction to deep reinforcement learning. Foundations and Trends in Machine Learning, 11(3–4):219–354, 2018. doi: 10.1561/2200000071.

[54] P. W. Anderson. Absence of difusion in certain random lattices. Physical Review, 109(5): 1492–1505, 1958. doi: 10.1103/PhysRev.109.1492.

[55] Elliott Lieb, Theodore Schultz, and Daniel Mattis. Two soluble models of an antiferromagnetic chain. Annals of Physics, 16(3):407–466, 1961. doi: 10.1016/0003-4916(61)90115-4.

[56] J. Hubbard. Electron correlations in narrow energy bands. Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences, 276(1365):238–257, 1963. doi: 10.1098/ rspa.1963.0204.

[57] Chanchal K. Majumdar and Dipan K. Ghosh. On next-nearest-neighbor interaction in linear chain. I. Journal of Mathematical Physics, 10(8):1388–1398, 1969. doi: 10.1063/1.1664978.

[58] Ernst Ising. Beitrag zur theorie des ferromagnetismus. Zeitschrift f¨ur Physik, 31:253–258, 1925. doi: 10.1007/BF02980577.

[59] W. P. Su, J. R. Schriefer, and A. J. Heeger. Solitons in polyacetylene. Physical Review Letters, 42(25):1698–1701, 1979. doi: 10.1103/PhysRevLett.42.1698.

[60] M. J. Rice and E. J. Mele. Elementary excitations of a linearly conjugated diatomic polymer. Physical Review Letters, 49(19):1455–1459, 1982. doi: 10.1103/PhysRevLett.49.1455.

[61] Alexei Yu. Kitaev. Unpaired majorana fermions in quantum wires. Physics-Uspekhi, 44(10S): 131–136, 2001. doi: 10.1070/1063-7869/44/10S/S29.

[62] Xie Chen, Zheng-Cheng Gu, and Xiao-Gang Wen. Classification of gapped symmetric phases in one-dimensional spin systems. Physical Review B, 83(3):035107, 2011. doi: 10.1103/PhysRevB. 83.035107.

[63] Subir Sachdev. Quantum Phase Transitions. Cambridge University Press, Cambridge, 1999. ISBN 9780521582548. doi: 10.1017/CBO9780511622540.

[64] Anders W. Sandvik. Evidence for deconfined quantum criticality in a two-dimensional heisenberg model with four-spin interactions. Physical Review Letters, 98(22):227202, 2007. doi: 10.1103 PhysRevLett.98.227202.

[65] T. Senthil, Ashvin Vishwanath, Leon Balents, Subir Sachdev, and Matthew P. A. Fisher. Deconfined quantum critical points. Science, 303(5663):1490–1494, 2004. doi: 10.1126/science.1091806.

[66] C. Monroe, W. C. Campbell, L.-M. Duan, Z.-X. Gong, A. V. Gorshkov, P. W. Hess, R. Islam, K. Kim, N. M. Linke, G. Pagano, P. Richerme, C. Senko, and N. Y. Yao. Programmable quantum simulations of spin systems with trapped ions. Reviews of Modern Physics, 93(2):025001, 2021. doi: 10.1103/RevModPhys.93.025001.

[67] Antoine Browaeys and Thierry Lahaye. Many-body physics with individually controlled rydberg atoms. Nature Physics, 16:132–142, 2020. doi: 10.1038/s41567-019-0733-z.

[68] T. Hensgens, T. Fujita, L. Janssen, Xiao Li, C. J. Van Diepen, C. Reichl, W. Wegscheider, S. Das Sarma, and L. M. K. Vandersypen. Quantum simulation of a fermi-hubbard model using a semiconductor quantum dot array. Nature, 548(7665):70–73, 2017. doi: 10.1038/nature23022.

[69] U. Las Heras, A. Mezzacapo, L. Lamata, S. Filipp, A. Wallraf, and E. Solano. Digital quantum simulation of spin systems in superconducting circuits. Physical Review Letters, 112(20):200501, 2014. doi: 10.1103/PhysRevLett.112.200501.

[70] Alexei Kitaev. Anyons in an exactly solved model and beyond. Annals of Physics, 321(1):2–111, 2006. doi: 10.1016/j.aop.2005.10.005.

[71] P. W. Anderson. Resonating valence bonds: A new kind of insulator? Materials Research Bulletin, 8(2):153–160, 1973. doi: 10.1016/0025-5408(73)90167-0.

[72] Leon Balents. Spin liquids in frustrated magnets. Nature, 464(7286):199–208, 2010. doi: 10. 1038/nature08917.

[73] Lucile Savary and Leon Balents. Quantum spin liquids: a review. Reports on Progress in Physics, 80(1):016502, 2017. doi: 10.1088/0034-4885/80/1/016502. Published online 8 November 2016.

[74] D. M. Basko, I. L. Aleiner, and B. L. Altshuler. Metal–insulator transition in a weakly interacting many-electron system with localized single-particle states. Annals of Physics, 321(5):1126–1205, 2006. doi: 10.1016/j.aop.2005.11.014.

[75] Rahul Nandkishore and David A. Huse. Many-body localization and thermalization in quantum statistical mechanics. Annual Review of Condensed Matter Physics, 6(1):15–38, 2015. doi: 10.1146/annurev-conmatphys-031214-014726.

[76] Dmitry A. Abanin, Ehud Altman, Immanuel Bloch, and Maksym Serbyn. Colloquium: Manybody localization, thermalization, and entanglement. Reviews of Modern Physics, 91(2):021001, 2019. doi: 10.1103/RevModPhys.91.021001.

[77] Don N. Page. Average entropy of a subsystem. Physical Review Letters, 71(9):1291–1294, 1993. doi: 10.1103/PhysRevLett.71.1291.

[78] H. J. Lipkin, N. Meshkov, and A. J. Glick. Validity of many-body approximation methods for a solvable model. I. exact solutions and perturbation theory. Nuclear Physics, 62(2):188–198, 1965. doi: 10.1016/0029-5582(65)90862-X.

[79] Daniel M. Greenberger, Michael A. Horne, and Anton Zeilinger. Going beyond bell’s theorem. In Menas Kafatos, editor, Bell’s Theorem, Quantum Theory and Conceptions of the Universe, pages 69–72. Kluwer Academic Publishers, Dordrecht, 1989. doi: 10.1007/978-94-017-0849-4 10.

[80] Kenneth G. Wilson. Confinement of quarks. Physical Review D, 10(8):2445–2459, 1974. doi: 10.1103/PhysRevD.10.2445.

[81] John B. Kogut and Leonard Susskind. Hamiltonian formulation of wilson’s lattice gauge theories. Physical Review D, 11(2):395–408, 1975. doi: 10.1103/PhysRevD.11.395.

[82] John B. Kogut. An introduction to lattice gauge theory and spin systems. Reviews of Modern Physics, 51(4):659–713, 1979. doi: 10.1103/RevModPhys.51.659.

[83] Pascual Jordan and Eugene P. Wigner. Uber das paulische ¨aquivalenzverbot.<sup>¨</sup> Zeitschrift f¨ur Physik, 47(9–10):631–651, 1928. doi: 10.1007/BF01331938. English title: About the Pauli exclusion principle.

[84] Sergey B. Bravyi and Alexei Yu. Kitaev. Fermionic quantum computation. Annals of Physics, 298(1):210–226, 2002. doi: 10.1006/aphy.2002.6254.

[85] Andrew Tranter, Peter J. Love, Florian Mintert, and Peter V. Coveney. A comparison of the bravyi–kitaev and jordan–wigner transformations for the quantum simulation of quantum chemistry. Journal of Chemical Theory and Computation, 14(11):5617–5630, 2018. doi: 10.1021/acs.jctc.8b00450.

[86] J. Goldstone. Field theories with superconductor solutions. Il Nuovo Cimento, 19:154–164, 1961. doi: 10.1007/BF02812722.

[87] Yoichiro Nambu. Quasi-particles and gauge invariance in the theory of superconductivity. Physical Review, 117(3):648–663, 1960. doi: 10.1103/PhysRev.117.648.

[88] Assa Auerbach. Interacting Electrons and Quantum Magnetism. Graduate Texts in Contemporary Physics. Springer, New York, 1994. ISBN 978-0-387-94286-5. doi: 10.1007/ 978-1-4612-0869-3.

[89] Christopher J. Turner, Alexios A. Michailidis, Dmitry A. Abanin, Maksym Serbyn, and Zlatko Papi´c. Weak ergodicity breaking from quantum many-body scars. Nature Physics, 14:745–749, 2018. doi: 10.1038/s41567-018-0137-5.

[90] Hannes Bernien, Sylvain Schwartz, Alexander Keesling, Harry Levine, Ahmed Omran, Hannes Pichler, Soonwon Choi, Alexander S. Zibrov, Manuel Endres, Markus Greiner, Vladan Vuleti´c, and Mikhail D. Lukin. Probing many-body dynamics on a 51-atom quantum simulator. Nature, 551(7682):579–584, 2017. doi: 10.1038/nature24622.

[91] Ian Afleck, Tom Kennedy, Elliott H. Lieb, and Hal Tasaki. Valence bond ground states in isotropic quantum antiferromagnets. Communications in Mathematical Physics, 115:477–528, 1988. doi: 10.1007/BF01218021. The related PRL titled ”Rigorous results on valence-bond ground states in antiferromagnets” is 1987, DOI 10.1103/PhysRevLett.59.799.

[92] C. N. Yang. Concept of of-diagonal long-range order and the quantum phases of liquid He and of superconductors. Reviews of Modern Physics, 34(4):694–704, 1962. doi: 10.1103/RevModPhys. 34.694.

[93] Subir Sachdev and Jinwu Ye. Gapless spin-fluid ground state in a random quantum heisenberg magnet. Physical Review Letters, 70(21):3339–3342, 1993. doi: 10.1103/PhysRevLett.70.3339.

[94] Alexei Kitaev and S. Josephine Suh. The soft mode in the Sachdev–Ye–Kitaev model and its gravity dual. Journal of High Energy Physics, 2018(5):183, 2018. doi: 10.1007/JHEP05(2018)183.

[95] Juan Maldacena and Douglas Stanford. Remarks on the sachdev-ye-kitaev model. Physical Review D, 94(10):106002, 2016. doi: 10.1103/PhysRevD.94.106002.

[96] I. Dzyaloshinsky. A thermodynamic theory of “weak” ferromagnetism of antiferromagnetics. Journal of Physics and Chemistry of Solids, 4(4):241–255, 1958. doi: 10.1016/0022-3697(58) 90076-3.

[97] Tˆoru Moriya. Anisotropic superexchange interaction and weak ferromagnetism. Physical Review, 120(1):91–98, 1960. doi: 10.1103/PhysRev.120.91.

[98] K. I. Kugel and D. I. Khomskii. The jahn-teller efect and magnetism: transition metal compounds. Soviet Physics Uspekhi, 25(4):231–256, 1982. doi: 10.1070/ PU1982v025n04ABEH004537.

[99] Zohar Nussinov and Jeroen van den Brink. Compass models: Theory and physical motivations. Reviews of Modern Physics, 87(1):1–59, 2015. doi: 10.1103/RevModPhys.87.1.

[100] Dian Wu, Riccardo Rossi, Filippo Vicentini, Nikita Astrakhantsev, Federico Becca, Xiaodong Cao, Juan Carrasquilla, Francesco Ferrari, Antoine Georges, Mohamed Hibat-Allah, et al. Variational benchmarks for quantum many-body problems. Science, 386(6719):296–301, 2024.

[101] Fred Glover, Gary Kochenberger, and Yu Du. A tutorial on formulating and using qubo models. arXiv preprint arXiv:1811.11538, 2018.

[102] Fred Glover, Gary Kochenberger, Rick Hennig, and Yu Du. Quantum bridge analytics i: a tutorial on formulating and using qubo models. Annals of Operations Research, 314(1):141–183, 2022.

[103] Rudolph Pariser and Robert G Parr. A semi-empirical theory of the electronic spectra and electronic structure of complex unsaturated molecules. i. The Journal of Chemical Physics, 21 (3):466–471, 1953.

[104] Rudolph Pariser and Robert G Parr. A semi-empirical theory of the electronic spectra and electronic structure of complex unsaturated molecules. ii. The Journal of Chemical Physics, 21 (5):767–776, 1953.

[105] John A Pople. Electron interaction in unsaturated hydrocarbons. Transactions of the Faraday Society, 49:1375–1385, 1953.

[106] Federico Becca and Sandro Sorella. Quantum Monte Carlo approaches for correlated systems. Cambridge University Press, 2017.

[107] Shi-Jian Gu. Fidelity approach to quantum phase transitions. International Journal of Modern Physics B, 24(23):4371–4458, 2010.

[108] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jefrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

[109] Kimio Ohno. Some remarks on the pariser–parr–pople method. Theoretica Chimica Acta, 2: 219–227, 1964. doi: 10.1007/BF00528281.

[110] Zolt´an G. Soos and S. Ramasesha. Valence-bond theory of linear hubbard and pariser–parr–pople models. Physical Review B, 29:5410–5422, 1984. doi: 10.1103/PhysRevB.29.5410.

[111] Shou-Shu Gong, Wei Zhu, D. N. Sheng, Olexei I. Motrunich, and Matthew P. A. Fisher. Plaquette ordered phase and quantum phase diagram in the spin-1/2 j<sub>1</sub>–j<sub>2</sub> square heisenberg model. Physical Review Letters, 113(2):027201, 2014. doi: 10.1103/PhysRevLett.113.027201.

[112] Amazon Web Services. Amazon braket pricing. https://aws.amazon.com/braket/pricing/, 2026. Accessed 3 August 2026.

[113] David Jansen, Donato Farina, Luke Mortimer, Timothy Heightman, Andreas Leitherer, Pere Mujal, Jie Wang, and Antonio Ac´ın. Mapping phase diagrams of quantum spin systems through semidefinite-programming relaxations. Physical Review Letters, 136(5):050401, 2026.

[114] Dorit Aharonov. Quantum computation. Annual Reviews of Computational Physics VI, pages 259–346, 1999.

[115] Michel Ledoux, Ivan Nourdin, and Giovanni Peccati. Stein’s method, logarithmic sobolev and transport inequalities. Geometric and Functional Analysis, 25(1):256–306, 2015.

[116] Elena Orlova, Aleksei Ustimenko, Ruoxi Jiang, Peter Y Lu, and Rebecca Willett. Deep stochastic mechanics. arXiv preprint arXiv:2305.19685, 2023.

[117] G. Carleo and M. Troyer. Solving the quantum many-body problem with artificial neural networks. Science, 355(6325):602–606, 2017. doi: 10.1126/science.aag2302.

[118] Toby S Cubitt, Ashley Montanaro, and Stephen Piddock. Universal quantum hamiltonians. Proceedings of the National Academy of Sciences, 115(38):9497–9502, 2018.

[119] Roberto Oliveira and Barbara M Terhal. The complexity of quantum spin systems on a twodimensional square lattice. arXiv preprint quant-ph/0504050, 2005.

[120] Julia Kempe, Alexei Kitaev, and Oded Regev. The complexity of the local hamiltonian problem. Siam journal on computing, 35(5):1070–1097, 2006.

[121] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019.

[122] Yann N. DauphVaswaniin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 933–941. PMLR, 2017. URL https://proceedings.mlr.press/v70/dauphin17a.html.

[123] Noam Shazeer. GLU variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. URL https://arxiv.org/abs/2002.05202.

[124] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[125] Soham De and Sam Smith. Batch normalization biases residual blocks towards the identity function in deep networks. In Advances in Neural Information Processing Systems, volume 33, pages 19964–19975. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/ paper/2020/hash/e6b738eca0e6792ba8a9cbcba6c1881d-Abstract.html.

[126] Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, and Boris Ginsburg. nGPT: Normalized transformer with representation learning on the hypersphere. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=se4vjm7h4E.

[127] Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. Advances in neural information processing systems, 33:11525–11538, 2020.

[128] John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Z´ıdek, Anna Potapenko, et al. Highly<sup>ˇ</sup> accurate protein structure prediction with alphafold. nature, 596(7873):583–589, 2021.

[129] Jin-Yi Cai, Martin F¨urer, and Neil Immerman. An optimal lower bound on the number of variables for graph identification. Combinatorica, 12(4):389–410, 1992.

[130] Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, and Yaron Lipman. Provably powerful graph networks. Advances in neural information processing systems, 32, 2019.

[131] David Ha, Andrew M. Dai, and Quoc V. Le. HyperNetworks. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=rkpACe1lx.

[132] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. FiLM: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, pages 3942–3951, 2018. doi: 10.1609/aaai.v32i1.11671.

[133] Kenneth G Wilson. The renormalization group and critical phenomena. Reviews of Modern Physics, 55(3):583, 1983.

[134] Leo P Kadanof. Scaling laws for ising models near t c. Physics Physique Fizika, 2(6):263, 1966.

[135] Kenneth G. Wilson. The renormalization group: Critical phenomena and the Kondo problem. Reviews of Modern Physics, 47(4):773–840, 1975. doi: 10.1103/RevModPhys.47.773.

[136] Guifr´e Vidal. Entanglement renormalization. Physical Review Letters, 99(22):220405, 2007. doi: 10.1103/PhysRevLett.99.220405.

[137] Ofir Press, Noah A Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409, 2021.

[138] Ofir Press, Noah A Smith, and Mike Lewis. Shortformer: Better language modeling using shorter inputs. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5493–5505, 2021.

[139] Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer networks. In Advances in Neural Information Processing Systems, volume 28, 2015. URL https://proceedings.neurips.cc/ paper/2015/hash/29921001f2f04bd3baee84a12e98098f-Abstract.html.

[140] Emil Julius Gumbel. Statistical theory of extreme values and some practical applications: a series of lectures, volume 33. US Government Printing Ofice, 1954.

[141] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.

[142] Luitpold Babel, Irina V Chuvaeva, Mikhail Klin, and Dmitrii V Pasechnik. Algebraic combinatorics in mathematical chemistry. methods and algorithms. ii. program implementation of the weisfeiler-leman algorithm. arXiv preprint arXiv:1002.1921, 2010.

[143] IA Faradzev, Aleksandr Anatolievich Ivanov, M Klin, and AJ Woldar. Investigations in algebraic theory of combinatorial objects. Springer Science & Business Media, 2013.

[144] Nicholas Metropolis, Arianna W Rosenbluth, Marshall N Rosenbluth, Augusta H Teller, and Edward Teller. Equation of state calculations by fast computing machines. The journal of chemical physics, 21(6):1087–1092, 1953.

[145] W Keith Hastings. Monte carlo sampling methods using markov chains and their applications. Biometrika, 57(1):97–109, 1970.

[146] Fabio Falcioni, Elena Orlova, Timothy Heightman, Philip Mantrov, and Aleksei Ustimenko. Benchmarking simulacra ai’s quantum accurate synthetic data generation for chemical sciences. arXiv preprint arXiv:2511.07433, 2025.

[147] Gareth O Roberts, Andrew Gelman, and Walter R Gilks. Weak convergence and optimal scaling of random walk metropolis algorithms. The annals of applied probability, 7(1):110–120, 1997.

[148] Gareth O Roberts and Richard L Tweedie. Exponential convergence of langevin distributions and their discrete approximations. Bernoulli, 2(4):341–363, 1996.

[149] Gareth O Roberts and Jefrey S Rosenthal. Optimal scaling of discrete approximations to langevin difusions. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 60(1):255–268, 1998.

[150] Robert H Swendsen and Jian-Sheng Wang. Replica monte carlo simulation of spin-glasses. Physical review letters, 57(21):2607, 1986.

[151] Koji Hukushima and Koji Nemoto. Exchange monte carlo method and application to spin glass simulations. Journal of the Physical Society of Japan, 65(6):1604–1608, 1996.

[152] Christophe Andrieu and Johannes Thoms. A tutorial on adaptive mcmc. Statistics and computing, 18(4):343–373, 2008.

[153] Mark Girolami and Ben Calderhead. Riemann manifold Langevin and Hamiltonian Monte Carlo methods. Journal of the Royal Statistical Society, Series B, 73(2):123–214, 2011.

[154] Simon Byrne and Mark Girolami. Geodesic Monte Carlo on embedded manifolds. Scandinavian Journal of Statistics, 40(4):825–845, 2013.

[155] Saifuddin Syed, Alexandre Bouchard-Cˆot´e, George Deligiannidis, and Arnaud Doucet. Nonreversible parallel tempering: A scalable highly parallel mcmc scheme. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 84(2):321–350, 2022.

[156] Yves F Atchad´e, Gareth O Roberts, and Jefrey S Rosenthal. Towards optimal scaling of metropolis-coupled markov chain monte carlo. Statistics and Computing, 21(4):555–568, 2011.

[157] Frederick N Fritsch and Ralph E Carlson. Monotone piecewise cubic interpolation. SIAM Journal on Numerical Analysis, 17(2):238–246, 1980.

[158] James M Hyman. Accurate monotonicity preserving cubic interpolation. SIAM Journal on Scientific and Statistical Computing, 4(4):645–654, 1983.

[159] Herbert Robbins and Sutton Monro. A stochastic approximation method. The annals of mathematical statistics, pages 400–407, 1951.

[160] Ruichen Li, Haotian Ye, Du Jiang, Xuelan Wen, Chuwei Wang, Zhe Li, Xiang Li, Di He, Ji Chen, Weiluo Ren, et al. A computational framework for neural network-based variational monte carlo with forward laplacian. Nature Machine Intelligence, 6(2):209–219, 2024.

[161] Aleksandar Botev, Hippolyt Ritter, and David Barber. Practical gauss-newton optimisation for deep learning. In International Conference on Machine Learning, pages 557–565. PMLR, 2017.

[162] Sandro Sorella. Green function monte carlo with stochastic reconfiguration. Physical review letters, 80(20):4558, 1998.

[163] Timothy Heightman and Marcin P lodzie´n. Deep learning in classical and quantum physics. arXiv preprint arXiv:2508.10666, 2025.

[164] James Martens, Jimmy Ba, and Matt Johnson. Kronecker-factored curvature approximations for recurrent neural networks. In International Conference on Learning Representations, 2018.

[165] Jeremy M Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. arXiv preprint arXiv:2103.00065, 2021.
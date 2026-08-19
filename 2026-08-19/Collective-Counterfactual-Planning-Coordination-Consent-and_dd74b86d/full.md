# Collective Counterfactual Planning: Coordination, Consent, and Verification under Representational Constraints

Chainarong Amornbunchornvej

National Electronics and Computer Technology Center (NECTEC), NSTDA Pathum Thani, Thailand [chainarong.amo@nectec.or.th]

August 19, 2026

## Abstract

Groups routinely complete projects that no single member can plan, execute, or verify alone. We propose a formal model of this phenomenon, Collective Counterfactual Planning (CCP), in which the binding limitation on each agent is neither capability, nor knowledge, nor observability, but representational geometry: each agent i perceives the shared state, con ceives prospective moves, consents to actions, and certifies goal requirements only through an orthogonal projection onto an agent-specific subspace V<sub>i</sub> of a common task space (the subspaces themselves are common knowledge; only consent thresholds are private). Four gates — the exogenous implementation coalitions R(u), together with three representational gates: conception C<sub>i</sub>(x, O), consent thresholds on interpretability, and task-relative verification qualification Q — jointly determine whether a team can reach a conjunctive goal $O = \cap _ { k } O _ { k }$ and legitimately recognize that it has done so. We define the Collective Counterfactual Solvability (CCS) problem and separate three semantic levels: geometric feasibility, executable attainment, and validated completion. The main results expose a positive– negative duality. Iterated cross-agent relay can unlock a solution that no one-shot pooling of individual plans contains, but any goal requirement depending essentially on the subspace dark to the entire team is unverifiable and therefore not validly completable, even when the physical trajectory accidentally attains it. Memoryless and audited consent further constrain diferent objects — action directions versus cumulative trajectory states — and neither dominates the other. On the constructive side, a four-step exhaustive horizon-bounded solvability scheme (coverage preflight, relay closure, ratification, terminal inspection) is sound and complete under exact representation of the complete relay closure; restricted implementations remain sound on returned plans but need not be complete. The model thereby gives one geometry for sequential mutual enabling, competent execution of steps whose purpose is invisible to the executor, forced sub-teaming at expertise boundaries, and completion that cannot be validly declared.

## 1 Introduction

Consider building a house. No individual on the project — architect, structural engineer, electrician, plumber, inspector — represents the entire task. Each understands some aspects of the current state of the project and is blind to others; at each stage, someone sees a useful next move that others cannot yet see; every move requires the participation of specific tradespeople, each of whom will act only on work they suficiently understand; and the project is finished not when the building happens to satisfy the code, but when each requirement has been certified by someone qualified to check it. The team does not need to be able to build every possible house. It needs a collection of people whose combined representational and practical competencies sufice to plan, execute, and verify this house.

This paper proposes a minimal formal model of that situation. Its central commitment is that the limitation doing the work is representational: what an agent can perceive, think of, agree to, and certify is bounded by the subspace of the task’s state space that the agent can represent. This is a deliberately diferent modeling choice from the three dominant traditions in multi-agent planning, which locate agent limitation in capability — which operators an agent owns [7, 44] — in knowledge — which epistemic states an agent can distinguish [5, 14] — or in observability — which signals an agent receives [4, 32]. The point of the contrast is not that those frameworks cannot model persistent failure; each is expressive. The point is that CCP holds those channels conceptually separate and makes representational structure itself the explicit object of analysis — and, so isolated, it yields limits of its own kind: failures that persist because the missing dimension is not information an agent lacks but structure the agent cannot host.

The linear-algebraic substrate follows the cognitive-geometric framework of [1] and is consonant with the broader case that concepts and cognitive states are usefully modeled as vectors in structured spaces [17, 27, 33, 36]. Each agent i carries a subspace V<sub>i</sub> of a common task space. Four gates then govern collective action, and they are not of one kind. Implementation is exogenous task structure: each action names the coalition of agents physically or institutionally required to perform it, and nothing about R(u) is derived from geometry. The other three are induced by the agents’ representational geometry: conception (an agent can generate a move only if it makes apparent progress in the agent’s projection of the task), consent (a required agent participates only in moves that are suficiently interpretable to that agent), and verification (an agent is qualified to certify a goal requirement only if all distinctions relevant to that requirement are visible in the agent’s subspace).

## Contributions.

1. Model. A parsimonious formal setting (Section 3) that holds the exogenous implementation structure R(u) separate from three gates induced by the agents’ representational geometry — conception, consent, and qualification; none of the four implies another.

2. Problem. The Collective Counterfactual Solvability problem (Section 4), with three semantic levels separating geometric feasibility, executable attainment, and validated completion. Objective attainment and validated completion come apart by design.

3. Collective reach and collective blindness. Iterated cross-agent re-expansion can make a solution available that is absent from every one-shot individual tree, and the enabling signal must pass through a component visible to the next agent (Theorem 2, Corollary 1). Dually, a requirement that depends essentially on the team’s dark subspace has no qualified verifier and cannot be validly completed (Theorem 1).

4. Consent geometry. Memoryless consent gates action directions, whereas audited consent gates cumulative trajectory states. The resulting solvability notions are incomparable (Theorems 3–4), and the same geometry can force non-plenary sub-teaming (Proposition 2).

5. Exhaustive solvability scheme and correctness. A four-step exhaustive horizonbounded scheme — coverage preflight, relay closure, ratification, terminal inspection — is sound and complete under exact representation of the complete relay closure (Theorem 5). A restricted practical search is still sound on any returned plan, but may miss existing solutions (Corollary 2).

The model is deliberately non-strategic: agents are truthful, share the goal, and consent mechanically. What remains when incentives, deception, and disagreement are all removed is the paper’s subject — the purely representational obstacles to collective planning, and the sense in which teamwork, sub-teaming, and institutionalized verification are forced by geometry rather than chosen by convention.

## 2 Related work

Multi-agent planning. MA-STRIPS and its successors model heterogeneity through operator ownership [7]; cooperative multi-agent planning is surveyed in Torre˜no et al. [44]; the complexity landscape of classical planning is anchored by Bylander [8] and its textbook treatment [19]. Epistemic planning locates the limitation in knowledge, with Dynamic Epistemic Logic as the standard vehicle [5]. Closest to our relay phenomenon is implicit coordination [14], where perspective shifts allow one agent to plan on the assumption that another will recognize and continue the plan; the mechanism there is epistemic indistinguishability, whereas ours is projection geometry, and no analogue of our span-based impossibility or verification qualification arises. Decentralized POMDPs locate the limitation in observability and yield the field’s canonical hardness results [4, 32]. Required-cooperation analysis [50] characterizes when cooperation is unavoidable under capability heterogeneity; our Proposition 2 is the representational counterpart. Team formation with required skills [25, 28] is the discrete ancestor of the team-synthesis problem our model raises but defers.

Coordination and collective agency. The classical accounts locate what coordination needs in salience [40], equilibrium structure [11, 45], or common knowledge and its fragility [15, 39]; group agency is treated in List and Pettit [31], cooperative problem solving in Wooldridge [48], Wooldridge and Jennings [49], and open problems in cooperative AI in Dafoe et al. [12]. Agreement dynamics on time-varying interaction networks provides a complementary mathematical account of collective convergence [9]. Amornbunchornvej et al. formalize coordination events, initiators, and following mechanisms from dynamic interaction data [3]; Amornbunchornvej and Berger-Wolf model heterogeneous following strategies through which individuals respond to neighbors, particular individuals, or combinations of influences in producing grouplevel coordination [2]. Consensus among agents with dissimilar cognitive architectures is studied information-theoretically by Sowinski et al. [42], and mutual-intelligibility constraints between linguistic agents by Komarova and Niyogi [26]. Each of these presupposes the representational frame within which focal points are focal and messages are messages; the present model makes that presupposed layer the variable. Mathematically, the nearest neighbor is sheaf-theoretic coordination over heterogeneous stalks with linear restriction maps [21]; the aims difer — consensus dynamics there, planning, executability, and verification here.

The phenomenon. That groups accomplish what no member represents is documented in distributed cognition [24] and epistemic dependence [22, 23]; “who knows what” as a team resource is the subject of transactive memory [30]. On the organizational side, bounded rationality [41], organization design as information processing under cognitive limits [16], and organizations as interpretation systems [13] anticipate the sub-teaming result; verification as institution [37] anticipates the coverage preflight; and invisible or normalized failure [35, 46, 47] is the empirical face of representationally undetectable failure. Shared intentionality and common ground [10, 43] ground the shared task and the possibility of joint action.

Terminology. “Counterfactual” here means within-basis prospective rollout — forward simulation of moves from the current state — and is distinct from the backward-looking causal senses of Pearl [34] and Lewis [29], as well as from models in which the representational basis itself changes. The psychology of such forward simulation is prospection [20]. Actions as vectors in conceptual spaces are treated by G¨ardenfors and Warglien [18], for single events rather than multi-agent planning.

## 3 The model

![](images/a813192ff7386d139031c3eb45bdb9be25ccce6c5cf6150922c9e0b0bc4b09ed.jpg)  
Figure 1: The four gates of CCP. Implementation $R ( u )$ is exogenous: it fixes which coalition may perform a direction, independent of anyone’s geometry. The remaining three gates are all induced by the agents’ subspaces but ask diferent questions: conception asks whether a move is visible progress to the agent who would propose $i t ;$ consent asks whether the required implementers find the move (memoryless) or the resulting cumulative state (audited) suficiently interpretable; verification asks whether some agent’s projection decides a requirement exactly. A move must clear implementation, conception, and consent to execute (Section 3); a goal must clear verification, at every requirement, to be validly declared complete (Definition 9).

## 3.1 Agents, task, actions

Definition 1 (Agents and spans). $N = \{ 1 , \ldots , n \}$ agents act in an ambient task space $V = \mathbb { R } ^ { d } ,$ the frame induced by the goal. Each agent i carries a subspace $V _ { i } \subseteq V -$ the directions i can represent — with orthogonal projection $P _ { i }$ onto $V _ { i }$ and complement projection $P _ { i } ^ { \perp }$ . The threshold $\tau _ { i } \in ( 0 , 1 ]$ is the minimum interpretability i requires before consenting; its angular form is $\theta _ { i } = \operatorname { a r c c o s } \sqrt { \tau _ { i } } .$ with bundling constant $\kappa ( \tau _ { i } ) = \tan \theta _ { i }$ . The group span is $S = V _ { 1 } + \cdots + V _ { n }$ — everything somebody can see; its orthogonal complement $S ^ { \perp }$ is the dark subspace — what nobody can see. Point-to-set distance is dist $( y , A ) = \operatorname* { i n f } _ { a \in A } \| y - a \|$

Definition 2 (Task). The start is $x _ { 0 } \in V ;$ the goal is a public conjunction of requirements $\textstyle O = \bigcap _ { k = 1 } ^ { m } O _ { k }$ with $O _ { k } \subseteq V ;$ ; the demand set is $\Delta = { O } - { x } _ { 0 }$ . Diferent requirements may concern diferent subspaces; all competence notions below are relative to this task only.

Definition 3 (Actions as directions). The resource U is a finite set of unit vectors in V. Each direction $u \in U$ carries a required coalition $R ( u ) \subseteq N$ , its necessary implementers: u can be performed only by these agents acting jointly; if any one is missing, u is unavailable. A move is a token $( u , \lambda )$ with duration $\lambda > 0$ , under jump dynamics $x _ { t + 1 } = x _ { t } + \lambda _ { t } u _ { t }$

## 3.2 Conception

Definition 4 (Counterfactual availability). Agent i can conceive u at state x if moving along u makes progress toward the task in i’s projection:

$$
u \in C _ { i } ( x , O ) \iff \exists \lambda > 0 : \mathrm { ~ d i s t } \big ( P _ { i } ( x + \lambda u ) , P _ { i } O \big ) < \mathrm { d i s t } \big ( P _ { i } x , P _ { i } O \big ) .\tag{1}
$$

Progress here is apparent, not actual: projected progress need not be progress in V . Two consequences are free from the definition: $C _ { i } ( x , O )$ depends on x only through $P _ { i } x -$ conception is computed inside the agent’s subspace — and $P _ { i } u = 0 \Rightarrow u \not \in C _ { i } ( x , O )$ : no one conceives a move he cannot see at all. Conception and consent are distinct gates: $u \in C _ { i } ( x , O )$ means i can see the point of the move from here; $u \in K _ { i }$ (below) means i interprets it suficiently to consent to executing it. Neither implies the other.

## 3.3 Interpretability and consent

Definition 5 (Interpretability; consent cones). The interpretability of a vector $v \neq 0$ to agent j is

$$
I _ { j } ( v ) = \frac { \| P _ { j } v \| ^ { 2 } } { \| v \| ^ { 2 } } = \cos ^ { 2 } \angle ( v , V _ { j } ) ,\tag{2}
$$

the fraction of $v \mathrm { { s } }$ squared length lying inside $j ^ { \prime }$ s subspace; $I _ { j } ( \lambda v ) = I _ { j } ( v )$ for $\lambda > 0 -$ magnitude is invisible to interpretability. By convention, $I _ { j } ( 0 ) = 1$ . The consent cone is

$$
K _ { j } = \big \{ v \in V : \ \| P _ { j } ^ { \perp } v \| \leq \tan \theta _ { j } \| P _ { j } v \| \big \} ;
$$

for $v \ \ne \ 0 , \ v \in \ K _ { j } \iff I _ { j } ( v ) \ \ge \ \tau _ { j } \iff \ \angle ( v , V _ { j } ) \le \theta _ { j } . \ K _ { j }$ is a nonconvex double cone: $I _ { j } ( v ) = I _ { j } ( - v )$ , so consent sees axes, not arrows.

Definition 6 (Consent models; executability). Under memoryless consent, j consents to $( u , \lambda )$ if $u \in K _ { j } ;$ ; duration plays no role — consenting to a direction is consenting to any distance along it. Under audited consent, at cumulative displacement $D , j$ consents if the new total stays interpretable: $I _ { j } ( D + \lambda u ) \ge \tau _ { j }$ , audited at jump endpoints. Implementation requires the participation of every necessary agent, and each participates only in steps he interprets: $( u , \lambda )$ executes at D if every $j \in R ( u )$ consents under the chosen model. Sovereign execution: consent has no substitute.

## 3.4 Plans, perception, knowledge

Definition 7 (Plans). A plan is $p = ( ( u _ { 1 } , \lambda _ { 1 } ) , \dots , ( u _ { T } , \lambda _ { T } ) )$ ) with skeleton $w = ( u _ { 1 } , \dots , u _ { T } )$ coalitions $R ( u _ { t } )$ are public annotation. The crew is crew $\begin{array} { r } { ( p ) = \bigcup _ { t < T } R ( u _ { t } ) } \end{array}$ , the personnel roster. Consent is strictly per-step: step t needs yeses from $R ( u _ { t } )$ only; nobody ever consents to a plan. The trajectory is $x _ { t } = x _ { 0 } + D _ { t }$ with $\begin{array} { r } { D _ { t } = \sum _ { s < t } \lambda _ { s } u _ { s } } \end{array}$ and final state $x _ { T }$ . Durations are chosen at extension time (availability is evaluated at realized states), so plans are generated already instantiated. Tokens are public: every agent computes $P _ { j } D _ { t }$ himself; the executed past is public arithmetic.

Perception and knowledge. Agent i perceives the state and each requirement only through his projection, $P _ { i } ( x _ { 0 } + D )$ and $P _ { i } O _ { k } ;$ components in $V _ { i } ^ { \perp }$ are dark to i. All subspaces $V _ { 1 } , \ldots , V _ { n }$ are public (common knowledge); the thresholds $\tau _ { j }$ are the model’s only private information. All agents are truthful: queried consent values and reported inspections are returned as they are. The model concerns representational limits, not strategic behavior.

## 3.5 Verification

Definition 8 (Task-relative qualification). The qualified verifiers of requirement $O _ { k }$ are

$$
Q _ { k } = \big \{ i \in N : \ \forall x , y \in V , P _ { i } x = P _ { i } y \ \Rightarrow \ ( x \in O _ { k } \iff y \in O _ { k } ) \big \} .\tag{3}
$$

Equivalently, $i \in Q _ { k }$ if $O _ { k } = P _ { i } ^ { - 1 } ( P _ { i } O _ { k } )$ : membership in $O _ { k }$ is completely determined by i’s projection, and the qualified check $P _ { i } x \in P _ { i } O _ { k }$ is sound and complete for that requirement.

Qualification is task-relative only: $i \in Q _ { k }$ implies neither $V _ { i } = V$ nor qualification for any other requirement; diferent requirements may have diferent qualified verifiers. Competence is not completion: $Q _ { k } \neq \emptyset$ means someone can decide $O _ { k }$ , not that $O _ { k }$ holds.

Definition 9 (Cover; executable stopping). The verification cover holds if $Q _ { k } \neq \emptyset$ for every k — a state-independent property of the team–task pair. Executable stopping is

$$
{ \mathrm { S T O P } } ( x ) \iff \forall k \ \exists i \in Q _ { k } : \ P _ { i } x \in P _ { i } O _ { k } .\tag{4}
$$

Under the cover, STOP $( x ) \iff x \in O$ : qualified checks decide the objective exactly. Without the cover, landing in $O$ is invisible as an achievement. Objective attainment $\neq$ validated project completion.

## 3.6 Plan language: relay closure

Definition 10 (Extension; closure). Agent i may append token $( u , \lambda )$ to a plan whose terminal state is x if: $( 1 ) i \in R ( u )$ — the extender is one of u’s necessary implementers; (2) $u \in C _ { i } ( x , O )$ ， with λ chosen so that dist $( P _ { i } ( x + \lambda u ) , P _ { i } O ) < \mathrm { d i s t } ( P _ { i } x , P _ { i } O )$ — the move is sized to make apparent progress; (3) under memoryless consent, $u \ \in \ K _ { i }$ . Extenders leave no trace: who thought of a step is not part of $p .$ The closure: each agent grows his counterfactual tree from $x _ { 0 }$ and publishes it whole; take the union; every agent re-expands from every new node’s state; repeat to a fixed point, to horizon h (a depth bound; results are h-relative); write $L _ { h }$ for this horizon-h closure.

Prefix-dependence is real: availability at a node depends on the state its prefix created, so $L _ { h }$ has no closed form over a fixed alphabet — the relay iteration is essential, not distributed bookkeeping. In the comparison variant with exogenously state-independent availability, orderirrelevance returns for the memoryless model and $T \leq d$ sufices (Proposition 4).

## 4 The problem

Definition 11 (CCS). Collective Counterfactual Solvability.

Given $( V , \{ V _ { i } \} , \{ \tau _ { i } \} , U , R , x _ { 0 } , \{ O _ { k } \} _ { k = 1 } ^ { m } ;$ consent model; horizon h): does a plan $p \in L _ { h }$ exist such that every step executes and STOP(x<sub>T</sub>) holds? The synthesis version returns p or ⊥.

Three semantic levels. (1) Geometric: $\Delta \cap \operatorname { c o n e } ( U ) \neq \emptyset$ , where $\mathrm { c o n e } ( U )$ is the set of nonnegative combinations of directions — could the available physical moves ever add up? (2) Attainment: an executable relay plan lands in $O -$ the group can conceive and execute a route to the objective. (3) Validated $( = \mathrm { C C S } \ \mathrm { y e s } )$ : attainment together with a verification cover — the group can also legitimately recognize that it is done. The gap between levels 1 and 2 is created by conception and consent; the gap between 2 and 3 is created by verification. Discoverability is treated separately as an algorithmic property: for an exhaustive horizon-bounded search it coincides with validated solvability by Theorem 5, while a restricted practical search may be incomplete.

## 5 Results

All results are stated for the model of Section 3. We first fix a running example; all its numerical claims are machine-checked.

Example 1 (E1). Two agents, two actions, and a two-part goal in the plane $( d = 2 )$

1. Agents. Agent 1 sees only the horizontal axis $( V _ { 1 } = \mathrm { s p a n } ( e _ { 1 } ) )$ ; agent 2 sees only the vertical axis $( V _ { 2 } = \mathrm { s p a n } ( e _ { 2 } ) )$ . Both demand $\tau _ { 1 } = \tau _ { 2 } = 0 . 8$ , a consent half-angle of $\theta \approx$ $2 6 . 5 7 ^ { \circ }$

2. Actions. $U = \{ u _ { 1 } , u _ { 2 } \}$ . Direction $u _ { 1 } = ( \cos \alpha , \sin \alpha )$ with $\alpha = 2 0 ^ { \circ }$ is nearly horizontal but slightly tilted, and only agent 1 can perform it $( R ( u _ { 1 } ) = \{ 1 \} )$ . Direction $u _ { 2 } = ( 0 , - 1 )$ points straight down, and only agent 2 can perform it $\left( R ( u _ { 2 } ) = \{ 2 \} \right)$

3. Task. $O _ { 1 } = \{ x : \langle x , e _ { 1 } \rangle = 1 \}$ : bring the horizontal coordinate to 1. $O _ { 2 } = \{ x : \langle x , e _ { 2 } \rangle = 0 \}$ end with the vertical coordinate at 0. So $O = \{ ( 1 , 0 ) \}$ and $x _ { 0 } = 0$ . Each requirement is cylindrical over its owner’s axis, so $1 \in Q _ { 1 }$ and $2 \in Q _ { 2 }$ : the cover holds, and each agent certifies exactly his own requirement.

4. Why agent 2 cannot begin. At $x _ { 0 }$ the vertical coordinate is already 0: agent 2’s projected distance to his goal is zero, no move can strictly improve it, and so u $\not \in C _ { 2 } ( x _ { 0 } , O )$ From where agent 2 stands, the job looks finished before it has started.

5. Agent 1’s step. Agent 1 conceives $u _ { 1 }$ (it reduces his horizontal distance, so $u _ { 1 } ~ \in$ $C _ { 1 } ( x _ { 0 } , O ) )$ ) and consents to it $( I _ { 1 } ( u _ { 1 } ) = \cos ^ { 2 } 2 0 ^ { \circ } \approx 0 . 8 8 3 \geq 0 . 8 )$ . Choosing $\lambda _ { 1 } = 1 /$ cos α lands his coordinate exactly on target: the state becomes $x _ { 1 } = ( 1 , \tan \alpha ) \approx ( 1 , 0 . 3 6 4 )$ . The tilt is the point: since $P _ { 2 } u _ { 1 } = \sin \alpha \neq 0$ , the step disturbs the vertical coordinate that agent 2 watches.

6. The flip. At $x _ { 1 }$ agent 2 now sees a problem $( 0 . 3 6 4 \neq 0 )$ , so $u _ { 2 } \in C _ { 2 } ( x _ { 1 } , O )$ : a move inconceivable at $x _ { 0 }$ has become conceivable because of someone else’s action. Agent 2 consents $( I _ { 2 } ( u _ { 2 } ) = 1 )$ and executes $\lambda _ { 2 } = \tan \alpha$ , reaching $x _ { T } = ( 1 , 0 ) \in O$

7. Finish. Agent 1 inspects $O _ { 1 }$ , agent 2 inspects $O _ { 2 } ;$ both pass, so $\mathtt { S T O P } ( x _ { T } )$ holds. All quantities are machine-checked. This one instance witnesses the relay of Theorem 2, the loop inversion of Proposition 3, and the memoryless direction of Theorem 4.

Theorem 1 (Unverifiability from darkness). Suppose some requirement $O _ { k }$ depends essentially on $S ^ { \perp }$ : there exist $x \in O _ { k }$ and $x ^ { \prime } = x + z \notin O _ { k }$ with $z \in S ^ { \bot } \setminus \{ 0 \}$ . Then $Q _ { k } = \emptyset .$ ; hence no verification cover exists and $C C S = n o \mathrm { ~ - ~ } f o r$ every x<sub>0</sub>, $U , R _ { i }$ , and consent model, even $i f$ the physical trajectory lands in O.

Proof. For every i, $V _ { i } \subseteq S$ , so $S ^ { \bot } \subseteq V _ { i } ^ { \bot }$ and $P _ { i } z = 0 ;$ hence $P _ { i } x = P _ { i } x ^ { \prime }$ while $x \in O _ { k }$ and $x ^ { \prime } \notin O _ { k }$ . No i satisfies (3), so $Q _ { k } = \emptyset$ . STOP requires a qualified certifier for every requirement, so it is false at every state, and no plan satisfies Definition 11. □

Remark 1 (Conception is equally blind). If $u \in U$ lies wholly in $S ^ { \perp }$ then $P _ { i } u = 0$ for all i, so by Definition 4 u is never conceivable and never appendable: deliberate motion along $S ^ { \perp }$ occurs only as a side-component of visible directions. Unsolvability and its undetectability coincide on $S ^ { \perp }$

Theorem 2 (Relay). There exist instances on which the one-shot union of the agents’ counterfactual trees at x contains no plan reaching O, while the fixed-point closure contains a plan achieving STOP.

Proof. Example E1. One-shot: agent $2 \mathrm { { ^ { \circ } s } }$ tree is the bare root — the only direction with $2 \in R$ is $u _ { 2 }$ , and u<sub>2</sub> $\not \in { C } _ { 2 } ( x _ { 0 } , O )$ ; agent 1’s tree contains only $u _ { 1 } \mathrm { - c h a i n s }$ , and after any prefix the $e _ { 2 ^ { - } }$ coordinate equals $( \sum \lambda )$ sin $\alpha > 0$ , so no node lies in O. Closure: agent 1 publishes the branch to $x _ { 1 } = ( 1 , \tan \alpha )$ ; re-expanding from $x _ { 1 } .$ , agent 2 has $u _ { 2 } \in C _ { 2 } ( x _ { 1 } , O )$ with $\lambda _ { 2 } = \tan \alpha$ realizing projected progress tan $\alpha  0 , 2 \in R ( u _ { 2 } )$ , and $u _ { 2 } \in K _ { 2 } ;$ the extended plan reaches $( 1 , 0 ) \in O ;$ every memoryless consent passes; the cover holds and both inspections pass. □

Corollary 1 (Relays propagate only through visible components). Since $C _ { i }$ depends on the state only through $P _ { i } x , P _ { 2 } u _ { 1 } = 0 \Rightarrow P _ { 2 } x _ { 1 } = P _ { 2 } x _ { 0 } \Rightarrow C _ { 2 } ( x _ { 1 } , O ) = C _ { 2 } ( x _ { 0 } , O )$ : a move completely invisible to agent 2 cannot make any new counterfactual move available to agent 2. Genuine relay requires the preceding action to alter some component visible in the next agent’s representational space. In E1 the flip exists exactly because $P _ { \mathrm { 2 } } u _ { \mathrm { 1 } } = \sin \alpha \neq 0$

Theorem 3 (Dichotomy; visitation-local audit). (i) Memoryless consent imposes no magnitude constraint on an accepted direction: for any $j \in R ( u )$ with $u \in K _ { j }$ , consent to $( u , \lambda )$ holds for every $\lambda > 0$ , by scale invariance $I _ { j } ( \lambda u ) = I _ { j } ( u )$ Consent alone therefore never bounds $\| P _ { j } ^ { \perp } D \|$ whenever some accepted direction has $P _ { j } ^ { \bot } u \neq 0 .$ : auditing is consent’s only brake on magnitude. (The conception rule’s progress clause does bound the chosen duration at generation $- \textit { a }$ conception constraint, not a consent one; the claim here concerns consent $o n l y . \nearrow \left( i i \right)$ Under audited consent, for every step t with $j \in R ( u _ { t } )$ , consent is equivalent to $\| P _ { j } ^ { \perp } D _ { t } \| \leq$ tan $\theta _ { j } \parallel P _ { j } D _ { t } \parallel$ . The constraint is visitation-local: it binds only at $j ^ { \prime } s$ required steps; it is enforced at all $\textit { t i f f j }$ is a plenary auditor $( j \in R ( u _ { t } )$ for every t); and the endpoint is constrained by j if $j \in R ( u _ { T } )$

Proof. (i) is the scale-invariance of $I _ { j }$ in (2), read at the consent gate. (ii): for $D \neq 0 , I _ { j } ( D ) \geq$ τ ⇐⇒ $\begin{array} { r } { \| P _ { j } D \| ^ { 2 } \geq \tau ( \| P _ { j } D \| ^ { 2 } + \| P _ { j } ^ { \bot } D \| ^ { 2 } ) \iff \| P _ { j } ^ { \bot } D \| ^ { 2 } \leq \frac { 1 - \tau } { \tau } \| P _ { j } D \| ^ { 2 } } \end{array}$ , and $\sqrt { ( 1 - \tau ) / \tau } =$ tan $\theta _ { j } ;$ at $D = 0$ both sides hold trivially, using the convention $I _ { j } ( 0 ) = 1$ . The locality clauses restate Definition 6. □

Example 2 $( \mathrm { E 2 } ) . \ d = 2 , \ V _ { 1 } = \mathrm { s p a n } ( e _ { 1 } ) , \ V _ { 2 } = \mathrm { s p a n } ( e _ { 2 } ) , \ \tau _ { 1 } = 0 . 8 , \ \tau _ { 2 } = 0 . 0 5 ; \ U = \{ e _ { 1 } , e _ { 2 } \}$ with $R ( e _ { 1 } ) = \{ 1 \} , R ( e _ { 2 } ) = \{ 1 , 2 \} ; O _ { 1 } = \{ x : \langle x , e _ { 1 } \rangle = 1 \} , O _ { 2 } = \{ x : \langle x , e _ { 2 } \rangle = 0 . 3 \} ; x _ { 0 } = 0$ . The cover holds $( 1 \in Q _ { 1 } , 2 \in Q _ { 2 } )$

Theorem 4 (Consent-model incomparability). Neither consent model’s solvable set contains the other: E1 is memoryless-solvable and audited-unsolvable, while $E \mathcal { Q }$ is audited-solvable and memoryless-unsolvable.

Proof. E2, audited-solvable: the plan $( e _ { 1 } , 1 ) , ( e _ { 2 } , 0 . 3 )$ is generated (agent 1 extends the first step with P<sub>1</sub>-progress to 0; agent 2 extends the second, $e _ { 2 } ~ \in ~ C _ { 2 }$ at $( 1 , 0 )$ with $\lambda = 0 . 3$ realizing progress) and ratified: $I _ { 1 } ( ( 1 , 0 ) ) = 1 , I _ { 1 } ( ( 1 , 0 . 3 ) ) = 1 / 1 . 0 9 \approx 0 . 9 1 7 \geq 0 . 8$ , and $I _ { 2 } ( ( 1 , 0 . 3 ) ) \approx$ $0 . 0 8 3 \geq 0 . 0 5$ ; the endpoint lies in O and both inspections pass. $E \mathcal { Q } ,$ , memoryless-unsolvable: any plan reaching O must change the e<sub>2</sub>-coordinate, hence contains an $e _ { 2 }$ step, whose coalition includes agent 1; but $I _ { 1 } ( e _ { 2 } ) = 0 < \tau _ { 1 }$ , so agent 1 never consents memorylessly.

$E \boldsymbol { { 1 } }$ , memoryless-solvable is Theorem 2. E1, audited-unsolvable: note first that every plan in $L _ { h }$ begins with $u _ { 1 } \colon$ at x<sub>0</sub>, $C _ { 2 } ( x _ { 0 } , O ) = \emptyset$ blocks $u _ { 2 }$ , and $2 \notin R ( u _ { 1 } )$ blocks agent 2 entirely; consequently the cumulative $e _ { 1 }$ -coordinate is at least $\lambda _ { \mathrm { f i r s t } }$ cos $\alpha > 0$ forever, since $u _ { 2 }$ has zero $e _ { 1 ^ { - } }$ component and $u _ { 1 }$ adds positively. Reaching O requires final $e _ { 2 } – \mathrm { c o o r d i n a t e } \ 0 .$ , so a plan reaching O contains a last $u _ { 2 }$ step; let v be the $e _ { 2 ^ { - } }$ -coordinate immediately after it and $\mu \geq 0$ the total u -duration after it. Then $v = - \mu \sin \alpha .$ , and the e -coordinate at that step is $1 - \mu$ cos $\alpha > 0$ by the first-step argument. If $\mu = 0$ , the $u _ { 2 }$ step is last and its audit is $I _ { 2 } ( D _ { T } ) = I _ { 2 } ( ( 1 , 0 ) ) = 0 <$ τ<sub>2</sub>; reject. If $\mu > 0$ , the audit of that $u _ { 2 }$ step requires $| v | \geq$ cot $\theta _ { 2 } \cdot ( 1 - \mu$ cos $\alpha ) = 2 ( 1 - \mu$ cos α). But rule (2) of Definition 10 — the chosen duration realizes strict projected improvement — gives $| v | < | e _ { 2 }$ before the step|, and the $e _ { 2 } .$ -coordinate before any $u _ { 2 }$ step is at most A sin α with $A = ( 1 - \mu \cos \alpha ) /$ cos α the prior $u _ { 1 } \mathrm { - d u r a t i o n }$ (earlier $u _ { 2 }$ steps only shrink $| e _ { 2 } |$ , again by rule (2)). Hence $| v | < ( 1 - \mu \cos \alpha )$ tan α, and the audit demands $( 1 - \mu \cos \alpha )$ tan α $> 2 ( 1 - \mu \cos \alpha )$ , i.e. tan $\alpha > 2 -$ false at $\alpha = 2 0 ^ { \circ }$ □

Remark 2 (Mechanism). The two models control diferent objects. Memoryless consent gates directions: an uninterpretable direction is vetoed in every context, however small the step (E2). Audited consent gates cumulative trajectories: it forbids interpretable directions from accumulating into uninterpretable positions (E1), yet permits an uninterpretable direction to ride inside interpretable cumulative mass — in E2, $I _ { 1 } ( e _ { 2 } ) = 0$ while $I _ { 1 } ( ( 1 , 0 . 3 ) ) \approx 0 . 9 1 7$ . Neither model is simply stricter; they are two distinct readings of “I only join what I understand.”

Remark 3. The progress-realizing clause of Definition 10 is essential: without it, an overshooting $u _ { 2 }$ duration (large |v|) satisfies the audit and the separation fails. The separation holds for any witness angle with tan $\alpha \leq \cot \theta _ { 2 } ~ ( = 2 \mathrm { a t } \tau _ { 2 } = 0 . 8 )$

Remark 4 (Adaptive consent). Definitions 6 and 4 fix a single consent model for the whole plan. Nothing in the model forces this: a plan may instead carry a consent-policy sequence $\sigma = ( \sigma _ { 1 } , \dots , \sigma _ { T } ) \in \{ \mathrm { M } , \mathrm { A } \} ^ { T }$ , with step t ratified by

$$
\begin{array} { r } { \mathrm { C o n s e n t } _ { j } ( t ) = \left\{ \begin{array} { l l } { u _ { t } \in K _ { j } , } & { \sigma _ { t } = \mathrm { M } , } \\ { I _ { j } ( D _ { t } ) \geq \tau _ { j } , } & { \sigma _ { t } = \mathrm { A } , } \end{array} \right. \qquad j \in R ( u _ { t } ) . } \end{array}
$$

The two pure regimes are the constant sequences $\sigma \equiv \mathrm { M }$ and $\sigma \equiv \mathrm { A } ;$ ; nothing else in Steps 0–1 or 3 changes, and generation (Definition 10) is untouched $- \ \sigma$ governs ratification only. Write $S _ { \mathrm { M } } , S _ { \mathrm { A } } , S _ { \mathrm { a d a p t i v e } } \subseteq L _ { h }$ for the sets of plans ratifiable under, respectively, the pure memoryless model, the pure audited model, and some policy σ. Trivially $S _ { \mathrm { M } } \cup S _ { \mathrm { A } } \subseteq S _ { \mathrm { a d a p t i v e } } .$

Proposition 1 (Adaptive consent is strictly more permissive). There is an instance solvable under adaptive consent but under neither pure regime: $S _ { \mathrm { M } } \cup S _ { \mathrm { A } } \subsetneq S _ { \mathrm { a d a p t i v e } }$

Proof. Embed E1 and E2 in orthogonal coordinate blocks of $d = 4 \colon$ agents 1, 2 and directions $u _ { 1 } , u _ { 2 }$ act on coordinates 1, 2 exactly as in E1 $( R ( u _ { 1 } ) = \{ 1 \} , R ( u _ { 2 } ) = \{ 2 \} , \tau _ { 1 } = \tau _ { 2 } = 0 . 8 )$ , and agents 3, 4 with directions $u _ { 3 } = e _ { 3 } , u _ { 4 } = e _ { 4 }$ act on coordinates $^ { 3 , }$ 4 exactly as in E2 $( R ( u _ { 3 } ) = \{ 3 \}$ , $R ( u _ { 4 } ) = \{ 3 , 4 \} , \tau _ { 3 } = 0 . 8 , \tau _ { 4 } = 0 . 0 5 )$ . Requirements stack axis-wise $( x _ { 1 } = 1 , x _ { 2 } = 0 , x _ { 3 } = 1$ $x _ { 4 } = 0 . 3 )$ , each cylindrical over its owner’s axis, so the cover holds. Conception reduces blockwise — it is computed inside $P _ { i }$ , and every direction lies in a single block — so the closure contains the plan running the E2 block first and the E1 relay second,

$$
p ^ { \star } = \big ( ( u _ { 3 } , 1 ) , ( u _ { 4 } , 0 . 3 ) , ( u _ { 1 } , 1 / \cos \alpha ) , ( u _ { 2 } , \tan \alpha ) \big ) ,
$$

extended by agents 3, 4, 1, 2 in turn, with endpoint $( 1 , 0 , 1 , 0 . 3 ) \in O$

Ratify $p ^ { \star }$ under $\sigma = ( \mathrm { A } , \mathrm { A } , \mathrm { M } , \mathrm { M } )$ . At steps 1–2 the cumulative displacement still lies entirely in block 2, so the audits take exactly E2’s values: $I _ { 3 } ( D _ { 1 } ) = 1 , I _ { 3 } ( D _ { 2 } ) = 1 / 1 . 0 9 \approx 0 . 9 1 7 \geq 0 . 8$ and $I _ { 4 } ( D _ { 2 } ) = 0 . 0 9 / 1 . 0 9 \approx 0 . 0 8 3 \geq 0 . 0 5$ . Steps 3–4 are memoryless, hence state-independent: $I _ { 1 } ( u _ { 1 } ) = \cos ^ { 2 } 2 0 ^ { \circ } \approx 0 . 8 8 3 \geq 0 . 8$ and $I _ { 2 } ( u _ { 2 } ) = 1$ . All consents pass and STOP holds at the endpoint, so the instance is adaptive-solvable. Note that interpretability does not reduce blockwise — cumulative mass dark to the auditor dilutes $I _ { j } ( D )$ (Remark 2) — so the ordering is forced: with the E1 block first, the $u _ { 4 }$ step fails under both modes, $I _ { 3 } ( u _ { 4 } ) = 0 < 0 . 8$ memorylessly and, at its audit, $D = ( 1 , 0 , 1 , 0 . 3 )$ gives $I _ { 3 } ( D ) = 1 / 2 . 0 9 \approx 0 . 4 7 8 < 0 . 8$ and $I _ { 4 } ( D ) = 0 . 0 9 / 2 . 0 9 \approx$ $0 . 0 4 3 < 0 . 0 5$

Neither pure regime solves the instance. Memoryless: any plan reaching O must move coordinate 4, hence contains a u step, and $3 \in R ( u _ { 4 } )$ with $I _ { 3 } ( u _ { 4 } ) = 0 < \tau _ { 3 }$ . Audited: any plan reaching O has total $u _ { \mathrm { 1 } } \mathrm { - d u r a t i o n }$ $1 /$ cos α and a last $u _ { 2 }$ step, and the E1 argument of Theorem 4 applies to agent $2 \mathrm { { ^ { \circ } s } }$ audit there, with two adaptations: $u _ { 2 }$ is conceivable only at strictly positive e<sub>2</sub>-coordinate, so positive u<sub>1</sub>-duration precedes the last $u _ { 2 }$ step and $1 - \mu \cos \alpha > 0 ;$ and block-2 mass only enlarges $\lVert P _ { 2 } ^ { \perp } D \rVert$ , strengthening the audit’s demand $| v | \geq 2 ( 1 - \mu \cos \alpha )$ against the generation bound $| v | < ( 1 - \mu \cos \alpha )$ tan $\alpha .$ . The contradiction tan $\alpha > 2$ persists at $\alpha = 2 0 ^ { \circ }$ □

Proposition 2 (Cone shrinkage; forced non-plenary steps). $\cap _ { i \in R } K _ { j }$ is antitone in R. The two consent models confine diferent objects at plenary steps $( R ( u _ { t } ) = N ) \colon ( \mathrm { i } )$ Memoryless. Every executed step with coalition R has direction in $\cap _ { j \in R } K _ { j }$ ; hence an all-plenary memoryless plan has $\begin{array} { r } { D _ { T } \in \mathrm { c o n e } \big ( U \cap \bigcap _ { j \in N } K _ { j } \big ) } \end{array}$ , and

$\Delta \cap \mathrm { c o n e } \Big ( U \cap \bigcap _ { j \in N } K _ { j } \Big ) = \emptyset \implies$ every memoryless plan reaching O contains a non-plenary step.

(ii) Audited. Every plenary step constrains the cumulative displacement: $\textstyle D _ { t } \in \bigcap _ { j \in N } K _ { j }$ whenever $R ( u _ { t } ) = N ,$ hence an all-plenary audited plan has $D _ { t } \in \bigcap _ { j \in N } K _ { j }$ for all $t ,$ in particular $D _ { T }$ , and

$$
\Delta \cap \bigcap _ { j \in N } K _ { j } = \emptyset \implies e v e r y \ a u d i t e d \ p l a n \ r e a c h i n g \ O \ c o n t a i n s \ a \ n o n - p l e n a r y \ s t e p .
$$

Proof. Antitonicity is set intersection over larger index sets. (i): memoryless executability of a plenary step is $u _ { t } \in \cap _ { i \in N } K _ { j } ;$ additivity gives $\begin{array} { r } { D _ { T } \in \mathrm { c o n e } ( U \cap \bigcap _ { i } K _ { j } ) } \end{array}$ for all-plenary plans; take the contrapositive. (ii): audited executability of a plenary step is $I _ { j } ( D _ { t } ) \ge \tau _ { j }$ for every j, i.e. $\textstyle D _ { t } \in \bigcap _ { j \in N } K _ { j }$ ; in particular $\begin{array} { r } { D _ { T } \in \bigcap _ { j \in N } K _ { j } } \end{array}$ for all-plenary plans; take the contrapositive. The clauses instantiate Theorem 4’s mechanism: memoryless gates action directions, audited gates cumulative trajectory states. □

Two further facts about the model — that E1’s relay is fully consented by its executor yet purposeless to him (loop inversion), and that removing state-dependence from conception collapses memoryless CCS to a single conic-feasibility test — are recorded as Proposition 3 and Proposition 4 in Appendix A, since neither is needed for the results that follow.

Remark 5 (Public arithmetic; stall and sound stop). Tokens, bases, and requirements are public, so every agent computes every $I _ { j } { \mathrm { - v a l u e } }$ exactly; the only queried objects are consent values $( \tau _ { j }$ private), truthfully returned. The qualification structure is likewise fixed by the public data: $Q _ { k }$ is determined by the task and the representational structure, and is established by the planning procedure (Step 0 of Section 6) rather than by any individual’s meta-cognition — an agent need only perform the qualified checks for the requirements on which that agent is itself qualified, not compute the full qualification structure of the team. Both ends of a plan are mechanically gated: a step whose required implementer fails the consent condition does not execute (the plan stalls there — sovereign execution enforces itself), and termination runs through the qualified checks, sound and complete for their requirements, so a declared completion is a completion.

## 6 Exhaustive solvability scheme, correctness, and complexity

For the fixed horizon h, recall that $L _ { h }$ is the relay closure of Definition 10 truncated at depth h. The mathematical scheme below is exhaustive: Step 1 ranges over every instantiated plan in $L _ { h }$ , including every duration choice satisfying the projected-progress clause. This extensional formulation separates correctness from the separate question of whether $L _ { h }$ admits an eficient finite representation.

![](images/0390026d8b903715a259b6993c0bc6feffb601b10d0793de4239557e577bfe96.jpg)  
Figure 2: The exhaustive horizon-bounded solvability scheme of this section. Branch labels abbreviate the exit condition: empty is $Q _ { k } \ = \ \mathcal { O }$ for some $k ;$ covered is the cover holding; none/found refer to candidate plans reaching O; passes/none pass refer to a surviving plan clearing every qualified check in Step 3. Steps 0–1 are generative and never reject a true witness (Theorem 5); Steps 2–3 filter candidates against the model’s exact predicates. A restricted search that explores only $\widehat { L } _ { h } \subseteq L _ { h }$ in Step 1 remains sound on any plan it returns, but $\textrm { a \_ l }$ from such a search is not a certificate of unsolvability (Corollary 2).

Step 0 — coverage preflight (state-independent). Determine each $Q _ { k }$ from the public task and representational structure. If some $Q _ { k } = \emptyset$ , return ⊥ immediately: by Theorem 1, the task cannot be validly completed by this team, regardless of trajectory. Confirm you have an inspector before anyone lifts a hammer.

Step 1 — exhaustive relay closure (generation). Construct the complete horizon-bounded closure $L _ { h }$ by Definition 10: every agent re-expands from every generated node, and every improving duration allowed by the definition is retained. Candidate solutions are the generated plans whose endpoints lie in O. If there is no such candidate, return ⊥.

Step 2 — ratification. For each candidate plan, query each required agent’s consent value per step (memoryless: on $u _ { t } ;$ audited: on the realized $D _ { t } )$ ; delete refused candidates. Audited feasibility of a candidate is the constraint system $x _ { T } \in O$ and $\| P _ { j } ^ { \perp } D _ { t } \| \leq \tan \theta _ { j } \| P _ { j } D _ { t } \|$ for all t and all $j \in R ( u _ { t } )$ . Any symbolic or numerical manipulation of durations must preserve the conception inequalities along the realized trajectory, or the resulting plan is not a member of $L _ { h }$ . Thresholds are private, so synthesis is interactive (open problem O2).

Step 3 — terminal inspection. For each surviving plan, every requirement k is inspected by some $i \in Q _ { k }$ , who checks $P _ { i } x _ { T } \in P _ { i } O _ { k }$ in his own projection; diferent requirements may be covered by diferent agents. If all checks pass, return the plan. If none passes, return ⊥.

Theorem 5 (Correctness of the exhaustive horizon-bounded scheme). Assume that Step 0 determines the qualification sets $Q _ { k }$ exactly, Step 1 represents the complete closure $L _ { h }$ , and the consent and projected-membership tests in Steps 2–3 are evaluated exactly. Then the four-step scheme returns a plan if and only if CCS is yes for the given horizon h. Every returned plan is a valid CCS witness.

Proof. Soundness. Suppose the scheme returns a plan $p .$ Step 1 guarantees $p \in L _ { h }$ . Step 2 retains p only if every required implementer consents at every step, so every step executes under Definition 6. Step 3 returns p only if, for every requirement $O _ { k }$ , some qualified verifier $i \in Q _ { k }$ finds $P _ { i } x _ { T } \in P _ { i } O _ { k }$ . By Definition $9 , \mathsf { S T O P } ( x _ { T } )$ holds. Hence $p$ satisfies Definition 11.

Completeness. Suppose CCS is yes. Then there exists a witness $p ^ { \star } \in L _ { h }$ whose every step executes and for which $\mathtt { S T O P } ( x _ { T } ^ { \star } )$ holds. Because STOP requires a qualified verifier for every requirement, Step 0 does not reject. Completeness of Step 1 places $p ^ { \star }$ among the candidates (and $\mathtt { S T O P } ( x _ { T } ^ { \star } )$ implies $x _ { T } ^ { \star } \in O )$ . Since every step of $p ^ { \star }$ executes, Step 2 does not delete it. Since $\mathtt { S T O P } ( x _ { T } ^ { \star } )$ holds, every requirement has a passing qualified check in Step 3. Thus the scheme returns a plan. □

Corollary 2 (Soundness of restricted search). Let a practical implementation explore any subset $\widehat { L } _ { h } \subseteq L _ { h }$ but apply ratification and terminal inspection exactly. Every plan it returns is a valid CCS witness. However, returning ⊥ is a certificate of CCS = no only when the explored set is complete, $\widehat { L } _ { h } = L _ { h }$

Proof. The soundness argument of Theorem 5 uses only membership in $L _ { h }$ , exact consent, and exact terminal inspection, so it applies to every returned plan from $\widehat { L } _ { h }$ . Completeness may fail when a valid witness lies in $L _ { h } \backslash \widehat { L } _ { h }$ h. □

Computational observations and open problems. The general model places no representation constraints on the requirements $O _ { k } \subseteq \mathbb { R } ^ { d }$ , and for arbitrary subsets membership, projected distance, and the qualification test need not be computable. For algorithmic and complexity statements we therefore restrict to efectively represented requirements — those for which these operations are computable — with afine or polyhedral requirements as the principal example. This makes the input finite, but it does not by itself make the continuous duration branching of $L _ { h }$ finite; exact symbolic representation of the state-dependent relay closure remains part of the computational problem. In the state-independent comparison variant, memoryless CCS collapses to a single conic-feasibility test (Proposition 4), LP-shaped [6]. In general, prefixdependent enabling mirrors a central source of combinatorial search in classical planning [8], while audited CCS additionally imposes nonconvex cumulative-trajectory constraints. O1: the exact representation and complexity of state-dependent relay closure — including which finite length or symbolic bounds survive. O2: the query complexity of interactive plan synthesis under private thresholds.

## 7 Computational illustration

A reference implementation accompanies the paper for afine requirements. To keep the search finite, it explores a restricted subset $\widehat { L } _ { h } \subset L _ { h }$ by choosing the projected-optimal improving λ at each extension. Every generated token still satisfies Definition 10, and ratification and terminal inspection use the model’s exact predicates; therefore every returned plan is sound by

![](images/02128c58f8e591f8b3f8a1d6f07832af99a1b3898824316376d54b801d293822.jpg)  
Figure 3: A two-step counterfactual relay in E1. Agent 1 changes the state so that agent 2 can conceive a continuation that was unavailable at the start. In agent $2 \mathrm { { ^ { \circ } s } }$ projection the whole plan is the closed loop $0  \tan \alpha  0 .$ , even though the group makes goal-relevant progress along a dimension dark to agent 2. The same instance is memoryless-solvable but audited-unsolvable (Theorem 2, Proposition 3, Theorem 4).

Corollary 2. The implementation is not claimed complete for general CCP: a valid solution may require a diferent improving duration. Every numerical claim of Examples E1 and E2 is an executable assertion in the test suite. Figure 3 draws the E1 relay geometry; Figure 4 sweeps structured random instances across span coverage and consent thresholds; Figure 5 varies the shared representational core. All three are structured illustrations of the formal results, not empirical validation.

## 8 Discussion

What the results formalize. The first phenomenon is sequential mutual enabling: a team can fail under one-shot pooling yet succeed when members repeatedly reconsider the evolving state, because one person’s move can make another person’s next move newly conceivable (Theorem 2 and Corollary 1). The second is competence without global purpose: an executor may fully understand the step he performs while the reason the step matters lies outside his projection (Proposition 3). The third is forced sub-teaming: when plenary participation is confined by the shared representational core, boundary progress may require delegation to proper subcoalitions (Proposition 2). The negative limit is equally sharp: if a task requirement varies along a dimension dark to the whole team, the team cannot validly declare completion even if it accidentally reaches the right physical state (Theorem 1). These phenomena are documented, separately, in distributed cognition [24], epistemic dependence [22], and high-reliability organizing [46, 47]; CCP supplies explicit geometric conditions under which they arise.

![](images/21d1b6e25cb9f3d76c4d4ba287dd1fb466fb3fdc4873301f7912bd92745d680d.jpg)

![](images/797205631a5ea6c060334af13798c98aec01385f2b31ce34e0234e0cc0530e57.jpg)  
Figure 4: Success rates across structured random instances. Left: greater representational coverage increases attainment and validated completion. Right: audited consent helps at lenient thresholds but hurts at strict thresholds, illustrating its incomparability with memoryless consent.

Scope. Three deliberate exclusions define the regime. No strategy: agents are truthful and share O; this is the cognitive skeleton of cooperation with the political flesh removed. No learning: bases are fixed, so the model covers collaboration on timescales shorter than teaching; changes to the representational basis lie outside the present model. And sovereign execution picks out professionalized collaboration — “I do not perform steps I do not understand” is the working ethics of medicine, aviation, and licensed trades — rather than authority-gated organizations. Within the regime, the claims are face-valid, not validated: whether real teams fail by these mechanisms is an empirical bet, and the model’s prediction shapes (sub-teaming density at basis boundaries; verification stafing predicting valid completion) say where to test it.

Future work. The hardness of general CCS (O1, O2); and team synthesis — given a task $\{ O _ { k } \}$ and a pool of candidate agents with subspaces and thresholds, construct the minimum team for which CCS is yes — the representational successor of team formation with required skills [25, 28].

## 9 Conclusion

Collective Counterfactual Planning models a team as a set of projections of one task space, and derives relay, consent, sub-teaming, and sign-of from representational geometry interacting with the task’s implementation structure. Its positive result explains how a group can reach a goal no member can plan alone; its negative result identifies requirements that no amount of efort, honesty, or luck can let the team validly finish. The exhaustive horizon-bounded solvability scheme is correct for the formal CCS problem, while practical restricted searches remain sound but may be incomplete. You do not need a team that can build every possible house. You need people whose combined representational and practical competencies sufice to plan, execute, and verify this one.

![](images/20448fe18a30c45472f500479eb46fc114f12de37c49591558c5c8298d43c984.jpg)  
Figure 5: Shared representation changes who can act together. As the common representational core grows, successful plans use larger coalitions and more whole-team steps; progress outside that core is carried by smaller subteams. All instances remain solvable.

Reproducibility. Code and reproduction materials are available at https://github.com/ DarkEyes/Coll-Counterfactual-Plan. All numerical claims of Examples E1 and E2, and of Proposition 1, are executable assertions in the accompanying test suite (test e1.py); the figures are produced by experiments.py from the library ccp.py, seeded for reproducibility. The code implements the restricted duration policy described in Section 7; it is used to check witnesses and illustrate the theory, not as a completeness certificate for general CCS. The audited\memoryless direction of Theorem 4 was discovered by this code: the first threshold sweep contradicted a draft containment claim, and E2 is the distilled witness. Proposition 1’s witness was likewise code-checked: an initial draft ordering failed to ratify under every consent policy, since audited interpretability does not decompose across orthogonal blocks (Remark 2); the corrected ordering in the proof is the one the test suite confirms.

## Use of generative AI and AI-assisted technologies

During the preparation of this work the author used Claude (Anthropic)/chatGPT (OpenAI) to edit and polish the draft. After using these tools, the author reviewed and edited the content as needed and takes full responsibility for the content of the publication.

## A Additional results

These two observations sharpen the reading of E1 and of the closure mechanism, respectively, but neither is needed for the paper’s main line of argument (Sections 5–6), so they are collected here rather than in the main text.

Proposition 3 (Loop inversion). In the E1 relay plan, executor 2 fully interprets his step $( I _ { 2 } ( u _ { 2 } ) = 1 )$ , yet the plan’s point is dark to him: his projected view of the whole trajectory is the closed loop $0  \tan \alpha  0 , P _ { 2 } ( x _ { T } - x _ { 0 } ) = 0$ , and the goal-relevant progress lies along $e _ { 1 }$ with $P _ { 2 } e _ { 1 } = 0$ . Interpretability gates motion, never purpose.

Proof. By the E1 computations.

Proposition 4 (State-independent comparison variant). Under the concrete progress rule (1), availability is generically state-dependent; for comparison, consider the variant in which availability is exogenously state-independent, $C _ { i } ( x , O ) \equiv C _ { i }$ (replacing Definition $\it 4 )$ . In this variant, under memoryless consent, if $\Delta \cap$ cone $( U ^ { * } ) \neq \emptyset$ and the verification cover holds, where $U ^ { * } = \{ u : \forall j \in R ( u ) , u \in K _ { j }$ and $\exists i \in R ( u ) , u \in C _ { i } \}$ , then a witnessing plan with $T \leq d$ exists; hence for every horizon $h \geq d ,$ , CCS holds if the cover holds and $\Delta \cap \mathrm { c o n e } ( U ^ { * } ) \neq \emptyset$

Proof. Availability and consent are then prefix-independent, so appendability and executability of a token depend only on its direction, and reachable endpoints are exactly $x _ { 0 } + \mathrm { c o n e } ( U ^ { * } )$ by additivity. By conic Carath´eodory [38], any point of cone(U<sup>∗</sup>) is a nonnegative combination of at most d linearly independent members; merging equal-direction tokens gives $T \leq d ,$ which fits within any horizon $h \geq d .$ The variant isolates exactly what prefix-dependent enabling contributes: with it removed, order-irrelevance returns, reachable endpoints form a cone, and short plans sufice. □

## References

[1] Chainarong Amornbunchornvej. Interpretation as linear transformation: A cognitivegeometric model of concepts and meaning. Minds and Machines, 36(3):36, 2026.

[2] Chainarong Amornbunchornvej and Tanya Berger-Wolf. Framework for inferring following strategies from time series of movement data. ACM Transactions on Knowledge Discovery from Data (TKDD), 14(3):1–22, 2020.

[3] Chainarong Amornbunchornvej, Ivan Brugere, Ariana Strandburg-Peshkin, Damien R. Farine, Margaret C. Crofoot, and Tanya Y. Berger-Wolf. Coordination event detection and initiator identification in time series data. ACM Trans. Knowl. Discov. Data, 12(5), June 2018. ISSN 1556-4681. doi: 10.1145/3201406. URL https://doi.org/10.1145/3201406.

[4] Daniel S. Bernstein, Robert Givan, Neil Immerman, and Shlomo Zilberstein. The complexity of decentralized control of Markov decision processes. Mathematics of Operations Research, 27(4):819–840, 2002. doi: 10.1287/moor.27.4.819.297.

[5] Thomas Bolander and Mikkel Birkegaard Andersen. Epistemic planning for single- and multi-agent systems. Journal of Applied Non-Classical Logics, 21(1):9–34, 2011. doi: 10. 3166/jancl.21.9-34.

[6] Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, 2004.

[7] Ronen I. Brafman and Carmel Domshlak. From one to many: Planning for loosely coupled multi-agent systems. In Proceedings of the 18th International Conference on Automated Planning and Scheduling (ICAPS), pages 28–35, 2008.

[8] Tom Bylander. The computational complexity of propositional STRIPS planning. Artificial Intelligence, 69(1–2):165–204, 1994. doi: 10.1016/0004-3702(94)90081-7.

[9] Bernard Chazelle. The total s-energy of a multiagent system. SIAM Journal on Control and Optimization, 49(4):1680–1706, 2011. doi: 10.1137/100791671.

[10] Herbert H. Clark. Using Language. Cambridge University Press, 1996.

[11] Russell Cooper. Coordination Games. Cambridge University Press, 1999.

[12] Allan Dafoe, Edward Hughes, Yoram Bachrach, Tantum Collins, Kevin R McKee, Joel Z Leibo, Kate Larson, and Thore Graepel. Open problems in cooperative ai. arXiv preprint arXiv:2012.08630, 2020.

[13] Richard L Daft and Karl E Weick. Toward a model of organizations as interpretation systems. Academy of Management Review, 9(2):284–295, 1984.

[14] Thorsten Engesser, Thomas Bolander, Robert Mattm¨uller, and Bernhard Nebel. Cooperative epistemic multi-agent planning for implicit coordination. Electronic Proceedings in Theoretical Computer Science, 243:75–90, 2017. doi: 10.4204/EPTCS.243.6.

[15] Ronald Fagin, Joseph Y. Halpern, Yoram Moses, and Moshe Y. Vardi. Reasoning About Knowledge. MIT Press, 1995. doi: 10.7551/mitpress/5803.001.0001.

[16] Jay R. Galbraith. Organization design: An information processing view. Interfaces, 4(3): 28–36, 1974.

[17] Peter Gardenfors. Conceptual spaces: The geometry of thought. MIT press, 2004.

[18] Peter G¨ardenfors and Massimo Warglien. Using conceptual spaces to model actions and events. Journal of Semantics, 29(4):487–519, 2012. doi: 10.1093/jos/fs007.

[19] Malik Ghallab, Dana Nau, and Paolo Traverso. Automated Planning and Acting. Cambridge University Press, 2016. doi: 10.1017/CBO9781139583923.

[20] Daniel T. Gilbert and Timothy D. Wilson. Prospection: Experiencing the future. Science, 317(5843):1351–1354, 2007. doi: 10.1126/science.1144161.

[21] Jakob Hansen and Robert Ghrist. Opinion dynamics on discourse sheaves. SIAM Journal on Applied Mathematics, 81(5):2033–2060, 2021. doi: 10.1137/20M1341088.

[22] John Hardwig. Epistemic dependence. The Journal of Philosophy, 82(7):335–349, 1985. doi: 10.2307/2026523.

[23] John Hardwig. The role of trust in knowledge. The Journal of Philosophy, 88(12):693–708, 1991. doi: 10.2307/2027007.

[24] Edwin Hutchins. Cognition in the Wild. MIT Press, 1995.

[25] Julio Ju´arez, Cipriano Santos, and Carlos A. Brizuela. A comprehensive review and a taxonomy proposal of team formation problems. ACM Computing Surveys, 54(7):153:1– 153:33, 2021. doi: 10.1145/3465399.

[26] Natalia L. Komarova and Partha Niyogi. Optimizing the mutual intelligibility of linguistic agents in a shared world. Artificial Intelligence, 154(1–2):1–42, 2004.

[27] Nikolaus Kriegeskorte and Rogier A Kievit. Representational geometry: integrating cognition, computation, and the brain. Trends in cognitive sciences, 17(8):401–412, 2013.

[28] Theodoros Lappas, Kun Liu, and Evimaria Terzi. Finding a team of experts in social networks. In Proceedings ofthe 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 467–476, 2009. doi: 10.1145/1557019.1557074.

[29] David Lewis. Counterfactuals. Blackwell, 1973.

[30] Kyle Lewis. Measuring transactive memory systems in the field: Scale development and validation. Journal of Applied Psychology, 88(4):587–604, 2003. doi: 10.1037/0021-9010. 88.4.587.

[31] Christian List and Philip Pettit. Group agency: The possibility, design, and status of corporate agents. Oxford University Press, 2011.

[32] Frans A. Oliehoek and Christopher Amato. A Concise Introduction to Decentralized POMDPs. Springer, 2016. doi: 10.1007/978-3-319-28929-8.

[33] Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 39643–39666. PMLR, 21–27 Jul 2024. URL https: //proceedings.mlr.press/v235/park24c.html.

[34] Judea Pearl. Causality. Cambridge University Press, 2009.

[35] Charles Perrow. Normal Accidents: Living with High-Risk Technologies. Basic Books, New York, 1984.

[36] Steven T Piantadosi, Dyana CY Muller, Joshua S Rule, Karthikeya Kaushik, Mark Gorenstein, Elena R Leib, and Emily Sanford. Why concepts are (probably) vectors. Trends in Cognitive Sciences, 28(9):844–856, 2024.

[37] Michael Power. The audit society: Rituals of verification. OUP Oxford, 1997.

[38] R. Tyrrell Rockafellar. Convex Analysis. Princeton University Press, 1970.

[39] Ariel Rubinstein. The electronic mail game: Strategic behavior under” almost common knowledge”. The American Economic Review, pages 385–391, 1989.

[40] Thomas C Schelling. The Strategy of Conflict: with a new Preface by the Author. Harvard university press, 1980.

[41] Herbert A. Simon. Models of Man: Social and Rational. Wiley, New York, 1957.

[42] Damian Rados law Sowinski, Jonathan Carroll-Nellenback, Jeremy DeSilva, Adam Frank, Gourab Ghoshal, and Marcelo Gleiser. The consensus problem in polities of agents with dissimilar cognitive architectures. Entropy, 24(10):1378, 2022. doi: 10.3390/e24101378.

[43] Michael Tomasello, Malinda Carpenter, Josep Call, Tanya Behne, and Henrike Moll. Understanding and sharing intentions: The origins of cultural cognition. Behavioral and Brain Sciences, 28(5):675–691, 2005.

[44] Alejandro Torre˜no, Eva Onaindia, Anton´ın Komenda, and Michal Stolba. Cooperative<sup>ˇ</sup> multi-agent planning: A survey. ACM Computing Surveys, 50(6):84:1–84:32, 2017. doi: 10.1145/3128584.

[45] John B. Van Huyck, Raymond C. Battalio, and Richard O. Beil. Tacit coordination games, strategic uncertainty, and coordination failure. The American Economic Review, 80(1): 234–248, 1990.

[46] Diane Vaughan. The Challenger launch decision: Risky technology, culture, and deviance at NASA. University of Chicago Press, 1996.

[47] Karl E. Weick, Kathleen M. Sutclife, and David Obstfeld. Organizing for high reliability: Processes of collective mindfulness. In Robert I. Sutton and Barry M. Staw, editors, Research in Organizational Behavior, volume 21, pages 81–123. JAI Press, 1999.

[48] Michael Wooldridge. An introduction to multiagent systems. John wiley & sons, 2009.

[49] Michael Wooldridge and Nicholas R. Jennings. The cooperative problem solving process. Journal of Logic and Computation, 9(4):563–592, 1999. doi: 10.1093/logcom/9.4.563.

[50] Yu Zhang, Sarath Sreedharan, and Subbarao Kambhampati. A formal analysis of required cooperation in multi-agent planning. In Proceedings of the 26th International Conference on Automated Planning and Scheduling (ICAPS), volume 26, pages 335–343, 2016. doi: 10.1609/icaps.v26i1.13770.
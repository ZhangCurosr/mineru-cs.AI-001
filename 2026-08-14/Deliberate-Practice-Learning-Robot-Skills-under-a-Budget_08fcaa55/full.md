# Deliberate Practice: Learning Robot Skills under a Budget

Shivam Vats<sup>1</sup> Sudarshan Harithas<sup>1</sup> Mete Tuluhan Akbulut<sup>1</sup>

Arvind Raghunathan<sup>2</sup> George Konidaris<sup>1</sup>

<sup>1</sup>Brown University <sup>2</sup>Mitsubishi Electric Research Laboratories Correspondence to shivam vats@brown.edu

Abstract: We consider the problem of autonomously learning robot skills under a limited practice budget for sequential tasks. We propose an active skill learning algorithm, Deliberate Practice (DP), that computes a provably budget-optimal allocation—practicing skills that maximize expected cumulative reward while being learnable within the budget. DP estimates both the time needed to master skills and the cumulative reward of the task plans that the skills unlock. Computing a budget-optimal allocation is challenging as it requires reasoning about combinatorially many skill plans over a large practice budget. Our key contribution is a bilinear program that can compute this exactly using off-the-shelf solvers. Through simulated and real-world experiments on long-horizon manipulation tasks, we show that our approach allows robots to optimally use limited practice time to acquire useful policies and improve long-horizon planning.

Keywords: Task and Motion Planning, Active Learning, Robot Skills

## 1 Introduction

Recent progress in large-scale robot learning has produced general-purpose robot policies that can perform diverse skills and transfer to novel environments [1, 2, 3]. When combined with classical planning methods, such as Task and Motion Planning (TAMP) [4, 5, 6] and search [7, 8], these policies can allow robots to perform long-horizon tasks in homes and factories, charting a path for more autonomous robots. However, pretraining alone does not cover all the possible scenarios a robot may face in the real-world, often leading to unreliable task execution. Therefore, a promising solution is to allow robots to learn from experience through autonomous practice [9, 10, 11]. Unfortunately, current RL algorithms remain highly sample inefficient [12, 13], limiting their applicability to deployment-time learning. Since robots typically have only limited downtime during deployment, they require methods explicitly designed for budgeted practice.

Our key insight is that downtime is often known in advance, and robots should leverage this information to adapt their learning process to the resulting practice budget. Intuitively, a large practice budget should encourage robots to practice more rewarding skills even if they are hard, while a small budget should lead to conservative learning behaviour. We formalize the problem of autonomou skill practice for sequential tasks as budgeted skill learning, and propose an active learning algorithm, Deliberate Practice (DP), that optimally uses the budget to maximize task reward. We consider the standard TAMP setting in which a robot is given high-level skill specifications, including preconditions, termination conditions, and effects. For example, in the breakfast domain in Section 6, the StartToaster skill requires moving above the toaster lever to press it and has the effect of turning on the toaster. The robot must learn parameterized control policies that ground skills into low-level actions through practice in the environment. At decision time, a task planner computes a plan consisting of skills, e.g., (Pick(bread), Place(bread, toaster), Start(toaster)) to toast bread. Figure 1 illustrates how our approach allows the robot to decide which skills to practice to make breakfast. The robot can either practice one skill to toast bread or use additional practice time to learn two skills required to microwave oatmeal and achieve a higher reward. As shown in Section 6, with a budget of 30 episodes, DP correctly estimates that only one skill can be learned reliably and hence practices to toast bread. Under a budget of 60 episodes, it instead practices to microwave oatmeal to take advantage of the additional practice time.

![](images/ccf84c95839a12998ea823cf1d3d6d37884de2872a26861feed297118a81be62.jpg)  
Figure 1: Deliberate Practice allows robots to intelligently use a limited practice budget to actively learn skills to maximize long-horizon task reward. (top) Under a small budget, the robot practices an easy task plan that toasts bread and provides reward 1. (bottom) Under a larger budget, it practices a more difficult but higher-reward task plan that microwaves oatmeal and provides reward 2.

Deliberate Practice first estimates the difficulty of learning each skill as a function of the practice budget. It then computes a budget allocation that provably maximizes expected planning performance, and practices its skills accordingly. We show in Section 4 that computing an optimal allocation requires solving a challenging bilevel optimization problem. Existing active learning algorithms approximate this problem greedily, leading to myopic and suboptimal learning (Section 4.1). By contrast, we derive an exact single-level reformulation of the bilevel problem using the linear programming (LP) formulation of Markov decision processes. This allows us to compute the optimal budget allocation using off-the-shelf solvers. Through experiments in simulated and real-world long-horizon manipulation tasks, we demonstrate that Deliberate Practice is uniquely budget-aware and enables robots to learn effectively under deployment constraints.

## 2 Related Work

Active Skill Learning. Deliberate practice is a widely studied mechanism underlying expert human performance [14], characterized by structured, goal-directed training that allocates effort toward the most limiting components of a skill. In robotics, active learning has similarly been central for skill acquisition due to the high cost of data collection. Prior work includes methods for actively learning parameterized robot skills from practice [15], human demonstrations [16], and across multiple tasks [17, 18, 19]. However, these approaches focus on efficiently acquiring individual skills without considering the sequential nature of long-horizon problems. By contrast, our approach studies active skill learning embedded in a complex sequential decision process under explicit budget constraints.

Task and Motion Planning (TAMP). TAMP [4, 20, 5] addresses robotic planning problems that require joint reasoning over discrete task structure and continuous motion feasibility. Most TAMP systems adopt a hierarchical approach, where a high-level planner [21, 22] solves an abstract version of the problem to compute a task plan, which is then grounded into low-level robot actions using motion planning and control [23]. Recent TAMP approaches leverage machine learning to improve scalability and performance, for example, by learning heuristics to guide symbolic search [24], or by learning symbolic operators and task representations [25, 26]. Our work is closely related to approaches that plan to practice skills for autonomous improvement in long-horizon tasks [27, 28]. However, these methods rely on greedy practice strategies and often allocate practice time suboptimally. By contrast, our method computes a budget-optimal allocation of practice time.

## 3 Background

We model the environment as a Markov decision process (MDP) with continuous state and action spaces, defined by the tuple $\mathcal { M } : = ( S , \mathcal { A } , P , r , \gamma )$ , where $s$ is the low-level state space, $\mathcal { A }$ is the lowlevel robot action space, $P ( s ^ { \prime } | s , a )$ is the transition function, $r ( s , a )$ is the reward function, γ is the discount factor. We assume an abstract MDP $\mathcal { M } _ { h } : = ( S _ { h } , \mathcal { A } _ { h } , P _ { h } , r _ { h } , \gamma )$ , where a state abstraction function typically $\alpha : { \mathcal { S } }  { \mathcal { S } } _ { h }$ projects raw sensory observations to abstract states.

Abstract Dynamics. We assume that the robot is provided with a library of skills $u ,$ which are defined using an extension of the options framework, consisting of the following components: a) $P r e \cdot$ condition (PRE) defines the states in which a skill can be initiated, b) Effect (EFF) describes the outcome of executing a skill successfully, c) Control policy (π) computes the low-level actions to be executed by the robot, d) Termination condition (β) specifies when a skill should terminate, e) Competence $( p )$ is the probability that the skill will successfully achieve the desired effect. The abstract action space $\boldsymbol { \mathcal { A } } _ { h }$ is defined by skill library U that the robot can use to transition between abstract states. Skill effect and competence together define the high-level transition function $P _ { h } \colon P _ { h } ( s ^ { \prime } | s , a ) = p ,$ if $s ^ { \prime }$ corresponds to the effect of skill $u \in \mathcal { U }$ when executed in state s, otherwise transitioning to an absorbing failure state. We use a task planner to solve this abstract MDP $\mathcal { M } _ { h }$ to compute a sequence of skills with the correct parameters to reach the goal, $\mathrm { e . g . }$ , (Open(top-drawer), Pick(toy), Place(toy, top-drawer)). The skills then ground this abstract task plan by computing feasible robot trajectories.

Structured Robot Skill. We use a combination of motion planning and parameterized policies to handle contact-rich interactions with the environment. We structure every contact-rich policy as an object-centric Composable Interaction Primitive (CIP) [29], which consists of three phases: a) $P r e \mathrm { - }$ interaction: motion plan to a pose near the object, b) Interaction: execute a parameterized policy, and c) Post-interaction: motion plan away from the object. The interaction policy is learned by the robot by practicing in the environment.

Task and Skill Reward. The robot is tasked with maximizing a task reward function R with one or more absorbing goal states, defined by a goal function $g : { \mathcal { S } }  \{ 0 , 1 \}$ . We also define skill-specific reward functions to learn the policy parameters through practice.

## 4 Budgeted Skill Learning for Planning

Given the high-level descriptions of a library of skills $u ,$ consisting of their preconditions, termination conditions, and effects, the robot must learn the policy parameters of its skills to improve overall planning performance under a limited practice budget B, e.g., maximum number of trials. We adopt the standard episodic reinforcement learning setting without access to expert demonstrations. Our goal is to compute a budget allocation $b _ { u }$ across all skills U to maximize the expected planning performance. Formally, this can be expressed as the following optimization problem:

$$
\begin{array} { r l } & { \displaystyle \operatorname* { m a x } _ { b } \sum _ { s \in \cal S } e _ { s } v _ { s } } \\ & { \displaystyle \mathrm { s u b j e c t ~ t o } } \\ & { \displaystyle \sum _ { u \in \cal U } b _ { u } \le B , } \\ & { \displaystyle v _ { s } \in \mathrm { S o l v e M D P } ( \mathcal { M } _ { h } , b ) , } \end{array}\tag{1}
$$

where $e _ { s }$ is the initial state distribution, and $v _ { s }$ is the state value function. This is a bilevel optimization problem where the outer problem computes a budget allocation b for practicing policies, and the inner problem solves the resulting task MDP $\mathcal { M } _ { h }$ to compute the expected planning performance after practicing with budget allocation b.

## 4.1 Greedy Active Learning Converges to Local Minima

Prior active skill learning methods [27, 28] for sequential tasks greedily practice the skill that will most improve the expected task performance $J _ { \mathrm { t a s k } }$ after one round of training. These methods can be viewed as performing one-step reasoning to solve equation 1. However, this is inherently myopic: it can overlook higher-performing task plans that require deliberate practice over multiple episodes to be mastered. We provide an illustrative example in Figure 2. This MDP consists of 5 states: A, B, C, D and a terminal failure state. The robot always starts at A, while B and D are terminal goal states. The MDP can be solved either by using $\pi _ { 1 }$ to reach B or by using $\pi _ { 2 }$ and π<sub>3</sub> to reach D. All skills have identical competence $p = 0 . 1$ ; they transition to the failure state with probability 0.9. The robot must decide which skills to practice among $\pi _ { 1 } , \pi _ { 2 } , \pi _ { 3 }$ under a budget of 20 episodes given each skill has the same rate of

![](images/a301d17b71d6217f1937c858418d252a4a13c8fbabd8411634268a911e488974.jpg)  
Figure 2: An example illustrating how greedy active learning can make suboptimal decisions.

improvement $\Delta p _ { i } = 0 . 1$ per practice episode. The optimal policy for the MDP before practice is to use $\pi _ { 1 }$ to reach B. Hence, $J _ { \mathrm { t a s k } } = p _ { 1 } \times R _ { B } = 0 . 1$ . Let $\pi _ { i } ^ { \prime }$ be the skill after a round of practice. The greedy task improvement measure of each skill is computed below:

$$
{ \mathrm { P r a c t i c e } } \pi _ { 1 } \implies J _ { \mathrm { t a s k } } \big ( \{ \pi _ { 1 } ^ { \prime } , \pi _ { 2 } , \pi _ { 3 } \} \big ) = ( p _ { 1 } + \Delta p _ { 1 } ) \times R _ { B } = 0 . 2 \times 1 = 0 . 2
$$

$$
\mathrm { P r a c t i c e } \ \pi _ { 2 } \implies J _ { \mathrm { t a s k } } \big ( \{ \pi _ { 1 } , \pi _ { 2 } ^ { \prime } , \pi _ { 3 } \} \big ) = ( p _ { 2 } + \Delta p _ { 2 } ) \times p _ { 3 } \times R _ { D } = 0 . 2 \times 0 . 1 \times 4 = 0 . 0 8
$$

$$
\mathrm { P r a c t i c e } \ \pi _ { 3 } \implies J _ { \mathrm { t a s k } } \big ( \{ \pi _ { 1 } , \pi _ { 2 } , \pi _ { 3 } ^ { \prime } \} \big ) = p _ { 2 } \times ( p _ { 3 } + \Delta p _ { 3 } ) \times R _ { D } = 0 . 1 \times 0 . 2 \times 4 = 0 . 0 8 .
$$

Therefore, greedy methods practice only $\pi _ { 1 }$ since it always maximizes one-step $J _ { \mathrm { t a s k } }$ improvement resulting in the final reward 1. However, this misses the optimal solution $\{ \pi _ { 2 } , \pi _ { 3 } \}$ which will provide reward 4 under 20 rounds of practice, because identifying the optimal budget allocation requires reasoning about deliberate practice across $\pi _ { 2 }$ and $\pi _ { 3 }$ over multiple rounds.

## 5 Deliberate Practice

Our algorithm allocates the practice budget provably optimally by jointly reasoning about all robot skills across the entire practice budget. As shown in Figure 3, our approach comprises three steps:

1. Competence Prediction. Competence improvement is modeled as a function of the practice budget. This model is initialized with a domain-specific prior and updated online based on the robot’s actual improvement.

2. Budget Allocation. A budget-optimal task plan $\Pi ^ { * }$ is computed along with the corresponding budget allocation $b ^ { * }$ needed to master it.

3. Skill Practice. The robot sequentially masters the skills that are reachable from the start and then uses them to plan to reach and practice downstream skills.

## 5.1 Competence Prediction

The competence prediction function $f _ { \mathrm { i m p r o v } } ( p _ { u } , b _ { u } )$ predicts the expected skill competence after practicing a skill with a competence of $p _ { u }$ for $b _ { u }$ rounds. While our approach supports linear, piece-wise linear and exponential models of competence prediction, empirically a piece-wise linear

![](images/5d8e5efcc39a482f26a885adcd00a61bf57e68db6a267cae655e44fe252a1632.jpg)  
Figure 3: Overview: Our approach predicts skill competence as a function of practice budget, computes a budget allocation for practice, and practices skills by interacting with the environment.

model provides a good balance between performance and efficiency.

$$
f _ { \mathrm { i m p r o v } } ( u , b ) = \operatorname* { m i n } ( 1 , p _ { u } + \Delta _ { u } b ) ,\tag{2}
$$

where $\Delta _ { u }$ is the rate of improvement of a skill u estimated from practice. $\Delta _ { u }$ is estimated online as a running average of competence improvement observed after practicing skill u over time: $\Delta _ { u } ^ { t } =$ $\epsilon \Delta _ { u } ^ { t - 1 } + ( \bar { 1 } - \epsilon ) \bar { ( p _ { u } ^ { t } - p _ { u } ^ { t - 1 } ) }$ , where $\epsilon \in [ 0 , 1 ]$ is a smoothing factor. To capture highly non-linear dynamics (such as deep reinforcement learning), our approach also supports more complex models, such as the saturating exponential: $f _ { \mathrm { i m p r o v } } ( u , b ) = p _ { u } + ( 1 - p _ { u } ) ( 1 - e ^ { - \alpha b } )$ , where α is the learning rate that is estimated online.

## 5.2 Budget Allocation

Next, we use the competence prediction model to compute the best task plan feasible within the practice budget. Long-horizon manipulation tasks require deliberately practicing multiple skills over many episodes to be reliably solved. As the bilevel optimization problem (Equation 1) shows, this requires jointly reasoning about all skills and the entire practice budget. However, this optimization is notoriously difficult due to its nested structure: the feasible set of the outer problem is defined implicitly by the solution of the inner optimization. This coupling leads to highly nonconvex and often non-smooth objectives, even when both levels are individually convex. We derive an exact single-level reformulation using duality, yielding a structured single-level bilinear program that enables efficient optimization using standard nonlinear programming solvers, eliminating the need for nested optimization, and greedy approximations. To our knowledge, this is the first exact formulation ofrobot learning on a budget as an optimization problem.

We first state the bilinear program and then outline the derivation. The lines in black in equation 3 correspond to the standard dual linear program of an MDP [30], and our proposed changes are highlighted in blue. $\mu _ { s } ^ { a }$ are dual variables that correspond to state-action occupancy, $e _ { s }$ is the initial state distribution, $P _ { s ^ { \prime } s } ^ { a }$ is the current transition function, and $\bar { P } _ { s ^ { \prime } s } ^ { a } ( b )$ is the predicted transition function after practicing with budget allocation b. Surprisingly, we show that practice budget constraints can be directly incorporated into the standard dual linear program.

$$
\begin{array} { r l } & { \displaystyle \operatorname* { m a x } _ { b , b , \psi } \displaystyle \sum _ { v _ { s } \in S } \sum _ { v \alpha \in A } r _ { s } ^ { a } \mu _ { s } ^ { a } } \\ & { \displaystyle \mathrm { s u b j e c t ~ t o } } \\ & { \displaystyle \sum _ { v _ { \alpha } \in A } \mu _ { s } ^ { a } - \gamma \sum _ { s ^ { \prime } } \sum _ { v _ { \alpha } } \bar { P } _ { s ^ { \prime } s } ^ { a } ( b ) \mu _ { s ^ { \prime } } ^ { a } = e _ { s } , \forall s } \\ & { \displaystyle \bar { P } _ { s ^ { \prime } s } ^ { a } ( b ) = f _ { \operatorname* { m p m } } ( P _ { s ^ { \prime } s } ^ { a } , b ) } \\ & { \displaystyle \sum _ { b , j \leq B } B } \\ & { \displaystyle \mu _ { s } ^ { a } \leq \frac { 1 } { 1 - \gamma } , \forall s , a . } \end{array}\tag{3}
$$

This is a nonsmooth bilinear mathematical program because of the piece-wise linear model in equation 2 and bilinear terms $\bar { P } _ { s ^ { \prime } s } ^ { a } ( b ) \mu _ { s ^ { \prime } } ^ { a }$ <sub>′</sub> in the constraints. While being a nonsmooth, nonconvex nonlinear problem, there exist powerful techniques to solve it to global optimality. In particular, piecewise McCormick envelopes can be used to develop convex relaxations of the bilinear constraints and are supported by popular solvers, such as Gurobi [31].

Theorem 1. Deliberate Practice is budget-optimal, i.e, it computes a globally optimal allocation of the practice budget.

Proof. First, we show that the bilinear program in Equation 3 is an exact reformulation of the bilevel program in Equation 1. Our key idea is to replace the inner optimization with its linear programming formulation. This allows us to convert the bilevel problem into a max min problem. Then, we utilize LP duality to reformulate it as a single-level max problem with bilinear constraints. We solve equation 3 to global optimality using spatial branch and bound methods [32], which handle bilinear constraints via convex relaxation techniques, such as Piecewise McCormick Envelopes. Constructing these relaxations requires valid finite bounds on all the variables appearing in the bilinear constraints. We derive tight lower and upper bounds for all variables $\mu$ and $P$ appearing in bilinear constraints, enabling the certification of global optimality. In particular, $P _ { s s ^ { \prime } } ^ { a } \leq 1$ since it corresponds to transition probability, and $\textstyle \mu _ { s } ^ { a } \leq \frac { 1 } { 1 - \gamma }$ since it is the discounted state-action occupancy measure. A detailed proof is in Appendix A.2. □

## 5.3 Skill Practice

Once the robot computes a budget allocation and a corresponding budget-optimal task plan, it constructs a curriculum to practice the selected skills. This requires the robot to plan with its existing skills to first reach the precondition of the skill being practiced [28]. Specifically, the robot sequentially masters each of the skills on the budget-optimal plan computed by Deliberate Practice and uses each newly acquired skill to reach and practice subsequent skills.

## 6 Experimental Evaluation

We evaluate our method in three long-horizon table-top manipulation tasks. Our simulated environments are implemented in MuJoCo [33] using LIBERO [34]:

1. Cleanup (simulated). A Franka Panda robot must clear a table by placing an object into one of three drawers. The task MDP has 47 abstract states and 10 skills in total, with the longest task plan consisting of 4 skills. The robot is provided abstract specifications of Pick, Place, OpenDrawer, and CloseDrawer skills.

2. Cleanup-Multi (simulated). This is a challenging multi-object version of the Cleanup task in which four objects must be placed into a drawer. The task MDP has 5000 abstract states and 22 skills in total, with the longest task plan consisting of 10 skills.

![](images/ce3d9b3a86e0a5db02d7ad985792608f12162e7bce9d07655e4aa32633627ed4.jpg)  
Figure 4: (a) Breakfast task requires learning forceful manipulation skills to interact with novel articulated objects. (b) Cleanup task requires the robot to place objects in the top, middle, or bottom drawer with rewards 1, 2 and 4 respectively.

3. Breakfast (real-robot). A Franka Panda robot can toast bread or make hot oatmeal using a microwave to achieve a higher reward. The former requires learning a StartToaster skill, while the latter requires learning OpenMicrowave and CloseMicrowave skills, and hence is harder. These skills require forceful interaction with novel articulated objects and hence must be learned by practicing. The robot has additional pick and place skills for transporting objects using motion planning that need not be learned.

Baselines. We compare Deliberate Practice (DP) against the following baselines: (1) Estimate, Extrapolate, Situate (EES) estimates the expected improvement of each skill and greedily practices the skill that most improves task performance [28]. (2) Competence Improvement (CI) practices the skill with the highest expected competence improvement [35, 36]. (3) Least Competent First (LCF) practices the skill with the lowest estimated competence. (4) Random (R) randomly samples a reachable skill for practice. Implementation details are provided in the Appendix.

Skill Practice. We implement all learnable skills as Cartesian-space impedance controllers [37] parameterized by waypoints and impedance parameters. For each skill, we define a reward function and optimize the corresponding interaction policy parameters using CMA-ES [38]. We use CMA-ES because our objective is evaluated through rollouts and is not differentiable with respect to the policy parameters. CMA-ES is a robust gradient-free optimizer for such continuous black-box objectives.

## 6.1 Experimental Results

DP is budget-aware. Figure 4 shows three possible task plans in the Cleanup task that place items in the top, middle and bottom drawers to achieve rewards of 1, 2, and 4, respectively. We compare the effect of different practice budgets on all methods in Figure 5. DP intelligently chooses which skills to learn based on the available budget: conservatively placing items in the top drawer when the budget is 100 episodes, placing items in the middle drawer under a budget of 150 episodes, and maximizing reward by placing items in the bottom drawer under a budget of 250 episodes. By contrast, baselines cannot adapt their behavior to the budget and greedily practice the easiest (and lowest-reward) task plan, irrespective of the budget.

DP significantly outperforms prior active learning methods. As shown in Figure 5, DP and EES perform similarly under a low budget, but DP significantly outperforms all baselines in medium (150 episodes) and high budget (250 episodes) settings. The greedy active learning baselines, EES and CI, reason myopically and, therefore, fail to discover higher-reward plans which require practicing skills that do not immediately improve task performance. This trend continues in the more complex Cleanup-Multi task, which requires sequencing 10 skills. Interestingly, random practice outperforms EES in this setting because EES lacks an exploration mechanism when the myopic task improvement measure $\Delta J _ { \mathrm { t a s k } }$ is zero for all skills. By contrast, DP can look ahead and hence does not suffer from this limitation.

![](images/150dd631bc01a6a9919779145ee2bf1e516225ebaa5f259375a2aebb25fac9ba.jpg)  
Deliberate PracticeEESLeast Competent FirstCompetence Improvement Random  
Figure 5: We evaluate all methods on Cleanup with budgets (B) of 100 (low), 150 (medium), and 250 (high) episodes. While they perform similarly under a low budget, DP is significantly better under medium and high budgets. We report the mean and standard deviation averaged over 5 seeds.

DP for larger problems. We evaluate the scalability of DP in Cleanup-Multi, which has 22 skills and 5000 abstract states resulting in a large bilinear program. Despite the size of this problem, DP consistently computes the optimal budget allocation within a maximum solve time of 6 minutes. In larger settings where the solver may not prove global optimality within the allotted time, the same optimization procedure can still return a bounded-suboptimal allocation as Gurobi maintains an incumbent feasible solution and a global bound from convex relax ations of the bilinear constraints, providing an optimality-gap certificate upon early termination.

Real-world Validation. We conduct experiments on a real Franka Panda robot to validate our design choices. To complete the Breakfast task, the robot must learn either to toast bread or microwave oatmeal. As shown in Figure 1, toasting provides a reward of 1 and requires practicing only the StartToaster skill, while microwaving oatmeal provides a reward of 2, but requires practicing two skills: OpenMicrowave and CloseMicrowave. The robot is provided with additional Pick and Place skills that can be used to transport items. We evaluate DP under two practice budgets: 30 episodes and 60 episodes. As shown in Figure 6, DP correctly chooses the conservative toast-bread plan under the smaller budget, but learns to microwave oatmeal under the larger budget, thereby achieving higher task reward.

![](images/e008199b81fbe4bea550d02ef20894803cfa4ccbe870b5cfd41b3e9267f3fd20.jpg)  
Figure 6: Real-world Breakfast task.

## 7 Conclusion

We propose Deliberate Practice, an active skill learning algorithm that allocates a limited practice budget to many robot skills to provably maximize long-horizon planning performance. Our algorithm models each skill’s competence as a function of the practice budget, computes an optimal budget allocation, and sequentially practices the skills in the environment. Computing this allocation requires solving a challenging bilevel optimization problem, for which we derive an exact single-level reformulation that can be solved using off-the-shelf solvers. Simulated and real-robot experiments show that our method is uniquely budget-aware and allows robots to better leverage their practice budgets compared to existing methods to master long-horizon manipulation tasks.

## 8 Limitations

One key assumption of our approach is access to approximate priors over skill competence. If these priors are overly optimistic, the robot may allocate practice to task plans that are actually infeasible within the available budget. This limitation could be addressed by using conservative priors or by explicitly incorporating uncertainty in skill-competence predictions. A second limitation is that solving the bilinear program to global optimality may become challenging for very large problems. An important direction for future work is therefore to develop bounded-suboptimal optimization strategies. Finally, we plan to extend our approach to practicing generalizable skills acros multiple tasks on mobile manipulators, moving toward the broader goal of general-purpose robots.

## Acknowledgments

This work was supported by the Office of Naval Research (ONR) under REPRISM MURI N000142412603 and ONR grant N00014-22-1-2592, as well as by the National Science Foundation (NSF) via grant 1955361 and ARL grant W911NF-18-2-0218. Partial funding was also provided by the Robotics and AI Institute. Sudarshan was supported by the Office of Naval Research (ONR) grant N00014-22-1-259.

## References

[1] P. Pastor, M. Kalakrishnan, S. Chitta, E. Theodorou, and S. Schaal. Skill learning and task outcome prediction for manipulation. In 2011 IEEE international conference on robotics and automation, pages 3828–3834. IEEE, 2011.

[2] Y. Zhu, Z. Jiang, P. Stone, and Y. Zhu. Learning generalizable manipulation policies with object-centric 3d representations. arXiv preprint arXiv:2310.14386, 2023.

[3] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.

[4] S. Cambon, R. Alami, and F. Gravot. A hybrid approach to intricate motion, manipulation and task planning. The International Journal ofRobotics Research, 28(1):104–126, 2009.

[5] C. R. Garrett, R. Chitnis, R. Holladay, B. Kim, T. Silver, L. P. Kaelbling, and T. Lozano-Perez.´ Integrated task and motion planning. Annual review of control, robotics, and autonomous systems, 4(1):265–293, 2021.

[6] B. Hedegaard, Z. Yang, Y. Wei, A. Jaafar, S. Tellex, G. Konidaris, and N. Shah. Beyond task and motion planning: Hierarchical robot planning with general-purpose policies. arXiv preprint arXiv:2504.17901, 2025.

[7] J. Liang, M. Sharma, A. LaGrassa, S. Vats, S. Saxena, and O. Kroemer. Search-based task planning with learned skill effect models for lifelong robotic manipulation. In 2022 International Conference on Robotics and Automation (ICRA), pages 6351–6357. IEEE, 2022.

[8] I. Mishani, Y. Shaoul, and M. Likhachev. Mosaic: A skill-centric algorithmic framework for long-horizon manipulation planning. arXiv preprint arXiv:2504.16738, 2025.

[9] J. Kober, J. A. Bagnell, and J. Peters. Reinforcement learning in robotics: A survey. The International Journal ofRobotics Research, 32(11):1238–1274, 2013.

[10] M. P. Deisenroth, G. Neumann, and J. Peters. A survey on policy search for robotics. Founda tions and Trends® in Robotics, 2(1-2):1–142, 2013.

[11] S. Gu, E. Holly, T. Lillicrap, and S. Levine. Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates. In 2017 IEEE international conference on robotics and automation (ICRA), pages 3389–3396. IEEE, 2017.

[12] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.

[13] D. Kalashnikov, A. Irpan, P. Pastor, J. Ibarz, A. Herzog, E. Jang, D. Quillen, E. Holly, M. Kalakrishnan, V. Vanhoucke, et al. Scalable deep reinforcement learning for vision-based robotic manipulation. In Conference on robot learning, pages 651–673. PMLR, 2018.

[14] K. Anders Ericsson. Deliberate practice and acquisition of expert performance: a general overview. Academic emergency medicine, 15(11):988–994, 2008.

[15] B. Da Silva, G. Konidaris, and A. Barto. Active learning of parameterized skills. In International Conference on Machine Learning, pages 1737–1745. PMLR, 2014.

[16] S. Chernova and M. Veloso. Interactive policy learning through confidence-based autonomy. Journal ofArtificial Intelligence Research, 34:1–25, 2009.

[17] A. Fabisch and J. H. Metzen. Active contextual policy search. The Journal of Machine Learning Research, 15(1):3371–3399, 2014.

[18] S. Vats, O. Kroemer, and M. Likhachev. Synergistic scheduling of learning and allocation of tasks in human-robot teams. In 2022 International Conference on Robotics and Automation (ICRA), pages 2789–2795. IEEE, 2022.

[19] S. Vats, M. Zhao, P. Callaghan, M. Jia, M. Likhachev, O. Kroemer, and G. Konidaris. Optimal interactive learning on the job via facility location planning. Robotics: Science and Systems (RSS), 2025.

[20] S. Srivastava, E. Fang, L. Riano, R. Chitnis, S. Russell, and P. Abbeel. Combined task and motion planning through an extensible planner-independent interface layer. In 2014 IEEE international conference on robotics and automation (ICRA), pages 639–646. IEEE, 2014.

[21] M. Fox and D. Long. Pddl2. 1: An extension to pddl for expressing temporal planning domains. Journal ofartificial intelligence research, 20:61–124, 2003.

[22] K. Hauser. Task planning with continuous actions and nondeterministic motion planning queries. In Proc. of AAAI Workshop on Bridging the Gap between Task and Motion Planning, 2010.

[23] C. R. Garrett, T. Lozano-Perez, and L. P. Kaelbling. Pddlstream: Integrating symbolic planners´ and blackbox samplers via optimistic adaptive planning. In Proceedings of the international conference on automated planning and scheduling, volume 30, pages 440–448, 2020.

[24] R. Chitnis, D. Hadfield-Menell, A. Gupta, S. Srivastava, E. Groshev, C. Lin, and P. Abbeel. Guided search for task and motion plans using learned heuristics. In 2016 IEEE International Conference on Robotics and Automation (ICRA), pages 447–454. IEEE, 2016.

[25] T. Silver, R. Chitnis, J. Tenenbaum, L. P. Kaelbling, and T. Lozano-Perez. Learning sym- ´ bolic operators for task and motion planning. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3182–3189. IEEE, 2021.

[26] N. Shah and S. Srivastava. Hierarchical planning and learning for robots in stochastic settings using zero-shot option invention. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 10358–10367, 2024.

[27] S. Vats, M. Likhachev, and O. Kroemer. Efficient recovery learning using model predictive meta-reasoning. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 7258–7264. IEEE, 2023.

[28] N. Kumar, T. Silver, W. McClinton, L. Zhao, S. Proulx, T. Lozano-Perez, L. P. Kaelbling, ´ and J. Barry. Practice makes perfect: Planning to learn skill parameter policies. In Robotics: Science and Systems (RSS), 2024.

[29] B. Abbatematteo, E. Rosen, S. Thompson, T. Akbulut, S. Rammohan, and G. Konidaris. Composable interaction primitives: A structured policy class for efficiently learning sustainedcontact manipulation skills. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 7522–7529. IEEE, 2024.

[30] M. L. Puterman. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.

[31] Gurobi Optimization, LLC. Gurobi Optimizer Reference Manual, 2024. URL https:// www.gurobi.com.

[32] M. Tawarmalani and N. V. Sahinidis. Convexification and global optimization in continuous and mixed-integer nonlinear programming: theory, algorithms, software, and applications, volume 65. Springer Science & Business Media, 2013.

[33] E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 5026–5033. IEEE, 2012.

[34] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310, 2023.

[35] A. Stout and A. G. Barto. Competence progress intrinsic motivation. In 2010 IEEE 9th international conference on development and learning, pages 257–262. IEEE, 2010.

[36] C. Colas, P. Fournier, M. Chetouani, O. Sigaud, and P.-Y. Oudeyer. Curious: intrinsically motivated modular multi-goal reinforcement learning. In International conference on machine learning, pages 1331–1340. PMLR, 2019.

[37] N. Hogan. Impedance control: An approach to manipulation: Part i—theory. Journal of Dynamic Systems, Measurement, and Control, 107(1):1–7, 1985.

[38] N. Hansen and A. Ostermeier. Completely derandomized self-adaptation in evolution strategies. Evolutionary computation, 9(2):159–195, 2001.

[39] G. P. McCormick. Computability of global solutions to factorable nonconvex programs: Part i—convex underestimating problems. Mathematical programming, 10(1):147–175, 1976.

[40] S. Garrido-Jurado, R. Munoz-Salinas, F. J. Madrid-Cuevas, and M. J. Mar˜ ´ın-Jimenez. Auto-´ matic generation and detection of highly reliable fiducial markers under occlusion. Pattern Recognition, 47(6):2280–2292, 2014.

[41] G. Bradski. The opencv library. Dr. Dobb’s Journal: Software Tools for the Professional Programmer, 25(11):120–123, 2000.

[42] I. A. Sucan, M. Moll, and L. E. Kavraki. The open motion planning library. IEEE Robotics & Automation Magazine, 19(4):72–82, 2012.

[43] S. Chitta, I. Sucan, and S. Cousins. Moveit![ros topics]. IEEE robotics & automation magazine, 19(1):18–19, 2012.

## A Appendix

## A.1 Linear Programming Formulation of MDPs

We leverage the linear programming (LP) formulation of MDPs to instantiate SolveMDP in Equation 1. Using the primal LP formulation for computing the optimal value function of an MDP, we get the following bilevel problem:

$$
\operatorname* { m a x } _ { b } \sum _ { s \in \mathcal { S } } e _ { s } v _ { s }
$$

subject to

$$
\sum _ { u \in \mathcal { U } } b _ { u } \le B
$$

$$
v _ { s } \in \arg \operatorname* { m i n } _ { v } \sum _ { s } e _ { s } v _ { s }
$$

subject to

$$
v _ { s } \geq r _ { s } ^ { a } + \gamma \sum _ { s ^ { \prime } } \bar { P } _ { s s ^ { \prime } } ^ { a } ( b ) v _ { s ^ { \prime } } , \forall a , s\tag{4}
$$

v unconstrained, ∀s

This is a bilevel program (also called a Stackelberg game), where the inner program (follower) is conditioned on decision variables that are chosen by the outer program (leader). In particular, the follower’s constraints depend on the budget decided by the leader since the transition function $\bar { P }$ is a function of the budget b via the competence improvement function $\bar { P } _ { s s ^ { \prime } } ^ { a } = f _ { \mathrm { i m p r o v e } } ( P _ { s s ^ { \prime } } ^ { a } , b )$

## A.2 Single-Level Reformulation

The follower is a b-parameterized LP for which Slater’s conditions hold.

Lemma 1. Suppose b is given. Then the LP min $\begin{array} { r } { , \{ e ^ { T } v \ : \ v _ { s } \ge \ r _ { s } ^ { a } + \gamma \sum _ { s ^ { \prime } } \bar { P } _ { s s ^ { \prime } } ^ { a } ( b ) v _ { s ^ { \prime } } , \ \forall s , a \} } \end{array}$ satisfies Slater’s condition.

Proof. The satisfaction of Slater’s condition is equivalent to showing that there exists vˆ such that inequalities are strictly satisfied. Let $\alpha = \operatorname* { m a x } _ { s , a } r _ { s } ^ { a }$ . Then, it is easy to verify $\begin{array} { r } { \hat { v } _ { s } = \frac { \alpha } { \gamma ^ { 2 } } \forall . } \end{array}$ s satisfies the inequalities strictly. This proves the claim. □

It is intractable to directly solve the bilevel program. A standard approach for reformulating such bilevel problems into single-level problems is to replace the follower with its KKT conditions since they are both necessary and sufficient for optimality. This results in a single-level optimization with complementarity constraints which can be solved using mixed integer programming. However, this formulation is usually difficult to solve in practice. Our key observation is that both the leader and the follower are optimizing the same objective. Hence, the bilevel problem can be rewritten as a max-min problem:

$$
\operatorname* { m a x } _ { b \in \mathcal { B } } \operatorname* { m i n } _ { v } \{ e ^ { T } v : v _ { s } \ge r _ { s } ^ { a } + \gamma \sum _ { s ^ { \prime } } \bar { P } _ { s s ^ { \prime } } ^ { a } ( b ) v _ { s ^ { \prime } } , \ \forall s , a \}
$$

where,

$$
{ \boldsymbol { B } } : = \{ b _ { u } : \sum b _ { u } \leq { \boldsymbol { B } } \}\tag{5}
$$

Since the LP satisfies Slater’s condition (Lemma 1) strong duality holds, i.e. the primal and dual are both feasible and attain their respective optimal values which are equal. By strong duality, the dual of the follower attains the same optimum as the primal. Hence, we can replace the inner min

problem with its dual max LP.

$$
\begin{array} { l } { { \displaystyle \operatorname* { m a x } _ { b \in { \mathcal { B } } } \operatorname* { m a x } _ { \mu } \sum _ { \forall s \in { \mathcal { S } } } \sum _ { \forall a \in { \mathcal { A } } } r _ { s } ^ { a } \mu _ { s } ^ { a } } } \\ { { \mathrm { s u b j e c t ~ t o } } } \\ { { \displaystyle \qquad \sum _ { \forall a \in { \mathcal { A } } } \mu _ { s } ^ { a } - \gamma \sum _ { s ^ { \prime } } \sum _ { \forall a } P _ { s ^ { \prime } s } ^ { a } \mu _ { s ^ { \prime } } ^ { a } = e _ { s } , \forall s } } \\ { { \mathrm { w h e r e } , \mathcal { B } : = \{ b _ { u } : \sum _ { b } b _ { u } \leq B \} } } \end{array}\tag{6}
$$

The inner problem is feasible for every choice of budget. Hence, we can jointly maximize the budget and dual variables in single max optimization:

$$
\begin{array} { l } { \displaystyle \underset { b , \mu } { \operatorname* { m a x } } \displaystyle \sum _ { \forall s \in \mathcal { S } } \displaystyle \sum _ { \forall a \in \mathcal { A } } r _ { s } ^ { a } \mu _ { s } ^ { a } } \\ { \displaystyle \mathrm { s u b j e c t ~ t o } } \\ { \displaystyle \quad \sum _ { \forall a \in \mathcal { A } } \mu _ { s } ^ { a } - \gamma \displaystyle \sum _ { s ^ { \prime } } \displaystyle \sum _ { \forall a } \bar { P } _ { s ^ { \prime } s } ^ { a } ( b ) \mu _ { s ^ { \prime } } ^ { a } = e _ { s } , \forall s } \\ { \displaystyle \sum _ { b _ { u } \leq B } } \end{array}\tag{7}
$$

This is a surprisingly simpler single-level program, where the budget constraints have been added to the dual LP formulation of the MDP. Note that it is a non-linear problem even when the competence improvement function is linear. However, standard optimizers, such as Gurobi [31] can natively handle such bilinear constraints via the classic McCormick Envelope [39].

## A.3 Experimental Details

## A.3.1 Skill Practice

We use CMA-ES to practice robot skills. CMA-ES is a gradient-free evolutionary-based optimization approach for reward maximization. For each skill, we define a reward function and optimize the corresponding interaction policy parameters using CMA-ES [38]. We use CMA-ES because our objective is evaluated through rollouts and is not differentiable with respect to the policy parameters. CMA-ES is a robust gradient-free optimizer for such continuous black-box objectives. We implement all learnable skills as Cartesian-space impedance controllers [37] parameterized by waypoints and impedance parameters. The optimizer takes as input the policy parameters to be optimized, together with a Gaussian prior over these parameters, specified by a mean and diagonal covariance matrix.

At each iteration of CMA-ES, we sample $N = 6$ candidate parameter vectors from the current Gaussian search distribution. Each candidate is evaluated by executing the corresponding policy on the real robot in the environment. At the end of each episode, the robot receives a binary reward, and the top half candidates are used to update the mean and covariance of the search distribution. Over successive iterations, the search distribution shifts toward regions of the parameter space that provide higher rewards and improve competence.

## A.4 Baselines

1. Estimate, Extrapolate, Situate (EES) estimates the expected improvement from practicing each skill independently using $f _ { \mathrm { i m p r o v } } ( \pi , 1 )$ . For each skill, it uses a task planner to compute the expected improvement in task performance after practicing that skill for one round:

$$
\underset { \pi } { \arg \operatorname* { m a x } } \Delta J _ { \mathrm { t a s k } } ( \pi ) = \underset { \pi } { \arg \operatorname* { m a x } } J _ { \mathrm { t a s k } } ( \bar { P } _ { \pi } ) - J _ { \mathrm { t a s k } } ( P ) ,
$$

where $\bar { P } _ { \pi }$ is the skill transition matrix after practicing π for one round. $J _ { \mathrm { t a s k } }$ is computed by solving the MDP induced by P using its linear programming formulation. The robot greedily practices the skill predicted to yield the largest task-performance improvement in the next round [28].

Bimanual Robot  
Intel RealSense Camera  
![](images/abf1ac1773d2baa3b673640d3d21475cea484388634d4a366563240a55ee406a.jpg)  
(a) Breakfast Domain

![](images/f7fc8c2cf4006ff4b85e536d0b4b947c84776c0525e359ea81ac5a8cdb32e363.jpg)  
(b) Cleanup-Multi Domain  
Figure 7: (a) Breakfast domain. The robot can either toast bread to achieve reward 1 or microwave oatmeal to achieve reward 2. The former requires practicing StartToaster while the latter requires practicing OpenMicrowave and CloseMicrowave. (b) Cleanup-multi domain. The achieves reward 1 for placing all objects in the top drawer, reward 2 for placing them in the middle drawer and reward 4 for placing them in the bottom drawer. The lower drawers are blocked by the upper drawers, so the robot must learn CloseDrawer skills to place objects in the lower drawers.

2. Competence Improvement (CI) practices the skill with the highest expected competence improvement [35, 36], i.e.,

$$
\arg \operatorname* { m a x } _ { \pi } { f _ { \mathrm { i m p r o v } } ( \pi , 1 ) }
$$

CI is biased towards skills that are easy to learn and, hence often wastes time on skills that are not relevant to the task.

3. Least Competent First (LCF) practices the skill with the lowest current competence, i.e,.

$$
\arg \operatorname* { m i n } _ { \pi } p _ { \pi }
$$

This encourages the robot to focus on underdeveloped skills and gradually master the full skill set. However, it can waste time on skills that are not relevant to the task or are too hard to learn within the practice budget.

4. Random (R) unformly randomly samples a reachable skill for practice.

## A.5 Real-Robot Breakfast Domain

As shown in Figure 7, the Breakfast task is conducted on a bimanual robot with two Franka Panda arms.. The skill library comprises 7 skills: PickBread, PlaceBread, StartToaster, OpenMicrowave, PickBowl, PlaceBowl, CloseMicrowave. Our perception system includes a wrist mounted Intel-Realsense D435 camera. Auroco Markers are mounted on the interior of the microwave door. Once the door is opened, the markers are detected using OpenCV Auroco Detection [40, 41], and depth measurements are used to estimate the dimensions and position of the microwave door. The resulting detected pose is used to construct a collision model of the scene, which enables collision-aware motion planning using OMPL [42] through MoveIt [43]. Contactrich skills StartToaster, OpenMicrowave, CloseMicrowave require forceful interaction with the toaster and microwave and are implemented using Cartesian-space impedance controllers parameterized by relative actions in the end-effector frame and impedance parameters that determine the forces applied by the robot during interaction. These impedance parameters are sensitive

![](images/3e76c96a5714172fac9b29df6648c4de00f030ae3f0717255fec4391611515ec.jpg)  
Door Pose Estimation using Camera

![](images/f75326e4f24c3373d7b58dfd302167f1afa286204c3792801ad0a9ec1ea3ccf0.jpg)  
Collision Model for Motion Planning

Figure 8: Breakfast domain. After pressing the microwave button, the robot estimates the position of the microwave door using a RealSense wrist camera. This information is used to construct a collision model of the scene for motion planning.

to the object properties and task context, and therefore must be learned through practice. To ensure safe exploration, we specify a bounded range of impedance parameters that the robot may explore and use a reward function to encourage the robot to apply the minimum force necessary to solve the task.

## A.5.1 Cleanup Domain

The Cleanup domain requires a Franka Panda robot to place an item in one of three open drawers to clean the table. Cleanup-multi is a harder version with 4 items on the table (shown in Figure 7). Both of them are implemented in MuJoCo [33] using LIBERO [34] which provides realistic physics simulation. The robot receives higher reward for placing items in lower drawers. However, these placements are more challenging to learn because they require the robot to first learn how to close the upper drawers. The Cleanup domain has 41 abstract states, and Cleanup-multi has 5001 abstract states consisting of the following variables: DrawerState (open, closed) for every drawer, ObjectState (in-hand, on-table, in-top-drawer, in-middle-drawer, in-bottom-drawer) for every object, and a terminal failure state. The robot is provided with abstract specifications for 22 manipulation skills including their preconditions, termination conditions and effects:

1. OpenDrawer × {top, middle, bottom} drawer: opens a drawer by grasping the handle and pulling it. The pull action is learned through practice.

2. CloseDrawer × {top, middle, bottom} drawer: closes a drawer by pushing. The push action is learned through practice.

3. Pick × 4 items: picks an object with a grasp pose learned through practice.

4. Place × 4 items × 3 drawers: moves the object to the target drawer and places it. The object release position depends on the object size and the target drawer and is learned from practice.

All skills are implemented as Cartesian-space impedance controllers with a fixed impedance and waypoints defined with respect to the target object. We use privileged information about object positions from the simulator to prameterize all object-centric skills. The highest reward task plan requires sequencing 10 skills to first close the top two drawers and place all items in the lowest drawer.
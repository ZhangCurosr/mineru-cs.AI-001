# Orbit-Planner: Towards Latent World Models for On-Orbit Obstacle Avoidance of Satellite Agents

Zhijian Li , Chao Ren<sup>∗</sup> , Peijin Wang , Xian Sun Senior Member, IEEE

Aerospace Information Research Institute, Chinese Academy of Sciences, 100094 Beijing, China

lizhijian25@mails.ucas.ac.cn, renc0003@e.ntu.edu.sg

Code HuggingFace Project Page

Abstract—Satellite agents for on-orbit navigation tasks need to predict collision risks using limited onboard observations. However, conventional planners often rely on predefined maps and fixed environmental assumptions, limiting their adaptability in dynamic on-orbit scenarios. In this paper, we propose Orbit-Planner, a two-stage latent world model for on-orbit obstacle avoidance. Orbit-Planner learns action-conditioned spacecraft dynamics to perform future-state rollouts in latent space, and introduces a Physics Probe to decode physical state changes from imagined latent trajectories. Experiments demonstrate that Orbit-Planner can perform long-horizon latent rollouts and recover physical states from imagined trajectories. In closedloop obstacle-avoidance navigation in Isaac Sim, it attains a success rate of 91.7%. Code is available at https://github.com/ ZhijianLi2003/Orbit Planner.

Index Terms—Latent world models, Space robotics, Autonomous navigation, On-orbit obstacle avoidance

## I. INTRODUCTION

With the rapid increase in space exploration and on-orbit assembly tasks, autonomous satellite agents are playing an increasingly critical role [1]. In these complex and dynamic orbital environments, effective obstacle avoidance is paramount to ensure the safety and success of space missions. Traditional methods often rely on predefined environmental parameters and classical path-planning algorithms, such as Artificial Po-<sup>[</sup> tential Fields and A\* search [2]. Imitation-learning policies [3] can generate reactive maneuvers from demonstrations, but they often generalize poorly when the test-time dynamics deviate from the demonstrated distribution.

To overcome these issues, world models have emerged as a transformative paradigm. By extracting compact predictive representations from high-dimensional sensory inputs, these models empower agents to internally simulate future states and evaluate actions prior to physical execution [4, 5]. Notably, recent breakthroughs in Joint-Embedding Predictive Architectures (JEPA) have enabled stable, end-to-end learning of such models [6], demonstrating remarkable efficacy in highly dynamic control scenarios like agile quadrotor flight [7]. Inspired by these advancements, latent world models present a compelling pathway for space robotics.

![](images/e8e88d689c7878047b51f1ed8a4ee4a28e2c44b658880e379c47643445c4a941.jpg)  
Fig. 1. Overview of the on-orbit obstacle-avoidance task and the overall pipeline. Left: a CubeSat navigates from a start position to a target through a cluttered orbital field of obstacles. Right: the pipeline of data collection, world-model training, and testing.

The main contributions of this paper are as follows: 1) we construct an on-orbit obstacle-avoidance dataset for satellite agents based on Isaac Sim; 2) we propose Orbit-Planner, a two-stage latent world model that learns action-conditioned dynamics for future-state rollout, with a Physics Probe to recover physical states from imagined trajectories; 3) based on this world model, we achieve online on-orbit obstacleavoidance navigation, with a 91.7% success rate.

## II. METHODOLOGY

## A. Problem Formulation

We consider a discrete-time dynamical system for a CubeSat agent operating in a dynamic and cluttered orbital environment. As illustrated in Fig. 1, the primary objective is to navigate the satellite safely to a target destination while minimizing the probability of collision. The system is characterized by its state $\boldsymbol { s } _ { t } \in \mathbb { R } ^ { 1 \bar { 6 } }$ and control action $\mathbf { \boldsymbol { a } } _ { t } \in \mathbb { R } ^ { 8 }$

![](images/03917933309120f10592231d722d3008a3d0c31e6381475bd0ebf31564d94454.jpg)  
Fig. 2. Overview of the proposed Orbit-Planner framework. Stage I pre-trains the latent world model to represent multimodal observations (RGB sequences and spacecraft states) and perform action-conditioned latent rollouts. In Stage II, the Physics Probe maps rolled-out latents to future spacecraft-state increments, while the depth decoder recovers current obstacle geometry; both are used for trajectory planning.

For the CubeSat, we define the state as:

$$
\begin{array} { r } { \pmb { \mathscr { s } } _ { t } = [ \pmb { p } _ { t } ^ { \top } \quad \pmb { v } _ { t } ^ { \top } \quad \pmb { r } _ { t } ^ { \top } \quad \pmb { \omega } _ { t } ^ { \top } \quad \phi _ { t } ] ^ { \top } . } \end{array}\tag{1}
$$

Here, ${ \pmb p } _ { t } \in \mathbb { R } ^ { 3 }$ denotes the position displacement in the inertial (world) frame relative to the initial position $( { \pmb p } _ { 0 } = { \bf 0 } )$ The linear velocity ${ \pmb v } _ { t } \in \mathbb { R } ^ { 3 }$ and angular velocity $\boldsymbol { \omega } _ { t } \in \mathbb { R } ^ { 3 }$ are expressed in the body frame. The attitude is represented by a continuous 6D rotation vector $\boldsymbol { r } _ { t } ~ \in ~ \mathbb { R } ^ { 6 }$ , derived from the first two columns of the rotation matrix $\pmb { R } _ { t } ~ \in ~ S O ( 3 )$ representing the attitude of the body frame in the inertial frame. Additionally, $\phi _ { t } \in [ 0 , 1 ]$ represents the remaining fuel ratio. The discrete time step is set to $\Delta t = 0 . 0 4 \mathrm { s } \ ( 2 5 \mathrm { H z } )$

The control action $\mathbf { \boldsymbol { a } } _ { t } \in \mathbb { R } ^ { 8 }$ denotes the normalized thrust commands for the 8 onboard thrusters:

$$
\begin{array} { r } { \pmb { a } _ { t } = [ F _ { 1 , t } , F _ { 2 , t } , \ldots , F _ { 8 , t } ] ^ { \top } \in [ 0 , 1 ] ^ { 8 } . } \end{array}\tag{2}
$$

The system operates under the assumption that the agent’s physical state $\mathbf { } _ { s _ { t } }$ is fully observable. To capture surrounding environmental information, the agent utilizes an onboard RGB camera to obtain visual observations $\pmb { o } _ { t } \in \mathbb { R } ^ { H \times W \times 3 }$

## B. Orbit-Planner World Model Framework

As illustrated in Fig. 2, Orbit-Planner consists of two stages: world-model pre-training, followed by physics probing and planning.

1) Stage I: World Model Pre-training: In the first stage, we pre-train a latent world model to capture the complex dynamics of the space environment. This stage fundamentally consists of two processes: representation and rollout. During the representation phase, the model takes RGB observations o<sub>t</sub> and low-dimensional spacecraft states $\mathbf { } _ { s _ { t } }$ as inputs. A Vision Transformer (ViT) acts as the visual encoder, extracting a global class token and dense patch tokens. Concurrently, an MLP-based State Encoder maps the physical state into a state embedding. A Projector module then fuses the class token and the state embedding to generate the latent representation $z _ { t } .$ which jointly encodes the agent’s own physical state and the surrounding environmental geometry. Crucially, this projector employs Batch Normalization to counteract the LayerNorm effects from the ViT, ensuring that the variance-based regularization (SIGReg [6]) functions effectively on the latent distribution without collapsing. The dense patch tokens are fed to a dedicated depth decoder for depth prediction.

During the rollout phase, the control action $\mathbf { } \mathbf { a } _ { t }$ is introduced. The thruster commands are processed by an MLP-based Action Encoder to produce an action embedding. The Latent Dynamics Prediction module, parameterized by an autoregressive Transformer equipped with Adaptive Layer Normalization (AdaLN), then utilizes the current latent state ${ \boldsymbol { z } } _ { t }$ and the action embedding to predict the future latent state $z _ { t + 1 }$ . Specifically, the action embedding acts as the conditioning signal in the AdaLN blocks [8], dynamically modulating the latent features to ensure the physical actions effectively guide the future-state rollout. To ensure the learned representations are physically meaningful and geometrically consistent, the training is guided by a composite loss function $\mathcal { L } \mathrm { : ~ }$

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { M S E } } + \mathcal { L } _ { \mathrm { S I G R e g } } + \mathcal { L } _ { \mathrm { D e p t h } } , } \end{array}\tag{3}
$$

which incorporates the mean squared error $( \mathcal { L } _ { \mathrm { M S E } } )$ for latent dynamics prediction, variance-based regularization constraints $( \mathcal { L } _ { \mathrm { { S I G R e g } } } )$ to prevent feature collapse, and depth supervision $( \mathcal { L } _ { \mathrm { D e p t h } } )$ via a dedicated depth decoder. The depth loss is included mainly to encourage the visual encoder to capture depth cues that are critical for navigation.

![](images/8fb79686c39a6cc4d973c253380261118c54f46d8f8be906fe3109aba8d38768.jpg)  
Fig. 3. Latent-space rollout prediction error versus horizon. Thin blue curves denote individual trajectories; the red curve denotes the mean MSE.

2) Stage II: Physics Probing and Planning: After pretraining, we freeze the world model and use it to roll out future states. Given a sequence of past observations $\left( o _ { t - m + 1 } , \ldots , o _ { t } \right)$ , the corresponding states $\left( \pmb { s } _ { t - m + 1 } , \ldots , \pmb { s } _ { t } \right)$ and a sequence of proposed future actions $( \pmb { a } _ { t } , \dots , \pmb { a } _ { t + n - 1 } )$ the world model predicts the corresponding future latent states $( z _ { t + 1 } , \ldots , z _ { t + n } )$ . During training, the observation context length is set to $m = 8$ and the rollout horizon is set to $n = 1 2$ To map latent states to physical quantities, we introduce a Physics Probe and train only this probe while keeping the world model frozen. The probe translates the latent states into physical state transitions $( \Delta \pmb { s } _ { t + 1 } , \dots , \Delta \pmb { s } _ { t + n } )$ . Simultaneously, the frozen depth decoder extracts obstacle information, converting it into a Depth-to-Point Cloud representation.

## C. Autonomous Control and Decision-Making

Given the probe-predicted state rollouts and obstacle point clouds, Orbit-Planner employs MPPI [9] with K = 256 samples and a horizon of $H = 5 0$ steps, and selects the sequence with the lowest predicted collision risk while progressing toward the target. Only the first action of the selected sequence is executed before replanning.

## III. EXPERIMENTS

## A. Experimental Setup

We collect data in a space robotics simulation environment [1] based on Isaac Sim. The dataset contains 8000 on-orbit obstacle-avoidance trajectories, split into 7200 for training and 800 for testing. To improve the robustness of the learned representations under visual and geometric variations, extensive domain randomization is applied during data collection, encompassing variations in obstacle positions and lighting conditions. Furthermore, to improve behavioral diversity, trajectories are gathered under three distinct levels. Expert trajectories are generated by RRT<sup>∗</sup> path planning followed by PD tracking, with larger safety margins and lower control noise. Risky trajectories fly straight toward the goal without obstacle-aware planning. Exploratory trajectories inject strong random action noise for unstructured exploration.

## B. Latent Space Rollout

We evaluate action-conditioned multi-step prediction in the latent space z. Starting from an encoded context, the frozen world model recursively rolls out future latents under the recorded action sequence, and we measure the MSE against encoder-derived target latents at each horizon. Fig. 3 shows the per-trajectory errors (blue) and their mean (red). As expected for autoregressive imagination, the error accumulates with the horizon; nevertheless, the mean MSE grows gradually and remains moderate even at a 50-step horizon, indicating that the learned latent dynamics support stable long-horizon rollouts for action-conditioned trajectory imagination. Fig. 4 shows that the depth decoder recovers obstacle structure, indicating that z preserves useful geometric cues.

![](images/bf60dedd058aef295ec6754ff5d0a7989a3628eb628e89aac9a8bbe314d886b8.jpg)  
Fig. 4. RGB-to-depth prediction results. Both the ground-truth and predicted depth maps are downsampled to 16 × 16.

TABLE I  
PHYSICS PROBE ERRORS ON ABSOLUTE STATES RECOVERED VIA KINEMATIC INTEGRATION FROM PREDICTED INCREMENTS ∆s (MEAN ± STD OVER 100 RANDOMLY SELECTED TESTING TRAJECTORIES).
<table><tr><td>Quantity</td><td>MAE  $( h = 2 5 , t = 1 \mathrm { s } )$ </td><td>MAE  $( h = 5 0 , t = 2 \mathrm { s } )$ </td></tr><tr><td>Position p (m)</td><td> $0 . 0 3 9 4 \pm 0 . 0 5 5 4$ </td><td> $0 . 0 8 1 3 \pm 0 . 0 6 6 1$ </td></tr><tr><td>Velocity v (m/s)</td><td>0.0636 ± 0.0963</td><td> $0 . 1 1 9 1 \pm 0 . 1 1 0 2$ </td></tr><tr><td>Rotation (geodesic, rad)</td><td> $0 . 0 6 7 6 \pm 0 . 0 9 0 9$ </td><td> $0 . 2 0 3 4 \pm 0 . 1 8 2 6$ </td></tr><tr><td>Angular velocity ω (rad/s)</td><td> $0 . 0 8 8 3 \pm 0 . 1 0 8 5$ </td><td> $0 . 1 5 5 7 \pm 0 . 1 4 5 3$ </td></tr><tr><td>Fuel φ (normalized)</td><td> $0 . 0 0 1 9 \pm 0 . 0 0 1 5$ </td><td> $0 . 0 0 3 3 \pm 0 . 0 0 2 1$ </td></tr></table>

## C. Physics Probe

We evaluate the accuracy of physical-state recovery from action-conditioned latent rollouts. With the world model’s encoder and predictor frozen, an 8-step observation context is encoded into initial latent states. The model then autoregressively rolls out future latents conditioned on the recorded action sequences. Subsequently, the Physics Probe maps each predicted latent state to physical state increments $\Delta s .$ , which are kinematically integrated from the last context state to reconstruct the full physical trajectory. We compare these recovered states against the ground truth and report the Mean Absolute Error (MAE) across 25-step and 50-step prediction horizons. As shown in Table I, position, velocity, and fuel are recovered with low error, whereas angular velocity exhibits slightly higher variance due to its highly dynamic nature. Fig. 5 provides a qualitative comparison. Within a limited time horizon, the collision-free rollout remains aligned with the ground-truth trajectory. In a prediction window that contains a collision event, the error between the rollout and the recorded trajectory becomes substantially larger, which indirectly suggests that the model has captured action-conditioned dynamics rather than merely replaying the observed outcome. These results indicate that the action-conditioned latent rollouts preserve physically meaningful dynamics that can be decoded by the probe.

![](images/8884af62ba07306054dc4ec6527deb4830ff29db740eafcbea334fcd1b56d04d.jpg)  
(a) Physical state prediction  
(b) 3D Visualization

Fig. 5. Physics Probe qualitative results on a 50-step (2.0 s) prediction horizon. (a) Predicted versus ground-truth physical-state components on a collision-free trajectory. (b) 3D comparison of a collision-free window and a collision window.  
![](images/3620fffdb27ab315879336b536c8ddf7cb59b3c14a45f9294c38d8b58e68a939.jpg)  
Fig. 6. Closed-loop success rates of Orbit-Planner and Diffusion Policy [3] in Isaac Sim under six settings (1/3/5 obstacles × v<sub>0</sub> = 1/2 m/s) and their average.

## D. On-Orbit Obstacle Avoidance

We further evaluate closed-loop obstacle-avoidance navigation in Isaac Sim. Orbit-Planner uses the frozen world model to imagine action-conditioned futures and select collisionfree maneuvers, and is compared against an imitation-learning baseline, Diffusion Policy [3]. Both methods are tested under six settings that vary the number of obstacles and the initial velocity, with 10 episodes per setting. In each episode, the obstacles are randomly placed and unseen during training. As shown in Fig. 6, Orbit-Planner attains an average success rate of 91.7%, substantially outperforming Diffusion Policy (55.0%). The advantage is consistent across all six settings and remains pronounced at higher speed and denser obstacle fields. These results indicate that action-conditioned latent rollouts enable anticipatory, physics-aware decisions, whereas a purely reactive imitation policy struggles to generalize when the scene dynamics deviate from the demonstrated distribution.

## IV. CONCLUSION

In this paper, we presented Orbit-Planner, a latent world model for on-orbit obstacle avoidance by autonomous satellite agents. Orbit-Planner learns action-conditioned spacecraft dynamics to perform future-state rollouts in latent space, employing a Physics Probe to decode physical state transitions from imagined trajectories. Experimental results demonstrate that the proposed model achieves long-horizon latent rollouts, physical-state readouts, and higher closed-loop navigation success than an imitation-learning baseline. In future work, we will study the sim-to-real gap under more realistic sensing and dynamics conditions.

## REFERENCES

[1] A. Orsula, M. Geist, M. Olivares-Mendez, and C. Martinez, “Space robotics bench: robot learning beyond earth,” arXiv preprint arXiv:2509.23328, 2025.

[2] Y. Li, S. Yue, and Z. Du, “Obstacle avoidance method for onorbit assembly based on artificial potential field and improved a\* path-planning algorithm,” IFAC-PapersOnLine, vol. 59, no. 20, pp. 1350–1355, 2025.

[3] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, vol. 44, no. 10-11, pp. 1684–1704, 2025.

[4] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse control tasks through world models,” Nature, vol. 640, no. 8059, pp. 647–653, 2025.

[5] N. Hansen, H. Su, and X. Wang, “TD-MPC2: Scalable, robust world models for continuous control,” in International Conference on Learning Representations, 2024.

[6] L. Maes, Q. L. Lidec, D. Scieur, Y. LeCun, and R. Balestriero, “Leworldmodel: Stable end-to-end joint-embedding predictive architecture from pixels,” arXiv preprint arXiv:2603.19312, 2026.

[7] P. Rao, W. Zhang, R. Balestriero, Y. LeCun, and G. Loianno, “Skyjepa: Learning long-horizon world models for zero-shot simto-real control of quadrotors,” arXiv preprint arXiv:2606.23444, 2026.

[8] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4195–4205.

[9] G. Williams, A. Aldrich, and E. Theodorou, “Model predictive path integral control using covariance variable importance sampling,” arXiv preprint arXiv:1509.01149, 2015.
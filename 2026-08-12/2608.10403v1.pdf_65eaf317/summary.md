---
title: "Threat-guided Policy-aware Scene Perturbation for Safe Autonomous Driving with Online Reinforcement Learning"
source: https://arxiv.org/pdf/2608.10403v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:31:24"
field: "自动驾驶安全强化学习"
keywords: ["安全关键场景生成", "在线强化学习", "策略感知扰动", "自动驾驶", "威胁引导优化", "NAVSIM", "场景增强"]
innovations: ["冻结策略投影构建策略感知场景表示，实现扰动与当前策略弱点对齐", "以扰动前后同策略轨迹威胁差作为归一化优势训练场景生成网络", "在NAVSIM v2上以约400万公里交互刷新NC与TTC安全记录"]
benchmarks: ["NAVSIM v2 navhard_two_stage", "GPUDrive仿真训练"]
---

# 论文速读：Threat-guided Policy-aware Scene Perturbation for Safe Autonomous Driving with Online Reinforcement Learning

## 一句话总结
本文提出 TPSP（Threat-guided Policy-aware Scene Perturbation）框架，通过提取当前策略的隐藏表征构造策略感知场景编码，对关键车辆进行定向高斯扰动，并以扰动前后策略轨迹的威胁分差异作为优化信号，引导场景生成网络发现高训练价值的碰撞风险场景，从而在约400万公里在线RL交互预算下实现 NA VSIM v2  benchmark 上最安全的驾驶表现。

## 研究问题与动机
- 在线 RL 驾驶策略的安全关键长尾场景稀缺，依赖常规采样难以覆盖足以训练碰撞规避能力的交互。
- 现有对抗/关键场景生成以环境级难度为目标，未与正在更新的策略耦合，导致“对成熟策略 trivial、对幼稚策略危险”的错配。
- 纯随机或固定幅度扰动提升数据多样性但对当前策略弱点无针对性，训练信号低效。
- 缺乏闭环机制将策略薄弱点显式反馈到场景扰动目标上。

## 核心贡献（创新点）
- 提出策略感知场景编码器：将自车运动学、邻车交互特征、道路结构、风险指标与**冻结策略网络隐藏表征**融合，使扰动生成对齐当前策略行为。
- 定向关键目标高斯扰动：基于对象重要性 Softmax 选取 top-K 物体，由生成网络预测均值与方差并采样受边界约束的扰动，避免全场均匀修改。
- 威胁差异闭环优化：以原场景与扰动场景在同策略同种子 rollout 上的威胁差作为信号，归一化后驱动场景生成网络最大化高训练价值样本出现概率，并附加熵正则。
- 在 NAVSIM v2 navhard_two_stage 上刷新安全记录：约400万公里模拟数据下 S1 获 99.8% NC / 99.6% TTC、S2 获 96.7% NC / 94.6% TTC，较上一最优提升约 2.2% / 1.8%。
- 消融与定性验证策略感知与定向扰动的必要性：TPSP 优于 Vanilla PPO、随机扰动与去掉策略感知的变体；可视化显示 TPSP 能在切入场景中主动避让、在急刹场景中保持安全距离。

## 方法详解
- **状态空间形式化**：自动驾驶建模为 MDP $\mathcal{M}=(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma)$，策略 $\pi_\theta(a_t|s_t)$ 输出动作；场景扰动网络 $G_\phi$ 输出速度调整、生成时间偏移、自车初态修改三类扰动。
- **策略感知编码器**：
  - 自车特征 $e^{\text{ego}}$（速度、航向、目的地）、邻车特征 $e_i^{\text{obj}}$（类型、几何、相对位姿、相对速度、TTC）、道路特征 $e^{\text{road}}$、风险特征 $e^{\text{risk}}$（反向最小距离、反向最小TTC、邻车数量）。
  - 策略特征：用当前策略副本 $\pi_{\bar{\theta}}$（训练期间冻结）对原始观测做前向得到隐藏表征 $\mathbf{z}_{\pi_{\bar{\theta}}}$，经投影 MLP 得 $e^{\text{policy}}=\text{MLP}^{\text{policy}}(\mathbf{z}_{\pi_{\bar{\theta}}})$，参数解耦防止梯度流入策略网络。
  - 以自车为 query 的注意力聚合邻车交互：$c^{\text{obj}}=\text{Attention}(e^{\text{ego}},\{e_i^{\text{obj}}\})$。
  - 多源融合：$h=\text{MLP}^{\text{fusion}}([e^{\text{ego}},c^{\text{obj}},e^{\text{road}},e^{\text{risk}},e^{\text{policy}}])$。
- **定向扰动生成**：
  - 对象重要性打分：$w^{\text{obj}}=\text{Softmax}(\{f^{\text{score}}([e_i^{\text{obj}},h])\})$，选 top-$K$ 表示 $E_K^{\text{obj}}$。
  - 高斯采样与有界映射：$\mu=g_\mu([E_K^{\text{obj}},h]),\;\sigma=\exp(g_\sigma([E_K^{\text{obj}},h]))$，$z^{\text{raw}}\sim\mathcal{N}(\mu,\text{diag}(\sigma^2))$，再经 $z=\lambda(z^{\max}\odot\tanh(z^{\text{raw}}))$ 得到物理可行的扰动。
- **威胁评分与优化**：
  - 步级威胁：$c_t=w^{\text{ttc}}c_t^{\text{ttc}}+w^{\text{dist}}c_t^{\text{dist}}+w^{\text{edge}}c_t^{\text{edge}}+w^{\text{col}}c_t^{\text{col}}+w^{\text{off}}c_t^{\text{off}}$。
  - 轨迹级关键性（取 top-k 高危步平均）：$\mathcal{C}(\tau)=\frac{1}{k'}\sum_{t\in\text{Top}(\tau,k)}c_t$。
  - 时序不稳定性：$\mathcal{V}(\tau)=\frac{1}{T}\sum_{t=0}^{T-1}|c_{t+1}-c_t|$。
  - 最终威胁分：$\xi(\tau)=\mathcal{C}(\tau)\frac{1+\beta\mathcal{V}(\tau)}{1+\beta}$。
  - 威胁差：$\Delta\xi=\xi(\tau^{\text{per}})-\xi(\tau^{\text{org}})$，同策略同模拟器种子对照。
  - 归一化优势：$\hat{A}_s=(\Delta\xi_s-\text{mean})/(\text{std}+\epsilon)$。
  - 场景生成网络损失：$\mathcal{L}^{\text{per}}=-\mathbb{E}_s[\hat{A}_s\log p_\phi(z_s^{\text{raw}}|E_K^{\text{obj}},h)]-c^{\text{ent}}H_z$。
- **闭环流程**：每轮从原始数据集 $D$ 与缓冲区 $B$ 混合采样 → 冻结策略提取特征 → $G_\phi$ 生成扰动 → 同策略 rollout 训练驾驶策略 $\pi_\theta$（PPO）→ 按间隔更新 $G_\phi$ 并累积高威胁扰动场景到 $B$ → 同步策略副本与扰动强度调度。

## 实验与结果
- **环境与数据**：GPUDrive 模拟器 + PPO；训练数据为 NAVSIM v2 navtrain；评估使用 NAVSIM v2 navhard_two_stage（S1 评估原始场景，S2 评估更难合成场景）。
- **指标**：NC（无责碰撞率↑）、DAC（可行驶区域合规↑）、TTC（碰撞时间↑）。
- **消融对比**（表1）：Vanilla PPO S1/S2 为 96.2/85.7% NC；Random 为 97.1/83.2% NC；TPSP w/o PA 为 98.8/91.1% NC；TPSP 达 99.8/96.7% NC，S2 的 DAC/TTC 同样显著领先（97.8/93.7 与 99.6/94.6）。
- **与排行榜对比**（表2）：超越 NavFormer、RAP、ZTRS、SimScale、DrivoR 等，S1 的 NC/TTC 分别为 99.8%/99.6%，S2 为 96.7%/94.6%，较上一最优提升约 2.2%/1.8%。
- **效率**：2M miles 即达到 Vanilla PPO 在 4M miles 的碰撞奖励水平；定性可视化显示 TPSP 更早做出危险响应并维持安全距离。

## 相关工作脉络
- AdvSim / KING：通过优化邻车行为构造安全关键交互，但目标函数与环境级可行性相关，未耦合当前策略。
- SafeBench / CAT：提供基准与闭环对抗训练框架，强调环境难度，策略适配性有限。
- FREA / Dif-Scene：可行性感知对抗与扩散式关键场景生成，仍以场景失败概率为主目标。
- CaRL / 自适应课程：按策略能力调度经验或调整训练难度，但侧重于课程/体验编排而非目标级定向扰动。
- AlphaZero / 自动驾驶自博弈：展示策略相关交互生成的价值，TPSP 将其思想引入连续驾驶领域的闭环威胁引导。

## 局限性与未来方向
- 当前扰动空间限于场景级运动学与生成时序，未包含更丰富的语义、地图拓扑与多智能体协同扰动。
- 依赖 GPUDrive 的白盒结构化输入（自车/邻车状态、道路、风险指标），与主流基于多视图相机的端到端范式和真实传感器噪声存在鸿沟。
- 训练里程约400万公里，扩展到更大规模或真实世界部署时的泛化与仿真到现实的差距尚需验证。
- 当前仅演示 PPO 与固定策略冻结拷贝，与其他在线RL算法或动态教师策略的兼容性未系统评估。

## 研究启发与可借鉴点
- **冻结策略投影提取特征**：用 detach 的当前策略隐藏层作为辅助条件，既保留策略信号又避免梯度冲突，可迁移到其他“任务条件生成器”设计。
- **威胁差作为生成信号的归一化 Advantage**：在同策略同种子对照下用 batch 内标准化优势驱动生成网络，减少不同训练阶段威胁量纲漂移的影响。
- **目标选择 + 高斯边界映射**：对场景中关键对象做重要性加权后生成带界的连续扰动，兼顾可解释性与可微性，适合其他多主体交互仿真环境。
- **缓冲区混合与动态比例**：将历史高威胁扰动场景存入缓冲并与原始分布混合，渐进提升挑战比例，易于接入现有 on-policy RL 流水线。
- **评估双阶段对照**：S1（原始）与 S2（合成）分开报告可清晰区分泛化与强对抗鲁棒性，建议作为后续工作标配。

## 关键术语表
- **TPSP**：Threat-guided Policy-aware Scene Perturbation，面向在线RL自动驾驶的策略感知威胁引导场景扰动框架。
- **Policy-aware scene representation**：融合自车、邻车、道路、风险与冻结策略隐藏特征的联合场景表示。
- **Threat difference**：同一策略在同初始状态下于扰动与原场景上的轨迹威胁评分之差，用于评估扰动训练价值。
- **Rollout criticality $\mathcal{C}(\tau)$**：取轨迹中最危险 k 个时间步威胁分的均值，刻画整体风险强度。
- **Threat instability $\mathcal{V}(\tau)$**：轨迹步间威胁变化率的均值，衡量交互状态的时变剧烈程度。
- **Targeted top-K perturbation**：按对象重要性选取少量高相关目标进行速度/生成时机/自车初态的高斯扰动。
- **Detached policy feature**：从冻结策略网络投影得到的特征，用于指导扰动生成并断开对策略更新的梯度。
- **NAVSIM v2 navhard_two_stage**：NAVSIM v2 的两阶段安全评估基准，S1 为原始场景、S2 为更严苛合成场景。

## 可复现要素
- **数据集**：NAVSIM v2 navtrain（训练）、navhard_two_stage（评估）；NAVSIM 场景经转换接入 GPUDrive，论文未提供公开转换脚本声明。
- **代码/权重**：论文未明确声明开源代码或预训练权重。
- **关键超参**：PPO 学习率 3e-4、clip 0.2、epoch 2、discount 0.99、GAE 0.95、batch 204800、mini-batch 6400；扰动网络学习率 1e-4、Adam、entropy 0.01、clip 0.5、更新间隔 4 rollouts、初始扰动强度 λ=0.2、编码器维度 128、融合 256、策略特征 64、选 K=8、初始 log σ=-1.0；策略网络 MLP、隐层 64、动作离散（转向13档、加速7档）。硬件为 16×A100、512 并行环境。

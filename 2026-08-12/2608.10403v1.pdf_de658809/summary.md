---
title: "Threat-guided Policy-aware Scene Perturbation for Safe Autonomous Driving with Online Reinforcement Learning"
source: https://arxiv.org/pdf/2608.10403v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:32:27"
field: "自动驾驶安全强化学习"
keywords: ["自动驾驶", "强化学习", "安全关键场景", "场景扰动", "威胁引导", "策略感知", "NAVSIM"]
innovations: ["策略感知场景编码器，将冻结策略隐层表征融入场景编码以对齐策略弱点", "目标场景扰动模块，基于重要性评分选择 top‑K 物体进行定向速度/时间/状态修改", "威胁引导优化，利用扰动前后策略 rollout 威胁分差异作为监督信号实现闭环优化"]
benchmarks: ["NAVSIM v2 navhard_two_stage"]
---

# 论文速读：Threat-guided Policy-aware Scene Perturbation for Safe Autonomous Driving with Online Reinforcement Learning

## 一句话总结
论文提出 **TPSP**（威胁引导的策略感知场景扰动）框架，通过策略感知的场景编码器识别当前策略弱点，并生成针对性、高威胁的场景扰动，从而在有限交互预算下显著提升在线强化学习自动驾驶策略的安全学习效率与最终鲁棒性。

## 研究问题与动机
1. **安全关键长尾场景稀缺**：在线强化学习（RL）依赖大量交互数据，但真实世界中危险、罕见的安全关键场景难以通过常规采样充分暴露，导致策略学习能力受限。
2. **现有场景增强方法缺乏策略感知**：优先回放、课程学习、对抗场景生成等方法通常基于环境级目标（如碰撞概率、预定义安全违规）优化场景难度，未显式建模生成扰动与**当前演化策略弱点及学习需求**的关系。
3. **策略无关扰动训练价值有限**：不同策略对同一交通事件（如加塞、急刹）的敏感程度不同，固定或随机扰动无法针对性地强化策略薄弱环节，造成交互数据利用率低。
4. **需要闭环自适应扰动机制**：为提升安全学习效率，场景扰动应动态适应当前策略特性，自动发现并生成与策略弱点相关的高威胁交互场景。

## 核心贡献（创新点）
1. **提出策略感知场景编码器**：融合自车状态、周边物体特征、道路上下文、风险信息及**冻结策略网络的隐层表征**，构建策略感知的场景表示，使扰动生成能对齐当前策略弱点。*与已有工作的本质区别：首次将策略行为表征显式集成到场景编码中，实现扰动与策略弱点的动态对齐。*
2. **设计目标场景扰动模块**：基于重要性评分（Softmax over scoring network）选择 top-K 关键物体，并对其进行定向修改（速度调整、出生时间偏移、自车初始状态），而非全局均匀扰动。*与已有方法的区别：从环境级扰动转向策略相关的目标级扰动，聚焦于对当前策略影响最大的物体。*
3. **开发威胁引导优化策略**：定义综合威胁严重度与时序不稳定性的威胁分，通过比较扰动前后相同策略 rollout 的威胁分差异（$\Delta\xi$）作为监督信号，优化扰动网络以最大化威胁增量。*与同类工作的区别：利用策略 rollout 的威胁增量作为闭环优化信号，而非固定环境指标或预设难度。*
4. **在 NAVSIM v2 上实现显著安全性能提升**：约 400 万英里仿真数据下，Stage 1 达 99.8% NC、99.6% TTC，Stage 2 达 96.7% NC、94.6% TTC，较现有最强方法提升约 2.2% 和 1.8%。*与基线的本质区别：在相同交互预算下，通过策略感知扰动显著提升安全学习效率与最终鲁棒性。*

## 方法详解
TPSP 包含三个核心组件，形成**驾驶策略 $\pi_\theta$** 与 **场景扰动网络 $G_\phi$** 之间的闭环优化流程。

### 1. 策略感知场景编码器
- 提取五类特征：
  - **ego feature**：自车运动状态（速度、航向、目的地）。
  - **object features** $\{e_i^{\text{obj}}\}$：第 $i$ 个周边物体的类型、几何属性、相对位置/速度、Time-to-Collision (TTC)。
  - **road feature**：周围道路结构信息。
  - **risk feature**：场景级风险指标（最近距离倒数、最小 TTC 倒数、场景内物体数量）。
  - **policy feature**：从**冻结策略网络** $\pi_{\bar\theta}$ 的隐层表示 $\mathbf{z}_{\pi_{\bar\theta}}$ 经 MLP 投影得到：$e^{\text{policy}} = \text{MLP}^{\text{policy}}(\mathbf{z}_{\pi_{\bar\theta}})$，冻结策略参数防止梯度回传。
- 通过注意力机制聚合物体交互特征：$c^{\text{obj}} = \text{Attention}(e^{\text{ego}}, \{e_i^{\text{obj}}\})$。
- 最终融合得到策略感知场景表示：$h = \text{MLP}^{\text{fusion}}([e^{\text{ego}}, c^{\text{obj}}, e^{\text{road}}, e^{\text{risk}}, e^{\text{policy}}])$。

### 2. 目标场景扰动
- **物体重要性评分**：$w^{\text{obj}} = \text{Softmax}(\{f^{\text{score}}([e_i^{\text{obj}}, h])\})$，选择 top‑K 物体作为扰动目标 $E_K^{\text{obj}}$。
- **扰动生成**：扰动网络 $G_\phi$ 输出高斯分布的均值 $\mu$ 和标准差 $\sigma$（经指数函数保证正值），采样原始扰动变量 $z^{\text{raw}} \sim \mathcal{N}(\mu, \text{diag}(\sigma^2))$，再经边界变换得到物理可行的扰动向量：$z = \lambda (z^{\text{max}} \odot \tanh(z^{\text{raw}}))$。
- 扰动变量 $z$ 对应三类修改：**物体速度调整、物体出生时间偏移、自车初始状态修改**。

### 3. 威胁引导优化
- **步级威胁分** $c_t$：综合五个安全因素，$c_t = w^{\text{ttc}} c_t^{\text{ttc}} + w^{\text{dist}} c_t^{\text{dist}} + w^{\text{edge}} c_t^{\text{edge}} + w^{\text{col}} c_t^{\text{col}} + w^{\text{off}} c_t^{\text{off}}$，各分量取值 $[0,1]$。
- **rollout 级威胁严重度** $\mathcal{C}(\tau)$：取 rollout 中 top‑k 最高步级威胁分的平均值，聚焦最危险时刻。
- **威胁时序不稳定性** $\mathcal{Y}(\tau)$：$\frac{1}{T}\sum_{t=0}^{T-1} |c_{t+1} - c_t|$，衡量威胁变化的剧烈程度。
- **最终威胁分**：$\xi(\tau) = \mathcal{C}(\tau) \frac{1+\beta\mathcal{Y}(\tau)}{1+\beta}$。
- **威胁差异**：对同一初始状态，用相同冻结策略 $\pi_{\bar\theta}$ 和相同随机种子分别运行原始场景与扰动场景，计算 $\Delta\xi = \xi(\tau^{\text{per}}) - \xi(\tau^{\text{org}})$。
- **扰动网络损失**：归一化威胁优势 $\hat{A}_s = \frac{\Delta\xi_s - \text{mean}(\Delta\xi)}{\text{std}(\Delta\xi)+\epsilon}$，优化目标为：
  $\mathcal{L}^{\text{per}} = -\mathbb{E}_s[\hat{A}_s \log p_\phi(z_s^{\text{raw}}|E_K^{\text{obj}}, h)] - c^{\text{ent}} H_z$，其中 $H_z$ 为扰动分布熵，鼓励探索。
- **训练循环**（Algorithm 1）：每轮迭代从原始数据集 $\mathcal{D}$ 和场景缓冲区 $\mathcal{B}$（存储高威胁扰动场景）混合采样 → 提取策略特征 → 生成扰动 → 用冻结策略执行 rollout → 更新驾驶策略 $\pi_\theta$（PPO）→ 更新扰动网络 $G_\phi$ → 将高威胁扰动场景存入 $\mathcal{B}$ → 同步 $\pi_{\bar\theta} \leftarrow \pi_\theta$。

## 实验与结果
- **数据集与评估基准**：训练使用 NAVSIM v2 `navtrain` 数据集（经转换适配 GPUDrive 模拟器）；评估在官方 **NAVSIM v2 `navhard_two_stage`** 基准上进行，含 Stage 1（原始场景）和 Stage 2（更具挑战的合成场景）。
- **评估指标**：No‑At‑Fault Collisions (NC)、Driveable Area Compliance (DAC)、Time To Collision (TTC)，均为百分比，越高越好。
- **基线方法**：Vanilla PPO、Random Perturbation、TPSP w/o PA（无策略感知）、以及外部对比方法 NavFormer、RAP、ZTRS、SimScale、DrivoR。
- **主要结果**（Table 2）：
  - **Stage 1**：TPSP 达到 **99.8% NC、99.6% TTC**，超越 SimScale（99.5% NC、99.5% TTC）。
  - **Stage 2**：TPSP 达到 **96.7% NC、94.6% TTC**，超越 SimScale（94.5% NC、92.8% TTC）约 **+2.2%、+1.8%**。
  - DAC 指标：Stage 1 为 97.8%，Stage 2 为 93.7%。
- **学习效率**（Figure 3）：在相同交互预算下，TPSP 的安全奖励（碰撞避免、离路 avoidance）提升更快；在 200 万英里时已达 Vanilla PPO 在 400 万英里时的碰撞奖励水平。
- **消融实验**（Table 1）：
  - Random Perturbation 仅带来有限提升，证明单纯增加场景多样性不够。
  - TPSP w/o PA 性能低于完整 TPSP，验证策略感知特征的关键作用。
  - 完整 TPSP 在 Stage 2 取得最高 NC（96.7%）和 TTC（94.6%）。

## 相关工作脉络
1. **对抗场景生成方法**（AdvSim, KING, SafeBench, CAT）：优化环境级目标（碰撞概率、 adversary 行为）生成危险场景，但未显式建模与**当前策略弱点**的关联，易生成策略无关的无效扰动。
2. **优先回放与课程学习**（Prioritized Replay, Curriculum Learning）：侧重于经验选择或难度调度，未引入策略表征进行场景扰动生成，缺乏针对策略薄弱环节的定向强化。
3. **策略感知强化学习**（Self‑play, CaRL, Adaptive Curriculum）：关注策略优化或经验安排，而非**安全关键场景的主动构造**，未形成“策略→扰动→策略”的闭环。
4. **数据驱动端到端自动驾驶**（NavFormer, RAP, ZTRS, SimScale, DrivoR）：多采用离线学习范式，直接输入原始传感器观测（如多视角图像），而 TPSP 为**在线 RL 框架**，使用结构化白盒信息（状态、物体、道路）并通过交互优化策略。
5. **扩散模型场景生成**（Diff‑Scene 等）：基于扩散过程生成安全关键场景，计算成本高且缺乏与在线策略演化的实时对齐机制。

## 局限性与未来方向
- **扰动空间有限**：当前仅支持场景级修改（物体速度、出生时间、自车初始状态），未涵盖更丰富的语义、地图级别或多智能体交互扰动。
- **仿真到现实差距**：实验基于 GPUDrive 模拟器，在真实驾驶场景中的泛化能力有待验证。
- **训练依赖专用模拟器**：需要将 NAVSIM 场景转换为 GPUDrive 兼容格式，可能损失部分原始数据细节。
- **未来方向**：扩展扰动空间（语义、地图、多智能体交互）；在更大规模仿真环境和真实世界设置中验证；探索与更多 RL 算法及自动驾驶策略架构的集成。

## 研究启发与可借鉴点
1. **策略感知表示的提取方式**：通过冻结策略网络隐层并经 MLP 投影得到策略特征，可作为通用模块嵌入各类对抗性场景生成系统，实现扰动与策略弱点的动态对齐。
2. **威胁分的组合设计**：同时考虑威胁严重度（top‑k 平均）与时序不稳定性（差分幅度），提供更全面的场景难度评估，可迁移至需要衡量序列风险的其他任务（如机器人控制、无人机避障）。
3. **闭环优化机制**：扰动网络与策略网络交替更新，利用策略 rollout 的威胁增量（$\Delta\xi$）作为监督信号，形成自我强化的训练循环，适用于交互数据昂贵的在线学习场景。
4. **实验设计借鉴**：固定交互预算对比安全奖励曲线，直观展示学习效率提升；通过 Stage 1/Stage 2 分层评估验证策略在未知挑战场景中的泛化能力。
5. **缓冲区管理策略**：维护一个场景缓冲区存储历史高威胁扰动场景，并动态调整原始与扰动场景的采样比例，可实现训练难度的渐进式提升。

## 关键术语表
- **TPSP**：Threat‑guided Policy‑aware Scene Perturbation，威胁引导的策略感知场景扰动框架。
- **策略感知场景编码器**：融合自车状态、物体特征、道路上下文、风险信息及冻结策略隐层特征的编码模块，输出策略感知的场景表示 $h$。
- **目标场景扰动**：基于重要性评分选择 top‑K 关键物体，并对其施加速度、出生时间或自车初始状态的定向修改，而非全局均匀扰动。
- **威胁分**：综合时间碰撞、距离、路边接近、碰撞和离路等因素的步级风险度量，取值 $[0,1]$。
- **威胁不稳定性**：衡量 roll‑out 中威胁分时序变化剧烈程度的指标，反映安全条件的快速演变。
- **威胁优势**：扰动前后威胁分差异的归一化版本，用于指导扰动网络的策略梯度更新。
- **NAVSIM v2**：包含 `navtrain` 训练集与 `navhard_two_stage` 评测基准的自动驾驶数据集，提供录制驾驶场景用于 benchmark 评估。
- **GPUDrive**：支持大规模并行仿真（1M FPS）的驾驶模拟器平台，为 TPSP 的在线 RL 训练提供交互环境。

## 可复现要素
- **数据集**：NAVSIM v2 `navtrain`（训练）、`navhard_two_stage`（评估），后者为公开基准；训练前需将场景转换至 GPUDrive 兼容格式（论文未提供转换脚本）。
- **代码/权重**：论文未提及代码与权重开源情况。
- **关键超参**：
  - 扰动网络：编码器隐藏维度 128，融合维度 256，策略特征维度 64，选择物体数 $K=8$，初始对数标准差 $-1.0$，学习率 $1\times10^{-4}$，熵系数 0.01，梯度裁剪 0.5，更新间隔 4 rollout，初始扰动尺度 $\lambda=0.2$。
  - PPO：学习率 $3\times10^{-4}$，clip ratio 0.2，epoch 2，折扣因子 $\gamma=0.99$，GAE 系数 0.95，experience batch size 204800，mini‑batch size 6400，熵系数 0.01，梯度裁剪 0.5。
  - 硬件：16 × NVIDIA A100 GPU，512 并行环境，PyTorch 框架。

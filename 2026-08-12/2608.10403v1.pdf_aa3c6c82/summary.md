---
title: "Threat-guided Policy-aware Scene Perturbation for Safe Autonomous Driving with Online Reinforcement Learning"
source: https://arxiv.org/pdf/2608.10403v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:31:56"
field: "自动驾驶安全强化学习"
keywords: ["在线强化学习", "自动驾驶安全", "场景扰动", "策略感知", "威胁引导", "长尾场景", "NAVSIM v2", "GPUDrive"]
innovations: ["提出策略感知场景编码器，将冻结策略隐层表示融入场景联合特征以对齐扰动生成", "设计定向扰动模块，基于物体重要性评分选取 top-K 关键对象进行速度/时间/状态修改", "引入威胁差异优化信号，通过扰动前后同策略 rollout 威胁分数变化驱动扰动网络迭代"]
benchmarks: ["NAVSIM v2 navhard_two_stage", "GPUDrive simulation"]
---

# 论文速读：Threat-guided Policy-aware Scene Perturbation for Safe Autonomous Driving with Online Reinforcement Learning

## 一句话总结
论文提出 **TPSP**（Threat-guided Policy-aware Scene Perturbation）框架，通过在线强化学习训练自动驾驶策略时，生成与当前策略弱点相匹配的**高威胁安全关键场景**，显著提升有限交互预算下的碰撞避免与交互安全性。

## 研究问题与动机
1. **在线 RL 面临长尾危险场景稀缺**：真实交通中高风险交互（如碰撞）发生频率低，常规采样难以充分覆盖，导致策略在安全关键行为上学习不足。
2. **现有场景增强方法与策略脱节**：已有方法（如 AdvSim、SafeBench）主要从环境级目标（碰撞概率、对抗目标）出发构造困难场景，未显式建模扰动与当前策略弱点及学习需求之间的关系；同一场景对成熟策略可能 trivial，对不成熟策略却极具挑战。
3. **均匀扰动效率低下**：对整个场景施加均匀修改无法针对策略薄弱环节提供高价值训练信号，资源浪费且提升有限。
4. **缺乏闭环策略-场景协同优化**：场景生成与策略更新通常独立进行，缺少以策略反馈为信号指导场景扰动优化的机制。

## 核心贡献（创新点）
1. **提出策略感知的场景编码器**：融合 ego 状态、周围物体特征、道路上下文、风险指标以及**冻结策略网络的隐层表示**，构建同时包含场景物理交互信息与策略行为特征的联合表征，使后续扰动生成能够对齐当前策略。
2. **设计针对关键物体的定向扰动**：基于物体重要性评分选出 top-K 关键对象，仅对它们施加速度调整、生成时间偏移与 ego 初始状态修改，而非全场景均匀扰动，提升扰动的针对性与训练效率。
3. **引入威胁差异作为场景扰动优化信号**：通过同策略在原始与扰动场景上的 rollout 计算威胁分数差异（Δξ），驱动场景扰动网络 $G_\phi$ 最大化该差异，实现以策略学习需求为导向的高价值安全关键场景生成。
4. **建立策略与扰动网络的闭环迭代优化**：在每一训练轮次中先用扰动场景收集 rollout 更新驱动策略，再根据威胁差异更新扰动网络，形成动态适应策略弱点的自我进化循环。
5. **在 NAVSIM v2 基准上取得领先安全性能**：约 400 万车公里的模拟数据下，Stage 2 场景达到 96.7% NC 与 94.6% TTC，较之前最优方法分别提升 2.2% 与 1.8%，验证了方法的有效性与数据效率。

## 方法详解
TPSP 框架包含三个核心组件：

1. **策略感知场景编码器（Policy-aware Scene Encoder）**
   - 提取五类特征：ego 特征 $e^{\text{ego}}$（速度、航向、目的地）、周围物体特征 $e_i^{\text{obj}}$（类型、几何、相对位置/速度、TTC）、道路特征 $e^{\text{road}}$、场景风险特征 $e^{\text{risk}}$（最小距离倒数、最小 TTC 倒数、周围物体数量）、策略特征 $e^{\text{policy}}$（由冻结策略网络 $\pi_{\bar{\theta}}$ 的隐层经 MLP 投影得到）。
   - 使用注意力机制聚合物体交互信息：$c^{\text{obj}} = \text{Attention}(e^{\text{ego}}, \{e_i^{\text{obj}}\}_{i=1}^N)$。
   - 通过 MLP fusion 融合所有特征得到最终表示 $\boldsymbol{h}$，该表示同时编码场景物理状态与当前策略行为特性。

2. **定向场景扰动模块（Targeted Scene Perturbation）**
   - **物体选择**：对每个物体计算重要性 logit $f^{\text{score}}([e_i^{\text{obj}}, \boldsymbol{h}])$，经 Softmax 归一化后选取 top-K 个关键物体作为扰动目标。
   - **扰动生成**：将选定物体表示与 $\boldsymbol{h}$ 输入扰动网络 $G_\phi$，预测高斯分布的均值 $\boldsymbol{\mu}$ 与标准差 $\boldsymbol{\sigma}$，采样原始扰动变量 $z^{\text{raw}}$。
   - **边界转换**：通过 $z = \lambda (\boldsymbol{z}^{\max} \odot \tanh(z^{\text{raw}}))$ 将无约束采样转化为物理可行的扰动变量，涵盖物体速度调整、生成时间偏移与 ego 初始状态修改三种扰动类型；$\lambda$ 为训练期动态调整的缩放因子。

3. **威胁引导的场景扰动优化（Threat-guided Optimization）**
   - **步级威胁分数**：综合 TTC、车间距、道路边缘接近度、碰撞发生与离线道路五种安全因素加权求和：$c_t = w^{\text{ttc}}c_t^{\text{tts}} + w^{\text{dist}}c_t^{\text{dist}} + w^{\text{edge}}c_t^{\text{edge}} + w^{\text{col}}c_t^{\text{col}} + w^{\text{off}}c_t^{\text{off}}$。
   - **rollout 级关键性**：选取 top-k 个最高威胁时间点求平均：$\mathcal{C}(\tau) = \frac{1}{k'}\sum_{t \in \text{Top}(\tau,k)} c_t$。
   - **威胁不稳定性**：衡量威胁随时间的变化幅度：$\mathcal{V}(\tau) = \frac{1}{T}\sum_{t=0}^{T-1}|c_{t+1}-c_t|$。
   - **最终威胁分数**：$\xi(\tau) = \mathcal{C}(\tau)\frac{1+\beta\mathcal{V}(\tau)}{1+\beta}$，$\beta$ 控制不稳定性贡献。
   - **扰动网络损失**：计算扰动与原始 rollout 的威胁差异 $\Delta\xi = \xi(\tau^{\text{per}}) - \xi(\tau^{\text{org}})$，经 batch 内标准化得到优势 $\hat{A}_s$，优化目标为：$\mathcal{L}^{\text{per}} = -\mathbb{E}_s[\hat{A}_s \log p_\phi(z_s^{\text{raw}}|E_K^{\text{obj}},\boldsymbol{h})] - c^{\text{ent}}H_z$，其中熵正则项防止过早收敛。
   - **闭环流程**：每轮迭代混合原始数据集 $\mathcal{D}$ 与高威胁扰动场景缓冲区 $\mathcal{B}$ 采样场景，提取策略特征、生成扰动、收集 rollout 更新策略 $\pi_\theta$；定期更新扰动网络 $G_\phi$ 并将高威胁扰动场景存入 $\mathcal{B}$，同时同步冻结策略 $\pi_{\bar{\theta}}$。

## 实验与结果
- **数据集与仿真平台**：训练使用 GPUDrive 模拟器与 NAVSIM v2 navtrain 数据集；评估在 NAVSIM v2 navhard_two_stage 基准上进行。
- **评估指标**：无责碰撞率（NC↑）、可行驶区域合规率（DAC↑）、碰撞时间（TTC↑）；Stage 1 为原始场景评估，Stage 2 为更具挑战性的合成场景评估。
- **主要基线**：Vanilla PPO、Random Perturbation、以及 NAVSIM v2 排行榜上的 NavFormer、RAP、ZTRS、SimScale、DrivoR。
- **最强结果**：TPSP 在 Stage 2 获得 **96.7% NC** 与 **94.6% TTC**，较上一最优方法 SimScale 分别提升 **2.2%** 与 **1.8%**；Stage 1 达到 99.8% NC 与 99.6% TTC。
- **数据效率**：在约 400 万车公里模拟里程内，TPSP 在 2M 英里时已达到 Vanilla PPO 在 4M 英里时的碰撞避免奖励水平，证明其能更高效地利用有限交互预算。
- **消融验证**：随机扰动仅带来有限提升；去除策略感知模块（TPSP w/o PA）在 Stage 2 显著下降至 91.1% NC 与 89.7% TTC，表明策略感知与定向扰动不可或缺。
- **定性分析**：可视化显示 TPSP 生成的扰动在切入场景（cut-in）中增加威胁（Δξ=+0.1115），在跟车场景降低威胁（Δξ=-0.0540），符合直觉且说明扰动网络能辨别不同场景的风险变化。

## 相关工作脉络
1. **AdvSim / KING**：通过优化周围智能体行为构造安全关键场景，但未考虑 evolving 策略的特定弱点；TPSP 以策略特征为指导进行定向扰动。
2. **SafeBench / CAT**：提供评估框架或闭环对抗训练，但场景难度优化独立于策略学习；TPSP 建立策略与扰动网络的闭环协同优化。
3. **CaRL / 自适应课程学习**：根据策略能力调整训练体验调度，但聚焦于规划优化而非安全关键场景构造；TPSP 专门针对碰撞等高风险交互生成高价值训练样本。
4. **Diff-Scene / FREA**：基于扩散模型或可行性引导生成对抗场景，属于离线或环境级目标优化；TPSP 采用在线 RL 框架，以策略 rollout 威胁差异为信号动态调整扰动。
5. **RAD / Raw2Drive**：探索 3DGS 仿真或世界模型对齐的 RL 训练范式；TPSP 侧重于在结构化白盒信息基础上通过场景扰动提升策略鲁棒性，两者可互补。
6. **GPUDrive / MetaDrive**：提供大规模并行仿真平台；TPSP 直接构建于 GPUDrive 之上，利用其高效场景生成能力进行在线策略学习。

## 局限性与未来方向
1. **扰动空间相对有限**：当前仅考虑物体速度、生成时间与 ego 初始状态修改，未涵盖更丰富的语义、地图级或多智能体交互扰动。
2. **依赖结构化白盒信息**：需要模拟器提供 ego 状态、周围物体状态、道路信息等结构化特征，难以直接迁移至纯视觉传感器输入场景。
3. **模拟到现实鸿沟**：实验均在 GPUDrive 仿真环境中验证，实际部署时需解决 sim-to-real 泛化问题。
4. **未来可扩展方向**：论文提及将探索更复杂的扰动空间、更大规模仿真环境、真实驾驶场景，以及与更多样化 RL 算法和策略架构的结合。

## 研究启发与可借鉴点
1. **策略特征作为扰动生成条件**：从冻结策略网络提取隐层表示并投影为辅助特征，使场景生成模块能够感知当前策略行为特性，这一设计可迁移至其他需要“策略适配”的数据增强任务。
2. **威胁差异作为生成信号**：通过比较扰动前后同一策略的 rollout 威胁分数差异来指导生成网络优化，避免依赖预设规则或环境级目标，该思路可应用于对抗样本生成、鲁棒性训练等领域。
3. **定向扰动而非均匀扰动**：先对物体进行重要性排序再针对性修改，能显著提升训练效率；类似思想可用于机器人控制、游戏 AI 等需要高效探索的场景。
4. **缓冲区存储高价值样本**：维护一个场景缓冲区保留已发现的高威胁扰动场景，并动态调整原始与扰动场景的采样比例，实现训练过程中难度的渐进提升，可借鉴于 curriculum learning 设计。
5. **安全相关奖励与威胁度量解耦**：策略优化使用标准安全奖励（碰撞惩罚、离路惩罚），而扰动网络优化使用独立计算的威胁分数，两者分工明确且互不干扰，这种模块化设计便于替换不同 RL 算法或安全度量标准。

## 关键术语表
- **TPSP**：Threat-guided Policy-aware Scene Perturbation，一种面向在线强化学习自动驾驶的安全场景扰动框架。
- **Policy-aware scene representation**：融合 ego 状态、周围环境、道路信息、风险指标及当前策略隐层表示的联合特征向量。
- **Targeted perturbation**：仅对关键物体（top-K）施加速度、时间或状态修改，而非全场景均匀扰动。
- **Threat score**：综合 TTC、距离、边缘接近、碰撞与离路事件的加权步级风险度量。
- **Threat difference (Δξ)**：同策略在扰动与原始 rollout 上的威胁分数之差，作为扰动网络优化的核心信号。
- **GPUDrive**：支持百万帧每秒并行仿真的高速多智能体驾驶仿真平台。
- **NAVSIM v2 navhard_two_stage**：NAVSIM v2 基准中的两阶段安全评估协议，Stage 2 使用更具挑战性的合成场景。
- **NC / DAC / TTC**：No-Collision 无责碰撞率、Drivable Area Compliance 可行驶区域合规率、Time-to-Collision 碰撞时间，三项核心安全指标。

## 可复现要素
- **数据集**：NAVSIM v2 navtrain（训练）、navhard_two_stage（评估），源自 CAMMA Labs 公开基准。
- **代码/权重**：论文未提及代码开源状态与模型权重提供情况。
- **关键超参**：
  - 策略网络：MLP actor，隐藏维度 64，潜层维度 64，tanh 激活，离散动作空间（转向 13 档，加速 7 档）。
  - 扰动网络：编码器隐藏维度 128，融合维度 256，策略特征维度 64，选择物体数 K=8，初始 log 标准差 -1.0，Adam 优化器，学习率 1×10⁻⁴，熵系数 0.01，梯度裁剪 0.5，每 4 轮 rollout 更新一次。
  - PPO：AdamW，学习率 3×10⁻³，clip ratio 0.2，2 epochs，γ=0.99，GAE λ=0.95，batch size 204800，mini-batch 6400，价值损失系数 0.5，熵系数 0.01，梯度裁剪 0.5。
  - 初始扰动缩放因子 λ=0.2。
- **硬件**：16 张 NVIDIA A100 GPU，1024GB 内存，128 核 CPU，512 个并行仿真环境。

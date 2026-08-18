---
title: "CRITIC-FREE PRETRAINING FOR EFFICIENT ONLINE REINFORCEMENT LEARNING FINE-TUNING"
source: https://arxiv.org/pdf/2608.10473v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:29:30"
field: "离线到在线强化学习"
keywords: ["offline-to-online reinforcement learning", "critic-free pretraining", "flow matching policy", "behavior cloning", "online fine-tuning", "distribution shift", "actor-critic decoupling"]
innovations: ["完全摒弃离线 Critic 训练，仅用 BC 预训练 Actor 以避免分布偏移导致的悲观值偏差", "从零初始化 Critic 并进行极短时离线数据 Warm-up 完成校准，无需额外在线预训练阶段"]
benchmarks: ["OGBench (Cube-Double/Triple/Quadruple, Puzzle-4x4, Scene)", "Robomimic (Lift, Can, Square)"]
---

# 论文速读：CRITIC-FREE PRETRAINING FOR EFFICIENT ONLINE REINFORCEMENT LEARNING FINE-TUNING

## 一句话总结
本文提出 **Critics-Free Pretraining（CFP）**，一种离线到在线（O2O）强化学习的新范式：完全摒弃离线阶段的 Critic 训练，仅用行为克隆（BC）预训练 Actor，在线微调前从零初始化 Critic 并进行短时 Warm-up，从而避免离线 Critic 的悲观值偏差对在线策略优化的干扰。

## 研究问题与动机
1. **离线 Critic 的分布偏移问题**：O2O 范式中直接复用离线训练好的 Critic，但其价值估计基于静态数据集分布，随着在线策略快速演化，Critic 的值估计会与实际在线环境失配，导致策略更新不准确。
2. **Critic 悲观性（Pessimism）的隐性累积**：现代离线数据集以次优/失败轨迹为主，TD 学习的自举机制使得这些低回报从数据中向后传播，使 Critic 产生系统性悲观偏差，阻碍有效探索。
3. **Critic 与 Actor 非对称角色被忽视**：离线阶段的目标应是让 Actor 学会在有效轨迹区域内采样以提升采样效率，而非同时训练一个可能有害的 Critic；当前方法将两者捆绑训练缺乏理论依据。
4. **计算与内存开销可进一步压缩**：消除离线 Critic 训练可显著降低计算成本和内存占用，对大规模 RL 系统的扩展有益。

## 核心贡献（创新点）
1. **提出 Critic-Free 离线训练范式**：完全取消离线阶段的 Critic 训练，仅用 BC 预训练 Actor；与已有 O2O 方法（如 CQL、Cal-QL、WSRL）的本质区别在于不依赖保守正则或 replay 设计来"矫正"Critic，而是从根本上避免引入带偏的 Critic。
2. **设计简洁有效的 Warm-up 校准机制**：在线微调开始前，从零初始化 Critic 并在离线数据上进行短时（10K 步，约 30 秒）Warm-up；与 OPT（需额外在线预训练阶段）、WSRL（仅用在线数据校准）等方法相比，CFP 的 Warm-up 完全基于现有离线数据集，无需额外在线数据收集。
3. **提供可解释的理论与实证分析**：通过 Toy MDP 实验可视化证明 O2O 的 Critic Q 值结构与真实在线 Q 函数存在显著失配，而 CFP 的 Fresh Critic 在线上微调期间 RMSE 更低、收敛更快。
4. **广泛的算法兼容性与性能提升**：CFP 可无缝接入 FQL、QC、QCFQL、QCFQL-nstep 四种主流 Flow-based O2O 算法，在 OGBench 多个挑战性任务上实现显著提升（如 Cube Triple 上 QCFQL-nstep-CFP 从 0% 提升至 98.4%）。

## 方法详解
**CFP 整体流程分为三阶段：**

1. **离线阶段（Critic-Free）**：仅在静态数据集 $\mathcal{D}$ 上对 Flow-based Actor 进行行为克隆训练，目标函数为：
   $$L_\pi(\omega) = \alpha \mathbb{E}_{s \sim \mathcal{D}, z \sim \mathcal{N}(0, I^d)}[\|\mu_\omega(s, z) - \mu_\theta(s, z)\|_2^2]$$
   无 Critic 更新，Actor 梯度不来自价值网络。

2. **Warm-up 阶段**：在线微调开始前，从零初始化 Critic $Q_\phi$，在离线数据集上同时进行 Actor 和 Critic 的短时联合训练（10K 步）。此时 Actor 仍保留 BC 项，Critic 通过 TD loss 学习初步的 Q 值估计：
   $$L_Q(\phi) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}[(Q_\phi(s,a) - [r + \gamma Q_{\bar{\phi}}(s',a')])^2]$$
   关键发现：即使 Warm-up 极短（仅占总步数 0.5%），Critic 能快速收敛并为 Actor 提供有效梯度。

3. **在线微调阶段**：在真实环境中收集轨迹，使用标准在线 Actor-Critic 更新（TD loss + Actor 损失）。Online 与 Warm-up 阶段的更新目标相同，仅数据来源不同（Online 含在线样本）。

**核心洞察**：Fresh Critic 不继承离线分布导致的悲观偏差，虽初始不准确但不会持续误导 Actor；短时 Warm-up 使其获得初步校准，同时 Actor 的 BC 先验保障了稳定的数据采样。

## 实验与结果
- **数据集**：OGBench（5 个稀疏奖励操作域：Cube-Double/Triple/Quadruple、Puzzle-4×4、Scene）+ Robomimic（3 个域：Lift、Can、Square）；使用默认 Multi-Human 数据集，Cube Quadruple 使用 100M 大规模数据集。
- **评估基线**：4 种 Flow-based O2O 算法（FQL、QC、QCFQL、QCFQL-nstep）的原始版本 vs. 各算法的 CFP 版本。
- **关键结果**：
  - **Cube Triple**：所有 CFP 变体均显著优于对应 O2O 基线；QCFQL-nstep-CFP 最终成功率达 **98.4%**（vs. O2O 基线 0%~4%）。
  - **Puzzle 4×4**：FQL-CFP 达到约 **0.7** 最终成功率（vs. FQL-O2O < 0.4），且超越所有 QCFQL/QC 变体。
  - **Cube Double/Quadruple/Scene**：CFP 与基线表现相当或更优。
  - **Robomimic**：CFP 在大多数任务上与基线持平；**Square 任务**上 QCFQL-nstep-CFP 实现显著提升（82.4% vs. 73.2%）。
- **最强结果**：QCFQL-nstep-CFP 在 Cube Triple Task 4 实现 **91.6%** 成功率（O2O 基线为 75.6%），提升 **16个百分点**。
- **消融结论**：
  - Warm-up 期间引入 Q-loss 对 Actor 更新有益（图6）；
  - CFP 对 Warm-up 步数不敏感（10K~50K 均可），但 **0 步 Warm-up 会导致训练不稳定**；10K 步仅需约 30 秒（A800 GPU）。

## 相关工作脉络
1. **保守/悲观离线 RL（CQL、Cal-QL）**：通过保守正则化抑制 OOD 动作的价值过估计；CFP 选择完全不同的策略——不修正 Critic，而是替换它。
2. **行为克隆增强方法（BCQ、TD3+BC、AWAC）**：在 Actor 损失中显式加入 BC 项以约束策略不偏离数据集；CFP 在离线阶段**完全等价于纯 BC**，但去除了 Critic 训练。
3. **Replay 设计（RLPD）**：混合离线数据和在线经验训练离线 Actor-Critic；CFP 同样使用混合 replay，但不保留离线 Critic 参数。
4. **Critic 校准方法（WSRL、OPT）**：WSRL 用纯在线数据校准预训练 Critic；OPT 引入新 Critic 并在线预训练；CFP 的创新在于直接丢弃离线 Critic、从零初始化并用极短 Offline-data Warm-up 完成校准，无需额外在线数据阶段。
5. **Flow-based 策略（FQL、QCFQL 等）**：CFP 作为通用范式，可直接套用于上述多种 Flow 架构，体现了方法论层面的解耦思想。

## 局限性与未来方向
1. **在 Robomimic 上未持续超越 O2O**：CFP 并非在所有任务上都优于传统 O2O，其收益依赖于离线 Critic 与在线分布的失配程度。
2. **缺乏自动诊断机制**：目前无法预测何时离线 Critic 的分布偏移会导致显著性能下降，从而决定是否需要启用 CFP。
3. **Warm-up 策略仍有优化空间**：当前仅尝试了固定步数的 Warm-up，尚未探索自适应校准或更精细的 Critic 初始化策略。
4. **Toy MDP 的简化性**：理论分析使用的 16 状态 MDP 过于简单，复杂高维环境下 Fresh Critic 的动态可能有所不同。

## 研究启发与可借鉴点
1. **Actor-Critic 解耦训练的普适性**：将 Actor 预训练（负责采样效率）与 Critic 初始化（负责价值估计）解耦的思路，可迁移到其他 O2O 算法（如 SAC、SD3 等）及更大规模的视觉-动作任务中。
2. **"丢弃而非修正"的创新视角**：面对分布偏移问题，常规思路是添加正则或校准项，CFP 展示了"干脆不用"的简约方案，对其它存在类似偏置的场景（如离线生成模型微调）具有启发意义。
3. **超参数不敏感性的实验设计**：通过系统 sweep Warm-up 步数并观察到稳健性，为后续工作提供了"简单方案往往足够"的实证证据，值得在类似研究中复现验证。
4. **与团队方向结合机会**：若团队涉及大型机器人操作或具身智能的在线微调，CFP 可无缝集成至现有 Flow-based 框架，在 Cube Triple 等挑战任务上可能有显著收益。

## 关键术语表
**Offline-to-Online (O2O) RL**：利用静态离线数据集预训练策略，再通过与环境交互进行在线微调的两阶段强化学习范式。
**Critic-Free Pretraining (CFP)**：本文提出的方法，离线阶段仅训练 Actor（行为克隆），不训练 Critic；在线阶段从零初始化 Critic 并进行短时 Warm-up。
**Flow Matching Policy**：基于流匹配（连续归一化流）的连续动作策略，通过求解 ODE 从噪声分布生成动作。
**Behavior Cloning (BC)**：从专家演示数据中直接学习策略的行为克隆方法，CFP 离线阶段即采用此目标。
**Pessimism / Conservative Q-learning**：离线数据中由于 TD 自举对次优轨迹的价值传播导致的 Critic 系统性低估现象。
**Warm-up Stage**：在线微调前对 Fresh Critic 进行短时训练的阶段，使 Critic 获得初步 Q 值估计能力而不继承离线偏置。
**Best-of-N Sampling (QC)**：从 N 个候选动作中选择 Q 值最大者作为策略输出的推理方式，不反向传播 Critic 梯度。
**Action Chunking (QCFQL)**：将连续 h 步动作视为一个 chunk 联合生成和评估，适用于长 horizon 任务。

## 可复现要素
- **数据集**：OGBench 和 Robomimic 均为公开基准；OGBench 使用官方 Standard Play 数据集（Cube Quadruple 使用 100M 大规模版本）；Robomimic 使用 Multi-Human 数据集。**论文未明确声明代码开源**。
- **代码/权重**：论文未提及代码仓库链接或权重开源信息，写有"论文未提及"。
- **关键超参**：Batch Size=256，Discount Factor γ=0.99，Optimizer=Adam，Learning Rate=$3\times10^{-4}$，Target Network Update τ=$5\times10^{-3}$，Critic Ensemble Size=2，Flow Integration Steps=4/10，Action Chunk Length=5，Offline Training Steps=$1\times10^6$，Online Steps=$1\times10^6$，Network Width=512，Depth=4，Warm-up Steps=10K（默认）。

---
title: "CRITIC-FREE PRETRAINING FOR EFFICIENT ONLINE REINFORCEMENT LEARNING FINE-TUNING"
source: https://arxiv.org/pdf/2608.10473v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:34:16"
---

# 论文速读：CRITIC-FREE PRETRAINING FOR EFFICIENT ONLINE REINFORCEMENT LEARNING FINE-TUNING

## 一句话总结
本文提出 **Critic-Free Pretraining (CFP)** 范式，在离线阶段完全放弃 Critic 训练、仅通过行为克隆（BC）预训练 Actor；在线微调初期从零初始化新鲜 Critic 并执行短暂 Warm-up 校准，从而避免继承离线数据集的悲观值偏差，显著提升离线到在线（O2O）强化学习的微调效率与最终性能。

## 研究问题与动机
- **传统 O2O 的 Critic 复用困境**：现有方法在离线阶段同步训练 Actor 与 Critic，但直接复用离线训练好的 Critic 会与快速演变的在线策略及数据分布产生失配，导致价值估计偏差、策略更新低效甚至性能退化。
- **离线数据的悲观偏差累积**：现代离线数据集以次优/随机/失败轨迹为主，TD 自举学习会使低回报反向传播并累积系统性悲观性（$Q_{\beta} \leq Q^{*}$），该偏差在在线阶段难以通过常规微调消除。
- **计算与内存冗余**：离线阶段训练 Critic 消耗额外资源，但对后续在线阶段的实际增益有限，尤其在大规模强化学习系统中造成不必要开销。
- **角色解耦洞察**：离线阶段的核心目标应是为 Actor 提供高效的采样先验（使其进入易收集有效轨迹的区域），而非训练泛化价值网络；因此 Actor 预训练与 Critic 预训练在目标上应当解耦。

## 核心贡献（创新点）
- **提出 Critic-Free 离线训练范式**：证明离线阶段训练 Critic 反而损害在线微调性能，通过完全移除离线 Critic 训练大幅降低计算成本与内存需求。
- **设计简洁高效的 CFP 实现流程**：明确给出“保留预训练 Actor + 初始化新鲜 Critic + 短暂 Warm-up 校准”的执行方案，平衡计算效率与实验性能，并提供玩具 MDP 可视化直观验证有效性。
- **验证广泛兼容性与显著性能提升**：CFP 可无缝嵌入多种主流 O2O 算法（FQL, QC, QCFQL, QCFQL-nstep），在多样化稀疏奖励任务上保持一致或更优表现，尤其在 Cube Triple 等高难度任务上优势突出。

## 方法详解
- **离线阶段（Critic-Free Pretraining）**：仅更新 Actor，采用基于 Flow Matching 的行为克隆损失：
  $L_{\pi}(\omega) = \alpha \mathbb{E}_{s \sim \mathcal{D}, z \sim \mathcal{N}(0, I^d)} [\|\mu_{\omega}(s, z) - \mu_{\theta}(s, z)\|_2^2]$
  此阶段完全不更新 Critic，Actor 仅通过模仿数据集分布获得高效采样先验，避免被离线 Critic 的悲观梯度误导。
- **Warm-up 阶段**：在线交互开始前，从头初始化新鲜 Critic $Q_{\phi}$，在离线数据集 $\mathcal{B}$ 上进行少量步数（默认 10K）的联合更新。Actor 损失同时保留 BC 项与 Q-loss（$-\mathbb{E}[Q_{\phi}(s,a)]$），Critic 通过标准 TD 损失更新。该阶段让新鲜 Critic 快速获得初步 Q 估计能力，同时借助 Q-loss 的正向反馈引导 Actor 输出更高价值动作，打破数据集中的悲观循环。
- **在线微调阶段**：从真实环境收集轨迹存入 Buffer，随后同步更新 Actor 与 Critic。由于 Critic 未继承离线分布的系统性偏差，其 Q 值能随在线微调快速向真实 Online Q-function 收敛，避免了传统 O2O 中 Q 均值长期停滞于低水平的偏置问题。
- **理论机制**：Fresh critic 虽初始不准确，但缺乏隐性悲观先验；配合短暂 Warm-up 可实现快速校准。消融实验表明，若 Warm-up 期间剔除 Q-loss，Actor 将持续输出数据集低质动作，导致 Critic Q 估计随步数增加而进一步下降，验证了 Q-loss 在打破悲观闭环中的关键作用。

## 实验与结果
- **数据集与基准**：OGBench（Cube-Double/Triple/Quadruple, Puzzle-4×4, Scene，5 个稀疏奖励域，其中 Cube Quadruple 使用官方 100M 规模数据集）与 Robomimic（Lift, Can, Square，3 个机器人操作任务，使用 Multi-Human 数据集）。
- **评估基线**：4 种代表性 Flow-based O2O 算法：FQL, QC, QCFQL, QCFQL-nstep。
- **主要结果**：
  - CFP 变体在全部 OGBench 任务上保持一致或提升性能。在 **Cube Triple** 上提升最为显著，所有 CFP 变体均优于对应 O2O 基线，QCFQL-nstep-CFP 最终成功率接近 100%。
  - 在 **Puzzle 4×4** 上，FQL-CFP 达到约 0.7 的最终成功率，而 FQL-O2O 不足 0.4；FQL-CFP 在该域甚至超越所有 QCFQL/QC 基线，说明 CFP 对需要组合操作的场景互补性极强。
  - **Robomimic** 上多数任务性能相当，例外为 **Square** 任务中 QCFQL-nstep-CFP 取得显著改进（最高成功率提升明显）。
- **消融实验**：
  - Warm-up 期间加入 Q-loss 普遍提升后续在线微调表现，尤其在 Cube Triple Task 4 与 Cube Double Task 4 上增益清晰。
  - CFP 对 Warm-up 步数不敏感（ Sweep 0~0.1M 步），选择 **10K 步** 可在稳定性与效率间取得最佳平衡（NVIDIA A800 上仅需约 30 秒）。
  - 若 Warm-up 不使用 Q-loss，更长 Warm-up 反而导致性能下滑；归因于 Actor 持续被困于低质数据集动作，Critic Q 估计随之单调下降。
- **玩具 MDP 验证**：在 16 状态表格型 MDP 中，CFP 的 Critic 在整个在线微调过程中与真实 Online Q-function 的 RMSE 始终显著低于 O2O，直观证实了“重新初始化避免偏置继承”的核心假设。

## 相关工作脉络
- **保守性/悲观性算法（CQL, Cal-QL, BCQ, TD3+BC, AWAC）**：通过正则化或显式约束缓解分布外动作的价值过估计；本文反其道而行，直接摒弃离线 Critic 训练从源头规避悲观偏差，无需额外保守正则项。
- **Replay 设计与 Critic 校准（RLPD, WSRL, OPT）**：RLPD 混合离线/在线经验；WSRL 仅用在线数据校准预训练 Critic；OPT 引入新 Critic 并加权融合离线/在线预训练结果。CFP 与之相比更彻底——完全跳过离线 Critic 训练，仅依赖极短 Warm-up 完成校准，实现更轻量。
- **Flow-based Policy 算法（FQL, QC, QCFQL, QCFQL-nstep）**：本文聚焦将 CFP 范式无缝嵌入各类 Flow-based O2O 算法，验证了 Actor-Critic 解耦在

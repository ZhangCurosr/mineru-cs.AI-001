---
title: "CRITIC-FREE PRETRAINING FOR EFFICIENT ONLINE REINFORCEMENT LEARNING FINE-TUNING"
source: https://arxiv.org/pdf/2608.10473v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:30:53"
---

# 论文速读：CRITIC-FREE PRETRAINING FOR EFFICIENT ONLINE REINFORCEMENT LEARNING FINE-TUNING

## 一句话总结
本文提出 Critic-Free Pretraining (CFP)，一种高效的离线到在线（O2O）强化学习微调范式。该方法在离线阶段完全放弃 Critic 训练，仅通过行为克隆（BC）预训练 Actor，随后在线上线阶段用从零初始化的全新 Critic 配合极短 Warm-up 完成校准，从而避免离线分布偏差对在线策略更新的干扰。

## 研究问题与动机
1. **核心问题**：传统 O2O 方法直接复用离线阶段训练好的 Critic 进行在线微调，但当策略与数据分布快速变化时，继承的 Critic 价值估计会与在线环境失配（misaligned），导致策略更新不准、探索低效。
2. **分布偏移引发悲观偏差**：现代离线数据集充斥次优/随机/失败轨迹，高质量演示极少；TD 自举机制会将这些低回报向后传播，使离线 Critic 产生系统性的价值低估（pessimism），进而拖累在线微调性能。
3. **现有方法的局限**：CQL、Cal-QL 等依赖保守正则或参考策略下界抑制过估计；RLPD、OPT 等依赖混合回放或复杂校准流程。它们仍保留离线 Critic 训练，无法彻底切断离线分布偏差向在线阶段的传递。
4. **重新定位离线阶段目标**：作者指出离线训练的核心目的是让 Agent 在更可能收集到有效轨迹的区域运行（提升采样效率），而非泛化一个精准的价值网络，因此主张将 Actor 预训练与 Critic 预训练解耦。

## 核心贡献（创新点）
1. **提出 Critic-Free 离线训练范式**：彻底摒弃离线阶段的 Critic 训练，证明其反而会损害后续在线微调性能；与已有工作的本质区别在于打破了传统 Actor-Critic 对称预训练假设，从源头切断离线悲观偏差向在线阶段的继承。
2. **设计极简且有效的 Warm-up 校准机制**：在线开始前用全新初始化的 Critic 在离线数据集上进行短时联合微调；与 WSRL/OPT 等方法相比，无需大量在线数据混合或复杂回放设计，仅靠 10K 步即可建立可用的价值估计基准。
3. **验证多基线兼容性与跨任务泛化能力**：CFP 可无缝嵌入 FQL、QC、QCFQL、QCFQL-nstep 等多种流模型骨干；与已有方法聚焦单一算法不同，本文证明该范式与不同归纳偏置的底层策略可实现正交叠加。
4. **提供机制层面的理论与仿真解释**：构造 16 状态表格 MDP 玩具示例并结合真实任务曲线，定量证明 CFP 的 Critic 在整个在线微调过程中 RMSE 更低、Q 值结构更贴近真实在线最优策略；弥补了现有 O2O 工作缺乏因果机制验证的不足。

## 方法详解
CFP 将训练流程拆分为离线、Warm-up、在线三个独立阶段，关键设计如下：
- **离线阶段（Critic-Free）**：仅训练 Actor，完全跳过 Critic 更新。Actor 损失退化为纯行为克隆形式（流策略单步蒸馏损失）：
  $L_{\pi}(\omega) = \alpha \mathbb{E}_{s \sim \mathcal{D}, z \sim \mathcal{N

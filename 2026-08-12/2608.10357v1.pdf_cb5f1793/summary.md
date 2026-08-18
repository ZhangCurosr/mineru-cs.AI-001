---
title: "Efficient Reinforcement Learning for Long-Horizon Tool-Use Agentic Tasks"
source: https://arxiv.org/pdf/2608.10357v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:49:38"
---

# 论文速读：Efficient Reinforcement Learning for Long-Horizon Tool-Use Agentic Tasks

## 一句话总结
本文提出了 SINKFLEX-RL，一套面向长程工具使用智能体的模块化强化学习训练系统；通过集成 Gymnasium 兼容环境封装、无独立价值网络的 GRPO 策略更新以及支持可微 Sink 归一化的 FlexAttention 路径，有效缓解多轮 On-policy Rollout 上下文过长导致的 HBM 瓶颈，在 τ²-Bench 零售任务上实现验证奖励 0.25→0.44 的提升，并在 4096 token 处节省 19.7% 峰值显存、成功运行 8192 token 训练。

## 研究问题与动机
- **长轨迹 On-policy 采样的显存压力**：长程工具使用智能体需在多轮交互中维持用户指令、领域策略、工具调用与模拟器状态的一致性；On-policy RL 要求反复生成完整轨迹，上下文长度随交互轮数线性增长，Eager 实现的 n×n 注意力分数结构会在反向传播前耗尽 HBM。
- **固定 Attention 算子无法覆盖模型特定语义**：FlashAttention 等融合算子虽降低显存搬运，但生产级 MoE 模型常依赖混合 Mask（因果+滑动窗口）与可学习的 Sink 归一化机制，固定内核接口无法暴露此类可微的分数修改路径。
- **传统 PPO 的价值网络开销过大**：多步工具调用场景下程序化轨迹奖励通常稀疏且仅在episode末尾可验证，训练独立 Critic/Value 网络会显著增加参数与显存负担，且对延迟信号的学习效率有限。
- **双控环境的系统集成缺口**：现有 RL 框架（如 verl）侧重分布式后训练数据流编排，缺乏对双控交互（Agent 与模拟用户共同驱动状态）、程序化奖励校验与长上下文 Attention 语义的统一整合方案。

## 核心贡献（创新点）
- 提出 SINKFLEX-RL 模块化训练系统，将环境接口、RL 数据流与 Attention 内核设计深度整合。与以往仅关注算法或单点算子优化的工作不同，本文强调系统级协同以保障长程 Agent RL 的显存可行性。
- 采用 GRPO 替代 PPO，在不训练独立价值网络的前提下，基于采样轨迹组进行组内相对优势归一化。与 DeepSeekMath 的动机一致，但本文将其与双控环境接口及长上下文可微 Attention 路径做了端到端耦合。
- 推导零值 Sink 代数等价式，将显式 Sink Token 物化转化为对标准 Attention 输出的可微缩放（α_sink = σ(lse − s_η)）。与 StreamingLLM 仅用于推理期 KV-cache 管理不同，该机制完整保留在训练前向/反向链路中。
- 基于 PyTorch FlexAttention 与 torch.compile 实现内存优化路径，支持因果+滑动窗口+前缀的混合 Mask 编译与 Block-Sparse 跳过。与纯工程调优工作不同，本文提供了模型特定 Attention 语义在 RL 训练中的可扩展实现范式，并在 8192 token 处消除 Eager 基线的 OOM 问题。

## 方法详解
- **双控环境封装**：任务实例表示为 E = (g, π_SOP, A_tool, s_0, u)，其中 u 为可主动提供信息或执行动作的模拟用户。通过 Gymnasium 兼容接口暴露 reset/step，Step 阶段将模型输出路由至自然语言响应、工具调用或终止处理器，标准化输出轨迹 τ = {(o_t, a_t, r_t

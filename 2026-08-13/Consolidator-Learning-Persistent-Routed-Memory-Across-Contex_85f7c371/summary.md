---
title: "Consolidator-Learning-Persistent-Routed-Memory-Across-Contex"
source: https://arxiv.org/pdf/2608.11701v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:55:57"
---

# 论文速读：Consolidator-Learning-Persistent-Routed-Memory-Across-Contex

## 一句话总结
提出 Consolidator，一个在 Phasor Memory Network (PMNet) 中运行的共享槽位局部算子，无需重放源 token 即可将短期记忆 (STM) 转换为可持久化的长期记忆 (LTM)，并使 LTM 同时作为可检索内容与引导后续插槽选择的访问状态。在控制性同地址更新任务上，仅训练 0.041% 参数（12.35K）即可使更新映射的 LTM 召回率从 18.32% 跃升至 87.02%，直接路由贡献了 +42.64 pp 的纯机制增益。

## 研究问题与动机
- **跨边界状态保留与修订的缺失**：现有可微记忆系统（如 NTM/DNC、Transformer-XL、Compressive Transformer 等）多关注跨段压缩或循环状态延续，但未解决“如何在上下文边界处将已路由的 STM 转换为可更新/修订的 LTM，且无需重放输入序列”。
- **LTM 仅作为静态内容库的局限**：单纯将 STM 快照复制/累加至 LTM 只能实现状态携带，但无法保证保留的状态能反作用于后续的显式记忆访问（如插槽选择）。
- **冻结接口下的适应性瓶颈**：预训练好的记忆接口是否需要一个学习的转换操作来处理冲突内容（而非原始累加）？保留的 LTM 是否只能供读取路径使用，还是可以同时作为访问状态调节路由？
- **前向非参数化适应的需求**：区别于 continual fine-tuning 或 test-time gradient descent，研究者希望推理时通过固定参数的非参数化状态更新实现 episode-level 的适应性，以降低计算与存储开销。

## 核心贡献（创新点）
1. **无重放的槽位局部相位转换算子**：Consolidator 对已占用的 STM 槽应用共享的 gated phase transform，将转换结果模 $2\pi$ 累加至 LTM 后清空 KV/STM，仅含 12.35K 参数。与已有工作的本质区别在于不依赖源 token 重放、不随树容量增长参数，且以零权重+单位相位初始化保证起始于精确恒等变换，便于梯度稳定传播。
2. **LTM 直接条件化插槽路由的架构归纳偏置**：将保留的 LTM 相位向量直接注入同级分层路由器的候选槽评分 $a_{t,b,g,j} = u_{t,b,g,j} + e_{b,g,j} + L_{b,g,j}$，使冻结的路由器能基于经验依赖的偏移量主动选择后续读写插槽。与已有工作的本质区别在于打破“LTM 仅作读取内容”的传统设定，确立其作为 access state 的结构化作用。
3. **机制隔离的同地址顺序更新任务与干预套件**：设计双段 modulo-10 映射任务，通过 learned vs. identity consolidation、direct routing on/off、mismatched/fresh LTM 等配对消融，精确拆解“跨重置携带、内容修订、内容检索、路由引导”四项记忆功能。与已有工作的本质区别在于提供机制层面的因果证据，而非仅报告端到

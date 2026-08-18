---
title: "Adversarial-Resilience-of-Poisson-Process-Submodular-Maximiz"
source: https://arxiv.org/pdf/2608.12134v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:06:31"
---

# 论文速读：Adversarial-Resilience-of-Poisson-Process-Submodular-Maximiz

## 一句话总结
本文证明了 SGS-Poisson 算法在预言机存在持久有界对抗误差时仍能保持经典近似比；将该离线韧性结果作为黑盒接入 offline-to-online 归约框架，首次为一般拟阵约束下的 full-bandit CMAB 达到极限近似后悔因子 1/e（非单调）与 1−1/e（单调），累积遗憾为 Õ(n^{1/5}k^{4/5}T^{4/5})。

## 研究问题与动机
- **核心问题**：在值预言机 f̂ 满足 ∀S, |f̂(S)−f(S)|≤ξ 的持久有界可控误差条件下，拟阵约束下的非负子模最大化离线算法能否保持原有的近似保证？
- **现有方法不足**：已有关一般拟阵的 full-bandit CMAB 结果仅能获得 1/2（单调）与 1/3（非单调）的近似因子，远低于经典离线边界 1−1/e 与 1/e。
- **技术障碍**：微小扰动可导致最大权残基、基交换映射、预处理停止时间及 Poisson 轨迹发生不连续跳变；且 f̂ 本身未必保持子模性或单调性，传统 Lipschitz 扰动分析失效。
- **动机**：验证 SGS-Poisson 的 Poisson 强度、单元素交换规则与 spiteful drop 完全不改动的情况下，其内在结构证书仍可在受控预言机下保持，从而打通通往最优在线后悔界的通道。

## 核心贡献（创新点）
- **证明适应性势函数漂移引理**：即便基于 f̂ 的预处理轨迹与精确轨迹完全不同，势函数 M_t=f(Q_t∪O_t)+½f(Q_t) 仍满足 E[M_{t+1}−M_t|F_t]≥(8OPT−kξ)/(k−t)，从根本上绕开了轨迹逼近的困难。
- **建立鲁棒 almost-above-average 交换引理**：在控制预言机误差下，SGS-Poisson 的基和浓度论证依旧成立，每次交换仅需额外付出 O(kξ) 项，保证 η≤C₁εOPT+C₂kξ。
- **给出 SGS-Poisson 对抗韧性定理**：返回的可行集满足 E[f(A_out)]≥(1/e−ε)OPT−O(kξ)（非单调）或 (1−1/e−ε)OPT−O(kξ)（单调），预言机复杂度 Õ(nk²ε⁻²)，算法结构零修改。
- **实现离线到在线的最优归约**：将韧性参数代入 [8] 的 black-box 归约，首次使 general matroid 上的 full-bandit CMAB 达到极限近似后悔因子 1/e 与 1−1/e，遗憾界 Õ(n^{1/5}k^{4/5}T^{4/5})。
- **本质区别**：区别于“对比精确/噪声轨迹”或“假设 f̂ 保持子模性”的既有思路，本文直接在受控预言机生成的离散轨迹上证明结构证书的保持，属于非平凡的自适应分析。

## 方法详解
- **问题形式化**：设拟阵 M=(U,I) 秩为 k，基础集 |U|=n，目标 f:2^U→[0,1] 非负子模。预言机为 ξ-controlled，即 ∀S, |f̂(S)−f(S)|≤ξ，误差固定且可具对抗性。
- **算法骨架（SGS-Poisson 保持不变）**：以速率 k/t 驱动非齐次 Poisson 过程。在每个事件时刻对当前基 A 执行 valid swap：按多线性边际权重在收缩拟阵中找最大权残基 Z，构造确定性交换映射 h_t:Z→O_t，均匀采样 j∈Z 进入，并以概率 t 执行 spiteful drop。
- **鲁棒最优值证书**：对增广拟阵（添加 k 个零边际虚拟元素）运行 RRG r=⌈k/2⌉ 次，

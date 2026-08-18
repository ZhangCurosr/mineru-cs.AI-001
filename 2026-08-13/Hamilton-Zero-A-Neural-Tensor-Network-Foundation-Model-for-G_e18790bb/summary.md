---
title: "Hamilton-Zero-A-Neural-Tensor-Network-Foundation-Model-for-G"
source: https://arxiv.org/pdf/2608.11911v1.pdf
model: agnes-2.5-flash
chunks: 7
summarized_at: "2026-08-18 12:21:16"
---

# 论文速读：Hamilton-Zero: A Neural Tensor Network Foundation Model for Ground State Computation

## 一句话总结
Hamilton-Zero 是首个面向任意二次 qubit Hamiltonian 基态计算的 Foundation Model，以约 0.5B 参数在 SU(2) 流形上预训练，通过零样本推理与轻量微调实现跨系统尺寸、相互作用拓扑与 Hamiltonian 类型的严格保变分基态求解。

## 研究问题与动机
- **核心问题**：任意二次 qubit Hamiltonian 的基态能量与波函数计算面临 Hilbert 空间组合爆炸与从头训练成本高昂。
- **现有方法不足**：传统变分蒙特卡洛（VMC）与神经量子态（NQS）在离散 Hilbert 空间参数化，不严格保证变分上界，且缺乏跨拓扑/系统尺寸的泛化能力。
- **理论瓶颈**：标准 ansatz 未显式编码 SU(2) 对称性与规范不变性，导致采样效率低、优化 landscapes 崎岖。
- **工程瓶颈**：大规模量子系统训练路径耗时 >1 年，缺乏对称性感知的离散决策机制与高效数据增强范式。

## 核心贡献（创新点）
- **首个量子基态 Foundation Model**：预训练后支持零样本与仅微调 ~4.7M 参数的轻量适配，泛化至 8100 qubits 规模。
- **流形变分上界严格保证**：基于 Peter-Weyl 定理将波函数约束为 spin-1/2 sector，替代离散 Hilbert 振幅，确保 $\langle\psi|\hat{H}|\psi\rangle/\langle\psi|\psi\rangle \ge E_0$。
- **双输入 Transformer 架构**：原生融合自旋构型流（$q \in \text{SU(2)}^N$）与 Hamiltonian 耦合张量上下文，支持 per-bond/per-site/global 多尺度表征。
- **2-WL 对称商路由策略**：用自回归 masked slot picks 与 WL 颜色折叠替代暴力置换搜索，将策略支撑集压缩至能量最小化轨道，显著降低 REINFORCE 梯度方差。
- **高效训练管线**：集成 SU(2) replica-exchange Langevin 采样、精确规范增强、hot-swapped 数据增强与扩展 KFAC，训练耗时从 >1 年压缩至约 4 天。

## 方法详解
- **Lie 导数参数化**：将 Pauli 算符映射为 SU(2) 流形上的微分算子（$\hat{\sigma}_i^a \longleftrightarrow -iL_i^a$，$\hat{\sigma}_i^a \hat{\sigma}_j^b \longleftrightarrow -L_i^a L_j^b$），通过自定义 JAX 自动微分原语计算二阶微分算子作用。
- **双输入流与分层架构**：Trunk 采用 LLM 风格 Transformer（attention + FFN 残差 + global context stream）；Featuriser 经谱/RMS 归一化与列行 attention 生成 per-bond/per-site/global embedding，缺失 bond 用 learned absent token；Leaf reader 将每 site 提升为线性奇数据流并与 Hamiltonian 上下文融合。
- **Rank-4 Merge Tree**：平衡二叉树收缩结构，共享 merge 层保证 sign flip 传递性；context stream 采用 nGPT 风格归一化插值避免饱和。
- **自因子化路由策略**：$\pi_\theta(p|h,\ell^{(L)},e^{(L)},m)=\prod_{t=0}^{\hat{N}-1}\pi_t(p_t|\cdot)$，每步通过掩码槽选择解码。核心组件包括：候选构建器（RMS 归一化+线性投影+FFN）、树前缀编码器（物化为偏合并树，含因果 2-FWL 精炼与 cover-query/candidate-to-cover/post-prefix 块）、Pointer（共享 bias-free $W_q,W_k$，零初始化 $W_q$ 与温度 $\tau$）、对称商（2-WL 有序对折叠 logits，消除 $(\hat{N}-N)!$ 冗余，Gumbel-max 采样）。
- **梯度分解与方差控制**：策略梯度拆解为 VMC 梯度（中心化为路由专属均值 $\bar{E}^{(p)}$）与 REINFORCE 梯度 $\nabla_\theta\log\pi_\theta(p)$；对 local energy 做 batch clipping 以降低梯度噪声。
- **训练工程**：Hot-swapped 数据增强（每 epoch 预计算 ±bond/噪声/对称场扰动，快训练循环 2 轮）；精确规范增强（Haar 随机 $u\in\text{SU}(2)$ 同时变换 $(J,h,q)$ 保持局部能量不变）；SU(2) replica-exchange Langevin 采样；扩展 KFAC 优化器；Folx forked JAX 加速。

## 实验与结果
- **规模与设置**：预训练 5000 种拓扑，扩展至数十万 Hamiltonian；预训练最大 64 qubits，zero-shot 评估达 1024 qubits，fine-tune 评估达 8100 qubits。
- **缩放律**：V-score 与 ED 相对 gap 均随计算量幂律下降，指数为 $C^{-0.53}$（相对 gap $C^{-0.61}$），优于 LLM scaling 约 5 倍。
- **外推表现**：$N>22$ 预测中位相对 gap 为 3.45%（IQR 1.86%–10.21%）。
- **Zero-shot 三轴评估**：轴 A（组合优化）、轴 B（未见拓扑）、轴 C（最难样本）的中位带符号 gap 分别为 4.02%、4.21%、12.63%。
- **效率**：全 256 系统微调耗时 234 GPU h（单 A100），zero-shot 评估耗 47 GPU h（8×H200）。

## 相关工作脉络
- **神经量子态（NQS）**

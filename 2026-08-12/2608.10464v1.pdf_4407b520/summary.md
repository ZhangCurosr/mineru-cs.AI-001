---
title: "Quantum Incremental Learning with Mixed State Prototypes"
source: https://arxiv.org/pdf/2608.10464v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:24:22"
field: "量子机器学习与增量学习"
keywords: ["quantum incremental learning", "mixed state prototype", "Hilbert-Schmidt distance", "CCPS ansatz", "catastrophic forgetting", "NISQ"]
innovations: ["基于CCPS混合态原型的量子增量学习框架，避免电路宽度扩展", "HS距离经SWAP test分解实现硬件友好的度量学习", "两阶段解耦训练：骨干表征学习后独立拟合类原型"]
benchmarks: ["CIFAR-100 32-class incremental", "Tiny-ImageNet 32-class incremental"]
---

# 论文速读：Quantum Incremental Learning with Mixed State Prototypes

## 一句话总结
论文提出了一种基于**可训练混合态原型**的量子增量学习框架，通过在共享量子骨干网络上附加轻量级、可扩展的经典模块来管理类别扩展，利用 CCPS ansatz 构建的混合态原型实现基于 Hilbert-Schmidt 距离的语义匹配，在有限量子比特资源下有效缓解增量学习中的灾难性遗忘。

## 研究问题与动机
1. **量子分类器的正交基态容量瓶颈**：传统量子分类器依赖固定基态概率或期望值计算类别分数，随标签空间增长需扩展测量结构或增加量子比特，但追加量子比特会改变全局量子态与希尔伯特空间，破坏已学习的纠缠相关性，加剧灾难性遗忘。
2. **NISQ 时代硬件约束**：含噪中等规模量子（NISQ）设备限制电路宽度，盲目增加电路宽度或深度易触发 barren plateau 现象，导致量子网络不可训练。
3. **纯态表示能力不足**：纯态将单个输入编码为希尔伯特空间中的单一状态向量，缺乏表示类别内多样分布的统计容量；混合态通过正交态凸组合封装数据随机性与类内方差。
4. **经典增量学习策略难以直接量子化**：经典深度学习扩展最终线性层即可容纳新类，但量子网络的结构差异使得直接迁移困难，需探索基于状态距离的距离度量学习范式。

## 核心贡献（创新点）
1. **可扩展的量子原型框架**：通过将新增类别表示为类原型而非增加共享量子骨干的电路宽度，突破正交基态数量的容量约束；与已有工作本质区别在于不依赖显式参数化分类头，采用隐式距离匹配机制。
2. **混合态原型设计（CCPS ansatz）**：通过动态加权纯态构建混合态原型，避免冗余量子比特，提供类 PCA 的噪声滤波特性；与纯态原型（如 Fidelity Classifier）的本质区别在于单原型的低秩近似可过滤低贡献分量。
3. **可分解的 HS 距离度量**：利用 SWAP test 将 HS 距离计算分解为经典概率评估与量子重叠度测量，计算复杂度从 $O(d^3)$ 降至更低；与 trace distance、Bures 距离等需要谱分解的度量本质区别在于硬件友好性。
4. **两阶段解耦训练范式**：第一阶段用辅助 MLP 头优化量子骨干特征提取，第二阶段冻结骨干独立拟合混合态原型；与端到端联合优化本质区别在于避免原型参数干扰骨干表征空间。

## 方法详解
**整体架构**：混合经典-量子框架，包含四个组件——经典预处理模块、量子特征提取骨干（QCNN）、辅助 MLP 头（训练时用于表征学习，推理时丢弃）、每类独立的混合态原型电路。

**1) 特征提取与编码**：
- 输入图像经 ResNet-18 提取 512 维特征，经 Dropout + ReLU + 线性投影压缩至 256 维。
- L2 归一化后通过幅度编码（amplitude encoding）映射到 8 量子比特的初始纯态 $|\psi(\mathbf{x})\rangle = \sum_{j=0}^{255} a_j |j\rangle$。

**2) 量子骨干（QCNN）**：
- 采用金字塔结构：量子卷积层（参数化 SU(4) 双量子比特门块）+ 量子池化层（受控旋转 + 部分迹操作）。
- 池化层迹掉一半量子比特，将 8 量子比特系统缩减至 4 量子比特，输出 $16 \times 16$ 密度矩阵 $\rho(\mathbf{x})$。
- 部分迹操作自然抽象结构不确定性，使保留子系统进入混合态。

**3) 辅助 MLP 头（仅训练阶段）**：
- 对 4 量子比特系统计算基测量，获得 16 维概率向量 $\mathbf{p}(\mathbf{x})$。
- 经两层 MLP（隐藏层 32 维）映射到类别 logits，使用标准交叉熵损失 $\mathcal{L}_{rep}$ 优化骨干。
- 骨干训练完成后丢弃此头。

**4) 混合态原型构造（CCPS ansatz）**：
- 每类独立原型电路：3 层 VQC，每层 4 个 SU(4) 块（交替环状拓扑），共 180 个参数。
- 原型密度矩阵：$\rho_c = \sum_{i=0}^{K-1} w_{c,i} U_c |i\rangle\langle i| U_c^\dagger$，其中 $w_{c,i}$ 通过 softmax 参数化保证概率单纯形约束。
- 原型秩 $K=7$（固定，不做数据集特异性选择）。

**5) 分类度量（HS 距离 + SWAP test）**：
- HS 距离平方：$\mathcal{D}_c(\mathbf{x}) = \|\rho(\mathbf{x}) - \rho_c\|_F^2 = \text{Tr}[\rho(\mathbf{x})^2] + \text{Tr}[\rho_c^2] - 2\text{Tr}[\rho(\mathbf{x})\rho_c]$。
- 由于查询纯度 $\text{Tr}[\rho(\mathbf{x})^2]$ 对所有类相同，分类简化为最大化 logit：
$$\text{logit}_c(\mathbf{x}) = \mathcal{O}_c(\mathbf{x}) - \frac{1}{2}\sum_{i=0}^{K-1} w_{c,i}^2$$
其中 $\mathcal{O}_c(\mathbf{x}) = \sum_{i} w_{c,i} \text{Tr}[\rho(\mathbf{x})|\psi_{c,i}\rangle\langle\psi_{c,i}|]$ 通过 SWAP test 估计。
- SWAP test 仅需 1 个辅助量子比特：$P(0) = \frac{1}{2} + \frac{1}{2}\text{Tr}[\rho\sigma]$，故重叠度 $\mathcal{F}_i = 2P(0)-1$。

**6) 原型拟合损失**：
$$\mathcal{L}_{proto}^c(\theta_c, s_c) = \sum_{i=0}^{K-1} w_{c,i}^2 - \frac{2}{N}\sum_{n=1}^N\sum_{i=0}^{K-1} w_{c,i}\mathcal{F}_{c,i}^{(n)}$$
等价于对类别均值密度矩阵应用变分量子 PCA。

**7) 增量学习流程（Algorithm 2）**：
- **Step 1**：骨干表征学习——用当前任务数据 + 旧类回放示例联合更新骨干和辅助头。
- **Step 2**：混合态原型拟合——冻结骨干，独立优化每类原型参数。
- **Step 3**：回放记忆管理——按 HS 距离选取每类最近样本存入缓冲区（内存预算 M=640，均匀分配）。
- **Step 4**：基于原型的全局推理——查询样本经冻结骨干提取密度矩阵，与所有可见类原型计算距离，取最近邻。

## 实验与结果
**数据集**：CIFAR-100（3 个固定互斥的 32 类子集 Split A/B/C）和 Tiny-ImageNet（32 类），均采用 16→20→24→28→32 类的五阶段增量协议。

**评估指标**：Last Acc（最终任务准确率）和 Avg Acc（各阶段平均准确率）。

**静态分类实验（Table I）**：
- 在 4 个数据集上的平均准确率：Split A = 0.8323, Split B = 0.7766, Split C = 0.8222, TinyImageNet = 0.6997，总体 Avg = **0.7827**。
- 超越所有对比量子分类器（HEA-VQC: 0.6167, QCNN: 0.6372, TTN-QNN: 0.1236, Fidelity Classifier: 0.7362）。
- 超越 Split A 上最强经典分类器（Linear Softmax: 0.8281）；在 Split B/C 上略低于最强经典方法 1.63pp / 0.23pp。
- 相比直接测量的 QCNN，平均提升 12.88~15.83 个百分点。

**增量学习实验（Tables II-III）**：
- CIFAR-100 Split A：Last Acc = **0.5728**，Avg Acc = **0.7178**。
- Split B：Last Acc = 0.4947，Avg Acc = 0.6465。
- Split C：Last Acc = 0.5481，Avg Acc = 0.7097。
- TinyImageNet：Last Acc = 0.3838，Avg Acc = 0.5371。
- 虽未超越最强经典基线（如 FOSTER: Last 0.6847, Avg 0.7796），但优于 BiC、Eucl-NCM、NCM、MUC-LwF、SSRE、DeeSIL、TOPIC+、SDC、LUCIR 等多个方法。
- 性能随增量阶段逐渐下降（图 7），呈现与原型/统计类基线相当的渐进式衰退，而非 abrupt collapse。

**消融实验（Table IV, Split A）**：
- 移除混合态（纯态原型）：Last Acc 从 0.5728 降至 0.5450，Avg Acc 从 0.7178 降至 0.6668。
- 冻结骨干：Last Acc = 0.4788，Avg Acc = 0.6586。
- 移除回放：Last Acc = 0.5122，Avg Acc = 0.6735。
- 混合态原型贡献最显著（+5.10pp Avg Acc）。

**原型秩敏感性（Table V, Fig. 9）**：
- $K=7$ 为全局固定设置，非数据集特异性调优。
- Split A 最优 $K=12$（Avg 0.7286），Split B 最优 $K=7$（Avg 0.6465），Split C 最优 $K=15$（Avg 0.7159），TinyImageNet 最优 $K=7$（Avg 0.5371）。
- $K=16$（满秩）并非最佳，支持低秩近似的去噪假设。

**实验配置**：8 量子比特骨干 + 4 量子比特原型；ResNet-18 预处理；SGD+动量 0.9 优化骨干，Adam 优化原型；初始任务 240 epoch，原型拟合 60 epoch，增量微调 36 epoch；内存预算 640 样本（~20 样本/类）。

## 相关工作脉络
1. **iCaRL [6]**：基于 exemplar replay 的经典增量学习基线，使用贪心算法选择代表性样本存入缓冲区；本文在其基础上引入量子混合态原型替代经典最近中心分类器。
2. **QCNN [45]**：将量子卷积/池化层应用于图像分类；本文继承其金字塔结构与部分迹去噪机制，但将直接概率测量改为混合态原型距离匹配。
3. **Fidelity Classifier [50]**：使用可训练纯态原型 + 态保真度度量；本文本质区别在于混合态原型具有更高表示容量和类 PCA 去噪能力。
4. **HEA-VQC [48] / TTN-QNN [49]**：基于直接测量基态概率的分类器；本文指出此类方法将固定希尔伯特空间均分给每类，无法处理非均匀特征分布，而混合态原型可自适应划分特征空间。
5. **FeCAM [15] / FeTrIL [14]**：无 exemplar 的原型/统计方法；本文结合 exemplar replay + 混合态原型，在有限内存下取得更有意义的增量学习能力。
6. **QMSC (Quantum Mixed State Compiling) [41]**：提出 CCPS ansatz 用于混合态制备；本文将其创新性地应用于增量学习的类原型构建，实现硬件友好的低秩近似。

## 局限性与未来方向
1. **绝对性能低于最强经典方法**：在 TinyImageNet 上 Last Acc 仅 0.3838，显著落后于 FOSTER (0.3588 但内存 640) 和其他基线；作者承认当前结果旨在验证量子方法的有效增量学习能力，而非追求 SOTA。
2. **固定秩 $K$ 的普适性**：不同数据集最优 $K$ 不同（Split A: 12, Split B: 7, Split C: 15），当前统一 $K=7$ 为折中设置，缺乏自适应秩选择机制。
3. **模拟实验为主**：所有实验基于经典模拟器（PennyLane），未在真实量子硬件上验证；NISQ 设备噪声对混合态原型的影响未充分探索。
4. **小规模增量场景**：仅测试 16→32 类、5 个任务阶段的场景，扩展至更大规模类别序列和更复杂数据集的可行性待验证。
5. **内存预算固定**：采用均匀分配策略（每类等量样本），未探索基于原型距离的自适应记忆分配。

未来方向（作者自述）：探索更硬件友好的原型电路、研究真实量子设备上的噪声感知训练方法、扩展至更大规模增量场景和多样化数据集。

## 研究启发与可借鉴点
1. **混合态作为类原型的表示优势**：将类别表示为密度矩阵而非纯态/点估计，天然捕获类内分布的不确定性；该方法可迁移至经典度量学习中，用协方差矩阵替代均值向量作为原型。
2. **CCPS ansatz 的低秩去噪机制**：原型秩 $K$ 控制相当于量子 PCA，过滤低贡献分量；该思想可启发电经典降维或字典学习中的稀疏表示设计。
3. **两阶段解耦训练范式**：先冻结骨干再独立优化原型，避免端到端训练时的参数干扰；该策略可推广至其他量子-经典混合模型的可扩展性设计。
4. **HS 距离的硬件友好分解**：通过 SWAP test 将距离计算分解为单量子比特测量，避免完整密度矩阵操作；该技巧适用于 NISQ 设备的度量学习任务。
5. **量子特征提取中的部分迹去噪**：池化层的 partial trace 操作天然产生混合态，这一物理过程可类比经典网络中的 dropout/pooling 去噪机制，为量子-经典混合架构设计提供新思路。

## 关键术语表
**Incremental Learning（增量学习）**：模型从连续数据流中逐个学习任务/类别，同时最小化对已有知识的灾难性遗忘。
**Catastrophic Forgetting（灾难性遗忘）**：神经网络在适应新任务时，对先前学习到的表征的急剧遗忘现象。
**Mixed State / Density Matrix（混合态/密度矩阵）**：描述开放量子系统statistical ensemble的算子 $\rho$，可表达纯态无法捕获的概率混合与不确定性。
**CCPS (Convex Combination of Pure States)**：一种混合态制备 ansatz，将 rank-R 混合态表示为 R 个变换后计算基态的加权混合，无需额外辅助量子比特。
**Hilbert-Schmidt (HS) Distance**：基于 Frobenius 范数的量子态距离度量 $\|\rho-\sigma\|_F^2$，计算复杂度低于 trace distance 和 Bures 距离，适合重复原型拟合。
**SWAP Test**：通过 1 个辅助量子比特的受控 SWAP 门测量两个量子态重叠度 $\text{Tr}[\rho\sigma]$ 的实验电路。
**QCNN (Quantum Convolutional Neural Network)**：将经典 CNN 的局部感受野和权值共享思想推广到量子域，通过参数化酉变换和量子池化（部分迹）提取语义特征。
**Amplitude Encoding（幅度编码）**：将 $2^n$ 维经典向量编码为 $n$ 量子比特纯态的概率振幅，实现指数级压缩但需 $O(2^n)$ 量子门。

## 可复现要素
**数据集**：CIFAR-100 和 Tiny-ImageNet（公开数据集）；类划分子集协议由代码指定。
**代码**：已开源，地址 https://anonymous.4open.science/r/QPIL-3E43。
**框架**：PyTorch + PennyLane（量子电路模拟）。
**硬件**：NVIDIA GeForce RTX 3090 (24GB) GPU + x86 CPU。
**关键超参**：
- 骨干：8 量子比特 QCNN，ResNet-18 (512 维) → 256 维线性投影 → 幅度编码
- 原型：4 量子比特系统，3 层 VQC（每层 4 个 SU(4) 块），秩 $K=7$
- 训练：初始任务 240 epoch LR=0.01 batch=128；原型拟合 60 epoch LR=0.04 batch=16；增量微调 36 epoch LR=0.01
- 优化器：SGD (momentum=0.9, weight decay=1e-4) + Adam；warmup cosine scheduler
- 内存：总预算 640 样本，均匀分配至各可见类

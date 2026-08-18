---
title: "CABS-Efficient-and-Scalable-Model-Merging-via-Conflict-Aware"
source: https://arxiv.org/pdf/2608.12842v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:21:20"
field: "模型合并与知识融合"
keywords: ["Model Merging", "Task Vector", "Sparsification", "CMA-ES", "Model Mergeability", "Adaptive Weight Allocation", "Conflict-Aware Pruning"]
innovations: ["提出AWA梯度自由权重分配策略，将GPU显存降至推理级别", "设计非对称适应度函数缓解多任务损失尺度差异导致的优化失衡", "系统实证研究六大可合并性影响因素并提出RSS量化指标"]
benchmarks: ["LLM Leaderboard", "Open LLM Leaderboard 2", "GLUE", "FusionBench"]
---

# 论文速读：CABS-Efficient-and-Scalable-Model-Merging-via-Conflict-Aware

## 一句话总结
本文提出 CABS+ 框架，通过梯度自由自适应权重分配（AWA）替代原有网格搜索，降低时间复杂度与 GPU 显存开销，同时引入非对称适应度函数缓解多任务优化失衡问题；此外系统性地研究了影响模型可合并性的关键因素并提出 RSS 度量指标。

## 研究问题与动机
1. 现有模型合并方法（如 AdaMerging）依赖梯度反向传播构建计算图，GPU 显存消耗随模型规模、序列长度和任务数量线性增长，难以在消费级 GPU 上运行数十亿参数的大语言模型。
2. 原 CABS 方法采用网格搜索确定任务缩放系数，时间复杂度随任务数呈指数增长，且其优化目标为各任务性能求和，易被高性能任务主导导致其他任务性能下降。
3. 社区缺乏对"可合并性"的系统性实证研究，缺少可量化评估模型合并潜力的指标，难以在实际应用前指导模型选择。
4. 直接部署多个专用模型面临存储和维护成本高、多任务数据获取受隐私限制等问题，模型合并成为高效替代方案，但需解决参数冲突与知识干扰。

## 核心贡献（创新点）
1. **提出 AWA 梯度自由权重分配策略**：基于 CMA-ES 进化算法进行系数优化，无需构建计算图，GPU 显存降至推理级别；与已有梯度搜索方法（如 AdaMerging）的本质区别在于完全规避反向传播开销。
2. **设计边界约束与非对称适应度函数**：通过上下界限制系数搜索空间，并利用相对变化率的非对称惩罚机制平衡不同量级的任务损失；与标准 CMA-ES 直接最小化绝对损失之和的区别在于消除了任务间损失尺度差异导致的优化偏差。
3. **系统研究模型可合并性影响因素**：首次实证分析学习率、训练轮数、任务异质性、数据分布差异、模型架构和模型规模六个关键因素对合并性能的影响；与既有工作仅关注合并算法的差异在于建立了合并前模型筛选的科学依据。
4. **提出 RSS（Relative Synergy Score）度量指标**：用于量化模型合并的协同增益与破坏性干扰程度；相对于以往定性描述的优势在于提供了标准化、可比较的合并潜力评估工具。

## 方法详解
**整体框架分为两个阶段：**

**阶段一：冲突感知稀疏化（CA）与平衡稀疏化（BS）**
- **冲突感知稀疏化（CA）**：采用顺序剪枝策略消除任务向量间的参数重叠。首先对 $\tau_A$ 执行 $n:m$ 剪枝并生成 mask，然后在 $\tau_B$ 中排除 mask_A 覆盖的位置后再次剪枝，确保保留参数不重叠。当保留比例之和超过1时，采用符号选择和参数平均策略处理不可避免的重叠区域。经 CA 处理的任务向量在 Frobenius 内积下正交。
- **平衡稀疏化（BS）**：将权重矩阵划分为 $m$ 个非重叠连续块，每块保留绝对值最大的 $n$ 个参数，在全局各层均匀分布保留权重，防止因参数集中于少数层导致严重冲突。注意 BS 生成的合并模型仍为稠密模型。

**阶段二：自适应权重分配（AWA）**
- 将缩放系数优化建模为连续空间中的无梯度自动优化问题，基于 CMA-ES 实现：
  - **梯度自由采样与边界约束**：从多元正态分布采样候选系数 $\lambda_k^{(g)} \sim m^{(g)} + \sigma^{(g)}\mathcal{N}(0, C^{(g)})$，并通过投影函数 $P_\Omega$ 将系数约束在 $[l=0.1, u=2]$ 范围内。
  - **非对称适应度评估**：定义相对变化量 $\Delta_t(\lambda) = \frac{L_t(\theta(\lambda)) - L_{\text{base}}^{(t)}}{L_{\text{base}}^{(t)}}$，构造非对称惩罚函数：$\Delta_t > 0$ 时惩罚系数 $\alpha=100$，$\Delta_t \leq 0$ 时系数 $\beta=1$，最终适应度 $F(\lambda) = \sum_t f_t(\lambda)$。
  - **种群选择与均值更新**：选取前 $\mu = K/2$ 个最优样本，以对数加权平均更新分布均值 $m^{(g+1)}$。
  - **步长控制与协方差矩阵自适应**：利用共轭演化路径 $p_\sigma$ 和 $p_c$ 动态调整步长 $\sigma$ 和协方差矩阵 $C$，使搜索分布自动适应目标函数的等高线形状。
- **时间复杂度**：网格搜索为 $O(S^T)$，AWA 为 $O(K \times G)$，有效避免维度灾难。
- **合并公式**：$W_{\text{final}} = W_{\text{base}} + \sum_{t=1}^{T} \lambda_t^* \cdot \tilde{\tau}_t$

**效率优势机制**：AWA 优化主要在 CPU 端通过 Numpy 矩阵运算完成，GPU 仅用于前向推理计算损失，任何时刻仅需保留基础模型参数和单层合并参数，GPU 显存与任务数量无关（$O(1)$），而 AdaMerging 和 WUDIMerging 为 $O(n)$。

## 实验与结果
**实验设置**
- 数据集：LLM Leaderboard（MMLU, HellaSwag, TruthfulQA, ARC, Winogrande, GSM8K）、Open LLM Leaderboard 2（IFEval, BBH, MATH, GPQA, MUSR, MMLU-Pro）、GLUE（CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2）、RACE、SQuAD，以及 ViT-B/32 的6个视觉任务，共27个数据集。
- 模型：Mistral-7B-v0.1、Qwen-2.5-7B-Instruct、RoBERTa、GPT-2、ViT-B/32。
- 基线：Task Arithmetic、TIES-Merging、AdaMerging、WUDIMerging、CABS（原版）、DARE、Magnitude Pruning。
- 硬件：V100 GPU（32GB 显存）。

**主要结果**
- **大模型性能**：在 Qwen2.5 系列上，CABS+（fqFirst）平均 45.09，超越 WUDIMerging（44.09）和 AdaMerging（43.03）；在 Mistral 系列上，CABS+ 平均 76.71，略低于 WUDIMerging（76.82）但显著优于 AdaMerging（75.90）。
- **小模型性能**：RoBERTa 上 4任务/2任务/6任务合并均取得最优或次优结果；GPT-2 上同样表现稳定。
- **跨模态实验**：ViT-B/32 上平均 82.50，优于 CABS（81.08）和 WUDIMerging（82.30）。
- **效率对比（Table VIII）**：Mistral 上 CABS+ 仅需 15.95GB 显存，约为 AdaMerging（66.70GB）的 25%；运行时间与 AdaMerging 相当（1h），但仅为 WUDIMerging（4h）的 1/4。
- **整体提升**：相比 AdaMerging 平均提升 16.97%，相比 WUDIMerging 平均提升 12.93%。

**理想模型基准**：各任务在所有微调变体中的最优成绩的平均值作为性能上界，CABS+ 在多数设置下接近甚至超越理想模型。

## 相关工作脉络
1. **Task Arithmetic / Model Soups**：直接对参数进行加权平均或任务向量加法，未处理参数冲突，是基础但性能有限的 baseline。
2. **TIES-Merging**：通过幅度剪枝去除冗余参数并结合符号冲突解决，但幅度剪枝在模型合并场景中可能加剧权重分布不均，与 CABS 的交叉层均匀剪枝形成对比。
3. **DARE**：借鉴 Dropout 机制随机丢弃大量参数并重新缩放，证明稀疏性可缓解冲突，但与 CABS 的结构化剪枝目标不同（压缩加速 vs 冲突消减）。
4. **AdaMerging**：在无测试数据情况下通过梯度下降优化合并系数，需同时维护所有任务向量的计算图和中间激活，显存开销大；CABS+ 的 AWA 从根本上避免了这一瓶颈。
5. **WUDIMerging**：利用任务向量的线性子空间几何关系抑制干扰，采用逐层优化策略，存在频繁的 CPU-GPU 数据传输开销；CABS+ 全程在单个推理通道中完成，避免 I/O 通信瓶颈。
6. **Fisher Merging / RegMean**：分别通过 Fisher 信息矩阵和预测差异最小化确定权重，需额外训练或优化过程，而 CABS+ 为纯数据自由的无训练合并。

## 局限性与未来方向
1. **AWA 计算耗时**：虽然显存占用低，但无梯度优化仍需多代迭代（默认 G=50），在超大规模任务合并场景下仍有时间开销。
2. **仅针对同源基模型**：实验主要验证同一基模型微调出的任务向量合并，跨架构或跨预训练基础的模型合并效果未充分验证。
3. **超参数敏感性**：AWA 的边界约束范围 $[0.1, 2]$ 和 Population size $K=6$ 等超参需人工设定，可能影响不同场景下的最优性能。
4. **仅测试数据自由场景**：方法适用于无训练数据情况，但在可访问部分验证数据时可能仍有优化空间。
5. **作者自述**：未来将探索多模态和异构模型合并以扩展适用范围。

## 研究启发与可借鉴点
1. **无梯度优化替代梯度优化**：AWA 策略通过进化算法实现合并系数优化，规避了计算图显存瓶颈，为资源受限环境下的大模型合并提供了新思路；可迁移到任何需要优化模型参数组合但显存不足的场景。
2. **非对称适应度函数设计**：利用相对变化率而非绝对损失，并引入非对称惩罚抑制性能下降，这一思路可推广至多目标优化中的尺度对齐问题。
3. **系统化可合并性研究范式**：本文对六个影响因素的对照实验设计（固定合并算法以隔离源模型影响）值得借鉴；RSS 指标可作为后续研究的通用评估工具。
4. **结构剪枝与合并系数的解耦**：先通过 CA+BS 消除参数冲突，再进行平滑空间的系数优化，这种分阶段策略提高了优化可靠性；类似"先预处理后优化"的范式可应用于其他组合优化问题。
5. **CPU-GPU 分离计算策略**：将重量级优化放在 CPU 上执行、GPU 仅负责前向推理的模式，为内存受限场景下的大模型处理提供了实用工程方案。

## 关键术语表
**Model Merging（模型合并）**：在参数空间直接组合多个专家模型以构建统一多任务模型，无需额外训练或多任务数据。

**Task Vector（任务向量）**：微调后模型参数与预训练模型参数之差，用于表示任务特异性知识更新。

**Sparsification（稀疏化）**：通过剪枝去除冗余或冲突参数以降低任务间干扰的技术，本文采用结构化 $n:m$ 剪枝。

**CMA-ES（Covariance Matrix Adaptation Evolution Strategy）**：一种无梯度进化优化算法，通过协方差矩阵自适应调整搜索分布。

**Adaptive Weight Allocation (AWA)**：本文提出的基于 CMA-ES 的梯度自由合并系数优化策略，支持边界约束和非对称适应度评估。

**Relative Synergy Score (RSS)**：衡量合并协同效应的指标，定义为 $(Score_{merged} - Score_{ideal}) / Score_{ideal} \times 100\%$，正值表示协同增益，负值表示破坏性干扰。

**Conflict-Aware Sparsification (CA)**：通过顺序剪枝和掩码排除消除任务向量间参数重叠的冲突消减机制。

**Balanced Sparsification (BS)**：采用逐层 $n:m$ 块剪枝确保保留权重在全网均匀分布的平衡稀疏化策略。

## 可复现要素
- **数据集**：GLUE、LLM Leaderboard、Open LLM Leaderboard 2、MMLU-Pro、BBH、MATH、GPQA、MUSR、RACE、SQuAD、FusionBench 等；部分开源。
- **代码**：已开源，链接为 https://anonymous.4open.science/r/CABS-Plus-70C1。
- **模型权重**：Hugging Face 公开可用。
- **关键超参**：AWA 种群大小 $K=6$、初始步长 $\sigma=0.05$、迭代次数 $G=50$、边界约束 $[0.1, 2]$、早期停止条件（连续6代损失无显著变化）；BS 稀疏度小模型 0.90、大模型 0.75；CMA-ES 标准参数。
- **硬件环境**：V100 GPU（32GB）；评测使用 lm-evaluation-harness 0.4.0，batch size 为 auto。

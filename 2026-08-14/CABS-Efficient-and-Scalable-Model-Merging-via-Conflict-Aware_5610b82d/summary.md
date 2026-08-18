---
title: "CABS-Efficient-and-Scalable-Model-Merging-via-Conflict-Aware"
source: https://arxiv.org/pdf/2608.12842v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:21:07"
field: "模型合并与多任务学习"
keywords: ["Model Merging", "Task Vector", "Structured Pruning", "CMA-ES", "Model Mergeability", "Gradient-Free Optimization", "Sparsification"]
innovations: ["提出AWA无梯度进化策略替代网格搜索，将合并时间复杂度从指数级降至多项式级且显存降至O(1)", "设计非对称适应度函数处理多任务损失尺度差异，防止高损失任务主导优化", "提出RSS指标并系统实证量化六大因素对模型可合并性的影响"]
benchmarks: ["LLM Leaderboard", "Open LLM Leaderboard 2", "GLUE", "FusionBench", "ViT-B/32 六视觉任务"]
---

# 论文速读：CABS-Efficient-and-Scalable-Model-Merging-via-Conflict-Aware

## 一句话总结
本文提出 **CABS+**，通过引入无梯度的自适应权重分配（AWA）策略替代原 CABS 的网格搜索，将合并时间复杂度从指数级降至多项式级，同时以非对称适应度函数缓解高损失任务主导优化问题，并在百亿参数大模型上实现了不到 AdaMerging 25% 的 GPU 显存占用与约 4 倍加速。此外，论文系统研究了模型可合并性（mergeability）的影响因素，并提出新指标 **RSS（Relative Synergy Score）** 用于量化模型合并潜力。

## 研究问题与动机
1. **参数冲突与知识干扰**：现有模型合并方法在合并多任务专家模型时，因参数空间中广泛存在的冲突与知识干扰，合并后模型在单个任务上的性能通常劣于对应的单任务模型。
2. **原 CABS 的网格搜索瓶颈**：CABS（ICML 2025）通过结构化剪枝减少冲突，但其确定缩放系数的网格搜索方法随任务数呈指数增长，难以在实际中扩展使用。
3. **优化目标偏置高表现任务**：CABS 的优化目标聚合全任务性能，导致系数容易被少数高表现任务主导，削弱其余任务性能，难以实现全面均衡的提升。
4. **现有数据驱动方法的显存瓶颈**：AdaMerging 等无测试数据梯度优化方法在十亿参数大模型上需同时保留基座模型与所有任务向量的计算图，显存开销随任务数线性增长，超出 V100 等消费级 GPU 容量，限制了实际部署可行性。

## 核心贡献（创新点）
1. **提出 AWA（Adaptive Weight Allocation）无梯度优化策略**：基于 CMA-ES 进化算法，通过边界约束搜索空间与对数加权均值更新，将时间复杂度从 O(S^T) 降至 O(K×G)，消除梯度计算与反向传播带来的显存开销。与网格搜索的本质区别在于其能在连续空间中高效探索而非离散枚举；与 AdaMerging 的本质区别在于无需构建计算图、仅需前向推理即可评估。
2. **设计非对称适应度函数（Asymmetric Fitness）**：通过相对于基线损失的归一化变化量构建惩罚项（α=100 惩罚性能下降，β=1 奖励性能提升），有效处理不同任务的损失尺度差异，避免优化被高损失任务主导。传统黑盒优化（如标准 CMA-ES）直接最小化绝对损失之和，易受尺度差异干扰。
3. **系统性可合并性实证研究与 RSS 指标**：首次系统量化任务异构性、数据分布差异、训练配置、模型架构与模型规模六大因素对合并性能的影响，并提出 RSS = (Score_merged − Score_ideal) / Score_ideal × 100% 作为标准化可合并性度量。这是对现有合并研究聚焦"如何合并"而忽视"合并什么"这一空白的填补。
4. **跨模态验证与效率突破**：在 ViT-B/32 视觉模型（6 任务）上验证了 CABS+ 的泛化能力（平均 82.50，较 CABS 提升 1.75%），并在 Mistral-7B 大规模实验中小于 25% 的 AdaMerging 显存与约 4× 速度优势。

## 方法详解
CABS+ 框架分为两阶段：冲突感知剪枝（CA）+ 平衡稀疏化（BS），以及自适应权重分配（AWA）。

**Phase 1：Conflict-Aware Sparsification（CA）**
- **顺序剪枝与掩码**：对任务向量 τ_A 执行 n:m 剪枝生成 mask_A，随后令 τ_B_remaining = τ_B ⊙ (1 − mask_A) 以消除与 τ_A 的参数重叠，再对 τ_B_remaining 独立剪枝生成 mask_B，得到正交化的剪枝向量 $\tilde{\tau}_A = \text{mask}_A \odot \tau_A$ 与 $\tilde{\tau}_B = \text{mask}_B \odot \tau_B$。当保留比例之和 >1 时不可避免重叠，此时采用类 TIES-Merging 的符号选择与参数平均策略。正交化保证合并更新范数中消除交叉项，各任务向量贡献可独立缩放。
- **Balanced Sparsification（BS）**：将权重矩阵划分为 m 个连续块，每块保留绝对值最大的 n 个参数。该策略与模型压缩目的下的 n:m 剪枝本质不同：BS 的目标是在合并前均匀分布各任务向量信息、降低参数冲突，最终合并模型仍为稠密模型，优先考虑合并性能而非推理加速。

**Phase 2：Adaptive Weight Allocation（AWA）**
- **无梯度采样**：从多元正态分布采样 K 个候选系数向量：$\lambda_k^{(g)} \sim m^{(g)} + \sigma^{(g)} \mathcal{N}(0, C^{(g)})$，初始协方差为单位阵 I，初始步长 σ=0.05。
- **边界约束**：定义可行域 Ω = {v ∈ R^D | l ≤ v_i ≤ u}，其中 l=0.1、u=2，对每个采样进行投影裁剪：$\lambda_{k,i}^{(g)} = \min(u, \max(l, [\lambda_k^{(g)}]_i))$。
- **非对称适应度函数**：
  - 基线损失：$L_{\text{base}}^{(t)} = \mathcal{L}_t(\theta(\lambda^{(0)}))$
  - 归一化相对变化：$\Delta_t(\lambda) = (L_t(\theta(\lambda)) - L_{\text{base}}^{(t)}) / L_{\text{base}}^{(t)}$
  - 非对称惩罚：$f_t(\lambda) = \alpha \cdot \Delta_t(\lambda)$ 若 $\Delta_t > 0$（性能下降），否则 $f_t = \beta \cdot \Delta_t(\lambda)$（α=100, β=1）
  - 总适应度：$F(\lambda) = \sum_t f_t(\lambda)$
- **种群选择与均值更新**：选取 Top μ=K/2 个体，以 log 加权均值更新分布中心：$m^{(g+1)} = \sum_{i=1}^{\mu} w_i \lambda_{i:K}^{(g)}$，其中 $w_i' = \ln(\mu+0.5) - \ln(i)$，归一化后满足 $\sum w_i = 1$。
- **步长与协方差更新**：利用演化路径 $p_\sigma$（共轭路径控制步长 σ）与 $p_c$（各向异性路径更新协方差 C）自适应调整搜索空间尺度与形状，使分布呈超椭球形沿相关方向高效下降，这是网格搜索无法实现的隐式学习任务间系数相关性。
- **早停机制**：loss 连续 6 代无显著变化时提前终止。关键超参：K=6，G=50，σ=0.05。
- **最终合并**：$\theta_{\text{merged}} = \theta_{\text{base}} + \sum_{t=1}^T \lambda_t^* \cdot \tilde{\tau}_t$。

**效率优势**：AWA 将 GPU 仅作前向推理模块使用，系数优化与矩阵运算在 CPU（Numpy）完成，GPU 显存始终仅保留基座模型+单层参数，与任务数无关（O(1)），而 AdaMerging 显存随任务数线性增长（O(n)）。

## 实验与结果
**数据集与模型**：27 个数据集、5 种模型，涵盖 LLM（Mistral-7B、Qwen2.5-7B-Instruct 及其微调变体）、小语言模型（RoBERTa、GPT-2）与视觉模型（ViT-B/32）。LLM 评测使用 LLM Leaderboard（6 任务）与 Open LLM Leaderboard 2（6 任务），用小模型使用 GLUE（7 任务）+ RACE + SQuAD，视觉用 6 任务。

**主要结果（大模型）**：
- Qwen2.5-7B 系列（Open LLM Leaderboard 2）：CABS+（fqFirst）AVG=45.09，较 CABS 提升 +2.37%，优于 WUDIMerging（44.09）与 AdaMerging（43.03）；CABS+（TFirst）AVG=45.10，较 CABS 提升 +2.86，逼近 Ideal Model（45.04）。
- Mistral-7B 系列（LLM Leaderboard）：CABS+ AVG=76.71，略低于 WUDIMerging（76.82），但整体高度可比；在多个子任务上显著超越 AdaMerging 与其他基线。

**主要结果（小模型）**：
- RoBERTa 4 任务合并：CABS+ 平均 82.40~82.47，较 CABS（81.64~81.70）提升约 +0.71~+0.80，超过 AdaMerging（80.52）与 WUDIMerging（81.79）。
- RoBERTa 6 任务合并：CABS+ 平均 70.49（+0.87），超 AdaMerging（66.56）与 WUDIMerging（68.19）。
- GPT-2 6 任务合并：CABS+ 平均 68.71（+1.56），超 AdaMerging（65.84）与 WUDIMerging（66.31）。

**跨模态验证**：ViT-B/32 六任务合并平均 82.50，优于 CABS（81.08）与 WUDIMerging（82.30）。

**效率对比（Table VIII）**：
- RoBERTa 4 任务：CABS+ GPU 显存 3.77 GB vs AdaMerging 6.69 GB，耗时 3 min vs 12 min；
- Mistral 2 向量：CABS+ GPU 显存 15.95 GB vs AdaMerging 66.70 GB（<25%），耗时 1h vs WUDIMerging 4h（约 4× 加速）。

**总体提升（Figure 3）**：相较 AdaMerging 综合提升 +16.97%，相较 WUDIMerging 提升 +12.93%，在 Qwen2.5、RoBERTa 多任务设置下提升尤为显著。

## 相关工作脉络
1. **Task Arithmetic（Ilharco et al., ICLR 2023）**：基础任务向量方法，以线性缩放方式合并任务向量；CABS+ 在此基础上引入结构化剪枝降冲突与进化优化定系数，解决其未经剪枝直接合并带来的性能劣化问题。
2. **TIES-Merging（Yadav et al., NeurIPS 2023）**：通过幅度剪枝去冗余+符号冲突消解；CABS+ 的核心区别是顺序掩码而非幅度剪枝，避免幅度剪枝保留的高重叠不均匀分布，并进一步引入 AWA 优化系数。
3. **DARE（Yu et al., ICML 2024）**：受 Dropout 启发的随机稀疏化；CABS+ 借鉴稀疏化思想但采用结构化 n:m 剪枝+顺序掩码，目标是从"加速推理"转向"合并友好"的均匀信息分布。
4. **AdaMerging（Yang et al., ICLR 2024）**：同样无需额外训练数据的无测试数据方法，通过梯度下降最小化测试熵；CABS+ 的关键区别在于梯度-free（零阶优化），显存从 O(n·L·T) 降至 O(1)，可在单卡 V100 上合并十亿参数模型。
5. **WUDIMerging（Cheng et al., ICML 2025）**：基于任务向量几何关系抑制层间干扰的逐层优化方法；CABS+ 在效率上大幅优于 WUDIMerging（4× 加速、更低显存），同时在多个设置下取得更高性能。
6. **CABS（原版，ICML 2025）**：论文前一版本，已使用顺序掩码+ n:m 剪枝；CABS+ 的核心增量是 AWA 替换网格搜索，解决指数时间复杂度与高表现任务主导问题。

## 局限性与未来方向
1. **搜索迭代次数的固定超参**：AWA 默认 G=50、K=6，对于任务数极多或冲突严重的场景可能需要更多迭代或早停触发不及时，自适应迭代次数设计仍有优化空间。
2. **固定边界约束（l=0.1, u=2）的通用性**：当前硬边界依赖经验设定，对于某些极端任务配置可能过于保守或宽松，缺乏自适应边界学习机制。
3. **仅验证同源基座模型合并**：所有实验基于同一基座微调出的任务向量，尚未探索跨架构/跨预训练模型的异构合并场景。
4. **RSS 指标的理论边界待完善**：RSS 为经验性指标，其对极端低/高理想分场景的归一化稳定性及与人类主观判断的相关性尚需更多实证支撑。
5. **论文自述的未来方向**：将 CABS+ 扩展至多模态与异构模型合并，进一步拓宽适用场景。

## 研究启发与可借鉴点
1. **无梯度进化优化用于合并系数搜索**：AWA 将 CMA-ES 引入模型合并领域是一个干净高效的范式，可作为通用插件嵌入任意基于任务向量的合并框架（如 Task Arithmetic、TIES），无需改动主干只需替换系数搜索模块。
2. **非对称适应度函数的设计思想**：相对基线变化的归一化惩罚机制对处理多任务损失尺度差异具有通用迁移价值，可推广至多任务蒸馏、联邦平均等同样存在尺度不对齐的场景。
3. **顺序掩码实现正交化的数学直觉**：通过掩码运算强制使剪枝后任务向量 Frobenius 内积为零，这一几何构造可直接复用于其他需减少参数重叠的模型融合场景。
4. **可合并性系统性实证研究的范式**：RSS 指标与六因素分析框架为后续合并类工作提供了可比对的评估基准，团队可在此基础上引入更多维度（如数据集规模、模型预训练域）构建更完整的 mergeability 理论。
5. **CPU-GPU 分离的内存优化策略**：AWA 将计算卸载至 CPU 而 GPU 仅作推理模块的做法，为大规模模型合并提供了可在消费级硬件上运行的工程模板，适合部署在内存受限的推理平台。

## 关键术语表
- **Model Merging（模型合并）**：在参数空间直接组合多个微调模型，无需额外训练或多任务数据即可构建统一多任务模型的范式。
- **Task Vector（任务向量）**：微调后模型参数与预训练基座模型参数之差，表征任务特异性知识更新。
- **Conflict-Aware Sparsification（CA，冲突感知稀疏化）**：通过顺序掩码剪枝使不同任务向量在参数空间中互不重叠，消除合并时的参数冲突。
- **Balanced Sparsification（BS，平衡稀疏化）**：采用 n:m 块级剪枝确保各任务向量信息在整网中均匀分布，而非集中于少数层。
- **Adaptive Weight Allocation（AWA，自适应权重分配）**：基于 CMA-ES 的无梯度进化搜索策略，自动优化各任务向量的缩放系数。
- **Asymmetric Fitness Function（非对称适应度函数）**：对任务性能下降施加 100 倍惩罚、对性能提升仅施加 1 倍惩罚，防止优化被高损失任务主导。
- **Relative Synergy Score（RSS，相对协同得分）**：(Score_merged − Score_ideal) / Score_ideal × 100%，用于量化合并性能的相对增益/损失。
- **Model Mergeability（模型可合并性）**：源模型之间可通过合并获得理想综合性能的内在可行程度，受学习率、训练轮数、任务异构性、数据分布、架构与规模等六大因素影响。

## 可复现要素
- **数据集**：GLUE（CoLA/MNLI/MRPC/QNLI/QQP/RTE/SST-2）、RACE、SQuAD、LLM Leaderboard（ARC/HellaSwag/MMLU/TruthfulQA/Winogrande/GSM8K）、Open LLM Leaderboard 2（IFEval/BBH/MATH/GPQA/MUSR/MMLU-Pro）、ViT 六视觉任务；**大部分公开**，部分checkpoint 来自 Hugging Face 与 FusionBench 仓库。
- **代码开源**：是，已匿名开源，地址：https://anonymous.4open.science/r/CABS-Plus-70C1
- **权重开源**：基座模型（Mistral-7B-v0.1、Qwen2.5-7B-Instruct、RoBERTa、GPT-2、ViT-B/32）及微调 checkpoint 均在 Hugging Face 公开。
- **关键超参**：AWA — K=6，G=50，σ=0.05，l=0.1，u=2，早停6代；BS — 小模型稀疏度 0.90，大模型 0.75；评测使用 lm-evaluation-harness v0.4.0，batch_size=auto。
- **硬件**：V100 32GB；论文未提及多卡扩展方案。

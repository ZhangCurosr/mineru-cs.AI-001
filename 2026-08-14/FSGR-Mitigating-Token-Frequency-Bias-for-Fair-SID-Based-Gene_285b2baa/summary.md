---
title: "FSGR-Mitigating-Token-Frequency-Bias-for-Fair-SID-Based-Gene"
source: https://arxiv.org/pdf/2608.12845v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:28:17"
field: "生成式推荐系统公平性"
keywords: ["Token Frequency Bias", "Semantic ID", "Generative Recommendation", "Fairness", "Optimal Transport", "Hierarchical Frequency Calibration"]
innovations: ["首次定义并量化SID生成式推荐中的Token Frequency Bias公平性问题", "提出FSGR框架联合优化SID构建（平衡码本利用）与推荐训练（分层频率校准）", "设计层次感知频率校准HFC，适配SID token的层次语义结构进行差异化去偏"]
benchmarks: ["Amazon Luxury Beauty", "Amazon Industrial and Scientific", "Amazon Software"]
---

# 论文速读：FSGR-Mitigating-Token-Frequency-Bias-for-Fair-SID-Based-Gene

## 一句话总结
论文首次揭示了基于语义ID（SID）的生成式推荐系统中存在的 **Token Frequency Bias（词频偏差）** 问题，并提出 **FSGR** 公平优化框架，通过平衡语义量化（BSQ）与分层频率校准（HFC）在提升 SID token 曝光公平性的同时，保持了具有竞争力的推荐精度。

## 研究问题与动机
- **核心问题**：在 SID 生成式推荐中，高频 SID token 被系统性过度预测，低频 token 被低估，导致推荐曝光向头部 token 对应的商品类别集中，长尾类别被边缘化。
- **现有方法不足**：
  1. 现有 SID 构建方法（如 LC-Rec、QuaSID）主要关注改善码本质量和减少 token 冲突，忽视了 token 频率不平衡对下游推荐公平性的影响。
  2. 直接移植自 NLP 的 LLM 去偏方法（如 MiLe、WAKL）对 SID 效果不佳，因为 SID token 具有层次语义结构，不同层编码不同粒度的语义，需要差异化的去偏强度。
  3. 传统推荐公平性方法（重排序、对抗学习、正则化）作用于推荐列表或 Item 表征层面，无法直接应用于由自回归 SID 生成决定曝光的生成式推荐范式。

## 核心贡献（创新点）
1. **首次定义并量化 Token Frequency Bias**：系统揭示了 SID 构建阶段码本不平衡与推荐训练阶段 popularity bias 及 MLE 目标共同导致的公平性问题，填补了该领域空白。
2. **提出端到端的公平优化框架 FSGR**：与以往仅优化下游推荐或仅改进码本质量的方法不同，FSGR 联合优化 SID 生成（构造平衡的语义空间）和推荐训练（缓解生成阶段的频率偏差）。
3. **设计 Balanced Semantic Quantization（BSQ）模块**：通过最优传输分配优化（OTA）与双标准重锚定机制（DCR）主动平衡码本利用率，从源头减少 SID 表示空间的结构性偏差，这是与仅关注码本利用率指标的前序工作（如 RT、QuaSID）的本质区别。
4. **提出分层频率校准（HFC）策略**：采用两阶段训练，在语义对齐预训练后，对 SID 预测 logits 引入基于层号的自适应校准温度（$\tau_l = l/L$），比均匀去偏或反向校准更能兼顾语义保真度与公平性。

## 方法详解
FSGR 框架包含两个核心组件：

**1. 平衡语义量化（BSQ）**
- **最优传输分配优化（OTA）**：在 RQ-VAE 的每一量化层，将样本到码本的软分配矩阵 $Q$ 与通过 Sinkhorn-Knopp 算法求解的熵正则化最优传输计划 $P^*$（设定样本和码本的边际分布均为均匀分布）进行 KL 散度对齐：$\mathcal{L}_{OTA} = D_{KL}(P^* \| Q)$。总损失为 $\mathcal{L}_{total} = \mathcal{L}_{rq} + \lambda_o \mathcal{L}_{OTA}$。
- **双标准重锚定机制（DCR）**：定期识别“死码”（使用计数 $n_k < \delta$ 的码本向量）。策略 A（OT 成本空洞检测）：将死码重新锚定到 OT 成本最高（即处于语义空间稀疏区域）的样本编码向量上，以填充几何空洞。策略 B（密度感知需求重锚定）：根据码本利用频率构建采样分布，将剩余死码重新锚定到高密度区域的代表性样本编码向量上，以增加拥挤区域的表示容量。

**2. 两阶段推荐训练**
- **阶段一：语义对齐预训练**：使用标准交叉熵损失 $\mathcal{L}_{CE} = -\sum_{l=1}^{L} \log P(s^l | \mathcal{H}, s^{<l})$ 训练 LLM，使其学习用户行为历史与 SID 序列之间的语义对应关系，此阶段不引入任何频率约束。
- **阶段二：分层频率校准（HFC）公平微调**：
  - 根据训练集实证 token 分布计算对数频率先验：$\mathbf{b}_l = -\log(\mathbf{f}_l + \epsilon)$。
  - 对第 $l$ 层的预测 logits $\mathbf{z}_l$ 进行校准：$\hat{\mathbf{z}}_l = \mathbf{z}_l + \tau_l \mathbf{b}_l$，其中校准温度按层号递增 $\tau_l = l/L$（深层捕获细粒度语义，故施加更强校准；浅层编码粗粒度类别，保持稳定）。
  - 微调阶段总损失为：$\mathcal{L} = \mathcal{L}_{non-SID} + \lambda_h \mathcal{L}_{SID}$，其中 $\mathcal{L}_{non-SID}$ 对非 SID 位置使用原始 logits，$\mathcal{L}_{SID}$ 对 SID 位置使用校准后 logits $\hat{\mathbf{z}}$。

## 实验与结果
- **数据集**：Amazon Review 的三个子集：Luxury Beauty、Industrial and Scientific、Software。
- **评估基线**：
  - SID 生成方法：vanilla RQ-VAE、RT、QuaSID。
  - LLM 去偏方法：MiLe、WAKL。
- **骨干模型**：TIGER（Transformer-based）、Llama3.1-8B、Qwen3-8B。
- **评估指标**：推荐精度（Recall@K、NDCG@K）与公平性（基于生成 SID token 频率分布计算的 Gini 系数，值越低越公平）。
- **主要结果**：
  - **公平性显著提升**：在三个数据集和三种骨干模型上，FSGR 均取得了最低的 Gini 分数。相较于最佳基线，**平均 Gini 公平性提升超过 20%**。
  - **精度保持竞争力**：在绝大多数设置下，FSGR 的 Recall 和 NDCG 与最优基线相当，未出现显著下降。
  - **消融验证**：移除 BSQ 或 HFC 任一组件均会导致公平性下降，证明两者互补；分层温度策略（HFC）在精度-公平性权衡上优于反向校准和固定温度策略。
  - **码本利用分析**：Our-BSQ 在三个数据集上均实现了最高的码本覆盖率（Industrial 达 100%）和最低的 Gini 系数，证实了码本利用的均衡性。

## 相关工作脉络
1. **LC-Rec / TIGER（Rajput et al., 2023; Zheng et al., 2024）**：开创性的 SID 生成式推荐工作，使用 RQ-VAE 构建离散 token 序列。本文与其定位差异：LC-Rec 追求推荐精度与语义泛化，本文首次将公平性（token 频率偏差）作为核心研究对象。
2. **QuaSID（Hu et al., 2026）**：通过 Hamming 距离引导和对比学习改善 SID 质量、减少碰撞。本文认为 QuaSID 等同样聚焦表征质量，未触及因 token 频率不均衡引发的公平性退化问题。
3. **传统推荐公平性方法（Zehlike et al., 2017; Zhang et al., 2023; Chang et al., 2024）**：作用于重排序、Item 表征或损失正则化层面。本文指出这些方法无法直接迁移，因为生成式推荐的曝光机制完全由自回归 SID 生成决定，而非显式排序分数。
4. **LLM 文本去偏方法（MiLe, Su et al., 2024; WAKL, Shrestha & Srinivasan, 2025）**：针对自然语言生成的词频偏差设计。本文发现直接套用效果次优，因为 SID token 具有层次语义结构，需要层级的差异化校准，而非 uniform 去偏。
5. **最优传输在表示学习中的应用（如 RT, Fifty et al., 2025）**：利用 Sinkhorn 算法解决码本使用不均。本文继承并扩展了这一思想，不仅用于最后一层均匀映射（如 LC-Rec），还将其引入每层量化过程（OTA），并辅以 DCR 机制处理死码和局部空洞。

## 局限性与未来方向
- **局限性**：
  1. 公平性优化可能在极少数设置下对推荐精度造成轻微负面影响（如部分 G@10 设置中 Recall 略降）。
  2. 实验仅在 Amazon 三个相对中小规模的文本型商品数据集上进行，未验证于更大规模或更复杂的 multimodal 商品数据。
  3. 两阶段训练增加了训练流程的复杂性，第一阶段预训练的质量会直接影响第二阶段微调的效果。
- **未来方向**（论文自述）：
  1. 将 FSGR 扩展至更先进的生成式推荐架构（如结合多模态特征或更复杂的因果推理）。
  2. 探索自适应的频率感知优化策略，以更精细地平衡公平性与准确性。

## 研究启发与可借鉴点
1. **解耦训练策略的通用性**：将“语义/结构学习”与“公平性/偏差修正”解耦为两个阶段，可有效避免因多目标竞争导致的优化不稳定。此思想可迁移至其他需要兼顾性能与公平性的生成式任务。
2. **层次结构感知的去偏设计**：HFC 中根据 SID 层的语义粒度动态调整校准强度（$\tau_l = l/L$）是一项精巧设计。类似的分层差异化处理思路可用于任何具有层次编码结构的生成模型（如分层 VQ-VAE、树状分类器）。
3. **最优传输用于表征空间平衡**：OTA 将均匀边际分布约束引入最优传输问题，以鼓励码本均匀利用。这一思路可推广到其他基于码本的离散表示学习任务（如图像分类、异常检测）中以避免码本坍塌。
4. **公平性评估指标的选择**：采用基于生成 token 频率分布的 Gini 系数直接量化 token-level 公平性，为生成式推荐系统的公平性评估提供了可直接度量、可优化的新视角，区别于传统的列表级公平性指标。
5. **死码复活机制的借鉴**：DCR 中基于 OT 成本和采样密度的双标准重锚定策略，为 VQ-VAE 类模型中常见的死码问题提供了一种几何感知的解决方案。

## 关键术语表
- **Semantic ID (SID)**：通过残差向量量化（RQ-VAE）将商品特征映射为离散的 token 序列，作为商品的全局唯一标识符，供生成式推荐模型自回归预测。
- **Token Frequency Bias**：SID 生成式推荐中系统性过预测高频 SID token、低估低频 SID token 的偏差现象，导致推荐曝光在物品类别间分布不均。
- **Balanced Semantic Quantization (BSQ)**：FSGR 中用于 SID 构建的模块，通过 OTA 和 DCR 机制促进码本各 token 的均衡利用，构建更公平的语义表示空间。
- **Optimal Transport-based Assignment Optimization (OTA)**：在每一量化 mini-batch 内构建样本到码本的最优传输问题（设定均匀边际），并通过 KL 散度约束软分配矩阵跟随最优传输计划，以实现码本利用均衡。
- **Dual-Criteria Re-anchor (DCR)**：定期识别并重新初始化死码（极少使用的码本向量），通过空洞填充（基于高 OT 成本）和密度感知（基于采样分布）两种策略分别优化稀疏区和密集区。
- **Hierarchical Frequency Calibration (HFC)**：在推荐训练的第二阶段，根据训练集 token 频率先验对各层 SID 预测 logits 进行加法校准，且校准强度随层号递增（$\tau_l = l/L$），以适应 SID 的层次语义。
- **Gini Coefficient (Gini)**：衡量分布不平等程度的指标，值域 [0,1]。本文用于计算生成 SID token 频率分布的不均匀性，值越低表示 token 曝光越公平。
- **Maximum Likelihood Estimation (MLE)**：标准生成模型训练目标，最大化训练数据中出现概率的乘积。其固有特性会放大高频信号的预测倾向，是导致 Token Frequency Bias 的成因之一。

## 可复现要素
- **数据集**：Amazon Review 公开数据集（Luxury Beauty, Industrial and Scientific, Software），需自行下载并遵循论文预处理流程（移除交互少于5次的用户/物品，截断用户序列至长度20）。
- **代码/权重**：**论文未明确声明代码是否开源**（注：arXiv 论文通常会在附录或脚注提供匿名代码链接，需查看原文完整版确认）。权重文件未提及开源。
- **关键超参数**：
  - RQ-VAE：4 层，每层码本大小 256。
  - OTA：温度系数 $\tau_q$、权重 $\lambda_o$（按 warm-up 调度递增）。
  - DCR：死码阈值 $\delta = 1$。
  - HFC：层校准温度 $\tau_l = l/L$，损失权重 $\lambda_h$。
  - 优化器：AdamW-8bit，LoRA 微调 LLM。
  - *详细超参数见论文补充材料（supplementary material）*。

---
title: "LOOKBACK-Where-and-How-to-Score-LVLM-Responses-via-Visual-Re"
source: https://arxiv.org/pdf/2608.11847v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:35:18"
field: "多模态大模型可靠性与幻觉评估"
keywords: ["LVLM", "response scoring", "visual hallucination", "best-of-n selection", "self-evaluation", "visual lookback", "training-free"]
innovations: ["提出 visual lookback score 量化 token 级视觉引用，弥补输出空间置信度的图像不敏感性缺陷", "设计无需训练的响应评分方法 LOOKBACK，通过 token 级校准与响应级加权聚合实现高效 Best-of-N 选择"]
benchmarks: ["VQAv2", "CHAIR", "AMBER", "HallusionBench"]
---

# 论文速读：LOOKBACK: Where and How to Score LVLM Responses via Visual Reference Usage

## 一句话总结
针对 LVLM 响应评分中现有置信度指标对图像不敏感的问题，本文提出 LOOKBACK——一种无需训练的评分方法，通过引入 token 级"视觉回顾分数"（visual lookback score）来校准输出空间置信度，在 Best-of-N 选择任务中显著提升响应质量。

## 研究问题与动机
1. **视觉幻觉问题**：LVLM 会产生流畅但未被图像支持的响应（视觉幻觉），使得响应评分比纯文本 LLM 更困难。
2. **现有置信度方法不足**：诊断实验表明，删除输入图像后，Self-Certainty（SC）等基于输出空间置信度的评分选择结果几乎不变（top-1 一致率高达 0.36–0.64，远高于随机基线 0.04），说明置信度主要捕获的是文本合理性而非图像对齐度。
3. **图像条件性与图像敏感性存在差距**：尽管 LVLM 生成时以图像为条件，但其输出空间置信度对图像存在与否并不敏感。
4. **需要轻量且无需外部验证器的方案**：现有 LVLM 评分方法多依赖外部多模态奖励模型或额外训练，限制了适用性。

## 核心贡献（创新点）
1. **首次系统诊断了 LVLM 输出空间置信度的图像不敏感性**：通过 SC w/ image vs. w/o image 的选择一致性分析，揭示了 confidence-based scoring 在 LVLM 中的根本缺陷——无法可靠检测视觉幻觉。
2. **提出了视觉回顾分数（visual lookback score）这一 token 级指标**：通过计算每个生成 token 对输入视觉 token 的注意力比例，轻量地量化"模型在生成该 token 时多大程度上查看了图像"。
3. **设计了 LOOKBACK 训练自由响应评分框架**：在 token 级将输出置信度与视觉回顾分数结合（lookback-calibrated token score），在响应级根据视觉相关性分布进行加权聚合，无需额外训练、外部验证器或推理步骤。
4. **在四个基准和三个 LVLM 上验证了有效性**：LOOKBACK 在 Best-of-N 选择中一致超越语言和视觉侧基线，平均得分提升显著（相对随机选择提升 4.97%）。

## 方法详解

### 整体流程
给定图像 $I$、文本查询 $\mathbf{x}$ 和编码后的视觉 token $\mathbf{v}$，LVLM 自回归生成响应 $\mathbf{y} = (y_1, \dots, y_T)$。通过随机解码采样 $N$ 个候选响应，LOOKBACK 无需额外推理即从内部信号计算每个候选的响应级分数 $S(\mathbf{y}|\mathbf{x}, \mathbf{v})$，选取最高分者。

### 视觉回顾分数（Visual Lookback Score）
对于第 $t$ 个生成 token $y_t$，其因果上下文 $C_t$ 包含文本查询 token、视觉 token 和此前输出 token。令 $\mathcal{P}_v \subset C_t$ 为视觉 token 的位置集合，$a_{t,k}^{(\ell,h)}$ 为层 $\ell$、头 $h$ 中查询位置到上下文位置 $k$ 的注意力权重：

$$
A_t = \frac{1}{LH} \sum_{\ell=1}^{L} \sum_{h=1}^{H} \frac{\sum_{k \in \mathcal{P}_v} a_{t,k}^{(\ell,h)}}{\sum_{k \in C_t} a_{t,k}^{(\ell,h)}}
$$

$A_t \in [0,1]$ 表示第 $t$ 步生成时分配到视觉 token 的注意力比例，越大说明该 token 越依赖图像。

### Lookback-Calibrated Token Score
结合 token 级置信度 $p_t$ 和视觉回顾分数 $A_t$：

$$
u_t = \log(p_t) + \alpha \log(A_t) = \log(p_t \cdot A_t^{\alpha})
$$

其中 $\alpha$ 控制视觉校准强度。该形式等价于 Product-of-Experts：高置信度但低视觉回顾的 token 将被降分。

### 响应级聚合（Visual Relevance Weighting）
基于熵正则化的视觉相关性分布最大化问题，得到闭式解：

$$
q_\lambda(t) = \frac{A_t^{\lambda}}{\sum_{j=1}^{T} A_j^{\lambda}}
$$

最终 LOOKBACK 响应分数为：

$$
S(\mathbf{y}|\mathbf{x}, \mathbf{v}) = \sum_{t=1}^{T} \frac{A_t^{\lambda}(\log(p_t) + \alpha \log(A_t))}{\sum_{j=1}^{T} A_j^{\lambda}}
$$

即对 token 级校准分数按视觉相关性分布做加权平均（等价于视觉相关性加权的几何平均的对数）。超参数 $\lambda$ 控制聚合的尖锐程度：$\lambda=0$ 退化为均匀平均，$\lambda$ 越大越聚焦于高视觉回顾的 token。

## 实验与结果

### 数据集与模型
- **四个基准**：VQAv2（VQA 准确率）、CHAIR（对象幻觉 F1）、AMBER（多维度幻觉 F1）、HallusionBench（GPT-evaluated correctness）
- **三个 LVLM**：LLaVA-1.5-7B、Qwen2.5-VL-7B、InternVL3-8B
- **解码设置**：nucleus sampling (temperature=1.2, top-p=0.9)，每个输入生成 N=25 个候选

### 对比基线
- **语言侧**：Self-Certainty (SC)、Universal Self-Consistency (USC)
- **视觉侧**：CLIP-Score、VAUQ
- **随机选择**（Random）

### 主要结果
LOOKBACK 在全部三模型的跨基准平均性能最优：

| 模型 | LOOKBACK Avg (N=25) | 次优 Baseline Avg | 相对 Random 提升 |
|---|---|---|---|
| LLaVA-1.5-7B | **63.67** | VAUQ 62.13 | +6.59 |
| Qwen2.5-VL-7B | **69.91** | VAUQ 68.17 | +5.96 |
| InternVL3-8B | **72.27** | VAUQ 71.53 / USC 71.58 | +3.82 |

- 整体平均得分从 65.37% 提升至 68.62%，相对随机选择提升 **4.97%**。
- 在 HallusionBench 上 LOOKBACK 对 Qwen2.5-VL-7B 的提升尤为显著（N=25 时 61.93 vs. 次优 58.21，+3.72）。
- Scaling 实验：随 N 增大，LOOKBACK 持续稳定领先，而 CLIP-Score 和 VAUQ 收益不明显。

### 消融与效率
- 消融显示 token 级校准（$\alpha$）与响应级加权（$\lambda$）互补，$\alpha$ 与 $\lambda$ 需协同调优。
- 评分开销仅为毫秒级/响应，远低于 VAUQ 和 USC（后者需扰动或额外推理）。

## 相关工作脉络
1. **LLM 输出空间评分**：Self-Certainty (Kang et al., 2025) 基于 token 分布与均匀分布的 KL 散度进行 BoN 选择；LOOKBACK 保留了此类轻量内源信号的优点，但补充了图像敏感性。
2. **多模态奖励模型**：如 VisualPRM (Wang et al., 2025)、VrPRM (Chen et al., 2025) 等需要额外训练和标注；LOOKBACK 无需任何训练即可利用模型内部注意力。
3. **VAUQ (Park et al., 2026)**：通过对视觉注意力加权重的掩码扰动来估计不确定性；LOOKBACK 直接使用原始注意力权重，无扰动开销，且更针对性地捕捉 token 级的视觉引用。
4. **视觉幻觉检测**：如 Halc (Chen et al., 2024)、MITIGATING HALLUCINATION (Leng et al., 2024) 等关注生成过程干预；LOOKBACK 关注 post-hoc 响应评分，可在不修改生成过程的前提下提升选择质量。
5. **CLIP-Score (Hessel et al., 2021)**：通过外部 CLIP 编码器计算图文余弦相似度；LOOKBACK 完全基于模型内部信号，无需额外编码器。
6. **Universal Self-Consistency (Chen et al., 2023)**：通过多次提示让模型自我验证；LOOKBACK 无需额外推理步骤，开销更低。

## 局限性与未来方向
1. **需要访问内部注意力权重**：限制了在 black-box LVLM API 上的应用。
2. **视觉回顾仅为代理指标**：高视觉注意力不等于事实正确性——模型可能强关注图像但仍生成错误响应。
3. **模型特异性依赖**：注意力分布和校准方式因模型架构而异，跨模型泛化需谨慎。
4. **当前仅评估短响应场景**：长文多模态推理、多图/视频输入、复杂 grounding 场景下的泛化待验证。
5. **未来可扩展至其他源导向生成**：如 RAG 中检索文档引用、工具调用输出引用等场景的评分。

## 研究启发与可借鉴点
1. **"图像条件性 ≠ 图像敏感性"的诊断思路**：通过 image-removed 对照实验揭示评分信号的本质缺陷，这一诊断范式可直接迁移到其他多模态模型评估场景中。
2. **注意力比例作为视觉引用的轻量代理**：用 causal attention 中视觉 token 的比例来衡量"视觉参考使用度"，无需额外模型，这一思路可推广到任何具有多源输入的生成场景（如 RAG 中检索 chunk 的引用）。
3. **熵正则化相关性加权的设计**：通过最大熵优化得到闭式解 $q_\lambda(t) \propto A_t^\lambda$，将软选择问题转化为易于调参的幂律分布，该技巧可复用。
4. **Token 级 complementarity 分析与 Aggregation 分层的两阶段设计**：先在 token 级融合互补信号，再在响应级按语义重要性加权，这种分层设计兼顾了细粒度与全局一致性。
5. **无需训练的 test-time 增强思路**：在已有 LLM/LVLM 基础上，仅利用内部信号即可实现性能提升，这对资源受限场景具有高实用价值。

## 关键术语表
**Best-of-N (BoN)**：从同一输入生成的 N 个候选响应中，通过评分函数选出最优响应的推理时策略。
**Visual Lookback Score ($A_t$)**：每个生成 token 预测时分配到视觉 token 的注意力比例，衡量该 token 对图像的依赖程度。
**Lookback-Calibrated Token Score ($u_t$)**：结合 token 级输出置信度 $p_t$ 与视觉回顾分数 $A_t$ 的校准分数，形式为 $\log(p_t) + \alpha \log(A_t)$。
**Visual Relevance Distribution ($q_\lambda$)**：基于熵正则化优化的 token 级权重分布，使高视觉回顾的 token 获得更大聚合权重。
**Self-Certainty (SC)**：基于 token 预测分布与均匀分布的 KL 散度来衡量模型输出置信度的 BoN 评分方法。
**Visual Hallucination**：LVLM 生成流畅但未被图像支持的内容（如不存在的对象、属性或关系）。
**Product-of-Experts Interpretation**：LOOKBACK 的 token 级分数可解释为输出空间置信度与视觉回顾分数两个 "expert" 的乘积形式。
**Image-conditioning vs. Image-sensitivity**：前者指生成过程以图像为条件，后者指评分结果随图像变化而变化；论文发现两者存在显著 gap。

## 可复现要素
- **数据集**：VQAv2、CHAIR、AMBER、HallusionBench（均为公开数据集）；论文未声明是否重新托管数据。
- **代码/权重**：论文未明确声明代码开源（全文无 GitHub 链接或代码可用性声明）。
- **关键超参**：$\alpha$ 和 $\lambda$ 按模型设置：LLaVA-1.5-7B $(\alpha, \lambda)=(7.0, 1.5)$；Qwen2.5-VL-7B $(0.5, 1.25)$；InternVL3-8B $(0.25, 1.25)$；解码参数 temperature=1.2, top-p=0.9, N=25。
- **硬件**：LLaVA-1.5-7B 使用 NVIDIA RTX A6000；其余使用 NVIDIA H200。

---
title: "LOOKBACK-Where-and-How-to-Score-LVLM-Responses-via-Visual-Re"
source: https://arxiv.org/pdf/2608.11847v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:36:33"
field: "多模态大模型评估与可靠性"
keywords: ["LVLM", "响应评分", "视觉幻觉", "Best-of-N", "视觉回溯", "无需训练评分", "多模态大模型"]
innovations: ["揭示LVLM输出置信度对图像不敏感的缺陷并提出视觉回溯分数作为互补信号", "提出LOOKBACK框架，通过token级看回校准与响应级视觉相关性加权实现无需训练的视觉感知响应评分", "证明视觉回溯分数与输出置信度在词性层面的互补性，并提供熵正则化软选择聚合方法"]
benchmarks: ["VQAv2", "CHAIR", "AMBER", "HallusionBench"]
---

# 论文速读：LOOKBACK: Where and How to Score LVLM Responses via Visual Reference Usage

## 一句话总结
论文提出 LOOKBACK，一种无需训练的 LVLM 响应评分方法，通过将 token 级输出置信度与视觉回溯分数（visual lookback score）相结合，在 Best-of-N 响应选择任务中显著提升 LVLM 的视觉对齐评估效果。诊断分析揭示：现有基于置信度的评分对图像输入不敏感，而视觉回溯分数与之形成 token 级别的互补。

## 研究问题与动机
- **LVLM 存在视觉幻觉问题**：模型能生成流畅且看似合理的响应，但包含图像中未出现的内容（对象、属性、关系等），传统文本级幻觉评估不足以覆盖此问题。
- **输出空间置信度对图像不敏感**：作者通过诊断实验发现，移除输入图像后 SC（Self-Certainty）评分的分布几乎不变，Top-1 一致率高达 0.36–0.64（随机基线仅 0.04），表明置信度主要捕捉文本流畅性而非视觉依据。
- **缺乏轻量级、无辅助的视觉感知识别信号**：现有 LVLM 评分方法依赖外部多模态奖励模型或额外推理，成本高；而纯输出空间信号无法区分"视觉上支持"与"语言上 plausible"的响应。
- **视觉回溯与置信度存在 token 级互补**：词性分析显示，视觉回溯分数在名词/形容词等内容词上更高，而 SC 在功能词上更高，两者选择的最佳响应一致率最低仅 0.01，说明可从不同维度捕捉响应质量。

## 核心贡献（创新点）
1. **揭示 LVLM 输出置信度的图像不敏感性**：通过 SC w/ image vs. w/o image 的诊断实验，证明现有置信度评分与图像条件解耦，为视觉感知的响应评分研究提供了关键动机。
2. **提出 LOOKBACK 评分框架**：将 token 级视觉回溯分数与输出置信度结合，并在响应级通过熵正则化的视觉相关性分布进行聚合，无需外部模型、额外训练或推理pass。
3. **定义视觉回溯分数（visual lookback score）**：基于模型内部注意力权重计算每个生成 token 对视觉 token 的注意力比例，作为轻量级的视觉参考使用代理。
4. **系统性实验验证**：在 4 个基准（VQAv2、CHAIR、AMBER、HallusionBench）和 3 个 LVLM（LLaVA-1.5-7B、Qwen2.5-VL-7B、InternVL3-8B）上，LOOKBACK 均取得最高模型级平均性能。

## 方法详解
**视觉回溯分数（Visual Lookback Score）**：对于第 t 个生成 token，计算其在所有层 ℓ 和所有注意力头 h 中，注意力权重分配到视觉 token 位置的比例：

$$A_t = \frac{1}{LH}\sum_{\ell=1}^{L}\sum_{h=1}^{H}\frac{\sum_{k\in\mathcal{P}_v}a_{t,k}^{(\ell,h)}}{\sum_{k\in C_t}a_{t,k}^{(\ell,h)}}$$

其中 $\mathcal{P}_v$ 为视觉 token 位置集合，$C_t$ 为完整因果上下文。该分数直接从生成前向传播的注意力权重中获得，无需额外计算。

**看回校准的 Token 分数（Lookback-Calibrated Token Score）**：将 token 概率 $p_t$ 与视觉回溯分数 $A_t$ 结合：

$$u_t = \log(p_t) + \alpha\log(A_t) = \log(p_t \cdot A_t^\alpha)$$

超参数 $\alpha$ 控制视觉校准强度；当 $\alpha=0$ 时退化为纯输出置信度。

**视觉相关性分布（Visual Relevance Distribution）**：通过熵正则化最大化问题求解最优权重分布：

$$q_\lambda = \arg\max_{q\in\Delta_T}\left[\lambda\mathbb{E}_{t\sim q}[\log(A_t)] + H(q)\right]$$

闭式解为 $q_\lambda(t) = \frac{A_t^\lambda}{\sum_j A_j^\lambda}$，$\lambda$ 控制分布的集中程度。

**最终响应评分（LOOKBACK）**：

$$S(\mathbf{y}|\mathbf{x},\mathbf{v}) = \sum_{t=1}^{T}\frac{A_t^\lambda(\log(p_t)+\alpha\log(A_t))}{\sum_{j=1}^{T}A_j^\lambda}$$

等价于视觉相关性加权的 token 级乘积得分的几何平均，避免了简单求和对响应长度的依赖。

## 实验与结果
- **数据集与模型**：VQAv2、CHAIR、AMBER、HallusionBench 四个基准；LLaVA-1.5-7B、Qwen2.5-VL-7B、InternVL3-8B 三个模型；N=5 和 N=25 两种候选预算。
- **基线方法**：语言侧（SC、USC）、视觉侧（CLIPScore、VAUQ）、Random。
- **主要结果**：LOOKBACK 在所有模型-基准组合中均取得最高平均分，相对 Random 提升约 4.97%。例如 LLaVA-1.5-7B 在 N=25 时平均从 60.16% 提升至 63.67%；Qwen2.5-VL-7B 从 66.95% 提升至 69.91%；InternVL3-8B 从 69.00% 提升至 72.27%。
- **最强结果**：在 Qwen2.5-VL-7B 上，LOOKBACK 于 HallusionBench（N=25）达到 61.93%，超越最佳基线 SC（58.04%）约 3.9 个百分点。
- **消融与扩展**：token 级校准与响应级加权组件互补；随 N 增大 LOOKBACK 性能稳定增长，优于其他基线；评分开销远低于 VAUQ 和 USC。

## 相关工作脉络
- **Self-Certainty (SC, Kang et al., 2025)**：基于 token 分布与均匀分布的 KL 散度衡量置信度，在 LLM 中表现优异，但本文证明其对 LVLM 的视觉依据不敏感。
- **Universal Self-Consistency (USC, Chen et al., 2023)**：通过多次提示模型自身进行自我一致性选择，效果依赖模型自身的选择能力，跨模型稳定性不如 LOOKBACK。
- **CLIPScore (Hessel et al., 2021)**：利用预训练 CLIP 编码器计算文本-图像余弦相似度，需外部模型且无法区分 token 级别的视觉依据。
- **VAUQ (Park et al., 2026)**：通过掩码固定比例视觉注意力并计算熵估计不确定性，需要扰动式额外推理，计算成本显著高于 LOOKBACK。
- **VrPRM / VisualPRM**：基于过程奖励模型的 LVLM 评估方法，需任务特定标注和额外训练，LOOKBACK 完全无需。
- **Output-space scoring in LLMs**：本文继承"纯模型内部信号"的思路，但首次在 LVLM 中引入显式视觉参考使用度量以弥补纯文本置信度的不足。

## 局限性与未来方向
- **需访问内部注意力权重**：无法直接应用于黑盒 LVLM API。
- **视觉回溯仅为代理指标**：强注意力不代表事实正确性，模型可能高度关注图像但仍生成错误响应。
- **跨模型架构可靠性待验证**：注意力分布及校准方式因模型而异，泛化性需进一步检验。
- **当前仅评估简短响应**：未覆盖长形多模态推理、多图/视频输入等复杂场景。
- **未来方向**：可扩展至检索增强生成（RAG）中的文档依据感知评分、工具调用输出评估、指令遵循等源感知生成任务。

## 研究启发与可借鉴点
- **诊断驱动的方法设计**：先通过图像移除实验诊断现有方法的本质缺陷（置信度对图像不敏感），再针对性地引入互补信号（视觉回溯），这一研究路径值得借鉴。
- **Token 级互补信号融合**：证明两种信号在 POS 层面呈现相反模式（视觉词 vs. 功能词），为多模态内部信号融合提供了可复用的分析框架。
- **熵正则化软选择机制**：用 $q_\lambda(t) \propto A_t^\lambda$ 实现加权聚合而非硬选择，在聚焦与多样性间取得平衡，可迁移至其他需要软分配权重的场景。
- **无额外训练的高效评分**：完全利用生成过程中的内部信号，无需外部模型或额外推理 pass，为资源受限场景提供了实用方案。
- **跨域通用性**：框架可自然推广至任何"响应需锚定输入某一部分"的任务（如 RAG 中的检索文档、工具调用的输出），具有较广泛的适用潜力。

## 关键术语表
- **LVLM (Large Vision-Language Model)**：将视觉感知与语言生成整合的大规模多模态模型，能从图像生成文本响应。
- **Best-of-N (BoN)**：从 N 个候选响应中根据评分函数选择最优响应的策略，广泛用于提升 LLM/LVLM 可靠性。
- **Self-Certainty (SC)**：基于预测分布与均匀分布的 KL 散度衡量 token 级置信度的 LLM 响应评分方法。
- **Visual Lookback Score ($A_t$)**：生成过程中第 t 个 token 对视觉 token 的注意力权重比例，作为视觉参考使用的轻量级代理。
- **Visual Hallucination**：LVLM 生成与图像内容不一致的流畅响应，属于多模态特有的事实错误类型。
- **Visual Relevance Distribution ($q_\lambda$)**：基于视觉回溯分数经熵正则化得到的响应级 token 权重分布，用于聚合 token 级评分。
- **Output-Space Confidence**：模型自生成的响应概率信号，不依赖外部验证器或奖励模型。
- **Product-of-Experts Interpretation**：LOOKBACK 可解释为输出置信度与视觉回溯分数的乘积型专家集成。

## 可复现要素
- **数据集**：VQAv2、CHAIR、AMBER、HallusionBench（均为公开基准）。
- **代码/权重**：论文未明确声明开源仓库；模型使用 LLaVA-1.5-7B、Qwen2.5-VL-7B、InternVL3-8B（公开权重）。
- **关键超参**：核采样 temperature=1.2, top-p=0.9, N=25 候选数；LOOKBACK 超参 $(\alpha, \lambda)$ 分别为 (7.0, 1.5)、(0.5, 1.25)、(0.25, 1.25)（按模型设置）。
- **硬件配置**：LLaVA-1.5-7B 使用 NVIDIA RTX A6000；Qwen2.5-VL-7B / InternVL3-8B 使用 NVIDIA H200。

---
title: "Heterogeneous-Vision-Language-Ensemble-with-Disagreement-Awa"
source: https://arxiv.org/pdf/2608.12843v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:30:24"
field: "多模态检索与跨模态对齐"
keywords: ["Text-Based Person Anomaly Retrieval", "Vision-Language Models", "Multimodal Ensemble", "Reranking", "AI City Challenge"]
innovations: ["异构VLM嵌入迭代集成+分数对齐解决跨模型排序不一致问题", "分歧感知VLM选择性重排序路由策略大幅降低推理成本", "RRF融合异构双编码器与VLM排名实现无需校准的跨模型聚合"]
benchmarks: ["PAB (Pedestrian Anomaly Behavior)", "AI City Challenge 2026 Track 4"]
---

# 论文速读：Heterogeneous-Vision-Language-Ensemble-with-Disagreement-Aware Reranking for Text-Based Person Anomaly Retrieval

## 一句话总结
本文提出了针对AI City Challenge 2026 Track 4（基于文本的行人异常检索，TBAPS）的解决方案，通过**异构视觉-语言嵌入模型的迭代集成**与**分歧感知VLM选择性重排序**相结合，在PAB基准上取得90.92% mAP（完整36,773图库设置），显著优于已有方法。

## 研究问题与动机
1. **任务定义**：基于文本的行人异常检索（TBAPS）要求通过自然语言描述从大规模图像库中检索具有异常行为的行人，需对行人外观、行为语义、物体交互及场景上下文进行细粒度跨模态推理。
2. **现有方法不足**：既有TBAPS方法（如CMP、HUI）主要依赖同质嵌入模型集成，未能充分利用不同架构/训练目标的预训练VLM模型所蕴含的互补表示。
3. **异构集成的技术障碍**：独立开发的嵌入模型对gallery图像的排序可能不一致，直接进行分数级融合会导致匹配错误。
4. **效率-精度权衡**：穷举式调用计算昂贵的VLM cross-encoder重排序不可行，需要智能路由机制仅在模糊查询上触发重排序。

## 核心贡献（创新点）
1. **异构嵌入迭代集成框架**：将Voyage Multimodal、BGE-VL-v1.5-mmeb、Qwen3-VL-Embedding-8B等异构预训练VLM嵌入模型引入迭代集成，扩展了基线HUI的单一嵌入空间，本质区别在于利用了不同架构/训练目标带来的表示多样性而非单纯增加同质模型数量。
2. **Score Alignment分数对齐机制**：针对不同嵌入模型内部gallery排序不一致的问题，设计了统一gallery索引映射，确保跨模型分数融合的正确性，这是已有迭代集成框架未解决的工程性瓶颈。
3. **Disagreement-Aware VLM重排序路由**：提出基于集合分歧的自适应路由策略，仅在top-1预测不一致时（严格零容忍阈值）才调用gemini-3.1-flash-lite进行cross-encoder重排序，实现了84.9%的推理延迟降低；相比AnomalyLMM等对全量查询调用VLM的方法，在保持精度的同时大幅提升了计算效率。
4. **RRF融合策略**：采用Reciprocal Rank Fusion（k=60）将异构双编码器集成结果与VLM重排序结果融合，无需分数归一化即可处理不同尺度的排名信号。

## 方法详解
**整体流程**：粗到细的两阶段范式——先通过异构双编码器集成获得初始排名，再对分歧查询进行选择性VLM交叉编码器重排序，最后通过RRF融合。

**1. 基础检索框架（HUI [11]）**：采用Local-Global Hybrid Perspective (LHP) 模块（概率局部/全局图像变换增强视觉表示）和Unified Image-Text (UIT) 建模（联合优化ITC、ITM、MLM、MIM四种损失）。图文编码为共享嵌入空间：$z_q = f_q(q)$，$z_i = f_i(i)$，相似度用余弦相似度计算：$S(i,q) = \frac{z_i^\top z_q}{\|z_i\|_2 \|z_q\|_2}$。

**2. 异构嵌入迭代集成**：设嵌入模型集合$\mathcal{M} = \{M_1, ..., M_K\}$，每个模型独立计算相似度矩阵$S^{(k)} \in \mathbb{R}^{N_q \times N_g}$。初始化$S = S_{init}$（来自基线模型），迭代更新：
$$S \leftarrow w_k S + (1 - w_k) S^{(k)}$$
其中$w_k$控制累积预测与新增模型预测的贡献权重。

**3. Score Alignment**：对每个嵌入模型的内部gallery排序构建映射到参考gallery索引的对齐方案，重排相似度矩阵后再进行融合，保证不同模型的分数对应相同的gallery图像。

**4. 分歧感知VLM重排序**：设定零容忍阈值——若多个集成配置的top-1预测存在任何不一致，则判定为歧义查询并路由至VLM（gemini-3.1-flash-lite）。VLM作为cross-encoder对top-10候选进行精细推理（提示词设定为"expert surveillance analyst"）。最终仅15.1%（299/1978）的查询被路由。

**5. RRF融合**：对集成排名与VLM重排序结果采用Reciprocal Rank Fusion：
$$\text{RRF}(d) = \sum_{r \in \mathcal{R}} \frac{1}{k + \text{rank}(r, d)}$$
其中$k=60$，最终按RRF分数排序输出。

## 实验与结果
**数据集**：PAB（Pedestrian Anomaly Behavior）基准，训练集>100万合成图像-文本对（1,000常规+1,600异常行为类别），评估集1,978个查询、36,773个gallery图像（含34,795个干扰项）。

**评估指标**：mAP、Recall@1、Recall@5、Recall@10。

**主要结果（完整挑战测试集，36,773图库）**：

| 方法 | mAP (%) | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---|---|---|---|
| HUI [11]基线 | — | 89.23 | 99.70 | 99.90 |
| Ours: Iterative Ensemble | **90.90** | **85.08** | **97.72** | **98.68** |
| Ours: + VLM Rerank & RRF | **90.92** | **85.13** | 97.72 | 98.68 |

**最强结果**：最终提交方案取得90.92% mAP、85.13% Recall@1，较基线HUI的89.23% Recall@1提升约1.9个百分点（注：基线未报告mAP，但集成版本已达90.90% mAP，相较之前最佳方法SSDC [17]的92.74% mAP仍略低，但完整设置下更具挑战性）。

**消融关键发现**：
- 引入Voyage Multimodal作为首个增量模型即带来大幅提升，证明其嵌入空间与基线高度互补。
- Qwen3-VL-Embedding放在最后一步且需精细调权（权重0.88）可带来+1.29% mAP跃升（88.96%→90.25%）。
- BGE-VL-v1.5-mmeb作为中间步骤在含大规模干扰的Setting B下最优（90.90% mAP），而Rzen-VL在标准Setting A下更强（95.12% mAP），说明不同模型的互补性具有场景依赖性。
- 分歧感知重排序以极低代价（仅15.1%查询）带来额外+0.02% mAP提升，同时节省84.9%推理时间。

**消融结论**：最强的集成不一定由最强的独立模型组成，**表示互补性**（embedding space complementarity）比单模型性能更重要。

## 相关工作脉络
1. **HUI框架 [11]**：本文的直接基线，提出LHP+UIT+迭代集成的TBAPS框架，但在集成中使用了相对同质的嵌入模型集合；本文在此基础上扩展至异构VLM模型并引入分歧感知路由。
2. **CMP/PE/IHNM [18]**：PAB基准的首批基线方法，基于Cross-Modal Pose-aware架构，代表了TBAPS任务的起点，但性能显著低于后续方法。
3. **AnomalyLMM [6]**：将生成式LVLM应用于行人异常检索的cross-encoder方法，对全量查询进行VLM推理；本文与之形成对比——通过分歧路由仅对15.1%查询调用VLM，实现相近精度但大幅提升效率。
4. **SSDC [17]**：近期在标准PAB设置上取得最高mAP（92.74%）的方法，但未在含大规模干扰的完整挑战设置下评估；本文在更严格的36,773图库设置下给出更有实践价值的结果。
5. **CLIP [13]/BLIP-2 [7]/BEiT-3 [16]**：经典VLM预训练模型，本文将其作为嵌入源纳入异构集成，而非直接作为检索器使用。
6. **RRF [3]**：Rank Fusion经典方法，本文将其引入异构双编码器与VLM重排序结果的融合环节，避免分数校准问题。

## 局限性与未来方向
1. **推理成本增加**：多嵌入模型集成加上选择性VLM重排序导致比单一模型更高的计算开销，尽管路由策略已大幅降低延迟。
2. **超参数经验设定**：集成权重schedule和路由阈值均通过验证集手动调优，缺乏自适应学习机制。
3. **未来方向**：探索自适应集成权重学习和可学习的查询路由策略，进一步提升精度与效率的平衡。

## 研究启发与可借鉴点
1. **异构集成优先于同质堆叠**：本文揭示了"最强的独立模型≠最优集成成员"这一关键洞见，强调embedding space互补性的重要性；对于任何多模型集成任务，应有意识地选择架构/训练目标差异化的模型而非简单地叠加最强模型。
2. **Score Alignment的工程技巧**：针对独立模型gallery排序不一致的问题，通过显式映射对齐再融合的简单但有效手段，可推广至任何跨模型分数融合场景。
3. **分歧路由的自适应推理设计**：将集合共识/分歧作为查询难度代理信号，据此动态调度计算资源（轻量双编码器vs.重VLM cross-encoder），是一种可扩展的"聪明地省钱"策略，适用于各类两阶段检索系统。
4. **RRF在异构排名融合中的实用性**：无需分数归一化即可融合不同尺度、不同分布的排名列表，适合作为异构模型集成的通用融合层。
5. **可结合方向**：本团队的检索/排序任务可借鉴"分歧感知选择性地调用更强模型"的范式，在成本约束下实现精度-效率的最优折衷；也可将BGE-VL、Qwen3-VL等开源嵌入模型直接纳入自有系统的集成框架进行验证。

## 关键术语表
**TBAPS（Text-Based Person Anomaly Retrieval）**：基于文本的行人异常检索，通过自然语言描述从大规模图像库中检索具有异常行为的行人，需理解行为语义而不仅依赖外观。
**HUI（Hybrid, Unified, Iterative）**：本文采用的基线检索框架，整合局部-全局混合视角增强、统一图文建模和迭代集成三个核心组件。
**LHP（Local-Global Hybrid Perspective）**：基线框架中的视觉增强模块，通过概率性的局部和全局图像变换丰富视觉表示。
**UIT（Unified Image-Text）**：基线框架中的联合预训练目标，同时优化ITC、ITM、MLM、MIM四种跨模态学习任务。
**Score Alignment**：针对不同嵌入模型内部gallery排序不一致问题，将各模型的相似度矩阵投影到统一gallery索引上的预处理步骤。
**Disagreement-Aware Routing**：基于多模型top-1预测分歧程度判断查询模糊性的路由机制，仅在存在分歧时触发昂贵的VLM重排序。
**RRF（Reciprocal Rank Fusion）**：通过倒数排名求和融合多个排序列表的方法，无需分数归一化即可处理异构排名信号。
**PAB（Pedestrian Anomaly Behavior）**：AI City Challenge 2026 Track 4使用的行人异常检索基准，包含>100万合成训练对和含大规模干扰的真实gallery。

## 可复现要素
- **数据集**：PAB（Pedestrian Anomaly Behavior）基准，由AI City Challenge 2026提供，论文中引用了[18]。
- **代码**：论文致谢中提到baseline框架[11]的代码已公开；本文提出的异构集成、Score Alignment、分歧路由等组件未明确声明代码开源情况。
- **关键超参**：UIT微调15个epoch、LHP微调3个epoch；集成权重（Voyage: 0.9, BGE-VL: 0.92, Qwen: 0.88）；VLM重排序阈值（零容忍）；RRF常数k=60；VLM为gemini-3.1-flash-lite（zero-shot）。
- **硬件**：单张NVIDIA RTX A6000 GPU（48GB VRAM）用于训练，特征提取通过本地GPU+云服务+商业API完成。
- **开源模型**：Voyage Multimodal、BGE-VL-v1.5-mmeb、Qwen3-VL-Embedding-8B（部分以API形式调用）。

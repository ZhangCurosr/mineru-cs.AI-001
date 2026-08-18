---
title: "Predicting Space Groups of Double Perovskites by LLM with Dynamic Few-Shot Learning"
source: https://arxiv.org/pdf/2608.10483v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:36:39"
---

# 论文速读：Predicting Space Groups of Double Perovskites by LLM with Dynamic Few-Shot Learning

## 一句话总结
本文提出 DyRIS，一个基于 LLM agent 的预测框架，通过多样性增强的动态 few-shot 检索与规则引导推理，从双钙钛矿（DPs）化学组成预测最可能的 Top-3 空间群（SG）候选；在严重类别不平衡的数据集上，DyRIS 以较低训练数据量实现了与最强基线相当的总体精度，并显著提升对少数 SG 的预测性能。

## 研究问题与动机
- **类别严重不平衡制约数据驱动模型**：ICSD 与 Materials Project 等数据库中的 DP 结构高度集中于 SG 14 与 SG 225（合计占 72%），导致纯统计模型对少数 SG 的预测性能大幅下降。
- **传统计算方法成本高且依赖先验**：高通量第一性原理计算需预先指定原胞信息，且低对称结构的计算代价随原子数急剧增加，难以直接用于大规模成分筛选。
- **LLM 隐式知识不足以可靠锁定 SG**：尽管 LLM 蕴含广泛的材料知识，但仅凭隐式参数难以在复杂成分空间中稳定缩小 SG 候选范围，需显式引入晶体学规则与证据整合机制。
- **现有基线缺乏类平衡感知设计**：CrabNet、CRYSPNet 等传统模型在 Overall Top-1 accuracy 上表现尚可，但 macro-F1 与少数类指标明显落后，缺乏针对长尾分布的结构化缓解策略。

## 核心贡献（创新点）
- **多样性约束的动态 few-shot 检索机制**：在加权嵌入空间中为每个查询动态选取上下文示例，并强制“每个 SG 类最多入选 1 个”的约束，打破多数类对 prompt 的主导，使候选覆盖更均衡。（本质区别：不同于固定示例集或纯 kNN，显式注入类多样性先验。）
- **多证据融合的规则引导 LLM 推理流程**：将 B/B′ 阳离子序构概率、Combined score、Global fit、特征级一致性指标与多数类偏置控制规则结构化输入 LLM agent，由 agent 按预定义原则完成 Top-3 筛选与排序。（本质区别：将 LLM 从端到端分类器转为多源证据的符号-数值混合推理器。）
- **证明规则推理优于传统 ML 重排序器**：在相同候选证据下，Logistic Regression、XGBoost、LightGBM、LambdaMART 等监督重排序模型在各项指标上均全面落后 DyRIS，尤其在少数类 Top-1 上差距显著。（本质区别：验证了规则引导的 LLm 推理在类不平衡场景下不可被固定阈值或黑盒分类器简单替代。）
- **揭示数据量-性能非单调性并提出混合策略**：训练比例 0.5→0.8 时部分 Top-1 指标反而下降，归因于排序规则与检索邻近证据的冲突；同时构建 DyRIS 与 PyCaret 基线的启发式融合方案，实现全指标提升。（本质区别：首次系统刻画 LLM 规则推理在高数据 regime 下的偏差来源，并给出互补集成路径。）

## 方法详解
- **数据预处理与划分**：保留 $E_{hull} \le 0.1$ eV/atom 的热力学稳定 DP，剔除样本数 $\le 3$ 的 SG 类，最终得 3,528 个样本、19 个 SG 类。SG 14（1,081）与 SG 225（1,460）定义为多数类（占 72%），其余 17 类为少数类。测试集每 SG 上限 30 样本以抑制多数类对整体指标的支配。
- **四组成分特征与嵌入空间**：拼接 161 维特征向量：① DP 特异性特征（容忍因子 $t$、八面体因子 $\delta$、Shannon 离子半径、氧化态）；② 位点特征（A/B/B′/X 的原子序、周期、族、电负性）；③ B 位衍生特征（B/B′ 氧化态差、半径差、电负性差）；④ Magpie 特征（元素属性统计描述）。通过 leave-one-out 网格搜索（步长 0.05，权重和为 1）优化四类特征块权重，以 Overall Top-1 macro-F1 为目标。
- **多样性增强动态 few-shot 检索**：在加权嵌入空间中执行 kNN 检索，强制约束“每 SG 类最多取 1 个示例”，返回 Top-5 候选供后续推理使用。
- **B/B′ 阳离子序构软先验**：训练 PyCaret 代理模型预测岩盐、层状、柱状及稀有序构的概率；将概率与各类序构兼容的 SG 集合映射表一同输入 LLM，作为软结构约束而非硬标签。
- **定量指标设计**：
  - **Combined score**：$CS(s) = \log(1+n_{local}(s)) \cdot \frac{n_{local}(s)}{N_s+1}$，衡量候选 SG 在查询邻居中的局部连续密度，并对全局高频 SG 归一化。
  - **Global fit**：基于 6 个关键特征的标准化偏差平方和开方，评估候选 SG 与查询的整体分布吻合度。
  - **特征级一致性指标**：$z_{abs\_mean}(s)$（平均偏差）、$z_{max\_abs}(s)$（最大偏差）、$z_{best\_count}(s)$（最优特征计数）。
- **规则引导 LLM 推理**：LLM agent 按五条原则输出排名 Top-3 SG：① 优先保留高 Combined score 候选；② 仅当序构兼容性、Global fit、特征一致性与检索距离一致时才替换候选；③ 稀有序构兼容候选在多证据支持下可入选；④ 多数类 SG 需多指标明确支持才能排 Top-1（偏置控制）；⑤ 最终排序综合检索排名、Combined score、Global fit 与特征一致性。 backbone 为 `gpt-5.4-mini-2026-03-17`，temperature=1.0，Agent 框架为 Microsoft AutoGen。

## 实验与结果
- **实验设置**：

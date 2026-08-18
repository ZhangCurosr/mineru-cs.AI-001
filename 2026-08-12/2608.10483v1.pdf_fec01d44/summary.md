---
title: "Predicting Space Groups of Double Perovskites by LLM with Dynamic Few-Shot Learning"
source: https://arxiv.org/pdf/2608.10483v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:37:13"
field: "材料信息学/晶体结构预测"
keywords: ["double perovskites", "space group prediction", "large language models", "few-shot learning", "class imbalance", "in-context learning", "materials informatics"]
innovations: ["多样性增强的动态few-shot检索机制，通过每SG最多取1示例约束防止主类主导prompt", "基于B/B′阳离子排序先验与定量指标（Combined score/Global fit/z-score一致性）的规则引导LLM推理框架", "推理层主SG偏置控制策略，在不修改训练分布下缓解类别不平衡"]
benchmarks: ["Overall Top-1 accuracy", "Overall Top-1 macro-F1 score", "Overall Top-3 accuracy", "Minor-SG Top-1 accuracy", "Minor-SG Top-1 macro-F1 score", "Minor-SG Top-3 accuracy"]
---

# 论文速读：Predicting Space Groups of Double Perovskites by LLM with Dynamic Few-Shot Learning

## 一句话总结
本文提出 DyRIS，一种基于 LLM 智能体的双钙钛矿（DP）空间群预测框架，通过多样性增强的动态 few-shot 检索与基于晶体学先验（B/B′阳离子排序、定量指标、主类偏置控制）的规则引导推理，在严重类别不平衡的数据集上实现了对少数类空间群的可靠预测。

## 研究问题与动机
- **双钙钛矿空间群预测的重要性**：DPs（通式 $A_2BB'X_6$）的功能性质不仅取决于化学成分，还受稳定结构的空间群（SG）控制；相同成分不同 SG 可导致电子迁移率、光学吸收等性质显著差异。
- **数据驱动方法的不足**：现有数据库（ICSD、Materials Project）中 DP 结构强偏向高对称性 SG（如 Fm-3m/SG 225 和 SG 14），严重类别不平衡使纯统计模型对少数类 SG 预测性能大幅下降。
- **LLM 的潜力与局限**：LLM 隐式编码广泛材料知识，可通过 prompt 工程显式融入晶体学规则；但仅靠隐式知识不足以可靠缩小候选 SG 范围，需结合检索与定量证据。
- **高计算成本替代方案的需求**：高通量 ab initio 计算需先验原胞信息且低对称结构计算代价剧增；生成模型（如 MatterGen）仍依赖训练分布，亟需融合领域知识的混合方法。

## 核心贡献（创新点）
- **多样性增强的动态 few-shot 检索**：按嵌入空间距离动态检索相关上下文示例，并通过"每 SG 最多取 1 个示例"的多样性约束防止主类主导 prompt，这是与固定示例集或纯 kNN 检索的本质区别。
- **基于 B/B′阳离子排序与定量指标的规则引导 LLM 推理**：将软性晶体学先验（排序概率→兼容 SG 集合映射）与 Combined score、Global fit、特征级一致性指标整合进 LLM prompt，由 LLM 作为推理智能体而非简单分类器，这与传统 ML 重排序模型形成本质差异。
- **主 SG 偏置控制机制**：在推理规则中显式限制 SG 14 和 SG 225 的过度排名，要求其必须在多项指标上均提供强支持才可位列 Top-1，这是针对类别不平衡问题的结构性设计。
- **在严重不平衡数据集上对少数类 SG 的系统性提升**：在训练数据比例 0.5 时，DyRIS 获得最佳 Overall Top-1 macro-F1 及全部 Minor-SG 指标，Minor-SG Top-1 准确率较最强基线 CrabNet 提升 3.26 个百分点。

## 方法详解
**整体框架 DyRIS** 分为三个阶段：特征提取与嵌入构建 → 多样性增强动态 few-shot 检索 → 规则引导 LLM 推理。

- **特征工程（161维特征向量）**：
  1. **DP-specific 特征（10维）**：容忍因子 $t$、八面体因子 $\delta$、Shannon 离子半径、氧化态；
  2. **Site 特征（16维）**：A/B/B′/X 位点的原子序数、周期、族、电负性；
  3. **B-site-derived 特征（3维）**：B/B′氧化态差、离子半径差、电负性差；
  4. **Magpie 特征（132维）**：元素性质的均值、范围、方差等统计描述子。
  通过特征块加权（网格搜索优化 Overall Top-1 macro-F1）构建加权嵌入空间。

- **多样性增强动态 few-shot 检索**：
  对每个查询成分，在加权嵌入空间中按距离排序训练样本，检索 Top-5 候选，但**每个 SG 类最多选取 1 个示例**，确保 prompt 中 SG 标签多样性，避免主类 SG 14/225 主导。

- **B/B′阳离子排序预测**：
  使用 PyCaret 代理模型预测四种排序类型（rock-salt、layered、columnar、rare）的概率，作为软结构先验传入 LLM；提供排序类型→兼容 SG 集合的映射表（如 rock-salt 兼容 {1,2,7,12,14,31,87,139,146,148,201,225}）。

- **定量指标计算**：
  - **Combined score (CS)**：$CS(s) = \log(1+n_{local}(s)) \cdot \frac{n_{local}(s)}{N_s+1}$，衡量候选 SG 在查询邻域的局部密度，归一化项 $N_s+1$ 抑制全局高频 SG；
  - **Global fit**：$G_{fit}(s) = \sqrt{\sum_{f \in F} z(s,f)^2}$，聚合6个关键特征的标准化偏差；
  - **特征级一致性**：$z_{abs\_mean}(s)$（平均绝对偏差）、$z_{max\_abs}(s)$（最大绝对偏差）、$z_{best\_count}(s)$（最优特征计数）。

- **规则引导推理（LLM Agent）**：
  LLM（gpt-5.4-mini）作为推理智能体，依据五条原则从 Top-5 候选筛选并排序 Top-3：
  1. Combined score 最高的候选优先保留；
  2. 仅当排序兼容性、Global fit、特征一致性、检索距离排名一致时才执行替换；
  3. 稀有排序兼容候选在有序证据支持下可纳入；
  4. 主 SG 偏置控制：SG 14/225 仅在多指标一致支持时居 Top-1；
  5. 最终排名综合 Combined score、检索排名、Global fit、特征一致性。

## 实验与结果
- **数据集**：3,528 个热力学稳定 DP 条目（$E_{hull} \leq 0.1$ eV/atom），覆盖 19 个 SG 类；SG 14（1,081）和 SG 225（1,460）为主类（占 72%），其余 17 个为少数类。
- **评估设置**：训练数据比例 0.3/0.5/0.8，5 次随机分割；测试集每 SG 最多 30 样本；指标包括 Overall/Minor-SG 的 Top-1 准确率、Top-1 macro-F1、Top-3 准确率。
- **主要结果（训练比例 0.5）**：
  - DyRIS Overall Top-1 准确率 0.4266（与 CrabNet 0.4318 相当），**Overall Top-1 macro-F1 0.3846 最优**；
  - **Minor-SG Top-1 准确率 0.3762 最优**，较 CrabNet（0.3436）提升 **3.26 个百分点**；
  - Minor-SG Top-3 准确率 0.6658，优于 PyCaret 基线（0.6494）；
  - Overall Top-3 准确率 0.6994，仅次于 PyCaret（0.7074）。
- **数据敏感性**：DyRIS 在所有训练比例下保持最高 Overall Top-1 macro-F1 和 Minor-SG Top-1 准确率/ macro-F1；但 0.8 比例时 Overall/Minor-SG Top-1 准确率略有下降（0.4154/0.3534），根源在于最终排序步骤对主类的降级效应。
- **消融实验**：移除检索示例导致 Overall Top-3 准确率从 0.6994 降至 0.5424（降幅最大）；移除定量指标导致 Minor-SG Top-1 准确率下降约 5.5 个百分点；移除主 SG 偏置控制导致 Overall Top-1 macro-F1 下降约 3 个百分点。
- **ML 重排序替代实验**：用 LR/RF/XGBoost/LightGBM/LambdaMART 等替换 LLM 推理步骤，DyRIS 在所有指标和训练比例下均优于最强 ML 模型（如 0.5 比例下 Best learned Overall Top-1 accuracy 仅 0.3445 vs. DyRIS 0.4266）。
- **混合策略**：DyRIS + PyCaret 启发式融合在 0.8 比例下全面超越两者单独表现。

## 相关工作脉络
- **CRYSPNet**（Liang et al., 2020）：基于神经网络的 DP 空间群预测代理模型，纯数据驱动，受类别不平衡限制；DyRIS 通过检索+规则推理弥补其 minority SG 劣势。
- **CrabNet**（Wang et al., 2021）：组成约束注意力网络，在 Overall Top-1 准确率上表现最强；DyRIS 在其 macro-F1 和 Minor-SG 指标上全面超越。
- **MatterGen**（Zeni et al., 2025）：基于扩散的晶体结构生成模型，直接生成原子坐标；DyRIS 聚焦于从成分直接预测 SG 候选排名，计算开销更低且显式融入晶体学先验。
- **LLM 材料发现工作**（Miret & Krishnan, 2025; Jia et al., 2024; Lee et al., 2025）：验证 LLM 在材料科学中的 QA 与生成能力；本文定位差异在于将 LLM 作为**多证据整合推理智能体**而非直接预测器，并针对类别不平衡设计检索与偏置控制机制。
- **特征工程传统方法**（Ward et al., 2016 Magpie; Zhao et al., 2020）：使用组成描述子训练分类器；DyRIS 将这些特征用于构建嵌入空间与定量指标，而非直接输入分类器。
- **Imbalanced learning 方法**（López et al., 2013; Zhu et al., 2015）：重采样/重加权策略；DyRIS 不改变训练分布，而是通过检索多样性约束和推理偏置控制在 inference 阶段缓解不平衡。

## 局限性与未来方向
- **高数据比例下 Top-1 性能下降**：训练比例从 0.5 增至 0.8 时，Overall 和 Minor-SG Top-1 准确率均降低，根源在于规则引导的排序步骤对主类 SG 存在过度降级。
- **单一排序策略的类别适应性不足**：对主类 SG 检索证据已足够强，但额外定量指标可能覆盖近邻证据；对少数类 SG 检索稳定性差，需更积极的证据整合——需开发**类别自适应排序策略**。
- **混合策略尚处启发式阶段**：DyRIS 与 PyCaret 的融合依赖人工规则（Top-1 保留、一致性优先、主类偏置校正），缺乏系统性联合推断框架。
- **B/B′排序代理模型的间接性**：排序标签从 CIF/POSCAR 结构文件推导，非直接观测；稀有排序类型预测置信度低，依赖软概率而非硬标签。
- **LLM 成本与可重复性**：依赖 GPT-5.4-mini API，推理成本较高；模型版本依赖（2026-03-17）影响完全复现。

## 研究启发与可借鉴点
- **"检索+规则推理"的 LLM 智能体范式**：将 LLM 定位为多证据整合推理器而非直接预测器，适用于需要融合异构证据（统计邻近性+领域规则+定量指标）的材料属性预测任务，可有效缓解类别不平衡。
- **多样性约束的 few-shot 检索策略**："每类最多取 k 个示例"的约束设计简单有效，可直接迁移至其他类别不平衡的材料分类/回归任务（如带隙预测、形成能估计）。
- **定量指标嵌入 prompt 的标准化偏差方法**：利用训练集类内均值/标准差构建 z-score 特征级一致性指标，为 LLM 提供可解释的数值证据，该方法可泛化至其他晶体学特征预测。
- **主类偏置控制的推理层干预**：在 LLM prompt 中显式声明类别不平衡事实并施加排名约束，是一种无需修改训练分布的 inference-time 去偏策略，值得在其他 Imbalanced materials ML 任务中验证。
- **混合模型融合启发式**：数据驱动模型（强 Top-3 覆盖）与规则推理模型（强 Top-1 排序+少数类）的互补性提示了"生成+验证"双阶段架构的价值，可探索基于置信度与排序一致性的系统性融合。

## 关键术语表
**Double Perovskites (DPs)**：通式为 $A_2BB'X_6$ 的双钙钛矿，由两种不同 B 位阳离子有序排列形成的钙钛矿衍生结构，具有可调组分与丰富物性。
**Space Group (SG)**：描述晶体结构全部对称操作的 230 种空间群之一，决定材料的物理性质对称性约束。
**DyRIS**：Dynamic and Diversity-enhanced few-shot retrieval and Rule-Guided Inference for Space-Group Prediction，本文提出的 LLM 智能体框架。
**In-context Learning / Few-shot Prompting**：在 LLM 输入中提供若干输入-输出示例，使其无需微调即可执行目标任务。
**Combined score (CS)**：衡量候选 SG 在查询成分邻近区域内局部密度的指标，结合局部频次与全局归一化。
**Global fit**：聚合六个关键特征（容忍因子、八面体因子、A位半径、B/B′半径比、B位半径差、B位电负性差）的标准化偏差的标量距离度量。
**B/B′ Cation Ordering**：双钙钛矿中 B 位两种阳离子的有序排列方式，分为 rock-salt、layered、columnar 及稀有排序类型，直接影响允许的空间群。
**Macro-F1 Score**：对各类别 F1 分数求算术平均的指标，对少数类更敏感，适合评估类别不平衡任务。

## 可复现要素
- **数据集**：3,528 个 DP 条目，来源于 ICSD/Materials Project，经 $E_{hull} \leq 0.1$ eV/atom 稳定性过滤及最少 3 样本 SG 类过滤；论文未声明公开仓库链接。
- **代码/权重**：论文未提供代码开源声明；使用 OpenAI API（gpt-5.4-mini-2026-03-17）与 Microsoft AutoGen 框架；PyCaret 代理模型权重未单独开源。
- **关键超参**：检索 Top-5 示例、每 SG 最多 1 示例；特征块权重经步长 0.05 的网格搜索在训练集上优化；LLM temperature=1.0；训练数据比例 0.3/0.5/0.8；5 次随机种子重复。

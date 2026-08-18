---
title: "Predicting Space Groups of Double Perovskites by LLM with Dynamic Few-Shot Learning"
source: https://arxiv.org/pdf/2608.10483v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:36:42"
field: "材料计算与AI"
keywords: ["双钙钛矿", "空间群预测", "大语言模型", "少样本学习", "类别不均衡", "材料信息学", "检索增强生成"]
innovations: ["多样性增强的动态少样本检索防止多数类主导提示", "多源定量证据与晶体学先验的结构化规则引导推理", "主要SG偏差控制机制抑制少数类预测偏见"]
benchmarks: ["Overall Top-1 accuracy", "Overall Top-1 macro-F1 score", "Minor-SG Top-1 accuracy", "Minor-SG Top-3 accuracy"]
---

# 论文速读：Predicting Space Groups of Double Perovskites by LLM with Dynamic Few-Shot Learning

## 一句话总结
本文提出了 DyRIS（Dynamic and Diversity-enhanced few-shot retrieval and Rule-Guided Inference for Space-Group Prediction），一个基于 LLM-agent 的框架，通过多样性增强的动态少样本检索与晶体学先验规则引导的推理，解决双钙钛矿（DPs）空间群（SGs）预测中的严重类别不均衡问题，在少数类 SG 预测上显著优于传统数据驱动模型。

## 研究问题与动机
- **核心问题**：双钙钛矿（$A_2BB'X_6$）的物性高度依赖于稳定结构的空间群，而数据库（如 ICSD、Materials Project）中 SG 分布严重不均衡（SG 14 和 225 占比 72%），导致纯数据驱动模型对少数类 SG 预测性能差。
- **现有方法不足**：
  1. CRYSPNet 等代理模型和 MatterGen 等生成模型强依赖训练数据分布，难以弥补少数类样本匮乏的问题；
  2. 传统 kNN 检索策略在类别不均衡下会偏向多数类 SG，无法保证候选集多样性；
  3. LLM 虽隐式编码了材料科学知识，但仅靠隐式知识无法可靠缩小可能的 SG 候选空间，需显式整合晶体学域知识。

## 核心贡献（创新点）
- **多样性增强的动态少样本检索**：在嵌入空间中为每个查询成分动态检索 Top-5 相关上下文示例，并通过"每个 SG 最多选 1 个"的多样性约束防止多数类主导提示，与固定示例集的 few-shot 方法本质不同。
- **定量指标驱动的规则引导推理**：设计了 Combined score（局部密度）、Global fit（分布一致性）、特征级 z 一致性（$z_{abs\_mean}$、$z_{max\_abs}$、$z_{best\_count}$）等多源证据，由 LLM-agent 结构化整合而非简单加权求和，克服了固定阈值/权重无法适应不同查询的局限。
- **B/B′阳离子有序化软先验**：用 PyCaret 代理模型预测四种有序化类型（rock-salt/layered/columnar/rare）的概率，并将兼容 SG 集合以软约束形式输入 LLM，而非确定性标签，避免稀有有序化被低概率淹没。
- **主要 SG 偏差控制机制**：在提示中显式标记 SG 14 和 225 为多数类，并要求仅在多指标一致支持时才将其列为 Top-1，从规则层面抑制多数类过度排名。
- **替代实验证明 LLM 推理不可替代**：将最终规则引导推理替换为 Logistic Regression、XGBoost、LambdaMART 等 ML 重排序模型后，DyRIS 在所有指标上均更优，尤其在 Minor-SG Top-1 上差距显著，表明多源证据的结构化整合无法被单一可学习打分函数取代。

## 方法详解
**整体框架（DyRIS）**：输入化学成分 → 特征提取 → 多样性检索 → LLM-agent 规则引导推理 → 输出 Top-3 排名 SG 候选。

### 1) 特征工程（161 维向量）
四类特征块拼接：
- **DP-specific features（10 维）**：容差因子 $t$、八面体因子 $\delta$、Shannon 离子半径、氧化态；
- **Site features（16 维）**：A/B/B′/X 位点的原子序数、周期、族、电负性；
- **B-site-derived features（3 维）**：B/B′ 氧化态差、离子半径差、电负性差；
- **Magpie features（132 维）**：composition-level 统计描述符（均值、极差、方差等）。

嵌入空间构建时引入**特征块加权**（$a+b+c+d=1$），通过网格搜索优化 Overall Top-1 macro-F1 得分，提升检索性能。

### 2) 多样性增强动态少样本检索
- 对每个 query，在加权嵌入空间中按距离排序训练样本；
- 应用多样性约束：**每个 SG 类最多选取 1 个样本**作为 in-context 示例（Top-5）；
- 缓解多数类 SG（SG 14、225）在检索结果中占主导的问题。

### 3) B/B′ 有序化代理模型
- 使用 pymatgen 从 CIF/POSCAR 文件标注有序化类型（rock-salt/layered/columnar/other）；
- 训练 PyCaret 分类器预测四种有序化的概率；
- 将预测概率及兼容 SG 集合作为**软结构先验**输入 LLM 提示。

### 4) 定量指标设计
**Combined score（局部密度）**：
$$CS(s) = \log(1 + n_{local}(s)) \cdot \frac{n_{local}(s)}{N_s + 1}$$
其中 $n_{local}(s)$ 为从 query 出发连续相邻且同属 SG $s$ 的样本数，$N_s$ 为 SG $s$ 的全局样本数。该指标衡量候选 SG 在 query 邻域的局部密度，归一化项 $(N_s+1)$ 抑制全局高频 SG 的膨胀效应。

**标准化偏差**：
$$z(s, f) = \frac{x_f - \mu(s, f)}{\sigma(s, f)}$$

**Global fit（全局分布一致性）**：
$$G_{fit}(s) = \sqrt{z(s,tf)^2 + z(s,of)^2 + z(s,r_A)^2 + z(s,r_{B'/B})^2 + z(s,\Delta r_B)^2 + z(s,\Delta EN_B)^2}$$

**特征级一致性指标**：
- $z_{abs\_mean}(s)$：平均绝对偏差，越小越好；
- $z_{max\_abs}(s)$：最大绝对偏差，识别极端不匹配；
- $z_{best\_count}(s)$：在候选集中表现出最小偏差的特征数量，越大越好。

### 5) 规则引导推理（五原则）
1. **Combined score 锚定**：以 Combined score 最高的三个候选为初始 Top-3 候选集；
2. **KNN 共识保留**：KNN Top-3 与 Combined score Top-3 的交集候选被强保留；
3. **边界候选比较**：仅当多个信号（有序化兼容性、Global fit、特征一致性）一致矛盾时才替换候选；
4. **主要 SG 偏差控制**：SG 14/225 仅在多指标明确支持时才列为 Top-1；
5. **最终排序**：综合 Retrieval rank、Combined score、Global fit、特征一致性进行联合排序。

LLM-agent（GPT-5.4-mini）以 AutoGen 框架运行，Temperature=1.0，将上述所有证据结构化整合，输出最终 Top-3 排名。

## 实验与结果
**数据集**：3,528 个经 $E_{hull} \leq 0.1$ eV/atom 热力学过滤的 DP 条目，19 个 SG 类（SG 14: 1,081；SG 225: 1,460，合计 72% 为多数类；其余 17 类为少数类）。

**评估设置**：训练集比例 0.3/0.5/0.8，测试集每 SG 最多 30 个样本；五次随机分割取均值±标准差。

**主要结果（训练比例 0.5）**：
| 模型 | Overall Top-1 acc | Overall Top-1 macro-F1 | Overall Top-3 acc | Minor-SG Top-1 acc | Minor-SG Top-3 acc |
|------|-------------------|------------------------|-------------------|-------------------|-------------------|
| DyRIS | 0.4266 | **0.3846** | 0.6994 | **0.3762** | **0.6658** |
| CrabNet | 0.4318 | 0.3650 | 0.6546 | 0.3436 | 0.5860 |
| PyCaret-based | 0.4020 | 0.3002 | **0.7074** | 0.2958 | 0.6494 |

- DyRIS 在 **Overall Top-1 macro-F1** 和 **所有 Minor-SG 指标** 上均为最优；
- Minor-SG Top-1 accuracy 较 CrabNet 提升 **3.26 个百分点**；
- Minor-SG Top-3 accuracy 优于最强 PyCaret 基线。

**数据敏感性分析**：
- DyRIS 在所有训练比例下保持最高的 Overall Top-1 macro-F1 和 Minor-SG Top-1 指标；
- 训练比例从 0.5 增至 0.8 时，DyRIS 的 Overall Top-1 accuracy（0.4154）和 Minor-SG Top-1 accuracy（0.3534）略降，主要源于最终排序步骤对多数类 SG 的"降排"效应。

**消融实验**（比例 0.5）：
- 移除检索示例：Overall Top-3 acc 从 0.6994 降至 0.5424，影响最大；
- 移除定量指标：Minor-SG Top-1 acc 下降约 **5.5 个百分点**；
- 移除主要 SG 偏差控制：Overall Top-1 acc 和 macro-F1 各下降约 1.0 和 3.0 个百分点；
- 移除有序化信息：比例 0.8 时 Minor-SG Top-1 acc 下降约 **4.2 个百分点**，表明在高数据 regime 有序化先验作用更显著。

**ML 重排序替代实验**：
- DyRIS 在所有比例和指标上均优于最佳 ML 重排序模型（Logistic Regression/XGBoost/LightGBM/LambdaMART）；
- Minor-SG Top-1 accuracy 差距最大（如比例 0.5：DyRIS 0.3762 vs. Best learned 0.2270）。

**混合策略**（比例 0.8）：
- DyRIS + PyCaret-based 启发式混合方法在所有指标上均超越单一模型，验证了两者互补性。

## 相关工作脉络
- **CRYSPNet**（Liang et al., 2020）：基于神经网络的 SG 预测代理模型，纯数据驱动，受类别不均衡限制；DyRIS 通过检索+规则推理弥补其少数类预测不足。
- **CrabNet**（Wang et al., 2021）：成分约束注意力网络，在 Overall Top-1 accuracy 上最强，但在 macro-F1 和 Minor-SG 指标上弱于 DyRIS。
- **MatterGen**（Zeni et al., 2025）：扩散生成模型直接生成晶体结构，需先验原胞信息且低对称结构计算代价高；DyRIS 直接从成分预测 SG 排名，无需结构生成。
- **MaScQA / LLEMA / LLMatDesign**：LLM 在材料问答和发现中的应用；本文定位在于将 LLM 作为"推理 agent"而非单纯知识源，显式整合晶体学定量证据。
- **Ward et al. (Magpie, 2016)**：提供成分级统计描述符；本文扩展至 DP 特异性特征（容差因子、离子半径等）并与 Magpie 特征融合。
- **King & Woodward (2010)**：B/B′有序化晶体学规律；本文将其量化为软概率先验整合进 LLM 推理流程。

## 局限性与未来方向
- **高数据 regime 下 Top-1 性能下降**：训练比例从 0.5 增至 0.8 时，规则引导的最终排序对多数类 SG（SG 14/225）产生"降排"，导致 Overall 和 Minor-SG Top-1 accuracy 略降；根本原因是检索邻近证据与定量指标之间出现冲突，单一排名策略无法同时优化两类 SG。
- **混合策略仍为启发式**：DyRIS 与 PyCaret 的当前融合基于手动规则，未系统考虑模型置信度、排名一致性、类别类型和证据一致性的联合建模。
- **未来方向**：
  1. 开发**类别自适应排名策略**：对多数类 SG 更强保留检索邻近性，对少数类 SG 更积极应用证据整合排名；
  2. 构建** principled hybrid inference 框架**，联合优化多模型置信度与证据一致性；
  3. 探索更多材料体系中 LLM-agent 与晶体学先验的结合方式。

## 研究启发与可借鉴点
- **多样性约束检索策略**：在类别不均衡的 few-shot 学习中，"每类最多选 k 个"的多样性约束可有效防止提示被多数类主导，这一设计可迁移至其他材料属性预测任务（如带隙、弹性模量预测）。
- **多源证据结构化整合 vs. 可学习打分**：本文证明规则引导的 LLM 推理比 ML 重排序模型更优，启示在复杂材料预测中，**显式整合多源异质证据**（检索+定量+先验）可能比端到端学习更有效，尤其在样本匮乏的少数类场景。
- **软先验而非硬约束**：B/B′有序化概率以"软约束"形式输入 LLM，避免了错误硬标签导致的 propagated error，这一设计可推广至其他晶体学约束（如对称性规则、占位规则）的 LLM 整合。
- **特征块加权嵌入**：通过网格搜索优化特征组权重以构建更有判别力的嵌入空间，是一种简单有效的表征学习策略，适用于多源特征融合的材料科学任务。
- **与数据驱动模型的互补性**：DyRIS（LLM-agent）与 PyCaret（监督分类器）的混合策略证明**机理/规则驱动与数据驱动方法的互补**是提升预测鲁棒性的有效路径，可启发混合 AI 架构的设计。

## 关键术语表
- **Double Perovskites (DPs)**：通式为 $A_2BB'X_6$ 的双钙钛矿，通过 B/B′位点阳离子有序化形成，具有可调的光电、磁学和热电性能。
- **Space Group (SG)**：描述晶体结构对称性的 230 种空间群之一，决定材料的物理性质，是本文预测目标。
- **In-context Learning**：LLM 在推理时将少量示例（in-context examples）嵌入提示词中，无需微调即可适应特定任务。
- **Tolerance Factor (t)**：$t = (r_A + r_X) / [\sqrt{2}(r_B + r_X)]$，衡量钙钛矿结构稳定性的几何参数，与八面体倾斜和对称性降低相关。
- **B/B′ Cation Ordering**：双钙钛矿中 B 和 B′位点阳离子的有序排列模式（rock-salt/layered/columnar），直接影响空间群选择。
- **Combined Score**：衡量候选 SG 在 query 邻域局部密度的定量指标，结合局部连续样本数与全局频率归一化。
- **Macro-F1 Score**：对每个类别独立计算 F1 后取均值，对类别不均衡敏感，能反映少数类预测性能。
- **DyRIS**：Dynamic and Diversity-enhanced few-shot retrieval and Rule-Guided Inference for Space-Group Prediction，本文提出的 LLM-agent 框架。

## 可复现要素
- **数据集**：3,528 个热力学过滤（$E_{hull} \leq 0.1$ eV/atom）的 DP 条目，来源为 ICSD 和 Materials Project；论文未声明公开，需自行从上述数据库构建。
- **代码**：论文未声明开源代码库；实现基于 GPT-5.4-mini API 和 Microsoft AutoGen 框架。
- **关键超参**：检索 Top-5 示例、多样性约束（每 SG 最多 1 个）、Temperature=1.0、特征块加权网格搜索步长 0.05、训练比例 0.3/0.5/0.8、五次随机分割。
- **补充材料**：SI 包含特征构建细节（S1）、特征加权结果（S2）、有序化代理模型（S3）、完整提示模板（S4）等，可复现关键设计。

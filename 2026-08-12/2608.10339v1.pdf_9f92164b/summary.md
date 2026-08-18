---
title: "Expert-Guided g-computation with Large Language Models for Estimating Causal Efects on Timings: Applications to Hospital Quality Improvement"
source: https://arxiv.org/pdf/2608.10339v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:26:06"
field: "因果推断与LLM交叉"
keywords: ["因果推断", "g-computation", "大型语言模型", "甘特图", "医院质量改进", "平均住院时间"]
innovations: ["首次将甘特图形式化为等待时间因果DAG并建立g-computation识别定理", "提出Type I/II/III节点分类的专家引导因果推断框架", "开发五步LLM辅助管道实现大规模专家推理规模化"]
benchmarks: ["模拟研究（n=4000，50次重复，偏差对比）", "ZSFG医院11个QI干预真实世界评估（2193条住院记录）"]
---

# 论文速读：Expert-Guided g-computation with Large Language Models for Estimating Causal Effects on Timings: Applications to Hospital Quality Improvement

## 一句话总结
论文提出"egg-computation"（专家引导的g-computation），将医院流程常用的甘特图形式化为因果DAG，结合专家推理与数据驱动建模，通过LLM辅助管道实现大规模专家引导的因果推断，用于估计新型质量改进（QI）干预措施对平均住院时间（LOS）的"平均节省时间"。

## 研究问题与动机
- **医院QI需要因果排序**：医院QI项目面临多个候选干预措施，只能实施少数，必须按预期因果效应排序优先级，而现有回归模型仅估计关联而非因果。
- **定性分析受限于认知偏见**：专家基于甘特图/价值流图进行定性推理时，能准确预测直接目标事件的时间偏移，但对下游事件的连锁影响难以预测，因事件间交互复杂且无法用固定规则刻画。
- **定量分析面临三大技术障碍**：① 干预效果的临床推理依赖非结构化临床记录而非表格特征；② 新干预措施无历史暴露数据，阳性假设（positivity）无法满足；③ 患者异质性导致统一因果图几乎不可构建，且数据不足以估计图中条件关系。
- **LLM扩展专家推理的可行性**：完全人工执行egg-computation需为每位患者构建DAG并模拟反事实，成本过高；LLM的推理能力使规模化成为可能，且仅需提示而非微调/重训练即可适配新场景。

## 核心贡献（创新点）
1. **首次将甘特图形式化为带概率与因果语义的DAG模型**：证明甘特图可转化为等待时间的局部马尔可夫DAG，建立Gantt chart与因果DAG文献的连接，为后续g-computation提供识别基础。
2. **提出egg-computation（专家引导的g-computation）**：将节点按与干预的关系分为Type I（非后代，直接复制观测时间）、Type II（后代非直接子节点，用数据模型预测）、Type III（干预直接子节点，依赖专家分布），通过g-computation变体递归生成反事实轨迹，证明其在Assumptions 1-7下为无偏估计量。
3. **开发五步LLM辅助管道实现专家推理规模化**：包含资格判定、自由形式临床推理、节点生成、边生成、结构化编码五个LLM调用阶段，采用迭代图扩展与disjunctive criterion自动搜索混杂因素，每步要求引用原始病历以保证可审计性。
4. **在模拟与真实医院数据上验证方法的准确性与专家一致性**：模拟显示当患者因果结构异质且隐性时，单一DAG方法偏差可达14小时，而egg-computation可实现无偏估计；真实数据中LLM生成的DAG与5位专家标注高度一致（特异性>96%，精确度>91%），时间节省估计差异均值仅-2.6至8.1小时。

## 方法详解
**因果模型构建**：
- 设患者基线协变量为$X$，事件集为$\mathrm{D}=\{0,1,\ldots,K\}$，其中$k=0$为入院（源节点），$k=K$为出院/死亡（汇节点，定义结局$Y:=T_K$）。
- 对事件$k$，定义墙钟时间$T_k$与等待时间$W_k$，满足递归关系$T_k = \max_{j\in\mathrm{Pa}(k)} T_j + W_k$（若$\mathrm{Pa}(k)\neq\emptyset$）。
- 局部马尔可夫性质：$W_k \perp T_{\mathrm{Nd}(k)} \mid T_{\mathrm{Pa}(k)}, X, \mathrm{D}$。

**干预表示与假设**：
- 引入非事件节点$A$（二元干预，$A=0$为当前政策，$A=1$为新QI政策），$\mathrm{Pa}(A)=\emptyset$。
- Assumption 1（一致性）：$T_k = T_k(0)$。
- Assumption 2（外生干预）：$W_k(a) \perp A \mid X, \mathrm{D}$。
- Assumption 3（排除限制）：Type I节点$W_k(1)=W_k(0)$；Type II节点条件分布不变。
- Assumption 4（局部马尔可夫）：$W_k(a) \perp T_{\mathrm{Nd}(k)}(a) \mid T_{\mathrm{Pa}(k)}(a), X, \mathrm{D}$。
- Assumption 5-7：专家正确指定DAG、Type II时序模型等于真实条件分布、Type III专家分布等于反事实分布。

**egg-computation算法**：
- Step 1（DAG构建）：专家判定干预适用性，构建因果DAG并按与$A$的关系分类节点。
- Step 2（反事实模拟）：按拓扑序递归生成$T'_k(1)$：
  - Type I（$k\in\mathrm{Nd}(A)$）：直接复制$T'_k(1)=T_k$。
  - Type III（$k\in\mathrm{Ch}(A)$）：从专家分布$p_{\mathrm{expert}}$抽样$W'_k(1)$。
  - Type II（$k\in\mathrm{De}(A)\setminus\mathrm{Ch}(A)$）：用训练模型预测$W'_k(1)\sim p(W_k\mid T_{\mathrm{Pa}(k)}, X, \mathrm{D})$。
- 估计量：$\hat{\tau} = \frac{1}{n}\sum_{i=1}^n (Y_i - Y'_i)$，定理3.1证明其为$\tau=\mathbb{E}[Y(0)-Y(1)]$的无偏估计。

**LLM五步管道**：
1. 自由形式临床推理：从病历中识别决定LOS的事件、干预目标、汇门控事件及沿反向路径的混杂。
2. Gantt图节点生成：去重过滤，为每个事件分配起止时间戳并标记干预直接影响的节点。
3. Gantt图边生成：固定节点集，记录每个事件的时序依赖关系。
4. 结构化节点编码：输出临床类别、事件类型、时间戳。
5. 结构化边编码：输出有向边及单句因果理由和原始病历引用。
- 提示设计原则：早期步骤侧重临床/因果推理，后期步骤侧重结构化输出；采用Guo & Zhao (2026)的迭代图扩展与VanderWeele (2019)的disjunctive criterion自动搜索混杂；粒度目标为因果链跨度不超过24小时。

**Type II时序建模三种选项**：
- 选项A：抽取预定义表格特征训练随机森林/岭回归。
- 选项B：使用sentence embeddings作为特征的岭回归。
- 选项C：基于in-context learning（ICL）的LLM推理，提供15个匹配的训练示例。

## 实验与结果
**模拟研究**（$n=4,000$患者，因果结构异质，50次重复）：
| 方法 | Priority-CT偏差(h) | Priority-MRI偏差(h) |
|------|-------------------|---------------------|
| DAG-extractor + Gantt-timing（oracle） | 0.00 ± 0.00 | 0.00 ± 0.00 |
| DAG-extractor + OLS-timing | -0.35 ± 0.02 | -0.22 ± 0.02 |
| DAG-extractor + kNN-timing | 2.75 ± 0.01 | 2.11 ± 0.01 |
| Single-DAG + Gantt-timing (CT→MRI) | 13.01 ± 0.06 | -2.56 ± 0.07 |
| Single-DAG + Gantt-timing (MRI→CT) | -5.37 ± 0.04 | 14.19 ± 0.07 |
| Single-DAG + OLS-timing (CT→MRI) | 10.73 ± 0.08 | -3.05 ± 0.16 |
| Single-DAG + OLS-timing (MRI→CT) | 7.06 ± 0.11 | 5.47 ± 0.16 |

**核心结论**：单一DAG方法偏差最高达14小时；患者特异性DAG+近似时序模型偏差<0.4小时（OLS）或<2.8小时（kNN）；证明结构误设是主要误差来源。

**真实世界评估**（ZSFG医院，2,193条住院记录，5种常见诊断，11个候选干预）：
- **LLM vs 专家DAG一致性**（24对干预-住院，12对/干预，8对双重标注）：
  - 干预目标特异性：96.5%（两种干预均）
  - 干预目标召回率：88.9%（早期出院规划）/ 95.7%（关键影像）
  - 边精确率：91.8% / 94.2%
  - 边召回率：80.2% / 86.0%
  - 时间节省估计差异均值：-2.6小时（早期出院规划）/ 8.1小时（关键影像），24对中19对估计完全一致。

- **Type II时序模型准确性**（4,790个下游节点，留一住院交叉验证）：
  - ICL（GPT-5）MAE = 24.12小时，优于所有基线（零基线28.30、配对中位数27.23、岭回归34.30、随机森林36.58、embedding+岭33.79）。
  - ICL在8/11个干预上显著优于所有基线；剩余3个干预因等待时间高度零膨胀（48-71%<1小时）而差异不显著。
  - ICL在较长等待时间（>6小时）预测上表现优于回归模型。

- **11个干预的因果排序**（预期影响=适用率×平均节省时间）：
  1. Consult response escalation：18.9小时/筛选住院（适用率71%）
  2. Disposition-critical imaging priority：15.6小时（适用率80%）
  3. Early discharge planning：10.8小时（适用率69%）
  - 与单纯按适用率排序不同：Consult response escalation适用率仅排第2，但因果效应最高；Urgent IR procedure block适用率仅10%但效应排第6（4.8小时）。
  - 89%的bootstrap重采样中前3名保持稳定。
  - 4位临床专家盲评中，3/4将Consult response escalation排第一，与egg-computation一致；无人复现单纯适用率排序。

**计算成本**：1,531个LLM生成DAG，总API成本约\$490（GPT-5-mini \$0.25/百万输入token，GPT-5 \$1.25/百万输入token）。

## 相关工作脉络
1. **Robins (1986) g-computation**：经典纵向因果推断框架，假设固定DAG与条件可交换性；本文将其扩展至甘特图等待时间场景，并引入专家输入处理数据不可识别部分。
2. **Local Independence Graphs（Didelez 2008; Røysland et al. 2025）**：连续时间图的局部独立性模型；本文关注离散事件序列的因果DAG，不假设固定图结构。
3. **Lean healthcare与价值流图（Rother & Shook 1999; Vidal-Carreras et al. 2022）**：定性流程改进工具，缺乏因果识别保证；本文桥接操作实践与形式化因果推理。
4. **LLM用于因果发现（Kıcıman et al. 2024; Antonucci et al. 2024）**：仅从变量名或摘要推断成对/小型因果图；本文扩展至14节点患者特异性DAG的全端到端因果推断。
5. **Cotta et al. (2024) 端到端LLM因果推断**：从互联网文本提取暴露/结局/混杂并用IPW估计；本文关注医院QI干预对时序的因果效应，方法论完全不同。
6. **Guo & Zhao (2026) 迭代图扩展；VanderWeele (2019) disjunctive criterion**：混杂选择准则；本文将其嵌入LLM提示设计，自动搜索干预目标与汇门控事件的反向路径混杂。

## 局限性与未来方向
- **资源竞争未建模**：当前估计量解释为给定相同人群层下的效果，或假设资源无限的条件下界；加速某一患者可能占用资源影响其他患者轨迹。
- **置信区间未建立**：缺乏同时考虑抽样变异与专家输入不确定性的推断程序。
- **专家校准待加强**：目前仅通过少量患者进行提示微调，更广泛的专家校准可进一步提升性能。
- **干预改变轨迹的假设**：Remark 2.1指出若干预大幅改变患者轨迹（如化疗改为手术），D(0)=D(1)假设不成立，框架不适用。
- **DAG结构验证不可检验**：Assumption 5（专家正确指定DAG）无法从数据验证，需在实践中评估合理性。

## 研究启发与可借鉴点
1. **甘特图→因果DAG的形式化转换**：将操作研究中的甘特图（含start-start/finish-start等依赖类型）转化为等待时间的局部马尔可夫DAG，为流程改进提供因果识别框架，可迁移至制造业调度、软件开发流程等场景。
2. **Type I/II/III节点分类策略**：按节点与干预的因果关系分类并选择最优信息来源（专家vs数据），为混合定性-定量因果推断提供通用设计模式。
3. **LLM五步管道设计原则**：早期步骤侧重推理、后期步骤侧重结构化；每步要求引用原始文本保证可审计性；采用disjunctive criterion自动确定混杂集合——这套提示工程模式可直接复用于其他需要专家级因果推理的领域。
4. **ICL用于时序预测的竞争优势**：在长尾等待时间预测上，LLM的in-context学习优于传统回归模型，提示在小样本+高噪声时序预测场景中的潜力。
5. **预期影响=适用率×平均效应**的排序指标：比单纯适用率或单纯效应更能反映干预的实际价值，可作为QI项目优先级的通用评估框架。

## 关键术语表
- **egg-computation**：expert-guided g-computation的简称，将专家推理与数据驱动建模结合的因果推断变体，通过甘特图DAG递归生成反事实轨迹。
- **Gantt chart（甘特图）**：用水平条形表示事件、箭头表示依赖关系的时间轴流程图，本文证明其可形式化为因果DAG。
- **Type I/II/III节点**：按与干预节点A的关系分类：Type I为非后代节点（时间直接复制）；Type II为后代非直接子节点（用数据模型预测）；Type III为干预直接子节点（依赖专家分布）。
- **disjunctive criterion（析取标准）**：VanderWeele (2019)提出的混杂选择准则，保留作为干预目标或结局（或两者）原因的任何前置变量。
- **average time saved（平均节省时间）**：目标因果效应$\tau=\mathbb{E}[Y(0)-Y(1)]$，衡量新干预相比当前政策的期望住院时间减少量。
- **local Markov property（局部马尔可夫性质）**：本文定义的等待时间条件独立关系$W_k \perp T_{\mathrm{Nd}(k)} \mid T_{\mathrm{Pa}(k)}, X, \mathrm{D}$，区别于标准贝叶斯网络的节点独立性。
- **SUTVA（稳定单位处理值假设）**：No interference假设，本文假设每位患者的干预效果独立，不受其他患者资源竞争影响。
- **FFRCISTG（单世界干预因果图）**：Robins (1986)的因果语义框架，本文据此定义潜在等待时间$W_k(a)$与DO算子。

## 可复现要素
- **数据集**：Zuckerberg San Francisco General Hospital（ZSFG）成人住院记录，5种最常见诊断，2,193条包含临床笔记、医嘱、检验结果的时间线；论文未声明完全公开，但暗示数据来自prior work（Vossler et al. 2026）。
- **代码/权重**：提示模板与软件包将随开源包发布（论文多次提及"all prompts will be available in an open-source software package"，见Appendix C/D）；使用GPT-5与GPT-5-mini API。
- **关键超参**：LLM管道每DAG平均14节点21边；Type II ICL使用15个训练示例；粒度目标为因果链跨度≤24小时；模拟$n=4,000$，50次重复。

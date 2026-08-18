---
title: "Expert-Guided g-computation with Large Language Models for Estimating Causal Efects on Timings: Applications to Hospital Quality Improvement"
source: https://arxiv.org/pdf/2608.10339v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:47:52"
field: "因果推断与大模型交叉"
keywords: ["因果推断", "g-computation", "大型语言模型", "Gantt图", "医院质量改进", "反事实模拟", "专家引导推理"]
innovations: ["将Gantt图形式化为具有概率与因果语义的DAG模型并提出egg-computation识别框架", "设计五步LLM辅助流水线规模化专家级DAG构建与反事实等待时间模拟", "在真实医院11个QI干预上验证LLM-egg-computation与人工专家高度一致且优于传统因果方法"]
benchmarks: ["模拟研究（4000患者，2种干预，50 replications）", "ZSFG医院真实数据（2193住院轨迹，11个候选干预）"]
---

# 论文速读：Expert-Guided g-computation with Large Language Models for Estimating Causal Efects on Timings: Applications to Hospital Quality Improvement

## 一句话总结
本文提出**expert-guided g-computation（egg-computation）**框架，将Gantt图转化为因果DAG并结合g-computation变体，仅需专家提供数据无法识别的干预直接效应部分；通过LLM辅助流水线规模化专家推理，在模拟和真实医院数据上验证了其对估计候选QI干预"平均节省时间"因果效应的高准确性与专家一致性。

## 研究问题与动机
1. **医院QI项目需要因果排序**：医院面临多个候选干预措施，但只能实施少数几个，需基于预期影响（如对平均住院时长LOS的因果效应）进行优先级排序。
2. **定性方法存在认知偏差**：传统基于专家判断的Gantt图分析依赖临床推理，难以预测干预对下游事件的间接影响（如周末服务限制、多事件交互等）。
3. **定量方法存在根本性局限**：回归模型估计的是关联而非因果；新干预措施历史上从未实施（违背positivity假设）；患者轨迹复杂且异质，难以构建统一的因果图；结构化数据不足以满足混杂控制需求。
4. **已有因果推断方法不适用**：传统g-computation等方法假设共同因果图且依赖完整数据支持，而医院干预的因果机制常含非结构化临床文本信息且个体结构各异。

## 核心贡献（创新点）
1. **将Gantt图形式化为具有概率与因果语义的DAG模型**：通过等待时间（waiting time）变量定义DAG上的局部马尔可夫性质，建立干预节点A与事件节点的因果联系，证明在Assumptions 1-7下目标 estimand $\tau = \mathbb{E}[Y(0)-Y(1)]$ 可识别。
2. **提出egg-computation（专家引导的g-computation）**：将节点分为Type I（非干预后代，直接复制观测时间）、Type II（干预后代但非直接目标，用数据模型预测）、Type III（干预直接目标，由专家/LLM提供分布），按拓扑序递归模拟反事实轨迹，实现混合专家知识与统计建模的因果估计。
3. **设计五步LLM辅助DAG构建流水线**：从临床自由文本推理→Gantt图节点/边定义→结构化编码，通过迭代图扩展与disjunctive criterion自动搜索混杂因素，平均生成14节点/21边的患者特异性DAG，并包含可审计的引用链。
4. **在真实医院场景中规模化验证**：在ZSFG医院11个候选QI干预、2,193条住院轨迹上运行LLM-egg-computation，生成1,531个DAG，成本仅\$490；LLM生成的DAG与人工专家高度一致（edge precision 91.8%~94.2%，target specificity 96.5%），时间节省估计差异均值为-2.6~8.1小时。

## 方法详解
**因果模型设定（Section 2）**：
- 每个事件$k$有等待时间$W_k$和壁钟时间$T_k$，关系为$T_k = \max_{j \in \mathrm{Pa}(k)} T_j + W_k$（无父节点时$T_k = W_k$）。
- DAG满足局部马尔可夫性质：$W_k \perp T_{\mathrm{Nd}(k)} \mid T_{\mathrm{Pa}(k)}, X, \mathrm{D}$。
- 引入二元干预节点$A$，Assumption 3（排除限制）区分Non-descendants（$W_k(1)=W_k(0)$）与Descendants但不直接受干预（条件分布等价）。

**egg-computation三步分类与模拟（Box in Section 3.1）**：
1. **Type I（$k \in \mathrm{Nd}(A)$）**：直接复制观测时间$W_k(1) = W_k$。
2. **Type II（$k \in \mathrm{De}(A) \setminus \mathrm{Ch}(A)$）**：用观测数据训练的$p(W_k \mid T_{\mathrm{Pa}(k)}, X, \mathrm{D})$模型预测，Assumption 6要求模型等于真实条件分布。
3. **Type III（$k \in \mathrm{Ch}(A)$）**：由专家/LLM提供$p_{\mathrm{expert}}(W_k(1) \mid T_{\mathrm{Pa}(k)}(1), X, \mathrm{D})$分布，Assumption 7要求专家指定分布等于真实反事实分布。

最终估计量：$\hat{\tau} = \frac{1}{n}\sum_{i=1}^n (Y_i - Y_i')$，其中$Y_i'$为反事实LOS。

**LLM流水线（Section 3.2）**：
- **Step 1（资格判定+DAG构建）**：五步LLM调用——自由文本临床推理→Gantt图节点→Gantt图边→结构化节点→结构化边。通过从sink节点和Type III节点反向追溯搜索混杂，采用disjunctive criterion停止规则，平均14节点/21边。
- **Step 2（Type II Timing模型）**：三种选项——（a）提取表格特征+随机森林/OLS；（b）sentence embeddings + ridge；（c）ICL（GPT-5，15个相似训练示例few-shot）。推荐cross-validation选择。
- **Step 2（Type III Timing）**：通过详细prompt_specification干预描述（动机、规则、示例、纳入/排除标准），必要时迭代调优对齐专家分布，建议对干预直接效应做敏感性分析。

## 实验与结果
**模拟研究（Section 4.1）**：
- 设置4,000名患者，因果关系（CT→MRI or MRI→CT）因人而异，两种干预分别限制CT/MRI报告时间≤12小时。
- **关键结果**：DAG-extractor + Gantt-timing（oracle配置）偏差为**0.00±0.00小时**；Single-DAG方法偏差高达**13.01±0.06**（Priority-CT）和**14.19±0.07**（Priority-MRI）；即使Type II使用 imperfect timing模型（OLS偏差-0.35~-0.22小时，kNN偏差2.75/2.11小时），患者特异性DAG仍大幅优于单一共享DAG。

**真实数据评估（Section 4.2）**：
- **数据集**：ZSFG医院2,193条住院轨迹（5种最常见诊断），11个候选QI干预。
- **DAG一致性**：LLM与5位专家 annotators 对比（24对干预-住院对，8对双重标注）。Type III target specificity 96.5%（两干预均），recall 88.9%/95.7%；edge precision 91.8%/94.2%，recall 80.2%/86.0%。重跑egg-computation后平均时间节省差异：-2.6h（[-8.3, 0.8]）和 8.1h（[0.0, 24.2]），19/24对估计完全相同。
- **Type II Timing模型**：ICL (GPT-5) MAE = **24.12小时**，优于Zero gap (28.30)、Pair-type median (27.23)、Ridge (34.30)、Random forest (36.58)、Embedding+ridge (33.79)。ICL在8/11个干预上显著优于所有baseline。
- **最终干预排名（按expected impact = eligibility rate × avg time saved）**：
  1. Consult response escalation: 18.9h [11.4, 26.9]（eligibility 71%）
  2. Disposition-critical imaging priority: 15.6h [10.2, 22.1]（eligibility 80%）
  3. Early discharge planning: 10.8h [5.0, 17.5]（eligibility 69%）
  Top-3在89%的bootstrap resamples中稳定。
- **临床验证**：4位clinician独立评审，3/4认为egg-computation排名比纯eligibility排名更符合临床经验直觉。

**成本**：总LLM API成本\$490（含prompt caching折扣），生成1,531个DAG，平均每DAG约\$0.14。

## 相关工作脉络
1. **Robins (1986) g-computation**：经典纵向因果推断方法，通过iterative imputation估计时间加权干预效应；本文将其扩展至Gantt图场景并引入专家输入处理数据不可识别部分。
2. **Kıcıman et al. (2024)**：证明LLM可从变量名推断pairwise因果方向；本文相比其在小规模（≤5节点）well-understood setting下的零样本能力，扩展到大规模（~14节点）、复杂临床文本、多步骤迭代构建的因果图。
3. **Cotta et al. (24)**：端到端LLM因果推断，从互联网文本提取暴露/结局/混杂后运行IPW；本文不同在于（a）使用Gantt图作为中介表示；（b）混合专家+数据建模而非纯LLM推理；（c）面向医院QI的counterfactual trajectory simulation。
4. **Lean healthcare / Gantt chart实践**（Rother & Shook, 1999; Vidal-Carreras et al., 2022）：定性流程优化工具，缺乏形式化因果识别保证；本文 bridging操作研究与因果推断，赋予Gantt图概率与因果语义。
5. **Guo & Zhao (26) iterative graph expansion** & **VanderWeele (19) disjunctive criterion**：混杂因素选择理论；本文将其嵌入LLM prompt design，指导DAG构建时从干预目标和sink节点反向追溯以保留充分混杂调整集。
6. **Hegselmann et al. (22) TabLLM**：few-shot LLM分类表格数据；本文Type II timing中的ICL选项与其发现一致——在复杂数据、小样本场景下few-shot LLM可与训练好的表格模型竞争。

## 局限性与未来方向
1. **SUTVA假设限制**：当前框架假设患者间无interference，但医院资源有限，加速某一患者的护理可能延迟其他患者；框架给出的是"无限资源"下的upper-bound估计。
2. **DAG正确性不可检验**：Assumption 5（专家DAG正确指定）无法从数据验证，若因果图结构错误则估计有偏；虽可通过专家审核缓解，但规模扩大后难以保证。
3. **未处理事件集变化**：当前假设$\mathrm{D}(0) = \mathrm{D}(1)$，即干预不改变事件集合本身；对于从根本上改变治疗路径的干预（如化疗换手术）不适用。
4. **不确定性量化不足**：目前仅提供point estimate，缺乏confidence interval来同时捕获抽样变异和专家输入的不确定性；future work需形式化inferential procedures。
5. **LLM依赖与漂移风险**：pipeline基于GPT-5-mini/5的prompt而非fine-tuning，虽然避免了retraining成本，但模型升级可能导致behavior shift，需持续calibration。

## 研究启发与可借鉴点
1. **DAG构建的"分步细化"策略**：五步LLM流水线中前两步侧重临床/因果推理（低结构约束），后三步侧重格式化和验证（高结构约束），这种"reasoning-then-formatting"的token分配策略值得迁移至其他复杂结构化输出任务。
2. **Type I/II/III节点分类思想**：将因果图中节点按"干预直接作用范围"分层，分别采用纯复制/数据建模/专家输入三种策略——这一分类框架可推广至其他需混合专家知识与数据的因果推断场景（如制造工艺优化、软件开发流程评估）。
3. **ICL for temporal prediction**：Type II waiting time预测中，15-shot ICL以24.12h MAE优于多种监督模型，提示在**事件描述为自由文本、条件配置罕见重复、等待时间分布零膨胀**的场景下，LLM的contextual reasoning可能天然适配，可作为强baseline。
4. **Prompt engineering中的正式统计语言**：pipeline刻意使用formal statistical language（如"conditional exchangeability"、"ancestral"）激发LLM更严谨推理——这一设计原则对任何需要LLM输出符合数学/统计约束的任务均有参考价值。
5. **与临床专家co-design的annotation interface**：PHI-compliant HTML接口支持逐节点/逐边verdict、引用溯源、inter-annotator双标注，为医疗AI中human-in-the-loop验证提供了可复用范式。

## 关键术语表
**egg-computation（expert-guided g-computation）**：本文提出的因果估计框架，结合专家对干预直接效应（Type III节点）的判断与数据模型对下游事件（Type II节点）的预测，沿Gantt DAG拓扑序递归模拟反事实轨迹以估计平均节省时间。

**Type I/II/III节点**：按与干预节点A的DAG关系分类：Type I为非后代（时间直接复制）；Type II为间接后代（用观测数据模型预测等待时间）；Type III为直接子节点（由专家/LLM提供反事实分布）。

**g-computation**：Robins (1986) 提出的因果效应估计方法，通过在DAG上按拓扑序iteratively impute每个节点的干预后分布，最终计算结局期望值的差。

**disjunctive criterion（联合标准）**：VanderWeele (2019) 提出的混杂选择准则：保留所有导致暴露或导致结局的变量，确保至少覆盖一个充分调整集以避免残余混杂。

**FFRCISTG（单世界故障分布因果图）**：Richardson & Robins (2013) 的形式化因果语义，将DAG与potential outcomes联系，本文Assumptions 1-4即在此框架下定义。

**Local Markov property（for waiting times）**：本文定义的DAG马尔可夫性质，表述为$W_k \perp T_{\mathrm{Nd}(k)} \mid T_{\mathrm{Pa}(k)}, X, \mathrm{D}$，区别于标准Bayesian network因wall times存在时序约束。

**In-context learning (ICL)**：Brown et al. (2020) 提出的LLM少样本推理范式，本文在Type II timing预测中使用15个相似患者案例作为prompt示例，无需额外模型训练。

**Average Time Saved ($\tau$)**：本文目标estimand，定义为$\mathbb{E}[Y(0) - Y(1)]$，即无干预与有干预条件下平均住院时长的期望差。

## 可复现要素
- **数据集**：ZSFG医院2,193条成人住院轨迹（5种最常见诊断），来自Vossler et al. (2026)先前工作；**论文未明确声明公开**，但提到代码/pipeline prompts将在open-source软件包中提供。
- **代码**：论文在Appendix D附了Type III和Type II的prompt模板，声称完整prompts将在 accompanying software package 中开源（链接未在全文正文给出，需在arXiv页面查看）。
- **模型**：DAG构建使用 **GPT-5-mini**，Type II timing ICL使用 **GPT-5**；非LLM baseline包括OLS、kNN、Random Forest、Ridge（含sentence embeddings）。
- **关键超参**：LLM pipeline每住院最多30例（因诊断组×干预）；Type II ICL用15个训练示例；DAG平均14节点/21边；kNN中k由leave-one-out CV选取但对结果不敏感（Appendix F显示MAE和bias在$k$ sweep下flat）。
- **模拟数据**：$n=4{,}000$患者，edge均匀随机采样，50次replication，代码/seed未在论文中提供。

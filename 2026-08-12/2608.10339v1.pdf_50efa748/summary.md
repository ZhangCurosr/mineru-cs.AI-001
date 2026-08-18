---
title: "Expert-Guided g-computation with Large Language Models for Estimating Causal Efects on Timings: Applications to Hospital Quality Improvement"
source: https://arxiv.org/pdf/2608.10339v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:25:17"
field: "因果推断与大语言模型交叉"
keywords: ["因果推断", "g-computation", "大型语言模型", "甘特图", "医院质量改进", "反事实推理", "专家知识"]
innovations: ["将甘特图形式化为因果 DAG 并建立等待时间因果模型，首次为 Gantt 图提供因果识别保证", "提出 egg-computation，按三类型节点分别采用专家输入与数据统计建模实现专家-数据互补", "构建五步 LLM 辅助管道，实现从非结构化临床笔记到患者级个性化 DAG 的大规模自动化构建"]
benchmarks: ["ZSFG 医院 2193 条住院记录（5 个诊断组，11 项 QI 干预）", "模拟研究（n=4000，CT/MRI 依赖结构异质性）"]
---

# 论文速读：Expert-Guided g-computation with Large Language Models for Estimating Causal Effects on Timings: Applications to Hospital Quality Improvement

## 一句话总结
本文提出 **expert-guided g-computation（egg-computation）**，将专家对甘特图（Gantt chart）的定性推理与基于数据的因果建模相结合，借助 LLM 自动化专家推理流程，用于估计尚未实施的新干预措施对平均住院时长（LOS）的因果效应。模拟与真实医院数据分析表明，该方法在处理患者特异性因果结构异质性时显著优于传统因果推断方法，且 LLM 提取的 DAG 与人类专家高度一致。

## 研究问题与动机
- **医院质量改进（QI）优先级排序缺乏因果支撑**：医院需从多个候选干预措施中选择 1–2 个实施，依据是预期对 LOS（平均住院天数）的因果影响，但现有方法无法准确估计假设性新政策的因果效应。
- **定性方法的局限**：依赖专家基于 Gantt 图的手工推理，虽能准确估计直接目标事件的时间偏移，但对下游事件的级联效应难以预测（受多事件交互及周末/节假日等实际约束影响），且存在认知偏差。
- **定量因果推断方法的局限**：回归模型估计的是关联而非因果；面对完全假设的新政策，**positivity 假设**无法满足；临床笔记等非结构化数据中的因果信息被忽略；甚至统一的因果 DAG 在异质性患者群体中几乎不可构建。
- **因果推断理论框架缺失**：将操作研究中的 Gantt 图与正式因果 DAG 文献建立等价关系，并推导识别条件，填补了这一空白。

## 核心贡献（创新点）
1. **首次为 Gantt 图建立形式化因果模型**：将 Gantt 图等价于事件等待时间 DAG，证明在标准因果假设下可识别干预对平均时间节省（$\tau = \mathbb{E}[Y(0)-Y(1)]$）的效应；与既有工作本质区别在于：Gantt 图此前仅在运营研究中作为可视化工具使用，本文赋予其概率与因果语义。
2. **提出 egg-computation（专家引导的 g-computation 变体）**：将节点按与干预的关系分为三类（Type I/II/III），分别采用事实复制、统计建模、专家输入三种信息源；与经典 g-computation 的本质区别在于：经典方法仅用数据，本文在数据不足处引入专家判断，实现"专家强项 + 数据强项"的互补。
3. **构建五步 LLM 辅助管道实现大规模专家推理扩展**：以分步提示驱动 LLM 从非结构化临床笔记中抽取患者级 DAG（平均 14 节点/21 边），并结合 in-context learning（ICL）预测 Type II 节点等待时间；与既有 LLM 因果发现工作的本质区别在于：不追求群体级因果图，而是构建患者级个性化 DAG 并执行完整反事实模拟。
4. **在真实医院场景中验证 LLM 管道与人类专家高度一致**：LLM 生成的 DAG 在节点分类（ specificity 96.5%，recall 88.9%）和边抽取（precision 91.8%，recall 80.2%）上与人类专家注释高度吻合，且 11 项候选 QI 干预的因果排序经临床医生评估比单纯基于覆盖率的排序更符合临床直觉。
5. **ICL 在等待时间预测任务上优于传统回归基线**：GPT-5 ICL 的 MAE 为 24.1 小时，显著优于零延迟基线（28.3h）、配对类型中位数（27.2h）及多种表格回归模型，且在 8/11 干预中差异统计显著；与既有表格因果推断工作的本质区别在于：证明少样本 LLM 推理可直接替代复杂的特征工程与模型训练。

## 方法详解
- **Gantt 图的因果建模**：每个事件 $k$ 有等待时间 $W_k$ 和墙钟时间 $T_k$，关系为 $T_k = \max_{j \in \text{Pa}(k)} T_j + W_k$（无父节点时 $T_k = W_k$）。DAG 满足局部马尔可夫性质：$W_k \perp\!\!\!\perp T_{\text{Nd}(k)} \mid T_{\text{Pa}(k)}, X, D$。引入二元干预节点 $A$（$A=0$ 当前政策，$A=1$ 新 QI 政策），形成潜在等待时间 $W_k(a)$。
- **识别假设**：一致性（$W_k = W_k(0)$）、外生干预（$W_k(a) \perp\!\!\!\perp A \mid X, D$）、排除限制（Type I 节点等待时间不变；Type II 节点条件分布不变）、局部马尔可夫性质。在此基础上，反事实联合分布可因子化：$p(T_D(1)|X,D) = \prod_k p(T_k(1)|T_{\text{Pa}(k)}(1), X, D)$。
- **三类型节点处理**：
  - **Type I**（干预的非后代）：$W_k(1) = W_k$，直接复制观察时间。
  - **Type II**（干预的后代但非直接子节点）：用观测数据拟合的时序模型预测，$W_k(1) \sim p(W_k | T_{\text{Pa}(k)}, X, D)$。
  - **Type III**（干预的直接子节点）：由专家（或对齐的 LLM）提供分布 $p_{\text{expert}}$ 采样。
- **egg-computation 算法**（Box 中描述）：按拓扑序递归生成反事实时间戳 $T'_k(1)$，最终 $\hat{\tau} = \frac{1}{n}\sum_i (Y_i - Y'_i)$；定理 3.1 证明其在假设 1–7 下是无偏估计。
- **LLM 五步 DAG 构建管道**：Step 1 自由文本临床推理（识别目标节点、sink-gating 事件、向后追踪混杂因素）；Step 2–3 形式化为 Gantt 图节点与边；Step 4–5 编码为结构化图格式（含引用引用以保证可审计性）。提示设计借鉴迭代图扩展与析取标准（disjunctive criterion）。
- **Type II 时序建模**：三种方案对比——表格特征 + 回归、sentence embedding + ridge、**ICL（GPT-5，few-shot 示例）**。推荐通过交叉验证选择。Type III 节点通过详细干预描述 + 迭代 prompt tuning 使 LLM 输出与专家分布对齐。

## 实验与结果
- **模拟研究**（$n=4000$，患者因果结构两两不同：CT→MRI vs. MRI→CT）：
  - **DAG-extractor + Gantt-timing**（理论最优）：偏差 $0.00 \pm 0.00$ 小时（两项干预均）。
  - **Single-DAG + Gantt-timing（错误方向）**：偏差高达 **13.01 ± 0.06h**（Priority-CT）和 **14.19 ± 0.07h**（Priority-MRI）。
  - DAG-extractor + OLS/kNN-timing 偏差仅 -0.35/2.75h，表明**结构误设是主要误差来源**。
- **真实世界评估**（ZSFG 医院，2,193 条入院记录，5 个常见诊断组，11 项候选 QI 干预）：
  - LLM DAG 与专家一致性（Table 2）：specificity 96.5%，precision 91.8%，recall 80–86%；重新运行 egg-computation 后估计的平均时间节省差异仅 -2.6h / 8.1h（19/24 对结果相同）。
  - **Type II 时序预测**（Table 3，4,790 个下游节点）：**ICL (GPT-5) MAE = 24.1h**，优于 Zero gap（28.3h）、Pair-type median（27.2h）、Ridge（34.3h）、Random forest（36.6h）、Embedding+ridge（33.8h）。
  - **11 项干预排序**（Table 10）：按预期影响排序前三为 **Consult response escalation（18.9h/screened）**、**Disposition-critical imaging priority（15.6h）**、**Early discharge planning（10.8h）**；该排序与仅按覆盖率排序的结果不同，且 4 位临床医生中 3 位认为 egg-computation 排序更合理。
  - LLM API 总成本约 **$490**（1,531 个 DAG 构建约 \$215）。

## 相关工作脉络
1. **Robins (1986) g-computation**：经典纵向因果推断方法，假设共同 DAG 与条件交换性；本文扩展至患者级个性化 DAG 且允许专家填补数据盲区。
2. **Lean healthcare / Gantt chart 运营研究**（Rother & Shook, 1999；Vidal-Carreras et al., 2022）：Gantt 图长期用于流程改进可视化但缺乏因果识别保证；本文将其形式化为因果 DAG 并给出识别定理。
3. **LLM 因果发现**（Kıcıman et al., 2024；Antonucci et al., 2024；Cotta et al., 2024）：已有工作证明 LLM 可从变量名或文本推断因果边；本文独特之处在于面向患者级 DAG 构建并执行完整反事实模拟而非仅图推断。
4. **专家知识在因果推断中的作用**（VanderWeele, 2019；Guo & Zhao, 2026）：传统方法依赖专家指定混杂变量；本文将专家角色延伸至等待时间分布的指定，并通过 LLM 规模化。
5. **离散事件仿真与排队论在医疗流程中的应用**（Banks, 2005；Green, 2006）：缺少因果识别框架；本文桥接运营研究与因果推断，提供识别保证。
6. **LLM 模拟人类推理**（Argyle et al., 2023；Aher et al., 2022）：证明 LLM 可模仿人类思维；本文具体应用于医院 QI 领域的专家引导因果推理。

## 局限性与未来方向
- **SUTVA 假设不现实**：假设患者间无干扰，但医院资源有限，加速一位患者的护理可能延迟另一位患者（讨论部分已提及）。
- **干预不改变事件集**：当前框架要求 $\text{D}(0) = \text{D}(1)$，不适合根本性改变治疗路径的干预（如化疗改为手术）。
- **不确定性量化不足**：缺乏同时考虑抽样变异和专家输入不确定性的置信区间构造方法。
- **LLM 管道的泛化性待验证**：提示设计和 prompt tuning 需针对新临床场景和新干预重新适配；当前仅在一个安全网医院验证。
- **未来方向**：扩展到批次患者联合分析以处理资源共享约束；与专家判断更 extensive 校准；推广至制造业排程、软件开发等通用甘特图场景。

## 研究启发与可借鉴点
1. **专家-数据互补的节点分类策略可迁移**：Type I/II/III 三分法为"何时用专家、何时用数据"提供了通用决策框架，适用于任何具有任务依赖关系和等待时间的过程建模（制造、物流、项目调度）。
2. **分步提示设计降低 LLM 复杂推理难度**：五步管道将 DAG 构建拆分为临床推理→节点提取→边提取→结构化编码→验证，每一步聚焦不同认知任务，此设计模式可复用于其他需要 LLM 生成结构化因果图的任务。
3. **ICL 在时序预测中的潜力**：少样本 LLM 推理在等待时间预测上优于传统表格模型，提示在"精确重复的条件配置几乎不存在"的场景下，ICL 可利用语义相似性而非精确匹配，值得在其他序列预测任务中验证。
4. **反事实甘特图可视化增强专家信任**：LLM 生成 counterfactual Gantt chart 并附患者级示例，使临床专家可审计因果推理过程，这种"因果推断 + 可视化解释"的组合可推广至其他需专家验证的黑盒模型场景。
5. **从覆盖率先行到因果效应排序的范式转换**：传统 QI 优先按 eligible 比例排序，本文证明因果排序与之显著不同（如 Consult response escalation 覆盖率仅排第 2 但因果效应最高），为资源约束下的政策优先级排序提供了新方法。

## 关键术语表
- **egg-computation**：专家引导的 g-computation，结合专家对 Gantt 图的干预修改与统计采样，生成反事实轨迹以估计平均时间节省的因果推断方法。
- **g-computation**：Robins (1986) 提出的因果效应估计框架，通过递归代入干预后的条件分布来模拟反事实结果。
- **Type I / II / III 节点**：按与干预节点 A 的图关系分类：Type I 为非后代（时间不变）；Type II 为间接后代（用数据模型预测）；Type III 为直接子节点（由专家/LLM 提供分布）。
- **Gantt chart（甘特图）**：用水平条形表示事件及其依赖关系的流程图，本文将其形式化为事件等待时间 DAG。
- **Disjunctive criterion（析取标准）**：混杂选择准则，要求纳入至少是暴露因果后代或结局因果前因的基线变量，本文用于指导 LLM 向后追踪混杂因素。
- **In-context learning (ICL)**：利用少量示例提示 LLM 直接输出预测，无需微调；本文用于 Type II 等待时间预测。
- **SUTVA（稳定单元处理值假设）**：假设个体间无干扰、干预只有一个版本；本文指出的重要局限之一。
- **Average Time Saved（平均时间节省）**：目标因果量 $\tau = \mathbb{E}[Y(0) - Y(1)]$，即当前政策与候选干预下的平均住院时长之差。

## 可复现要素
- **数据集**：Zuckerberg San Francisco General Hospital (ZSFG) 2,193 条成人住院记录（5 个常见诊断组），基于 Vossler et al. (2026) 同数据管道提取；**数据受 HIPAA 保护，论文未公开**。
- **代码/权重**：LLM 提示模板在配套开源软件包中提供（论文未给出具体 GitHub 链接）；基础模型为 GPT-5 和 GPT-5-mini，需 API 访问。
- **关键超参**：LLM 管道每干预最多处理 30 条 eligible 住院记录；DAG 平均 14 节点 / 21 边；ICL 使用 15 个 few-shot 示例；结构验证 tolerance 为几分钟级时间不一致自动修正。
- **其他**：模拟数据为人工合成（$n=4000$），因果结构随机采样；所有提示模板见附录 D。

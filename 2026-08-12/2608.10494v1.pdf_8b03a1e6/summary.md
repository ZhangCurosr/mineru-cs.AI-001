---
title: "GeoForge: Non-Parametric Self-Evolving Agents for Earth-Observation Reasoning"
source: https://arxiv.org/pdf/2608.10494v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:39:55"
field: "地球观测智能体"
keywords: ["Earth Observation", "LLM Agent", "Self-Evolving", "Non-Parametric Memory", "Tool-Use Planning", "Remote Sensing"]
innovations: ["三级非参数记忆架构（工作流图+动作级经验+适配SOP）协同指导EO工具使用", "安全门控轨迹蒸馏实现无参数更新的自进化", "任务条件化检索增强结合模态感知工具过滤的免训练执行框架"]
benchmarks: ["Earth-Bench", "ThinkGeo", "GeoPlan-Bench"]
---

# 论文速读：GeoForge: Non-Parametric Self-Evolving Agents for Earth-Observation Reasoning

## 一句话总结

GeoForge 提出了一种免训练、自进化的地球观测（EO）智能体框架，通过将历史执行轨迹蒸馏为三层结构化非参数记忆（工作流图记忆、动作级经验库、适配技能SOP），在推理前进行任务条件过滤与检索，持续指导工具选择与工作流编排，而无需更新骨干 LLM 参数。

## 研究问题与动机

- **核心问题**：EO 分析工作流受传感语义、产品依赖关系、时空兼容性以及参数约束的严格限制，现有智能体暴露于宽泛异构工具空间后，容易选取不匹配的操作、产生无效参数或不完整的工作流。
- **现有 EO 智能体的不足**：基于通用 ReAct 范式构建工作流，缺乏对遥感数据流（数据发现→产品生成→时空聚合→科学解释）的显式先验指导，导致冗余调用、不兼容操作和轨迹断裂（图1示例）。
- **自进化方法的局限**：已有自进化系统（如 GeoEvolve 演化算法实现、GeoEvolver 积累交互级经验）未能跨"全局操作顺序→局部动作修正→程序与数据约束"三个粒度组织可复用知识；原始轨迹保留过多实例细节，而文本摘要又可能遗漏决定工作流有效性的产品依赖与参数条件。
- **设计动机**：需要将已完成轨迹转化为领域有效、结构化、任务条件化且安全可更新的执行先验，使智能体在每次任务后持续改进规划，而不扰动 LLM 权重。

## 核心贡献（创新点）

1. **免训练自进化框架 GeoForge**：通过持续将遥感执行轨迹蒸馏为结构化知识表示，实现跨异构地理空间模态的持续规划与工具使用能力提升，与微调类方法本质区别在于完全不扰动 LLM 权重。
2. **三级非参数记忆+任务感知执行策略**：融合模态感知工具过滤、工作流图记忆（全局顺序）、动作级经验库（局部修正）和适配技能 SOP（程序/数据/参数约束），提供结构化规划先验，与仅存储交互结果或演化代码的实现形成对比。
3. **多层次互补记忆的协同机制**：工作流图提供全局拓扑与顺序约束，SOP 提供流程语义与失败预防规则，经验库提供局部条件-动作修正；三者在不同决策粒度上互补，区别于单粒度记忆（如纯经验库或纯 SAGE 图内存）的设计。

## 方法详解

**问题建模**：EO 任务表示为 $\boldsymbol{x} = (q, d, \boldsymbol{y})$（科学请求、数据集合、答案空间），操作库为 $\mathcal{T} = \{T_i\}_{i=1}^n$，轨迹 $\tau = (q, a_1, o_1, \ldots, a_m, o_m, y)$，其中 $o_t = T_i(\theta_t)$ 为工具调用产出。有效轨迹须遵循遥感数据流顺序：识别相关观测→生成产品→传播产物至下游工具→检查时空兼容性后进行聚合。

**工作流图记忆（Workflow Graph Memory）**：将高层执行知识表示为有向可靠性感知图 $\mathcal{G} = (\mathcal{W}, \mathcal{A}, \mathcal{U})$，节点 $w_i = (c_i, r_i, \mathbf{z}_i, \mathbf{p}_i, \mathbf{b}_i, Q_i, n_i^+, n_i^-)$ 编码任务类型、工作流描述、压缩工具序列、参数提示、注意事项、溯源信息及成败计数。检索时计算混合评分：
$$s_G(q, w_i) = \alpha \cdot J_{\mathrm{tool}}(q, w_i) + \beta \cdot J_{\mathrm{text}}(q, w_i) + \lambda_C \mathbf{1}[c_i = \hat{c}(q)] - \lambda_L \max(0, |\mathbf{z}_i| - L_0)$$
即工具序列 Jaccard 相似度（结构）+ 文本语义相似度 + 任务类型一致性奖励 + 序列长度惩罚。仅超阈值的节点序列化为规划先验。

**动作级经验库（Action-Level Experiences）**：存储细粒度条件-动作规则 $e_i = (\chi_i, \alpha_i, \mathbf{a}_i, \rho_i, m_i)$，修正局部 EO 决策（模态/时间对齐、参数校准、产物复用、空间不兼容处理、充分证据后停止）。检索评分：
$$s_E(q, e_i) = \frac{|V(q) \cap V(\chi_i \| \alpha_i \| \rho_i)|}{\max(|V(q)|, 1)} + \lambda_E \mathbf{1}[\hat{c}(\chi_i \| \rho_i) = \hat{c}(q)]$$
经验库通过新颖性保持合并演化：仅接受不在已有等价类中的规范化条目，从成功轨迹提炼模式、从成败对比推导错误归因。

**适配技能 SOP（Adapted Skill SOP）**：存储通用任务分解与程序约束 $s_j = (\mu_j, \omega_j, \pi_j, \beta_j, \kappa_j)$ 编码任务、工作流、数据/论证约束、传感器语义、失败预防规则。通过 LLM 压缩器 bounded by $L_S$ 执行任务条件适配：$S_q = A_\Theta(S, q, \mathcal{E}_q; L_S)$，去除不相关/实例细节，保留与当前查询相关的流程与注意事项；更新采用替换式而非追加式。

**证据锚定执行**：推理时拼接上下文 $\mathcal{C}_q = \mathrm{Trunc}_{L_C}[\mathcal{G}_q \| \mathcal{E}_q \| \mathcal{S}_q \| \mathcal{H}_q]$，其中 $\mathcal{H}_q$ 为确定性 EO 指导（先检查数据资产、按模态/时间组织输入、优先兼容批操作、传播产物路径等）。执行标准 ReAct 循环，最终答案通过确定性+LLM 标准化器映射至规范格式，**记忆仅作为处理先验，最终答案必须基于当前观测**。

**安全门控自进化**：执行后轨迹 $\bar{\tau}$ 经蒸馏器生成 $(\Delta \mathcal{E}, \Delta \mathcal{S}, \Delta \mathcal{W})$，更新需通过安全门控 $\psi(\tau, \hat{y})$ 检查：工具调用数>0、答案非截断、调用次数 $\leq L_{\max}$、单工具重复 $\leq R_{\max}$、SOP 候选不含禁止模式。更新后 $\mathcal{M}_{t+1} = (U_G(\mathcal{G}_t, \Delta\mathcal{W}), U_E(\mathcal{E}_t, \Delta\mathcal{E}), U_S(\mathcal{S}_t, \Delta\mathcal{S}; \psi))$。

## 实验与结果

**数据集**：Earth-Bench（Feng et al. 2025）、ThinkGeo（Shabbir et al. 2025）、GeoPlan-Bench（Li et al. 2025），覆盖可执行 EO 分析、光学/SAR 工具推理、高层地理空间规划。

**评估基线**：Earth-Agent、GeoEvolver、OpenEarth-Agent、CoT、ReAct、Plan&Execute、Debate、AFlow、EarthAgent-MAS、GPT-Agent、MGX、Manus、Coze 等。

**主要结果**：
- **Earth-Bench**：GeoForge 在5个 LLM 骨干上均取得最佳或最优之一：GPT-5 达 **74.33%**（+11.17 vs Earth-Agent 63.16%）、DeepSeek-V3.1 达 **77.09%**（+24.86 vs Earth-Agent 52.23%）、Gemini-2.5-Flash 达 **67.91%**（+12.85）、Qwen3-Max 达 **69.72%**（+22.35）。Tool-Any-Order 最高达 83.58%（DeepSeek-V3.1）。
- **ThinkGeo**：Instruction Alignment **97.27%**，Tool Accuracy **80.31%**，Answer Accuracy **60.98%**，均超越 GeoEvolver（46.88/53.74）和所有通用模型。
- **GeoPlan-Bench**：$\mathrm{Recall}_{key}$ **1.00**，$\mathrm{F1}_{key}$ **0.77**，Structural **0.79**，Holistic **1100.65**，全面超越 EarthAgent-MAS、GeoEvolver 及传统 planner。
- **模态泛化**（Table 4）：平均精度 **61.85%**（vs Earth-Agent 47.84%，+14.01）；Spectrum +27pp 至 77.00%，Products +29.15pp 至 71.26%。
- **消融**：去掉 Skill 导致准确率最大降幅（74.33%→52.66%）；去掉 Graph 对轨迹质量影响最大（Tool-Any-Order 85.89%→75.69%）。
- **误差分析**：改进主要来自工具规划错误的显著减少，推理错误对多数模型已消除；剩余错误主要为执行轨迹与参数相关问题。

## 相关工作脉络

- **ReAct / Memento / MemSkill / SAGE**：通用 agent 自进化方向，通过反思学习、可读可写记忆或图记忆引擎实现无参数更新改进；本文区别于它们的是针对 EO 领域特有约束（传感语义、时空兼容）进行多层记忆组织。
- **GeoEvolve (Luo et al. 2025)**：多智能体协作自动化发现地理空间算法，聚焦于算法层面的演化；本文聚焦于工具使用顺序与工作流可复用性，不生成新代码。
- **GeoEvolver (Dai et al. 2026)**：积累交互级工具执行经验库；本文扩展至三级记忆（全局工作流图+局部经验+SOP），覆盖比 GeoEvolver 更宽的粒度层次。
- **Earth-Agent (Feng et al. 2025) / OpenEarthAgent (Dai et al. 2026)**：基于 ReAct 的 EO 智能体基线；本文通过检索增强规划与安全门控蒸馏突破其工具选择与顺序执行瓶颈。
- **TerraBench (Nguyen et al. 2026) / TerraLogic (Yan et al. 2026)**：EO agent 基准测试；本文方法可在这些基准上验证，且实验显示在 TerraBench 类指标（Argument Accuracy、Path Similarity）上优于基线。
- **XSkill (Jiang et al. 2026)**：多模态 agent 的持续经验学习；本文在非参数记忆的结构化程度（图+经验+SOP）和 EO 领域适配性上不同于 XSkill 的通用框架。

## 局限性与未来方向

- **冷启动依赖**：框架依赖初始可用的历史轨迹进行蒸馏，全新任务域在初期可能缺少足够先验，尚未讨论零样本场景下的启动策略。
- **检索质量敏感性**：消融与灵敏度分析表明 Top-k 与阈值设置对性能有影响，过度检索引入噪声，阈值过严丢失有用经验，需在复杂跨模态任务中动态调整。
- **非参数记忆的容量与一致性管理**：多级记忆长期累积可能引发冗余或冲突（如不同任务的工作流节点存在矛盾顺序），文中仅提到规范化与阈值筛选，缺乏大规模持续学习的内存管理策略。
- **未探索多智能体协同**：当前为单智能体框架，结合多智能体分工协作（如 GeoEvolver 的多智能体架构）可能进一步提升复杂 EO 任务的并行处理能力。

## 研究启发与可借鉴点

- **三级分层非参数记忆架构**可直接迁移至其他领域 agent 系统（如医疗诊断、金融分析），将全局流程、局部修正与标准操作程序分离存储，提升结构化知识的可维护性。
- **安全门控蒸馏机制**（$\psi$ 检查：工具调用数、答案完整性、调用频率上限、禁止模式过滤）可作为通用 agent 自进化模块，防止错误经验污染知识库。
- **模态感知工具过滤 + 混合相似度检索**（工具序列 Jaccard + 文本语义）为异构工具集下的检索增强规划提供了可复用的打分范式，适用于任何具有层次化工具库的场景。
- **确定性 EO 指导指令 $\mathcal{H}_q$ 的注入方式**（先检查数据资产→组织输入→传播产物→验证兼容性→延迟科学推断）可作为领域 agent 的"推理链脚手架"，与任意 backbone LLM 配合。
- **本团队可结合的方向**：将三级记忆架构与参数高效微调（如 LoRA adapter）结合，实现"非参数先验引导 + 轻量参数适配"的混合自进化方案；或将安全门控蒸馏应用于遥感大模型的持续预训练数据选择策略。

## 关键术语表

**GeoForge**：本文提出的免训练、自进化地球观测智能体框架，通过三层非参数记忆持续改进规划与工具使用。
**Earth-Bench**：Feng et al. (2025) 提出的 EO agent 综合基准，涵盖可执行遥感分析任务，用于评估任务精度与轨迹质量。
**Workflow Graph Memory**：将高层执行知识表示为有向可靠性感知图，节点编码任务类型、工具序列、成败统计，提供全局操作顺序先验。
**Action-Level Experiences**：存储细粒度条件-动作修正规则的经验库，用于纠正局部 EO 决策（模态/参数/产物复用等），来源于成功与失败轨迹的对比分析。
**Adapted Skill SOP**：任务条件化的标准操作程序，经 LLM 压缩器过滤无关细节后保留通用流程、数据约束与失败预防规则。
**ReAct**：推理-行动循环范式（Yao et al. 2022），LLM 交替进行思考与工具调用，本文在此基础上加入记忆检索层。
**Safety-Gated Distillation**：更新非参数记忆前通过多项安全检查（工具调用有效性、答案完整性、重复限制、禁止模式）的门控蒸馏机制。
**MCP（Model Context Protocol）**：本文工具服务的通信协议，确保工具接口标准化与安全性。

## 可复现要素

- **数据集**：Earth-Bench、ThinkGeo、GeoPlan-Bench；论文未明确说明是否公开，但提及引用来源（arXiv），通常配套项目页/代码库公开。
- **代码**：论文未明确声明开源仓库链接；提到使用 LangChain 与 LangGraph 实现。
- **权重**：框架为 training-free，不更新 LLM 权重；使用商业/开源 LLM backbone（GPT-5、Gemini-2.5-Flash、DeepSeek-V3.1、Qwen3-Max）。
- **关键超参**：温度 temperature=0.1，超时 120s，递归限制 80，最多 2 次续接尝试；记忆检索 Top-k=3（经验）、Top-1（工作流），检索阈值 0.6；SOP 压缩长度上限 $L_S$、上下文截断 $L_C$；工作流序列长度惩罚阈值 $L_0$。

---
title: "GeoForge: Non-Parametric Self-Evolving Agents for Earth-Observation Reasoning"
source: https://arxiv.org/pdf/2608.10494v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:39:03"
field: "地球观测智能体"
keywords: ["Earth Observation", "Self-Evolving Agent", "Non-Parametric Memory", "Tool Use Planning", "Workflow Graph", "Remote Sensing"]
innovations: ["三层非参数记忆架构（工作流图+动作级经验+适配型SOP）实现跨决策层级知识组织", "安全门控蒸馏机制在无参数更新前提下持续改进工具使用轨迹", "混合检索（结构Jaccard+语义相似度+任务一致性）提升任务条件化先验召回精度"]
benchmarks: ["Earth-Bench", "ThinkGeo", "GeoPlan-Bench"]
---

# 论文速读：GeoForge: Non-Parametric Self-Evolving Agents for Earth-Observation Reasoning

## 一句话总结
GeoForge 是一种免训练的自进化地球观测智能体框架，通过三层非参数记忆（工作流图记忆、动作级经验库、适配型技能 SOP）将已完成的执行轨迹蒸馏为结构化先验，在不更新骨干 LLM 权重的情况下，持续改进工具选择、执行顺序与最终准确率。

## 研究问题与动机
1. **EO 工作流的领域约束难以静态编码**：遥感工作流受感知语义、产品依赖、时空兼容性和参数条件约束，通用 Agent 盲目搜索宽泛操作空间，易产生工具不匹配、参数错误和断裂轨迹。
2. **现有自进化系统未跨决策层级组织可重用知识**：GeoEvolve 进化算法实现，GeoEvolver 累积交互级经验，但缺乏从全局操作顺序到局部动作纠正再到程序约束的完整知识组织。
3. **轨迹知识表示的两难**：原始轨迹保留过多实例特定细节，而紧凑文本摘要可能遗漏决定工作流有效性的产品依赖与执行结构。
4. **缺乏任务条件化的执行先验**：科学兼容性与操作顺序规则随任务与感知上下文变化，需要能被检索并复用、同时适应新感知场景的动态知识。

## 核心贡献（创新点）
1. **提出三层非参数记忆架构**：将完成轨迹蒸馏为 Workflow Graph Memory、Action-Level Experiences 和 Adapted Skill SOP，实现跨全局顺序、局部纠正与程序约束的互补知识表示，区别于仅存原始轨迹或纯文本摘要的方法。
2. **任务感知的混合检索策略**：结合工具序列 Jaccard 结构相似度与文本语义相似度（工作流图）、领域感知词法投影与任务一致性（经验库），实现更精准的上下文构建，区别于通用 RAG 仅依赖语义检索的方式。
3. **安全门控蒸馏与替换式记忆更新**：通过完整性、安全性与有效性门控（ψ=1）决定是否采纳新 SOP，采用替换而非追加更新 SOP，区别于多数自进化工作基于追加或经验累积的策略。
4. **免训练跨多骨架 LLM 的一致提升**：在 GPT-5、Gemini-2.5-Flash、DeepSeek-V3.1、GPT-4o、Qwen3-Max 五个骨干上均提升准确率与轨迹质量，证明框架的骨架无关性。

## 方法详解

### 整体流程
GeoForge 采用 ReAct 循环，外部维护一个**非参数执行状态**（三层记忆），每次任务前检索任务条件化先验注入提示，任务后通过安全门控蒸馏更新记忆，形成「执行→蒸馏→复用」闭环。

### 三层非参数记忆

**① Workflow Graph Memory（工作流图记忆）**  
有向可靠感知图 $\mathcal{G}=(\mathcal{W},\mathcal{A},\mathcal{U})$，节点 $w_i$ 包含任务类型 $c_i$、工作流描述 $r_i$、压缩工具序列 $\mathbf{z}_i$（去除重复以分离结构）、参数提示 $\mathbf{p}_i$、注意事项 $\mathbf{b}_i$、来源 $Q_i$ 及成功/失败计数 $(n_i^+, n_i^-)$。检索分数：
$$s_G(q, w_i) = \alpha \cdot J_{\text{tool}} + \beta \cdot J_{\text{text}} + \lambda_C \mathbf{1}[c_i=\hat{c}(q)] - \lambda_L \max(0, |\mathbf{z}_i|-L_0)$$
$J_{\text{tool}}$ 与 $J_{\text{text}}$ 分别为工具序列重叠度与词汇相似度，$\alpha+\beta=1$。

**② Action-Level Experiences（动作级经验库）**  
存储条件-行动对 $e_i = (\chi_i, \alpha_i, \mathbf{a}_i, \rho_i, m_i)$，用于修正局部决策（如模态/时间对齐、参数接地、空间不兼容处理）。检索分数：
$$s_E(q, e_i) = \underbrace{\frac{|V(q)\cap V(\chi_i\|\alpha_i\|\rho_i)|}{\max(|V(q)|,1)}}_{\text{词汇亲和}} + \lambda_E \underbrace{\mathbf{1}[\hat{c}(\chi_i\|\rho_i)=\hat{c}(q)]}_{\text{任务一致性}}$$
更新通过新奇性保持整合：新条目仅当 $\nu(e)\notin\nu(\mathcal{E}_t)$ 时加入，并截断至 $N_E$ 条。

**③ Adapted Skill SOP（适配型技能标准操作程序）**  
$S=\{s_j\}$，每项 $s_j$ 编码 EO 任务类型 $\mu_j$、处理流程 $\omega_j$、数据/论据约束 $\pi_j$、传感器/模态语义 $\beta_j$、防错规则 $\kappa_j$。通过 LLM 压缩器（限制长度 $L_S$）生成任务条件化 SOP $S_q = A_\Theta(S, q, \mathcal{E}_q; L_S)$，更新方式为**替换式**：仅当蒸馏候选 $\Delta S$ 通过安全门控才替换，否则保留原 SOP。

### 证据 Grounding 执行
拼接上下文 $\mathcal{C}_q = \text{Trunc}_{L_C}[\mathcal{G}_q\|\mathcal{E}_q\|\mathcal{S}_q\|\mathcal{H}_q]$，其中 $\mathcal{H}_q$ 为确定性 EO 引导（先检查数据资产、按模态/时间组织、优先兼容批处理、传播产品路径等）。执行后通过正则或 LLM 归一化将响应映射为有效答案，不产生新证据。

### 安全门控蒸馏（Safety-Gated Distillation）
$D_\Theta(\bar{\tau}, q, \hat{y}) = (\Delta\mathcal{E}, \Delta\mathcal{S}, \Delta\mathcal{W})$，过滤样本特定路径、答案捷径、外部 API 及非 MCP 代码。门控函数：
$$\psi(\tau,\hat{y})=\mathbf{1}[\text{Tools}>0]\mathbf{1}[\phi(\hat{y})=0]\mathbf{1}[|\text{Calls}|\leq L_{\max}]\mathbf{1}[\max_T n_T\leq R_{\max}]\mathbf{1}[\Delta S\cap\mathcal{F}=\emptyset]$$
仅 $\psi=1$ 时更新 SOP，工作流节点合并靠加权 Jaccard 相似度。

## 实验与结果

**数据集**：Earth-Bench（含可执行 EO 分析）、ThinkGeo（光学/SAR 工具推理）、GeoPlan-bench（高层地理空间规划）三个基准。

**主要结果（Earth-Bench，Table 1）**：
- GPT-5：GeoForge 准确率 **74.33%**（Earth-Agent 63.16%，GeoEvolver 70.85%），Tool-A-O 79.39、Tool-I-O 61.98、Tool-E-M 47.39。
- DeepSeek-V3.1：准确率 **77.09%**（Earth-Agent 52.23%），Tool-A-O 83.58。
- Gemini-2.5-Flash：准确率 **67.91%**，Tool-A-O 80.35。
- Qwen3-Max：准确率 **69.72%**，Tool-A-O 79.26。
- GPT-4o：准确率 **57.98%**，Tool-E-M 52.46。

**ThinkGeo（Table 2）**：指令对齐 97.27%、工具准确率 80.31%、答案准确率 60.98%。

**GeoPlan-Bench（Table 2）**：Key Recall 1.00、Key F1 0.77、Structural 0.79、Holistic 1100.65，全部最优。

**跨模态泛化（Table 4）**：平均准确率 61.85%（Earth-Agent 47.84%，提升 +14.01pp）；Spectrum 77.00%（+27pp）、Products 71.26%（+29.15pp）。

**消融（Table 3）**：无记忆 52.23%；去 Skill 降至 52.66%（最大降幅）；去 Graph 使 Tool-A-O/I-O/E-M 分别从 85.89/70.01/51.18 降至 75.69/61.74/45.09。

**超参敏感性**：Top-k=3、Min-Retrieve-Score=0.6 为最佳平衡点。

## 相关工作脉络
1. **Earth-Agent (Feng et al. 2025)**：ReAct 风格 EO Agent，报告最终准确率但缺乏轨迹结构化评估；GeoForge 通过记忆机制系统性提升其轨迹质量。
2. **GeoEvolver (Dai et al. 2026)**：经验驱动多 Agent，累积交互级工具专长；GeoForge 进一步组织跨全局-局部-程序多层级的知识。
3. **GeoEvolve (Luo et al. 2025)**：通过多 Agent 协作自动化发现地理空间算法（代码进化路线）；GeoForge 走「记忆驱动规划」路线，不涉及模型参数更新。
4. **Memento / Memento-Skills (Zhou et al. 2025/2026)**：通用 LLM Agent 的读写记忆与可进化技能；GeoForge 将其思想迁移至 EO 领域，增加遥感领域约束（模态、时空兼容、数据流）。
5. **SAGE (Wang et al. 2026a)**：自进化图记忆引擎用于通用关联记忆；GeoForge 的图是「工作流节点+工具转移边」，编码遥感数据流依赖，定位更具体。
6. **TerraBench (Nguyen et al. 2026)**：统一 EO 影像与 GIS 推理的基准，揭示参数接地与数值推理瓶颈；GeoForge 的经验库直接针对此类本地决策错误。

## 局限性与未来方向
1. **冷启动依赖初始轨迹质量**：三层记忆初期为空，需一定数量的种子任务才能形成有效先验，对小样本场景可能不稳定。
2. **经验库容量受限**：$N_E$ 截断策略可能丢弃有价值但略显不同的历史经验，长期积累的知识密度存在上限。
3. **仅支持 MCP 工具生态**：安全门控排除非 MCP 模式，限制了与某些专有遥感 API 或内部代码工具的集成。
4. **未涉及多 Agent 协同蒸馏**：当前为单 Agent 蒸馏，多 Agent 间的经验交叉复用尚未探索。
5. **SOP 替换式更新可能丢失历史**：一旦旧 SOP 被新 SOP 替换，若新 SOP 不适配后续任务则无法回滚。

## 研究启发与可借鉴点
1. **三层非参数记忆架构可迁移**：将"全局结构→局部纠正→程序约束"的分层知识组织思路应用于其他领域 Agent（如医疗、金融分析），有望复现类似提升。
2. **混合检索策略（结构 Jaccard + 语义相似度 + 任务一致性）**：在工具调用丰富的领域，同时利用操作序列结构和文本语义可提高检索精度，值得借鉴。
3. **安全门控蒸馏机制**：通过完整性、长度、模式等多重门控过滤无效/有害更新，避免记忆系统积累噪声，对任何基于经验的自进化系统均有参考价值。
4. **替换式而非追加式 SOP 更新**：保持知识表示紧凑、去重，避免记忆膨胀，适合资源受限的 Agent 系统。
5. **任务条件化压缩（Adapted SOP）**：用 LLM 作为压缩器将全局知识库裁剪为任务局部上下文，可在长上下文场景中显著降低提示开销。

## 关键术语表
**Workflow Graph Memory**：以有向图形式记录任务类型、工具调用顺序、参数提示与成功/失败统计的结构化知识，捕捉全局工作流依赖。
**Action-Level Experiences**：存储条件-行动对的细粒度经验库，用于修正模态对齐、参数接地、空间不兼容等局部决策错误。
**Adapted Skill SOP**：通过 LLM 压缩器针对具体任务从全局技能库中提取的紧凑程序规范，包含任务分解、数据约束与防错规则。
**Safety-Gated Distillation**：在执行后对轨迹进行蒸馏并施加多重门控（有效工具调用、完整答案、调用次数与重复率限制、禁止模式过滤），仅合格内容更新记忆。
**Evidence-Grounded Execution**：以当前 EO 观测数据与产品为最终答案依据，记忆仅作为执行先验，不替代实际数据产出结论。
**ReAct-based Self-Evolving Loop**：通过"推理→行动→观察"循环与外部非参数记忆的结合，实现不更新 LLM 权重的持续规划改进。

## 可复现要素
- **数据集**：Earth-Bench、ThinkGeo、GeoPlan-bench（均为公开基准，论文引用自 Feng et al. 2025、Shabbir et al. 2025、Li et al. 2025）
- **代码/权重**：论文未提及开源声明
- **关键超参**：temperature=0.1、timeout=120s、recursion limit=80、Top-k=3（经验检索）、Top-1（工作流检索）、min-retrieve-score=0.6、SOP 长度限制 $L_S$、经验库上限 $N_E$（论文未给出具体数值，详参 Appendix）

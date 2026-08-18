---
title: "GeoForge: Non-Parametric Self-Evolving Agents for Earth-Observation Reasoning"
source: https://arxiv.org/pdf/2608.10494v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:39:28"
field: "地球观测智能体"
keywords: ["Earth Observation", "Self-Evolving Agents", "Non-Parametric Memory", "Tool Use", "Workflow Reasoning"]
innovations: ["三层非参数记忆架构（工作流图+动作经验+技能SOP）", "任务条件化检索与安全门控蒸馏机制"]
benchmarks: ["Earth-Bench", "ThinkGeo", "GeoPlan-Bench"]
---

# 论文速读：GeoForge: Non-Parametric Self-Evolving Agents for Earth-Observation Reasoning

## 一句话总结
GeoForge 是一种无需训练、自进化的地球观测（EO）智能体框架，通过三层非参数记忆（工作流图、动作级经验、技能 SOP）将执行轨迹转化为可复用的结构化知识，在不更新骨干 LLM 权重的前提下持续提升 EO 任务的规划与工具使用质量。

## 研究问题与动机
1. **EO 工作流的领域约束复杂性**：遥感分析受传感语义、产品依赖、时空兼容性等多重约束，现有通用 Agent 在宽泛操作空间中盲目搜索，容易产生工具错配、无效参数和不完整工作流。
2. **已有自进化系统的知识组织不足**：GeoEvolve 聚焦算法层面演化，GeoEvolver 仅积累交互级工具经验，两者均未系统性地在跨决策层级（全局顺序→局部修正→程序约束）组织可复用知识。
3. **轨迹知识的压缩与复用难题**：原始轨迹包含过多实例特定细节，而纯文本摘要又会丢失产品依赖、参数条件和执行结构，难以形成可直接指导执行的领域先验。

## 核心贡献（创新点）
1. **提出三层非参数记忆架构**：首次将全局工作流图（Workflow Graph Memory）、动作级经验（Action-Level Experiences）和技能 SOP（Adapted Skill SOP）统一整合，区别于 GeoEvolver 仅存储交互级错误归因，实现了从宏观到微观的完整知识覆盖。
2. **任务条件化的检索与适配机制**：通过感知上下文约束操作空间，并结合工具序列 Jaccard 相似度与文本语义相似度进行混合检索，区别于传统基于纯语义检索的方法，显式保留了工作流结构信息。
3. **安全门控的蒸馏与更新范式**：设计了基于工具调用数、回答完整性、代码安全性等多重约束的知识提炼流程，确保只有高质量、可执行的轨迹才能更新记忆，区别于 GeoEvolve 的累积式经验库。

## 方法详解
**整体流程**：给定任务 q，GeoForge 首先推断传感上下文以约束操作空间，然后从三层记忆中检索任务条件化的先验知识构建执行上下文，最后通过 ReAct 循环执行工具调用；任务完成后，安全门控蒸馏将轨迹转化为可复用知识并更新外部记忆。

**Workflow Graph Memory**：将高执行知识表示为有向图 G = (W, A, U)，节点 w_i 包含任务类型 c_i、工作流描述 r_i、压缩工具序列 z_i、参数提示 p_i、注意事项 b_i 及成功/失败计数 n_i^+/n_i^-。检索时计算混合得分：s_G = α·J_tool + β·J_text + λ_C·1[c_i = ĉ(q)] - λ_L·max(0, |z_i| - L_0)，其中 J_tool 衡量工具序列重叠，J_text 衡量词汇相似度，λ_C 奖励任务类型一致，λ_L 惩罚过长序列。

**Action-Level Experiences**：经验库存储细粒度执行修正，条目 e_i = (χ_i, α_i, a_i, ρ_i, m_i)，其中 χ_i 为触发条件，α_i 为修正动作，a_i 为源操作。检索采用门控相关性模型：s_E = 词汇亲和力 + λ_E·1[任务一致性]，并通过证据约束排除无执行基础的自由形式反思。经验库通过新颖性保持的合并更新：E_{t+1} = Tail_{N_E}(E_t ∪ {e ∈ ΔE : ν(e) ∉ ν(E_t)})。

**Adapted Skill SOP**：S = {s_j} 编码 EO 任务、处理流程、数据和参数约束、传感器语义及防错规则。通过 LLM 压缩器构建任务条件化 SOP：S_q = A_Θ(S, q, E_q; L_S)，保留相关工作流并移除不相关细节。SOP 采用替换式更新而非追加式：仅当蒸馏候选 ΔS 满足完整性、紧凑性、可执行轨迹支撑且通过安全门控 ψ(τ, ŷ)=1 时才替换。

**Safety-Gated Self-Evolution**：蒸馏器从执行轨迹 τ̄ 中提取至多 3 条经验、1 个修订 SOP 和至多 2 个工作流候选，同时排除实例特定路径、答案捷径、外部 API 和无效工作流。更新门控条件包括：工具调用数 >0、非不完整答案、调用次数 ≤ L_max、单工具重复次数 ≤ R_max、无禁用代码模式。

## 实验与结果
**数据集**：Earth-Bench（EO 分析基准）、ThinkGeo（光学/SAR 工具推理）、GeoPlan-Bench（高级地理空间规划）。

**主要结果（Earth-Bench）**：
- GPT-5  backbone：准确率 74.33%，Tool-Any-Order 79.39%，Tool-In-Order 61.98%，Tool-Exact-Match 47.39%
- DeepSeek-V3.1 backbone：准确率 77.09%（最高），Tool-Any-Order 83.58%，Tool-In-Order 68.30%
- Gemini-2.5-Flash backbone：准确率 67.91%，Tool-Any-Order 80.35%
- Qwen3-Max backbone：准确率 69.72%，Tool-Any-Order 79.26%

**ThinkGeo 基准**：Instruction Alignment 97.27%，Tool Accuracy 80.31%，Answer Accuracy 60.98%。

**GeoPlan-Bench 最强结果**：Recall_key = 1.00，F1_key = 0.77，Structural = 0.79，Holistic = 1100.65。

**消融实验**：移除 Skill 模块导致准确率从 74.33% 骤降至 52.66%（下降最大）；移除 Graph 模块导致 Tool-Any-Order 从 85.89% 降至 75.69%。

**跨模态泛化**：Spectrum 模态 77.00%（+27.00pp），Products 模态 71.26%（+29.15pp）。

**误差分析**：GeoForge 显著减少工具规划错误和推理错误，剩余错误主要集中在执行轨迹和参数相关层面。

## 相关工作脉络
1. **GeoEvolve (Luo et al. 2025)**：通过多智能体协作自动化发现地理空间算法，关注算法实现层面的演化；GeoForge 聚焦于工作流层面的知识组织与复用，不修改算法代码。
2. **GeoEvolver (Dai et al. 2026)**：构建经验库存储交互级工具专长和失败归因；GeoForge 进一步整合全局工作流结构、局部修正规则和功能化 SOP，形成更完整的知识体系。
3. **Memento (Zhou et al. 2025)**：引入读写反思学习，通过情景记忆存储结果；GeoForge 采用非参数外部记忆，专注于 EO 领域的领域约束和工作流知识。
4. **Earth-Agent (Feng et al. 2025)**：基于 ReAct 框架集成专用工具的 EO Agent；GeoForge 在其基础上引入三层记忆和自进化机制，显著改善工具选择和执行顺序可靠性。
5. **SAGE (Wang et al. 2026a)**：自进化图记忆引擎用于结构化关联记忆；GeoForge 针对 EO 领域特化，显式编码传感语义、时空兼容性等领域约束。

## 局限性与未来方向
1. **轨迹质量依赖**：蒸馏过程依赖成功或高质量执行轨迹，若 LLM 初始规划能力弱，可能难以生成可复用知识。
2. **记忆规模膨胀**：随着任务累积，记忆结构可能增长，需设计更精细的淘汰和压缩机制。
3. **罕见模式覆盖不足**：基于历史轨迹的归纳可能难以应对从未出现过的新型传感场景或工具组合。
4. **LLM 压缩器依赖**：SOP 适配依赖 LLM 的压缩能力，复杂任务的紧凑表示可能丢失关键细节。
5. **可扩展性验证**：目前仅在有限 LLM backbone 上验证，需进一步测试在其他开源模型上的效果。

## 研究启发与可借鉴点
1. **三层分层记忆架构**：从全局工作流→局部修正→功能化 SOP 的分层设计思路可迁移至其他领域 Agent（如科学计算、代码生成），提供可复用的知识组织范式。
2. **安全门控蒸馏机制**：通过多重约束（调用数、完整性、安全性）过滤知识更新的设计，可有效防止记忆污染，适用于需要持续学习的 Agent 系统。
3. **结构-语义混合检索**：同时利用工具序列 Jaccard 相似度和文本语义相似度进行检索，兼顾了工作流结构和领域知识的检索效果，可推广至其他工具使用场景。
4. **任务条件化 SOP 压缩**：通过 LLM 压缩器将通用知识适配为任务局部 SOP 的方法，可在多领域零样本设置中减少提示长度同时保持知识质量。

## 关键术语表
**Workflow Graph Memory**：以有向图结构存储全局执行知识，节点编码任务类型、工作流描述、工具序列和参数提示，边反映工具转换关系。

**Action-Level Experiences**：存储细粒度执行修正知识的经验库，每条记录包含触发条件、修正动作和来源操作，用于局部决策纠正。

**Adapted Skill SOP**：通过 LLM 压缩器从通用技能库中提取任务条件化的标准作业程序，包含可复用工作流、数据约束和防错规则。

**Safety-Gated Distillation**：基于工具调用数、回答完整性、代码安全性等多重约束的知识提炼流程，确保只有高质量轨迹才能更新记忆。

**Non-Parametric Self-Evolution**：通过外部结构化记忆存储和更新知识，不修改骨干 LLM 权重即可实现持续的规划能力改进。

**Tool-Any-Order / Tool-In-Order / Tool-Exact-Match**：分别衡量工具覆盖率、执行顺序保真度和完整序列精确度，用于评估轨迹质量。

**ReAct Framework**：结合推理（Reasoning）与行动（Acting）的迭代循环，Agent 在每步生成思考并调用工具，广泛用于多步骤任务求解。

## 可复现要素
- **数据集**：Earth-Bench、ThinkGeo、GeoPlan-bench（均为公开基准，论文未提及自定义数据集）
- **代码/权重**：论文未明确提及开源，但提到使用 LangChain 和 LangGraph 实现
- **关键超参**：temperature=0.1，timeout=120s，recursion limit=80，top-3 经验检索，top-1 工作流检索（阈值 0.6）

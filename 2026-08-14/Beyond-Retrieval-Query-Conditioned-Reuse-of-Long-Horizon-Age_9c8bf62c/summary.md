---
title: "Beyond-Retrieval-Query-Conditioned-Reuse-of-Long-Horizon-Age"
source: https://arxiv.org/pdf/2608.12847v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:53:38"
field: "Agent Memory & Retrieval"
keywords: ["agent memory", "trajectory reuse", "retrieval-augmented agent", "long-horizon planning", "query-conditioned reuse", "binding shift", "WebArena", "AppWorld"]
innovations: ["提出查询条件化复用(QCR)框架，在固定检索后对长轨迹进行目标导向重构，显式分离可迁移工作流与需重新绑定的值", "构建统一的查询条件化轨迹复用评测框架，隔离检索质量与复用效果的因果贡献", "揭示长轨迹复用中绑定漂移的失效模式，证明Full Trajectory在长轨迹下仅保留15.8%的短期效用而QCR保留60.3%"]
benchmarks: ["WebArena", "WorkArena", "AppWorld"]
---

# 论文速读：Beyond-Retrieval-Query-Conditioned-Reuse-of-Long-Horizon-Age

## 一句话总结
论文提出了查询条件化复用（QCR, Query-Conditioned Reuse）框架，在检索之后、执行之前对长轨迹记忆进行目标导向的重构，使智能体能够从历史轨迹中提取可迁移的工作流程并重新绑定当前任务的参数；在 WebArena、WorkArena 和 AppWorld 共 2,391 个目标实例上，QCR 达到 62.3% 平均成功率，较 Full Trajectory 提升 10.7 个百分点，同时减少 48.9% 的在线 token。

## 研究问题与动机
- **检索有效 ≠ 复用有效**：现有 agent 记忆研究多停在"能否检索到相关历史"这一阶段（如 LongBench、LongMemEval），但没有回答一个关键问题：检索到的长轨迹进入上下文后，是否真正帮助了当前任务的完成？
- **长轨迹的源-目标绑定漂移**：成功的轨迹编码了有价值的工具工作流、决策规则和验证序列，但同时也携带了源任务特定的用户、对象、路径、日期等绑定信息；直接注入原轨迹会导致过时的绑定被复制到新任务中。
- **上下文窗口不是唯一瓶颈**：长原始轨迹不仅带来 token 开销，更会将当前目标淹没于旧观察和无关分支中，模型未必能可靠地使用长上下文的相關部分。
- **评测需要分离检索与复用**：现有基线（无记忆、通用摘要、全轨迹）混淆了"检索质量"和"复用效果"，缺乏在固定检索条件下隔离比较不同记忆使用方式的能力。

## 核心贡献（创新点）
1. **提出查询条件化复用（QCR）框架**：在冻结检索之后插入一步目标导向的记忆重构，输出包含工作流不变量、需重新获取的绑定、适用条件和验证护栏的紧凑笔记——与已有方法仅依赖摘要压缩或原样注入的本质区别在于，QCR 显式分离了"可迁移的程序知识"和"需当前环境重新确认的绑定值"。
2. **构建统一的查询条件化轨迹复用评测框架**：固定候选检索、目标状态、模型、解码策略和工具预算，仅改变记忆呈现形式；涵盖 WebArena、WorkArena、AppWorld 三个环境的 2,391 个目标实例，为"检索后阶段"提供了可复现的隔离比较基准。
3. **揭示长轨迹复用中的绑定漂移失效模式**：通过轨迹长度分层和源-目标绑定偏移两个维度的系统分析，证明 Full Trajectory 在长轨迹下仅保留 15.8% 的短期效用，在大型绑定偏移下保留 8.2%；而 QCR 分别保留 60.3% 和 67.9%，建立了"绑定漂移越大，直接注入越失效"的定量证据链。
4. **设计轻量级 reranking 选择机制**：在 top-5 候选集合上进行摘要重排序，使最终可复用记忆准确率达 94.8%，仅低于 oracle 选择 1.8 个百分点，证明候选选择是必要但不充分的环节，真正的增益来自后检索阶段的复用格式转换。

## 方法详解
**整体流程（三段式冻结评测）**：
1. **离线记忆库构建**：从 WebArena（228 条）、WorkArena（201 条）、AppWorld（194 条）三个环境中收集 623 条经验证的成功轨迹，构建统一混合索引的冻结库 B，每条记录包含源指令、有序观察/动作序列、工具调用及返回、终端产物和验证结果。
2. **绑定感知目标构造**：对每条源轨迹 $\tau_s$，以 3.84 个/轨迹的平均比例生成目标变体：保留源工作流意图，但改写一个或多个目标特定绑定（small/medium/large 四个级别），共 2,391 个目标实例。目标从自身初始状态出发，由自身验证器检查。
3. **查询条件化复用（QCR）支持生成**：给定共享候选集 $Z_t$ 和 ranker 选定的单条轨迹，QCR writer 输出紧凑支持对象 $r_t$，包含四个字段：
   - **Workflow Invariant（工作流不变量）**：保留适用于目标的动作模式（如"检查→验证→修改→确认"），剔除源任务特有步骤。
   - **Bindings to Re-obtain（需重新获取的绑定）**：列出必须在当前目标环境中重新确认的值（实体、路径、日期、用户等），明确禁止复制源绑定。
   - **Applicability Conditions（适用条件）**：声明何时可以复用、何时应放弃复用（如前置条件不满足）。
   - **Verification Guardrail（验证护栏）**：指明源任务中建立完成判定的检查步骤，要求在目标侧重新执行。

**关键数学约束**：
- 候选检索：$Z_t = R(q_t, o_{t,0}, B)$，对所有比较方法固定相同
- 在线 token 总量：$C_{\mathrm{online}} = I_{\mathrm{base}} + I_{\mathrm{mem}} + I_{\mathrm{syn}} + O_{\mathrm{syn}} + O_{\mathrm{act}}$，用于公平比较各条件的实际成本
- 复用增益：$U = S_{\mathrm{memory}} - S_{\mathrm{no\ memory}}$，按轨迹长度和绑定偏移分层报告

**模型配置**：全部使用 DeepSeek-V4-Pro；源 rollout 和目标 actor 采样温度 0.2，支持生成和选择步骤采用确定性解码（temperature=0）。

## 实验与结果
**数据集**：WebArena（228 源/874 目标）、WorkArena（201 源/772 目标）、AppWorld（194 源/745 目标），统一混合库，每个目标 3 次 seed 匹配运行。

**评估指标**：Success（%）、Milestone（%）、API Calls（均值）、Online Tokens（均值）。

**主要结果**：

| 方法 | WebArena Success | WorkArena Success | AppWorld Success | 平均 Success | 平均 API Calls | 在线 Token |
|---|---|---|---|---|---|---|
| No Memory | 31.5 | 36.6 | 47.1 | — | 24.6 | 15.2k |
| Generic Summary | 40.2 | 45.9 | 57.6 | — | 20.8 | 8.1k |
| Full Trajectory | 43.8 | 49.6 | 61.4 | — | 21.9 | 18.4k |
| **QCR** | **54.7** | **60.4** | **71.8** | **62.3** | **16.7** | **9.4k** |

**关键数字**：
- QCR 平均 Success 62.3%，较 Full Trajectory 提升 **+10.7 个百分点**，较 No Memory 提升 **+23.2/23.8/24.7 点**（三环境分别）
- 在线 token 从 Full Trajectory 的 18.4k 降至 9.4k，减少 **48.9%**
- Reranking 将 top-5 候选中的可复用记忆准确率从 retriever top-1 的 78.9% 提升至 94.8%，仅低于 oracle 1.8 点
- 长度分层：Very Long 组下 Full Trajectory 仅保留 +2.9 点增益（短期 +18.4 的 15.8%），QCR 保留 +13.2 点（60.3%）
- 绑定偏移分层：Large shift 下 Full Trajectory 增益缩至 +2.2（无偏移 +26.9 的 8.2%），QCR 保留 +20.1（67.9%）；QCR 的过时绑定错误率 10.9% vs Full Trajectory 的 46.9%

**最强结果**：QCR 在所有环境和所有指标上均显著领先，AppWorld 上达到 71.8% 成功率和 82.9% Milestone，较 No Memory 提升 24.7/21.9 点。

## 相关工作脉络
1. **MemGPT / MemoryBank / HippoRAG / Mem0 / A-MEM**：关注记忆的存储、索引和上下文投递，但未解决检索后的复用格式问题——本文将其视为"访问层"完成，聚焦访问之后的转化操作。
2. **LongBench / LoCoMo / LongMemEval**：评测长上下文下的记忆保持和访问能力，本质上是"检索质量"代理——本文认为这是必要不充分条件，要求追踪到目标任务最终完成。
3. **Reflexion / Generative Agents / Voyager / ExpeL / Synapse**：从先验运行中提取反思、技能或工作流——本文固定候选集和选定轨迹，隔离测量"表示形式"对后续目标的影响。
4. **Agent Workflow Memory (Wang et al. 2024) / SAM (Hu et al. 2026) / Agentic Memory (Yu et al. 2026) / OCR-Memory (Li et al. 2026)**：各自学习不同的记忆操作或表示——本文定位为诊断性干预，证明即使最简的目标导向笔记也能在固定检索下带来显著增益。
5. **WebArena / WorkArena / AppWorld / τ-Bench**：提供交互式多步 agent 评测环境——本文复用其验证器，但新增跨环境统一库和绑定偏移构造，强调"验证完成而非孤立执行"。
6. **ReAct / MemoryBank 系列**：标准 RAG 模式"检索-then-条件化"——本文指出检索完成后仍有独立的"复用"阶段，需要专门评估。

## 局限性与未来方向
- **仅评估成功源轨迹**：未包含部分失败的历史运行或多条记忆的组合利用，真实场景中失败轨迹可能蕴含重要排除信息。
- **单一选定记忆假设**：每目标仅使用一条 ranker 选定的轨迹，未测试多记忆协同复用或跨轨迹程序组合。
- **绑定偏移为受控构造**：人工改写绑定的规模和类型有限，未覆盖开放环境中的自然任务变化和长尾漂移。
- **未评估不可逆副作用**：只测量验证完成的成功率，未考虑复用轨迹可能引入的策略违规或环境破坏性操作。
- **检索器固定为 BGE-M3**：未探索更好检索器或端到端可学习的复用策略，未来可在此框架基础上替换检索模块进行增量评估。
- **Token 节省不等于普遍目标**：安全敏感任务可能需要额外检查而非更少 token，本文的成本指标是辅助而非替代成功率的衡量。

## 研究启发与可借鉴点
1. **检索-复用解耦评测范式**：将"检索质量"和"复用效果"分离为独立可测量的阶段，通过冻结候选集和选定轨迹来隔离变量——此范式可直接迁移至任何 RAG-based agent 系统的诊断评测。
2. **绑定漂移分析框架**：通过受控的源-目标绑定改写（small/medium/large）量化复用的脆弱性，为记忆系统的稳健性评估提供了可复用的实验设计模板。
3. **紧凑目标导向笔记的结构设计**：QCR 的四字段结构（工作流不变量 + 需重新获取的绑定 + 适用条件 + 验证护栏）提供了一种简洁但信息完整的支持表示，可直接嵌入现有 agent 的 memory replay 模块作为后处理步骤。
4. **在线 token 的精确会计方法**：将基础 prompt、记忆输入、支持生成输入/输出、actor 输出分项记录，避免重复计数——为 agent 效率评估提供了可审计的核算标准。
5. **跨环境统一记忆库的可行性验证**：在同一混合索引库中进行跨 WebArena/WorkArena/AppWorld 的检索和复用，证明了跨领域经验迁移的可能性，为构建通用 agent 记忆库提供了实证基础。

## 关键术语表
- **Query-Conditioned Reuse (QCR)**：在固定检索结果后，针对当前目标查询和初始状态生成紧凑复用支持的机制，核心是分离可迁移流程与需重新绑定的值。
- **Binding（绑定）**：智能体在当前任务中必须 grounding 的具体值，包括实体、用户、文件路径、日期、记录 ID、参数或当前环境状态。
- **Stale-binding Error（过时绑定错误）**：智能体在动作、输出或工具参数中复制了源任务特定值，与新任务查询或观察相冲突的错误。
- **Workflow Invariant（工作流不变量）**：QCR 支持对象中包含的可迁移动作模式，即在新目标中仍适用的工具调用顺序和决策规则。
- **Verification Guardrail（验证护栏）**：指明源任务中用于判定完成的关键检查步骤，要求目标侧在类似位置重新执行验证。
- **Frozen Retrieval（冻结检索）**：所有比较方法共享同一份固定检索候选集和 ranker 选定结果，确保差异仅来自复用格式而非检索质量。
- **Binding Shift（绑定偏移）**：源-目标任务之间绑定值的改写程度，分为 None/Small/Medium/Large 四级，用于量化复用场景的相似性变化。
- **Usable-memory@k**：top-k 候选中存在可转移工作流的记录比例，用于诊断检索是否暴露了有潜力的历史经验。

## 可复现要素
- **数据集**：WebArena、WorkArena、AppWorld（原生公开）；作者构建的统一混合记忆库含 623 条验证轨迹和 2,391 个目标实例，附录包含完整清单（Table A2）
- **代码/权重**：论文未明确说明代码开源状态
- **关键超参**：Top-5 检索，每目标 3 次 seed-matched 运行；DeepSeek-V4-Pro 模型，actor 采样 temperature=0.2、top-p=0.95；QCR writer 使用 deterministic decoding（temperature=0）；max tokens=4,096（source/actor）或 1,024（writer/ranker）
- **嵌入检索器**：BGE-M3
- **模型**：全部使用 DeepSeek-V4-Pro

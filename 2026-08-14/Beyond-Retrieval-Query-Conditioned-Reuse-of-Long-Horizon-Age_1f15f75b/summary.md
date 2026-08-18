---
title: "Beyond-Retrieval-Query-Conditioned-Reuse-of-Long-Horizon-Age"
source: https://arxiv.org/pdf/2608.12847v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:54:09"
field: "Agent 记忆与轨迹重用"
keywords: ["Agent memory", "trajectory reuse", "query-conditioned reuse", "long-horizon agents", "retrieval-augmented agents", "binding shift", "WebArena", "WorkArena", "AppWorld"]
innovations: ["提出QCR框架，将历史轨迹转化为目标绑定的紧凑支持对象，隔离重用表示的影响", "构建跨环境统一记忆库与绑定偏移梯度评估协议，揭示长轨迹重用瓶颈从检索转移到重用", "发现Full Trajectory效用在长轨迹和大绑定偏移下骤降，而QCR能保留60%以上增益"]
benchmarks: ["WebArena", "WorkArena", "AppWorld"]
---

# 论文速读：Beyond-Retrieval-Query-Conditioned-Reuse-of-Long-Horizon-Age

## 一句话总结
本文指出长_horizon Agent 记忆的瓶颈已从检索转移到**检索后的重用**，提出 Query-Conditioned Reuse（QCR）框架，将历史轨迹转化为目标绑定的紧凑支持对象，在 WebArena/WorkArena/AppWorld 三个评测集上平均成功率达 62.3%，较直接注入完整轨迹提升 10.7 个百分点，同时在线 token 减少 48.9%。

## 研究问题与动机
- **核心问题**：Agent 在长_horizon 任务中如何有效重用历史交互轨迹？现有工作多关注检索质量，但检索到的轨迹如何被目标任务安全、高效地复用仍是盲区。
- **现有方法不足**：
  1. 检索型记忆（如 MemGPT、RAG）只保证“找到相关历史”，未解决“如何使用”；
  2. 直接注入完整轨迹（Full Trajectory）会带入源任务的过时绑定（如用户、路径、状态），导致错误复用；
  3. 通用摘要（Generic Summary）过于笼统，丢失关键的工作流和验证信息；
  4. 现有长上下文基准（LongBench、LoCoMo 等）仅评估“能否检索/回忆”，不评估“能否提升新任务成功率”。

## 核心贡献（创新点）
1. **提出查询条件化重用（QCR）框架**：在检索后插入一步目标绑定支持生成，仅保留工作流不变量、待重新获取的绑定、适用条件和验证护栏。
2. **构建统一的跨环境记忆评估协议**：固定检索候选、目标状态、模型、解码和工具预算，隔离“重用表示”对最终成功率的因果影响。
3. **发现长轨迹重用中的瓶颈转移**：随着轨迹长度和源-目标绑定偏移增加，直接注入效用骤降，而 QCR 能保留大部分增益。
4. **建立端到端评测基准**：基于 623 条已验证历史轨迹构造 2,391 个目标实例，涵盖 WebArena、WorkArena、AppWorld，提供可复现的绑定偏移梯度分析。

## 方法详解
- **冻结记忆库构建**：从三个环境收集 623 条通过验证器的成功轨迹，统一索引，排除环境标签和任务标识，仅保留指令、动作序列、工具调用与返回、终端产物。
- **绑定感知目标构造**：对每条源轨迹进行小/中/大三级绑定重写（替换用户、路径、日期等实体，或初始环境状态），保留可复用工作流，生成最多 4 个目标变体。
- **QCR 支持对象生成**：给定检索到的候选集 \(Z_t\)、目标查询 \(q_t\) 和初始观测 \(o_{t,0}\)，QCR writer 输出四个字段：
  1. **Workflow invariant**：保留目标仍需要的动作模式（如“检查→验证→修改→确认”）；
  2. **Bindings to re-obtain**：列出必须从当前任务重新获取的实体、路径、日期等，禁止复制源值；
  3. **Applicability conditions**：说明何时应拒绝重用（如前置条件不满足）；
  4. **Verification guardrail**：保留源任务的验证步骤，要求目标端重新确认。
- **评估协议**：固定 top-5 检索候选，轻量 ranker 选择一条轨迹，四条件（No Memory、Generic Summary、Full Trajectory、QCR）接收相同候选和选定记录，仅表示方式不同。

## 实验与结果
- **数据集**：WebArena（228 源/874 目标）、WorkArena（201/772）、AppWorld（194/745），共 623 源轨迹、2,391 目标实例。
- **基线**：No Memory、Generic Summary、Full Trajectory、QCR。
- **主要结果**（Table 1）：
  - QCR 平均成功率 **62.3%**，比 Full Trajectory（51.6%）高 **10.7 点**，比 No Memory（38.4%）高 23.9 点；
  - QCR 在线 token 为 9.4k，较 Full Trajectory（18.4k）减少 **48.9%**；
  - 跨环境一致性：WebArena 成功 54.7%（+10.9 vs Full）、WorkArena 60.4%（+10.8）、AppWorld 71.8%（+10.4）。
- **检索与选择分析**：Top-5 覆盖 97.8% 可重用轨迹，ranker 后最终 reusable-memory 准确度达 94.8%，仅比 oracle 低 1.8 点，说明**重用表示**是主要瓶颈而非检索。
- **长度敏感性**（Table 2）：轨迹从短到超长，Full Trajectory 效用从 +18.4 点降至 +2.9 点（保留 15.8% 短期效用），QCR 从 +21.9 点降至 +13.2 点（保留 60.3%）。
- **绑定偏移分析**（Table 3）：大偏移下 Full Trajectory 效用仅 +2.2 点（源值失效占 46.9%），QCR 保持 +20.1 点，正确重绑定率从 31.7% 升至 77.8%。

## 相关工作脉络
1. **MemGPT、MemoryBank 等系统记忆架构**：专注存储/检索/索引，本文指出它们未明确分离“检索质量”与“重用效用”，本文评估协议可检验其真实增益。
2. **LongBench、LoCoMo 等长上下文基准**：评估历史回忆能力，本文强调“能否解决新任务”才是 Agent 记忆的最终目标。
3. **Reflexion、ExpeL、Agent Workflow Memory 等经验提取方法**：从历史中提取反思/技能/工作流，本文固定提取结果，比较其表示方式对目标任务的影响。
4. **WebArena、WorkArena、AppWorld 等 Agent 评测**：通常独立评分单次执行，本文构造绑定偏移目标，测试跨环境轨迹重用的稳健性。
5. **OCR-Memory、Agentic Memory 等长程记忆表示**：探索更丰富的记忆结构，本文认为无论存储多丰富，**作用于 actor 的表示必须目标绑定**，否则引入陈旧值。

## 局限性与未来方向
- **仅评估成功源轨迹**：未涵盖部分失败、多记忆组合或开放获取场景。
- **单一目标绑定偏移控制**：真实环境中偏移更复杂且不可预测。
- **未评估不可逆副作用**：仅测量验证器得分，未追踪策略违规或有害操作。
- **未来方向**：可结合更强检索器、学习型重用策略，但应保持本文分离评估边界；扩展至多轨迹融合、动态记忆获取、安全敏感任务。

## 研究启发与可借鉴点
1. **评估协议设计**：固定检索候选和目标状态，只改变重用表示，可干净隔离“记忆使用方法”的贡献，适合记忆系统评测。
2. **QCR 四字段 schema**：工作流不变量、待重新获取绑定、适用条件、验证护栏，可作为通用目标绑定记忆表示模板。
3. **绑定偏移梯度分析**：小/中/大三级重写可用于诊断记忆重用的脆弱点，尤其适合调试 agent 的过时值依赖。
4. **跨环境统一记忆库**：打破环境隔离，测试工作流跨域迁移能力，更具挑战性也贴近真实多工具场景。
5. **在线 token 与成功率联合报告**：避免单纯追求高分而忽略成本，QCR 以一半 token 实现更高成功，体现效率意识。

## 关键术语表
- **Query-Conditioned Reuse (QCR)**：在检索后针对当前目标生成紧凑支持对象，明确区分可复用工作流与需重新获取的绑定。
- **Binding shift**：目标任务中相对于源轨迹被改写或替换的实体、路径、日期等具体值。
- **Stale-binding error**：Agent 错误复制源轨迹中的过时值，导致与当前任务状态冲突。
- **Workflow invariant**：源轨迹中仍可移植给目标任务的动作模式或决策规则。
- **Verification guardrail**：保留自源任务的验证步骤，确保目标端重新确认关键状态。
- **Frozen memory bank**：构建后不再更新的成功轨迹集合，用于隔离评估重用表示的影响。
- **Reusable-memory accuracy**：ranker 选中的轨迹确实能为目标提供可执行工作流的比例。

## 可复现要素
- **数据集**：WebArena、WorkArena、AppWorld 部分任务；历史轨迹与目标构造细节见附录，**代码与数据未明确开源**（论文未提及）。
- **模型**：DeepSeek-V4-Pro（生成、排名、QCR writer 均使用）。
- **检索器**：BGE-M3，top-5 候选，环境标签排除在索引外。
- **关键超参**：源 rollout 和 actor 温度 0.2、top-p 0.95；support 写作与排名温度 0（确定性）；max tokens 4096（rollout/actor）、1024（summary/QCR）、512（ranker）。
- **种子**：每个目标条件 3 次 seed-matched 运行取平均。

---
title: "From Faulty Memories to Corrected Actions: Dependency-Guided Rollback Repair for Memory-Augmented Agents"
source: https://arxiv.org/pdf/2608.10502v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 06:43:43"
field: "多智能体系统可靠性与鲁棒性"
keywords: ["Memory-Augmented Agents", "Rollback Repair", "Dependency Graph", "Fault Injection", "Agent Robustness", "Tool Use"]
innovations: ["依赖图建模 memory-store 与 execution trace 联合关系", "Rule-Guided Rollback 互斥动作决策机制", "多维评估指标体系与受控故障注入基准"]
benchmarks: ["Controlled Memory Repair Benchmark (150 cases)"]
---

# 论文速读：From Faulty Memories to Corrected Actions: Dependency-Guided Rollback Repair for Memory-Augmented Agents

## 一句话总结
本文提出一种**依赖关系引导的回滚修复方法**，通过构建 memory-store 与 execution trace 之间的依赖图，精准定位污染源头并选择性重放受影响步骤，实现对 memory-augmented agents 的后失效内存恢复，在 150 个跨领域任务上显著优于基线方法。

---

## 研究问题与动机
1. **Memory-augmented agents 的故障传递问题**：当 agent 的记忆存储被污染（如错误写入、残留 stale 记录）时，后续依赖这些记忆的推理步骤会产生级联错误，最终导致答案错误。
2. **现有修复方法的不足**：
   - 全量重放代价高昂（LLM 调用次数剧增）；
   - 朴素删除可能误伤 benign 记忆，破坏已构建的有效知识；
   - 缺乏对故障传播路径的可追溯建模。
3. **缺乏系统性的故障评估基准**：现有工作缺少标准化的 fault injection 协议和多维修复评估指标。
4. **可审计性需求**：agent 执行过程需要可验证、可复现，依赖图提供了一条可审计的执行轨迹。

---

## 核心贡献（创新点）
1. **依赖图（Dependency Graph）建模**：首次显式联合建模 memory store 与 execution trace 之间的多类型边关系（cite/support/produce/delete/update/consolidate/supersede/derive 等），实现故障传播的可追溯追踪。
2. **Rule-Guided Rollback 决策机制**：基于诊断标记（diagnosed fault identifiers）和依赖图约束，定义一组互斥动作集合（delete/quarantine/invalidate/replay/preserve），实现精准回滚规划。
3. **受控内存修复基准测试（Controlled Memory Repair Benchmark）**：构造 150 个工具使用案例（Shopping/Travel/Customer Support），设计 4 类故障注入模式（Poisoned/Stale/Wrong-user/Summary-drift），并提供多粒度评估指标体系。
4. **结构化 Prompt Contracts**：定义 base agent、rollback executor、LLM-judge baseline 三类角色在信息访问边界、证据引用要求与 JSON 输出契约上的规范，保障实验可复现性。
5. **多维度评估指标体系**：提出 Recovery / Recurrence / Faulty Removal / Benign Preservation / Claim-invalidation F1 / Replay Ratio / LLM Count 七项指标，全面衡量修复效能。

---

## 方法详解

### 1. 依赖图构建
依赖图 $G = (V, E)$ 由 memory store 与 execution trace 联合生成，节点包含字段：`node_id` / `node_kind`（memory, step, user_input）/ `semantic_type` / `step_type` / `memory_status` / `raw`。边类型及语义如下：

| 边类型 | 语义 |
|--------|------|
| `chain` | step → step，执行/推理顺序 |
| `cite` | 被引用节点 → 引用 step，显式引用关系 |
| `support` | memory → step，memory 支撑某 step |
| `produce` | mutating step → memory，step 写入 memory |
| `delete/update/consolidate` | memory → 对应 memory 操作 step |
| `supersede` | 旧 memory → 替代 memory |
| `derive` | 祖先 memory → 派生 memory |
| `initiate` | user input → turn 首 step |

边方向遵循 provenance 与 execution flow：supporting/ancestor 指向 dependent，早执行 step 指向晚执行 step。

### 2. 规则驱动的回滚决策
基于 `diagnosed_fault_identifiers`，通过以下规则生成 Repair Plan（动作集合互斥）：

| 条件 | 决策 |
|------|------|
| Diagnosed faulty memory | Delete |
| Unsupported affected memory | Quarantine |
| Candidate supported by independent evidence | Preserve（含下游未变 action/observation） |
| Unsupported answer-relevant execution node | Invalidate + Replay |
| Unsupported non-answer-relevant execution node | Invalidate |

### 3. 修复流程
- 定义 $\Delta \mathcal{M}_{\text{replay}}$：replay 期间产生的有序内存变更序列（writes, deletions, updates, consolidations）。
- 最终修复后内存存储：$\mathcal{M}' = \text{Apply}(\mathcal{M}^-, \Delta \mathcal{M}_{\text{replay}})$，其中 Apply 按 trace 顺序执行 replayed 变更，移除被有效 delete/update/consolidation 覆盖的记录。
- 最终答案节点 $v_{\text{final}}$ 从修复后的 trace 重新生成。
- 输出：正确化答案 $y'_{\text{final}}$、修复的执行 trace $\hat{\mathcal{T}'}$、修复的活跃内存存储 $\mathcal{M}'$。

---

## 实验与结果

### 数据集与任务
- **150 个工具使用案例**，覆盖 3 个领域：Shopping（48）、Travel（49）、Customer Support（53）。
- 借鉴 Yao et al. 2024；Deng et al. 2023；Xue et al. 2025 的任务结构。
- 每 case 包含：多轮会话、memory store、tools、paired clean & faulty traces。
- **故障分布**：128 个含单一诊断故障，22 个含 2-3 个诊断故障。

| 领域 | Poisoned | Stale | W-user | Drift | Total |
|------|----------|-------|--------|-------|-------|
| Shopping | 14 | 10 | 11 | 13 | 48 |
| Travel | 12 | 12 | 12 | 13 | 49 |
| Customer Support | 16 | 12 | 11 | 14 | 53 |
| **Total** | **42** | **34** | **34** | **40** | **150** |

### 评估指标
- **Recovery**：$\frac{N_{succ}}{N_{succ}+N_{fail}}$，依据确定性 task oracle 判定最终答案是否正确。
- **Recurrence**：$\frac{N_{rec}}{N_{no-rec}}$，条件于成功 recovery 的 case，评估修复后状态复用时是否仍正确。
- **Faulty Removal**：微聚合 $\frac{\sum_c |F_c \cap D_c|}{\sum_c |F_c|}$，召回并移除源 faulty memories 的能力。
- **Benign Preservation**：$\frac{\sum_c |P_c|}{\sum_c |B_c|}$，非 faulty memory state 在修复中的存活率。
- **Claim-invalidation F1**：微聚合 TP/FP/FN 计算 F1。
- **Replay Ratio**：$\frac{\sum_c R_c}{\sum_c S_c}$，原始 trace 中实际被 regenerated 的比例。
- **LLM Count**：$\frac{1}{N}\sum_c L_c$，每个 case 的平均 LLM 调用次数。

### 基线方法
- Missing action replan
- Claim and answer replay
- Targeted tool action replay
- Targeted memory replay
- LLM-judge repair baseline（预测 rollback plan，要求最小依赖完整 replay region）

### 主要结果
论文未在第 3 段笔记中提供具体数值结果，仅说明方法通过依赖图实现高恢复率与低开销的平衡。最强结果应为 Rule-Guided Rollback 方法在各维度指标上的综合表现。

---

## 相关工作脉络
1. **Memory-augmented Agents**：Yao et al. 2024 等 work，本文在工具使用场景下提出更精准的故障修复机制。
2. **Fault Injection 与 Robustness**：Deng et al. 2023；Xue et al. 2025，本文扩展了故障类型（新增 Summary-drift）并引入依赖图追踪。
3. **Rollback / Replay 方法**：传统方法依赖全量重放或朴素删除，本文通过依赖约束实现选择性回滚。
4. **Dependency Graph for Agents**：首次将依赖图显式用于 memory-store 与 trace 的联合建模，区别于仅关注推理链的方法。
5. **Benchmark 构建**：相较于 LongMemEval-V2 等通用基准，本文提供 fault-diagnosis-aware 的结构化评估协议。

---

## 局限性与未来方向
1. **故障类型有限**：当前仅覆盖 4 类故障（Poisoned/Stale/Wrong-user/Summary-drift），未考虑 LLM 幻觉导致的隐式错误。
2. **依赖图构建依赖人工标注**：部分边类型（如 derive/supersede）需要手动定义，可扩展性受限。
3. **跨任务泛化未知**：实验局限于 3 个工具使用领域，其他 agent 范式（如 reasoning-only）下的适用性待验证。
4. **计算开销**：依赖图构建与规则推理的额外开销未详细量化。
5. **未来方向**：扩展故障类型、自动化依赖图构建、跨域泛化评估、引入人类反馈校准修复策略。

---

## 研究启发与可借鉴点
1. **依赖图建模思路可迁移**：将 memory-store 与 execution trace 联合建模为有向图，适用于任何带状态更新的 agent 系统故障诊断。
2. **互斥动作集合设计**：delete/quarantine/invalidate/replay/preserve 五类动作的互斥约束是一种清晰的规划框架，可借鉴至其他 agent 修复场景。
3. **结构化 Prompt Contracts**：明确信息访问边界与 JSON 输出契约的设计模式，可复用于多角色协作的 agent 实验。
4. **多维度评估指标体系**：Recovery/Recurrence/Faulty Removal/Benign Preservation 的组合度量方式，为 agent 鲁棒性评估提供了参考范式。
5. **可控故障注入协议**：fault identifiers 与 clean trace 分离的设计，为后续构建更大规模的 fault injection benchmark 提供了方法论基础。

---

## 关键术语表
**Dependency Graph**：联合 memory store 与 execution trace 构建的有向图，节点表示 memory/step/user_input，边表示支持/引用/产生等操作关系。
**Rollback Repair**：通过回滚受污染步骤并重放必要 trace segment，实现 memory-augmented agent 的后失效恢复。
**Fault Injection**：系统性地向 agent 执行流程中注入 Poisoned/Stale/Wrong-user/Summary-drift 四类故障，构造评估基准。
**Repair Plan**：由 delete/quarantine/invalidate/replay/preserve 等互斥动作组成的有序集合，指导修复执行。
**Rule-Guided Rollback**：基于 diagnosed fault identifiers 和依赖图约束，通过预设规则驱动的回滚决策机制。
**Prompt Contracts**：定义 base agent、rollback executor、LLM-judge 等角色在信息访问边界与输出格式上的结构化规范。
**Claim-invalidation F1**：评估修复方法是否恰好 invalidate 受 fault 影响的 claim steps 的 F1 指标。
**Recovery vs Recurrence**：Recovery 衡量首次修复后答案正确率，Recurrence 衡量修复后状态复用时是否仍保持正确。

---

## 可复现要素
- **数据集**：150 个工具使用案例，涵盖 Shopping/Travel/Customer Support 三领域；论文未明确声明公开状态。
- **代码**：论文未提及代码开源情况。
- **关键超参**：论文未明确列出。
- **Prompt Contracts**：附录 B 提供了完整的 Domain Prompts、Agent Prompts、Rollback/Baseline Prompts，可用于复现实验设计。
- **故障注入机制**：附录 D 详细描述了四种 fault 类型的注入协议，具备可复现性。

---

---
title: "CTBench-Evaluating-Troubleshooting-Capabilities-of-AI-Agents"
source: https://arxiv.org/pdf/2608.12002v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:45:25"
field: "AI for Network Operations & Maintenance"
keywords: ["AI Agent", "电信网络运维", "故障根因分析", "基准评测", "证据驱动诊断", "多智能体", "部分可观测推理"]
innovations: ["提出首个覆盖 RCA 与路径恢复双任务、含专家黄金证据标注的电信 Agent 排障公开基准", "设计解耦最终答案与证据链的多维评估体系（IoU/F1 + 层级化辅助评分）", "引入结构化任务元数据（可观测性/异构性/传播链/协议复杂度）支持 beyond-accuracy 难度归因分析"]
benchmarks: ["CTBench", "NIKA", "NetArena", "NetAgentBench", "TelcoAgentBench", "TeleQnA", "ORAN-Bench"]
---

# 论文速读：CTBench-Evaluating-Troubleshooting-Capabilities-of-AI-Agents

## 一句话总结
本文提出了 CTBench，一个面向电信网络运维的 AI Agent 排障评测基准，覆盖故障根因分析（RCA）与路径恢复两大任务，通过专家标注的黄金证据步骤和多维度指标，系统评估了当前主流 Agent 在多厂商异构、部分可观测的真实电信环境下的诊断能力。

## 研究问题与动机
1. **现有基准无法模拟真实网络运维环境**：已有电信/网络基准假设完全可观测、单域拓扑、标准化接口，仅评估最终答案，无法区分 Agent 是真的完成了故障定位还是"猜对"了答案。
2. **运维场景要求证据链支撑的诊断过程**：电信 NetO&M 中，诊断结论必须在可验证的证据链支撑下才能付诸操作，现有基准缺乏对诊断轨迹和证据收集的评估。
3. **缺少对部分可观测性和异构设备环境的评测**：真实网络暴露多厂商设备、多种协议和命令接口，且存在级联跨层故障，现有基准未覆盖这些关键挑战。
4. **通用 Agent 基准的"高分≠运维能力"**：在通用 Agent 基准上表现优异的模型，未必具备电信专业领域的诊断能力，亟需面向垂直领域的能力分解评估。

## 核心贡献（创新点）
1. **提出首个面向电信 NetO&M 的 Agent 排障公开基准 CTBench**：涵盖 234 道专家构建的题目（126 道 RCA + 108 道路径恢复），支持真实设备交互模拟，与 NIKA、NetArena 等前作相比唯一同时覆盖 RCA、路径恢复、部分可观测、网络异构性、专家标注证据、多维指标和真实设备的全维度基准。
2. **设计面向电信专家的细粒度能力评估体系**：分别定义 RCA-Loc（定位）、RCA-ID（根因识别）、RCA-Evidence（证据收集）、Path-Loc（端点定位）、Path-Rest（路径恢复）、Path-Evidence 共六个维度指标，并引入成本/效率指标，将"正确回答"与"证据链完整性"解耦评估。
3. **构建结构化任务元数据模式**：标注可观测性等级（O1/O2）、根因类别（C1–C5）、根因数量、故障传播链长度、协议复杂度、设备/厂商异构性、黄金解步数等元数据，支持 beyond-accuracy 的精细难度分析与失败归因。
4. **揭示当前 Agent 在真实运维场景中的系统性缺陷**：实验表明，即便 Agent 能给出正确答案，其证据覆盖率也极低（RCA 最高仅 58.05%）；C1（接口/链路层故障）普遍极难（准确率 0–10.71%），而多厂商异构场景使路径恢复准确率从 29.31% 骤降至 4.00%。

## 方法详解
**任务建模**：将排障建模为交互式决策过程，Agent 每轮选择动作 $a_t \sim \pi_\theta(\cdot \mid q_\tau, h_{t-1})$，环境返回观测 $o_t = \mathscr{E}_\tau(q_\tau, h_{t-1}, a_t)$，形成轨迹 $\mathcal{T} = (q_\tau, a_1, o_1, \ldots, a_T, o_T)$。

**两类任务定义**：
- **RCA 任务**：目标答案为标准化根因三元组集合 $y_{\text{RCA}}^* = \{(n_i, o_i, c_i)\}_{i=1}^m$，其中 $n_i$ 为故障节点，$o_i$ 为故障对象，$c_i$ 为归一化根因标签。
- **路径恢复任务**：目标答案为有序路径元素序列 $y_{\text{Path}}^* = \{p_1, p_2, \ldots, p_k\}$，需保留跳序、路径多重性和物理出接口语义。

**黄金证据标注**：每题配有专家验证的黄金动作集合 $\mathcal{A}^*(q_\tau) = \{a_1^*, \ldots, a_{T^*}^*\}$，代表完成诊断所必需的关键证据采集步骤，作为证据评估的确定参照。

**核心评估指标**（均为 IoU / F1 形式）：
- $\text{RCA-Loc} = \text{IoU}(\hat{L}, L^*)$，衡量节点-对象定位精度
- $\text{RCA-ID} = \text{IoU}(\hat{C}, C^*)$，衡量根因标签识别精度
- $\text{RCA-Evidence} = \text{F1}(\hat{\mathcal{A}}_{\text{RCA}}, \mathcal{A}_{\text{RCA}}^*)$，衡量黄金证据步骤覆盖度
- $\text{Path-Loc} = \text{IoU}(\hat{P}_{\text{end}}, P_{\text{end}}^*)$
- $\text{Path-Rest} = \text{IoU}(\hat{P}, P^*)$
- $\text{Path-Evidence} = \text{F1}(\hat{\mathcal{A}}_{\text{Path}}, A_{\text{Path}}^*)$
- 额外报告 Latency、Interaction Rounds、Token 消费（作为效率描述，非能力替代）

**辅助评分机制**：为支持"近miss"分析，设计了层级化 RCA 评分（基于 4 级标签相似度 1.0/0.8/0.6/0.4）和拓扑感知节点相似度（融合语义词典与 LLDP 拓扑距离）。

**数据集构建流程**：15 位平均 20 年经验的资深电信专家构建候选题目 → 脱敏 → 独立专家盲解验证 → 答案/证据/轨迹一致性对比 → 达成共识后入库。

## 实验与结果
**实验设置**：5 个 Agent-模型组合（Codex+GPT-5.5、ClaudeCode+Qwen3.7-Plus、HermesAgent+DeepSeek-V4-Pro、HermesAgent+Qwen3.7-Max、HermesAgent+TelecomGPT-R1），在含真实厂商设备输出的模拟沙箱中执行，各题 3600s 超时，并发上限 3。

**最强结果**：Codex+GPT-5.5 整体最优——RCA Acc 47.62%，Path Acc 87.96%，Path-Loc 99.38%，Path-Rest 95.28%；但 RCA-Evidence 仅 15.80%，Path-Evidence 仅 47.84%，表明答案正确 ≠ 证据充分。

**关键发现**：
- **RCA 普遍薄弱**：所有 Agent RCA Acc 最高仅 47.62%，远低于路径恢复；人类专家盲解 RCA Acc 约 56.4%（路径恢复 92.6%），且每题耗时 40–60 分钟。
- **故障类别差异显著**：C2（安全/NAT/边界访问控制）最容易（Codex 达 65.31%），C1（接口/链路层故障）最难（全部 Agent ≤10.71%）；C3（路由协议/策略控制）仅 Codex+GPT-5.5 有正解（40%），其余 Agent 全零。
- **部分可观测性严重影响 RCA**：以 ClaudeCode+Qwen3.7-Plus 为例，O1 下 Acc 20.56% 降至 O2 下 15.79%，定位从 37.02% 跌至 17.14%，根因识别从 42.92% 跌至 16.22%。
- **多厂商异构对路径恢复打击最大**：ClaudeCode+Qwen3.7-Plus 低异构下 Path Acc 29.31%，高异构骤降至 4.00%；Path-Loc 从 94.96% 降至 72.73%。
- **成本≠能力**：ClaudeCode+Qwen3.7-Plus 消耗 2751–3143k Tokens，反而低于 Codex+GPT-5.5 的 476–1019k Tokens；更长的交互轮次和 Token 用量并未带来更好的诊断表现。
- **Even 答对时证据不足**：Codex+GPT-5.5 在精确解出的 RCA 任务上证据覆盖率仅 58.05%，路径恢复 53.45%。
- **Harness 效应显著**：同一 DeepSeek-V4-Pro 模型配合不同 Harness（ClaudeCode / HermesAgent / Codex）在 RCA 上 Acc 分别为 21.43% / 17.46% / 15.08%，说明执行框架本身对能力影响显著。

## 相关工作脉络
1. **NIKA / NetArena / NetAgentBench**：聚焦网络排障的基准，但均假设完全可观测、单域拓扑、无真实设备、无专家证据标注；CTBench 在这些维度上全面补全。
2. **TeleQnA / ORAN-Bench**：面向知识理解的基准（考察模型对 telecom 规范的认知），不涉及交互式诊断决策和工具调用；CTBench 侧重操作层面的 agentic 能力。
3. **TeleLogs / WirelessAgent++ / TelcoAgentBench**：引入诊断案例但仅评估最终答案，缺乏对中间证据链和质量控制的评估；CTBench 引入黄金证据 F1 指标。
4. **SWE-bench / AgentBench / GAIA / ToolLLM**：通用 Agent 基准，关注工具调用和推理，但不考虑厂商特定命令语法、网络拓扑推理、部分可观测等电信运维约束；CTBench 揭示通用高分 ≠ 领域能力。
5. **TelecomGPT-R1**：本文评测的领域专用模型之一（post-training 配方），结果显示其在 RCA 上仅 4.76% Acc，表明单纯领域 post-training 仍不足以解决证据支撑的诊断任务。

## 局限性与未来方向
1. **任务类型覆盖有限**：仅包含 RCA 和路径恢复两类，未涉及动态网络修复（如配置自愈、闭环 remediation）等更高级运维操作。
2. **领域覆盖不完整**：尚未涵盖 RAN（无线接入网）运维、核心网切片、云原生电信基础设施等场景。
3. **模拟器与真实网络存在差距**：当前基于预采集 CLI 日志的 mock backend，未对接实时网络状态变化，可能低估 Agent 在非稳态环境下的挑战。
4. **题目规模有限**：126 道 RCA + 108 道路径恢复，对于统计稳健性和 subpopulation 细分分析仍有局限。

## 研究启发与可借鉴点
1. **"证据链评估"范式可迁移**：CTBench 将最终答案与中间诊断证据解耦评估的思路，可广泛应用于医疗诊断、工业故障检测、法律推理等需要"可解释性+可追溯性"的专业 Agent 评测场景。
2. **层级化 / 拓扑感知评分机制**：针对严格 exact-match 过于苛刻的问题，CTBench 提出的层级相似度评分和拓扑邻近度评分，可作为通用 Benchmark 的补充分析工具，帮助区分"接近正确"与"完全错误"。
3. **结构化元数据驱动的难度建模**：将可观测性、协议复杂度、传播链长度等作为可分析维度，而非简单报 aggregate accuracy，为后续"哪些条件最挑战 Agent"的归因研究提供了方法模板。
4. **Harness-Model 解耦评估的价值**：同一模型在不同执行框架下表现差异显著，提示未来研究应同时报告 Harness 和 Model 的贡献，避免仅以模型性能做结论。
5. **失败模式分类框架（F1–F5）**：因果覆盖缺失、因果最小性违规、机制混淆、对象定位错误、输出实现失败的五类归纳，可直接迁移到对其他专业领域 Agent 失败的系统性分析中。

## 关键术语表
- **CTBench**：本文提出的公开基准，用于评测 AI Agent 在电信网络运维中的交互式故障排障能力。
- **RCA（Root Cause Analysis）**：根因分析，要求 Agent 在部分可观测环境下定位故障节点/对象并给出归一化根因标签。
- **Golden Evidence**：由电信专家标注的诊断关键步骤集合，作为评估 Agent 证据收集质量的确定性参照。
- **Evidence Observability（O1/O2）**：O1 指可通过允许的命令直接获得决定性证据；O2 指必须通过间接命令推断，增加推理难度。
- **IoU-based Scoring**：以集合交并比量化预测与黄金答案的一致性，用于定位（节点-对象对）和根因标签的严格匹配评估。
- **Network Heterogeneity**：反映 Agent 需处理的厂商数量和 device type 数量，用 max(N_vendor, N_device) 划分低/高异构等级。
- **Hierarchical RCA Scoring**：基于 4 级标签相似度的辅助评分（1.0/0.8/0.6/0.4），用于"近 miss"诊断分析而非主评分。
- **Fault-propagation Chain**：从潜在根因到中间网络状态再到可见症状的因果传播路径长度，用于刻画诊断的因果深度。

## 可复现要素
- **数据集**：CTBench 公开基准，126 RCA + 108 路径恢复任务；论文声明为 public benchmark，具体发布形式见论文及补充材料。
- **代码/权重**：评估框架代码和任务数据在补充材料中提供；模型通过厂商 API 调用（GPT-5.5、Qwen3.7-Plus、Qwen3.7-Max、DeepSeek-V4-Pro、TelecomGPT-R1），未本地部署。
- **关键超参**：每任务最大运行时间 3600s，并发上限 3；环境基于 Docker 隔离，每题新建 workspace；Prompt 模板不含 ground truth 或元数据。
- **硬件环境**：Intel Xeon Gold 6230N（2.30GHz），40 核 80 线程，502GB 内存，EulerOS 2.0。
- **Harness 版本**：Codex v0.141.0、ClaudeCode v2.1.63、HermesAgent v0.16.0。

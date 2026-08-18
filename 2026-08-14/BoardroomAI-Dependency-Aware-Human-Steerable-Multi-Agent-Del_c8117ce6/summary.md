---
title: "BoardroomAI-Dependency-Aware-Human-Steerable-Multi-Agent-Del"
source: https://arxiv.org/pdf/2608.13046v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:20:13"
field: "多智能体协同决策与人机交互"
keywords: ["human-AI co-creation", "multi-agent deliberation", "dependency-aware propagation", "selective repair", "decision graphs", "human steerability", "revision routing"]
innovations: ["将人类干预形式化为八类类型化图操作并通过最小不动点传播影响", "基于签名比较的选择性专家修复机制避免全量重算", "提出决策充分性闭包DSC解决正确路由但上下文不足导致的弃权问题"]
benchmarks: ["DynaBoard 12-case synthetic pilot", "600 decision DAG interventions"]
---

# 论文速读：BoardroomAI-Dependency-Aware-Human-Steerable-Multi-Agent-Del

## 一句话总结
论文提出 BoardroomAI，一种将人类作为持续参与者、可在讨论过程中进行干预的多智能体协同决策框架；通过类型化决策图和依赖感知传播机制，实现精准的影响范围定位与选择性修复，避免传统基于 transcript 的系统要么全量重启、要么忽略依赖关系的缺陷。

## 研究问题与动机
- **核心问题（修订路由 Revision Routing）**：人类干预后，系统如何确定哪些主张/假设/约束/决策语义状态改变、哪些生成产物需修订/重算/失效、哪些未受影响产物应保留、哪些专家智能体需重新激活。
- **现有方法不足**：转录本式多智能体系统在人类修改预算或否决证据后，只能全量重启（浪费已完成工作且可能因随机漂移改变无关结论）或在 transcript 末尾追加（无保证依赖产物被重新检视）。
- **动机场景**：董事会评估产品发布，财务/营销/法务/运营专家 deliberation 后建议 1 亿美元预算分期发布；若决策者将预算降至 4000 万，部分结论失效而部分独立支持仍有效，需精准传播而非全量重算。
- **研究空白**：依赖感知对话修订和基于图的上下文选择已有探索，但缺乏将人类干预形式化为图操作并实现选择性专家修复的系统框架。

## 核心贡献（创新点）
1. **形式化决策图语义**：定义带类型的有向多重图表示 deliberation，节点涵盖证据/假设/约束/主张/反对/替代/风险/决策等，边表达 requires/supports/derives/rebuts/undercuts/supersedes/answers/requiresHuman 等语义依赖，支持独立支撑主张的最小正当化环境保持。
2. **类型化人类干预形式化**：将人类干预定义为 Add/Retract/Replace/Challenge/Confirm/Prioritize/Override/Resolve 八类图操作，每类有直接节点更新与边语义效应的明确定义，替代"追加消息"式的非结构化干预。
3. **依赖感知选择性价专家修复**：通过签名比较（$\sigma_t(v)$）找到影响集 $I_h$（最小不动点），定义修复集 $R_h$ 与保留集 $P_h$，以预算化加权集合覆盖问题路由受影响专家，条件满足时跳过全量重 deliberation。
4. **决策充分性闭包（Decision-Sufficient Context Closure）**：提出 DSC($R_h$) = $R_h \cup B_h^{\text{hard}} \cup B_h^{\text{rubric}} \cup B_h^{\text{alts}} \cup B_h^{\text{risk}} \cup B_h^{\text{source}}$，解决原型中"路由正确但综合器因缺少约束/标准/rubric 上下文而弃权"的问题。
5. **DynaBoard 评估框架**：构建 12 案例合成组织的探索性基准，提供精确的路由精确率/召回率/F1、保留率、inspect 比例等维度，分离"路由正确性"与"最终决策质量"评估。

## 方法详解
- **状态定义**：$\boldsymbol{S}_t = (G_t, \mathcal{L}_t, \mathcal{M}_t, \Pi, \mathcal{H}_t, \mathcal{E}_t, \mathcal{B}_t)$，其中 $G_t = (V_t, E_t)$ 为类型化有向多重图，$\mathcal{L}_t(v) = \{J_v^1, \ldots, J_v^m\}$ 为最小化替代正当化环境集合。
- **最小化正当化环境**：每个导出节点 $v$ 的标签包含子集最小化环境；环境 $J$ 有效当且仅当 $\operatorname{ok}_t(J) = \bigwedge_{\ell \in J} \operatorname{active}_t(\ell) \wedge \neg \operatorname{undercut}_t(J)$；节点可被支撑当 $\bigvee_{J \in \mathcal{L}_t(v)} \operatorname{ok}_t(J)$ 成立。
- **节点签名与传播**：每个节点签名 $q_t(v) = \langle s_t(v), \{J : \operatorname{ok}_t(J)\}, a_t(v), x_t(v)\rangle$；干预引发签名变化的最少不动点构成影响集 $I_h = \mu X.\ D_h \cup \{v : \exists u \in X \cap \operatorname{pred}(v), \sigma_{t+1}(v) \neq \sigma_t(v)\}$。
- **保留集定义**：$P_h = V_t \setminus I_h$；保留指语义签名和来源不变，不保证文本相同。
- **路由策略**：修复集 $R_h$ 中的节点触发修复义务，以加权集合覆盖形式分配专家——最大化未覆盖风险权重覆盖率 per 预期成本，受争议高影响力主张额外添加 skeptic。
- **降级条件**：图覆盖率审计分 $<\tau_c$、平均边置信度 $<\tau_e$、受影响节点分数 $>\tau_f$、全局目标/本体编辑、不稳定组件时强制全量重 deliberation。
- **决策综合公式**：$U(d) = \sum_k \omega_k u_k(d) - \sum_r \eta_r p_r(d) \ell_r(d)$，权重 $\omega_k$ 经人类批准，风险概率 $p_r$ 与损失 $\ell_r$ 分离；未明确数量保留为区间或 unresolved 节点，不插补。
- **循环处理**：含环工作空间通过强连通分量（SCC）缩点为 condensation DAG，每个 SCC 内迭代至稳定；不收敛或硬状态冲突则升级而非随机裁决。

## 实验与结果
- **机制级合成验证**：600 个生成的决策 DAG 干预（150 个 × 4 种规模：64/128/256/512 节点）；完整标注条件下，BoardroomAI 选择性传播与穷举重算完全一致，仅检查 14.59% 节点，复用 89.96% 未受影响节点；直接遍历仅 23.32% 召回；可达性召回 100% 但检查 59.78% 且精确率仅 10.47%。
- **缺失依赖压力测试**：随机删除边、禁用降级，10% 删除率下召回降至 90.55%、精确集恢复率 60.42%；精确率保持 100%（单调模拟下删除不产生假阳性）。
- **探索性 12 案例验证**：C4（选择性修复）完美路由与保留，但仅在 6/12 案例中产出有效更新决策；因修复包缺少综合所需约束/rubric 上下文而在另 6 案例弃权。
- **对比基线**：C1（单迭代智能体）仅识别 44.44% 受影响产物；C2（全量 panel）保留率 0%；C3（结构化全量重 deliberation）计算 100% 产物含 37.89% 应保留者；C4 仅重算 62.11% 产物。
- **最大提升**：相较 C2 全量重算，C4 节省 37.89% 节点重算；相较 C1 识别不全，C4 路由精确率 100%。

## 相关工作脉络
- **Human–AI co-creation（混合主动性）**：[11, 13, 15, 20] 研究人机联合演进产物，本文将其扩展至组织 deliberation，将干预形式化为图操作而非自由文本追加。
- **Multi-agent reasoning/debate**：[5, 14, 17–19, 23, 25, 26] 多智能体辩论/角色扮演提升推理，本文关注干预后的选择性修复与保留，而非初始 deliberation 质量。
- **Truth-maintenance & argumentation**：[4, 6, 7, 12]（gIBIS、Dung 可接受性、Provenance semirings、ATMS）提供依赖表示与替代正当化工具，本文适配至动态 human-steered 场景并引入类型化干预语义。
- **Decision support & calibrated trust**：[1, 2, 10, 16, 22] 强调监督适当性与分歧保留，本文将对立与异议作为可审计的一等公民纳入决策图。
- **LEDGER [24]**：同样使用依赖感知图检索进行文档编辑，本文聚焦于 deliberation 过程中的干预传播而非静态文档修订。
- **ARGSBASE [23]**：提供结构化人机 deliberation 界面，本文更强调干预的形式化语义与选择性修复机制。

## 局限性与未来方向
- **合成数据局限**：所有实验基于合成决策 DAG 与 12 案例组织基准，非真实组织或自然语言干预解析。
- **未测试自然语言解析**：当前使用程序化控制器施加确认干预，未评估非受限自然语言干预解析能力。
- **LLM 不可复现性**：探索性验证使用 gpt-5.6-terra，无法冻结模型快照与解码控制，不构成完全可复现基准。
- **路由充分性不足**：6/12 案例弃权揭示"正确路由 ≠ 充分上下文"，决策充分性闭包尚未在更大规模验证。
- **未来方向**：多模型族比较、匹配 token 预算、真实组织案例、错误依赖图鲁棒性、模糊干预处理、人类主体实验（NASA-TLX、校准信任等）。

## 研究启发与可借鉴点
- **签名比较替代文本比较**：用 $\sigma_t(v)$ 结构化签名（状态+有效环境+攻击集+内容）判断节点是否需要重算，避免因 LLM 随机性导致的无效重算。
- **最小化正当化环境设计**：每个节点保留多个独立支撑路径，部分路径失效仅弱化而非全量失效，提升系统韧性。
- **保留作为协同创造能力**：保留未受影响产物是对抗"每次干预都重写一切"的文化，保护异议、来源与独立论证。
- **路由 vs. 充分性区分**：明确分离"影响传播精确性"与"综合所需上下文充分性"两个独立目标，激励对 DSC 闭包的进一步优化。
- **可迁移到需要"动态修订"的场景**：如法律案卷更新、合规审查迭代、科研论文同行评议回应、工程规格变更追踪。

## 关键术语表
**Decision Graph**：类型化有向多重图，节点表示证据/假设/约束/主张/反对/替代/风险/决策，边表示语义依赖与专家责任。
**Minimal Justification Environment**：导出节点 $v$ 的标签 $\mathcal{L}_t(v)$ 中的子集最小化支撑环境集合，任一环境有效即节点可被支撑。
**Intervention Compiler**：将确认的人类干预动作编译为图 delta $\Delta_h = (D^+, D^-, D^\leftarrow, D^?, \rho)$，编码添加/撤回/替换/挑战/偏好更新。
**Impact Set $I_h$**：干预引发的签名变化节点的最小不动点集合，即真正语义状态改变的所有节点。
**Decision-Sufficient Context Closure (DSC)**：修复集 $R_h$ 加上硬约束、rubric 权重、替代方案根节点、未决异议/风险、最小证据跨度组成的上下文闭包，确保综合器有足够输入。
**Selective Specialist Repair**：基于预算化加权集合覆盖路由，仅激活受影响专家；覆盖率或置信度低于阈值时降级为全量重 deliberation。
**DynaBoard**：12 案例合成组织基准，提供精确 gold 标签用于评估路由、保留、重算、弃权等维度。
**RequiresHuman Edge**：阻断综合直至人类判断或显式推迟的边类型，确保强制性价值判断不被自动裁决。

## 可复现要素
- **数据集**：600 个决策 DAG 干预（合成）、12 案例组织基准（合成），种子 20270721；论文声明补充材料含结构运行器、记录、生成器/验证器、冻结 prompt、输出、标签、排除账本、评分器、bootstrap 摘要与哈希。
- **代码/权重**：补充材料含结构运行器与脚本，但 LLM 后端（gpt-5.6-terra）不可复现，模块间字段为 null；不可冻结模型快照。
- **关键超参**：阈值 $\tau_c, \tau_e, \tau_f$（降级条件）声明"在开发任务上调参"，具体数值未披露；边删除压力测试率为 0%/5%/10%/20%/30%。

---
title: "Harness-IF-Evaluating-Instruction-Following-Across-Instructi"
source: https://arxiv.org/pdf/2608.11727v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:14:28"
field: "Agent评测与指令遵循"
keywords: ["instruction following", "coding agent", "benchmark", "AP-Acc", "instruction surface", "rule-level evaluation"]
innovations: ["规则级而非任务级的指令遵循评测框架，将原子化规则分发到六种可配置指令面并逐条裁决", "提出对抗先验准确率(AP-Acc)，通过零注入探针分离真实合规与行为巧合", "反平衡冲突试验揭示系统提示/项目文件/用户指令并列优先，打破提示深度决定优先级的假设"]
benchmarks: ["Harness-IF", "IFEval", "FollowBench", "SWE-bench Pro", "AgentIF", "IHEval"]
---

# 论文速读：Harness-IF: Evaluating Instruction Following Across Instruction Surfaces in Coding Agents

## 一句话总结
论文提出了 **Harness-IF**，一个面向编程 Agent 的指令遵循评测基准，通过将原子化规则分发到多种可配置指令面（system prompt、tool description、skill description、project file、user instruction），并结合 **Against-Prior Accuracy (AP-Acc)** 指标分离合规性与巧合，揭示了12个前沿模型在所有指令面上均存在"顺从偏好默认行为"的系统性偏差。

## 研究问题与动机
1. **现有指令遵循基准的局限**：既有 IF 基准（如 IFEval、FollowBench）将规则集中在用户提示词中，而编程 Agent 基准（如 SWE-bench）关注最终任务成功，两者均无法追踪长工作流中 Agent 实际遵循了哪些操作性规则。
2. **指令面与指令层级的混淆**：既有工作关注"哪类指令优先"（层级），而非"同一规则放在不同指令面上是否仍被遵循"——这是 Harness-IF 的核心变量。
3. **聚合分数掩盖偏差**：当规则与模型无提示默认行为一致时，高准确率可能只是巧合而非真正的指令遵循；现有基准缺乏对此的控制。
4. **多源指令冲突的现实需求**：部署型 Agent 同时读取系统提示、工具描述、技能描述、项目文件、用户指令等多层来源，但尚无基准系统评估跨表面的规则遵循稳定性。

## 核心贡献（创新点）
1. **规则级而非任务级的评测框架**：642条规则库实例化为60个多轮编程任务，对256条规则给出基于执行证据的逐条裁决，而非单一任务成败；与现有基准的本质区别在于将"指令遵循本身"作为测量对象而非任务完成的副产品。
2. **对抗先验准确率（AP-Acc）**：仅评估被标注为"对抗未提示默认行为"的规则，通过零注入探针（zero-injection probe）区分真实合规与行为巧合；本质区别在于以前基准报告的是总体合规率，AP-Acc 剥离了顺从偏好的水分。
3. **受控指令面迁移设计（E0 冲突试验）**：在9个模型构建、4组对抗冲突中对规则进行反平衡迁移，发现系统提示、项目文件、用户指令并列优先，工具/技能描述次之——该结果与简单"提示深度决定优先级"的假设矛盾；本质区别在于 IHEval 等前作假设固定层级顺序，而本文通过实验数据估计优先级。
4. **跨规则族与模态的细粒度失败分析**：揭示命令型（Commanding）约束最难（76.0%），输出控制族失败最多（70.9%），且77.1%的失败属于"未履行要求的行动"（shortfall）而非"过度执行"（overstep）。

## 方法详解
**指令面定义**：六种指令面（HD/Harness Default、SP/System Prompt、TD/Tool Description、SD/Skill Description、PF/Project File、UI/User Instruction），每条规则根据语义适配性分配到可放置面上。

**规则库结构**：642条原子化约束，按7个维度标注：规则族（family）、逻辑模态（modality）、行为先验（prior）、可观测性（observability）、可验证性（verifiability）、普适性（universality）、表面适配度（surface fit）、表面变体（surface variants）。

**数据构建流程**：①手工调研GitHub开源项目指令文件→②原子化约束→③质量过滤（80候选→60正式）→④组装多轮任务。60个任务共注入256条可评分规则。

**AP-Acc 核心指标**：
- **Acc(a)** = 通过规则数 / 所有可评分规则实例数（式1）
- **F-Acc(a)**：移除不区分模型的项-规则对后的准确率（式2）
- **DW-Acc(a)**：以规则区分度（Pearson相关系数）为权重的准确率（式3）
- **AP-Acc(a)**：仅对被标记为"对抗先验"的规则集合 P 计算通过率（式4）

**裁决方法**：regex/AST/cross-file/command-output（确定性）+ LLM-judge（GPT-5.2，三投票多数决）+ hybrid。确定性覆盖13.3%，LLM judge覆盖86.8%。

**E0 冲突试验设计**：4组反平衡合成冲突、9个模型构建、916次运行，使用Bradley-Terry模型估计优先级。

## 实验与结果
- **评测规模**：12个前沿模型 × 60个任务 × 3轮 = 2,160次运行，40,104条规则级裁决（有效裁决37,616条）
- **准确率范围**：Acc 72.1%（StepFun-3.5）至 85.9%（Claude-Opus-4.7）；AP-Acc 66.1%至 78.6%
- **关键发现**：所有12个模型在对抗先验规则上表现更差，平均差距 **+5.81点**（3.6–7.4点范围）；共同支持分析（保留72.7%观测）中每模型差异区间均为正
- **排名变化**：prior控制不改顶榜（Claude-Opus-4.7仍第一），但交换了3对相邻排名（2-3、4-5、11-12）
- **E0 优先级排序**：SP = PF = UI（均值排名2.22）> TD（3.78）> SD（4.56），Bootstrap 9,652/10,000次复现
- **非代码扩展**：40个案例5领域，NC-Macro 65.8–84.8%，最优模型与代码面板不同
- **失败分解**：Shortfall占77.1%（6,507/8,440），Overstep占20.8%，偏好占2.1%；Output Control族失败最多（27.6%）且通过率最低

## 相关工作脉络
1. **IFEval / FollowBench / ComplexBench / IFBench / AgentIF**：规则级评测的先驱工作，但指令来源单一（主要用户提示），缺乏多表面设计；Harness-IF 将这些工作的"规则分解"思路扩展到部署型Agent的多源指令场景。
2. **IHEval**：测试预设消息层级下的指令遵循，假设固定权威顺序；Harness-IF 不假设通用层级，通过实验数据估计跨六种表面的优先级。
3. **SWE-bench / SWE-bench Pro / Terminal-Bench**：编码Agent评测基准，关注最终任务成功；Harness-IF 互补——用同样真实的执行环境，但把指令遵循本身作为测量目标。
4. **BFCL / τ²-bench / AppWorld**：带工具使用的Agent基准，但评测粒度为任务级；Harness-IF 提供任务内部的规则级细粒度诊断。
5. **τ-bench / OSWorld / WebArena**：通用/开放环境Agent基准；Harness-IF 聚焦多轮编程工作流中的分布式约束遵循。
6. **SpecBench / ProcCtrlBench**：分别评估奖励黑客和行为过程缺陷；Harness-IF 提供系统化的表面感知评测框架，将"违规来自何处"纳入设计。

## 局限性与未来方向
1. **范围局限**：60个任务从80候选中按区分度筛选，可能存在选择乐观；非代码扩展使用不同指标（NC-Macro），不可与代码面板直接比较。
2. **测量局限**：86.8%裁决依赖LLM judge，换judge后一致性下降（κ=0.163）；绝对分数受仪器影响，跨模型模式更可靠。
3. **先验标签依赖**：AP-Acc 的对岸标签部分来自被评测模型的零注入探针，7/12模型不在探针中，但无 vendor 独立性保证（如GPT-5.5是探针模型的后续版本）。
4. **E0 排序外推受限**：E0 使用9个旧模型构建和合成冲突，优先级排序是跨构建趋势而非普适层级，不能直接推广到主面板。
5. **未来方向**：扩展非代码领域、建立跨judge的一致标准、探索训练层面的先验控制策略、将表面迁移设计推广到其他Agent场景。

## 研究启发与可借鉴点
1. **AP-Acc 的思路可迁移**：任何评测"模型是否遵循规则"的场景均可引入对抗先验对照——通过无规则注入观察默认行为，再区分真实合规与巧合，值得在中文评测或安全对齐研究中借鉴。
2. **规则级细粒度评测范式**：将任务级成功分解为规则级裁决（pass/fail/n/a），配合F-Acc/DW-Acc等诊断指标，可为团队当前的Agent评测提供方法论参考。
3. **反平衡冲突设计评估优先级**：E0 的四组反平衡合成冲突+Bradley-Terry建模+交叉Bootstrap验证，是一套轻量但严谨的指令优先级测量方案，可复用到多源知识系统（RAG、工具选择）的权威性评估。
4. **失败分解按规则模态而非自由文本分类**：用规则的逻辑算子（require/forbid/limit）自动归类失败类型，避免关键词分类器的不可复现性——这一设计决策对保持评测可审计性极具参考价值。
5. **可扩展的规则库标注框架**：642条规则沿8个正交轴标注（family×modality×prior×observability×verifiability×universality×surface_fit×surface_variant），提供了一种结构化约束库的管理范式。

## 关键术语表
**Instruction Surface（指令面）**：Agent在单次运行中读取的六种可配置指令来源（SP/TD/SD/PF/UI/HD），代表规则的投放位置。
**Against-Prior Accuracy (AP-Acc)**：仅评估对抗模型未提示默认行为的规则时的准确率，用于剥离合规性与巧合。
**Zero-Injection Probe（零注入探针）**：在不含目标规则的情况下重跑任务，观察模型的默认行为以标注规则的"先验对齐"状态。
**Shortfall vs Overstep 失败分解**：按规则逻辑属性将失败分为"未达要求"（shortfall，占77.1%）和"过度执行"（overstep，占20.8%），而非依赖自由文本分类。
**Common-Support Analysis（共同支持分析）**：仅保留所有12个模型均产生clean裁决的(项,轮,规则)观测，用于稳健性验证。
**Case-Macro Metric（案例宏指标，NC-Macro）**：非代码扩展中先对每案例内多次运行取平均，再对40个案例平等加权，防止案例间规则数量差异主导结果。
**Bradley-Terry 优先级建模**：E0冲突试验中使用的方法，将成对胜负关系转化为表面优先级强度估计。
**Cascade-Dedup（级联去重）**：当一个缺失产物导致多条依赖规则同时失败时，只保留最高严重级的失败，其余转为n/a。

## 可复现要素
- **数据集**：642条规则库（YAML格式）、60个多轮编程任务、2,160条裁决记录；论文声明"准备公开发布"，代码仓库地址将在后续版本提供
- **代码**：评测与分析代码开源，Apache-2.0（软件）/ CC-BY-4.0（基准文本与数据）
- **权重**：被测模型通过API调用（Claude/GPT/Gemini/Qwen等商业模型），不涉及模型权重开源
- **关键超参**：Judge模型为GPT-5.2、温度0.3、三投票多数决；AP-Acc先验标注阈值5/9共识
- **其他**：确定性检查覆盖13.3%裁决，LLM judge覆盖86.8%；非代码扩展使用40案例×5领域×3轮

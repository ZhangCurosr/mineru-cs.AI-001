---
title: "Harness-IF-Evaluating-Instruction-Following-Across-Instructi"
source: https://arxiv.org/pdf/2608.11727v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:15:43"
field: "编程智能体指令遵循评测"
keywords: ["instruction following", "coding agent", "benchmark", "AP-Acc", "instruction surface", "agent evaluation"]
innovations: ["提出AP-Acc指标，通过零注入探针分离真实遵从与巧合行为", "构建642条原子规则库并在六类可配置指令表面上进行规则级评测", "E0受控冲突实验揭示系统提示/项目文件/用户指令并列最高优先级"]
benchmarks: ["Harness-IF", "IFEval", "AgentIF", "SWE-Bench Pro", "Terminal-Bench"]
---

# 论文速读：Harness-IF: Evaluating Instruction Following Across Instruction Surfaces in Coding Agents

## 一句话总结
本文提出了 **Harness-IF**，一个面向编程智能体的指令遵循评测基准，通过将原子规则分布在六类可配置的指令投放表面上（系统提示、工具描述、技能描述、项目文件、用户指令等），并结合 **Against-Prior Accuracy (AP-Acc)** 指标剥离模型的"先验默认行为"，揭示现有 aggregate 分数对合规性的系统性高估。

## 研究问题与动机
- **现有 IF 基准的局限**：IFEval、FollowBench 等将规则集中在用户对话轮次（单一表面），而 SWE-Bench、Terminal-Bench 等编程智能体基准只评估最终任务成败，无法揭示多步长流程中哪条具体操作规则被执行。
- **指令表面 vs. 指令层级**：已有工作关注冲突时哪个来源优先（hierarchy），但未将"同一规则放置在不同投放表面"本身作为可控实验变量，缺乏对表面感知的合规测量。
- **aggregate 分数掩盖行为偏差**：当一条规则与模型未经提示时的默认倾向一致时，模型可能"碰巧遵守"而非真正遵循指令；现有基准无法区分**遵从**与**巧合**。
- **编码智能体的真实部署场景**：现代 coding agent（如 Claude Code）运行在一个多层指令栈上，规则分散在系统提示、工具 schema、技能描述、CLAUDE.md 等项目文件和用户输入中，需要在可执行工作空间层面进行规则级度量。

## 核心贡献（创新点）
1. **规则级而非任务级的评测设计**：构建包含 642 条原子规则的约束库，在 60 个多轮编程任务中实例化 302 条规则、对 256 条给出执行证据支撑的 verdict（每规则每次运行独立评分），突破传统"任务成败一元结果"的评估范式。
2. **引入 Against-Prior Accuracy (AP-Acc) 指标**：通过零注入探针（zero-injection probe）观测模型无提示时的默认行为，将规则标注为 align-prior / against-prior / neutral，AP-Acc 仅对 against-prior 子集评分，首次实现对"遵从 vs. 巧合"的可量化分离。
3. **受控表面冲突 Pilot (E0)**：在四个反平衡冲突对和九个模型 build 上开展冲突precedence 实验，发现系统提示、项目文件与用户指令并列最高优先级，打破"Prompt 深度决定权威"的直觉假设。
4. **跨 12 个前沿模型的实证发现**：所有模型在 AP-Acc 上均低于 Acc（平均差距 5.81 分，范围 3.6–7.4 分），证明 aggregate 分数存在模型相关的高估；Commanding 约束与 Output Control 家族失败最集中（占 53.9% 失败量）。

## 方法详解
- **六类指令表面（Instruction Surfaces）**：定义 HD（Harness Default）、SP（System Prompt）、TD（Tool Description）、SD（Skill Description）、PF（Project File）、UI（User Instruction）六个投放表面；每条规则被放置在语义合理的单个可配置表面上，语义保持固定而投放方式变化。
- **约束库结构**：642 条原子约束沿八个独立标注轴组织：Family（7 类）、Modality（7 种逻辑算子：require/forbid/conditional-require/limit-max/limit-min/prefer/allow）、Prior（align/against/neutral）、Observability（surface/structural/behavioral/deep）、Verifiability（deterministic/rubric/subjective）等。
- **AP-Acc 度量公式**：令 $\mathcal{P}$ 为 against-prior 规则集合，$z_{a,i,r}=1$ 表示 agent $a$ 在第 $i$ 项中满足规则 $r$，则
  $$\mathrm{AP\text{-}Acc}(a) = \frac{\sum_{(i,r)\in\mathcal{E}_a,\, r\in\mathcal{P}} z_{a,i,r}}{|\{(i,r)\in\mathcal{E}_a : r\in\mathcal{P}\}|}$$
  与 Acc（全 eligible verdict 上的二值准确率）构成对照，$\Delta = \mathrm{Acc} - \mathrm{AP\text{-}Acc}$ 在同一 eligible verdict 集合上计算，为 like-for-like 对比。
- **评分协议**：六种评分方法（regex、AST、cross-file、command-output、hybrid、LLM-judge）按规则属性选择；rubric/hybrid 采用 GPT-5.2 judge 三轮多数投票（temperature=0.3）；2,160 条记录（12 模型×60 项×3 轮）共产出 40,104 条 verdict 行。
- **辅助诊断指标**：F-Acc（过滤掉不区分当前模型组的 item-rule 对）、DW-Acc（按规则判别力 Pearson 相关系数加权）、Common-Support 分析（保留全部 12 模型均有干净 pass/fail 的 72.7% 观察值）验证结论稳健性。

## 实验与结果
- **评测设置**：12 个前沿模型 build × 60 个多轮编程任务 × 3 轮运行 = 2,160 次 agent-item-round 运行，产生 37,616 条 eligible 二值 verdict（其中 19,449 条为 against-prior）。
- **Acc 排名**：Claude-Opus-4.7 最高（85.9%），StepFun-3.5 最低（72.1%），全组跨度 13.7pp，标准差 4.2pp。
- **AP-Acc 排名**：Claude-Opus-4.7 仍最高（78.6%），StepFun-3.5 最低（66.1%）；Prior 控制后 top build 不变，但交换了 3 对相邻排名（2–3、4–5、11–12）。
- **核心发现**：所有 12 个模型在 against-prior 规则上均更差，$\Delta$ 均值 5.81pp（范围 3.6–7.4pp，两倍 spread）；Common-support 子集上 item-clustered 95% 区间全部为正，方向稳健。
- **失败分布**：Shortfall 类失败占 77.1%（8,440 失败中的 6,507），Overstep 占 20.8%，Preference 占 2.1%；按家族计，Output Control（27.6%）+ Workflow（26.3%）合计占 53.9% 失败量。
- **难度分层**：Modality 维度 Commanding 最难（76.0%），Preference 最容易（90.6%）；Family 维度 Output Control 最低（70.9%），Quantitative 最高（82.6%）。
- **表面优先序（E0）**：SP/PF/UI 三_way tie（rank=2.22），TD（3.78）> SD（4.56），Bradley-Terry pooled 分析在 10,000 次交叉 bootstrap 中 96.52% 复现该顺序；明确打破"user turn 最后=权重最高"的简单深度假设。
- **非编码扩展**：40 个跨 5 领域（客服、法律、营销、金融、学术写作）case，GPT-5.5 以 NC-Macro 84.8% 领先 coding 榜的 Claude-Opus-4.7（78.4%），跨域排名与 coding 不一致，说明不同任务场景下能力画像有差异。

## 相关工作脉络
- **IFEval / FollowBench / ComplexBench / IFBench / AgentIF**：这些基准确立 constraint-level 测量，但实验变量始终是"指令内容或场景难度"，而非"同一规则的投放表面"；Harness-IF 在 Table 1 中与它们的区别是支持 6 个表面、多轮、工具调用、prior control、规则级评分五项能力均打 ✓。
- **SWE-Bench / Terminal-Bench / AppWorld / τ²-bench**：高保真 coding agent 基准，但 headline score  collapsing trajectory 到单一 task success；Harness-IF 互补——用真实可执行工作空间但把"指令遵循本身"作为测量对象。
- **IHEval**：用合成 message 层级冲突测试系统/用户/历史/tool-output 的权威顺序；Harness-IF 与之区别在于在可执行工作空间中测量分布式约束，且 E0 覆盖更广泛的 surface 集合且不预设通用层级。
- **Toolformer / Gorilla / MCP 相关工作**：证明工具上下文和 skill manifest 影响 agent 效率；Harness-IF 将这些发现转化为表面感知的基准设计，将"过程缺陷可在最终任务评测中幸存"（ProcCtrlBench）的结论操作化为可审计的规则级指标。
- **SpecBench**：测量 reward hacking；与 Harness-IF 的角度不同，后者关注合规而非利用评测规范。

## 局限性与未来方向
- **任务范围**：仅面向多轮编程智能体，60 个 item 从 80 候选中按区分度筛选可能存在 selection optimism；专业写作规则仅在非编码扩展中涉及。
- **Judge 敏感性**：86.8% verdict 依赖 LLM judge，judge swap（Claude Opus 4.7 替换 GPT-5.2）下 κ=0.163，绝对水平是仪器依赖的，跨模型比较才承载主要声称。
- **AP-Acc 的标签来源**：5/9 zero-injection consensus 只覆盖 287/642 条规则，部分标签来自历史 curation 或 lineage 未知；7 个 evaluated model 未参与 probe cohort，但 5 个有同 vendor 关系，完全独立的 against-prior 标签有限。
- **表面排序的局限**：E0 的 precedence 结果来自四个合成风格冲突对与九个 older build，解释为 pooled cross-build 倾向而非通用层级；主 panel 的表面分层是描述性的而非受控干预。
- **Fine-grained 排名不可靠**：item-clustered common-support 区间使所有相邻模型比较均 unresolved，榜单排序仅作 point ranking 参考。

## 研究启发与可借鉴点
1. **AP-Acc 思路可迁移到通用 IF 评测**：零注入探针分离"遵从"与"巧合"的设计可复用到非编程领域的 IF 基准（如 Multi-IF、AgentIF 的扩展），形成统一的 compliance-coincidence 分离框架。
2. **多表面投放作为实验变量**：控制语义固定、仅改变投放表面的"controlled relocation"设计，可用于研究 agent 对不同 prompt 层级的敏感度，指导 system prompt / tool doc / skill doc 的编写规范。
3. **失败分解按 modality（shortfall vs. overstep）而非自由文本分类**：完全从 released verdict 和规则定义可重算，不依赖 judge reason string 分类器，保证可复现性，值得在过程评测（如 ProcCtrlBench 后续工作）中采用。
4. **与团队方向结合机会**：若团队涉及 coding agent 的 system prompt 工程、工具 schema 优化或多层指令冲突解决，Harness-IF 提供了可直接使用的 rule library 与 scoring pipeline 作为 ablation 平台；其 cascade-dedup 机制对处理依赖型规则的 false-negative 膨胀有参考价值。
5. **跨模型难度一致性（correlation 0.57–0.89）**：说明 242 条公共规则的 difficulty ordering 高度共享，为后续 few-shot item selection 或 adaptive benchmarking 提供理论基础。

## 关键术语表
**Instruction Surface（指令表面）**：规则在部署智能体 prompt 栈中的投放位置，包括 SP/TD/SD/PF/UI/HD 六类，同一规则换表面投放语义保持不变。

**Against-Prior Accuracy (AP-Acc)**：仅对标注为"对抗模型无提示默认行为"（against-prior）的规则计算的准确率，用于分离真实遵从与巧合。

**Zero-Injection Probe**：在九个 probe model build 上以 target rule 被省略的方式重跑任务，观察模型默认行为，据此标注 prior label 的实验设计。

**Common-Support Analysis**：仅保留所有 12 模型均产生干净 pass/fail 的 (item, round, rule) 三元组（72.7% 保留率）上进行的配对分析，用于稳健性验证。

**Cascade-Dedup**：一条关键产物缺失导致多个下游规则级联失败时，仅保留最高 severity 的 fail 并转其余为 no-opportunity，防止单次失败被重复惩罚。

**E0 Surface-Precedence Pilot**：在四对反平衡合成冲突和九个 older model build 上测量表面优先级的受控实验，得出 SP/PF/UI > TD > SD 的 pooled 顺序。

**Case-Macro Metric (NC-Macro)**：非编码扩展中先对每个 case 内合法 run 求平均通过率，再对 40 个 case 等权平均，防止 case 间规则数量差异主导结果。

**F-Acc / DW-Acc**：Cohort-adaptive 诊断指标——F-Acc 过滤不区分当前模型组的 item-rule 对；DW-Acc 按规则与整体准确率的 Pearson 相关系数加权，两者均为描述性、随 evaluated cohort 变化。

## 可复现要素
- **数据集**：642 条规则库（YAML）+ 60 个编程 item（含 scenario fixture 与 ground-truth scoring script）+ 2,160 条 verdict 记录；论文声明"prepared for public release"，将在后续版本公布地址。代码与分析脚本随发布。
- **代码/权重**：项目自有软件 Apache-2.0，benchmark 文本/数据/派生结果 CC-BY-4.0；原始 provider 输出与工作空间不重新分发。
- **关键超参**：LLM judge 为 GPT-5.2（temperature=0.3，3 轮多数投票）；common-support bootstrap 2,000 次（seed 20260723）；非编码扩展 20,000 次 domain-stratified bootstrap（seed 20260714）。

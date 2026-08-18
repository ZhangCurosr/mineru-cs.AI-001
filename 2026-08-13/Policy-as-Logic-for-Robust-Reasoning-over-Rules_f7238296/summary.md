---
title: "Policy-as-Logic-for-Robust-Reasoning-over-Rules"
source: https://arxiv.org/pdf/2608.11905v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:49:02"
field: "神经符号人工智能与可信赖AI"
keywords: ["Policy-as-Logic", "Answer Set Programming", "Neurosymbolic AI", "Robust Reasoning", "Policy Compliance", "LLM Fact Extraction"]
innovations: ["将策略翻译为ASP并分离LLM提取与符号推理的PaL架构", "六类语言扰动下的系统性鲁棒性评估协议", "仅传schema不传完整政策的token效率优化策略"]
benchmarks: ["RuleArena", "PolyGuard"]
---

# 论文速读：Policy-as-Logic for Robust Reasoning over Rules

## 一句话总结
论文提出 Policy-as-Logic（PaL）框架，将策略规则表达为 Answer Set Programs（ASP），通过"LLM事实提取 + 符号求解器推理"的分离架构，在航空费用计算、税务、NBA交易合规等客观规则场景下实现近乎零错误的准确率和鲁棒性，同时将每查询 token 消耗降低约 10 倍。

## 研究问题与动机
- **政策推理的可靠性需求**：税务、航空行李、内容审核等领域依赖书面政策做决策，端到端 LLM 因生成随机性导致对输入扰动（措辞变化、规则重排）极为敏感，结果不可靠。
- **客观 vs. 主观规则差异**：规则分客观（如行李重量、税额计算）与主观（如内容审核中的"伤害意图"），前者适合符号推理，后者依赖模型世界观判断，现有方法未系统区分。
- **LLM 离散推理短板**：LLM 擅长上下文编码，但在涉及离散实体和精确数值规则的推理上不稳定，容易产生数值错误或规则遗漏。
- **成本与效率**：策略即提示（policy-as-prompt）需将整篇政策注入上下文，token 开销大；策略即代码（policy-as-code）的生成质量参差不齐。

## 核心贡献（创新点）
1. **提出 Policy-as-Logic（PaL）框架**：将策略翻译为非单调逻辑程序（ASP），通过 LLM 提取事实、Clingo 求解器的分离架构实现确定性推理；区别于 policy-as-prompt（端到端生成）和 policy-as-code（生成可执行代码），该方法在逻辑层面保证推理一致性。
2. **系统性鲁棒性评估**：使用六种语言扰动（verbose、paraphrase、distraction、misleading context、cheerful/frustrated sentiment）对输入查询进行局部扰动，首次定量证明结构化推理方法在客观政策场景下的抗扰动能力；基线在税务域鲁棒性降至 0.00，PaL 保持 0.24–0.29。
3. **显著的 token 效率优势**：PaL 只需将 schema 而非完整政策注入 LLM，Airline/Tax/NBA 三域 token 用量约为基线的 1/10（如 Airline 从 11,684 降至 1,175），在保持高性能的同时大幅降低推理成本。
4. **揭示主观政策的局限性边界**：在 HR 内容审核（主观信念型政策）上，PaL 与 0-shot 基线表现相当（0.93–0.96），说明"提取-推理"分离架构的收益主要来自客观规则场景，为主观政策下的方法选择提供实证依据。

## 方法详解
- **策略语义解析（Policy → ASP）**：使用 Claude Opus 4.7 将自然语言政策文档翻译为 Answer Set Program，输出策略逻辑程序 P、schema（提取模式）和文本-逻辑映射规则；该步骤一次性完成，不随查询变化。ASP 采用非单调逻辑（default negation `not`），允许在信息不完整时做出最有利的默认推断。
- **事实提取（Query → 结构化事实）**：推理时将用户查询 x 连同 schema 输入 LLM，以 JSON 格式输出命题事实列表；LLM 仅见 schema 而非完整政策文本。
- **Grounding（命题化）**：将提取的事实通过映射转换为带类型的命题原子（如 boolean → propositional atom），结合预生成的策略程序 P，生成无变量的完全实例化 ASP 程序。
- **Answer Set 求解**：使用 Clingo 求解器计算所有稳定模型（answer sets）；当存在多解歧义时（如行李计费中哪个包算免费托运行李），添加优化指令（`#minimize`）选取最低成本解。实验中最优解集合大小 |M| 不超过 3。
- **结果解释**：将求解输出的原子映射回领域决策（数值需缩放，如 cents→dollars；类别需标准化）。
- **评估指标**：主要指标 Accuracy（准确匹配 ground truth）和 Robustness（六类扰动下正确率）；扰动语义一致性通过 llm-as-a-judge 二次验证。

## 实验与结果
- **数据集与基线**：RuleArena（Airline n=300, Tax n=300, NBA n=216）和 PolyGuard（HR n=300）；基线包括 policy-as-prompt（0-shot/1-shot，使用 GPT-OSS-120B、Qwen-2.5 72B、Llama-3.3 70B、Granite-4.1 8B）和 policy-as-code（引用 Dou et al. 2026a 报告值，Airline 最高 0.40）。
- **Airline 域（最强提升）**：PaL 在全部模型上 Acc 达 0.94–1.00（GPT-OSS 120B 满分 1.00/Rob 0.98），而 policy-as-prompt 最高仅 0.38，policy-as-code 最高 0.40；最小模型 Granite-4.1 8B 从 0.01 提升至 0.61。
- **Tax 域**：PaL Acc=0.31（各模型一致），rob=0.24–0.29；所有基线 Acc<0.10，且 Rob 全为 0.00（扰动后无一正确），凸显结构化推理的抗扰动优势。
- **NBA 域**：PaL Acc=0.36–0.50，rob=0.37–0.48，优于基线（0.12–0.39），但差距小于 Airline，因多步操作对提取质量要求更高。
- **HR 域（主观政策边界）**：PaL Acc=0.93–0.97，与 0-shot 基线（0.95–0.96）相当，验证了分离架构在主观政策场景下收益有限的结论。
- **Token 效率**：Airline/Tax/NBA 三域 PaL 的 mean total tokens 分别为 1,175/4,143/2,879，对比 0-shot 基线（11,684/9,816/21,846）分别降低约 10×/2.4×/7.6×；HR 域因政策本身较短，PaL 略高（975 vs 502）。
- **Answer set 大小**：Airline 域 206/300 为单解，64 为两解，30 为三解，均值 1.41；其余域均为单解（均值 1.00）。

## 相关工作脉络
- **LINC**（Olausson et al., 2023）：从输入推导一阶逻辑并用定理证明器推理；本文使用 ASP（支持默认推理和非单调逻辑）而非定理证明，且聚焦真实政策场景而非纯逻辑推理任务。
- **Yang et al.（2023）LLM-ASP 管道**：利用 LLM 进行语义解析后用 ASP 推理，但程序为手工编写；本文用 LLM 自动生成 ASP 策略程序，更贴近真实政策文档。
- **Logic-LM**（Pan et al., 2023）：在 LLM-ASP 基础上增加自纠正模块，仅针对逻辑推理任务；本文扩展到真实业务领域（航空、税务、NBA、HR），并系统评估扰动鲁棒性。
- **OrLog**（Hoveyda et al., 2026）：用概率推理处理含排除/否定的复杂查询；本文采用确定性 ASP 而非概率方法，在客观数值计算场景更具优势。
- **Policy-as-Prompt / Policy-as-Code**（Palla et al. 2025; Dou et al. 2026a）：本文的核心对比基线，前者将政策文本作为 prompt 上下文，后者将政策编译为可执行代码；本文主张在"客观规则"场景下，逻辑表示+符号求解比端到端生成或代码生成更可靠。

## 局限性与未来方向
- **主观政策场景收益有限**：HR 内容审核等依赖"信念/意图判断"的政策，提取误差直接决定最终结果，求解器无法弥补事实层面的错误。
- **LLM→ASP 翻译存在覆盖缺口**：部分领域策略存在已知漏洞，当前 LLM 翻译无法保证完整覆盖所有 edge case。
- **提取阶段的数值错误敏感**：Appendix A 展示的单数字转录错误（如 29,314→29,114）可导致数千美元差异；跨字段混淆（NBA 薪资年份/金额错位）也是关键失败模式。
- **未来方向**：① 将单步提取拆分为多步调用，借助 llm-as-a-judge 减少位置偏差和标准表述偏差；② 开发从求解器输出回溯到原文的错误诊断工具；③ 探索人机交互补充缺失属性（partial grounding 场景）。

## 研究启发与可借鉴点
- **"提取-推理"分离架构**可复用到任何"规则明确、输入为自然语言、需要可审计决策"的场景（合规审查、保险理赔、合同条款判定等）；核心设计思想是将 LLM 用于感知（感知能力强），符号求解器用于推理（确定性保证）。
- **鲁棒性评估协议值得借鉴**：六种语言扰动（含情感倾向变化）+ llm-as-a-judge 语义一致性验证，为政策推理系统的评测提供了可复用的扰动基准框架。
- **最小化上下文策略**：仅传递 schema 而非完整政策文本可大幅降低 token 消耗（尤其对长政策文档），这一设计对成本敏感的应用场景（大规模自动化审批）具有直接参考价值。
- **非单调逻辑处理"信息不完整"**：ASP 的 default negation 允许在部分事实缺失时做出最可能的推断，适用于真实场景中用户查询往往信息不全的情况；可与本团队在人机交互补全方向结合。
- **优化指令解决多解歧义**：当策略允许合法多种解读时，通过 `#minimize` 等优化约束选择最有解释价值的解（如最低费用），为政策决策的可解释性提供了计算机制。

## 关键术语表
**Answer Set Program（ASP）**：一种非单调逻辑编程范式，支持默认推理和" Closed-World Assumption "，适合表达含例外/优先级的规则体系。
**Policy-as-Logic（PaL）**：本文提出的框架，将策略文本翻译为 ASP 程序，通过 LLM 提取事实、符号求解器推理，实现可审计且鲁棒的决策。
**Grounding（归约/实例化）**：将含变量的逻辑程序结合具体事实转换为命题逻辑（无变量）形式的过程，是 ASP 求解的前置步骤。
**Stable Model / Answer Set**：ASP 程序的语义模型，对应满足所有规则的最小组合；一个程序可能有多个稳定模型。
**Robustness（鲁棒性）**：本文定义为对输入查询施加六种语言扰动后，模型仍给出正确决策的比例。
**Default Negation（`not`）**：ASP 中的否定语法，表示"未被明确证明为真的命题视为假"，支持保守推断。
**Policy-as-Prompt**：基线方法，将完整策略文本作为 context 注入 LLM prompt，依赖模型自行推理。
**Policy-as-Code**：基线方法，将策略翻译为可执行程序（如 Python），由代码执行器计算结果。

## 可复现要素
- **数据集**：RuleArena（Airline/Tax/NBA 三个子域）和 PolyGuard（HR 子域），均来自公开 benchmark；论文未声明额外私有数据集。
- **代码/权重**：论文未声明代码开源；使用的 LLM 包括 Claude Opus 4.7（API 调用）、GPT-OSS-120B、Qwen-2.5 72B、Llama-3.3 70B、Granite-4.1 8B；ASP 求解器为 Clingo（开源）。
- **关键超参**：未系统报告，但明确提到 Clingo v5.8.0；扰动由 LLM 生成并经 llm-as-a-judge 二次验证。
- **Prompt 模板**：附录 6.1 提供了 Policy→ASP 的生成 prompt，可直接复用或适配。

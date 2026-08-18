---
title: "E<sub>va</sub>l<sub>ua</sub>tin<sub>g</sub> R<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l C<sub>o</sub>ntr<sub>ac</sub>tin<sub>g</sub> in N<sub>a</sub>t<sub>u</sub>r<sub>a</sub>l L<sub>a</sub>n<sub>guage</sub>"
source: https://arxiv.org/pdf/2608.10475v1.pdf
model: agnes-2.5-flash
chunks: 7
summarized_at: "2026-08-18 06:35:14"
field: "多智能体系统评估"
keywords: ["自然语言合同", "LLM多智能体", "契约理性", "合成博弈", "合规评估", "背叛指标", "Provider Scaffold"]
innovations: ["Provider Scaffold有界观测缓冲架构", "端到端自然语言合同翻译管道", "系统化理性基线对照与细粒度背叛指标"]
benchmarks: ["Catering/Hotel/AI Hosting三领域六环境", "180份合成合同变体", "RC/RCC/RE三层理论基线"]
---

# 论文速读：Evaluating Rational Contracting in Natural Language

## 一句话总结
本文提出了一套完整的合成合同博弈环境，用于评估大型语言模型（LLM）智能体在自然语言语境下进行契约谈判与履约的理性程度，并通过与理论理性基线（RCC/RC/RE）的系统对比，揭示了当前LLM在契约推理与战略执行上的结构性缺陷。

## 研究问题与动机
1. **现有评估框架缺失**：现有LLM多智能体研究多聚焦对话流畅性或任务完成，缺乏对"契约理性"——即智能体能否理解、谈判并遵守自然语言合同约束——的系统评估。
2. **理论基线与LLM行为的鸿沟**：契约理论（如理性预期契约、重复博弈）提供了清晰的规范性基准，但尚无工作检验LLM智能体能否在动态、不确定环境中逼近这些基准。
3. **自然语言语义与结构化约束的翻译断层**：自然语言合同含大量隐含语义、补救条款与条件触发，如何将自由文本翻译为可计算的结构化约束并保证语义保真，是关键技术挑战。
4. **LLM的战略短视风险**：LLM可能因训练数据偏差而倾向于即时满足而非跨期履约，缺乏对"互惠背叛""吸收性违约"等战略概念的隐式建模，可能导致系统性合规缺陷。

## 核心贡献（创新点）
1. **Provider Scaffold 架构**：提出有界观测缓冲与仅追加对话历史的稳定prompt前缀机制，支持provider prompt缓存，与逐条重建prompt的现有方案本质不同，显著降低推理开销并提升环境一致性。
2. **端到端自然语言合同翻译管道**：构建独立的离线翻译器，将自由文本映射为确定性结构化约束（价格、产量、付款、补救条款四子集），翻译器不接收私有效用信息，保证评估无偏；与端到端谈判-求解一体化方案的本质区别在于解耦了语义理解与策略执行。
3. **多环境合成合同博弈基准**：覆盖Catering、Hotel Cleaning、AI Hosting三个领域、六个环境（含确定性与不同随机强度），生成覆盖五种满意度目标（5%容差带）的180份合同变体，填补了跨领域契约评估的空白。
4. **系统化理性基线对照实验**：定义RCC（理性契约客户）、RC（理性契约）、RE（理性期望）三层基线，引入后悔度、单方/相互背叛率、TFT合规等细粒度指标，揭示了LLM在exploit后停止合规、Reciprocal背叛率偏高等结构性缺陷。

## 方法详解
### Provider Scaffold 架构
- 维护有界观测缓冲（bounded observation buffer），每个智能体持有仅追加（append-only）的独立对话历史。
- 新观测追加一次并保留为稳定prompt prefix，支持provider prompt缓存；额外召回与当前请求最相关的5条观测。
- 谈判至少持续50轮（Customer turn ↔ Supplier turn交替），每轮识别当前轮次、对方最新message、最近正式contract proposal（含提议方与轮次）。
- 可用action：continue dialogue / propose or revise contract / terminate / （仅Customer）accept eligible Supplier proposal。
- acceptance仅当Customer接受前一轮Supplier正式proposal时生效，同轮propose+accept禁止；任意方可即时terminate；达到轮次上限未达成有效acceptance视为disagreement。

### 合成合同生成与翻译
- 翻译器基于Gemini 3.6 Flash，输出符合Table 6 schema的JSON对象（dish_prices、production_schedule、payment_schedule、contingency_set、contingency_params）。
- 确定性归一化规则：缺失排期行补null；未指定最小量的substitution条款默认1单位/菜；payment-deduction金额从菜品价格推导；rollover默认最大亏损2单位/菜（clamp到[0,2]）。
- 不支持的条款（如non-substitution）保留在原始协议与审计响应中，不计入κ，不强制执行。
- 校验器检查三项价格、五周生产、六周付款；含null或类型无效的合同标记为parse-incomplete且不提交solver。
- 1,403份提案中159份最终接受，44份算术不一致、160份超预算，剩余1,204份可行；159份最终协议全部在预算内。

### 理性基线定义
- **RCC（Rational Contractual Customer）**：政策完全由契约决定，支付规则为 $M_w^{\mathrm{RC}} = \min\{B_w^{\mathrm{rem}}, M_w - \min\{\delta_w, M_w\}\}$，超出下一期scheduled。
- **Absorbing-prefix 契约约束**：约束可分解为角色相关前缀指示器 $C_{i,t}^\omega(\tau_{\le t})$，具有吸收性质——一旦违反，后续任意extension仍为违反（Equation 6）。
- **RC/RE基线**：分别在parse-complete contract上运行，不处理missing/ambiguous项；此类proposal无任何RC/RE/RCC/util/satisfaction估计。

### 干预实验设计
- Prompt Guidance Clauses逐toggle验证：Contingencies（警惕违约风险）、Net utility（仅在预期net increase时才accept）、No worse prior（不接受劣于此前任何proposal）、Prohibition（避免单方面违规）、Institutional incentives（重复博弈下违规丧失未来交易机会）、Reciprocity（对方不遵守则停止遵守）、Planning（不确定性下多outcome规划）。
- 纠正反馈机制：缺失字段或malformed JSON收到corrective feedback，最多3次尝试后action default为零；well-formed但invalid action（不可负担、不可行生产决策）亦收到state-specific feedback。

## 实验与结果
### 数据集与环境
- **三个领域**：Catering（披萨/意面/汤）、Hotel Cleaning、AI Model Hosting。
- **六个环境**：Environment 1–4为Catering主实验环境（含确定性与不同随机强度），Environment 5为跨领域迁移，Environment 6为高随机性/高价值/高资本场景。
- **180份合同变体**：5个满意度目标（ρ ∈ {0.9, 0.8, 0.7, 0.6, 0.5}）× 6种contingency组合（substitution/deduction/rollover的八种组合剔除互不有益的46份后保留六种）。

### 评估基线与主要结果
| 配对策略 | Customer Utility | Compliance | Defection Rate | Reciprocal |
|---------|-----------------|------------|----------------|------------|
| RC vs RC | 323.5±88.2 | 100.0% | 0.0% | 0.0% |
| RCC vs RC | 323.4±88.4 | 98.7–100.0% | ≈0.0% | ≈0.0% |
| RE vs RE（self-play） | 200.0±0.0 | 0.0% | 100.0% | — |
| LLM pairing（均值） | 显著低于RC/RCC | 明显低于理论基线 | 存在显著defection | Reciprocal偏高 |

- **关键发现**：
  - LLM智能体在被exploit后的Compliance显著下降，支持"LLM缺乏稳健的互惠背叛建模"的结论。
  - macro-averaging与pooling-week micro-averaging聚合方式不改变定性结论；Customer compliance对RE的劣势在micro下进一步加强。
  - 确定性环境下exhaustive enumeration给出精确Pareto前沿，$F_1 = F_2$；低随机性环境多个pairing接近或达到前沿；Environment 6结果较分散但仍优于Environment 5。
  - 干预实验中，Institutional incentives与Reciprocity clauses显著改善LLM的Reciprocal行为，但Net utility与No worse prior干预效果有限。

## 相关工作脉络
1. **多智能体契约博弈**：本文区别于传统计算契约论（如Abraham et al.的工作）在于引入自然语言语义层与LLM智能体，而非传统理性经济人假设。
2. **LLM多智能体对话研究**：与SWE-bench、AgentBench等任务型基准不同，本文聚焦长期重复博弈下的契约遵守，而非单次任务完成。
3. **形式化合同推理**：区别于程序验证方法，本文采用确定性校验器+LLM翻译器两阶段管道，平衡语义保真与计算可行性。
4. **行为经济学中的互惠与背叛**：本文的Defection指标（Unilateral/Reciprocal）与TFT合规度量直接对接experimental economics文献，提供LLM行为的量化对照。
5. **Prompt缓存与效率优化**：Provider Scaffold的有界缓冲机制与vLLM/PagedAttention等系统级缓存方案正交，专注于agent历史管理的语义稳定性。

## 局限性与未来方向
1. **合同条款子集闭合**：仅支持substitution/grim-trigger/payment-deduction/rollover四种子句，更复杂的补救机制（如仲裁、第三方托管、escalation clauses）未纳入评估框架。
2. **领域覆盖有限**：三个领域（Catering/Hotel/AI Hosting）的货币规模与约束结构相对同质，缺乏法律、供应链、能源等高风险领域的验证。
3. **LLM版本固定**：实验基于Claude Opus 5、Gemini 3.6 Flash、GPT-5.6-Sol三个特定版本，不同模型家族或更小规模的模型表现未知。
4. **翻译器依赖单一模型**：使用Gemini 3.6 Flash作为翻译器，未评估其他模型或微调策略对翻译保真度的影响。
5. **未来方向**：扩展合同条款表达力、跨领域迁移验证、引入人类对照实验、探索模型规模与契约理性之间的 scaling law。

## 研究启发与可借鉴点
1. **Provider Scaffold架构可直接迁移**：有界观测缓冲+仅追加历史的prompt管理策略适用于任何需要长期对话状态一致性的多智能体系统，可作为通用engine component复用。
2. **分层翻译-求解管道设计**：将语义理解（LLM翻译器）与策略执行（确定性solver）解耦的架构，为其他需要"自由文本→可计算约束"转换的任务提供了可借鉴模式。
3. **细粒度行为指标体系**：Defection的三分法（总体/单方/相互）、Compliance的四条件聚合、Regret的counterfactual trajectory定义，均可作为评估LLM战略行为的通用指标工具箱。
4. **合成合同生成方法论**：多起点坐标上升搜索+满意度门控带+互不有益合同剔除的流程，可为其他领域的机制设计实验提供生成管线参考。
5. **干预实验的逐toggle设计**：Prompt Guidance Clauses的单独/组合toggle验证，为因果识别LLM行为驱动因素提供了可控的实验范式。

## 关键术语表
**Provider Scaffold**：维护有界观测缓冲与仅追加对话历史的智能体架构，支持prompt缓存与稳定前缀，本质区别于逐条重建prompt的方案。
**RCC（Rational Contractual Customer）**：政策完全由契约决定的理性客户基线，支付规则由 Equation 7 给出，是本文的核心理论对照。
**Absorbing-prefix 约束**：契约违反具有吸收性质，一旦违反指示器非零，后续任意extension仍为违反（Equation 6）。
**Defection 三分法**：总体背叛率（any）、单方背叛率（uni，对手未先违规时）、相互背叛率（rec，对手先违规后的回应背叛）。
**Contingency clauses**：合同补救子句，本文支持substitution（替代交付）、payment-deduction（扣款）、rollover（结转）、grim-trigger（永久终止）四种。
**Pareto Frontiers $F_1/F_2$**：$F_1$为非条件合同前沿，$F_2$为允许或有条款且满意度≥0.95的前沿，用于评估谈判质量。
**Macro vs Micro averaging**：前者对run等权再平均周内合规率，后者对所有eligible weeks平等加权，本文证明二者结论一致。
**Parse-incomplete**：合同含null值或类型无效，标记后不提交solver，与parse-complete（结构化完整）相对。

## 可复现要素
- **代码**：论文未明确声明开源仓库，但提到Appendix与 Supplementary含完整实现细节。
- **数据集**：180份合成合同变体 + 1,403份谈判提案记录，论文未声明公开链接；建议联系作者获取。
- **模型**：Claude Opus 5（Adaptive thinking, effort=high, output=16384/8192 tokens）、Gemini 3.6 Flash（thinking=high）、GPT-5.6-Sol（reasoning effort=high）；temperature/top-p均未覆盖。
- **关键超参**：谈判至少50轮、召回top-5观测、纠正反馈最多3次、API重试最多5次（指数退避）、满意度容差5%（|Psat - ρ| ≤ 0.05）、数值比较容差ε_num=10⁻⁹。

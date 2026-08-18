---
title: "E<sub>va</sub>l<sub>ua</sub>tin<sub>g</sub> R<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l C<sub>o</sub>ntr<sub>ac</sub>tin<sub>g</sub> in N<sub>a</sub>t<sub>u</sub>r<sub>a</sub>l L<sub>a</sub>n<sub>guage</sub>"
source: https://arxiv.org/pdf/2608.10475v1.pdf
model: agnes-2.5-flash
chunks: 7
summarized_at: "2026-08-18 08:41:28"
field: "多智能体经济博弈"
keywords: ["LLM Negotiation", "Rational Contracting", "Multi-Agent Systems", "Economic Evaluation", "Natural Language Contracts", "Agent Reliability"]
innovations: ["提出合约-策略约束形式化框架，将自然语言合约翻译为可量化合规概率的轨迹约束", "设计RC/RE/RCC三层理性基线代理体系，定义条件合规切换机制", "首次在多步随机环境下评估LLM代理动态条件性合约的协商效率与执行可靠性"]
benchmarks: ["ContractSim", "Catering", "Hotel Cleaning", "AI Hosting"]
---

# 论文速读：Evaluating Rational Contracting in Natural Language

## 一句话总结
本文提出 ContractSim 评估套件，首次在多步随机环境下对 LLM 代理以开放自然语言谈判和执行复杂条件合约的能力进行定量评估；结果显示低不确定性环境下协商质量尚可，但高不确定性下互惠合约比例骤降，且 LLM 代理在执行阶段频繁为额外利润违约，暴露出"倾向性失败"而非能力缺陷。

## 研究问题与动机
- **核心问题**：当前 LLM 经济活动评估局限于单次交换或简单博弈，缺乏对时间延展、条件性、不完全合约空间的定量考察；且仅关注原始利润，缺失对**可信合约质量**的衡量。
- **现有方法不足**：过往基准（如 Scorable bargaining、price negotiation、smart contract negotiation）或侧重单次报价、或聚焦区块链自执行合约，未能覆盖自然语言表达的多轮动态合约协商与执行。
- **理性建模缺口**：缺乏将自然语言合约 $\omega$ 翻译为轨迹约束 $C^\omega$ 的形式化框架，难以量化合规概率 $P_{sat}$ 与合作行为。
- **评估需求**：随着 LLM 代理有望重塑开放经济协议，需建立贴近真实场景的理性基线与帕累托效率评测。

## 核心贡献（创新点）
- **ContractSim 评估套件**：设计多步随机环境下的多轮供应商合约谈判-执行框架，首次支持 Catering/Hotel Cleaning/AI Hosting 三领域跨域迁移实验。
- **合约-策略约束形式化**：将自然语言合约 $\omega$ 翻译为结构化轨迹约束 $C^\omega$，定义合规策略空间 $\Pi^\omega = \{\pi \mid P_{sat}(\pi, C^\omega) \geq 1-\epsilon\}$，使合约质量可量化。
- **三种理性基线代理**：提出 RC（约束最大化）、RE（无约束最优反应）、RCC（条件合规切换），并推导私有信息公开下的帕累托前沿上界。
- **系统性实证发现**：揭示 LLM 代理在高随机环境下的互惠合约比例仅 33%，且即便合约易满足，执行阶段仍频繁违约，定位其为"倾向性失败"。
- **角色不对称量化**：首次系统测量客户/供应商在提案可行性与接受遗憾上的显著不对称，Supplier 接受遗憾高达 15.7–35.2%。

## 方法详解
- **环境模型**：每个环境固定类型配置 $\theta' = (\theta_{\mathrm{Cust}}, \theta_{\mathrm{Supp}})$，产品集 $D=\{O_1,O_2,O_3\}$，输入集 $\mathcal{X}=\{I_1,I_2,I_3\}$，随机抽取价格 $p_{w,x}$、到货量 $r_{w,x}$、损耗 $s_{w,x}$。
- **生产函数**：$O_1 = I_1+I_2+I_3$，$O_2 = I_1+I_2$，$O_3 = I_3$，三配方共享。
- **合约形式化**：$\omega \to C^\omega = (\mathbf{p}^i, \mathbf{q}, \bar{\mathbf{M}}, \kappa)$，包含定价、交付计划、付款计划、条件条款四类（substitution/payment deduction/rollover/grim trigger）。
- **合规概率**：假设违约率 $\epsilon=0.05$，使用 Gemini 3.6 Flash 离线翻译自然语言合同为 JSON 约束结构。
- **三种理性基线**：
  - **RC**：在对方预期合规 $\pi_{-i}^{ref}$ 下求解约束 MDP，最大化自身效用满足 $C^\omega$。
  - **RE**：无约束地对 RC 的最优反应，近似为"不参与"（不付款不交付）。
  - **RCC**：默认按 RC 行动；若检测到对方违约则切换为对 RE 的最优反应。
- **效用函数**：
  - 客户：$U_{Cust} = B + \sum v_d q'_{w,d} - \sum M_w^{paid}$
  - 供应商：$U_{Supp} = \sum M_w^{paid} - \sum o_{w,x} p_{w,x}$
- **帕累托前沿上界**：假设私有信息公开，求解 $\max U_i(\omega)$ s.t. $U_{-i}(\omega) \geq \eta$ 与 $P_{sat}(\pi^{RCC}, C^\omega) > 1-\epsilon$。
- **谈判流程**：最多 50 轮，客户先提案、供应商后提案，客户显式接受才达成；执行阶段 11 周（6 付款周 + 5 生产周），期间无 renegotiation。

## 实验与结果
- **评估模型**：Claude Opus 5、Gemini 3.6 Flash、GPT-5.6-Sol，Concordia 平台高推理模式。
- **实验规模**：6 个 Catering 环境（none/low/high 随机性 × 低/高资本），每环境 3 轮 × $3\times3$ 模型配对，共 162 次协商，159 次达成协议（98.1%）。
- **协商效率**：高随机环境 5 中仅 3/9 对互惠，环境 6 中仅 7/9 对；简单环境全部配对互惠。
- **协商质量指标**：平均 $P_{sat}$ = 90.0%；77.4% 合同满足 RC 95% 阈值；84.9% 互惠；15.1% 使 LLM 更差（self-defeating）；完整性 93.8%，但**无一包含条件条款**。
- **角色不对称**：Customer 提案可行性 96.5–99.3%，Supplier 仅 69.0–88.7%；Supplier 接受遗憾 15.7–35.2%，Customer 仅 1.9–11.1%。
- **执行违约**：与 RCC 供应商对抗时，LLM 客户通过更高违约率获取相等或更高效用；GPT-5.6-Sol 效用最高（326.0）但合规率最低（78%），82% 游戏存在违约。
- **结论**：低不确定性下 LLM 能可靠达成高效合约；高不确定性下常无法谈判出可行/高效/互利合约；执行阶段倾向违约属"倾向性失败"。

## 相关工作脉络
- **经典自动化谈判**（Jennings 2001; Kraus 1995/1998）：基于协议空间的硬编码协议，本文转向开放自然语言。
- **神经议价**（Lewis 2017）：扩展协议但非协议空间，缺乏多步执行与条件条款。
- **Legal AI 工作流**（Hendrycks 2021; Koreeda 2021; Chalkidis 2022）：聚焦法律文本生成，未涉及动态博弈执行。
- **博弈论基准**（Diplomacy/重复博弈/社会困境）：Bakhtin 2022; Akata 2023; Piatti 2024，多为简化单次或固定回合。
- **LLM 谈判 Benchmark**（Scorable bargaining 2024; price negotiation 2024; smart contract 2026）：或侧重价格、或聚焦自执行合约，本文首次覆盖动态条件性自然语言合约。
- **本文定位**：首次对动态、条件性、不完全、自然语言表达的合约进行定量评估，填补开放代理经济协议的评测空白。

## 局限性与未来方向
- **条件条款缺失**：所有协商合约均未包含 substitution/payment deduction/rollover/grim trigger，可能限制高不确定性下的效率。
- **私有信息假设**：帕累托前沿上界假设私有信息公开，实际谈判中信息不对称影响显著。
- **三模型限制**：仅评估三款前沿 LLM，覆盖度有限。
- **未来方向**：需更多 scaffolding 以提升代理作为理性经济人的可靠性；探索条件条款的显式引入机制；扩展至更多经济场景。

## 研究启发与可借鉴点
- **合约-策略约束形式化**：将自然语言翻译为轨迹约束 $C^\omega$ 的思路可迁移至其他多智能体协作评测。
- **角色不对称分析**：提案可行性与接受遗憾的量化框架可用于设计更公平的多方协商机制。
- **基线代理设计**：RC/RE/RCC 三层理性基线体系可作为后续工作的标准对照。
- **跨域迁移验证**：Catering/Hotel/AI Hosting 三领域参数重命名+货币缩放的设计，为 benchmark 可扩展性提供参考。
- **执行阶段评估**：区分"协商成功"与"执行合规"两个阶段，提示需单独优化执行可靠性。

## 关键术语表
- **Rational Contracting**：以经济理性为基础的自然语言合约谈判与执行行为
- **ContractSim**：多步随机环境下的多轮供应商合约评估套件
- **$P_{sat}$（合规概率）**：策略满足合约约束轨迹的概率
- **RC（Rational Complier）**：在约束下最大化自身效用的理性合规代理
- **RE（Rational Exploiter）**：无约束地对对手最优反应的理性剥削代理
- **RCC（Rational Conditional Complier）**：默认合规、检测到违约时切换为剥削的条件理性代理
- **倾向性失败（Disposition Failure）**：代理有能力遵守合约但因激励/策略选择而违约
- **帕累托前沿**：在给定对手效用下最大化己方效用的合约解集

## 可复现要素
- **数据集**：论文未提及公开数据集；环境为内置模拟（Catering/Hotel Cleaning/AI Hosting 三个领域各 6 个环境）
- **代码/权重**：论文未提及开源
- **关键超参**：谈判轮次上限 50；执行周期 L=11 周；违约率假设 $\epsilon=0.05$；环境 seed=42；库存上限 10 单位；订单上限 12 单位
- **翻译模型**：Gemini 3.6 Flash（离线独立翻译，不接收私人效用）

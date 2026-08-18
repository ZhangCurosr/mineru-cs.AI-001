---
title: "BEYOND-HANDCRAFTED-SECURITY-TOWARDS-SELF-EVOLVING-DEFENSE-FO"
source: https://arxiv.org/pdf/2608.12977v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:51:43"
field: "LLM Agent安全与防御"
keywords: ["LLM Agent安全", "运行时防御", "提示注入", "自演化", "Harness优化", "AgentCanary"]
innovations: ["提出Harness中心化运行时防御形式化框架，将防御设计建模为安全-效用优化问题", "提出HARD自主演化框架，通过失败溯源路由至context/action双层evolver实现定向防御改进", "联合演化context-side policy与action-side gate两类防御工件，在四类攻击与两种自适应攻击下显著优于手工防御"]
benchmarks: ["AgentCanary", "AgentHazard", "PinchBench"]
---

# 论文速读：BEYOND-HANDCRAFTED-SECURITY-TOWARDS-SELF-EVOLVING-DEFENSE-FOR-LLM-AGENTS

## 一句话总结
本文提出 **HARD**（Harness-based Autonomous Runtime Defense Evolution）框架，将LLM Agent的运行时防御从人工设计范式转变为基于执行失败轨迹的自主演化过程，通过在上下文构建与动作解释两个harness接口上迭代改进防御工件，实现自适应抵御动态攻击。

## 研究问题与动机
- **运行时防御的手动瓶颈**：现有运行时防御（runtime defense）严重依赖人工设计的干预策略，部署后保持静态，无法适应持续演化的攻击模式。
- **开放失败空间的挑战**：LLM Agent的失败空间本质上是开放且无法先验穷举的，即便防御覆盖了已知攻击模式，仍可能对新型漏洞失效。
- **自适应攻击的威胁**：动态攻击演化（DAE）和长期渐进攻击（LPA）等自适应攻击会针对已部署的防御持续调整策略，导致防御面临的失败分布随时间偏移。
- **人工诊断不可扩展**：应对新兴失败目前依赖人工迭代诊断与防御重构，成本高昂且难以规模化，亟需一种能从观测失败中自主学习并改进的自主演化范式。

## 核心贡献（创新点）
1. **Harness中心化形式化**：首次建立统一的harness-centric运行时防御形式化框架，将防御设计建模为在可编辑agent harness上的安全-效用优化问题，提供了现有方法的统一视角。
2. **自主演化框架HARD**：提出HARD框架，将执行失败转化为定向防御改进，通过失败溯源路由到对应harness接口（上下文构建/动作解释），由专用evolver自动提取泛化模式并更新防御工件。
3. **分层防御工件协同演化**：将harness分解为context-side security policy与action-side execution gate两类可独立演化的防御工件，并通过trace-driven routing机制实现针对性优化，二者联合演化在静态攻击下取得最优安全-效用权衡。
4. **全面评估与新基准适配**：在AgentCanary基准上覆盖四类安全威胁与两种自适应攻击设置，同时将AgentHazard数据集翻译至AgentCanary格式进行公平比较，验证了HARD超越现有手工防御的鲁棒性。

## 方法详解
**Harness中心化建模**：将部署的LLM Agent形式化为 $A = (M_\theta, H)$，其中 $H = (\phi_H, \psi_H)$ 为运行时harness，$\phi_H$ 负责上下文构建（决定呈现给模型的信息），$\psi_H$ 负责动作解释（决定如何将模型输出转化为可执行操作）。

**防御优化目标**：
$$H^* \in \arg\max_{H \in \mathcal{H}} \mathbb{E}\left[J_{\text{safe}}(\tau) + \lambda_u J_{\text{util}}(\tau)\right]$$
其中 $J_{\text{safe}}$ 衡量轨迹安全性，$J_{\text{util}}$ 衡量任务效用，二者存在内在权衡。

**失败驱动演化流程（Algorithm 1）**：
1. **攻击轨迹收集**：从攻击分布 $\mathcal{A}$ 采样任务，在当前harness $H_t$ 下收集执行轨迹 $\mathcal{T}_t$。
2. **失败识别**：筛选出 $J_{\text{safe}}(\tau_i) < \delta_s \lor J_{\text{util}}(\tau_i) < \delta_u$ 的失败轨迹集合 $\mathcal{F}_t$。
3. **失败路由**：LLM-based trace router $\mathcal{R}$ 分析每条失败轨迹，将其分配至对应的防御工件 $d_k$（policy或gate）。
4. **工件演化**：每个防御工件 $d_k^{t+1} = \mathcal{E}_k(d_k^t, \mathcal{F}_{k,t})$ 由专属evolver基于路由后的失败反馈更新，约束条件为最小化对现有harness的改动 $\Delta(H_t, H')$。

**两类防御工件**：
- **HARD-Policy**：仅演化context-side安全策略，生成高层语义安全不变量（如未信内容处理、权限边界判断）。
- **HARD-Gate**：仅演化action-side执行门控，生成可机器匹配的执行时谓词（如危险工具调用模式拦截）。
- **HARD-Both**：联合演化两者，通过routing机制分流失败轨迹至对应evolver。

**Evolver提示设计要点**：
- Policy evolver强调提取通用安全不变量而非逐场景拒绝规则，要求合并/泛化已有规则，避免overfit到特定benchmark artifact。
- Gate evolver要求定位有害action的dangerous_operand，生成可复用但低误报的执行时谓词，禁止使用benchmark专属路径作为匹配条件。
- Router设置0.55置信度阈值，低于阈值的gate预测回退至policy evolver以降低假阳性。

## 实验与结果
**数据集与环境**：
- 主要基准：AgentCanary（held-out test split，四类攻击：DPI/IPI/MC/SP）
- 补充基准：AgentHazard（翻译为AgentCanary格式）
- 良性效用：PinchBench（25个工具使用任务）
- 自适应攻击：动态攻击演化（DAE）、长期渐进攻击（LPA）

**模型配置**：
- 执行Agent：DeepSeek-V4-Flash
- 安全/效用Judge：GLM-5
- Evolution backbone（主实验）：GLM-5.2
- 消融实验对比：Claude Opus-4.6、Qwen3.7-Max、GPT-5.5

**主要结果（静态攻击）**：
| 攻击类型 | HARD-Both ASR | 最佳手工防御ASR | 提升幅度 |
|---------|--------------|----------------|---------|
| DPI | **12.1%** | Shield 36.1% | ↓24pp |
| IPI | **1.3%** | Shield 19.2% | ↓18pp |
| MC | **6.7%** | Shield 36.5% | ↓30pp |
| SP | **7.4%** | ClawKeeper 54.3% | ↓47pp |

- **良性效用（BU）**：HARD-Both保持91.9%–95.0%，与手工防御相当。
- **攻击下效用（UA）**：MC场景从57.8%提升至85.9%，SP场景从49.3%提升至95.1%。
- **统计显著性**：paired McNemar test在所有四种攻击上 $p < 10^{-8}$。

**自适应攻击结果**：
- DAE：HARD-Both ASR 26.5% vs 最佳手工防御30.1%
- LPA：HARD-Policy ASR 4.8% vs HARD-Both 12.1% vs 最佳手工防御24.1%（策略演化在分散恶意意图场景中优势显著）

**演化回退实验**：
- 各backbone均显著优于无演化（ASR 63.9% → 7.7%~20.3%）
- Claude Opus-4.6获得最低ASR（7.7%），GLM-5.2获得最高UA（85.9%），Qwen3.7-Max获得最高BU（96.4%）
- GPT-5.5 UA仅7.2%，表明演化可能产生过度保守的防御

## 相关工作脉络
1. **运行时防御（Runtime Defenses）**：SecureClaw、ClawKeeper、OpenClaw Shield等手工防御依赖预定义策略且部署后冻结；本文通过harness形式化统一了这些方法的干预位置，并赋予其自主演化能力。
2. **模型级防御（Model-level Defenses）**：StruQ、SecAlign、Agent Safety Alignment等方法需访问模型参数进行fine-tuning或RL，存在安全-效用tradeoff且无法应用于固定/闭源模型；HARD无需修改模型参数即可提升安全。
3. **自演化Agent（Self-Evolving Agents）**：GEPA、ADAS、Darwin Gödel Machine等优化任务性能；FATE扩展至安全但需参数更新；Membrane演化外部对比安全记忆但不改变执行循环；本文首次将自演化应用于运行时harness层面。
4. **直接提示注入防御**：Spotlighting、GuardAgent、Task Shield等方法通过上下文标记、guard model或任务对齐验证进行静态干预；HARD能从失败中学习更通用的安全策略。
5. **间接注入与记忆污染防御**：PoisonedRAG、MINJA、MELON等针对特定攻击模式设计；HARD通过失败溯源可同步改进context与action两层的防御机制。
6. **自适应攻击**：Adaptive attacks（如per-defense tailored attacks、Rainbow Teaming、AutoRedTeamer）使失败分布非平稳；HARD的迭代演化设计正是为应对此类动态威胁而提出。

## 局限性与未来方向
- **演化稳定性未充分验证**：长期演化可能积累过多规则导致过度限制，论文仅展示5轮演化结果，未探讨无限演化的收敛性或退化风险。
- **Gate规则泛化局限**：Case 3显示，在LPA场景下literal gate规则无法匹配重构后的命令，过度依赖gate可能导致policy层失去学习语义原则的机会。
- **单backbone假设**：Router与Evolver共享同一LLM backbone，未探索heterogeneous evolution architecture的可能性。
- **演化预算敏感性**：batch size、轮数、置信度阈值等超参对性能的影响仅部分讨论，缺乏系统性ablation。
- **跨模型部署验证不足**：实验固定使用DeepSeek-V4-Flash作为执行agent，未评估HARD在不同底层agent架构上的泛化性。
- **Future direction**：作者指出可将演化机制扩展至更多harness接口（如persistent artifact管理）、结合online learning实现零延迟防御更新、以及探索multi-agent协同演化防御。

## 研究启发与可借鉴点
1. **Harness分解范式可迁移**：将复杂系统防御分解为"信息输入"与"动作输出"两个正交接口的设计思路，可应用于其他AI系统（如RAG系统、多模态agent）的安全工程。
2. **失败溯源路由机制**：LLM-based router根据失败特征（语义推理需求 vs 可复用执行谓词）自动分流至不同evolver的策略，可为其他自主系统调试/改进提供借鉴。
3. **Evolver提示工程技巧**：Policy evolver强调"泛化安全不变量而非逐场景拒绝"、Gate evolver强调"定位dangerous_operand而非匹配benchmark路径"的设计，展示了如何通过prompt约束防止overfitting。
4. **安全-效用联合优化的形式化**：将运行时防御建模为轨迹级多目标优化问题，并为后续理论分析（如Pareto前沿、 regret bound）提供基础框架。
5. **跨基准数据翻译策略**：将AgentHazard翻译为AgentCanary格式以复用同一评测管道，避免了多评测栈带来的不可比性，值得在基准整合工作中参考。

## 关键术语表
- **Harness**：运行时harness，介导LLM与外部环境及持久化artifact交互的执行框架，包含上下文构建函数$\phi_H$与动作解释函数$\psi_H$。
- **Direct Prompt Injection (DPI)**：攻击者直接控制当前用户任务输入，植入恶意指令诱导越权行为。
- **Indirect Prompt Injection (IPI)**：攻击者控制agent读取的外部内容（网页、邮件、工具返回），利用未信内容与可信任务共享context的漏洞注入恶意指令。
- **Memory Contamination (MC)**：攻击者在先前会话中植入恶意持久化记忆，在后续任务中被检索并触发Unauthorized action。
- **Skill Poisoning (SP)**：攻击者在已安装的skill/plugin/tool artifact中嵌入隐藏指令或trigger-gated backdoor，agent调用时触发恶意行为。
- **Dynamic Attack Evolution (DAE)**：攻击者固定恶意目标，根据agent执行响应与judge反馈迭代优化攻击prompt的黑盒自适应攻击。
- **Long-horizon Progressive Attack (LPA)**：将恶意目标分解为plant-then-trigger序列，通过多轮合理交互累积风险信号最终达成越权效果。
- **Benign Utility (BU) / Utility under Attack (UA)**：BU衡量无攻击时良性任务完成率；UA衡量攻击场景下合法用户目标仍被完成的比率。

## 可复现要素
- **数据集**：AgentCanary（论文引用arXiv:2606.10484）、AgentHazard（arXiv:2604.02947）、PinchBench（https://github.com/pinchbench/skill）；AgentCanary与PinchBench公开，AgentHazard论文未明确声明开源状态。
- **代码/权重**：论文未提及官方代码仓库或模型权重开源声明。
- **关键超参**：安全阈值 $\delta_s = 0.5$；Router置信度回退阈值 $0.55$；Evolution batch size = 8 traces；温度 $temperature = 0$；演化轮数 $T = 5$（主实验）；Train/test split比例约1:1（按seeded hash排序切分）。

---
title: "AQuA-Recursively-Self-Improving-Quantitative-Trading-Researc"
source: https://arxiv.org/pdf/2608.12841v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:46:12"
field: "量化投资与自主研究代理"
keywords: ["recursive self-improvement", "quantitative trading", "alpha mining", "LLM agent", "sealed sandbox", "factor discovery", "time series model"]
innovations: ["通过密封沙盒与受限 DSL 结构性隔离数据泄漏，实现因子发现与模型开发的递归自改进", "proposal-first 可证伪假设框架使 LLM 生成从黑箱输出变为可审计推理链", "config-diff 驱动的可比变体搜索确保搜索过程无泄漏且变体间仅差已改旋钮"]
benchmarks: ["crypto five-minute universe", "US equity 30-minute forward return prediction"]
---

# 论文速读：AQuA: Recursively Self-Improving Quantitative Trading Research Agents

## 一句话总结
论文提出 AQuA，一个包含两个独立递归自改进研究系统的量化投资框架：一个用于符号因子发现（Part I），一个用于可训练模型开发（Part II）。两个系统通过密封沙盒隔离数据路径与评估器，各自在固定评估合同下闭环迭代，验证有效证据并累积研究记忆，从而实现研究过程的自主改进。

## 研究问题与动机
1. **量化研究的可重复性危机**：小方法错误（如前视偏差、测试集选择、单一 regime 过拟合）可导致看似强大但无法复现的回测结果（Bailey et al., 2017; Harvey et al., 2016）。
2. **现有 LLM 代理的局限性**：现有定量代理通常仅聚焦因子发现或模型开发之一，且无约束的代理可能无意中引入数据泄漏，损害后续迭代所依赖的证据链。
3. **递归改进可能放大错误**：若一个 agent 生成的特征或标签包含未来信息，而 review agent 因语义合理性而遗漏，高分数结果会被存为"成功案例"并在迭代中传播，递归改进会放大未检测错误。
4. **prompt 指令与模型审查的不可靠性**：反复访问固定 holdout 会导致自适应过拟合，LLM agent 已被观察到利用目标函数或评估器的错误设定（Atinafu & Cohen, 2026）。

## 核心贡献（创新点）
1. **在因子发现与模型开发两个独立系统中实现递归自改进**：与已有工作仅聚焦单一研究方向不同，AQuA 同时覆盖量化研究的两大环节，且两系统不共享代理、记忆或搜索空间。
2. **构建自主因子发现系统（Part I）**：通过 manager 协调的多代理流水线，提出可证伪的经济假设并转化为公式化因子，结合跨轮记忆与信念更新机制，使后续搜索基于实证证据而非从零开始。
3. **设计混合时序模型与自主开发循环（Part II）**：通过 config-driven loop 在密封沙盒内直接训练可比较的混合时序模型变体，每次配置 diff 仅改变架构/损失/采样器/优化器参数，确保变体间可比且无泄漏。
4. **展示样本外预测与交易性能**：Part I 在加密市场达到组合信号 IC ≈ 0.190；Part II 在美国股票达到单只股票 IC +0.0843（相对最强基线 GRU 提升 37.5%），策略 Sharpe 达 +2.50（2bps 换手成本），且 2021–2025 每年均为正收益。

## 方法详解

### 3.1 密封沙盒与受限迭代
核心原则：**不对称自由（asymmetric freedom）**——agent 可在 DSL 内自由探索，但评估器位于自适应曲面之外。

- **数据路径密封**：在迭代开始前冻结数据划分 $\mathcal{D}$、特征定义 $\mathcal{F}$、标签定义 $\mathcal{L}$ 和评估器 $\mathcal{V}$，构成 $\mathcal{S} = (\mathcal{D}, \mathcal{F}, \mathcal{L}, \mathcal{V})$。
- **受限操作空间**：agent 只能生成 DSL 中的配置 $\theta_k \in \Theta$，该空间不允许修改或绕过密封数据路径。
- **评估公式**：$s_k = \mathcal{V}(C(\theta_k); \mathcal{S})$，其中 $\mathcal{S}$ 是密封的。

**两类泄漏通道及对应闭合策略**：
1. **生成泄漏（generation leakage）**：agent 编写特征/标签/归一化时可能引用预测时不可用的信息 → 通过构造闭合解决，DSL 中无操作可触及密封组件。
2. **选择泄漏（selection leakage）**：搜索过程读取即将被报告的指标 → 通过将优化指标与报告指标分离解决，搜索期间仅返回验证切片得分，最终测试窗口仅在配置冻结后一次性评分。

### 3.2 Part I：自主因子发现
- **提案优先**：每个因子以可证伪假设形式进入流程，包含机制、预测方向、证伪条件（Listing 1）。
- **六代理流水线**：Data Steward → Visual Analyst → Idea Miner → Factor Evaluator → Backtest Engineer → Research Librarian，由 AI Manager 协调。
- **算子注册表（Operator Registry）**：基于 Kakushadze (2016) 公式化 alpha 词汇表，包含三类算子：
  - 截面算子（rank, z-score, sector neutralization）
  - 时间序列算子（lag, difference, rolling correlation, rolling std）
  - 逐元素算术与条件
- **因果闭合**：每个时间序列算子仅读取当前时间戳之前的 trailing window，因此任意组合均因果闭合。

### 3.3 Part II：自主模型开发
- **配置驱动循环**：每次迭代生成一个 config diff（Listing 2），包含 split、sampler、arch、loss、optim 等组件。
- **混合时序架构**：
  - 前端：多尺度 1D 卷积提取局部表示
  - 时序 backbone：可选 LSTM / Mamba / Attention
  - 横截面混合：cross-entity mixer
  - 门控融合 + pooled readout
  - MLP / SwiGLU 预测头
- **可比较性保证**：一个 config diff 对应一个 variant，数据路径固定，两 variant 仅在不同旋钮上差异。

### 3.4 递归自改进形式化
对于 part $p \in \{\mathrm{I}, \mathrm{II}\}$：
$$
R_{t+1}^{(p)} = \mathcal{U}_p\left(R_t^{(p)}, H_t^{(p)}, C_t^{(p)}, E_t^{(p)}\right), \quad \left(H_{t+1}^{(p)}, C_{t+1}^{(p)}\right) \sim \mathcal{P}_p\left(\cdot \mid R_{t+1}^{(p)}\right)
$$
其中 $R$ 为持久研究状态，$\mathcal{U}_p$ 为更新函数，$\mathcal{P}_p$ 为 proposal 分布。两部分独立运行，互不更新对方的 $R$。

### 3.5 三部分反馈回路（Part I）
1. **方向校准回路**：单次 backtest 内，测试并修正因子预测方向。
2. **证伪驱动信念更新回路**：单次 run 内，失败提案更新信念状态。
3. **跨轮记忆与政策回路**：AI Manager 读取累积记忆，引导下次搜索方向。

## 实验与结果

### 数据集与评估设置
- **Part I**：Crypto 五分钟高频数据（crypto-five-minute universe）
- **Part II**：美国股票市场，30 分钟前向回报预测
  - 训练：2010–2019
  - 隔离期：2020（不参与训练或选择）
  - 测试：2021–2025（完全 untouched）
  - 模型选择由训练窗口末尾的内层验证切片驱动

### Part I 结果
- 组合信号 IC 随迭代累积提升至约 **0.190**
- 单因子 IC 范围约 0.026–0.037
- 强度来源于机制 grounded factor 的组合而非单个表达式

### Part II 结果
| 模型 | 类别 | raw IC |
|------|------|--------|
| Linear (ridge) | linear | +0.0251 |
| LGB | gradient boosting | +0.0397 |
| xLSTM | recurrent | +0.0434 |
| LSTM | recurrent | +0.0535 |
| GRU | recurrent | +0.0613 |
| **Ours (hybrid)** | **hybrid** | **+0.0843** |

- 单只股票 IC：**+0.0843**（vs GRU +0.0613，绝对提升 +0.0230，相对提升 **37.5%**）
- $R^2$ = mean(IC²)：**1.20%**
- 阈值多空策略 Sharpe：
  - 基础：**+2.15**（sector-neutral）
  - + causal volatility targeting：**+2.50**
  - 全因果 walk-forward（无后见之明）：**+2.00**
- 年度 Sharpe：2021(+1.7), 2022(+3.5), 2023(+1.9), 2024(+1.8), 2025(+2.7)，每年均为正

## 相关工作脉络

1. **LLM-driven alpha mining**：从遗传编程（AlphaEvolve, Cui et al. 2021）、强化学习（AlphaMemo, Yu et al. 2023）到近期 LLM 代理（QuantaAlpha, Han et al. 2026; AlphaAgent, Tang et al. 2025）。AQuA Part I 与这些工作共享 proposal-first、memory-driven 设计，但额外耦合独立模型开发系统。
2. **Autonomous research agents**：Lu et al. (2024)、Romera-Paredes et al. (2024) 等端到端科学代理；金融领域 Miyazaki et al. (2026)、Singhi (2025)。AQuA 的独特之处在于将递归自改进分别实例化于因子发现和模型开发，并通过密封沙盒结构性地处理泄漏而非依赖 agent 判断。
3. **Deep models for financial time series**：Gu et al. (2020)、Zhang et al. (2019) 等；近期混合 RNN/CNN/Transformer 架构（Kabir et al. 2025; Song et al. 2025）。AQuA 贡献非新原语，而是自主循环，使 agent 能组合这些原语并跨 variant 累积证据。
4. **Sealed evaluation / reproducibility**：Bailey et al. (2014, 2017) 证明回测过拟合概率；Harvey et al. (2016) 强调 skepticism。AQuA 通过物理隔离（非仅指令约束）实现结构正确性。
5. **Operator-based factor search**：Zhang et al. (2020)、Cui et al. (2021) 的遗传编程；Zhao et al. (2025) 的 REINFORCE。AQuA 引入 LLM 提案机制与可证伪假设框架。

## 局限性与未来方向

1. **单一市场与频率**：Part I 仅在 crypto 五分钟频率验证，Part II 仅在美国股票 30 分钟频率验证，未在其他市场或频率上验证可迁移性。
2. **半自主性**：循环仍依赖人类操作员设置研究目标、拥有沙盒并监督晋升，非完全无人值守。
3. **模拟验证**：报告指标在换手成本模型下模拟，尚未在实际交易中验证。
4. **泄漏保障不对称**：生成泄漏通过构造闭合保证；测试窗口隔离依赖密封协议与 operator discipline，非密码学保障，理论上操作员可直接访问存储并绕过隔离。
5. **未来方向**：耦合两系统——将 Part I 发现的因子作为 Part II 模型的输入，但需将已发现因子集视为 Part II 沙盒的冻结组件，避免共享信息导致隐性过拟合。

## 研究启发与可借鉴点

1. **密封沙盒设计可迁移至其他自主研究 Agent**：AQuA 将泄漏区分为"生成泄漏"与"选择泄漏"两类，并分别通过构造闭合与指标分离解决。任何 scored-by-evaluator 的自主研究 agent（不限于金融）均可借鉴此范式。
2. **proposal-first 可证伪假设框架**：Part I 要求每个因子提案以假设、机制、方向、证伪条件形式出现，而非直接输出表达式。这可将 LLM 的"黑箱生成"转化为可审计的推理链，值得在材料科学、药物发现等其他 autoML/autonomous research 领域尝试。
3. **config-diff 驱动的可比变体搜索**：Part II 通过单 config diff 对应单 variant 的设计，确保搜索过程中变体间仅差已改旋钮。这种设计可直接迁移至神经网络架构搜索（NAS）或超参优化场景，提升搜索效率与可解释性。
4. **跨轮记忆与信念更新机制**：Part I 的 Research Librarian 结构化记录每次迭代的 goal、观察、机制、信念，形成"经验积累"。这种 persistent research state 的设计可推广至任何需要长程学习的自主代理系统。
5. **方向校准回路**：Part I 在 backtest 阶段测试并修正因子预测方向，这一机制可应用于任何符号表达式搜索任务，避免因初始假设符号错误而误杀有效因子。

## 关键术语表

**Information Coefficient (IC)**：预测值与实际值的 Spearman/Pearson 相关系数，衡量因子或模型的预测能力。

**Formulaic Alpha**：基于可微分算子（如 rank、lag、rolling correlation）组合的符号化因子表达式。

**Sealed Sandbox**：在迭代开始前冻结数据划分、特征定义、标签定义和评估器的隔离环境，agent 无法修改这些组件。

**Causality Closure**：DSL 中每个时间序列算子仅读取当前时间戳之前的 trailing window，使得任意组合表达式均因果闭合。

**Generation Leakage**：agent 生成的特征/标签/归一化引用了预测时不可用的信息。

**Selection Leakage**：搜索过程读取即将被报告的评估指标，从而学会针对该指标过拟合。

**Config Diff**：Part II 中描述的单一配置变更（architecture/loss/sampler/optimizer 等），对应一个可比较的模型变体。

**Recursive Self-Improvement**：在研究过程层面， validated evidence 被累积到 persistent research state 中，指导后续 hypothesis 生成，但不改变底层模型或评估器本身。

## 可复现要素

- **数据集**：Part I 为 crypto 五分钟数据；Part II 为美国股票市场（2010–2025），论文未明确声明公开
- **代码/权重**：论文未提及开源
- **关键超参**：
  - Part II 训练：lr=3.0e-4，cosine schedule，bf16 precision，DDP=8
  - 历史长度：64
  - Batch size：8192
  - 换手成本：2 bps
  - 损失函数：[spearman_ic, huber_csz, turnover_reg]

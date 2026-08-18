---
title: "E<sub>va</sub>l<sub>ua</sub>tin<sub>g</sub> R<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l C<sub>o</sub>ntr<sub>ac</sub>tin<sub>g</sub> in N<sub>a</sub>t<sub>u</sub>r<sub>a</sub>l L<sub>a</sub>n<sub>guage</sub>"
source: https://arxiv.org/pdf/2608.10475v1.pdf
model: agnes-2.5-flash
chunks: 7
summarized_at: "2026-08-18 06:35:59"
---

# 论文速读：E<sub>va</sub>l<sub>ua</sub>tin<sub>g</sub> R<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l C<sub>o</sub>ntr<sub>ac</sub>tin<sub>g</sub> in N<sub>at</sub>u<sub>r</sub>a<sub>l</sub> L<sub>an</sub>gu<sub>ag</sub>e

## 一句话总结
本文提出**Rational Contracting**框架与 **ContractSim** 基准，将LLM合同谈判形式化为两方“谈判‑执行”两阶段博弈，在随机多步环境中系统评估LLM能否作为理性且合作的缔约伙伴；实验表明LLM虽具备鲁棒反剥削能力，但在高随机环境下难以达成帕累托最优合同，且普遍依赖高频违约（defection）而非合规来追求收益，距离理性条件合规（RCC）基线仍有显著差距。

## 研究问题与动机
- **核心问题**：LLM驱动的语言Agent能否作为**理性且合作**的合同谈判与执行伙伴？现有评估多停留在一次性交换或简单重复博弈，无法刻画可信合同所需的质量维度。
- **现有方法不足**：
  1. 缺乏对合同**可行性、互利性、效率与违约/剥削行为**的系统量化指标；
  2. 现有谈判基准（如NegotiationArena等）多忽略多步执行阶段的随机性与环境不确定性；
  3. 缺少统一的**理性合作基线**，难以区分LLM的“合作表象”与真正的“条件合规能力”；
  4. 自然语言协议到结构化可执行约束的映射缺乏确定性管道，导致评估结果不可复现。

## 核心贡献（创新点）
1. **提出理性合同博弈框架**：将合同形式化为两方谈判‑执行博弈 $\mathcal{G}$，在随机多步环境中同步评估Agent的理性与合作性，填补了两阶段契约评估的空白。
2. **构建ContractSim基准与度量体系**：推出6种环境×3个领域（catering、hotel cleaning、AI hosting）的多轮供应商合同平台，并定义可行性、互利性、$P_{sat}$、违约率等量化指标。
3. **设计RCC理性条件合规基线**：引入“遵守合同条款，遇违约切换至非参与”的基准策略，为衡量Agent的真实合作质量提供可计算的上界参考。
4. **搭建离线分离的合同翻译与合成流水线**：开发严格JSON schema的结构化翻译器，并配套坐标上升优化的合成合同生成模块，支持跨领域自然语言渲染与可审计执行。

## 方法详解
### 1. 博弈形式化
- 博弈 $\mathcal{G} = (\mathcal{T}, \Theta, \tilde{\Omega}, \mathcal{N}, \mathcal{P})$，其中 $\mathcal{T}=\{1,2\}$（客户/供应商），$\Theta$ 为私钥类型分布，$\tilde{\Omega}$ 为自然语言合同空间，$\mathcal{N}$ 为谈判协议，$\mathcal{P}$ 为执行阶段动力学。
- 谈判阶段最多 **50 轮**，客户先行动；每轮行动包括发消息、提议契约、正式接受或终止谈判。契约必须显式规定**单价、每轮交付数量、每周付款金额**。
- 执行阶段**无沟通、无重新谈判**：客户按周付款，供应商购买原料并生产；目标函数为客户最大化期望私人效用（剩余预算+累计私人价值），供应商最大化期望利润（收入‑原料成本）。

### 2. ContractSim 环境规范
- **类型剖面固定**：分配背景前预先生成 $\theta' = (\theta_{\text{Cust}}, \theta_{\text{Supp}})$。
- **抽象生产图**：输入集 $\mathcal{X}=\{I_1,I_2,I_3\}$，产出集 $D=\{O_1,O_2,O_3\}$，生产函数 $O_1=I_1+I_2+I_3$，$O_2=I_1+I_2$，$O_3=I_3$。
- **时间结构**：执行阶段 $\bar{L}=11$ 周，付款周 $W_{\text{pay}}$ 共6周（奇数周），生产周 $W_{\text{prod}}$ 共5周（偶数周）。
- **随机性**：每生产周 $w$ 与输入 $x$ 独立抽取价格 $p_{w,x}\sim\mathsf{P}_x$、收货量 $r_{w,x}|o_{w,x}\sim\mathsf{R}_x$、损耗冲击 $s_{w,x}\sim\mathsf{S}_x$。主要性能实验环境种子为 **42**。
- **领域映射**：Catering（1x/$1粒度）、Hotel Cleaning（100x/$100粒度）、AI Hosting（1,000x/$1,000粒度），底层生产图与约束完全共享。
- **关键约束**：库存上限10 units/原料；损耗均匀分布在0/1/2 units；交付损失使实际收货为 $[\lfloor x/2\rfloor, x]$ 均匀整数；超过上限部分丢失但仍计费；BATNA 为从外部提供商以等于累计私人价值的成本获取最低要求产出。

### 3. 理性基线设计
- **RC（Refuse/Non‑participation）**：直接拒绝参与，效用等于BATNA。
- **RE（Exploitative）**：最大化自身效用，允许剥削对手。
- **RCC（Rational Conditional Compliance）**：严格遵循合同条款，但若观测到对手违约则永久切换至非参与状态（grim‑trigger 式的理性防御），确保序列理性与子博弈完美。

### 4. 合同翻译与合成生成
- **离线翻译器**：使用 **Gemini 3.6 Flash**，将自然语言协议映射为结构化约束 $C^\omega = (\mathbf{p}^i, \mathbf{q}, \bar{\mathbf{M}}, \kappa)$。输出严格遵循 Table 6 JSON schema（含 `dish_prices`、`production_schedule`、`payment_schedule`、`contingency_set`、`contingency_params`），缺失字段保留 null，绝不填充模糊核心项。
- **校验与缓存**：顶层 contracts 列表标识必须与请求批完全匹配；畸形输出最多3次纠正；任意标识缺失/重复则整体拒绝；缓存依赖 schema 版本、记录ID、原始文本 SHA‑256 等全量匹配。
- **合成合同生成**：首选取含 grim‑trigger 但无 elective contingency 的基础合同；对目标 $\rho\in\{0.9,0.8,0.7,0.6,0.5\}$ 求解：
  $$\max_{\omega} U_{\text{Cust}}(\omega) + U_{\text{Supp}}(\omega) \quad \text{s.t.} \quad U_i(\omega) \geq u_i(\perp), \; |P_{\text{sat}} - \rho| \leq 0.05$$
  采用多起点坐标上升搜索三变量产量、三单价、六名义付款；附加8种 substitution/payment_deduction/rollover 组合，最终保留互益且符合预算的 **180 份合成合同**用于跨6个评估环境的执行测试。

## 实验与结果
### 实验设置
- **框架**：Concordia（Vezhnevets et al. 2023）
- **环境**：6 个 Catering 环境（Env 5/6 为高随机性，其余4个为低随机性）
- **LLM 配对**：Claude Opus 5、Gemini 3.6 Flash、GPT‑5.6‑Sol（共 9 对）
- **基线**：RC、RE、RCC
- **规模**：162 场谈判（98.1%=159 场达成协议）；执行测试 180 份合成合同×5 个 $P_{sat}$ 水平×6 个 contingency 变体 = **4,860 场博弈**

### 关键结果
- **谈判效率**：高随机性 Env 5 中仅 **3 对**（<33%）达成互利；Env 6 仅 **7 对**；其他4个易环境全部配对互利。LLM 未谈判出含 contingency 条款的合同，而此类合同在相同 $P_{sat}$ 下可实现更高效用（$F_2$ vs $F_1$）。
- **整体合同质量**：$P_{sat}$ 均值 **90.0%**；符合理性基线 95% 满意度阈值的合同占

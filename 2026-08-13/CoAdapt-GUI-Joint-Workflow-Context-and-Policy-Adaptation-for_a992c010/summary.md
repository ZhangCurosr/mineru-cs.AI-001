---
title: "CoAdapt-GUI-Joint-Workflow-Context-and-Policy-Adaptation-for"
source: https://arxiv.org/pdf/2608.11588v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 09:53:51"
field: "GUI Agent 跨应用泛化"
keywords: ["GUI Agent", "Test-Time Adaptation", "Workflow Context", "Policy Adaptation", "Mobile Automation", "Cross-App Generalization"]
innovations: ["提出CoAdapt-GUI联合TTA框架，同步更新structured workflow context与LoRA policy adapter", "设计transfer-constrained workflow表示与population-based evolution机制，避免negative transfer", "构建AndroidWorld Plus数据集，严格区分Category-Shared与Category-Novel支持细粒度消融"]
benchmarks: ["AndroidWorld-Generalization", "AndroidWorld Plus"]
---

# 论文速读：CoAdapt-GUI: Joint Workflow-Context-and-Policy-Adaptation for Mobile GUI Agents

## 一句话总结
论文提出 **CoAdapt-GUI**，一个面向移动 GUI 代理的测试时自适应（TTA）框架，在有限交互预算与无目标演示条件下，联合更新 **结构化的可迁移工作流上下文（transferable workflow context）** 与 **策略参数（LoRA adapter）**，在 AndroidWorld-Generalization 与自建的 AndroidWorld Plus 两个未见应用（unseen-app）设置下均取得最优性能。

## 研究问题与动机
- **核心问题**：现有移动 GUI 代理在部署到训练集内未出现的应用（novel-app）时表现脆弱，缺乏有效的跨应用泛化能力。
- **现有方法不足**：
  1. **Policy-Only TTA**（如 AndroidWorld-Generalization）仅更新策略，无法覆盖流程性 gap（例如目标应用的操作顺序与源应用不同）。
  2. **Source-side 静态知识固化**（如 UI-Mem）将经验在源端预训练阶段固化，无法在目标端根据真实 rollout 反馈动态修正。
  3. **External memory / working notes**（如 AppAgent、MobiMem）通常保持底层 policy fixed，仅提供显式记忆辅助，缺少与策略的联合自适应机制。
  4. **Few-shot 方法**（如 LearnAct、AdaptAgent）依赖目标端演示数据，在零演示（zero-demonstration）场景下失效。
- **关键洞察**：test-time behavior 依赖两个互补的 adaptive states——显式的 workflow state $M_t$ 与参数化的 policy state $\theta_t$，两者均需从 agent 自身 target-app rollouts 与 task-level rewards 中联合更新，单一通道无法同时覆盖流程性 gap 与界面特异 grounding gap。

## 核心贡献（创新点）
1. **提出 CoAdapt-GUI 联合 TTA 框架**：在测试时同步维护并更新 structured workflow context 与 frozen VLM backbone 上的 LoRA adapter，无需目标演示或 held-out 评估信号即可自适应 unseen-app。
   - *区别*：与 E-SPL 等高层 idea 相似的工作相比，本文以 FSM-grounded 结构化表示替代自由文本 prompt，并通过 eligibility predictor + schema validator + linter 三重约束确保跨应用可迁移性。
2. **设计 transfer-constrained workflow context 表示**：每条条目 $w = \langle c, P, F, V \rangle$ 显式分离 app-bound 状态（不可迁移）与 transferable 抽象流程，通过 eligibility predictor 自动过滤 application-specific 元素（包名、坐标、widget label 等）。
   - *区别*：区别于 UI-Mem 的源端固化 memory，本文的源工作流库（$\mathcal{L}_{\text{src}}$）仅在目标适配期提供初始化检索基础，target-grounded state $M_t^{\text{tgt}}$ 可独立从零 evolve。
3. **实现 reward-guided population-based context evolution**：基于 TrueSkill 评分维护 context variant 群体，通过 frozen reflector（Claude Opus 4.7）对比成功/失败 traces 提出 type-diff revision，经 schema/provenance/lint 检查后接入群体，适应末期选取最高分 validated 状态。
   - *区别*：不同于 Agent-SAMA 将 FSM 仅用于 planning/recovery 阶段，本文的 FSM-grounded context 直接参与 test-time 在线更新，并与 policy 更新形成 interaction-coupled 反馈闭环。
4. **构建 AndroidWorld Plus 数据集**：在原始 AndroidWorld（19 app）基础上扩展至 25 app 覆盖 12 个 Google Play Store 类别，严格区分 Category-Shared（6 类）与 Category-Novel（7 类），adaptation 与 evaluation task types 完全 disjoint，杜绝数据泄漏。
   - *区别*：AndroidWorld-Generalization 仅含 5 个新 app，本文 Plus 版提供更全面的类别覆盖与更严格的 held-out 划分，支持细粒度消融分析。

## 方法详解
### 整体架构
CoAdapt-GUI 为 online RL-based TTA 框架，在每轮 adaptation round 中从目标应用 rollout 收集轨迹，同时驱动两条互补更新通道：

**1. Transfer-constrained Workflow Context 通道**
- **源侧构建**：对每个源应用 $\mathcal{L}_{\text{src}}$ 构建 FSM-grounded workflow context，按 Google Play Store 类别分组；每条 entry $w = \langle c, P, F, V \rangle$：
  - $c$（applicability）：指定适用范围（category-level 或 app-level）
  - $P$（procedure）：描述抽象操作流程
  - $F$（failure/recovery）：记录失败模式与恢复条件
  - $V$（verification）：指定完成检查条件
- **Eligibility filtering**：通过 $\text{Eligible}_{\text{tr}}(w) \in \{0,1\}$ + schema validator + linter 三重约束，强制排除 app names、package/resource identifiers、widget labels、coordinates、task-instance values 等 app-bound 内容。
- **Target-side 初始化**：Category-Shared app（源池有同类代表）从检索到的 source workflow 初始化 root context；Category-Novel app（源池无同类）root 初始化为空。
- **Population-based evolution**：controller 维护 TrueSkill-rated context variant 群体（初始 mean=25.0, std=8.33）；每次 rollout 后 frozen reflector（Claude Opus 4.7）分析成功/失败 traces，提出 typed workflow revision（add/modify/remove）至高分 parent；child 经 schema/provenance/lint 检查后加入群体（不继承 parent reward）；适应末期选取 highest-rated validated workflow state。

**2. Task–Context-Matched Policy 通道**
- **模型配置**：基于 Qwen3-VL-8B-Instruct（bfloat16），冻结 VLM backbone，仅更新 rank-16 LoRA adapter（目标模块 $q_{\text{proj}}$、$v_{\text{proj}}$）。
- **Group-relative optimization**：按 $(q, \kappa)$ 任务-上下文对分组，组内计算 normalized advantage $A_j = (r_j - \bar{r}_G) / s_G$；优化目标：
  $$\mathcal{L}_{\text{policy}} = -\frac{1}{|\mathcal{B}_{\text{act}}|} \sum_{j \in \mathcal{B}_{\text{act}}} A_j \, \ell_j(\theta) + \beta \, \mathcal{R}(\theta; \pi_{\text{anchor}})$$
  其中 $\beta=0.05$ 为 frozen-policy anchor 系数，$\mathcal{R}$ 为 KL regularization 项；AdamW 优化器 LR=$3\times10^{-4}$，梯度裁剪至 1.0。
- **Buffer 管理**：Policy buffer 仅存最近一次 LoRA 更新后收集的 trajectories；每次 adaptation round 至多尝试一次 LoRA update，update 后清空 buffer。

**3. 耦合机制（Interaction-Coupled，非 Jointly Differentiable）**
- 当前 workflow state $M_t$ 塑造用于 policy learning 的 trajectories（context-conditioned rollout）；
- 当前 policy $\theta_t$ 生成用于 future context 修订的成功/失败 traces；
- 两种更新从相同 rollout groups 导出，通过 interaction 而非联合梯度实现耦合。

### 关键实现参数
| 组件 | 关键参数 |
|------|----------|
| Policy Backbone | Qwen3-VL-8B-Instruct / bfloat16 |
| Max image pixels / gen tokens | 1,605,632 / 512 |
| LoRA rank / scale / dropout | 16 / 32 / 0.0 |
| Optimizer / LR | AdamW / $3\times10^{-4}$ |
| Max gradient norm | 1.0 |
| Anchor coeff / log-ratio clip | 0.05 / 10.0 |
| TrueSkill init mean/std | 25.0 / 8.33 |
| Performance / dynamics params | 4.17 / 0.083 |
| Population window / child uncertainty | 15 / 1.5 |
| Selection optimism / softmax T | 1.0 / 1.0 |
| Frozen reflector / synthesizer | Claude Opus 4.7 |
| Adaptation rounds per app | 20 |
| Hardware per app | ~9–10 GPU-hours (H200 141GB) |

## 实验与结果
### 数据集与协议
- **AndroidWorld-Generalization**（Gu et al. 2026）：Source split 12 app / 62 templates / 905 训练实例；Target split 5 个新 app（Audio Recorder、Clock、OsmAnd、Tasks、Broccoli），40 adaptation 实例 + 48 held-out evaluation 实例；seed 完全隔离。
- **AndroidWorld Plus**（本文自定义构造）：25 app 覆盖 12 个 Google Play Store 类别；Source pool 12 app / 96 templates / 480 episodes；Target pool 13 app / 95 templates，分 Category-Shared（6 app）与 Category-Novel（7 app）；Adaptation 60 templates × 5 seeds = 300 实例；Evaluation 35 templates × 3 seeds = 105 实例；adaptation 与 evaluation task types 严格 disjoint。

### 主要结果
| 设置 | 方法 | 性能 |
|------|------|------|
| AndroidWorld-Generalization | Policy-Only TTA baseline | 37.5% |
| AndroidWorld-Generalization | **CoAdapt-GUI** | **45.0%**（+7.5 pp） |
| AndroidWorld Plus（Overall） | Base Policy | 38.6% |
| AndroidWorld Plus（Overall） | Context-Only TTA | 48.1% |
| AndroidWorld Plus（Overall） | **CoAdapt-GUI** | **52.9%** |
| AndroidWorld Plus（Category-Shared） | Context-Only TTA | +7.4 pp vs Static Transfer |
| AndroidWorld Plus（Category-Novel） | Context-Only TTA | +2.0 pp（从零起步） |
| AndroidWorld Plus（Category-Novel） | CoAdapt-GUI vs Context-Only | +2.9 pp |

**关键结论**：
- Source-side online RL 对 unseen instances 提升 26.1 pp，但对 unseen applications 仅提升 8.3 pp，印证「同界面新任务成功 ≠ 陌生应用有效行为」。
- Static Context Transfer 仅对 Category-Shared 有效（+9.3 pp），对 Category-Novel 无效（+0.0 pp），验证类别限制检索规则的正确性。
- Policy-Only TTA 整体 +1.4 pp，但对 Category-Novel 下降 3.9 pp，说明仅靠策略自适应对 novel 类别有 negative transfer 副作用。
- Target-side 上下文构造在两个类别上均有增益，两通道角色互补。

## 相关工作脉络
1. **AndroidWorld-Generalization（Gu et al. 2026）**：Policy-Only TTA，workflow context 固定；CoAdapt-GUI 额外支持 context 在线更新，弥补流程性 gap。
2. **UI-Mem（Xiao et al. 2026）**：在 source-side training 阶段联合学习 experience memory 与 policy，实现 zero-shot transfer；区别在于知识在源端固化，无法在目标端根据真实反馈动态修正。
3. **AppAgent / Agent Workflow Memory / MobileAgent-E / MobiMem**：外部 memory 记录 explored functionality / reusable routines / evolving notes；通常保持 underlying policy fixed，不提供 joint context-policy TTA。
4. **Agent-SAMA（Guo et al. 2026）**：将 app execution 表示为 FSM 用于 planning/recovery；CoAdapt-GUI 用 FSM-grounded context 约束可 transfer 的知识范围，并支持 test-time evolution。
5. **E-SPL（Zhang, Chen, and Stadie 2026）**：联合优化 global free-text prompt 与 policy；与 CoAdapt-GUI 仅有高层 idea 重叠，实现上以结构化表示替代自由文本。
6. **LearnAct / AdaptAgent**：few-shot GUI 方法，使用 target demonstrations；依赖演示数据，CoAdapt-GUI 仅需 agent 自身 reward-bearing interactions。
7. **MobileRL（Xu et al. 2025）**：在线 RL for Mobile GUI Agents；CoAdapt-GUI 在 MobileRL 思想基础上引入 context-policy 双通道联合适应。

## 局限性与未来方向
- **计算开销**：每 app 需 9–10 GPU-hours（H200 141GB），population-based context evolution 与多次 LLM call（reflector、synthesizer）导致部署成本较高。
- **仅覆盖 12 个 Google Play Store 类别**：AndroidWorld Plus 虽扩展至 25 app，但仍局限于常见类别，对垂直领域应用（如金融交易、医疗）泛化性待验证。
- **Reflector 依赖 Claude Opus 4.7**：frozen large-reflector 的质量直接影响 context revision 准确性，低成本替代方案未探索。
- **Interaction-coupled 而非 jointly differentiable**：两种更新的耦合通过 rollout 序列间接实现，缺乏联合梯度优化理论保障。
- **未探索多任务协同适应**：当前框架针对单任务 per-app，并行处理多任务目标时的上下文冲突与资源分配问题未讨论。

## 研究启发与可借鉴点
1. **Transfer-constrained 表示设计**：$w = \langle c, P, F, V \rangle$ 的四层结构化表示与 eligibility filtering 机制，可迁移至其他 domain adaptation 场景（如 Web GUI、桌面应用），通过显式分离 app-bound vs transferable 组件避免 negative transfer。
2. **Population-based context evolution**：TrueSkill 评分 + schema validator + linter 的三重验证 pipeline，为 structured knowledge evolution 提供了可复用的框架；可考虑扩展至代码生成、API 调用等需要强结构约束的任务。
3. **Interaction-coupled 双通道架构**：policy 与 context 通过 rollout 序列间接耦合的设计，避免了 jointly differentiable 带来的优化不稳定性，为多模块 TTA 系统提供了工程实践参考。
4. **Category-Shared vs Category-Novel 细粒度评估**：AndroidWorld Plus 的类别划分与 disjoint task-type 划分策略，为 GUI agent 泛化能力评估提供了更严谨的 benchmark 设计范式。
5. **Frozen reflector + typed diff 机制**：使用冻结大模型作为 reflector 提出 type-constrained revision（add/modify/remove），既保证输出规范性又降低 compute cost，值得在其他 knowledge distillation 场景中借鉴。

## 关键术语表
- **CoAdapt-GUI**：一种面向移动 GUI 代理的测试时自适应（TTA）框架，联合更新 structured workflow context 与 LoRA policy adapter。
- **Transferable workflow context**：以 $w = \langle c, P, F, V \rangle$ 结构表示的跨应用可迁移操作知识，显式排除 app-bound 元素。
- **Eligibility predictor**：$\text{Eligible}_{\text{tr}}(w) \in \{0,1\}$，结合 schema validator 与 linter 过滤非可迁移内容的机制。
- **Population-based context evolution**：基于 TrueSkill 评分维护 context variant 群体，通过 frozen reflector 对比成功/失败轨迹并提出 type-diff revision 的在线更新策略。
- **Task–context-matched group-relative optimization**：按 $(q, \kappa)$ 分组的 LoRA 更新机制，组内 normalized advantage $A_j = (r_j - \bar{r}_G) / s_G$ 驱动 policy 适应。
- **Interaction-coupled vs jointly differentiable**：两种更新通道通过 rollout 序列间接耦合而非联合梯度优化，避免优化不稳定性。
- **Category-Shared / Category-Novel**：AndroidWorld Plus 中 target app 的两种类型，前者在 source pool 有同类代表，后者从零开始 grounding。
- **Frozen reflector**：固定参数的 Claude Opus 4.7，用于分析 target rollouts 并提出 typed workflow revision。

## 可复现要素
- **数据集**：AndroidWorld（公开）、AndroidWorld-Generalization split（Gu et al. 2026 公开）、AndroidWorld Plus（本文构造，数据与方法描述详细，可复现）。
- **代码/权重**：论文未明确声明代码开源状态；Qwen3-VL-8B-Instruct（step-500 checkpoint from UI-TARS-7B）公开；Claude Opus 4.7 为 API 服务。
- **关键超参**：LoRA rank=16, scale=32, LR=$3\times10^{-4}$, $\beta=0.05$（anchor coeff）, TrueSkill mean=25.0/std=8.33, population window=15, adaptation rounds=20/app, max rollout calls=160/app。
- **硬件**：NVIDIA H200 141GB，每 app 9–10 GPU-hours。

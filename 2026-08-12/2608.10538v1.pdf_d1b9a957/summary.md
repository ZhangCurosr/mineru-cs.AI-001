---
title: "SKILLER: Language-Level Reinforcement Learning for Reusable Skill Extraction in Small Language Models"
source: https://arxiv.org/pdf/2608.10538v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:37:39"
field: "小模型 Agent 部署与技能抽取"
keywords: ["Agent Skill", "Small Language Models", "Reinforcement Learning", "Policy Optimization", "Skill Generation", "Natural Language RL"]
innovations: ["将文本型 skill 作为可优化策略变量，通过强模型 actor/critic 与弱模型执行环境构建纯自然语言 RL 闭环", "结构化 state 四元组 (task, trajectory, reference, diagnostics) 实现因果错误定位", "支持 helper script 合成的可执行抽象卸载机制"]
benchmarks: ["SkillsBench", "SkillLearnBench", "SWE-Skills-Bench", "GAIA", "EarthBench"]
---

# 论文速读：SKILLER: Language-Level Reinforcement Learning for Reusable Skill Extraction in Small Language Models

## 一句话总结
SKILLER 提出了一种自然语言驱动的分步强化学习框架，通过将小型 LLM 的 agent 循环作为环境、前沿模型作为 actor 和 critic，以结构化自然语言迭代优化文本型 skill 策略，无需更新任何神经权重即可为小模型自动生成可复用、执行器特定的 skill。

## 研究问题与动机
1. **强闭源模型成本高**：当前主流 agent harness（如 Codex、OpenClaw）依赖前沿闭源模型生成 skill，推理成本 prohibitive，难以规模化部署。
2. **小模型 skill 迁移失效**：为强模型设计的 skill 无法直接迁移到小规模 LVLM；行为约束、隐式推理步骤和错误恢复假设对小模型往往无效，轻则效果不佳、重则导致任务失败。
3. **自动化生成适配性不足**：现有 skill 生成/进化方法主要面向大规模前沿模型，缺少针对小模型独特行为空间和执行范式的专门设计。
4. **开源小模型潜力未被释放**：Qwen3.5、Gemma 4 等可在消费级 GPU 上部署的开源紧凑模型展现出快速能力提升，若辅以精准 skill 约束，可大幅降低 agent 部署成本。

## 核心贡献（创新点）
1. **语言级策略迭代框架**：将文本型 skill 本身作为可优化策略，通过前沿模型作为 actor/critic、目标小模型 agent 循环作为环境，实现纯自然语言反馈的强化学习闭环；与已有工作本质区别在于完全绕过梯度更新，以语言为媒介桥接强模型推理与弱模型执行。
2. **结构化状态四元组设计**：$s_i = (x, \tau_i, \tau^\star, v_i)$，同时包含任务实例、当前轨迹、成功参考轨迹和验证器诊断，使 critic 能精确定位首次因果分歧而非仅感知二元成败；已有方法多仅依赖最终 reward 或局部诊断。
3. **可执行抽象（Helper Scripts）**：actor 支持合成 task-local helper program，将复杂过程性推理从自然语言 prompt 卸载到确定性外部工具，避免小模型因长文本上下文产生的认知过载；现有方法主要依赖纯文本策略修改。
4. **渐进式 skill 披露机制**：环境在 RL 循环中以 progressive disclosure 方式逐步揭示 skill，防止紧凑模型被冗长文本淹没；同时支持 batch 并行优化与快照回退，保证已生效行为不被后续更新覆盖。

## 方法详解
**形式化定义**：给定冻结紧凑模型 $\pi$、任务实例 $x$ 和技能 $\mathcal{K}_i$，执行轨迹 $\tau_i = (o_{i,0}, a_{i,0}, ..., o_{i,T})$，环境转移函数 $(\tau_i, r_i, v_i) = \mathcal{E}(x, \mathcal{K}_i; \pi)$，目标为最大化验证器性能 $J_x(K) = \mathbb{E}[r]$。

**核心组件**：

1. **环境 $\mathcal{E}$**：封装 benchmark 专属 tool interface、workspace 和官方 verifier；条件于 $\mathcal{K}_i$ 驱动 $\pi$ 重复观测-动作-反射循环，记录轨迹并返回标量 reward $r_i \in [0,1]$ 及诊断 $v_i$。

2. **State 四元组**：$\mathbf{s}_i = (x, \tau_i, \tau^\star, v_i)$，其中 $x$ 固定任务实例、$\tau_i$ 当前执行轨迹、$\tau^\star$ 参考成功轨迹（离线提供，不注入运行时）、$v_i$ 验证器诊断（per-test outcomes、error messages）。

3. **Critic $\mathcal{C}_\phi$**：基于状态、reward、历史 skill 和 replay memory 生成自然语言修改建议 $\mathbf{g}_i = \mathcal{C}_\phi(\mathbf{s}_i, r_i, \mathcal{K}_i, \mathcal{M}_i)$；系统区分缺失程序引导、tool misuse、output-contract violation、non-actionable infrastructure failure 四类错误，输出局部化编辑指令。

4. **Replay Memory $\mathcal{M}_i$**：压缩文本历史，保留三类证据：(a) 带 verifier 诊断的 failure signatures，(b) critic-summary 历史，(c) accepted edits 及其 observed outcomes；支持 rollback 防御性能退化。

5. **Actor $\mathcal{A}_\theta$**：执行四种显式编辑操作（Insert、Replace、Create、Delete）生成更新 $\Delta_i = \mathcal{A}_\theta(x, \mathcal{K}_i, \tau^\star, \mathbf{g}_i, \mathcal{M}_i)$，并通过 Apply 运算合成新 skill $\mathcal{K}_{i+1}$；可选地合成 task-local helper scripts 以卸载确定性计算。

**迭代调度**：对 $i = 0, ..., I-1$ 执行优化循环；支持 batch 并行（$B = \{x^{(n)}\}_{n=1}^N$），独立或共享更新 $\mathcal{K}_i^{(n)}$；最终产出 $\mathcal{K}_I$。

## 实验与结果
**评估设置**：
- 目标模型：Qwen3.5-9B、Qwen3.5-4B（OpenCode agent loop）
- 前沿模型：GPT-5.4（仅用于 skill 生成阶段 actor/critic）
- 基准：SkillsBench（26 tasks）、SkillLearnBench（100 instances）、SWE-Skills-Bench（117 instances, 10 skills）、GAIA（165 validation tasks）、EarthBench（248 samples）
- 基线：No-skill、Manus（闭源）、AutoSkill、EvoSkill、SkillX（开源）

**主要结果（Table 1）**：
- **Qwen3.5-9B**：SKILLER 在 SkillsBench 达 73.91（vs 最佳基线 SkillX 60.87，+13.04pp）、SWE-Skills-Bench 达 82.80（vs Manus 62.40，+20.4pp）、SkillLearnBench 达 32.11（vs Manus 31.17，+0.94pp）、EarthBench 达 76.08（vs Manus 71.24，+4.84pp）。
- **Qwen3.5-4B**：SKILLER 在 SkillsBench 达 43.48（vs Manus 36.23，+7.25pp）、SWE-Skills-Bench 达 66.70（vs Manus 53.40，+13.3pp）、SkillLearnBench 达 33.00（vs Manus 31.17，+1.83pp）、EarthBench 达 71.51（vs Manus 71.51，持平）。
- **零样本泛化（Table 2）**：Qwen3.5-9B 在 GAIA 测试集达 49.59（vs SkillX 42.28，+7.31pp）、EarthBench 达 72.31（vs Manus 70.43，+1.88pp）；Qwen3.5-4B 在 GAIA 达 40.65（vs Manus 34.55，+6.1pp）。

**关键结论**：
1. SKILLER 在所有五基准上 consistently 超越四种开源方法与一种闭源方法。
2. 在 SWE-Skills-Bench 上，Qwen3.5-4B + SKILLER 的 pass rate 甚至超过 Qwen3.5-9B 配合所有基线 skill，证明 tailor-made 策略比参数规模更重要。
3. 学习动态（Figure 3）显示复杂软件工程任务需多步累积收敛，而格式化任务两步行即可稳定。
4. 成本分析（Table 4）：SKILLER 生成成本 $8.95（输入 2.68M + 输出 0.15M tokens），介于 AutoSkill ($2.53) 与 SkillX ($14.55) 之间，但平均性能最高（62.86 vs 48-52）。

## 相关工作脉络
1. **Agent Skill 基础定义**：Anthropic 将 skill 形式化为标准化 procedural knowledge + tool-use conventions 打包；本文定位在于解决 skill 如何适配不同执行器（executors），而非 skill 本身的语义定义。
2. **Skill 获取与进化**：EvoSkill（trajectory-based evolution）、AutoSkill（lifelong self-evolution）、SkillX（knowledge base construction）均面向大规模模型；本文填补"面向 compact model 的 skill 自适应生成"空白。
3. **RL 驱动 Agent 优化**：Reflexion（Shinn et al. 2023）、SKILLRL（recursive skill-augmented RL）通过 verbal RL 改进 agent；本文区别在于以 skill artifact 本身为策略变量，而非更新模型权重。
4. **小模型 Agent 部署**：Xu et al. (2026) 指出小/中模型在工业环境中通过 skill constraint 可达到可用水平；本文提供自动化生成适配 skill 的可行路径。
5. **Benchmark 生态**：SkillsBench（task-specific skill utility）、SkillLearnBench（continuous generation）、SWE-Skills-Bench（software engineering execution-based）、GAIA/EarthBench（open-ended info-seeking）共同构成 skill 评估基准族；本文在这些基准上建立新 SOTA。

## 局限性与未来方向
1. **前沿模型依赖**：actor/critic 仍依赖 GPT-5.4 等闭源强模型，skill 生成阶段的成本与延迟尚未完全消除。
2. **离线生成范式**：当前为 task-level 离线优化，缺少 online 动态 skill 调整机制以应对分布漂移。
3. **Helper Script 边界**：task-local helper 的合成能力受限于 small-scale LVLM 的代码生成质量，复杂逻辑仍可能出错。
4. **Benchmark 选择性偏差**：部分基准（如 SWE-Skills-Bench）仅保留"skill 有 measurable effect"的 subset，可能高估实际泛化。
5. **未探索方向**：多 skill 协同、cross-task transfer、human-in-the-loop refinement 等机制尚未系统研究。

## 研究启发与可借鉴点
1. **"Policy as Text" 范式**：将 skill 视为可优化策略变量、用强模型 reasoning 生成编辑指令、用弱模型执行反馈 reward 的闭环，可迁移至 prompt engineering、workflow orchestration 等场景。
2. **结构化 State 设计**：$(task, trajectory, reference, diagnostics)$ 四元组兼顾意图、行为、参照、验证，为任何 agent skill 评估提供通用诊断框架。
3. **渐进式披露与快照回退**：RL loop 中 progressive disclosure 防上下文过载、snapshot rollback 防性能退化，可复用于任何基于 trial-and-error 的策略搜索。
4. **可执行抽象卸载**：将确定性计算（数据处理、验证逻辑）从 prompt 剥离到 external scripts，是小模型提升可靠性的关键杠杆；未来可探索 script 模板库与自动 synthesis。
5. **Cost-Performance 权衡分析**：Table 4 展示 token 消耗与性能的分离关系，提示后续工作应优化 critic/actor 的 prompt efficiency 而非单纯增加 rollout 次数。

## 关键术语表
**SKILLER**：Natural-language-driven reinforcement learning framework for automatically generating executor-specific skills for small LVLMs without neural weight updates.

**Executor-specific Skill**：Tailored to the action distribution and failure modes of a target compact model, unlike generic prompts designed for frontier models.

**Language-Level Policy Iteration**：Iterative optimization where the textual skill is the policy variable, updated via natural language feedback loops instead of gradient descent.

**Progressive Skill Disclosure**：Mechanism that gradually reveals skill content during execution to prevent overwhelming small models with long textual contexts.

**Replay Memory**：Compact textual history storing failure signatures, critic diagnoses, and accepted edits across optimization steps, enabling rollback and avoiding repeated failures.

**Helper Script**：Task-local deterministic program synthesized by the actor to offload complex procedural reasoning from natural language to executable artifacts.

**Verifier-Grounded Reward**：Scalar reward $r_i \in [0,1]$ derived from official benchmark verifier (pass/fail or normalized pass rate), anchoring language feedback to ground-truth acceptance criteria.

## 可复现要素
- **数据集**：SkillsBench、SkillLearnBench、SWE-Skills-Bench、GAIA、EarthBench（均为公开 benchmark）
- **代码/权重**：项目已开源，链接 https://github.com/DANG-ai/SKILLER
- **关键超参**：
  - Executor sampling temperature: 0.2
  - Actor/Critic sampling temperature: 0.3
  - Optimization steps: 5（所有 task）
  - Max tool steps per rollout: 30
  - GPT-5.4 作为 actor/critic（具体 API 配置论文未详述）

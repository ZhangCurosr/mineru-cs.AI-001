---
title: "Beyond-Outcome-Rewards-Step-Level-Self-Distilled-Policy-Opti"
source: https://arxiv.org/pdf/2608.12764v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:53:29"
field: "Agent 强化学习训练"
keywords: ["self-distillation", "reinforcement learning", "search agents", "GRPO", "process supervision", "credit assignment", "Evidence Anchors"]
innovations: ["提出Evidence Anchors构造搜索代理特权信息，避免完整答案泄露", "将自蒸馏信号转为步级优势权重应用于错误轨迹，解耦更新方向与幅度"]
benchmarks: ["BrowseComp", "GAIA", "FRAMES"]
---

# 论文速读：Beyond-Outcome-Rewards-Step-Level-Self-Distilled-Policy-Opti

## 一句话总结
本文提出 SSPO（Step-Level Self-Distilled Policy Optimization），通过将自身 logits 作为教师信号并结合 Evidence Anchors（从网络提取的步级证据片段）作为特权信息，将师生分歧转化为 GRPO 中的步级优势权重，仅应用于错误轨迹，解决了深度搜索代理中结果奖励稀疏与特权信息泄露的核心矛盾。

## 研究问题与动机
1. **奖励稀疏导致信用分配困难**：深度搜索代理轨迹长达 20+ 步，但标准 RL 仅提供单个二值结果奖励，无法判断哪些步骤对成败负责。
2. **自蒸馏到多轮搜索的假设断裂**：现有 OPSD 方法依赖数学参考答案或代码执行反馈等现成特权信息，而开放式网页搜索缺乏此类自然存在的监督信号。
3. **信息不对称引发捷径学习**：教师模型有特权信息（如正确答案、筛选后的证据）仅需 3.5 步即可完成搜索，而学生需 17.7 步探索；直接蒸馏会让学生模仿教师的"捷径"行为，在测试时放弃工具调用。
4. **Token 级监督与搜索行为单元不匹配**：搜索行动的价值在于信息获取动作本身，而非单个 token 选择，token 级信号难以对齐搜索的自然粒度。

## 核心贡献（创新点）
1. **提出 Evidence Anchors 构造特权信息**：用 SOTA LLM 为每个 QA 对提取约 5 条关键证据片段作为教师前缀，既提供过程监督又避免暴露完整答案路径。
2. **提出 SSPO 将蒸馏信号转为步级优势权重**：不直接优化师生分布差异，而是将分歧程度转化为 GRPO 中的权重因子，解耦"更新方向"（由结果奖励决定）与"更新幅度"（由教师调制）。
3. **证明多轮搜索场景需要重新设计自蒸馏**：直接 OPSD 蒸馏反而低于 GRPO（11.8 vs 12.8），证实特权信息泄露在长轨迹搜索中尤为严重。
4. **验证步级粒度优于 token 级粒度**：Step-level 权重在 BC-Sub 上达到 14.5，显著高于 Token-level 的 11.8，且熵曲线更稳定。
5. **实现样本效率翻倍**：SSPO 训练 100 步即超越 GRPO 训练 200 步，额外计算开销仅约 5%。

## 方法详解
1. **Evidence Anchors 构造**：使用 SOTA LLM（实验中为 GPT-OSS-120B）识别支持 ground-truth 的关键证据，每个问题约 5.24 个锚点，包含来源标题和解释；为教师提供锚点及学生生成的错误答案，防止重复相同错误。
2. **步级概率计算**：对轨迹中每个步骤 τ，计算教师分布 P_T 与学生分布 P_S 的联合概率（对步骤内所有 token 做自回归连乘）。
3. **特权信息增益**：Δ_τ^step = sg(log(P_T / P_S))，stop-gradient 确保仅用于权重计算。
4. **步级优势权重**：w_τ^step = min(exp(sign(A^(i)) · Δ_τ^step), 1 + ε)，其中 A^(i) 为轨迹级 GRPO 优势，ε 为超参数（论文设为 0.2）。
5. **仅应用于错误轨迹**：当 R_final < 1 时用 w_t · A_t 替换原优势项，正确轨迹保持原 GRPO 目标不变以保留多样性。
6. **权重语义**：错误轨迹中，学生对教师认可的步骤过于自信时放大惩罚，对教师支持的步骤不确定时减小惩罚。

## 实验与结果
1. **数据集与设置**：冷启动使用 ~4000 条正确轨迹（WebExplorer 管线收集）；On-Policy 训练使用 ~6000 条 DeepForge QA 对（难度比 1.5:3.5:3.5:1.5）；基座模型 Qwen3-8B。
2. **评估基准**：BrowseComp（OpenAI 高难检索基准）、GAIA 文本子集、FRAMES（Google 事实性评测）。
3. **主要结果**：
   - BrowseComp：GRPO 13.6 → SSPO 15.7（+2.1）
   - GAIA：GRPO 47.3 → SSPO 49.3（+2.0）
   - FRAMES：GRPO 69.8 → SSPO 73.0（+3.2）
   - 三基准平均：SSPO 46.0 vs GRPO 43.6，相对冷启动提升 +4.8 vs +2.4
   - 平均搜索轮次：GRPO 20.0 vs SSPO 20.5（几乎无损效率）
4. **训练动态**：SSPO 在约 30-50 步 warm-up 后持续超越 GRPO，梯度范数维持较高水平，表明更可持续的学习信号。
5. **消融验证**：
   - 直接 OPSD 蒸馏：11.8 准确率，轨迹缩短至 30.9 步（证实捷径学习）
   - Token 级权重：11.8 准确率，35.5 步（恢复长度但精度未提升）
   - 仅 Evidence Anchors：14.0；仅错误答案反馈：12.4（锚点贡献为主）

## 相关工作脉络
1. **GRPO [37,11]**：Group Relative Policy Optimization，通过组内归一化奖励估计优势，是本文基线算法。
2. **OPSD / 自蒸馏系列 [63,54,13,38,35,17]**：Self-Distilled Reasoner、RLSD、SRPO 等工作在单轮推理/代码场景验证自蒸馏，本文扩展至多轮搜索。
3. **RLSD [54]**：首次指出直接 OPSD 优化会导致特权信息泄露，提出将蒸馏信号转为优势权重，本文沿用此思想并适配搜索场景。
4. **CriticSearch [62]、PPR [51]、SmartSearch [45]**：引入外部过程奖励模型进行步级信用分配，依赖额外 LLM 评分；SSPO 无需外部评分器，用自蒸馏信号替代。
5. **WebExplorer [26]、WebSailor [20]**：深度搜索代理的 RL 训练管线，本文沿用其冷启动轨迹收集与 scaffold 设计。

## 局限性与未来方向
1. **数据规模受限**：受 API 成本（Serper、Jina、LLM 服务）限制，仅使用数千样本。
2. **模型规模有限**：实验仅在 8B 模型上进行，未验证更大模型的泛化性。
3. **单语言限制**：训练与评估均为英文，教师模型（GPT-OSS-120B）的 chain-of-thought 为英文，未涉及多语言场景。
4. **未来方向**：扩展到更多样化代理场景、更大模型规模、多语言设置。

## 研究启发与可借鉴点
1. **特权信息构造策略**：Evidence Anchors 的思路——用 SOTA LLM 提取结构化证据片段而非完整答案，可作为其他开放式任务中构造"安全"教师信号的通用范式。
2. **解耦更新方向与幅度**：将结果奖励固定为梯度方向、教师信号调制更新幅度，避免了直接蒸馏的信息泄露风险，可迁移至其他长 horizon 决策任务。
3. **粒度匹配原则**：监督信号粒度应与行为自然单元对齐（搜索 Agent 用 step 级而非 token 级），这一设计原则可推广至代码生成、多轮对话等场景。
4. **仅修正错误轨迹**：保持正确轨迹的原始 GRPO 更新以维护多样性，同时集中资源修正失败路径，这一策略在 RL 中具有良好的可扩展性。
5. **低成本过程监督**：仅增加一次前向传播（约 5% 开销）即显著提升样本效率，对资源受限的研究团队极具吸引力。

## 关键术语表
**Evidence Anchors**：从网络提取的简洁步级证据片段，作为搜索代理自蒸馏的特权信息，平均每个问题 5.24 条。
**SSPO**：Step-Level Self-Distilled Policy Optimization，本文提出的方法，将师生分歧转化为 GRPO 中的步级优势权重。
**GRPO**：Group Relative Policy Optimization，通过组内归一化奖励估计优势的策略梯度算法，本文基线。
**OPSD**：On-Policy Self-Distillation，使用模型自身 logits 作为教师信号的自蒸馏方法。
**Privileged Information**：教师可见但学生在测试时不可见的额外信息，如参考答案、执行反馈或证据锚点。
**Credit Assignment**：信用分配，确定轨迹中哪些步骤对最终结果负责，是长 horizon 代理的核心挑战。
**RLVR**：Reinforcement Learning with Verifiable Rewards，可验证奖励的强化学习设置，适用于有明确对错判定的任务。
**On-Policy**：策略在与当前参数相同的分布下采样数据并进行更新，保证训练稳定性。

## 可复现要素
- **数据集**：DeepForge 数据集（~6000 条 QA 对），论文未明确说明是否公开；冷启动轨迹 ~4000 条（WebExplorer 管线收集）
- **代码/权重**：论文未明确说明是否开源
- **关键超参**：学习率 1×10^-6，batch size 64，每问题 8 rollouts，ε=0.2，teacher 每 50 步重置，最大 context length 128K，最大 agent steps 100
- **基座模型**：Qwen3-8B
- **工具**：Serper API（search）、Jina（browse/content extraction）

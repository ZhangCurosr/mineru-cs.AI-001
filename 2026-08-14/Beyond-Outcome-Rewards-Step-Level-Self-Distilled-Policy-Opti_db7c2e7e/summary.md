---
title: "Beyond-Outcome-Rewards-Step-Level-Self-Distilled-Policy-Opti"
source: https://arxiv.org/pdf/2608.12764v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:53:21"
field: "多轮搜索 Agent 训练"
keywords: ["agent reinforcement learning", "self-distillation", "search agents", "process rewards", "credit assignment", "on-policy learning"]
innovations: ["提出Evidence Anchors作为多轮搜索的步骤级特权信息，避免答案泄露", "将教师-学生分歧转化为GRPO中的步骤级优势权重，仅作用于错误轨迹"]
benchmarks: ["BrowseComp", "GAIA", "FRAMES"]
---

# 论文速读：Beyond-Outcome-Rewards-Step-Level-Self-Distilled-Policy-Opti

## 一句话总结
本文提出 **Step-Level Self-Distilled Policy Optimization (SSPO)**，通过构建 **Evidence Anchors** 作为多轮搜索代理的步骤级特权信息，并将教师-学生分歧转化为 **GRPO 中的步骤级优势权重**（仅应用于错误轨迹），为长 horizon 搜索任务提供更细粒度的过程监督。在 Qwen3-8B 上，SSPO 持续超越 GRPO，且只需约 5% 额外计算开销即可匹敌双倍训练步数的 GRPO。

## 研究问题与动机
- **稀疏奖励瓶颈**：深度搜索代理轨迹包含 20+ 步，但标准 RL 仅提供单一二值结果奖励，信用分配困难。
- **自蒸馏难以直接迁移**：现有 On-Policy Self-Distillation (OPSD) 主要在单轮数学/代码推理上有效，多轮搜索场景下信息不对称极端——教师（有特权信息）平均仅需 3.5 步，学生需 17.7 步，直接蒸馏会导致学生过早放弃工具调用。
- **特权信息构造挑战**：与数学有参考答案、代码有执行反馈不同，开放网络搜索缺少现成的、既能有效指导教师又安全蒸馏给学生的监督信号。
- **轨迹级 vs 步骤级粒度**：搜索行为的核心单元是"信息检索动作"（搜索/浏览），而非单个 token；现有方法若以 token 粒度施加蒸馏信号，与搜索行为的自然单元不匹配。

## 核心贡献（创新点）
- **Evidence Anchors 构造**：使用 SOTA LLM 为每个 QA 对提取若干紧凑的步骤级证据片段，作为教师模型的 privileged prefix，既能提供过程线索又不泄露完整答案路径（每问题平均约 5.24 个锚点）。
- **SSPO 框架**：将教师-学生分歧转化为步骤级优势权重 $w_\tau^{\text{step}}$，仅应用于最终奖励 $R_{\text{final}} < 1$ 的错误轨迹，解耦更新方向（由环境奖励决定）与更新幅度（由教师调制）。
- **步骤级而非 token 级**：对每个 Thought + Action 组合构成一个完整搜索步骤共享统一权重，而非逐 token 加权，更契合信息检索动作的自然粒度。
- **双路特权信息互补**：Evidence Anchors 提供中间步骤的过程监督信号，错误答案反馈聚焦最终步骤的惩罚放大，两者结合效果最优。

## 方法详解
- **证据锚点构造**：用 LLM 从网络提取支持 ground-truth 答案的证据片段（附 URL），经 Jina 服务验证 URL 可达性与标题一致性，过滤无效锚点后构建训练数据（6,000+ QA 对，平均 5.24 锚点/问题）。
- **步骤联合概率建模**：对每个学生生成的步骤 $\tau$（包含思考 token $\mathtt{t}_\tau$ 和工具调用 token $\mathtt{a}_\tau$），计算教师分布 $P_T$（带特权上下文）与学生分布 $P_S$ 的自回归条件概率乘积。
- **特权信息增益**：$\Delta_\tau^{\text{step}} = \text{sg}(\log P_T(\mathtt{t}_\tau, \mathtt{a}_\tau) / P_S(\mathtt{t}_\tau, \mathtt{a}_\tau))$，stop-gradient 防止梯度回传到教师。
- **优势权重设计**：$w_\tau^{\text{step}} = \min(\exp(\text{sign}(A^{(i)}) \cdot \Delta_\tau^{\text{step}}), 1+\epsilon)$，$\epsilon = 0.2$；对错误轨迹（$R_{\text{final}} < 1$）替换 GRPO 中的 advantage 项为 $\hat{A}_t^{(i)} = w_t A_t^{(i)}$，正确轨迹保持原 GRPO 不变。
- **教师参数更新策略**：每 50 步用当前学生策略重新初始化教师参数，避免教师分布过时。
- **训练格式**：冷启动 SFT（1k 步，lr=$10^{-5}$，batch=32）→ 在线策略学习（~6k 样本，lr=$10^{-6}$，batch=64，8 rollouts/问题，max context=128K，max steps=100）；loss 仅对 agent 生成的 Thought、Action、最终答案 token 计算，环境观察 token  masking。

## 实验与结果
- **数据集**：DeepForge 开源数据集 6,000 个英文 QA 对（难度比例 1.5:3.5:3.5:1.5），WebExplorer 收集 4,000 条正确轨迹用于冷启动。
- **基准**：BrowseComp（OpenAI 高难信息检索）、GAIA text-only 子集、FRAMES（Google 事实准确与推理）。
- **主要结果（Qwen3-8B）**：
  - BrowseComp：SSPO 15.7 vs GRPO 13.6 vs Cold-Start 11.7
  - GAIA：SSPO 49.3 vs GRPO 47.3 vs Cold-Start 44.8
  - FRAMES：SSPO 73.0 vs GRPO 69.8 vs Cold-Start 67.2
  - 平均：SSPO 46.0 vs GRPO 43.6 vs Cold-Start 41.2（SSPO 相对 cold-start 提升 +4.8，GRPO 仅 +2.4）
- **效率对比**：SSPO 100 步训练即超越 GRPO 200 步；平均步数 20.5 vs 20.0，几乎无推理效率损失。
- **额外开销**：teacher forward pass 仅占每步训练时间约 5%（Table 9）。

## 相关工作脉络
- **GRPO**：作为 baseline，提供轨迹级群组归一化优势估计，所有 token 共享同一 sequence-level advantage。
- **OPSD / Self-Distilled Reasoner [63]**：单轮推理中用参考答案构造特权前缀做 token 级 KL 蒸馏，未考虑信息泄露与搜索行为粒度。
- **RLSD [54] / SRPO [17]**：发现直接蒸馏会导致特权信息泄露并提出优势权重方案，但均在单轮场景验证；本文将其扩展到多轮搜索并证明步骤级粒度优于 token 级。
- **CriticSearch [62] / PPR [51] / SmartSearch [45]**：依赖外部冻结 LLM 作为步骤评估器提供过程奖励，引入额外推理成本；SSPO 完全通过自蒸馏消除对显式步骤奖励模型的需求。
- **WebExplorer [26] / WebSailor [20]**：同规模搜索代理工作，使用更大训练数据；本文强调方法层面的效率增益（5% 开销 vs 双倍步数）。

## 局限性与未来方向
- 数据规模有限：冷启动仅 ~4,000 条、在线训练仅 ~6,000 QA 对，受 API 成本约束。
- 模型规模受限：仅在 8B 模型上验证，未扩展到更大参数规模。
- 单语言限制：训练与评估均为英文，因教师模型（GPT-OSS-120B）的 CoT 特性为英语，未涵盖中文、日文等多语言场景。
- 未来方向：扩展至更多样化的 agent 场景、更大模型规模、多语言设置，并探索细粒度监督对其他 RLVR 方法的通用价值。

## 研究启发与可借鉴点
- **特权信息≠答案泄露**：Evidence Anchors 的设计思路（提取支撑性证据片段而非完整解答）可作为开放域搜索任务构造自蒸馏教师信息的范式。
- **解耦方向与幅度**：将蒸馏信号从优化目标改为 advantage 权重、仅作用于错误轨迹，这一"修正幅度不改方向"的思路可迁移至其他长程决策场景（如游戏 Agent、代码生成）。
- **步骤粒度优于 token 粒度**：搜索代理的训练信号粒度应与行为单元（一次搜索/浏览）对齐，而非默认 token 级，这一原则可推广至任何以工具调用为基本动作的 agent 系统。
- **双路信号互补验证**：通过 rescore 分析解耦两个特权信息源的作用（一个主导中间步骤评估、一个聚焦最终步骤惩罚），这种因果归因方法值得借鉴。

## 关键术语表
- **SSPO (Step-Level Self-Distilled Policy Optimization)**：本文提出的方法，将教师-学生分歧转化为步骤级优势权重应用于 GRPO，仅对错误轨迹进行调制。
- **Evidence Anchors**：从网络提取的紧凑步骤级证据片段，作为教师模型的特权信息前缀，提供过程监督而不泄露完整答案路径。
- **On-Policy Self-Distillation (OPSD)**：利用模型自身带特权信息的 rollout 作为教师、无特权信息的 rollout 作为学生进行的自蒸馏，避免外部教师依赖。
- **RLVR (Reinforcement Learning with Verifiable Rewards)**：具备可验证奖励的强化学习设定，通过精确匹配或语义等价判断答案正确性。
- **GRPO (Group Relative Policy Optimization)**：通过组内 reward 归一化估计 advantage 的 on-policy RL 算法，当前 agent 训练主流框架。
- **Credit Assignment**：在长轨迹中将成功/失败归因于具体中间步骤的问题，是深度搜索代理训练的核心瓶颈。
- **Privileged Information Leakage**：当蒸馏目标包含测试时不可用的特权信号时，模型在训练期利用捷径但在测试期失效的现象。

## 可复现要素
- **数据集**：DeepForge 开源数据集（部分 QA 对），冷启动轨迹来自 WebExplorer 流程收集；论文未提供完整训练数据下载链接。
- **代码/权重**：论文未明确声明代码开源状态；基座模型为 Qwen3-8B（HuggingFace 公开）。
- **关键超参**：$\epsilon = 0.2$（权重 clip 上界）、teacher 每 50 步同步一次、batch size=64、lr=$10^{-6}$、rollouts=8/问题、max context=128K、max steps=100、format reward 权重 0.2。

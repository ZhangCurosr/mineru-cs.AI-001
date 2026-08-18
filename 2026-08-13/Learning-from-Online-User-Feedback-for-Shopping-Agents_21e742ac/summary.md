---
title: "Learning-from-Online-User-Feedback-for-Shopping-Agents"
source: https://arxiv.org/pdf/2608.11604v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:44:28"
field: "对话式推荐系统"
keywords: ["shopping agent", "reinforcement learning", "on-policy distillation", "online user feedback", "recommendation", "conversational recommendation"]
innovations: ["双源在线反馈（行为+对话）联合学习框架LOFA", "基于特权未来信息的反馈感知teacher实现on-policy蒸馏", "RL→OPD顺序训练提升推荐质量与对话修复成功率"]
benchmarks: ["JD-Search", "JD-conv"]
---

# 论文速读：Learning-from-Online-User-Feedback-for-Shopping-Agents

## 一句话总结
本文提出 LOFA 框架，让基于 LLM 的购物 Agent 直接从真实电商交互日志中学习，无需人工标注；通过组合"基于购买结果的强化学习"与"基于对话反馈的 on-policy 蒸馏"两种互补监督信号，同时提升推荐质量与用户满意度对齐。

## 研究问题与动机
- **现有方法过度依赖离线静态数据**：当前购物 Agent 优化主要使用用户-物品交互记录或合成偏好数据，忽略了已部署 Agent 积累的丰富在线对话反馈。
- **在线反馈的异构性与稀疏性未被充分利用**：真实日志中同时存在显式行为反馈（点击/购买）和对话内指令反馈（用户后续纠偏），但前者稀疏延迟、后者噪声大，难以自动转化为可靠学习信号。
- **单一反馈类型无法同时优化推荐质量与对话质量**：仅用购买信号能改善推荐排序，但无法解释用户不满的原因；仅用对话反馈则缺乏结果层面的可靠监督。
- **推理能力在 SFT 训练中被削弱**：现有 SFT 方法去除了思维链（reasoning trace），导致 Agent 多步推理能力下降；而在线反馈驱动的方法可保留推理结构。

## 核心贡献（创新点）
1. **将"从在线用户反馈中学习"定义为购物 Agent 优化的新范式**，并提出统一的 LOFA 框架，自动将真实日志中的异构反馈转化为训练信号，无需人工标注。
2. **识别出两种互补的在线反馈形式——显式行为反馈与对话内指令反馈**，并设计了基于 LLM 的反馈挖掘流程，从噪声对话中提取可操作的指令性信号。
3. **提出双反馈优化策略：GRPO 强化学习 + on-policy 蒸馏（OPD）**，前者提供可靠的结果级监督，后者提供细粒度的 token 级修正指导，二者顺序组合实现优势互补。
4. **构建基于京东真实日志的两个评测数据集（JD-Search、JD-conv）**，在推荐排序指标与对话修复成功率上均显著超越强基线，验证了框架的有效性。

## 方法详解
### 整体框架（Figure 2）
LOFA 采用**顺序训练管道**：先用 GRPO 在含购买结果的 session 上做强化学习，再用 OPD 在 turn 级样本上做自蒸馏，两阶段均无需人工标注。

### 显式行为反馈学习（Section 3.2）
- **数据构建**：筛选包含成功购买的完整交互 session，提取用户画像（meta profile + short-term preference + shopping memory）及购买物品 $i^*$。
- **GRPO 优化**：设计两个互补奖励：
  - **格式奖励 $R_{\mathrm{fmt}}$**：违反预设输出格式（缺少 `<answer>` 标签、推荐不足等）罚 $-1$，否则为 $0$。
  - **推荐奖励 $R_{\mathrm{rank}}$**：以购买物品为目标，计算 NDCG@20 作为奖励值。
  - **总奖励**：若格式违规则 $R(\tau) = -1$，否则 $R(\tau) = R_{\mathrm{rank}}(\tau)$。

### 对话内指令反馈学习（Section 3.3）
- **反馈挖掘**：将多轮对话分解为 turn 级样本，使用 LLM 识别后续用户响应 $q_{t+1}$ 中的指令性信号，分类为五类：Explicit Criticism（显式批评）、Implicit Deduction（隐式推断）、Pure Negative Feedback（纯负面）、Comparative Preference（比较偏好）、Topic Shift/Abandonment（话题转移/放弃，丢弃）。
- **反馈感知 Teacher 构建**：Teacher 在部署时上下文 $C_t$ 基础上注入特权信息：先前响应 $a_t$、后续反馈 $q_{t+1}$、反馈类别 $c_t$ 及解释 $e_t$，从而生成 token 级的修正指导。
- **On-Policy Distillation (OPD)**：以反向 KL 散度优化 Student：
  $$\mathcal{L}_{\mathrm{OPD}} = \mathbb{E}_{\hat{a}_t \sim p_s}\left[\frac{1}{N}\sum_{j=1}^{N} D_{\mathrm{KL}}\left(p_s^{(j)} \parallel p_t^{(j)}\right)\right]$$
  其中 $p_s^{(j)}$ 与 $p_t^{(j)}$ 分别为 Student 与 Teacher 在第 $j$ 步生成分布，实现稀疏对话反馈到密集 token 级监督的转化。

### 训练顺序
实验表明 **RL → OPD** 顺序优于 OPD → RL，因先获得可靠的结果级监督再注入细粒度指令修正更有效。

## 实验与结果
- **数据集**：来自京东"京言"购物 Agent 的真实日志（匿名化），JD-Search（6,003 train / 698 test session，平均 2.07 轮），JD-conv（7,437 train / 812 test turn 样本）。
- **基线**：Qwen3-8B Base、NoThink、SFT、Self-Reflection、+RL、+OPD、+RL→OPD、+OPD→RL。
- **评估指标**：NDCG@K、Recall@K、MAP@K（推荐任务）；Success Rate（对话修复任务）。
- **主要结果**（Table 3）：
  - **+RL→OPD 达到最优**：NDCG@10 = **0.6512**（较 Base 提升 **+44.2%**），Recall@10 = **0.8911**（+46.0%），Success Rate = **0.6022**（较 Base 提升 +10.1%）。
  - 仅 +RL 对推荐指标提升显著（NDCG@10: 0.4112 → 0.6389），仅 +OPD 对 SR 提升显著（0.5443 → 0.6010），二者互补。
  - RL→OPD 优于 OPD→RL，验证顺序重要性。
- **不同反馈类型上的 SR**（Table 4）：OPD 类方法在 Explicit Criticism（+0.0461）、Implicit Deduction（+0.0723）、Pure Negative Feedback（+0.0244）、Comparative Preference（+0.2353）均持续超越所有基线；Comparative Preference 提升最大，Pure Negative Feedback 最难但仍有效。
- **超参**：温度 1.0 / top-p 0.95（训练），学习率 $1 \times 10^{-6}$，batch size 32，GRPO group size G=5，1-2 epochs 全参数微调；Judge 模型为 DeepSeek-V3.2。
- **规模泛化**（Figure 3）：在 Qwen3-4B 上也复现一致趋势，框架对 backbone 规模不敏感。

## 相关工作脉络
- **Agent-based Recommendation**：LLM4Rec、RecMind 等侧重架构改进与工具调用，优化方法主要依赖离线 SFT 或合成偏好；LOFA 首次系统利用在线部署日志中的双源反馈。
- **RLHF / DPO / GRPO**：现有偏好优化依赖人工标注或合成偏好数据集；LOFA 将 RL 直接应用于真实购买结果，无需额外标注。
- **On-Policy Distillation**：原用于从 self-generated mistakes 中学习（Agarwal et al., 2024）；LOFA 将其扩展为利用"已知未来用户反馈"构建 teacher，实现对话修正知识传递。
- **Self-Reflection / Inference-time Refinement**：无参数更新的推理时自我修正；LOFA 证明参数级在线反馈学习显著优于此类方法。
- **SFT-based Recommendation**：去除 reasoning trace 的 SFT 在本工作表现明显弱于 RL/OPD；本文强调保留推理能力的必要性。

## 局限性与未来方向
- **反馈挖掘依赖 LLM 判断**：反馈分类与解释生成由外部 LLM（DeepSeek-V3.2）完成，存在误分类风险，且引入额外推理成本。
- **仅限单 agent 场景**：当前框架针对单一购物 Agent 的日志，未探索多 agent 协作环境下的反馈利用。
- **购买信号仍较稀疏**：即使经过 session 筛选，成功购买的比例仍然有限，directive feedback 的覆盖面更广但噪声更大。
- **未涉及长期用户建模**：当前利用短期 session 级反馈，未考虑跨天/跨周期的用户偏好演化。
- **作者提及未来可扩展为数据飞轮（data flywheel）**，支持周期性模型更新。

## 研究启发与可借鉴点
1. **"显式结果 + 隐式对话"双源反馈的互补建模思路**可直接迁移至客服 Agent、决策 Agent 等其他对话式智能体场景。
2. **On-policy distillation 配合特权未来信息的 teacher 设计**是一种将稀疏/噪声用户反馈转化为 dense token-level 监督的有效范式，可推广至其他需要"事后修正"的任务。
3. **GRPO 在多目标奖励（格式约束 + 业务指标）下的调度方式**（惩罚式格式奖励 + 连续推荐奖励）对工业界部署的 RL 训练具有参考价值。
4. **顺序训练管道（RL then OPD）优于反向**的发现提示：在融合多种粒度监督时，应先建立稳定的结果级能力，再注入细粒度对话级修正。
5. **保留 reasoning trace 的在线反馈学习**比去除它的 SFT 更有效，为"是否保留 CoT"的工业训练选择提供了实证依据。

## 关键术语表
**LOFA**：Learning from Online User Feedback 的缩写，本文提出的从在线用户反馈学习购物 Agent 的统一框架。  
**GRPO**：Group Relative Policy Optimization，本文用于基于购买结果进行强化学习的算法变体。  
**On-Policy Distillation (OPD)**：在 agent 自身生成的轨迹上进行的蒸馏，本文用于将对话内反馈转化为 token 级修正监督。  
**Directive Feedback**：对话内指令反馈，指用户在后续 turn 中表达的不满、偏好细化或补充约束等对话信号。  
**Explicit Behavioral Feedback**：显式行为反馈，指点击、购买等可观测的最终行为结果。  
**Feedback-aware Teacher**：注入未来用户反馈信息的 teacher 模型，用于提供 token 级的修正指导分布。  
**Success Rate (SR)**：对话修复成功率，衡量 Agent 修正后响应是否解决了用户反馈中指出的问题。  
**JD-Search / JD-conv**：本文构建的两个基于京东真实日志的数据集，分别用于行为反馈学习与指令反馈学习。

## 可复现要素
- **数据集**：JD-Search 与 JD-conv（京东"京言"Agent 日志，已匿名化）；论文未声明是否对外公开。
- **代码/权重**：论文未提及开源。
- **关键超参**：学习率 $1 \times 10^{-6}$，batch size 32，训练 1-2 epochs，GRPO group size $G=5$，temperature 1.0 / top-p 0.95（训练），temperature 0.6 / top-p 0.95（推理）；Backbone 为 Qwen3-8B，Judge 为 DeepSeek-V3.2。

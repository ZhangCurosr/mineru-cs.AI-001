---
title: "Beyond-Final-Scores-A-Systematic-Evaluation-of-Agents-for-Lo"
source: https://arxiv.org/pdf/2608.13417v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:53:11"
field: "AI Agent评估与长周期自主研究"
keywords: ["long-horizon agents", "autonomous research", "process evaluation", "experience reuse", "agent harness", "benchmark design"]
innovations: ["提出C1/C2/C3过程维度的确定性规则基评估框架，超越单一最终分数", "设计反事实经验隔离实验（Intra-task擦除+Inter-task lessons提取）测量经验复用效应", "系统量化Harness对稳定性和峰值性能的解耦影响并提出自动化harness演化"]
benchmarks: ["AutoLab"]
---

# 论文速读：Beyond-Final-Scores-A-Systematic-Evaluation-of-Agents-for-Lo

## 一句话总结
论文提出了一个超越最终分数的系统化评估框架，从过程能力（Solution Framing/Execution/Feedback Control）和经验复用两个维度全面评估七个前沿模型在36个长周期AI研发任务上的表现，发现当前智能体更接近工程优化器而非完全自主的研究者。

## 研究问题与动机
- 现有基准（如MLAgentBench、MLE-bench、RE-Bench等）主要依赖单一最终分数，无法揭示研究循环中的过程瓶颈或经验积累效应
- 相同最终分数可能来自早期发现的有效方向，也可能来自大量试错后的偶然结果，过程评估缺失导致诊断困难
- 传统评估将每次运行视为独立样本，无法判断累积经验是否真正改善后续决策（正/负迁移并存）
- 模型能力与周围系统（harness）之间存在强交互，仅测试模型本体无法区分能力来源

## 核心贡献（创新点）
1. 提出四维评估框架（最终分数+过程C1-C3+经验复用M_intra/M_inter+harness对比），首次将过程诊断与经验迁移纳入长周期AI研发评估体系
2. 设计完全确定性的规则基过程指标，直接基于验证器信号和轨迹事件计算，无需LLM主观判断，保证可复现性和审计性
3. 引入反事实经验干预设计：Intra-task通过分支点前后对比隔离经验效应，Inter-task通过source-task经验向held-out target-task的转移测量跨任务迁移
4. 首次系统性量化harness对智能体性能的影响，发现harness主要作用于run-to-run稳定性而非提升峰值性能，并提出自动化harness演化作为改进方向

## 方法详解
**评估协议**：在AutoLab的36个任务（Model Development 7个、System Optimization 15个、Puzzle & Challenge 10个、CUDA 4个）上评估7个前沿模型，每模型-任务对进行3次独立rollout，报告avg@3和best@3。

**C1: Solution Framing（方案框架构造）**：将轨迹映射到公共 horizon H=20，用high-water-mark曲线 h_i = max(h_{i-1}, x_i) 记录累计最佳进展，在早/中/晚三个阶段（前5/中5/后10步）加权平均，评估方向发现的质量和速度。

**C2: Execution（执行可靠性）**：对每个评估checkpoint检查交付质量 g_i（是否运行且正确），并引入构建失败惩罚 d(n)，其中 n 为代码相关构建错误数：n=0→1.00, n=1→0.85, n=2→0.70, n=3→0.60, n≥4→0.50，最终 C2 = (1/|I|) Σ g_i·d(n_i)。

**C3: Feedback Control（反馈控制）**：包含两个分量：Retention A_1 = clip(f/p, 0, 1) 衡量峰值保留率（p为最高分，f为最终分）；Recovery A_2 对每个dip episode计算恢复 credit B_e = ρ_e/L_e（ρ_e为恢复分数比例，L_e为恢复所需步数），再乘以步进成本折扣 D_e，最终 C3 = 0.5·A_1 + 0.5·A_2（无dip时仅用A_1）。

**M_intra（任务内经验复用）**：在轨迹中段选择分支点，保留经验条件 S^exp 与擦除经验条件 S^no_exp 对比，ΔS_intra = S^exp - S^no_exp ∈ [-1, +1]，正值表示经验有帮助。

**M_inter（跨任务经验复用）**：从源任务提取lessons.md后应用于目标任务，ΔS_inter = S^(+) - S^(0) ∈ [-1, +1]，并对比显式lessons vs 原始工作空间、自我生成 vs 跨模型生成两种形式。

## 实验与结果
- **Opus-4.7** 综合最强：avg@3=0.739，best@3=0.790，C1=0.612，C2=0.967，C3=0.920，但单任务推理成本高达$89.9
- **avg@3与best@3差距揭示可靠性差异**：模型间avg@3极差为0.237，best@3极差仅0.122，GLM-5.2的avg@3 (0.682) 接近Opus但Kimi-K2.7-Code的best@3仅略低0.028，说明多数模型具备潜力但稳定性不足
- **GPT-5.5 vs Gemini-3.1-Pro案例**：两者avg@3相近（0.663 vs 0.652），C1完全相同（0.555），但GPT的C2更高（0.958 vs 0.889）而Gemini的C3更高（0.920 vs 0.858），说明相似终态可能掩盖完全不同的能力剖面
- **CUDA任务最难**：avg@3极差0.403，best@3极差0.414；Model Development C2最高(0.985)但C3最低(0.743)；Puzzle & Challenge最均衡
- **经验复用效果**：Intra-task经验普遍有益（除Kimi略负外所有模型均正增益）；Inter-task DeepSeek-V4-Pro获得最大提升（avg@3 +0.093），Gemini-3.1-Pro反而下降（-0.017）
- **新颖性稀缺**：252个最佳方案中仅3个（1.2%）经人工审核确认为novel-approach，而evaluation-hacking有16个（6.3%），是novel的5倍
- **Harness影响**：native harness和OpenCode相比Claude Code主要提升avg@3（稳定性），best@3变化很小，跨harness模型排序不变

## 相关工作脉络
- MLAgentBench / MLE-bench / RE-Bench / AutoLab 等基准聚焦最终性能评估，本文在此基础上扩展至过程诊断与经验迁移评估
- AgentBoard / TRAJECT-Bench / WebStep / AgentLens 等过程评估工作针对工具调用序列或子目标达成，本文针对开放研究循环设计无judge的确定性过程指标
- Reflexion / ExpeL / LifelongAgentBench / SEA-Eval 等研究经验复用，本文首次在AutoResearch场景中分离Intra-task与Inter-task并设计反事实对照
- SWE-agent / Harness-Bench / Holistic Agent Leaderboard 关注harness效应，本文首次在长周期研究任务上系统比较共享harness vs native harness vs OpenCode的影响差异

## 局限性与未来方向
- 过程指标C1-C3仅捕捉轨迹中可观测信号，无法评估未实现的语义级想法或潜在推理质量；C3的Recovery分量在无dip轨迹中无法有效计算
- 经验复用实验受限于特定的分支点选择、source-target配对和lesson表示形式，不能完全泛化到更长部署周期中的复杂记忆动态
- 结论基于AutoLab的任务分布和单一共享harness设定，跨领域和harness组合的绝对分数及相对排名可能变化
- 成本估算依赖API公开定价和特定token计数方式，实际部署成本可能有差异
- 未来方向包括：面向过程瓶颈的针对性训练数据构建、推理时分支/剪枝的compute re-allocation策略、选择性记忆系统与经验验证/修正机制、以及奖励新颖性和有效性的新评测目标

## 研究启发与可借鉴点
- **过程指标的可迁移设计**：C1/C2/C3的确定性规则计算、高水位线聚合、以及构建失败惩罚折扣方案可直接迁移到其他长周期agent评估场景
- **反事实经验隔离范式**：Intra-task的分支点擦除对比、Inter-task的isolated workspace+lessons.md传递设计，为经验效应研究提供了可复用的因果推断实验范式
- **显式经验提取优于原始工作空间**：Inter-task实验中extracted lessons相比raw workspace显著提升transfer（+0.035 vs -0.007），提示agent memory系统应优先设计结构化知识提取模块而非简单上下文压缩
- **harness-稳定性解耦观察**：harness主要影响avg@3（稳定性）而非best@3（峰值），这对benchmark设计启示：跨系统比较必须固定harness或使用标准化接口
- **评估捷径警示**：6.3%的solution利用evaluation-hacking而非novel-approach，提示后续benchmark设计需增加evaluator robustness检查以防止agent exploitation

## 关键术语表
**Solution Framing (C1)**：评估智能体研究方向的探索效率，通过早/中/晚期的高水位线加权分衡量发现优质方案的速度和质量
**Execution (C2)**：评估将改动可靠转化为可运行且正确结果的实现能力，以构建失败频率为折扣因子
**Feedback Control (C3)**：评估智能体保留已有成果并在性能回退后恢复的能力，包含Peak Retention和Recovery Credit两个分量
**Intra-task Self-Improvement (M_intra)**：通过反事实分支点对比，测量同一任务内累积经验对后续方案质量的孤立贡献
**Inter-task Self-Improvement (M_inter)**：测量从已解源任务提取的lessons对新的目标任务性能的跨任务迁移增益
**Best-of-N (best@N / avg@N)**：best@N为N次rollout中的最高分，avg@N为N次rollout的平均分，前者衡量峰值能力后者衡量稳定性
**Evaluation Hack**：利用benchmark评测器的确定性特征（如固定PRNG seed或重复调用模式）绕过实际计算取得虚假高分的方案
**Harness Evolution**：通过自动搜索优化agent的工具接口、上下文管理策略和任务管理机制以提升性能，本文探索了仅4轮即产生跨任务迁移的进化结果

## 可复现要素
- 数据集：AutoLab（Xu et al., 2026），36个任务，论文提供了task instruction示例和评估prompt（Appendix J）
- 代码/权重：Project Page为 AutoResearchEval（论文未给出具体URL但指向项目主页），评估框架实现应开源
- 模型：7个商业/开源前沿模型（Claude-Opus-4.7、GPT-5.5、Gemini-3.1-Pro、GLM-5.2、Kimi-K2.7-Code、DeepSeek-V4-Pro、LongCat-2.0），模型本身各有对应来源
- 关键超参：C1 horizon H=20，epsilon=0.01噪声容差，C2构建失败折扣函数d(n)，Inter-task 19个target任务，Source选取标准（best@3_score > 0.5且best@3_commits ≥ 5）
- 成本：约10万美元模型推理费用，单任务成本$3.9（DeepSeek）~ $89.9（Opus）不等

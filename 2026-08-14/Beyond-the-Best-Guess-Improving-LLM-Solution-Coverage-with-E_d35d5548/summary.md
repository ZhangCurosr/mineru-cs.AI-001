---
title: "Beyond-the-Best-Guess-Improving-LLM-Solution-Coverage-with-E"
source: https://arxiv.org/pdf/2608.12679v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:19:50"
field: "大语言模型后训练方法"
keywords: ["Evolution Strategies", "Reinforcement Learning", "pass@k", "Test-Time Scaling", "Distribution Collapse", "LLM Post-training", "Solution Coverage"]
innovations: ["系统证明ES在pass@k上优于RL且优势跨模型规模一致", "揭示RL分布坍缩导致bin 0.0增加而ES减少此现象的机制", "提出准确率分布/回归进展/熵分析的组合评估框架"]
benchmarks: ["GSM8K", "MATH500", "Olympiad Bench", "Minerva"]
---

# 论文速读：Beyond-the-Best-Guess-Improving-LLM-Solution-Coverage-with-E

## 一句话总结
本文系统比较了进化策略（ES）与强化学习（RL）在大型语言模型后训练中的表现，发现ES能在提升pass@1精度的同时保持更广泛的答案空间覆盖，而RL因分布坍缩导致解覆盖率下降，在测试时扩展（TTS）场景下表现劣于ES。

## 研究问题与动机
- **RL后训练导致分布坍缩**：RLVR通过策略梯度优化使模型输出分布集中于高奖励答案，但会剪枝低概率但正确的解，导致pass@k性能随k增大而下降，甚至base model在足够大k时超越RL微调模型。
- **测试时计算扩展的重要性**：随着LLM用于数学、科学等发现领域，仅凭单次采样（pass@1）不足以评估模型价值，需要通过多次采样（pass@k）探索解空间，因此需要保持模型输出分布的广泛支持。
- **ES作为替代方案潜力**：ES是群体式的无梯度优化方法，直接在权重空间通过随机扰动优化，理论上能优化更鲁棒的解分布，但缺乏系统性对比验证。
- **缺乏全面基准评估**：现有研究未对ES在pass@k指标上的优势进行全面实证，特别是跨模型规模和家族的系统性比较。

## 核心贡献（创新点）
- **首次系统性评估ES的pass@k优势**：对1.5B至32B参数规模的多模型族进行ES后训练实验，证明ES在pass@k上始终优于RL，且该优势跨模型规模保持一致。
- **揭示ES与RL在解覆盖上的本质差异**：通过准确率分布、回归/进展分析和答案熵分析，揭示RL增加不可解题数量（bin 0.0）而ES减少此现象，ES同时提升新知识获取并减少知识丢失。
- **提供ES机制解释**：阐明ES通过在权重空间寻找平坦区域进行优化，相比RL在动作空间的梯度更新，能更好地保留base model的输出分布特性。
- **验证ES在测试时扩展中的下游价值**：通过自洽性投票实验证明ES生成的高质量输出分布在非可验证设置中同样优于RL分布。

## 方法详解
- **ES后训练框架**：ES采用群体式无梯度优化，对于参数$\theta$和奖励函数$R(\cdot)$，在每个优化步对群体中每个个体$n$采样扰动$\epsilon_n \sim \mathcal{N}(0, I)$，缩放后评估$r_n = R(\theta + \sigma\epsilon_n)$，通过归一化奖励$\hat{r}_n$更新权重：$\theta_t = \theta_{t-1} + \alpha \cdot \frac{1}{N}\sum_{n=1}^N \hat{r}_n \epsilon_n$。
- **pass@k评估方法**：使用Chen et al.提出的无偏低方差估计器：$\text{pass}@k = \mathbb{E}_{x \sim \mathcal{D}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$，其中$n$为每题采样总数，$c$为正确响应数，$k$为考虑采样的答案数量。
- **实验设置对比**：ES使用ES-at-Scale库（population size=32, σ=0.001, α/2学习率），RL使用VERL库（GRPO算法，clip ratio=0.2，KL系数=0.0），两组训练均在500步、batch size=512条件下进行公平比较。
- **分析指标设计**：除pass@k外，还设计准确率分布分析（统计各题在k次采样中正确率分布）、模型回归/进展度量（base model错但微调模型对的题目数vs base model对但微调模型错的题目数）、答案熵分析（Shannon熵衡量失败时答案多样性）。

## 实验与结果
- **GSM8K实验（1.5B-8B）**：在Qwen2.5-Instruct和Qwen3模型族上，ES与RL在k=2处交叉后ES持续领先，RL曲线更早饱和；Qwen2.5-Instruct的base model在足够大k时超越RL，但从未被ES超越。
- **MATH实验（7B-32B）**：ES在MATH500、Olympiad Bench和Minerva上均优于SOTA RL检查点（OatZero、SimpleRL-Zoo）；Qwen2.5-Math-32B在Minerva上优势最大且随k增大而扩大。
- **准确率分布分析**：RL相比base model增加bin 0.0（完全不可解题）的频率，ES则减少此频率；RL增益集中在bin 1.0（全对），ES增益更分散在(0.9, 1.0]区间。
- **回归/进展分析**：Qwen2.5-Math-7B在MATH500上ES有6个进展vs RL的3个，3个回归vs RL的17个；Olympiad Bench上56 vs 33进展，15 vs 49回归；Minerva上22 vs 16进展，6 vs 40回归。
- **熵分析**：ES失败时维持更高熵（median entropy右移），超过25%的RL回归熵接近0（ confidently incorrect），而ES失败时保持不确定性而非 confident错误。
- **自洽性投票实验**：随k增大，ES的投票准确率超越RL；两个benchmark上RL的最大概率答案甚至差于base model，表明RL分布质量退化。

## 相关工作脉络
- **RLVR后训练范式**：Shao et al. (2024)将RL与可验证奖励结合用于CoT推理，成为math/coding后训练标准方法；本文指出其在pass@k场景的局限性。
- **pass@k与测试时扩展**：Brown et al. (2024)和Snell et al. (2024)提出通过重复采样扩展测试时计算；本文聚焦此场景下后训练方法的选择。
- **RL分布坍缩问题**：Yue et al. (2025)首次报告base model在pass@k上超越RL微调模型；Wu et al. (2025)、Nguyen et al. (2026)分析坍缩机制；本文提出ES作为根本性解决方案而非正则化修正。
- **分布坍缩缓解方法**：PPO的KL惩罚、reset机制、VPO等多目标优化；本文认为这些方法引入额外超参且未根本解决策略梯度与分布多样性的张力。
- **ES在LLM中的应用**：Qiu et al. (2025)首次将ES应用于LLM微调，证明与RL性能相当；本文进一步揭示ES在解覆盖上的独特优势。
- **ES变体研究**：Sarkar et al. (2025)的hyperscale ES、Liang et al. (2026)的维度 blessing分析、Schweighofer et al. (2026)的遗忘问题研究；本文聚焦pass@k优化的理论动机与实证验证。

## 局限性与未来方向
- **仅评估数学推理领域**：实验集中在GSM8K和MATH benchmark，未验证在coding、科学发现等其他发现领域的应用效果。
- **ES超参敏感性未充分研究**：population size=32、σ=0.001等关键超参的选取基于前人工作，未系统分析其对pass@k的影响。
- **未优化pass@k直接作为目标**：当前ES优化期望奖励，未来可直接将pass@k或其代理指标作为优化目标。
- **仅测试自洽性投票**：TTS方法限于简单投票，未探索tree search、agentic harness等更复杂策略与ES的结合。
- **计算成本未详细比较**：ES需维护32个模型副本进行评估，实际训练效率与RL的对比未深入分析。

## 研究启发与可借鉴点
- **pass@k作为后训练评估指标**：在需要多样化输出的场景（如科学发现、代码生成），应将pass@k而非仅pass@1纳入模型评估体系，指导后训练方法选择。
- **分布分析的可迁移框架**：准确率分布、回归/进展度量、答案熵分析可推广至其他领域评估后训练方法的质量，不仅关注精度更关注分布健康度。
- **ES在无梯度优化中的优势**：对于需要保持模型多样性的任务（如多解问题、creative generation），ES的权重空间优化比RL的动作空间梯度更合适，值得探索于其他领域。
- **避免bin 0.0增多的设计原则**：后训练方法应确保不增加"完全不可解题"数量，这是分布质量的硬约束，可作为方法设计的评估准则。
- **与TTS方法的协同优化**：ES生成的优质分布可与tree search、agent harness等TTS方法结合，未来工作可探索联合优化训练目标与测试时策略。

## 关键术语表
**pass@k**：从模型输出分布中采样k个答案，至少有一个正确的概率，衡量模型在测试时扩展下的解覆盖率。
**Reinforcement Learning with Verifiable Rewards (RLVR)**：使用可验证奖励（如数学答案正确性）进行强化学习后训练，通过策略梯度优化高奖励输出的log概率。
**Distribution Collapse (分布坍缩)**：RL优化过程中输出分布向高奖励模式集中，导致低概率但正确的解被剪枝，解覆盖范围缩小的现象。
**Evolution Strategies (ES)**：无梯度优化方法，通过对权重空间施加随机高斯扰动生成群体，基于归一化奖励聚合更新模型参数。
**Test-Time Scaling (TTS)**：通过增加测试时计算（如多次采样、搜索、迭代）提升模型性能的策略，依赖模型输出分布的质量。
**Progressions / Regressions**：微调后模型相对于base model新获得正确解答的题目数（progressions）和丧失正确解答能力的题目数（regressions）。
**Self-Consistency Voting**：生成多个独立答案并通过多数投票选择最终答案的测试时扩展方法，适用于不可验证场景。
**Answer Entropy**：对单题生成的答案分布计算Shannon熵，衡量模型回答的多样性与不确定性程度。

## 可复现要素
- **数据集**：GSM8K、MATH (level 3-5)、MATH500、Olympiad Bench、Minerva（公开数据集）
- **模型**：Qwen2.5-Instruct (1.5B/3B/7B)、Qwen3 (1.7B/4B/8B)、Qwen2.5-Math (7B/14B/32B)
- **代码**：ES-at-Scale库（ES训练）、VERL库（RL训练）
- **权重**：ES训练权重未明确说明开源状态；RL基线来自SimpleRL-Zoo和OatZero
- **超参**：ES: σ=0.001, α/2学习率, population size=32, batch size=512/1024, 500 steps；RL: GRPO, clip=0.2, KL=0.0, batch=512, rollout=1024, 500 steps
- **硬件**：GSM8K实验用8×H200，MATH实验用8×B200

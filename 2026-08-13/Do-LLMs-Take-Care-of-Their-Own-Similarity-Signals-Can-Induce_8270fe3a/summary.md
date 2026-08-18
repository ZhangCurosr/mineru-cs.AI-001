---
title: "Do-LLMs-Take-Care-of-Their-Own-Similarity-Signals-Can-Induce"
source: https://arxiv.org/pdf/2608.12125v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 10:24:43"
field: "合作型 AI / 多智能体博弈"
keywords: ["LLM cooperation", "similarity signaling", "b-similarity equilibrium", "evidential decision theory", "multi-agent game theory", "CoT analysis", "CoopEval"]
innovations: ["提出b-相似度均衡作为Nash与EDT间的连续插值框架，证明高相似度下可恢复社会福利最优", "构建首个端到端的LLM相似度信号评测框架，区分外生/内生信号并系统比较10个基准域", "发现LLM对相似度信号来源域不敏感（包括随机噪声也能诱导合作），揭示信号可靠性风险"]
benchmarks: ["Prisoner's Dilemma", "Public Goods Game", "Traveler's Dilemma", "Stag Hunt", "Chicken", "HLE", "TRAIT", "Moral Choice", "Newcomb-like Problems", "Greatest Good Benchmark"]
---

# 论文速读：Do-LLMs-Take-Care-of-Their-Own-Similarity-Signals-Can-Induce

## 一句话总结
本文构建了首个评估 LLM 智能体在接收到**分级相似度信号**后如何进行合作决策的框架，发现高相似度信号通常能促进合作，并提出了一个连接 Nash 均衡与证据决策理论（EDT）的 **b-相似度均衡** 行为博弈论模型。

## 研究问题与动机
1. **核心问题**：当 LLM 智能体知道对手与其具有不同程度的决策相似性时，是否会表现出更多合作行为？如何通过实验隔离并量化这种"相似性"对 pairwise 合作的影响？
2. **现实背景**：越来越多 AI 智能体在同一生态系统中部署（如同为 Claude/GPT 系列），它们可能共享训练数据、模型架构或设计模式，导致决策行为高度趋同，传统"独立决策"假设在 AI 经济中不再成立。
3. **理论动机**：经典博弈论中的 Nash 均衡假设独立个体进行单侧偏离分析；但 Evidential Decision Theory（EDT）和超理性（Superrationality）认为，当个体意识到自己与对手遵循相同推理模式时，应将这种相关性纳入决策——这在人类场景中常被视为哲学好奇，但在 AI 同质化部署中却十分合理。
4. **已有工作不足**：Oesterheld et al. (2023) 研究了可信差异信号下的合作，但相似性信号的构造是预设的，缺乏对 grounding（信号来源）和操作化机制的系统研究。

## 核心贡献（创新点）
1. **首个 LLM 相似度信号评估框架**：覆盖 9 个 LLM、5 种混合动机博弈、10 个基准测试（7+3），系统研究不同相似度信息下的战略决策，此前无类似全面评测。
2. **b-相似度均衡（b-similarity equilibrium）理论框架**：提出一个简单的标量形式化模型，在 Nash 均衡（b=0，独立偏离）与 EDT/Kantian 均衡（b=1，完全复制）之间建立连续插值；证明高相似度下均衡可近似恢复社会福利最优（Theorem 1）。
3. **自顶向下的操作化路径**：首次将相似性信号"落地"到真实可计算的基准行为上（7+3 个基准），区分外生（exogenous）与内生（endogenous）两种信号计算方式，揭示信号来源域对合作几乎无显著影响这一反直觉发现。
4. **与现有合作的机制比较定位**：在 CoopEval 排行榜上，基于行为基准的外生相似度机制位列第二（仅次于最优），内生相似度机制在最佳条件下可跃升至第一。

## 方法详解
**实验框架**：LLM 智能体在两方/多方对称博弈（囚徒困境、公共物品、旅行者困境、猎鹿、斗鸡等）中参与单轮博弈，输入包含标准游戏规则 + 相似度信号提示（如"对手决策方式与你 X% 相似"），输出混合策略概率分布（JSON 格式）。

**b-相似度均衡定义**：对对称博弈 G，相似度矩阵 $\pmb{b} \in [0,1]^{\mathcal{N}\times\mathcal{N}}$，对称策略配置 $\pmb{s}=(s,...,s)$ 是 b-相似度均衡当且仅当对每个玩家 $i$ 和替代策略 $s'$：
$$u_i(s) \geq u_i\big(s', \sigma_{-i}(s, s', \pmb{b}_i)\big)$$
其中 $\sigma(s, s', b_{ij}) = b_{ij}s' + (1-b_{ij})s$ 表示对手以概率 $b_{ij}$ 跟随偏离、以剩余概率保持原策略的混合。

**关键性质**：
- **b=0 ⇒ Nash 均衡**（Lemma 2）
- **b=1 ⇒ 全局最优对称策略**（Proposition 3），即 EDT/Kantian 风格
- **高相似度福利上界**（Theorem 1）：对于同质相似度 $b$，福利损失不超过 $\sum_i R_i(1-b^{n-1})$，在囚徒困境中 $b > 1/2$ 即可支持全合作作为唯一均衡。

**相似度计算方式**：
- **外生（exogenous）**：基于基准答案匹配率（或 QWK 等）直接计算；例如 HLE、TRAIT、Moral、Newcomb 基准。
- **内生（endogenous）**：由判读模型（Judge LLM）查看对手模型在基准上的最终答案和/或 CoT 推理，自主给出 0-100 相似度评分，再注入博弈提示。

**CoT 分析框架**：使用 Gemini 3.1 Flash Lite Preview 作为 Judge，以 17 类预定义范畴（个体效用最大化、超理性、社会福祉最大化、因果独立/相关决策等）标注每个模型的推理链。

## 实验与结果
- **模型**：Gemini 3 Flash, GPT 5.4 mini, Claude Haiku 4.5, Grok 4.20, DeepSeek V4 Pro, Kimi K2.6, Gemma 4 31B, Qwen 3.5 27B, GPT-4o
- **采样**：每模型每条件 10 次，报告均值 ± 标准误

**RQ1 主要发现（囚徒困境，Figure 2）**：
- GPT 5.4 mini：**完全不受相似度影响**，在所有水平上均选择背叛（defect）
- Claude Haiku 4.5：**非单调**，80% 相似度时合作率达 70%，再升高后回落到 0%
- GPT-4o：在全部相似度水平上随机化（合作率 49%-63%）
- 其余 6 个模型（Gemini, DeepSeek, Kimi, Gemma, Grok, Qwen）：单调递增，0%→100% 相似度下从完全背叛过渡到完全合作，过渡阈值在 60%-80% 间

**RQ2 变体实验**：
- **报酬结构变化**：提高合作收益使转换阈值降低（如高合作收益下 10%-30% 相似度即可实现近乎全合作）；序数（ordinal）偏好下行为定性一致
- **推理努力**：高推理强度下 DeepSeek 等模型的转换更加锐利，与理论模型预测一致
- **相似度框架**：将提示从"相似（similar）"改为"不同（different/dissimilar）"后，DeepSeek 合作率下降，Claude 完全转向背叛
- **多方博弈（PublicGood）**：合作难度显著上升，Gemini 是唯一在 ≥60% 相似度时仍维持较高合作的模型
- **猎鹿/斗鸡**：猎鹿中低相似度信号起到"避免冒险"作用；斗鸡中所有相似度均推动 toward 合作（chicken out）

**RQ4 基准域的影响（Figure 3）**：
- **惊讶发现**：7+3 个基准域对诱导合作率的差异**极小**，模型普遍不区分信号来源的"相关性"
- 仅 DeepSeek 和 Gemma 能识别 Random Die/Coin 基准应视为随机噪声

**RQ5/RQ6 外生 vs 内生信号（Table 2）**：
- 外生相似度： Moral/TRAIT 基准诱导最高合作（约恢复 72% 最优社会福利）；HLE 基准下几乎无合作
- 内生相似度（决策+解释）：回收约 55%-73% 最优福利；**仅展示解释（explanation-only）时合作最高**（TRAIT 基准达 80%）
- 内/外生相似度在数值上差异显著：模型自评相似度往往高于客观答案匹配率（HLE 外生仅 ~1 分 vs 内生 1.6 分）

## 相关工作脉络
1. **CoopEval（Tewolde et al., 2026）**：评测多种合作机制（声誉、重复、中介、契约等）在 LLM 上的表现，本文将该框架扩展至相似度信号机制，并与之对照排名。
2. **Oesterheld et al. (2023)**：提出"相似性基础合作均衡"概念，本文扩展其抽象信号为可操作化的端到端实验流程。
3. **Meulemans et al. (2026)**：在嵌入式 AI 框架下用 Bayesian 推理实现相似性推断，但需完整交互历史；本文通过分离信号构造与游戏阶段，仅暴露标量分数，更简洁实用。
4. **Monderer & Tennenholtz (2009)，Kalai et al. (2010)**：中介（Mediation）合作机制，依赖第三方协调；本文的相似度信号属于信息型而非制度型机制。
5. **Ahmed (2021)，Roemer (2010)**：证据决策理论（EDT）和 Kantian 均衡的哲学/经济学基础，本文 b-相似度均衡在其之间建立了可参数的连续插值。
6. **Huang et al. (2025)**：Values in the Wild（VITW）数据集，本文以其五大价值类别为锚点选取评测基准，确保相似性信号覆盖多元领域。

## 局限性与未来方向
1. **信号可靠性风险**：模型对相似性信号的来源域不敏感，甚至对随机噪声也产生合作反应，存在被"操纵性标签"利用的风险。
2. **单方合作倾向**：部分模型（如 GPT）完全遵循 Causal Decision Theory 框架，将对手视为完全独立决策者，导致相似度信号完全失效。
3. **单轮博弈局限**：当前仅评估一次性（single-shot）博弈，未考虑重复互动、时序博弈或情境化环境中的动态相似性演化。
4. **未来方向**：作者呼吁研究（a）时序/上下文博弈中的相似性信号；（b）随时间变化的 LLM 行为对相似性感知的影响；（c）该行为模型是否能推广到更广泛的人类-感知相似性场景。
5. **伦理风险**：同类智能体间的高相似性可能引发"合谋"（collusion）或对用户/第三方的系统性风险，需要部署时的安全约束。

## 研究启发与可借鉴点
1. **b-相似度均衡的形式化范式可迁移**：该标量插值模型（0=Nash，1=EDT）简洁且理论上优美，可作为后续研究 LLM 在相关性推理中的基线模型，应用于更广泛的博弈设定和多主体协作问题。
2. **CoT 作为可审计的合作机制**：内生相似度中"仅展示推理链"比"仅展示答案"更能激发合作（TRAIT 基准 80% vs 40%），说明 reasoning trace 本身蕴含丰富的合作倾向信号，可在系统设计中有意识地暴露 CoT 以促进协调。
3. **提示工程的精细迭代方法**：论文展示了相似度提示词从宽泛到精确的多轮迭代过程（先发现三种矛盾解读，再锚定到"决策推理过程"），这一提示工程方法论值得在多 agent 评估中借鉴。
4. **多模型、多基准的穷举式评测策略**：通过 9 模型 × 5 博弈 × 10 基准的组合矩阵建立全面画像，可识别出异常行为模型（GPT/Claude），为后续选择性研究提供候选集。
5. **VITW 价值分类作为基准筛选框架**：以 Values in the Wild 的五类价值观作为锚点来覆盖多样性评测域，是一个系统化选型策略，可复用于其他 LLM 行为评估场景。

## 关键术语表
**b-相似度均衡（b-similarity equilibrium）**：一种博弈论解概念，玩家假设对手以概率 $b_{ij}$ 跟随自己的偏离策略，在 $b=0$ 时退化为 Nash 均衡，$b=1$ 时退化为 EDT/Kantian 均衡。

**证据决策理论（Evidential Decision Theory, EDT）**：与因果决策理论相对，主张在决策时考虑自身行动与其他事件之间的统计相关性，而非仅考虑因果影响；本文将其形式化为 $b=1$ 的极端情况。

**内生相似度（Endogenous similarity）**：由参与博弈的 LLM 智能体自身根据对手的响应/推理链自主评定的相似度分数，而非外部计算给出。

**外生相似度（Exogenous similarity）**：由研究者基于预先定义的公式（如答案匹配率）从基准测试中客观计算并注入给 LLM 的相似度分数。

**Chain-of-Thought（CoT）分析**：通过 LLM-as-a-judge 框架，对模型的推理链进行范畴分类（共 17 类），以量化其决策理由。

**超理性（Superrationality）**：Hofstadter 提出的概念，指多个完全相同的理性决策者预期会得到相同结论，从而据此选择全局最优策略。

**Values in the Wild（VITW）**：Huang et al. (2025) 从数十万 Claude 对话中提取的五大价值类别分类体系（实用、认知、社会、保护、个人），本文用于指导基准测试的选择。

**相似性框架（Similarity Framing）**：提示中关于相似性的措辞设计，包括"相似/不同/不相似"以及百分比呈现方式，对模型合作行为有显著影响。

## 可复现要素
- **代码/框架**：实验基于开源的 **CoopEval** 框架（Tewolde et al. 2026）扩展，论文声明为 open-source evaluation framework（具体 GitHub 链接论文未明示，需在附录/项目页查找）
- **模型访问**：所有 LLM 通过 **OpenRouter** API 调用；温度参数设为 1
- **基准数据集**：使用了现有公开基准（HLE、TRAIT、Newcomb、GGB、Moral Choice、DailyDilemmas、CABIN）及自定义的 Random Die/Coin、Similarity Game
- **关键超参**：每模型每条件 10 次采样；benchmark 抽取各 150 题进行相似度计算
- **论文未提及**：具体 GPU/推理硬件配置；代码仓库 URL；训练后的权重文件

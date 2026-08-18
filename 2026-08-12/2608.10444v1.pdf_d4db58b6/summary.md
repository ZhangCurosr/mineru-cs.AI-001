---
title: "From Reasoning Depth to Reasoning Breadth: Evaluating Multi-Point Associative Reasoning in Large Language Models"
source: https://arxiv.org/pdf/2608.10444v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:20:36"
---

# 论文速读：From Reasoning Depth to Reasoning Breadth: Evaluating Multi-Point Associative Reasoning in Large Language Models

## 一句话总结
本文提出双语评测基准 **MPAR-Bench**，通过多源异构线索聚合任务系统性隔离并评估大语言模型的“推理广度”（多点联想推理）；实验表明当前前沿模型的能力仍以线性推理深度为主，开启思维模式虽提升标准设定准确率，却无法一致增强抗扰动鲁棒性，且易因过度思考（overthinking）推翻初始正确假设。

## 研究问题与动机
1. **现有基准严重偏重推理深度**：主流 benchmark（如 MATH、MMLU、Beyond the Imitation Game 等）主要评估单条推理链的逐步推导与程序性推理，衡量模型能沿一条路径推多远，却未回答模型能否跨多条推理链聚合证据。
2. **真实应用场景需要广度能力**：多文档综合、跨域类比、假设生成、在残缺或含干扰证据下推断等任务，要求模型同时持有多条部分关系并将其收敛为单一预测，而非仅依赖单一路径匹配。
3. **传统联想测试存在数据污染风险**：如 RAT（Remote Associates Test）等公开题库多为固定三线索搭配，LLM 极易通过预训练共现记忆匹配答案，难以区分真实推理与表面统计规律。
4. **推理深度扩展与广度鲁棒性未必正相关**：尽管 CoT、强化学习思维模式能拉长推理轨迹，但其在信息缺失、顺序扰动、干扰注入等现实条件下是否稳健，目前缺乏系统测量。

## 核心贡献（创新点）
1. **提出 MPAR-Bench 广度推理基准**：受合作桌游 *Just One* 规则启发，要求模型从开放数量的自由形式语义线索中整合出唯一目标词；与 RAT 固定三线索及 Codenames/Connections 等游戏基准的“一给多”或“固定词表分组”范式形成本质区别，首次将“猜词者侧的多对一整合”与可控扰动套件联合评测。
2. **设计多代理线索合成流水线**：引入多 agent 按不同联想角度迭代生成线索，结合基于 embedding 的冗余过滤与人工验证，显著降低预训练污染风险，同时保障线索集在词汇、文化、语音与世界知识层面的语义多样性。
3. **构建粗到精评估协议**：除精确匹配外，融合 ANLS、fastText embedding 相似度与推理轨迹验证（事实核查+逻辑核查），并针对线索掩码、顺序打乱、干扰注入、多步推理四种扰动轴分别分析鲁棒性，填补广度维度评估空白。
4. **揭示“深度≠广度”的关键发现**：通过超思辨分析、信息增益曲线与 scaling law 实验，实证表明思维模式延长虽提升标准设定准确率（尤其英文），但无法一致降低扰动敏感性，且长程推理可能主动覆盖正确中间假设。

## 方法详解
- **任务定义**：给定线索集 $C = \{c_1, c_2, ..., c_n\}$，模型需整合各线索提供的独立语义关系，预测目标词 $y$。推理广度体现为对非冗余多源信号的聚合与概念收敛能力。
- **数据集构建**：
  - **答案空间**：取自公开词表（RAT 衍生词汇与 *Just One* 游戏卡牌）。
  - **多代理生成**：多个 agent 分配不同联想角度（lexical/cultural/phonetic/world-knowledge 等）迭代生成候选线索；judge agent 过滤答案本身、直接同义词/翻译/谐音/形态变体、精确或近重复线索及低质量条目。
  - **Embedding 过滤**：使用 Qwen3-Embedding-8B 计算线索-答案、线索-线索相似度，阈值区间 0.3–0.8 作为初筛，剔除与答案过于接近或互为重复的线索。
  - **答案唯一性保障**：构造阶段 judge 过滤共现下定条件不足的条目；人工随机抽检 250 题（两位 NLP 硕士独立评估），92.8% 被判定为唯一答案（95% Wilson CI）。
  - **双语设计**：英/中各 500 题，中文子集额外融入成语、字形/字面属性及当代文化梗。
- **增强设定与扰动**：
  - **Clue Masking**：随机掩码部分线索，模拟信息缺失。
  - **Order Shuffling**：打乱线索顺序，检验模型对输入排列的敏感性。
  - **Distractor Injection**：注入语义误导或无关词，测试抗噪声与虚假关联能力。
  - **Multi-step Inferring**：增加线索与目标间的语义距离，迫使生成中间隐含连接。
- **评估协议**：
  - **Accuracy**：预测与 ground truth 完全一致。
  - **ANLS**：$\mathrm{ANLS}(\hat{y}, y) = 1 - \frac{d_{\mathrm{lev}}(\hat{y}, y)}{\max(|\hat{y}|, |y|)}$，衡量字面编辑距离。
  - **Embedding Similarity**：$\mathrm{Sim}(\hat{y}_{emb}, y_{emb}) = \frac{\hat{y}_{emb}^\top y_{emb}}{\|\hat{y}_{emb}\|_2 \|y_{emb}\|_2}$，基于 fastText 计算余弦相似度。
  - **Reasoning Trace Verification**：将 CoT 分解为独立原子步骤，分别进行 Factual Accuracy 与 Logical Soundness 核查；人工-LM 判断一致性达 98.7%（事实）/94.7%（逻辑）。

## 实验与结果
- **评测模型**：GPT-5.2、Gemini-3.1pro/flash、Sonnet-4.5、Qwen3-max、Kimi-k2、Deepseek-v3.2、Seed-2-pro，以及自部署 Qwen3 家族（0.6B–32B）。覆盖 thinking 与 non-thinking 模式，未做 per-model 超参调优。
- **标准设置表现**：Thinking 模式下 Gemini-3.1pro 领跑，英文准确率 86.8%，中文 72.2%；GPT-5.2 英文 77.6%、中文 64.4%。Non-thinking 模式下 Sonnet-4.5 英文 70.4%、中文 68.8% 最优。
- **扰动鲁棒性**：所有模型在 Enhanced 设定下均下降。Thinking 模式下，英文扰动平均降幅 9–18 分，中文 5–12 分。Clue Masking 对英文伤害最大（平均较 Order Shuffling 下降约 20.0%）；Distractor Injection 对 Qwen3-max 与 Seed-

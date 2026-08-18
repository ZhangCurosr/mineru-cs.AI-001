---
title: "Are-You-Sure-You-re-Sure-On-the-Impact-of-Instruction-Tuning"
source: https://arxiv.org/pdf/2608.13430v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:49:49"
field: "大语言模型评估与可靠性"
keywords: ["instruction tuning", "model confidence", "lexical diversity", "calibration", "rationale generation", "uncertainty estimation"]
innovations: ["首次系统研究指令微调对置信度和理由词汇多样性的联合影响", "提出受控对比设计以隔离答案选择和生成长度对多样性的混杂效应", "揭示跨理由多样性稳定下降与表面词汇多样性异质性变化并存的新现象"]
benchmarks: ["ARC-Easy", "MMLU", "CommonsenseQA"]
---

# 论文速读：Are You Sure You're Sure? On the Impact of Instruction Tuning on Confidence and Lexical Diversity

## 一句话总结
本文系统研究了指令微调（instruction tuning）对多选题问答中模型置信度和答案理由（rationale）词汇多样性的影响，发现指令微调一致性地提升了模型置信度（降低答案熵、提高口头置信度），但未带来预测准确率的相应提升，同时导致跨理由交叉多样性稳定下降，而表面词汇多样性变化则呈现模型和基准依赖的异质性。

## 研究问题与动机
1. **高可靠性场景下的置信度评估需求**：LLM正被广泛应用于医学、金融、法律等领域，这些场景中模型能否可靠估计自身不确定性至关重要。
2. **指令微调与置信度失调的已知矛盾**：已有研究表明指令微调/对齐训练会改变模型置信度分布，出现"口头过度自信"（verbalized overconfidence）现象，且准确性改善与置信度提升并不匹配。
3. **置信度变化与理由多样性的潜在关联未被探索**：在自由形式生成中，语义等价的回答可能使用完全不同的词表序列，但现有工作尚未研究置信度变化是否伴随理由词汇多样性的系统性变化。
4. **多样性与校准关系的空白**：尽管有大量关于输出多样性、模型置信度和校准的独立研究，但词汇/token级多样性与口头置信度或校准之间的关联从未被系统调查。

## 核心贡献（创新点）
1. **首次配对评估基础模型与指令微调模型在置信度、校准和理由多样性上的系统性对比**：覆盖3个模型家族（Qwen2.5-7B、Mistral-7B、Llama-3.1-8B）和3个推理基准（ARC-Easy、MMLU、CSQA），填补了该领域的研究空白。
2. **引入受控对比设计以隔离多样性变化的独立效应**：通过在"base和instruct模型选择相同答案"且"理由长度匹配"的子集上进行比较，证明多样性变化并非由答案切换或生成长度差异驱动。
3. **揭示置信度提升与多样性变化之间的异质性关系**：证明指令微调后置信度上升并不伴随一致性多样性下降，且两种多样性度量（Unique-2 vs 1-SelfBLEU）对同一模型呈现相反趋势，为不确定性估计研究提供了新的分解视角。

## 方法详解
1. **置信度评估**：
   - **答案熵（choice entropy）**：计算归一化候选答案概率分布的熵 $H_{\mathrm{choice}}(x) = -\frac{\sum_{j=1}^{M} p_j \log p_j}{\log M}$，值越高表示不确定度越大。
   - **口头置信度（verbalized confidence）**：两阶段协议——第一阶段通过最大似然确定答案，第二阶段提示模型输出该答案正确的数值概率（0~1）。
2. **词汇多样性度量**：
   - **Unique-2（U2）**：独特bigram比例，衡量表面词汇丰富度。
   - **1-SelfBLEU（1-SB）**：同一问题生成的多个理由之间的self-BLEU相似度的补数，衡量跨理由交叉多样性（cross-rationale diversity）。
3. **理由生成设置**：对每个问题使用zero-shot chain-of-thought提示"Answer: Let's think step by step."，采样 $K=5$ 个理由，温度 $T=0.7$，nucleus sampling $p=1.0$，最大100新token。
4. **受控分析（controlled analysis）**：筛选base和instruct模型选择相同答案的问题，按token长度配对理由并截断至较短者，隔离纯语言多样性差异。
5. **校准评估**：使用Expected Calibration Error (ECE)，将预测按置信度分箱计算准确率与平均置信度的加权偏差。

## 实验与结果
1. **数据集与基线**：使用ARC-Easy（小学科学）、MMLU（多学科）、CSQA（常识知识）三个英语多选题基准；对比三对base/instruct模型。
2. **置信度一致上升**：所有模型在所有基准上，答案熵显著下降（如Qwen在MMLU上从0.430降至0.131，Mistral在CSQA上从0.736降至0.268），口头置信度显著上升（如Llama在ARC-Easy上从49.2%升至90.4%，在CSQA上从46.8%升至91.1%）。
3. **准确率变化不一致**：Llama在ARC-Easy上准确率保持82.2%不变，与置信度大幅提升形成鲜明反差。
4. **多样性变化异质性**：
   - **1-SelfBLEU**（跨理由多样性）在所有模型和基准上均稳定下降（Mistral在ARC-Easy上从0.813降至0.626）。
   - **Unique-2**（表面多样性）变化方向不一致：Mistral在ARC-Easy/MMLU上下降，但在CSQA上从0.719升至0.750。
5. **不确定性-多样性方向分析**：表2/6显示，不确定性下降与多样性变化无一致对应关系——以Mistral为例，Lower不确定性+Higher U2占比61.8%，但Lower不确定性+Lower 1-SB占比77.6%。
6. **受控分析结论（Table 3）**：在相同答案+匹配长度条件下，Qwen的U2几乎不变（-0.001）而1-SB下降（-0.036）；Mistral和Llama的U2显著上升（+0.050/+0.053）而1-SB下降（-0.069/-0.012），差异依然显著。
7. **校准与多样性无关**：多样性变化与ECE变化无一致关联——如Qwen在ARC-Easy上口头ECE从35.3%降至22.8%时，两个多样性指标均下降；而Llama在MMLU上likelihood ECE从0.5%升至5.9%时，两个多样性指标也下降。

## 相关工作脉络
1. **置信度与校准估计（Jiang et al., 2021; Geng et al., 2024）**：本文继承了基于答案概率和口头置信度的评估框架，但创新性地将其与理由多样性关联，而非仅关注校准本身。
2. **指令微调对置信度的影响（Kadavath et al., 2022; Tian et al., 2023; Xiong et al., 2024; Zhang et al., 2024）**：已有工作发现post-training（尤其是RLHF）会导致口头过度自信，本文在此基础上进一步揭示了这一现象与多样性变化的非单调关系。
3. **生成多样性评估（Alihosseini et al., 2019; Post, 2018）**：本文采用Unique-2和Self-BLEU作为多样性度量，但这些指标此前主要用于文本生成质量评估，本文首次将其应用于QA理由分析。
4. **语义熵与不确定性（Farquhar et al., 2024）**：语义熵通过多次采样评估模型不确定性，本文与其互补——不仅关注不确定性本身，还研究其伴随的词汇层面变化。
5. **后训练对语言多样性的影响（Guo et al., 2025; Yun et al., 2025; Deshpande et al., 2025）**：近期研究发现格式约束等后训练可能引发"多样性坍缩"，本文从词汇角度提供了类似的实证证据。
6. **calibration-tuning（Kapoor et al., 2024; Xie et al., 2024）**：校准调优工作试图改善模型置信度校准，本文暗示此类方法可能进一步压缩多样性，为trade-off分析提供依据。

## 局限性与未来方向
1. **基准和语言的局限**：仅评估了三个英语多选题基准，未涵盖开放式生成、中文等非英语场景，也未涉及不安全数据或人口统计学敏感提示。
2. **多样性度量的局限**：仅考察了词表层面的Lexical Diversity（Unique-2和Self-BLEU），未评估语义多样性（semantic diversity）和句法多样性（syntactic diversity）。
3. **置信度估计方法的局限**：仅使用了答案熵和口头置信度两种代理，未比较其他不确定性估计方法（如bootstrap、MC dropout等）。
4. **未来方向**：扩展至语义/句法多样性分析；验证跨语言泛化；研究校准优化是否会以牺牲多样性为代价；探索生成任务中多样性变化的下游影响。

## 研究启发与可借鉴点
1. **配对评估设计的参考价值**：使用matched base-instruct模型对在多个基准上进行对比，可有效隔离指令微调本身的效应，避免模型规模差异的混淆，该设计可直接迁移到任何post-training影响分析。
2. **受控多样性对比的范式**：通过筛选"相同答案+匹配长度"子集来隔离纯语言多样性变化，这一控制变量思路对后续研究多样性-性能trade-off具有重要方法论启示。
3. **双重多样性度量（表面vs跨理由）的发现**：Unique-2和Self-BLEU呈现相反趋势的发现提示：单一多样性指标可能掩盖重要信息，未来研究应综合多维度多样性评估。
4. **置信度-多样性解耦的警示**：高置信度不等于高质量/多样化推理，这为构建"谨慎型"LLM应用（如医疗、法律决策支持）提供了实证依据——不能仅凭口头置信度判断模型可靠性。
5. **可结合的创新机会**：可将本文方法与语义熵（semantic entropy）结合，研究指令微调是否同时影响词汇层面和语义层面的多样性；也可探索calibration-aware decoding策略对跨理由一致性的影响。

## 关键术语表
**Instruction Tuning（指令微调）**：在预训练基础上使用自然语言指令-响应对进行微调，使模型能够遵循用户指令完成各类任务。
**Choice Entropy（答案熵）**：基于候选答案归一化概率分布计算的熵，用于量化模型在多选项中的不确定程度。
**Verbalized Confidence（口头置信度）**：通过提示模型直接输出数值概率来 elicited 的置信度估计，区别于基于log-probability的隐式置信度。
**Unique-2（Unique Bigram Ratio）**：生成文本中独特bigram占总bigram的比例，衡量表面词汇丰富度。
**Self-BLEU**：同一输入生成的多条文本彼此之间的BLEU相似度，用于衡量跨采样输出的多样性；本文使用1-SelfBLEU作为多样性指标。
**Calibration（校准）**：模型预测置信度与实际准确率之间的一致性程度，通常用Expected Calibration Error (ECE)衡量。
**Rationale（理由）**：模型生成的支持其最终答案的推理步骤文本，本文通过chain-of-thought prompting获得。
**Cross-rationale Diversity（跨理由多样性）**：同一问题多次生成理由之间的差异程度，反映模型在相同答案下论证表达的多样性。

## 可复现要素
- **数据集**：ARC-Easy、MMLU、CommonsenseQA（均为公开基准）。
- **代码/权重**：模型权重均来自HuggingFace公开链接（Qwen2.5-7B、Llama-3.1-8B、Mistral-7B-v0.3及其Instruct版本），代码实现基于LM Evaluation Harness和SacreBLEU库，附录提供了完整prompt和生成参数。
- **关键超参**：温度T=0.7，nucleus sampling p=1.0，最大新token数100，每问题采样K=5个理由。
- **统计检验**：使用配对t检验（two-sided paired t-test），显著性阈值p<0.01。

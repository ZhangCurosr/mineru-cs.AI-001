---
title: "Falsehood-and-Impossibility-Are-Diferent-Directions-in-an-AI"
source: https://arxiv.org/pdf/2608.12852v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:28:13"
field: "LLM 可解释性/机制可解释性"
keywords: ["mechanistic interpretability", "truth probing", "impossibility", "sparse autoencoders", "linear probes", "Gemma 3"]
innovations: ["揭示真值方向与不可能方向在 transformer 残差流中近乎正交，偶假与必然假非同一轴的两端", "构建匹配模态刺激集验证双解离：输出混淆两类假而内部表征分离", "SAE 特征 firing profile 印证探针几何，表明不可能性为分布式表征而非单 neuron"]
benchmarks: ["Philosophical set (85 prompts)", "Modality set (75 prompts, 15 topic families × 5 conditions)"]
---

# 论文速读：Falsehood-and-Impossibility-Are-Diferent-Directions-in-an-AI

## 一句话总结
本文通过激活探针与稀疏自编码器（SAE）对 Gemma 3 4B IT 进行探索性研究，发现该模型在语言输出中将偶假（contingent falsehood）与逻辑矛盾混为一谈，但其内部表征却将"必然假"与"偶假"区分于正交方向，且必然假的表征位置更接近语义异常而非偶假，为经典哲学中"虚假"与"不可能"的区分提供了实证注脚。

## 研究问题与动机
1. **核心问题**：AI 模型是否在内部表征层面区分"偶假陈述"（empirically false but possible）与"必然假/不可能陈述"（logically impossible），抑或仅将其视为同一轴上的极端点？
2. **动机一**：模型输出可以流畅且具有说服力地产生虚假、无根据或内部不一致的内容，而读者难以辨别——理解模型内部如何处理此类失败对评估其影响至关重要。
3. **动机二**：先前工作已证明事实陈述的真值可从残差流线性解码（AUC~0.93），但"不可能性"是否独立于"真值轴"尚无任何实证研究。
4. **动机三**：哲学传统（亚里士多德非矛盾律、维特根斯坦《逻辑哲学论》、乔姆斯基"colorless green ideas"）早已指出语言中存在多种不同的失败类型，但尚未有工作以可操作化刺激对此在 transformer 中展开检验。

## 核心贡献（创新点）
1. **首次将"不可能性"作为独立表征维度进行实验检验**：构建涵盖 17 个哲学家族与 15 个主题族系的匹配刺激集，并同步施加真值、不可能性与异常三类探针，证明了三个方向的几何分离。
2. **揭示语言输出与内部表征的双解离（double dissociation）**：模型在词表层面将偶假与矛盾归为同一类别（12/15 标记为"contradiction"），但在激活空间中真值方向与不可能方向近乎正交（cosine ≤ 0.12）。
3. **证明必然假的表征邻居是语义异常而非偶假**：不可能探针在保持主题族系 held-out 时达到 AUC 1.00、balanced accuracy 0.97（Bonferroni 校正 P = 0.018），表明必要假命题并非偶假轴的极端。
4. **SAE 特征分布印证探针几何**：Layer 15 的 Gemma Scope 2 稀疏自编码器特征中，对不可能性最敏感的特征在异常和矛盾刺激上高频触发，但在偶假刺激上极少触发，确认该方向分布式而非单神经元编码。

## 方法详解
1. **刺激集设计**：
   - **哲学集**：17 个种子案例（直接矛盾、定义性不可能、说谎者悖论、堆垛悖论、语义异常等），每个家族含 1 个 canonical、3 个表面变换与 1 个相干控制，共 85 提示。
   - **模态集**：15 个主题族系（地理、物理、生物、日常算术、亲属、制度），每族系 5 种条件（common truth、contingent falsehood、improbable claim、semantically anomalous、necessary falsehood），共 75 提示。两集合刺激互不相交。
2. **模型与激活提取**：Gemma 3 4B IT（34 层 transformer，residual width 2560），bfloat16，PyTorch MPS 后端。每个提示嵌入固定指令要求用 4 类标签之一分类并解释，greedy 解码至多 48 token。提取最终 prompt token 处各层残差流状态（160 × 35 × 2560，float16）。
3. **三类线性探针**：L2 正则 logistic regression（C = 0.1，class-balanced），5-fold 分组交叉验证（整族 held-out）。
   - 真值探针（truth probe）：false vs. true
   - 不可能探针（impossibility probe）：impossible vs. true + false + improbable
   - 异常探针（anomaly probe）：anomalous vs. true + false + improbable
   - 报告指标：mean balanced accuracy；跨对比转移用 AUC 衡量。
4. **几何分析**：各层拟合真值与不可能方向，计算绝对 cosine（depth > 10 时 ≤ 0.12）；置换检验 1999 次，Bonferroni 校正 35 层深度。
5. **表面基线**：TF–IDF（词 1–2-gram、字符 3–5-gram）在同一 fold 内拟合，最强 surface baseline AUC = 0.77。
6. **SAE 分析**：Gemma Scope 2 residual stream JumpReLU SAEs，layer 15，small checkpoint（16k dict，~16 active features/prompt）与 big checkpoint（~90 active features/prompt）。按 log(1+z) 均值的标准化差排序，报告 firing prevalence。

## 实验与结果
1. **语言输出行为**：哲学集 4 标签 exact accuracy 55.3%，频繁将异质案例标记为"paradox"（58/85）。模态集：真陈述 14/15 标为 coherent；**偶假陈述 12/15 被标为 contradiction**；不可可能陈述 5/15 contradiction、9/15 paradox。模型解释用语相同（称偶假"contradicts established knowledge"）。
2. **真值探针**：balanced accuracy 峰值 0.93（与先前结果一致），在 impossible vs. false 上 AUC = 0.20（接近随机），表明真值方向无法区分必然假与偶假。
3. **不可能探针**：balanced accuracy 峰值 **0.97（layer 15，depth 16）**，permutation P = 0.0005（fold），Bonferroni P = 0.018；**impossible vs. false AUC = 1.00**，impossible vs. true AUC = 0.98；对 true vs. false AUC = 0.51（无信息）。
4. **异常探针**：anomalous vs. others AUC = 0.98；**impossible vs. anomalous AUC = 0.89**，说明两方向可区分但部分重叠（cosine ~ 0.4，中间层）。
5. **交叉数据集转移**：impossibility probe 从模态集→哲学集 AUC 最高 0.72；反向 0.79，说明信号相关但不等同。
6. **SAE 特征**：feature 14761 在 impossible 上 firing 0.67、anomalous 0.40、false 0.07；feature 9201 在 impossible 上 firing 1.00、false 0.80、anomalous 0.67；偶假（contingent false）上 firing 极低，印证探针几何。

## 相关工作脉络
1. **Azaria & Mitchell (2023)**：证明 LLM 内部状态可解码"是否撒谎"；本文扩展至更细粒度——区分两类"假"（偶假 vs. 必然假），并发现二者正交。
2. **Marks & Tegmark (2024)**：发现真/假数据在 LLM 表征中的线性几何结构；本文在此基础上追问：是否存在独立于真值轴的"不可能性轴"？答案是肯定的。
3. **Hewitt & Liang (2019); Belinkov (2022)**：关于探针设计的批评——可解码不等于模型因果使用；本文作者明确承认此局限，将贡献定位为"记录痕迹"而非"证明概念"。
4. **Sahoo et al. (2026)**：指出线性探针可能捕获任务格式而非推理模式；本文用 surface TF–IDF baseline（AUC 0.67–0.77）与 held-out family 设计尽量规避此风险。
5. **Gemma Scope 2 (McDougall et al., 2025)**：提供公开 SAE 权重；本文复用其 layer 15 16k SAE 验证探针发现的分布式特征基础。
6. **Alain & Bengio (2016)**：线性探针方法奠基；本文沿用但新增 double dissociation 与跨对比 transfer 检验以强化因果推断力度。

## 局限性与未来方向
1. **单一模型与小刺激量**：仅使用 Gemma 3 4B IT（小模型）、15 个主题族系×5 条件 + 85 哲学提示，不足以描绘模型的完整"意义地理"。
2. **相关性而非因果性**：探针证明可解码性（accessibility），未通过因果干预（如 patching/intervention）证明模型在实际推理中使用了该方向。
3. **刺激设计的人为性**：5 类范畴是人类定义的；模型可能采用不同分类学；且句式模板单一，词汇覆盖有限。
4. **表面形式贡献未被完全剥离**：TF–IDF baseline AUC 0.67–0.77，部分可分性来自表面线索（如"married bachelor"中的lexical cue），不可能方向虽独立但仍与之混合。
5. **未扩展至悖论与自指**：哲学集中的 paradox 与 underdetermined 案例未能被不可能探针统一解释（transfer AUC 0.72/0.79），说明"不可能"与"悖论"在表征中可能不同。
6. **未来方向**：在更大模型中复现并追踪该几何是否增强/溶解；用因果干预验证模型是否"调用"该方向；扩展到模型自组织的分类体系对比。

## 研究启发与可借鉴点
1. **Double dissociation 探针范式**：同时训练多类探针并交叉 transfer 评估 AUC，是检验"两个概念在表征中是否独立"的黄金标准，可直接迁移至其他哲学/认知范畴（如"虚构 vs. 错误""义务 vs. 可能性"）的 AI 表征研究。
2. **Held-out family 交叉验证设计**：按主题族系而非随机样本分组 held-out，有效防止词汇泄漏，确保探针学到的是抽象范畴而非表面共现；可推广至所有概念探针研究。
3. **SAE 与线性探针的三角验证**：用开源 SAE（Gemma Scope）的 feature firing profile 交叉验证 probe 几何，既增强了结论可信度，又揭示了分布式表征而非单 neuron 的存在形式——此联合策略值得推广。
4. **人类输出与内部表征的对照分析**：先检验模型在词表层面的分类行为（发现混淆），再揭示内部正交几何，这种"表面 vs. 深层"的对照框架对 AI 对齐与可解释性研究有方法论价值。
5. **可迁移到虚假信息检测**：若必然假与偶假在表征中正交，则可在不依赖输出文本的情况下，通过探针分数区分"偶发错误"与"逻辑不可能"，为更精细的幻觉/矛盾检测器提供表征级信号。

## 关键术语表
**Contingent falsehood（偶假）**：与经验事实不符但逻辑上可能为真的陈述（如"珠穆朗玛峰在阿尔卑斯山"），与必然假在表征上正交。
**Necessary falsehood / Impossible（必然假/不可能）**：在所有可理解解释下均为假的陈述（如"已婚的单身汉""比自身更高的珠穆朗玛峰"），代表上接近语义异常而非偶假。
**Double dissociation（双解离）**：探针 A 能区分 X vs. Y 但不能区分 X vs. Z，探针 B 相反，证明 X 与 Y、X 与 Z 在表征中由独立方向承载。
**Linear truth probe（线性真值探针）**：在残差流上训练的 L2 正则 logistic regression，用于判断激活向量中是否线性可分真/假陈述。
**Gemma Scope 2 SAE（Gemma Scope 2 稀疏自编码器）**：Google DeepMind 开源的 Gemma 3 4B IT 各层 JumpReLU 稀疏自编码器，dictionary size 16k，提供可解释特征。
**Balanced accuracy（平衡准确率）**：正负类 recall 的均值，处理类别不均衡时比 accuracy 更可靠，本文探针核心指标。
**TF–IDF surface baseline（TF–IDF 表面基线）**：用词/字符 n-gram 频率作为分类特征，衡量刺激可分性中有多少来自表面词汇线索而非深层表征。
**Semantic anomaly（语义异常）**：选择限制违反型句子（如"Everest rehearses its patient snow"），语法合法但语义组合不合常规，与必然假在表征上部分重叠但可区分。

## 可复现要素
- **数据集**：哲学集（85 prompts）+ 模态集（75 prompts），完整 stimulus JSON 已公开于 https://github.com/sixticket/representing-the-impossible
- **代码**：激活提取、探针训练、SAE 编码、绘图全部开源（同上仓库）
- **模型权重**：google/gemma-3-4b-it（Hugging Face 公开，bfloat16，未修改）
- **SAE 权重**：Gemma Scope 2，layer 15 width 16k（small 与 big checkpoint），Hugging Face 公开
- **关键超参**：L2 正则 C = 0.1，class-balanced 权重，5-fold grouped CV（按 family held-out），greedy decoding ≤ 48 tokens，probe 用 feature-standardized 状态
- **软件环境**：Python 3.11, PyTorch 2.13.0, Transformers 5.14.1, SAELens 6.47.1, scikit-learn 1.9.0
- **训练设备**：PyTorch MPS（Apple Silicon）
- **置换检验**：1999 次 permutaion within family，Bonferroni 校正 35 层深度

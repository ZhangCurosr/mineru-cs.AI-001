---
title: "Falsehood-and-Impossibility-Are-Diferent-Directions-in-an-AI"
source: https://arxiv.org/pdf/2608.12852v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:28:16"
---

# 论文速读：Falsehood-and-Impossibility-Are-Diferent-Directions-in-an-AI

## 一句话总结
本文通过线性探针与稀疏自编码器分析多模态模型 Gemma 3 4B IT 的内部激活，发现模型在表层语言分类中将“偶发虚假”与“逻辑矛盾”混为一谈，但在残差流激活空间中二者由近乎正交的方向分别编码；且“必然虚假”语句的表征位置更接近语义异常，而非虚假轴上的极端延伸。

## 研究问题与动机
- **核心问题**：LLM 能否在内部表征中区分“虚假(falsehood)”“不可能(impossibility)”与“语义异常(anomaly)”？不可能性是否只是真假轴上的一个极端点？
- **现有方法不足**：既往工作已证实命题真值可从模型残差流中线性解码，但缺乏针对“不可能性”这一更严格负例的系统性探针实验；且多数研究依赖单一真值基准或自然语言输出，未控制构造类型与话题分布。
- **实践动机**：LLM 输出可流畅且具说服力，却可能包含虚假、不自洽或逻辑上不可能的内容；若模型内部未区分这些失败类型，将影响错误检测、事实核查与对齐机制的设计。
- **理论/哲学动机**：从亚里士多德非矛盾律到维特根斯坦《逻辑哲学论》、乔姆斯基语法自主性，哲学与语言学长期区分“假”“不可能”“无意义”。Transformer 未内置此类本体论，需通过实证考察其自发学习的表征几何。

## 核心贡献（创新点）
1. **构建双集对照刺激库**：设计哲学集（85条，17个经典逻辑/语言种子家族）与模态集（75条，15个话题×5条件），在控制话题词汇的前提下严格区分真、偶发假、不可能、语义异常四类陈述。  
   → 与已有工作的本质区别：突破单一真值基准，首次将哲学构造与语言学异常操作化为可计数的实验对照，使“不可能性”成为可独立检验的变量。
2. **发现真假轴与不可能性轴的双分离几何**：证明模型表层标签混淆偶发假与矛盾，但其残差流中 Truth Probe 与 Impossibility Probe 的方向近乎正交（depth>10 后余弦绝对值≤0.12），且 Impossibility Probe 在 held-out 主题族上分离不可能与偶发假达 AUC 1.00。  
   → 与已有工作的本质区别：以往工作仅验证“真假可解码”，本文证明不可能性并非真假轴的线性延伸，而是由独立方向承载。
3. **SAE 特征与探针几何的三角验证**：在探针峰值层 15 调用 Gemma Scope 2 稀疏自编码器，识别出对不可能性敏感的特征（如 #14761、#9201），其 firing profile 重复探针发现的分离模式，且特征呈分布式而非单点神经元。  
   → 与已有工作的本质区别：将连续空间的线性探针发现延伸至离散特征空间，证实不可能性表征具有可解释的稀疏基底支撑，且与语义异常特征部分重叠但可区分。

## 方法详解
- **模型与激活提取**：使用 `google/gemma-3-4b-it`（34 层，残差宽度 2560，bfloat16）。每条提示嵌入固定指令模板，要求模型从 `{coherent, contradiction, paradox, underdetermined}` 中选一标签并解释。贪心生成最多 48 token。保留最终 prompt token 在 embedding 输出及每一 Transformer 层后的残差流状态（共 160×35×2560，float16）。
- **线性探针设计**：每层对特征标准化后的残差状态训练 L2 正则逻辑回归（`C=0.1`，类别平衡权重，固定随机种子）。三类主探针：
  - **Truth Probe**：False vs. True
  - **Impossibility Probe**：Impossible vs. {True, False, Improbable}
  - **Anomaly Probe**：Anomalous vs. {True, False, Improbable}
  采用按主题家族分组的 5 折交叉验证（同家族所有条件共同 held-out）。
- **几何与转移评估**：
  - 主性能报告 mean balanced accuracy；交叉对比性能报告 AUC。
  - 计算 Truth 与 Impossibility 探针方向的余弦相似度轨迹。
  - 跨数据集 transfer：在模态集训练的 Impossibility Probe 直接应用于哲学集（反之亦然），评估信号的结构化泛化能力。
  - 表面基线：在同一分组折叠内拟合 token 长度、词 1-2-gram TF-IDF、字符 3-5-gram TF-IDF。
  - 显著性检验：在峰值层对标签进行 1999 次 family-level 置换，重算分组 CV，Bonferroni 校正 35 个深度后的 P 值。
- **稀疏自编码器分析**：在 layer 15 使用官方 Gemma Scope 2 residual stream JumpReLU SAE（width=16k，two checkpoints: small l0 平均 ~16 active features/prompt，big l0 平均 ~90 active features/prompt）。按条件统计特征 `firing prevalence`（非零激活比例），特征排序依据为各组间 `log(1+z)` 均值的标准差差异。

## 实验与结果
- **数据集**：哲学集 85 prompts（17 家族 × 5 变体）；模态集 75 prompts（15 话题 × 5 条件，话题涵盖地理、物理、生物、算术、亲属、制度等）。
- **评估基线**：词/字符 TF-IDF（最强 0.77）、token 长度、 prior truth probing [11,12]。
- **主要结果**：
  - **表层混淆**：模型将 12/15 条偶发虚假标记为 “contradiction”（如 “Paris is the capital of Germany”），与必要虚假同组；哲学集总体分类准确率 55.3%，倾向滥用 “paradox” 标签。
  - **探针性能**：Truth probe 峰值平衡准确率 0.93；Impossibility probe 峰值 0.97（depth 16 / layer 15，置换检验 Bonferroni-adjusted P = 0.018）。
  - **双分离几何**（Table 2）：Truth probe 在 Impossible vs. False 上 AUC = 0.20（随机水平）；Impossibility probe 在 False vs. True 上 AUC = 0.51。两方向余弦绝对值在 depth>10 后 ≤ 0.12，近乎正交。
  - **与语义异常的关系**：Anomaly probe 与 Impossibility 方向余弦约 0.4；Impossibility probe 分离 Impossible vs. Anomalous 达 AUC 0.89，表明二者部分重叠但仍可区分。必要虚假在激活空间中更靠近 “colorless green ideas” 类异常，而非 “capital of Germany” 类普通虚假。
  - **SAE 验证**：大 capacity checkpoint 下，Impossible 敏感特征 #14761 在 Impossible

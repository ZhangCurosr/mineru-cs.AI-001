---
title: "Who-Thinks-Best-Depends-on-How-Long-You-Let-Them-Budget-Depe"
source: https://arxiv.org/pdf/2608.12150v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:45:55"
---

# 论文速读：Who-Thinks-Best-Depends-on-How-Long-You-Let-Them-Budget-Dependent-Rankings-in-LLM-Evaluation

## 一句话总结
本文系统挑战了LLM评估中“模型排名在推理条件下保持稳定”的隐含假设，通过在7个token生成预算下对4个模型在3个推理基准完成超5万次推断，揭示了排名跨预算反转、非单调“过度思考”现象的模型特异性，以及模型间随预算变化的动态互补性，并提出将预算作为一阶信号的路由验证方案。

## 研究问题与动机
- 现有Leaderboard与benchmark评测默认单一准确率即可定序，忽视了实际部署中token预算（计算/延迟约束）差异对模型表现的实质性影响。
- Test-time compute scaling与overthinking文献分别指出增加推理token可能提升或损害性能，但缺乏多模型、多预算、多任务下的系统性量化与对照。
- 低预算下高频的截断（truncation）会导致性能归因失真，现有工作未能有效剥离截断 artifacts 与真实推理失败。
- 现行LLM路由与集成系统普遍忽略预算维度，难以在异构算力场景下实现最优的按题/按预算分配。

## 核心贡献（创新点）
1. **建立item级预算行为分类框架**：首次按轨迹单调性将模型-题目对划分为四类，量化非单调下降（overthinking）的比例与分布规律。与已有工作相比，本文从宏观平均准确率转向题目个体层面的细粒度行为刻画，证明overthinking是模型特异性而非题目固有属性，无法通过剔除“难题”缓解。
2. **实证模型排名随预算显著反转**：在三个基准上均观察到最优模型身份随budget变化发生统计显著的交叉转移（McNemar p<0.01）。与主流benchmark“单分定序”范式相比，本文证明评测排名本质上是预算参数的函数族，打破了静态评估假设。
3. **刻画Oracle Gap的动态互补性**：定义并度量理想按题选择集成的性能增益，发现约束预算下互补性最强（GPQA最高+27.8 pp），且高预算下Jaccard相似度仍未饱和（0.741）。与单模型scaling分析相比，本文视角从“单模型能走多远”转向“多模型如何按预算组合”。
4. **提出预算感知路由原型并验证其边界**：设计首个将token budget作为显式特征的路由器，跨域捕获14.1% Oracle Gap（+2.67 pp），并通过SHAP与消融揭示预算特征的主导性及其跨域不可迁移性。与现有语义/规模路由相比，本文首次将预算独立建模，同时明确指出实际部署需结合领域自适应。

## 方法详解
- **行为轨迹分类**：对模型 $m$ 与题目 $i$ 记录7个预算下的二值正确序列 $\mathbf{c}_{m,i} \in \{0,1\}^7$，按单调性归类为 always-correct、monotone-increasing、non-monotone（存在 $1 \to 0$ 跳变）、always-wrong。为排除截断干扰，引入三层分析：all-items（标准评分，截断计错）、stop-only（仅保留 finish_reason=stop 的样本）、common non-truncated（四模型均完成题目的配对子集，用于McNemar检验）。
- **Oracle Gap度量**：对每个 $(i, b)$ 取 $\max_m \text{Acc}(m, i, b)$ 构成理想集成，减去当前预算下单模型最佳准确率，得到互补潜力上界。
- **预算感知路由器**：为每个模型训练XGBoost二分类器 $f_m(x, b) \in [0,1]$，预测在该预算 $b$ 下答对题目 $x$ 的概率。特征包含：$\log_2(b)$、表面文本统计量（字符/词数、特殊字符、LaTeX存在性、词熵、最大数值）、以及 all-MiniLM-L6-v2 句向量经PCA降至20维的语义特征。推理时执行 $m^* = \arg\max_m f_m(x, b)$。损失函数为标准对数似然（XGBoost默认），优化目标为逐模型校准概率。
- **解释与消融**：采用SHAP值量化特征贡献；设计跨域（GSM8K+MATH-500训练，GPQA测试）与域内5折CV双路径，剥离预算特征的真实增益与域过拟合风险。

## 实验与结果
- **数据集与基线**：GSM8K (1,319题)、MATH-500 (500题)、GPQA-D

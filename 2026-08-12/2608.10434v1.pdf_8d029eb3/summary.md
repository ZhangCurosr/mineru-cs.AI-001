---
title: "Conversational versus Dashboard Explainable AI for UAV Intrusion Detection: An Empirical Study of Operator Trust and Reliance"
source: https://arxiv.org/pdf/2608.10434v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:16:36"
field: "可解释AI与人机协同"
keywords: ["Explainable AI", "Intrusion Detection System", "UAV Security", "Human-AI Collaboration", "Trust and Reliance", "Conversational Interface"]
innovations: ["构建共享证据库的对话式与仪表板式XAI严格对照实验框架", "引入RAIR与RSR适当依赖度量揭示对话式XAI的过度依赖风险", "提出认知强制功能设计原则以平衡交互流畅性与证据验证"]
benchmarks: ["UAV-ID"]
---

# 论文速读：Conversational versus Dashboard Explainable AI for UAV Intrusion Detection: An Empirical Study of Operator Trust and Reliance

## 一句话总结
本文在无人机入侵检测（UAV IDS）场景下，通过对照实验系统比较了对话式可解释AI（基于LLM）与传统XAI仪表板对用户理解、信任及行为依赖的影响，发现对话式界面虽提升了感知有用性，却可能导致操作者适当自我依赖下降，存在过度依赖AI的风险。

## 研究问题与动机
- **核心问题**：在高风险的无人机网络防御中，不同形式的XAI界面（对话式 vs. 仪表板）如何影响操作者的理解深度、信任水平与行为依赖？
- **现有方法不足**：静态XAI仪表板面对多维度多模态赛博物理数据时，可视化展示密集、导航费力，且要求较高的AI素养，难以满足操作者动态、个性化的信息需求。
- **对话式XAI的潜在风险**：自然语言响应使AI建议更易被接受，可能削弱操作者验证底层证据的动力，导致"流畅性幻觉"与不当依赖。
- **研究空白**：现有文献缺乏对对话式XAI与仪表板式XAI在UAV入侵检测场景中operator trust与reliance的系统对比研究。

## 核心贡献（创新点）
- **实现了对比实验框架**：构建了共享同一底层XAI证据库的对话式与仪表板式双界面，确保可比的解释质量，此前缺乏此类严格对照。
- **引入了适当依赖度量**：使用RAIR（相对正向AI依赖）与RSR（相对正向自我依赖）区分合理依赖与盲目跟随，揭示了对话式界面的过度依赖风险，而不仅是主观满意度。
- **揭示了交互流畅性与认知参与度的权衡**：发现对话式界面提升感知有用性与信任，但未转化为更高的决策准确率，表明"易接受性"可能以牺牲深度理解为代价。
- **提出了认知强制功能设计原则**：建议未来XAI系统需嵌入"认知强制机制"（如要求操作者关联具体SHAP/PDP证据后方可确认），以平衡无缝交互与证据验证。

## 方法详解
- **入侵检测模型**：采用XGBoost作为核心分类器，在UAV-ID数据集上训练，处理42,258个样本，按7:3划分训练/测试集，报告macro-averaged指标以应对类别不平衡。
- **XAI方法集成**：整合四种解释方法对应不同信息需求：
  - **全局解释（PDP）**：回答"特征如何一般性地影响攻击概率"，展示全局趋势。
  - **局部特征归因（SHAP）**：回答"为何此告警被判定为恶意/良性"，使用TreeSHAP计算实例级特征贡献，并通过删除测试（deletion-style test）验证解释忠实度：
    $$\varDelta_k = p_y(x) - \mathbb{E}_{\tilde{x} \sim \mathcal{D}_{train}}[p_y(x_{S_k \to \tilde{x}_{S_k}})]$$
    Faithfulness AUC = $\frac{1}{K}\sum_{k=1}^{K}\varDelta_k$，其中$K=5$。
  - **对比反事实解释（MACE）**：回答"此告警需如何变化才能被不同分类"，约束搜索空间为合法特征域。
  - **交互式模拟（What-If）**：支持操作者手动扰动特征值，模拟预测变化。
- **对话式XAI界面架构**：基于`llama3.1:70b-instruct-q4_K_M`，分三步：①意图识别（将自然语言查询映射到XAI信息需求）；②工具执行（利用function calling能力输出结构化程序调用）；③响应综合（将技术输出转化为连贯自然语言叙述）。
- **仪表板XAI界面**：将相同四种XAI方法组织为独立模块，通过导航面板访问，提供结构化数据呈现。
- **用户实验设计**：57名参与者均分至三组（无解释Control、Dashboard XAI、Conversational XAI），每人审计20条告警（AI正确14条、错误6条），置信度与正确性解耦以阻止简单启发式决策。

## 实验与结果
- **数据集与基线**：使用UAV-ID数据集；分类器基线包括XGBoost、DT、RF、KNN、MLP、GB，XGBoost以Accuracy 81.09%、F1-Score 85.66%最优被选中。
- **主观感知结果**：
  - Explanation Utility：Conversational 4.10 vs. Dashboard 3.90
  - Operator Trust：Conversational 4.23 vs. Dashboard 4.20
  - Operator Understanding：Conversational 3.78 vs. Dashboard 3.68
- **行为依赖关键结果**：
  - Agreement Fraction：Control 0.74 → Dashboard 0.86 → Conversational 0.89
  - RSR（适当自我依赖）：Control 0.57 → Dashboard 0.29 → Conversational 0.11（显著下降，$H=38.26, p<.001$）
  - Decision Accuracy：Control 0.62 → Dashboard 0.74 → Conversational 0.71（对话式未超越仪表板）
- **最强结果与提升**：两种XAI界面均显著提升决策准确率（从0.62提升至0.74/0.71）；但对话式界面在RAIR仅0.48（略低于Dashboard的0.50），且RSR降至最低0.11，表明其引发的额外依赖未转化为更好决策。

## 相关工作脉络
- **TalkToModel [9]**：探索用自然语言对话解释ML模型，但聚焦通用场景，未针对高 stakes 安全领域的依赖行为进行实证测量。
- **ConvXAI [22]**：在协作写作场景中应用对话式XAI，本文将其思路迁移至网络安全领域并对比两种交互模态。
- **XAI Question Bank [7]**：本文依据其四类信息需求（how/why/how to be that/what if）设计XAI方法组合，确保解释覆盖operator核心疑问。
- **Appropriate Reliance度量框架 [27]**：引入RAIR与RSR指标区分合理与盲目依赖，区别于仅报告主观信任量表的前续工作。
- **Cognitive Forcing Functions [12]**：本文建议将此干预机制嵌入XAI界面设计，以缓解对话式界面可能带来的过度依赖风险。
- **UAV-ID数据集 [3]**：作为UAV入侵检测的标准基准，本文沿用其多模态赛博物理特征融合设定，验证了XGBoost的最优性能。

## 局限性与未来方向
- **样本量有限**：57名参与者均分三组，统计功效可能不足，尤其对微小效应量的检测。
- **参与者预筛选偏差**：要求具备AI/ML基础经验，结论可能不适用于普通操作者或领域专家。
- **短期实验生态效度**：实验室环境下审计20条告警，难以完全模拟真实战场高压、时间紧迫的决策场景。
- **未测量认知负荷**：仅报告了理解度与信任，未量化操作者在不同界面下的认知负担差异。
- **未来方向**：嵌入认知强制功能（如要求关联SHAP/PDP证据方可确认）；探索不确定性提示（uncertainty cues）的显式呈现；延长实验周期以评估长期依赖演变。

## 研究启发与可借鉴点
- **度量设计可迁移**：RAIR与RSR的适当依赖度量框架可用于评估其他领域XAI系统的行为影响，不仅依赖主观问卷。
- **置信度-正确性解耦实验设计**：通过交错高/低/随机置信度预测并解耦其与实际正确性，有效阻止操作者使用简单启发式，此设计可复用于其他AI辅助决策研究。
- **共享证据库的双界面对比**：确保对话式与仪表板使用完全相同的底层XAI输出，排除解释质量差异的混淆，为后续界面对比研究树立了方法论标杆。
- **XAI方法选择策略**：依据信息需求（how/why/how to be that/what if）而非算法新颖性来组合XAI方法，确保解释覆盖用户真实疑问。
- **认知强制功能的设计灵感**：将"暂停-验证"机制嵌入交互流程，可为高 stakes 领域（医疗、航空、网络安全）的XAI系统提供具体设计指南。

## 关键术语表
**Intrusion Detection System (IDS)**：入侵检测系统，用于识别网络中异常或恶意活动的安全系统。
**Explainable AI (XAI)**：可解释人工智能，旨在使AI模型的决策过程对人类可理解的技术集合。
**Appropriate Reliance**：适当依赖，指操作者根据AI建议的可靠性合理使用AI输出，既不过度信任也不盲目拒绝。
**RSR (Relative Positive Self-Reliance)**：相对正向自我依赖，衡量操作者在AI错误时仍坚持自己正确判断的倾向。
**RAIR (Relative Positive AI Reliance)**：相对正向AI依赖，衡量操作者在AI正确时采纳AI建议的倾向。
**Cognitive Forcing Functions**：认知强制功能，设计机制要求用户在决策前暂停并进行证据验证，以防止草率判断。
**Partial Dependence Plot (PDP)**：偏依赖图，展示单个特征对模型预测的全局平均影响。
**TreeSHAP**：基于树的SHAP值计算方法，用于高效估计树集成模型中各特征对个体预测的贡献。

## 可复现要素
- **数据集**：UAV-ID数据集（引用[3]），论文未明确声明公开链接。
- **代码/权重**：论文未提及开源；使用`llama3.1:70b-instruct-q4_K_M`模型。
- **关键超参**：XGBoost为最优分类器（具体超参论文未详述）；训练/测试分割比7:3；删除测试中$k \in \{1,...,5\}$。
- **参与者**：57名，均需具备AI/ML基础经验，均分三组。

---
title: "RadFusion: Towards Threshold-Controllable Radiology Report Generation"
source: https://arxiv.org/pdf/2608.10505v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:43:48"
---

# 论文速读：RadFusion: Towards Threshold-Controllable Radiology Report Generation

## 一句话总结
提出 RadFusion 框架，通过融合多标签分类器、VQA 报告生成器与大语言模型重写器，实现放射科报告生成中的阈值可控性，使生成报告的诊断行为可沿敏感-特异度曲线调节，并首次支持基于 ROC 的定量监管验证。

## 研究问题与动机
- 现有报告生成模型仅输出单一固定文本，缺乏置信度估计与阈值控制机制，无法像感知模型那样显式调节敏感-特异度权衡。
- 临床场景需求分化：急诊分诊需高敏感以降低漏诊，确诊评估需高特异以避免不必要的干预，固定报告无法自适应不同风险容忍度。
- 监管审批（如 FDA）普遍要求 ROC-based 验证，而生成模型缺乏可量化的诊断操作点，难以满足医疗器械合规路径。
- 可解释性与可信部署缺口：生成模型无法提供明确的诊断决策边界，限制了其在高风险临床工作流中的可靠集成。

## 核心贡献（创新点）
1. 提出首个阈值可控的放射科报告生成框架 RadFusion，将生成文本的诊断内容对齐至分类器的 ROC 特征，实现操作点的显式可调。
2. 设计“分类器+VQA生成器+LLM重写器”三模块融合架构，在保留生成模型丰富临床细节的同时，强制报告中的阳性/阴性陈述与阈值化分类结果一致。
3. 构建闭环评估协议：通过阈值扫描与文本回标绘制报告级 ROC 曲线，使生成报告具备可直接用于监管审批的定量验证能力。
4. 系统对比三类分类器实现（MI2微调、QRad线性探针、单token logit）与不同 LLM 重写器，验证框架的可移植性并给出最优配置。
5. 扩展至三维阈值控制 $(T_a, T_b, T_c)$，将置信度区间与临床紧迫度解耦，支持分级诊断报告策略，无需重新训练模型。

## 方法详解
- **整体流程**：输入胸片后，感知模型输出 K 类疾病的校准置信度分数 $\{P(C_i=1)\}$；给定阈值 τ 二值化为 $\hat{y}_i = \mathbf{1}[P > \tau]$；报告生成模型产出初始自由文本并支持追问获取遗漏细节；LLM 重写器结合两者输出，改写报告使其诊断陈述与 $\hat{y}_i$ 一致，同时保留解剖位置、严重程度、病程变化等临床属性。
- **感知模型**：采用 MedImageInsight (MI2) 基础模型，基于 Image-Text-Class Hybrid Contrastive Loss 微调：
  $\mathcal{L} = \mathcal{L}_{\text{image-text}} + \lambda \mathcal{L}_{\text{image-class}}$
  其中 $\mathcal{L}_{\text{image-text}}$ 为 softmax CE 对比损失，$\mathcal{L}_{\text{image-class}}$ 为 sigmoid CE 对比损失，类别嵌入基于模板提示（如 “a chest X-ray showing [CLASS NAME]”）。推理时通过可学习温度 τ_temp 计算余弦相似度并过 sigmoid 得概率，再经温度缩放（Temperature Scaling）后验校准。
- **报告生成模型**：采用 QRad 的 Auto-VQA 范式 $Q = f_Q(I), \ Y = f_A(I, Q)$，将报告生成重构为自我导向的视觉问答过程。初始报告覆盖阴阳性发现；当低阈值将某类翻转为阳性但原始报告遗漏时，可通过追加 VQA 追问从图像中召回定位/严重度等依据，形成证据池 $\mathcal{E}$。
- **LLM 重写机制**：使用 GPT-5 等通用大模型，通过结构化指令与每类疾病的正负表达模板（few-shot examples）对齐跨模态语义；阳性类若缺失则补全、若矛盾则修正；阴性类若存在则删除或否定；非类别内容（如影像质量、support devices）保持不变。
- **三维阈值扩展**：$P < T_a$ 为阴性区，$P > T_b$ 为阳性区，$T_a \le P \le T_b$ 为半阳性区（报告需明确标注不确定性并建议进一步检查）；$T_c$ 阈值化紧迫度评分，实现高风险发现优先强调，扩展了单一阈值的临床适用性。

## 实验与结果
- **数据集与设置**：MIMIC-CXR（227,835 studies, 377,110 chest X-ray images），使用官方 test split（2,347 studies）与 CheXpert 14类标签（评估13类，排除 No Finding），以 Findings 部分为生成目标。
- **评估协议**：τ 从 0.0 到 1.0 步长 0.1 扫描产生 11 个操作点，将改写报告经 GPT-5 回标为二值标签，计算 TPR/FPR 绘制 ROC 曲线；对比三类系统：分类器 ROC（蓝）、阈值控制报告 ROC（红）、无控制单点报告（绿叉）。
- **分类器实现对比**：MI2 FT（AVG AUC 0.90）与 QRad Linear Prob（AVG AUC 0.91）整体相当；QRad Linear Prob 在 Pneumothorax 等难类上显著更强，MI2 FT 在 Support Devices 等直观类上更优；QRad Token Logit 表现最弱（AVG 0.65）。
- **重写器对比**：GPT-5 默认带模板示例表现最佳（AVG 0.90）；移除示例后 Fracture 从 0.89 降至 0.73；GPT-5.4 未超越 GPT-5；较小模型（GPT-5.4-mini）需提升 reasoning effort 才能稳定对齐（Lung Lesion 从 0.58 升至 0.81）。
- **核心结论**：阈值控制报告的 ROC 紧密贴合分类器 ROC；在匹配特异度下敏感度高出 6.9%，在匹配敏感度高出 20.7%；验证了框架在保留临床丰富性的同时显著提升诊断准确性与可调控性。

## 相关工作脉络
- **放射科报告生成**：R2Gen/R2GenCMN → LLaVA-Rad/MAIRA/CheXagent/MedVersa/Libra 等；本文定位差异在于上述方法均输出固定报告，缺乏诊断操作点调节与 ROC 可验证性。
- **医学视觉问答**：Med-Flamingo/LLaVA-Med/Rad-ReStruct/RaDialog；本文利用 VQA 作为报告生成与信息补全的统一接口，专门解决传统生成模型阴性发现常见遗漏问题。
- **可控文本生成**：PPLM/FUDGE/Classifier-free Guidance；本文聚焦疾病类别级阈值控制与操作点选择，而非 token 级风格或语义属性引导。
- **置信度估计与校准**：P(True) logit 提取、verbalized confidence、温度缩放、proper scoring rules；本文将校准后的分类器置信度与生成流程解耦融合，使数值置信度直接驱动文本改写。
- **报告生成中的不确定性建模**：Wang et al. (20

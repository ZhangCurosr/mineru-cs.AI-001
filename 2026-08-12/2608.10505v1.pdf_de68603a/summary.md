---
title: "RadFusion: Towards Threshold-Controllable Radiology Report Generation"
source: https://arxiv.org/pdf/2608.10505v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:47:35"
field: "医学影像报告生成"
keywords: ["放射学报告生成", "阈值可控生成", "医学视觉问答", "ROC评估", "多模态大模型", "置信度校准", "可控文本生成"]
innovations: ["提出RadFusion框架，首次实现放射学报告生成的阈值可控性，使报告诊断内容与分类器ROC行为对齐", "设计基于LLM的重写策略+per-class模板示例，在保持临床细节grounding的同时实现类别级诊断控制", "构建闭环ROC评估协议，首次支持对生成报告的定量ROC分析和监管验证"]
benchmarks: ["MIMIC-CXR"]
---

# 论文速读：RadFusion: Towards Threshold-Controllable Radiology Report Generation

## 一句话总结
RadFusion 是首个面向放射学报告生成的阈值可控框架，通过将多标签分类器的置信度分数与 VQA 报告生成器融合，并利用 LLM 重写报告，使得生成报告的诊断内容可随阈值调节灵敏度和特异性的权衡，同时保持对原始描述的临床 grounding。在 MIMIC-CXR 上，生成报告的 ROC 曲线与底层分类器高度一致，且在匹配特异性下灵敏度提升 6.9%、在匹配灵敏度下特异性提升 20.7%。

## 研究问题与动机
1. **临床场景对灵敏度-特异性权衡的需求**：急诊分诊需要高灵敏度以减少漏诊，而确认性解读需要高特异性以减少不必要的干预，现有报告生成模型仅输出固定报告，缺乏可调的诊断行为。
2. **监管审批需要 ROC 验证**：FDA 等监管机构期望通过 ROC 分析对医疗器械进行定量评估，但现有生成模型没有置信度估计和阈值化机制，无法支持此类验证。
3. **现有可控文本生成方法不适用于临床诊断**：PPLM、FUDGE 等方法在 token 层面引导生成，无法基于疾病类别的特定阈值控制诊断内容；现有 LLM 置信度估计工作（如 logit-based P(True)）也缺乏运行时阈值调控机制。
4. **感知模型与生成模型的互补性未被充分利用**：分类器有数值化的 ROC 性能但缺乏描述性细节，生成器有丰富临床描述但缺乏可调置信度，两者尚未有效融合以实现可控报告生成。

## 核心贡献（创新点）
1. **提出了 RadFusion——首个阈值可控的放射学报告生成框架**：将分类器的每类置信度分数、VQA 报告生成器和 LLM 重写器三者融合，实现报告诊断内容与分类器 ROC 行为的对齐；已有报告生成模型均无此类机制。
2. **设计了基于 LLM 的报告重写策略，使生成报告忠实于分类器的阈值决策**：通过在提示词中提供 per-class 模板示例和重写规则，确保正类被保留或补充、负类被移除或否定，同时保持临床细节；此方法不同于 token-level 可控生成，实现了疾病类别级别的可控性。
3. **构建了闭环 ROC 评估协议，首次支持对生成报告的定量 ROC 分析**：将重写后的报告通过 GPT-5 转回二值标签， sweep 阈值生成多条 ROC 曲线；相比已有方法，这是首次使报告生成模型能够接受监管期望的 ROC 验证。
4. **系统比较了三类分类器实现和多种 LLM 重写器配置**：MI2 FT 分类器、QRad Linear Prob、QRad Token Logit，以及 GPT-5/5.4/mini 和 DeepSeek V4 Pro，为后续研究提供了组件级最优配置参考。

## 方法详解
RadFusion 由三个组件构成：

**（1）感知模型（分类器）**：给定胸部 X 光图像，对 CheXpert 定义的 14 种疾病类别输出置信度分数 $\{ \bar{P}(C_i = 1) \}_{i=1}^{K}$。给定阈值 τ，得到二值预测 $\hat{y}_i = \mathbf{1}[P(C_i = 1) > \tau]$。研究了三类实现：
- **MI2 FT**（默认）：基于 MedImageInsight 编码器，使用 Image-Text-Class Hybrid Contrastive loss 微调，loss 为 $\mathcal{L} = \mathcal{L}_{\text{image-text}} + \lambda \mathcal{L}_{\text{image-class}}$（前者为 softmax cross-entropy，后者为 sigmoid cross-entropy）。推理时通过余弦相似度加可学习温度计算类别概率，再用 temperature scaling 进行校准。
- **QRad Linear Prob**：在 QRad 冻结的图像编码器顶部训练线性分类头，推理时与报告生成器共享编码器。
- **QRad Token Logit**：利用 QRad 的 VQA 能力，提取 [Yes]/[No] 的 softmax 概率作为置信度。

**（2）报告生成模型（QRad Auto-VQA）**：将报告生成重构为自我导向的视觉问答过程 $Q = f_Q(I), Y = f_A(I, Q)$，即问题生成器 $f_Q$ 根据图像预测临床相关问题序列，答案生成器 $f_A$ 逐个回答问题生成报告句子。当低阈值将某类别翻转为阳性时，可通过补充问答检索原始报告中遗漏的发现及其位置、严重程度等细节，作为重写证据池 $\mathcal{E}$。

**（3）LLM 重写器**：使用现成 LLM（默认 GPT-5）结合分类器的二值预测 $\{\hat{y}_i\}$ 和证据池 $\mathcal{E}$ 重写报告。重写指令包含三条规则：① 对正类（$\hat{y}_i=1$），若报告中已有则保留，若缺失或被否定则用 $\mathcal{E}$ 中的细节添加或修正；② 对负类（$\hat{y}_i=0$），若报告中提及则为阳性的描述则移除或否定；③ 非类别内容（影像质量、支持设备等）保持不变。提示词中包含 per-class 模板示例（如 "Pleural Effusion" 可表现为 "small bilateral pleural effusions"、"fluid in the costophrenic angles" 等），以建立类别名称与自然语言表述的映射。

**三维阈值扩展**：框架可扩展到三维阈值 $(T_a, T_b, T_c)$，其中 $T_a \leq T_b$ 将置信度分为负区、半阳区和阳性区（半阳区报告中标注不确定性并建议进一步检查），$T_c$ 基于紧迫感评分独立处理时间敏感发现。

## 实验与结果
- **数据集**：MIMIC-CXR（227,835 研究，377,110 张胸部 X 光片），使用官方 test split（2,347 研究），以 Findings 部分为生成目标。
- **评估协议**：sweep 阈值 τ ∈ [0, 1]（步长 0.1），共 11 个操作点，对 13 个疾病类别（排除 "No Finding"）分别计算 ROC 曲线和 AUC；用 GPT-5 将重写报告转回二值标签与 ground truth 比较，计算 TPR 和 FPR。
- **分类器 AUC-ROC**（Table 1）：MI2 FT 宏观平均 0.90，QRad Linear Prob 0.91，QRad Token Logit 0.73。QRad Linear Prob 在 Pneumothorax 等难类上更强，MI2 FT 在 Support Devices 等易类上更强。
- **LLM 重写器**（Table 2）：GPT-5 宏平均 AUC 0.90，去除 per-class 示例后降至 0.88（Fracture 从 0.89 降至 0.73）；GPT-5.4 与 GPT-5 持平，新一代模型不一定更优。
- **主结果**：阈值可控报告的 ROC 曲线与分类器高度一致；在匹配特异性下灵敏度提升 6.9%，在匹配灵敏度下特异性提升 20.7%；非控制报告的单一操作点始终低于 ROC 曲线。

## 相关工作脉络
1. **放射学报告生成**：R2Gen（记忆驱动 Transformer）、R2GenCMN（跨模态记忆）、CvT2DistilGPT2（ViT+蒸馏解码器）等早期方法，以及 LLaVA-Rad、MAIRA、CheXagent、MedVersa、Libra 等基于大视觉-语言模型的方法——均只输出固定报告，无诊断行为调控机制。本文与其本质区别在于引入了阈值可控性。
2. **医疗视觉问答**：Med-Flamingo、LLaVA-Med、Rad-ReStruct、RaDialog 等——提供信息查询接口，但未与报告生成的阈值控制结合。本文利用 QRad 的 Auto-VQA 能力补充被遗漏的发现细节。
3. **可控文本生成**：PPLM、FUDGE、Classifier-free Guidance——在 token 级别引导生成属性，不支持基于疾病类别特定阈值的诊断控制。本文实现的是类别级别而非 token 级别的可控性。
4. **置信度估计与校准**：logit-based P(True)（Kadavath et al.）、verbalized confidence、temperature scaling 校准等——用于 LLM 置信度量化，但 Wang et al. (2024) 仅在训练时加权 loss 而推理时无控制机制。本文在此基础上实现了推理时的阈值化操作点选择。
5. **QRad**（Jin et al.）：统一报告生成与医疗 VQA 的 SOTA 模型，基于 MI2 编码器，支持追问以获取遗漏信息，是本文报告生成组件的基座。

## 局限性与未来方向
1. **误差传播风险**：分类器错误会传播至重写报告，报告生成器可能遗漏发现，LLM 重写器可能引入细微的语言 artifacts。
2. **预定义类别限制**：阈值控制仅作用于预定义的 14 个 CheXpert 类别，其他发现不受阈值调节。
3. **提示词依赖与可迁移性**：重写提示词为模板化设计，在 GPT-5 上调优，对不同 LLM、成像模态或临床领域需重新调优；消融显示 GPT-5.4 在 Lung Lesion 上显著退化（0.73），需调整推理强度。
4. **自动评估的误差**：报告转标签依赖 GPT-5 或 VisualCheXbert，可能引入额外误差。
5. **未来方向**：扩展至三维阈值控制（$T_a, T_b, T_c$）以支持更丰富的分级报告策略；扩展到不同成像模态和临床领域；开发面向临床医生的可调阈值交互界面。

## 研究启发与可借鉴点
1. **"感知-生成-重写"三阶段融合架构**：将分类器的结构化置信度与生成器的描述性能力结合，通过 LLM 桥接两者——此范式可迁移至其他需要可控生成的领域（如法律合规文本、内容审核）。
2. **闭环 ROC 评估协议的设计思路**：通过 threshold → 报告重写 → 标签还原 → ROC 重建的闭环验证，为生成模型的诊断性能提供了可监管量化的评估方法；类似思路可用于其他生成式诊断任务。
3. **per-class 模板示例在重写提示词中的关键作用**：消融实验表明提供类别名称与自然语言变体的映射示例对忠实重写至关重要（Fracture AUC 从 0.89 降至 0.73），这一技巧值得在需要事实一致性的 LLM 重写任务中借鉴。
4. **Auto-VQA 补充证据池机制**：利用 VQA 的追问能力检索初始报告中遗漏的发现细节，解决了生成模型常见的事实遗漏问题；该机制可与任何支持多轮查询的生成模型结合。
5. **温度缩放校准保持 AUC 不变**：calibration 仅改善置信度准确度而不影响判别性能（AUC 不变），可在任何需要可靠置信度的分类器上无缝集成。

## 关键术语表
**Threshold-controllable generation**：通过调节决策阈值来控制生成文本中诊断陈述的灵敏度和特异性权衡的能力。
**ROC conformance**：生成报告经阈值 sweep 后重建的 ROC 曲线与底层分类器的 ROC 曲线高度一致的现象，验证了可控性。
**Auto-VQA**：Self-directed Visual Question Answering，报告生成器主动生成问题并逐个回答以构建完整报告的过程。
**Temperature scaling**：后校准方法，通过学习标量温度 $T$ 最小化 ECE（Expected Calibration Error），使置信度与实证正确率对齐，不改变 AUC。
**Per-class template examples**：重写提示中为每个疾病类别提供的正/负文本片段示例，建立类别标签与自然语言表述的映射关系。
**Closed-loop evaluation protocol**：从分类器阈值决策 → 报告重写 → GPT-5 标签还原 → 与 ground truth 比较的闭环 ROC 评估流程。
**Operating point**：ROC 曲线上对应特定阈值的选择点，决定灵敏度与特异性的权衡位置。
**ChiXpert/CheXbert**：CheXpert 数据集及其报告标注工具 VisualCheXbert，定义了 14 种胸部 X 光疾病类别及 "No Finding"。

## 可复现要素
- **数据集**：MIMIC-CXR（Johnson et al., 2019），公开可用。
- **代码**：论文未明确声明代码开源状态；QRad 在 referenced workshop 中介绍，MedImageInsight 为开源嵌入模型（Codella et al., 2024）。
- **权重**：MI2 基础模型开源；QRad 具体权重状态论文未明确说明。
- **关键超参**：阈值 sweep 步长 0.1（11 个操作点）；13 个评估类别（排除 "No Finding"）；温度缩放用于校准；Image-Text-Class Hybrid Contrastive loss 中的 λ 参数论文未明确给出具体值。
- **LLM**：GPT-5（Singh et al., 2025）作为默认重写器；实验中亦比较了 GPT-5.4、GPT-5.4-mini 和 DeepSeek V4 Pro。
- **评估工具**：VisualCheXbert 用于从参考报告提取 ground truth 标签；GPT-5 用于从重写报告提取预测标签。

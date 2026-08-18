---
title: "JieZi-A-Large-Scale-Expert-Audited-Dataset-and-Benchmark-for"
source: https://arxiv.org/pdf/2608.11741v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:34:28"
field: "古文字计算与多模态理解"
keywords: ["Ancient Chinese Character Exegesis", "Vision-Language QA", "Paleographic Dataset", "Multimodal LLM", "Domain-specific Fine-tuning", "Expert-audited Data"]
innovations: ["首次将古汉字考释形式化为ACCE四层次VQA任务", "构建首个50万级专家审核多模态考释数据集JieZi-Dataset", "设计专家介入的半自动数据生成流水线缓解大模型幻觉"]
benchmarks: ["JieZi-Bench"]
---

# 论文速读：JieZi-A-Large-Scale-Expert-Audited-Dataset-and-Benchmark-for-Ancient-Chinese-Character-Exegesis

## 一句话总结
本文首次将古汉字考释形式化为多模态问答（VQA）任务 ACCE，构建了首个大规模专家审核训练数据集 JieZi-Dataset（约50万QA对、13万字形图像）和高质量评估基准 JieZi-Bench（约8千QA对），验证了领域特定数据对多模态大模型古文字理解的关键作用。

## 研究问题与动机
- 现有计算研究聚焦字形识别、图像检索等狭窄子任务，无法形式化完整的学者考释工作流。
- 通用多模态大模型缺乏古文字学知识，在面对古汉字时容易产生事实性幻觉。
- 高质量人工标注成本高昂，全自动化生成又不可靠，缺乏可扩展的数据构建方案。
- 缺乏系统评估多维度考释能力的基准，导致当前模型在深度考释任务上的能力边界不明。

## 核心贡献（创新点）
- **首次提出 ACCE 任务**：将传统小学考释流程形式化为四个递进层次（基本信息、字形分析、字义阐释、历时演变），与现有单标签分类任务本质不同。
- **构建 JieZi-Dataset**：首个大规模专家审核的多模态VQA训练数据集，覆盖六种书体（甲骨文至楷书）、十个细粒度子任务，总量超50万QA对。
- **构建 JieZi-Bench**：完全由人文专家逐条校验的评估基准，参考答案由《康熙字典》《说文解字》等权威辞书独立编纂，与训练数据严格隔离防止泄露。
- **设计专家介入的生成流水线**：通过模板约束+源文本引用+分阶段人工抽检，有效缓解大模型幻觉，平衡数据规模与学术准确性。

## 方法详解
**任务定义（ACCE四层次）**：
- L1 基本信息：字符识别（CHAR）、书体分类（SCRC）
- L2 字形分析：结构分类（STRC）、部件识别（COMR）、部件功能（COMF）、部件释义（COMI）、造字法分类（FORC，含六书体系）
- L3 字义阐释：本义（ORIM）
- L4 历时演变：部件演变（COME）、演变阐释（EVOI）

**数据构建三阶段流水线**：
1. **收集阶段**：以《汉字源流字典》为主源（2000+页、500万token、1.3万+字头），辅以 ACCP、MegaHan97K 等公开数据集；使用 Gemini-2.5-Pro 进行OCR并手动纠错（约1000小时），YOLOv11 检测字形并分类书体。
2. **结构化阶段**：用 Gemini-3-Flash 将非结构化字典条目解析为JSON元数据，JieZi-Dataset 抽检10%，JieZi-Bench 逐条人工校验。
3. **VQA生成阶段**：基于专家设计模板随机生成5-10个QA对，确保L1-L4全覆盖；JieZi-Dataset 抽检5000条（约3%错误率），JieZi-Bench 全量校验。

**评估指标**：分类任务用Accuracy；部件识别/功能用字符级F1-Score；开放生成用BERTScore；EVOI用LLM-as-a-Judge（事实一致性FAC + 学术表达SCE两个维度）。

## 实验与结果
- **测试基线**：GPT-5.4、Gemini 3.1 Pro、Claude Opus 4.6、Doubao-Seed-2.0-pro、Kimi-K2.5（170B）、GLM-4.6V（107B）、Qwen3.5系列（2B/4B/9B/397B）等。
- **核心发现**：
  - 通用模型在基础识别（SCRC 40-77%）尚可，但在细粒度字形分析和历时推理上表现较差（EVOI最高仅38.7）。
  - 中文语料丰富的模型（Doubao、Kimi）显著优于同体量英文模型（GPT-5.4 CHAR仅10.4 vs Doubao 36.6），说明领域数据覆盖比模型规模更重要。
  - **微调效果**：Qwen3.5-9B + JieZi-Dataset 在全部子任务上达到SOTA，CHAR从22.3提升至52.6（+30.3），SCRC从51.0提升至80.1（+29.1），COME从22.4提升至45.4（+23.0）。
  - 小模型（2B）经微调后可超越更大参数的通用模型（如COMR 42.6 > Gemini 3.1 Pro的31.9）。
- **泛化分析**：未见过字形（UG）上CHAR骤降至接近0，但结构分析指标衰减有限（COMF仅下降3.5-7.8点），表明模型习得了可迁移的字形知识。
- **书体难度**：甲骨文→金文→小篆→隶书→楷书，性能递减，早期书体仍是挑战。

## 相关工作脉络
- **古文字数据集**：OBIMD、OracleSage、PD-OBS 等集中于甲骨文单书体，且多为稀疏符号标签；JieZi-Dataset 首个支持多书体+多维度自然语言标注。
- **古文字评估**：OBI-Bench、Oracle-Bench 聚焦甲骨文考古场景；C³-Bench、MCS-Bench、AC-EVAL 在篇章/知识层面评估古典素养，但未涉及单字视觉理解。
- **领域专项模型**：tonggu-vl-2B 虽经古文语料预训练，但在ACCE子任务上得分极低（多数<5%），证明阅读理解≠考释推理，凸显本任务的独特性。
- **多模态大模型**：GPT-5.4、Gemini 3.1 Pro 等通用模型在ACCE上表现不佳，揭示现有MLLM在专门化人文学科任务中的知识匮乏。

## 局限性与未来方向
- **书体覆盖不均**：金文和小篆占比高，战国文字、简帛等手写体覆盖有限。
- **疑难字处理能力**：部分甲骨文、金文存在多种释读争议，当前字典来源难以覆盖所有学术分歧。
- **自然退化图像**：仅12.3%样本来自真实历史文献（含锈蚀、污渍），对实际考古场景的泛化有待验证。
- **未来方向**：扩展至更多出土文献、融合多源学术争议观点、探索少样本/零样本考释能力。

## 研究启发与可借鉴点
- **专家介入的半自动数据构建范式**：模板约束+源文本引用+分阶段抽检，可迁移至其他专业知识密集型领域（如医学、法学VQA）的数据生产。
- **训练-评估数据隔离设计**：JieZi-Bench 使用与训练源完全不同的权威辞书，有效防止数据泄露，为评测可靠性提供范例。
- **多维度细粒度评估体系**：十个子任务覆盖从识别到推理的完整链条，每类任务匹配不同评估指标（Accuracy/F1/BERTScore/LLM-judge），值得复杂任务评测借鉴。
- **小模型+高质量领域数据的性价比**：9B微调模型超越百亿级通用模型，验证了"数据驱动而非单纯规模扩张"在专业领域的有效性。

## 关键术语表
- **ACCE（Ancient Chinese Character Exegesis）**：古汉字考释任务，将传统小学分析流程形式化为四层次VQA问题。
- **六书（Six Writings）**：传统汉字构造分类体系，包括象形、指事、会意、形声、转注、假借。
- **ORIM（Original Meaning）**：字的本义，即该字形在历史语境中的 foundational semantic value。
- **COME（Component Evolution）**：部件历时演变，追踪特定部件在不同书体中的形态变化。
- **EVOI（Evolution Interpretation）**：演变阐释，综合分析字形演变的动因及其在文字系统发展中的意义。
- **LLM-as-a-Judge**：利用大语言模型作为裁判评估开放生成任务，此处用于EVOI的事实一致性与学术表达打分。
- **Expert-in-the-Loop**：专家介入的半自动数据构建模式，在关键节点引入人工校验而非完全自动化或全人工标注。

## 可复现要素
- **数据集**：JieZi-Dataset 和 JieZi-Bench 已在 https://github.com/Ran00w/JieZi 开源。
- **代码**：数据构建流水线与实验代码已公开。
- **模型权重**：微调后的 Qwen3.5-2B/4B/9B 权重已提供。
- **关键超参**：Batch size 128（2B/4B）或64（9B），Learning rate 1e-5，cosine schedule，Warmup 0.05，Weight decay 0.1，Epoch 5，bf16精度，8×Ascend 910B NPU。

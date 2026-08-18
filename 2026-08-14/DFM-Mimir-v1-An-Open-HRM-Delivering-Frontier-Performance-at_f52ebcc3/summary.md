---
title: "DFM-Mimir-v1-An-Open-HRM-Delivering-Frontier-Performance-at"
source: https://arxiv.org/pdf/2608.13517v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:23:52"
field: "低资源语言模型与合规数据训练"
keywords: ["Hierarchical Reasoning Model", "Low-resource NLP", "Permissible Data", "Instruction Tuning", "Danish Language Models"]
innovations: ["仅用合规数据实现1B参数前沿性能", "合成移植数据集替代不合规数据", "从多选题向自由生成任务的训练数据再平衡"]
benchmarks: ["GSM8K", "MATH", "HumanEval", "MMLU", "BoolQ", "Winogrande", "DROP", "DaLA", "WikiQA"]
---

# 论文速读：DFM-Mimir-v1-An-Open-HRM-Delivering-Frontier-Performance-at

## 一句话总结
论文提出了 Mimir v1，一个基于 Hierarchical Reasoning Model (HRM) 架构的 1B 参数开源语言模型，仅使用符合许可规范的训练数据，从训练伊始即聚焦 post-training 数据，在英语和丹麦语任务上达到前沿级性能。

## 研究问题与动机
1. 当前大语言模型开发依赖大规模、往往非合规的数据集，对坚持开源和伦理数据的研究者构成极高进入门槛。
2. 丹麦等低资源语言的高质量数据池有限，难以通过传统从头预训练方式构建可用的基础模型。
3. 现有 HRM-Text 基线模型在数学与代码任务上表现不足，且未充分考虑指令遵循与生成式任务。
4. 部分原始训练数据不符合 DFM（Danish Foundation Models）项目的许可标准，需要替代方案。

## 核心贡献（创新点）
1. **提出 Mimir v1**：基于 HRM-Text 架构从训练伊始使用仅 70.5B tokens/epoch 的 permissible 数据，实现了与更大模型竞争的性能。
2. **合成替代数据集（Transplant Datasets）**：用 Gemma-4 31B 生成并审计的合成数据替换不合规的 Sapient 数据，实现可比甚至更优性能。
3. **训练数据重新平衡**：将原 HRM-Text 中以多选题为主的训练数据转向自由形式生成任务，更好地对齐评估基准（如 GSM8k、MATH）。
4. **丹麦语领域突破**：在丹麦语任务上设立新 SOTA，显著超越此前基线模型。

## 方法详解
1. **架构**：采用 HRM-Text 架构，hidden size=1,536，32 层（half layers=true），每层 12 个注意力头，FFN expansion factor=4，2 个 H-cycles 和 3 个 L-cycles，truncated backpropagation 限制为 5 步。
2. **位置编码**：使用 RoPE（Rotary Position Embedding），θ=10,000。
3. **归一化**：pre-norm 结构，ε=10⁻⁶。
4. **训练配置**：使用 Gemma-4 tokenizer（区别于 HRM-Text 的自定义 tokenizer），应用 chat template 学习对话范式；FSDP 分布式训练，bfloat16 计算，fp32 聚合；AdamW 优化器，峰值学习率 3×10⁻⁴，2,000 步线性 warmup 后保持恒定。
5. **数据混合**：161 个数据集共 70.48B tokens/epoch，包含丹麦语指令与知识（22.07%）、英语指令（19.26%）、Sapient 混合数据（17.02%）、数学与推理（14.76%）、合成移植数据（10.00%）等八个类别。
6. **数据加工方式**：包括重新格式化（Reformatted）、筛选后重新格式化、合成+审计（Synthetic+audited）、工具调用格式化、翻译+审计等七种形式。
7. **训练效率**：8 卡 NVIDIA B200 GPU，180GB HBM，1.65M 步训练约 3 周，平均每步不到 1.1 秒。

## 实验与结果
**评估基准**：20 个基准，涵盖英语、数学与代码、丹麦语三个领域。

**主要结果**：
- **英语**：Mimir 1B 平均 69.0，仅落后 Qwen 3.5 4B（69.3）0.3 分；在 BoolQ（87.8）、Winogrande（73.5）、DROP（83.1）上均超越所有对比模型。
- **数学与代码**：Mimir 1B 平均 64.1，超越 HRM-Text 1B（46.9）达 36.7%；GSM8K 达到 89.9（第二），HumanEval 达到 56.7（权重类第一）；略低于 SmolLM3 3B（67.9）。
- **丹麦语**：Mimir 1B 平均 56.8，设立新 SOTA；远超 HRM-Text 1B（21.7），在 DaLA（96.1 F1）、GEC（85.6 EM）、WikiQA（66.8 EM）等任务上大幅领先。

## 相关工作脉络
1. **HRM-Text (Wang et al., 2026)**：本文采用的基础架构，本文改进其数据混合策略并引入合成替代数据。
2. **Qwen 3.5 系列**：包括 0.8B、2B、4B 版本，作为同量级和更大规模的主流基线。
3. **Gemma 4 E2B**：5B 总参数（有效 2.3B），是本文在数学与代码任务上的最强对比对象。
4. **Sapient 数据集**：原 HRM-Text 训练中的核心数据集，包含 Flan、Platypus 等子集合，本文通过合成移植部分替代。
5. **SmolLM3 3B**：在数学与代码任务上表现最优的常规 2-3B 模型，作为效率对比基线。
6. **丹麦语基础模型**：包括 Munin-Apertus 8B、Munin-Mistral 8B、Munin-Qwen 9B 等，用于丹麦语任务对比。

## 局限性与未来方向
1. 在数学与代码领域仍落后于 Gemma 4 E2B（5B），有提升空间。
2. 作为助手的能力相比 SOTA 仍有局限，需要更多指令调优工作。
3. HRM 架构的强化学习尚未探索，未来可引入 RL 进一步提升性能。
4. 数据集的完全开源许可仍在推进中，部分数据因协议限制无法公开共享。
5. 模型的扩展行为（scaling behavior）有待进一步研究。

## 研究启发与可借鉴点
1. **合成数据替代策略**：用 LLM 生成 + 人工/自动审计的方式替换不合规数据，为低资源/合规敏感场景提供了可行方案。
2. **数据分布重新平衡**：从多选题分类任务转向自由生成任务，显著提升了对 exact-match 评估基准的适应性。
3. **HRM 架构的高效性**：在 1B 参数下实现前沿性能，证明 hierarchical reasoning 架构在数据效率上的优势，值得在低资源语言场景探索。
4. **数据重复策略**：对小规模高质量丹麦语数据集进行 10× 重复，确保覆盖度，是一种有效的数据稀缺应对策略。
5. **评估配置标准化**：统一使用 temperature=0 和固定 seed，并报告 Greedy decoding 与 thinking mode 两种模式的结果，便于复现与对比。

## 关键术语表
**HRM (Hierarchical Reasoning Model)**：一种分层推理架构，通过交替执行高层（H）和低层（L）推理周期，结合截断反向传播提升训练效率。

**Permissible Data**：符合许可规范、不含个人信息或版权侵权的训练数据，包括公开授权、协议提供或欧盟文本数据挖掘例外允许的数据。

**Transplant Datasets**：通过 LLM 生成并经过审计的合成数据，用于替代原始训练中不符合许可要求的数据集。

**Sapient**：由 Flan、Platypus、Tasksource 等 107 个子集合组成的大型指令微调数据集集合，本文使用了其中的精选版本。

**FSDP (Fully Sharded Data Parallelism)**：一种分布式训练策略，将模型参数、梯度和优化器状态在多个 GPU 间分片存储以减少显存占用。

**RoPE (Rotary Position Embedding)**：旋转位置编码，一种用于 Transformer 的位置编码方法，通过旋转矩阵实现相对位置信息的注入。

**Exact-match**：精确匹配评估指标，要求模型生成的答案与标准答案完全一致，常用于数学推理和问答任务。

**Common Pile**：一个 8TB 的公共领域和开放授权文本数据集，本文用于英文合成数据生成。

## 可复现要素
- **数据集**：161 个数据集几乎全部公开在 HuggingFace Hub，部分协议数据（DBC、Lex.dk）因许可限制无法公开；合成移植数据集已开源。
- **代码**：框架基于 Sapient 的 HRM-Text 代码开源，链接见论文脚注 4。
- **模型权重**：已发布在 Hugging Face Hub：https://huggingface.co/danish-foundation-models/DFM-Mimir
- **关键超参**：LR=3×10⁻⁴，warmup=2,000 步，global batch size=262,144，gradient accumulation=2，hidden size=1,536，32 layers，12 attn heads，expansion=4，H cycles=2，L cycles=3，BP max steps=5，RoPE θ=10,000。

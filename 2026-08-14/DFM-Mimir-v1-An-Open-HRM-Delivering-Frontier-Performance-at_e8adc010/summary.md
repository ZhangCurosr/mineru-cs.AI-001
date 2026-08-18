---
title: "DFM-Mimir-v1-An-Open-HRM-Delivering-Frontier-Performance-at"
source: https://arxiv.org/pdf/2608.13517v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:26:16"
field: "低资源语言大模型与可授权数据训练"
keywords: ["Hierarchical Reasoning Model", "permissible data", "low-resource language", "Danish foundation model", "synthetic data", "open-source LLM", "post-training"]
innovations: ["仅用可授权后训练数据从头训练1B参数HRM模型，实现英语/丹麦语前沿性能", "合成移植数据集策略：用Gemma4 31B生成+人工审核替代不可授权Sapient数据", "将多项选择题训练数据重构为自由生成范式以对齐exact-match评估基准"]
benchmarks: ["GSM8K", "MATH", "HumanEval", "MMLU", "BoolQ", "Winogrande", "DROP", "DaLA", "GEC", "WikiQA"]
---

# 论文速读：DFM-Mimir-v1-An-Open-HRM-Delivering-Frontier-Performance-at-1B-Parameters-Using-Only-Permissible-Post-Training-Data

## 一句话总结
论文提出了 **Mimir v1**，一个基于 Hierarchical Reasoning Model (HRM) 架构的 1B 参数语言模型，仅使用可授权的后训练数据从头训练，在英语和丹麦语任务上实现了具有竞争力的前沿性能，尤其为低资源语言丹麦语树立了新的最优基准。

## 研究问题与动机
1. **大型 LLM 开发依赖大量非可授权数据**：当前主流 LLM 依赖庞大且常涉及版权问题的多阶段训练流程，形成较高的研究门槛，尤其不利于坚持开源与伦理数据使用的团队。
2. **丹麦语等低资源语言缺乏高质量基础模型**：丹麦基础模型项目坚持仅使用可授权、公开许可或欧盟文本数据挖掘例外允许的数据，导致可用高质量数据有限，从头训练有竞争力的模型困难。
3. **现有 HRM 工作使用了不符合 DFM 授权的训练数据**：原始 HRM-Text 所依赖的部分数据集（如 Sapient）与丹麦基础模型的许可哲学不符，需要寻找替代方案。
4. **小参数模型的高效潜力未被充分挖掘**：希望通过合理的架构设计（HRM）和数据策略，在 1B 参数量级实现接近更大模型（如 Qwen 3.5 4B、Gemma 4 E2B）的性能。

## 核心贡献（创新点）
1. **提出了首个完全基于可授权后训练数据的 1B 参数 HRM 模型**：与已有 HRM-Text 工作相比，Mimir v1 排除了所有包含个人信息、版权侵权或非许可数据，在保持性能的同时满足伦理与法律合规要求。
2. **设计了"合成移植数据集"(Synthetic Transplant Datasets)策略**：用由 Gemma4 31B 生成并经人工审核的合成数据替代原始 Sapient 中不可授权的 Flan/Platypus/tasksource 数据，在保持功能覆盖的同时消除版权风险，部分任务性能甚至超越原版。
3. **将训练数据从多项选择题偏向自由生成范式**：原始 Sapient 数据以多选分类任务为主，Mimir v1 将其重构为开放生成形式（如 "amazonfood-summary-text-generation"），更好地对齐 GSM8K、MATH 等以精确匹配评分的评估基准。
4. **在 1B 参数下实现丹麦语任务的新 SOTA**：丹麦语 DaLA、GEC、WikiQA 等任务全面超越所有对比基线（包括 8-9B 的 Munin 系列），证明小模型在低资源语言上可通过数据策略实现突破。

## 方法详解
**架构设计**：
- 基于 HRM-Text 架构，隐藏维度 1536，32 层（半层结构），每层 12 个注意力头，FFN 扩展因子 4。
- 层级推理配置：2 个 H-cycles（高层推理循环）和 3 个 L-cycles（低层推理循环），截断反向传播限制 5 步。
- 位置编码：RoPE（θ = 10,000），预归一化（pre-norm，ε = 10⁻⁶）。

**数据处理流程**（七种数据形式）：
1. **Reformatted**（65.96%）：直接将 HuggingFace 仓库转为训练格式。
2. **Curated + reformatted**（16.91%）：从 Sapient 中精选 107 个子集合后再格式化。
3. **Synthetic + audited**（11.08%）：使用 Gemma4 31B 生成后经人工审核，接受率从个位数到 90%+ 不等，包括 span-filling、denoising、reordering、continuation 等任务类型。
4. **Tool-call formatted**（2.65%）：用于 agentic 训练的 native tool-calling 数据。
5. **Translated + audited**（2.26%）：OpenHermes 类数据和 DA/EN 翻译数据经过修复与审核。
6. **Agreement-supplied**（0.95%）：来自丹麦基础模型协议（DBC、Lex.dk），因许可限制无法公开共享。
7. **Derived task**（0.18%）：从已有数据集衍生出新任务格式。

**训练配置**：
- 使用 Gemma-4 tokenizer（区别于原始 HRM-Text 的自定义 tokenizer）。
- 应用 Gemma 4 chat template 学习对话结构。
- 优化器：AdamW，峰值学习率 3×10⁻⁴，2000 步线性 warmup，之后恒定（min ratio 1.0）。
- 全局批量大小 262,144 tokens，梯度累积 2，每加速器 batch size 16384，适配 4 个 4096 长度上下文。
- FSDP（Fully Sharded Data Parallelism），bfloat16 计算，fp32 收集精度。
- EMA decay = 0.9999，weight decay = 0.1。
- 训练 1.65M 步，8 块 NVIDIA B200 GPU，总耗时不到 3 周，平均每步约 1.1 秒。

**数据分布**：
- 161 个数据集，每 epoch 约 70.5B tokens。
- 英语 68.6%，丹麦语 24.7%，双语 da+en 6.5%，其他 0.2%。
- Top 10 数据集占总 token 的 66.5%，top 3 占 38.1%（高度集中）。
- lærebogen 重复 4×（8.32B tokens，占比 11.8%），8 个小丹麦数据集重复 10×。

## 实验与结果
**评测设置**：
- 20 个基准测试，分英语、数学&代码、丹麦语三类。
- temperature=0（greedy decoding），固定 seed=4242。
- 使用 vLLM-served 端点（Mimir 因需 FlashAttention 而特殊处理），结果以 HuggingFace Transformers 为准以确保可复现。
- 英语部分评测使用 few-shot（shooting 数遵循原 HRM-Text 配置），丹麦语全部 0-shot。

**核心结果**：

| 类别 | Mimir 1B 平均 | 最佳对比模型 | 差距 |
|---|---|---|---|
| 英语 | **69.0** | Qwen 3.5 4B (69.3) | -0.3 |
| 数学&代码 | **64.1** | SmolLM3 3B (67.9) / Gemma 4 E2B (75.4) | -3.8 vs SmolLM3 3B |
| 丹麦语 | **56.8** | 无更强1B以下基线 | 丹麦语新SOTA |

**关键数字**：
- **GSM8K**: Mimir 1B = 89.9（1B 组第一，全模型第二，仅次于 Gemma 4 E2B think 的 90.3）
- **HumanEval**: Mimir 1B = 56.7（超过 Qwen 3.5 2B 的 47.6）
- **MMLU**: 57.5，接近 Qwen 3.5 4B 的 75.8
- **DROP**: 83.1，大幅领先 Qwen 3.5 4B 的 48.0
- **DaLA**: 96.1 F1，远超所有对比模型（HRM-Text 仅 26.7）
- **GEC**: 85.6 EM，新 SOTA
- **WikiQA**: 66.8 EM，丹麦语 QA 任务最佳
- **相比 HRM-Text 1B**：数学&代码平均提升 36.7%（64.1 vs 46.9），HumanEval 从 0.0 提升到 56.7

## 相关工作脉络
1. **HRM-Text (Wang et al., 2026)**：Mimir v1 的直接前身架构，使用相同的 HRM-Text 框架但不同的 tokenizer 和训练数据。本文通过合成移植数据和自由生成任务重塑，弥补了原始工作在数据授权和部分任务类型上的不足。
2. **Sapient 训练数据集合**：包含 Flan NIV2、Platypus、tasksource 等子集合，原为多项选择题主导。本文通过合成重建将其转化为开放生成格式，改变了训练分布。
3. **Qwen 3.5 系列**：作为 0.8B/2B/4B 不同规模的对比基线，展示 Mimir 1B 在英语和丹麦语上与更大参数模型的竞争力。
4. **Gemma 4 E2B**：5B 总量/2.3B 有效参数的能耗优化模型，是本文重点对比对象，显示 Mimir 在 1B 下接近其部分能力。
5. **Danish Foundation Models 项目**：本研究所属的国家级计划，强调仅使用可授权数据，与商业 LLM 项目的数据策略形成鲜明对比。
6. **OpenMathInstruct-2、AceReason-1.1-SFT**：数学推理领域的重要开源数据集，在本文被作为关键训练数据引入（各贡献 6.6B 和 1.95B tokens）。
7. **Dolci、Tulu 3、Nemotron SFT**：高质量英语指令微调数据集的代表，支撑了本文 13.58B 英语指令数据。

## 局限性与未来方向
1. **数学&代码能力仍落后于更大模型**：在 GSM8K、MATH、HumanEval 上虽为 1B 组最优，但与 Gemma 4 E2B（5B/2.3B effective）仍有明显差距（如 MATH 45.8 vs 64.2）。
2. **助手能力有限**：尽管使用 Gemma 4 chat template 训练，但模型作为 assistant 的能力与 SOTA 仍有距离，需引入 reinforcement learning 等后续训练。
3. **部分数据仍无法完全公开**：Agreement-supplied 数据（如 DBC、Lex.dk 来源）因许可限制无法开源，尚未实现完全的 data openness。
4. **HRM 架构的 scaling behavior 未知**：当前仅验证了 1B 参数规模，更大参数下的表现尚待探索。
5. **单语言 tokenizer 限制**：虽然使用 Gemma-4 tokenizer 支持多语言，但相较专门设计的多语言 tokenizer 可能在丹麦语上有进一步优化空间。
6. **未来方向**：探索 HRM 的 scaling law、引入 RL 训练、推进数据完全开源、以及继续优化低资源语言性能。

## 研究启发与可借鉴点
1. **合成数据替代策略具有推广价值**：用高质量 LLM（如 Gemma4 31B）生成 + 人工审核的方式替代不可授权数据，可在不牺牲功能覆盖的前提下解决版权与合规问题，适用于其他坚持伦理数据使用的研究项目。
2. **从多项选择到自由生成的数据重构**：将原本以分类为核心的数据集（如 Flan、tasksource）转化为开放生成格式，能更好对齐以 exact match 评分的推理基准（GSM8K、MATH、DROP），这一策略对数学推理训练有借鉴意义。
3. **数据高度集中 + 小数据反复采样的策略**：Top 10 数据集占 66.5% tokens，同时小丹麦数据集 10× 重复，既保证质量又弥补低资源数据不足，可作为低资源语言模型训练的参考范式。
4. **HRM 架构在小参数下的效率优势**：1B 参数即可达到接近 4B 模型的英语性能和 5B 模型的数学性能，证明层级推理机制在资源受限场景下的实用价值，值得在其他语言/任务上验证。
5. **端到端可复现的训练记录**：公开完整的超参数、硬件配置（8× B200）、训练步数（1.65M）和耗时（<3 周），为其他团队复现和小模型训练提供了清晰的参考模板。

## 关键术语表
**Hierarchical Reasoning Model (HRM)**：一种分层推理的 Transformer 变体架构，通过多层级推理循环（H-cycles 和 L-cycles）在自回归生成过程中引入中间推理步骤，以提升推理能力。
**Permissible Post-Training Data**：符合版权、隐私和法律合规要求的数据，包括公开许可、协议授权使用或欧盟文本数据挖掘例外允许的研究用途数据。
**Synthetic Transplant Datasets**：通过 LLM 生成并经人工审核合成的数据集，用于替代原始训练中不可授权或不符合许可要求的数据，同时保持任务功能的等效性。
**Gemma-4 Tokenizer**：Google Gemma 4 模型使用的子词词表，本文替代原始 HRM-Text 的自定义 tokenizer 以更好地支持多语言和对话格式。
**Fully Sharded Data Parallelism (FSDP)**：PyTorch 中的分布式训练策略，将模型参数、梯度和优化器状态分片到多个 GPU 上以支持大模型训练。
**NVIDIA B200**：NVIDIA Blackwell 架构的高端 GPU，具备 180 GB HBM3e 内存，用于本文的高效分布式训练。
**Exact Match (EM) Scoring**：要求模型输出与标准答案完全一致才计分的评估方式，常见于 GSM8K、MATH、DROP 等推理基准。
**vLLM-served**：基于 PagedAttention 的高性能 LLM 推理服务框架，本文用于标准化基准评测中的模型部署。

## 可复现要素
- **数据集**：161 个数据集，绝大部分在 HuggingFace Hub 公开（如 `danish-foundation-models/laerebogen`、`nvidia/OpenMathInstruct-2`、`allenai/Dolci-Instruct-SFT-No-Tools` 等）；少数 Agreement-supplied 数据（DBC、Lex.dk）因许可限制未公开。完整数据集列表见 Appendix A。
- **代码**：训练框架开源，基于 Sapient 的 HRM-Text 代码改进（论文标注脚注 4 提供链接）。
- **模型权重**：已上传至 Hugging Face Hub：`https://huggingface.co/danish-foundation-models/DFM-Mimir`。
- **关键超参**：
  - Hidden size: 1536，Layers: 32，Heads: 12，Expansion: 4
  - H cycles: 2，L cycles: 3，BP max steps: 5，Warmup ratio: 0.2
  - RoPE θ: 10,000，Pre-norm ε: 10⁻⁶
  - LR: 3×10⁻⁴，Warmup: 2000 steps，Global batch: 262,144，Grad accum: 2
  - Adam β₁/β₂: 0.9/0.95，Weight decay: 0.1，EMA decay: 0.9999
- **训练硬件**：8 × NVIDIA B200 GPU (180 GB HBM3e)，总计约 1.65M 步，<3 周。
- **推理框架**：HuggingFace Transformers（确保可复现），vLLM 仅用于部分基线对比。

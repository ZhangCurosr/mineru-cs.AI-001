---
title: "Instruction-Alignment-for-Binary-Code-Representation-Learnin"
source: https://arxiv.org/pdf/2608.11766v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:33:23"
field: "二进制代码表示学习"
keywords: ["Binary Code Similarity", "Representation Learning", "Instruction Alignment", "Contrastive Learning", "Binary Analysis"]
innovations: ["首次将指令对齐作为辅助目标用于二进制表示学习", "提出多正样本InfoNCE损失处理一对多指令对应关系", "MAS指标与协同评分提供可解释的细粒度相似性证据"]
benchmarks: ["BinKit", "BinaryCorp"]
---

# 论文速读：Instruction-Alignment-for-Binary-Code-Representation-Learnin

## 一句话总结
本文提出 **InsnAlign**，通过利用编译器调试信息建立指令级语义对应关系，将指令对齐作为辅助训练目标，与函数级对比学习联合优化，从而提升二进制代码表示学习的质量与可解释性。

## 研究问题与动机
1. **现有方法仅利用粗粒度监督信号**：当前二进制代码表示学习方法主要依赖函数级符号匹配（通过不同编译配置下共享同一符号的函数构建正负对），忽略了编译器调试信息中蕴含的指令级细粒度语义对应。
2. **指令级语义未被显式利用**：现代编译器生成的调试信息（如DWARF格式）提供了每条汇编指令到源代码行的映射，这种细粒度映射为学习更准确的二进制表示提供了丰富监督信号，但被现有方法忽视。
3. **函数级训练已隐式改善指令对齐**：初步研究表明，经过函数级对比学习微调的模型在指令对齐检索任务上显著优于预训练模型，说明指令级理解与函数级嵌入质量存在强相关性。
4. **缺乏可解释性**：现有方法仅输出函数级嵌入距离，无法提供相似性判断的细粒度证据；显式指令对齐训练可识别哪些具体指令对应，提供可检查的相似度证据。

## 核心贡献（创新点）
1. **首次提出指令对齐度量方法**：将指令对齐形式化为检索任务，使用MRR和Recall@1进行定量评估，揭示现有模型在指令级语义捕捉上的能力边界。
2. **设计InsnAlign训练框架**：在Transformer基线模型（jTrans、CLAP）上增加指令对齐辅助目标，使用多正样本InfoNCE损失显式学习指令级语义对应关系。
3. **证明辅助目标的双重收益**：不仅提升指令级对齐质量（Recall@1提升50.9%-88.2%），还显著改善函数级二进制代码相似度分析性能（Recall@1提升2.7%-27.9%）。
4. **提供可解释的相似性证据**：通过指令对齐矩阵和Mean Alignment Score (MAS) 提供细粒度、可检查的相似性判断依据，在难负样本区分上优于函数级嵌入。
5. **验证方法的鲁棒性与泛化性**：在多种编译器（GCC/Clang）和优化级别（O0/O3）组合下保持一致提升，且对标签噪声具有较强容忍度（20%噪声下仍显著优于基线）。

## 方法详解
**整体框架**：InsnAlign在现有Transformer二进制代码嵌入模型基础上， augmentation函数级对比学习与指令对齐两个训练目标。

**关键设计**：

1. **指令嵌入提取**：
   - 对Transformer输出的token级hidden states $\mathbf{H} \in \mathbb{R}^{S \times d}$，通过指令索引映射 $\psi$ 将属于同一指令的token进行mean-pooling，得到指令级嵌入 $\mathbf{e}_k$：
   $$\mathbf{e}_k = \frac{1}{|\mathcal{T}_k|} \sum_{t \in \mathcal{T}_k} \mathbf{h}_t$$

2. **指令对齐矩阵构建**：
   - 利用调试信息建立指令-源码行映射 $\phi: b_i \mapsto l_j$
   - 对于来自同一源函数的两个二进制函数 $A$ 和 $B$，构建二值对齐矩阵 $\mathbf{M} \in \{0,1\}^{n \times m}$，其中 $\mathbf{M}_{ij}=1$ 当且仅当 $a_i$ 和 $b_j$ 映射到同一源码行

3. **指令对齐损失（InfoNCE）**：
   - 计算指令相似度矩阵 $\mathbf{S}_{ij} = \cos(\mathbf{e}_i^A, \mathbf{e}_j^B)$
   - 前向损失（A→B）：
   $$\mathcal{L}_i^{A \to B} = \log \sum_{j=1}^{m} \exp(\mathbf{S}_{ij}/\tau) - \log \sum_{j:\mathbf{M}_{ij}=1} \exp(\mathbf{S}_{ij}/\tau)$$
   - 采用多正样本泛化（受Supervised Contrastive Learning启发），所有 $\mathbf{M}_{ij}=1$ 的指令视为正样本
   - 总对齐损失为双向平均：
   $$\mathcal{L}_{\text{align}} = \frac{1}{|\mathcal{T}^{A\to B}| + |\mathcal{T}^{B\to A}|} \left( \sum_{i \in \mathcal{T}^{A\to B}} \mathcal{L}_i^{A\to B} + \sum_{j \in \mathcal{T}^{B\to A}} \mathcal{L}_j^{B\to A} \right)$$

4. **联合训练目标**：
   $$\mathcal{L} = \mathcal{L}_{\text{triplet}} + \lambda \cdot \mathcal{L}_{\text{align}}$$
   - $\mathcal{L}_{\text{triplet}}$ 为标准三元组对比损失
   - $\lambda = 0.001$ 为权重系数，平衡两个目标

5. **层冻结策略**：冻结嵌入层和前10层encoder，仅微调上层参数，保持预训练知识同时适配指令级对齐。

## 实验与结果
**数据集**：
- **训练集**：BinaryCorp的训练划分（含调试信息子集），包含1,655,011个二进制函数、2,326,328对正样本
- **测试集**：BinKit（去重后），32个项目作为测试集，Coreutils用于验证

**评估基线**：
- jTrans（λ=0）：原始fine-tuned版本
- CLAP（λ=0）：基于RoBERTa的跨模态模型
- InsnAlign_jtrans：jTrans + 指令对齐
- InsnAlign_clap：CLAP + 指令对齐

**主要结果**：

| 模型 | 平均Recall@1 (O0→O3) | 提升幅度 |
|------|---------------------|---------|
| jTrans (λ=0) | 0.4011 | - |
| **InsnAlign_jtrans** | **0.4282** | **+6.8%** |
| CLAP (λ=0) | 0.6396 | - |
| **InsnAlign_clap** | **0.6590** | **+3.0%** |

- **指令对齐质量**：InsnAlign使jTrans的Recall@1提升50.9%，CLAP提升88.2%
- **难负样本区分**：MAS的AUC达0.720（jTrans基线0.644），Cohen's d达0.778（基线0.435）
- **协同评分**：结合MAS与函数级余弦相似度的synergy scoring，jTrans基线Recall@1提升至0.5475（+27.9%）
- **硬负采样**：与hard negative mining兼容，InsnAlign_clap在硬负采样下达到0.6829平均Recall@1
- **标签质量**：编译器标签86.8%正确、5.3%合理、2.8%可疑/错误；20%人工噪声注入下仍保持0.634 Recall@1

## 相关工作脉络
1. **jTrans [60]**：jump-aware Transformer，本文主要基线之一； insnAlign在其fine-tuned checkpoint上继续训练，验证指令对齐的增益。
2. **CLAP [53]**：利用自然语言监督的跨模态二进制表示学习，达到当时SOTA；本文证明即使在该强基线上仍可进一步提升。
3. **Asm2vec [9]**：基于PV-DM随机游走的二进制克隆搜索；属于早期词向量类方法，仅使用函数级监督。
4. **Gemini [67] / SAFE [40]**：图神经网络和自注意力方法；代表BCSA主流研究方向，但均局限于函数级嵌入。
5. **PalmTree [28]**：学习指令级嵌入的预训练任务，但未设计跨函数指令对齐目标；本文明确将其排除在评估外。
6. **DeepBinDif [10]**：基本块级嵌入用于二进制diffing；依赖字符串等语义无关特征，与本文的语义对齐思路不同。

## 局限性与未来方向
1. **上下文窗口限制**：模型输入长度限制为1024 tokens，高度优化的函数可能因截断导致关键指令丢失。
2. **不可见调用语义**：O0下的包装函数委托给callee，其实际语义在query中不可见，导致对齐模糊。
3. **近亲双胞胎函数**：仅相差单个分支或调用目标的函数（如xvprintf vs xvfprintf）难以区分。
4. **操作数级差异**：涉及内存大小、魔法常量等细微差异的函数，指令级相似度高但语义不同。
5. **调试信息依赖**：方法需要带调试信息的二进制文件， stripped binary场景受限。
6. **潜在应用**：论文初步探索了patch presence detection应用，但需克服上述局限才能实用化。

## 研究启发与可借鉴点
1. **细粒度辅助目标的迁移价值**：将粗粒度任务（函数级相似度）与细粒度监督（指令级对齐）结合的思路，可迁移至其他代码表示学习场景（如AST级、CFG级对齐）。
2. **多正样本InfoNCE设计**：针对一对多指令对应关系（宏展开、指令拆分）的多正样本contrastive loss设计，适用于其他存在多对多语义对应的表示学习任务。
3. **可解释性评估指标**：提出Mean Alignment Score (MAS) 和Cohen's d来量化指令级信号的判别力，为模型可解释性评估提供了可复用的量化框架。
4. **层冻结策略的沿用**：冻结底层encoder仅微调上层的设计，既节省计算又保留预训练知识，可借鉴于其他二进制分析的fine-tuning场景。
5. **数据泄漏防护意识**：避免使用与训练集有重叠的测试集（原jTrans测试集会泄露指令对齐性能），这一严谨做法值得在后续工作中贯彻。

## 关键术语表
**Binary Code Similarity Analysis (BCSA)**：通过比较二进制函数嵌入来判断其功能相似性的任务，是二进制分析的核心问题。

**Instruction Alignment**：利用编译器调试信息建立不同二进制变体中来自同一源码行的汇编指令之间的语义对应关系。

**InfoNCE Loss**：信息噪声对比损失，本文将其泛化为多正样本版本，用于优化指令级对齐。

**Mean Alignment Score (MAS)**：双向最大指令相似度平均值，用于量化函数对的指令对齐质量。

**Hard Negative Mining**：从Top-K相似候选中选择非正样本作为负例，提升对比学习的判别能力。

**Synergy Scoring**：结合函数级余弦相似度与指令级MAS的加权融合评分，用于重排序提升检索精度。

**DWARF**：标准的调试信息格式，提供汇编指令到源代码行的映射关系。

**BinaryCorp / BinKit**：两个大规模二进制代码相似度分析数据集，前者用于训练，后者用于评估。

## 可复现要素
- **数据集**：BinaryCorp（训练）、BinKit（评估）；论文已去重处理并公开清洗后的数据
- **代码**：已开源，链接 https://github.com/whj0401/InsnAlign
- **权重**：Zenodo DOI: 10.5281/zenodo.19343892
- **关键超参**：学习率 $1 \times 10^{-5}$，冻结层数 $k=10$，损失权重 $\lambda=0.001$，训练2个epoch
- **硬件**：NVIDIA A6000 GPU (48GB)，256GB RAM，AMD Threadripper 3970X CPU
- **反汇编工具**：IDA Pro

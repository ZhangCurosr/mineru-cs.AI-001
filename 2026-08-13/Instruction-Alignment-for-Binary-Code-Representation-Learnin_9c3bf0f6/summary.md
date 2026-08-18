---
title: "Instruction-Alignment-for-Binary-Code-Representation-Learnin"
source: https://arxiv.org/pdf/2608.11766v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:34:53"
field: "二进制代码相似性分析"
keywords: ["Binary Code Similarity Analysis", "Instruction Alignment", "Contrastive Learning", "Representation Learning", "Debug Information", "InfoNCE"]
innovations: ["首次将指令级对齐作为显式辅助目标引入二进制备表征学习，提出多正样本InfoNCE对齐损失", "提出指令对齐检索评测任务，用MRR和Recall@1量化模型细粒度语义捕获能力", "指令对齐证据提供可解释的相似性判断依据，在hard negative设置下比函数级嵌入更具区分力"]
benchmarks: ["BinKit", "BinaryCorp"]
---

# 论文速读：Instruction-Alignment-for-Binary-Code-Representation-Learnin

## 一句话总结
本文提出 **InsnAlign**，利用编译器调试信息建立指令级对齐关系，将其作为辅助训练目标（多正样本 InfoNCE loss）与函数级对比学习联合训练，从而在二进制代码表征学习中同时提升指令级语义捕获能力和函数级检索精度。

## 研究问题与动机
- **现有方法粒度粗**：当前二进制代码表征学习主要依赖函数级符号匹配（同一源函数在不同编译配置下的函数构成正对），仅利用编译器产生的粗粒度监督信号，忽略了调试信息中包含的"汇编指令↔源代码行"细粒度映射。
- **指令级语义未被显式建模**：虽然函数级对比训练隐式提升了模型对指令语义的捕获能力，但缺乏显式的指令对齐优化目标，导致对难区分样本的可解释性证据不足。
- **初步实验验证了可行性**：作者在 jTrans 上测量了指令对齐检索性能，发现微调后的模型比预训练版本在 MRR/Recall@1 上均有显著提升（如 O0→O2 的 Recall@1 从 0.524→0.574），说明指令级理解质量与函数级嵌入质量强相关，值得显式优化。

## 核心贡献（创新点）
1. **首次提出指令对齐评测任务**：将指令对齐形式化为检索问题，用 MRR 和 Recall@1 量化评估现有模型在指令级语义捕获上的能力。与已有工作相比，此前文献仅关注函数级相似度，无细粒度指令层评估。
2. **提出 InsnAlign 训练框架**：利用调试信息构建指令对齐矩阵 M，设计基于多正样本 InfoNCE 的指令对齐损失 $\mathcal{L}_{\text{align}}$，与函数级 triplet loss 联合优化。与现有方法的区别在于：首次将指令级对齐作为显式辅助目标引入二进制备表征学习。
3. **提供可解释的相似性证据**：模型不仅能输出函数级相似度，还能通过指令对齐热力图展示具体哪些指令对对应，为难以区分的候选对提供更 discriminative 的判断依据。与 LLM 后验生成的自然语言解释不同，对齐证据直接源自同一组指令嵌入。
4. **系统验证了方法的稳健性与可扩展性**：证明对齐标签容忍 20% 随机噪声，且与 hard negative mining 和 synergy re-ranking 策略正交互补。

## 方法详解
- **指令嵌入提取**：将二元函数表示为 token 序列，经 Transformer 得到隐藏状态 $\mathbf{H} \in \mathbb{R}^{S \times d}$，通过均值池化将 token 级表示聚合为指令级嵌入：$\mathbf{e}_k = \frac{1}{|\mathcal{T}_k|}\sum_{t \in \mathcal{T}_k} \mathbf{h}_t$，其中 $\mathcal{T}_k$ 是归属第 $k$ 条指令的 token 集合。此过程不引入额外可学习参数。
- **指令对齐矩阵构建**：给定来自同一源函数的两个二元函数 $A$ 和 $B$，利用调试信息建立二元匹配矩阵 $\mathbf{M} \in \{0,1\}^{n \times m}$，其中 $\mathbf{M}_{ij}=1$ 当且仅当指令 $a_i$ 和 $b_j$ 源自同一源代码行。
- **指令对齐损失（InfoNCE）**：计算指令相似度矩阵 $\mathbf{S}_{ij} = \cos(\mathbf{e}_i^A, \mathbf{e}_j^B)$，对每个有至少一个正匹配的查询指令计算：
$$\mathcal{L}_i^{A \to B} = \log \sum_{j=1}^{m} \exp(\mathbf{S}_{ij}/\tau) - \log \sum_{j:\mathbf{M}_{ij}=1} \exp(\mathbf{S}_{ij}/\tau)$$
该损失是标准 InfoNCE 的多正样本扩展：所有正匹配共享概率质量，而非强制一对一匹配。双向对称计算后取平均得 $\mathcal{L}_{\text{align}}$。
- **联合训练目标**：$\mathcal{L} = \mathcal{L}_{\text{triplet}} + \lambda \cdot \mathcal{L}_{\text{align}}$，其中 $\lambda = 0.001$，$\tau$ 为温度参数。
- **层冻结策略**：冻结 embedding 层和前 10 个 BERT encoder 层，仅微调上层，保留预训练知识的同时适配指令级对齐。
- **训练数据**：复用 jTrans 的 BinaryCorp 训练子集（含调试信息的 1,655,011 个二元函数，2,326,328 对正样本）。

## 实验与结果
- **数据集**：训练使用 BinaryCorp 含调试信息子集；评估使用 BinKit（去重后 32 个项目为测试集，Coreutils 为验证集），覆盖 gcc-11/clang-13/gcc-4/clang-4 在 O0 vs O3 下的 16 组跨编译配置。
- **基线模型**：jTrans（自定义 Transformer）、CLAP（RoBERTa-base + 跨模态训练）。
- **主要结果（Recall@1）**：
  - InsnAlign_jtrans 平均 0.4282 vs jTrans(λ=0) 0.4011，提升 **+6.7%**；InsnAlign_clap 平均 0.6590 vs CLAP(λ=0) 0.6396，提升 **+3.0%**。
  - 结合 synergy re-ranking（MAS + 函数余弦相似度加权）后：InsnAlign_jtrans 达到 0.5475（较 baseline 提升 **+18.7%→+27.9%**），InsnAlign_clap 达到 0.6733（+2.17%）。
  - 指令对齐检索：InsnAlign_jtrans 在 Recall@1_insn 上比基线提升 **50.9%**，InsnAlign_clap 提升 **88.2%**。
- **可解释性**：MAS 在 hard negative（Top-5）设置下 AUC 达 0.720（jTrans）和 0.702（CLAP），显著优于函数级 cosine 的 0.584 和 0.698；Cohen's d 分别为 0.778 和 0.622，优于基线的 0.435 和 0.236。
- **最强结果**：InsnAlign_clap + hard negative mining + synergy re-ranking，Recall@1 达 **0.6978**（160K pool 规模）。

## 相关工作脉络
1. **jTrans [60]**：自定义 Transformer，jump-aware 编码，函数级对比学习。本文在其微调 checkpoint 上追加指令对齐辅助损失，属于即插即用的增强。
2. **CLAP [53]**：RoBERTa-base + 自然语言监督的跨模态训练，当前 BCSA SOTA。本文同样在此基础上验证指令对齐的通用性。
3. **Asm2vec [9] / SAFE [40] / Gemini [67]**：基于 Word2Vec、自注意力、GNN 的函数级二进制备表征方法，均仅利用函数级符号匹配作为监督，无指令级信号。
4. **PalmTree [28]**：学习指令级嵌入但通过预训练任务，未设计跨函数的指令对齐，故未纳入本文对比。
5. **DeepBinDif [10]**：基本块级嵌入用于二进制 diffing，但依赖字符串等语义无关特征，且需昂贵的成对计算。本文的指令对齐可直接用于大规模检索。
6. **LLM-based 方法 [24, 50, 66]**：提供可读的解译产物（反编译代码/摘要），但与模型内部相似度决策脱耦；本文的对齐热力图直接与嵌入空间关联，提供内生的可解释证据。

## 局限性与未来方向
- **输入长度限制**：模型 context window 为 1024 tokens，高度优化的 -O3 函数（大量内联调用）会截断判别性指令，导致 true match 的 MAS 偏低。
- **简单包装函数（wrapper）查询**：-O0 下的薄包装函数仅做参数传递，其指令缺乏判别性，容易与多个相似 wrapper 产生高相似度误匹配。
- **近双胞胎兄弟函数**：仅相差一个调用或分支的函数（如 xvprintf vs xvfprintf）在指令级几乎无法区分。
- **常数/尺寸差异敏感不足**：结构相同的初始化函数（如 md5_init_ctx vs sha224_init_ctx）仅常数和大小不同，指令级嵌入难以捕捉 operand 级差异。
- **调试信息质量依赖**：编译器优化可能导致指令↔源码行映射不准确（本文评估仅 2.8% 为可疑/错误，但仍存在）。
- **未来方向**：扩大上下文窗口、结合 operand 级分析、探索补丁存在检测（PPD）等下游应用。

## 研究启发与可借鉴点
1. **细粒度对齐可作为通用辅助目标**：指令对齐的 InfoNCE 框架可迁移到其他低级代码表征场景（如 LLVM IR、中间表示层面），利用编译器调试/映射信息提供额外监督。
2. **对齐指标作为评测探针**：指令级 MRR/Recall@1 可作为评估模型细粒度语义理解能力的补充指标，不仅看函数级准确率，还能诊断模型在难样本上的表现。
3. **多正样本 InfoNCE 的设计**：针对一对多/多对一匹配关系（一条源行可能展开为多条指令），采用多正样本泛化而非强制一对一匹配，此设计对代码相似性任务有普遍参考价值。
4. **Synergy re-ranking 策略**：先用粗粒度快速召回 top-100，再用细粒度 MAS 重排，兼顾效率与精度，可与 hard negative mining 组合使用。
5. **标签噪声鲁棒性验证范式**：通过随机注入 5%/10%/20% 噪声评估辅助损失的容错能力，为后续研究利用"银标"数据提供了参考实验设计。

## 关键术语表
- **Binary Code Similarity Analysis (BCSA)**：在不具备源代码的情况下，通过比较二进制函数嵌入向量判断两个函数是否源自同一源函数的任务。
- **Instruction Alignment**：利用编译器调试信息建立两个二元函数间指令级别的语义对应关系，即来自同一源代码行的汇编指令视为对齐。
- **InfoNCE Loss**：对比学习中的负采样损失函数；本文将其泛化为多正样本版本，使查询指令的概率质量分布在所有正匹配指令上。
- **Mean Alignment Score (MAS)**：双向指令最大相似度取平均，作为衡量一对函数指令级对齐质量的连续标量，可用于相似性判断的可解释证据。
- **Synergy Scoring**：将函数级余弦相似度与指令级 MAS 加权组合（γ=0.5），用于对 top-100 候选重排序以提升检索精度。
- **Hard Negative Mining**：从当前模型 top-5 最相似但语义不同的候选中采样负样本，使 triplet loss 更具挑战性。
- **BinaryCorp**：用于训练的开源数据集，包含 1544 个项目在多种编译配置下的二元函数，本文仅使用含调试信息的子集。
- **BinKit**：用于评估的大规模 BCSA 数据集，51 个开源项目、213,400 个二元文件，包含调试信息，本文去重后使用。

## 可复现要素
- **数据集**：BinaryCorp（训练，公开）、BinKit（评估，公开）；论文明确说明了去重和验证/测试划分方式。
- **代码/权重**：论文声明代码和数据已在 Zenodo（doi:10.5281/zenodo.19343892）公开，并提供 GitHub 仓库 https://github.com/whj0401/InsnAlign。
- **关键超参**：学习率 1e-5，冻结层数 10，损失权重 λ=0.001，训练 2 epochs，温度参数 τ 未明确给出具体值。
- **硬件**：NVIDIA A6000 GPU (48GB)，256GB RAM，AMD Threadripper 3970X。
- **反汇编工具**：IDA Pro。

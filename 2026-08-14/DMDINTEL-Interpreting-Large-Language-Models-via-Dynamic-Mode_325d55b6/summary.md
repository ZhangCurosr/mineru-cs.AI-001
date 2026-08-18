---
title: "DMDINTEL-Interpreting-Large-Language-Models-via-Dynamic-Mode"
source: https://arxiv.org/pdf/2608.13048v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:24:07"
field: "大语言模型可解释性"
keywords: ["可解释性", "大语言模型", "动态模态分解", "输入归因", "DMD", "HODMD", "LLM interpretability"]
innovations: ["将 SFT LLM 隐藏状态视为离散动力系统并用 DMD/HODMD 提取时空模态用于输入 token 归因", "提出 instruction bias 去噪与 cosine-similarity 启发式层选择两步预处理", "在 3 模型族×3 数据集上证明 DMDINTEL 在 MC/RBO/Recall 及 token-masking 保真度上全面优于 IG/SHAP/PCA"]
benchmarks: ["Sentiment (Amazon polarity)", "HateXplain", "FakeEdit"]
---

# 论文速读：DMDINTEL-Interpreting-Large-Language-Models-via-Dynamic-Mode

## 一句话总结
本文提出 **DMDINTEL**，一种将解码器类 LLM 的隐藏状态视为离散时间动力系统演化轨迹、利用动态模态分解（DMD）提取时空模态并对输入 token 进行排序归因的框架；在三个文本分类数据集和三种模型族上均优于 PCA、Integrated Gradients 与 SHAP。

## 研究问题与动机
- 现有 LLM 可解释性方法（IG、SHAP 等）将 token 视为离散实体，仅给出"静态快照"，忽略了隐藏状态沿序列的连续演化逻辑。
- MLP 层结构分析（如 Geva 等的 key-value memory 视角）仍以孤立向量表征各 token，无法刻画跨序列构建的上下文依赖与模式演化。
- 指令微调模型在隐藏表示中会编码大量 prompt 模板噪声（instruction bias），直接影响 DMD 等对噪声敏感的数据驱动方法的分解质量，需要显式去噪。
- 缺乏从"动力系统视角"统一刻画 decoder-only LLM 自回归推理过程中隐状态演化的输入归因框架。

## 核心贡献（创新点）
1. **提出 DMDINTEL 输入归因框架**：将 SFT LLM 的隐藏状态视为动力系统轨迹，用 DMD 分解出低维时空模态以识别驱动输出的关键输入 token；本质区别在于以序列演化视角替代传统"静态逐 token 打分"。
2. **设计 instruction bias 去噪与启发式层选择**：通过空 prompt 提取 bias 向量并作 $\hat{h}_i = h_i - b$ 去偏，并基于中后段各层 cosine similarity 曲线的梯度拐点自动选取 $down\_proj$ 最佳层；与既有方法相比，首次显式分离指令模板噪声与真实输入语义。
3. **同时支持 DMD 与 HODMD 两种分解并比较 ranking 策略**：采用初始幅值（amplitude）与时序平均幅值两种模态排序，并引入 Hankel 延迟嵌入以捕捉长程时序依赖；区别于只做单次 SVD/PCA 的静态方法，该方法显式建模 $h_{t+1} \approx A h_t$ 的演化算子。
4. **在 3 模型族 × 3 数据集的大规模实验中验证有效性**：DMDINTEL 在 matched count、RBO、recall@k 以及 token-masking 保真度上整体优于 IG/SHAP/PCA，且跨模型更稳定；这是首次将 DMD 作为 LLM 输入归因主线的系统评估。

## 方法详解
- **动力系统建模**：将 SFT LLM 看作离散映射 $\mathcal{F}$，第 $t$ 步隐藏状态 $h_t = \mathcal{F}(w_{\leq t})$；用线性算子近似 $h_{t+1} \approx A h_t$，其中 $A$ 由 DMD 从快照矩阵 $X = [h_1, \dots, h_n]$ 求解。
- **去噪（instruction bias removal）**：用一个不含输入句子的空 prompt 过模型，从目标层的 $down\_proj$ 提取 bias 向量 $b$，再作 $\hat{h}_i = h_i - b$，得到去偏快照矩阵 $\hat{X}$，以降低模板噪声对 DMD 的干扰。
- **启发式层选择（Algorithm 1）**：遍历中后段各层，计算每层所有 token 与 bias 的平均 cosine similarity $\bar{c}^{(\ell)}$；选取低于阈值 $\tau=0.25$ 且局部梯度 $\nabla \bar{c}^{(\ell)}$ 最大的层 $\ell^*$，作为"输入语义与指令跟随平衡"的最优层（一般为 18–27 层）。
- **标准 DMD**：对 $\hat{X}$ 做 SVD $\hat{X}=U\Sigma V^*$，构造约化算子 $\tilde{A}=U^* \hat{X}_2 V \Sigma^{-1} U$，特征分解得 DMD 模态 $\Phi$ 与复特征值 $\Lambda$；模态代表潜空间主导结构，特征值决定增长/衰减/振荡。
- **HODMD（高阶 DMD）**：构造 $d$-延迟 Hankel 矩阵 $\mathcal{H}$，把 $h_{t+d}$ 建模为前 $d$ 步的线性组合；再用相同 DMD 流程分解 $\mathcal{H}$，从而捕捉长程时序依赖（Appendix C）。
- **模态排名**：采用（i）初始幅值 $|\alpha_i|$、（ii）时序平均幅值两种方式选 top-$k$ 模态（实验中 $k=5$）；消融表明 top-7 会引入噪声导致性能下降。
- **Token 归因得分**：取状态增量 $\Delta h_t = h_t - h_{t-1}$ 作为该 token 的"贡献向量"，投影到每个选定的 DMD/HODMD 模态 $\phi_i$：
  $\alpha_{t,i} = \frac{|\langle \Delta h_t, \phi_i \rangle|}{\|\phi_i\|}$，总分为 $s_t = \sum_{i=1}^k \alpha_{t,i}$，按 $s_t$ 降序排列得到 token 重要性排名。
- **度量**：Average Matched Count（MC）、Rank-Biased Overlap（RBO，$p=0.95$）、Recall@k；保真度通过 top-k token 掩码后 Accuracy drop 与 Confidence drop 评估。
- **重建误差**：$ \hat{X}_{recon} = \Phi \mathrm{diag}(\alpha) [1, \lambda_i, \dots, \lambda_i^{N-1}]^\top$，相对 Frobenius 误差 $\epsilon \in [0.06, 0.13]$，说明线性近似足够精确。
- **效率**：仅需一次前向传播即可提取归因，GPU 内存与耗时均显著低于 IG/SHAP（Appendix G）。

## 实验与结果
- **模型**：Llama-3.2-3B-inst、Qwen3-4B-inst、Mistral-7B-v0.3-inst（全精度 SFT）。
- **数据集**：Sentiment（Amazon polarity）、HateXplain（仇恨文本）、FakeEdit（假新闻 Reddit 标题），每数据集取 1000 条测试样本。
- **基线**：IG、SHAP（GradientSHAP）、PCA（作用于 $\hat{X}^\top$ 后再用同样投影流程）。
- **Ground truth**：由 GPT-4.1 逐 token 排序生成（每句最多 20 个非停用词），两位人工标注员校验显示与 GPT-4.1 的 RBO 达 0.54–0.70。
- **Sentiment（Table 3，top-20）**：HODMD-avgamp 配置最优。Llama-3.2-3B-inst 负向 MC@20=5.35、RBO=0.25、Recall=0.68；正向 MC@20=6.12、RBO=0.27、Recall=0.76，普遍超越 IG/SHAP/PCA。
- **HateXplain（Table 4，top-10）**：标准 DMD-amp 最优。Llama MC@10=2.77、RBO=0.25、Recall=0.64；Mistral 表现最佳，RBO=0.30、Recall=0.67。
- **FakeEdit（Table 4，top-10）**：标准 DMD-amp 最优。Llama MC@10=5.46、Recall=0.76；Qwen MC@10=5.37、Recall=0.75，均在 recall 上领先所有基线。
- **定性（Table 5）**：DMDINTEL 与 GT 对齐最紧密，跨模型稳定；SHAP 在仇恨句上紧凑但遗漏 context，PCA/IG 不稳定且 IG 出现重复 token。
- **掩码保真度（Table 6）**：DMDINTEL 在多数设置下 accuracy/confidence drop 排名第一或第二，证明所定位关键 token 确实具有强因果贡献。
- **层敏感性（Table 7）**：$\ell^* \pm 1$ 造成的 Recall 最大偏差 $\Delta_{max} \le 0.02$，说明启发式选层稳健。
- **最强结果**：Sentiment 正向 Llama-3.2-3B-inst 上 Recall@20=0.76；FakeEdit 上 Llama MC@10=5.46、Recall=0.76。

## 相关工作脉络
- **IG / SHAP（Sundararajan 2017; Lundberg 2017）**：主流梯度/博弈论归因，视 token 为独立贡献单元；DMDINTEL 改用"状态增量投影到低维动力学模态"的视角，强调连续演化而非逐步扰动。
- **Probing / Mechanistic interpretability（Tenney 2019; Olsson 2022; Meng 2022）**：探测特定层/头的结构语义或电路；DMDINTEL 不针对单一头或层，而是从 MLP down\_proj 的序列轨迹整体提取跨层-跨 token 的低维模式。
- **LogitLens / TunedLens（Suhr 2021; Belrose 2023）**：把中间层表征解码回词表观察预测轨迹；DMDINTEL 则是在表征空间做谱分解，关注"哪些 token 驱动了哪类时空模式"。
- **MLP 作为 KV memory（Geva 2021）与高维线性近似（Hernandez 2023）**：揭示 MLP 内部线性性质；DMDINTEL 进一步把这些线性性利用为动力系统 $h_{t+1}=Ah_t$ 并做谱分解，从而从结构洞察走向可计算归因。
- **DMD 在 NLP 中的早期应用（Kumar 2019; Mao 2020; Vyshnav 2020）**：用于文本/语音序列特征提取；本文是首个将 DMD 系统化用于"decoder-only LLM 输入 token 归因"的工作。
- **PCA 作为归因基线**：PCA 仅对静态快照做方差最大化分解，无时序假设；DMD/HODMD 显式学习一步/多步转移算子，更适合刻画自回归过程中的连续信息流。

## 局限性与未来方向
- 仅评估分类任务（1–2 个输出 token），难以直接推广到多 token 生成（如机器翻译、长文本生成），因为自回归生成的 token 会反向进入序列，破坏 DMD 快照矩阵的构造假设（论文 Section 8 明确）。
- 仅针对监督指令微调模型；商业通用 instruction-tuned 大模型的归因尚未探索（Section 8）。
- DMD 本质为线性近似，对高度非线性/长程反馈结构（如复杂推理链）可能丢失细节；top-k 模态数、延迟参数 $d$ 的选择依赖启发式或调参。
- Ground truth 依赖 GPT-4.1 生成并与人工比对（RBO≈0.54–0.70），存在主观性（尤其是 HateXplain）。
- 未扩展到 encoder 架构（如 BERT）与多模态 LLM。

## 研究启发与可借鉴点
1. **动力系统视角归因的可迁移性**：将隐藏状态轨迹建模为 $h_{t+1}=Ah_t$ 并用 DMD/HODMD 做谱分解，可推广至其他序列任务（机器翻译、摘要、对话）中的"输入-生成对应"归因。
2. **Instruction bias 显式去噪**：空 prompt 减去 bias 向量的思路简洁有效，可作为通用预处理步骤用于其他依赖中间表示的方法（如 probing、attention rollout）。
3. **Cosine-similarity 启发式层选择**：基于 bias 相关性的单调曲线拐点定位最优层，无需网格搜索，可复用至其他需要选定解释层的 pipeline。
4. **Token 增量 $\Delta h_t$ 作为贡献代理**：用前后状态差代替原状态本身作为 token 的"输入贡献"，在逻辑上更贴近"该 token 带来了什么新信息"，可与 attention rollout、induced attention 等方法对比或结合。
5. **时序平均幅值 ranking 优于初始幅值（Sentiment）**：表明"持续存在的模式"比"瞬时爆发"更能捕捉语义；这一经验对选择模态排序准则有直接指导。

## 关键术语表
- **Dynamic Mode Decomposition (DMD)**：从时序快照中提取主导时空模态的谱分解方法，通过近似线性算子刻画非线性系统的演化特征。
- **Higher Order DMD (HODMD)**：在 DMD 基础上引入 $d$-步延迟嵌入（Hankel 矩阵），可捕捉更长程的时序依赖。
- **Koopman 算子 / 线性近似算子 A**：DMD 从数据中学习的最佳线性映射，满足 $h_{t+1} \approx A h_t$。
- **DMD 模态（$\phi_i$）**：算子 $A$ 的特征向量，代表系统中主导的空间结构。
- **Modal amplitude（模态幅值）**：各模态在初始/时序上的投影强度，用于模态重要性排序。
- **Instruction bias（指令偏差向量 $b$）**：空 prompt 过模型得到的隐藏表示，反映指令模板带来的系统性偏移，需从各 token 表示中减去。
- **Matched Count（MC）**：归因 Top-k 中与 ground truth 重合 token 的期望数量，不计 GT 总量。
- **Rank-Biased Overlap (RBO)**：考虑 rank 位置一致性的序列重合度度量，对 rank 越高位置权重越大（本文 $p=0.95$）。

## 可复现要素
- **代码/权重开源情况**：论文未提供公开代码仓库链接（PDF 正文与附录未提及 GitHub）；模型为开源 Llama-3.2-3B-inst、Qwen3-4B-inst、Mistral-7B-v0.3-inst 可通过 HuggingFace 获取。
- **数据集**：Sentiment（Amazon polarity，Zhang 2015）、HateXplain（Mathew 2021）、FakeEdit（Nakamura 2020）均为公开数据集。
- **关键超参**：去噪阈值 $\tau=0.25$、top-k 模态数 $k=5$、HODMD 延迟参数 $d$ 为自适应（论文未给出固定值）；RBO 持久因子 $p=0.95$。
- **训练环境**：单张 NVIDIA H100 进行 SFT；归因使用 NVIDIA L40。
- **PyDMD 库**：使用 PyDMD（Demo 2018）实现 DMD/HODMD。
- **GT 生成**：由 OpenAI GPT-4.1 通过提示词生成 token 级重要度排名（Appendix A 声明）。

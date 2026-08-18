---
title: "DMDINTEL-Interpreting-Large-Language-Models-via-Dynamic-Mode"
source: https://arxiv.org/pdf/2608.13048v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:25:17"
---

# 论文速读：DMDINTEL-Interpreting-Large-Language-Models-via-Dynamic-Mode

## 一句话总结
本文提出 DMDINTEL，一种基于动态模态分解（DMD）的输入归因框架，将解码器型 LLM 的隐藏状态序列视为离散时间动力系统，通过提取低维时空模态并计算 token 在该模态上的投影幅度，实现对分类任务关键输入 token 的高效排序，性能全面超越 IG、SHAP 与 PCA。

## 研究问题与动机
- 现有归因方法（IG、SHAP 等）将 token 视为离散静态实体，仅能给出“碎片化快照”，忽视了 LLM 处理序列时隐状态连续演化的动力学特征。
- 尽管 MLP 的 key-value 记忆机制已有揭示，但现有研究仍孤立分析单层向量，无法刻画跨 token 序列累积的决策逻辑。
- 指令微调模型会在隐藏状态中编码大量 prompt 模板噪声，如何剥离指令偏置并自动定位最能表征输入语义的 decoder 层尚缺乏系统化方案。
- 梯度/扰动类方法计算开销极高（需数十至数百次前向传播），亟需一种单次前向即可实现、兼具高精度与高效率的非侵入式归因范式。

## 核心贡献（创新点）
1. **首创 DMD 动力系统归因框架**：将 SFT 后 LLM 的隐藏状态演化抽象为离散动力系统，利用 DMD/HODMD 提取低维时空模态，并基于 token 投影幅度排序，本质区别在于从“静态切片”转向“时序流形”建模。
2. **指令偏置扣除与启发式选层**：通过空 prompt 提取偏差向量并相减去噪；提出基于余弦相似度曲线拐点的自动选层算法，避免人工网格搜索，直接定位输入语义与指令遵循的最佳平衡层。
3. **高效非侵入式实现与全面基准验证**：仅需单次前向传播完成归因，显存与耗时显著低于梯度类方法；在 3 种模型族、3 个分类数据集上统一超越 IG/SHAP/PCA，并在忠实度掩码实验中证明其定位 token 的强因果贡献。

## 方法详解
- **隐状态快照矩阵构建**：提取目标 decoder 层 MLP `down_proj` 的输出，按 token 时间步排列成 $\mathbf{X} = [h_1, h_2, \ldots, h_n]$，其中 $h_t = \mathcal{F}(w_{\le t})$。
- **指令偏置去噪**：运行仅含系统/用户 prompt 的空输入序列，提取同层隐状态得到偏差向量 $\pmb{b}$，去噪后 $\hat{h}_i = h_i - \pmb{b}$，构成 $\hat{\pmb X}$，消除模板带来的结构化噪声。
- **启发式最优层选择（Algorithm 1）**：遍历中间层至最后一层，计算各层平均余弦相似度曲线 $\mathcal{C}$。限制候选层 $\bar{c}^{(\ell)} < \tau$（默认 $\tau=0.25$），在其中取梯度最陡上升的下沿层 $\ell^*$，该层保留最多输入 token 语义且尚未过度偏向指令跟随。
- **DMD / HODMD 模态分解**：对 $\hat{\pmb X}$ 应用标准 DMD（线性 Koopman 近似 $h_{t+1} = \mathbf{A}h_t$）或 HODMD（构造 Hankel 矩阵引入延迟 $d$，捕捉长程时序依赖）。经 SVD 降维后特征分解获得空间模态 $\phi_i$ 与复特征值 $\lambda_i$。
- **Token 归因评分**：计算隐状态增量 $\Delta h_t = h_t - h_{t-1}$ 作为 token 贡献代理，投影至前 $k=5$ 个最优模态（按初始振幅或时间平均振幅排序），得分 $s_t = \sum_{i=1}^k \frac{|\langle \Delta h_t, \phi_i \rangle|}{\|\phi_i\|}$，按 $s_t$ 降序输出 token 重要性排名。

## 实验与结果
- **数据集与模型**：Sentiment（Amazon Polarity）、HateXplain、FakeEdit；Llama-3.2-3B-inst、Qwen3-4B-inst、Mistral-7B-v0.3-inst（全精度 SFT）。
- **评估指标**：MC@k、RBO@k（$p=0.95$）、Recall@k；忠实度通过 top-k token 掩码后的准确率/置信度下降度量；效率记录 GPU 显存与单样本耗时。
- **主要结果**：
  - **Sentiment**：HODMD + 时间平均振幅排序最优，Qwen3-4B-inst 正样本 Recall@20 达 0.76，MC@20=6.17，全面领先。
  - **HateXplain & FakeEdit**：标准 DMD + 初始振幅排序最优，FakeEdit 上 Llama-3.2-3B-inst Recall@10 达 0.76，MC@10=5.46。
  - **忠实度**：DMDINTEL 在掩码实验中准确率与置信度下降幅度稳居第一或第二，证明其定位 token 与模型决策高度因果相关。
  - **效率**：IG/SHAP 需多次前向/反向传播，DMDINTEL 仅需单次前向；Llama-3.2-3B-inst 在 Sentiment 上耗时约 1.03s，显存 13.01 GB，远低于 IG（~9.07s/15.23 GB）与 SHAP（~2.62s/38.80 GB）。
- **最强结果**：Qwen3-4B-inst + Sentiment 正样本（HODMD-avgamp）：MC@20=6.17，RBO@20=0.29，Recall@20=0.76。

## 相关工作脉络
- **IG / SHAP**：梯度或 Shapley 值归因基线，将 token 视为独立离散单元；本文以动力系统视角替代静态扰动，捕捉序列演化结构。
- **PCA 归因**：直接对隐藏状态矩阵转置做方差最大化降维，忽略时序先后；本文对比显示 DMD 因建模状态转移而更贴合 LLM 推理轨迹。
- **Mechanistic Interpretability（如 Geva et al., 2021 MLP 记忆库假说）**：侧重单层权重级电路解析；本文聚焦跨时间步的隐状态流形动态，无需拆解具体参数。
- **DMD 早期 NLP/语音应用**：仅用于情感轨迹特征提取或音频谱表示；本文为首次将 DMD 引入 LLM 隐状态动力系统建模与输入归因。
- **LogitLens / TunedLens**：在层间插入线性适配器映射预测轨迹；本文直接在 MLP down_proj 端分析动力学模态，无需额外微调适配器。

## 局限性与未来方向
- **任务范围受限**：仅验证于生成 1-2 个 token 的分类任务；在自回归长文本生成（如机器翻译）中，生成 token 会反哺后续输入，难以直接构建 DMD 快照矩阵。
- **模型覆盖不足**：仅测试监督指令微调模型，通用预训练 base model 及多轮对话场景尚未探索。
- **超参依赖经验**：阈值 $\tau$ 与模态数 $k$ 需人工设定；消融显示 $k>5$ 会引入噪声，$\tau=0.5$ 会削弱上下文信号，缺乏自适应优化机制。

## 研究启发与可借鉴点
- **动力系统视角迁移**：将 LLM/RNN 隐藏状态序列视为离散动力系统并应用 DMD/Koopman 理论，可作为序列表征分析的新范式，迁移至代码生成、多轮对话追踪等场景。
- **偏置扣除工程技巧**：空 prompt 提取并相减的

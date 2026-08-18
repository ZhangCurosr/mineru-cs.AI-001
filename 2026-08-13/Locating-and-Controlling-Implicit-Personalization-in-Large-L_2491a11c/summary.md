---
title: "Locating-and-Controlling-Implicit-Personalization-in-Large-L"
source: https://arxiv.org/pdf/2608.11735v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:46:11"
---

# 论文速读：Locating-and-Controlling-Implicit-Personalization-in-Large-L

## 一句话总结
本文通过匹配提示与中性对话的残差流激活差异，定位并量化了 LLM 中隐式人口统计线索触发的个性化偏移内部信号；发现该信号的归一化幅度与行为偏移强相关（最高 r=0.87），在多线索共现时内部表征呈线性可分解但外部行为严格次可加，并可通过推理时方向投影剥离目标维度激活，从而有效抑制刻板推荐且优于显式提示。

## 研究问题与动机
- **现象普遍但黑盒化**：LLM 会在用户未明示身份的情况下，仅凭对话中的文化引用、俚语或方言等隐式线索自动调整推荐内容（implicit personalization），产生对齐刻板印象的偏移；用户无法察觉、质疑或拒绝该隐性 conditioning。
- **内部表征机制缺失**：现有研究多停留于行为审计（如 StereoSet、BOLD、CEB 等基准或现实任务偏差检测），虽有探针表明属性信息可解码，但缺乏对“激活对比—行为偏移”因果链的量化验证，无法区分相关性是否具因果效力。
- **多线索叠加的现实复杂性**：真实对话往往同时携带种族、性别、年龄等多个线索，单轴线审计会遗漏交互效应；亟需回答混合线索的内部表示是否可分解为单线索分量，以及行为输出是否遵循相同的线性叠加规律。
- **干预手段的局限性**：现有的去偏多依赖显式 prompting（如“请忽略人口统计信息”），但提示可能失效甚至反向加剧刻板印象，且缺乏在保持通用能力的前提下进行维度特异性抑制的可靠机制。

## 核心贡献（创新点）
1. **内部信号的定位与量化追踪**：提出基于匹配对照的残差流激活对比 $\Delta A$ 及其归一化幅度 $s_{\Delta A}$，证明其在样本层面与行为偏移强相关（最高 $r=0.87$），且在无偏移条件下自然衰减至零，填补了隐式个性化“行为可观测、内部不可见”的空白。
2. **内部线性几何与外部次可加行为的解耦发现**：揭示多线索共存时，混合对比向量可由样本自身的单线索方向线性重构（拟合度 $\bar{\rho}=0.576\sim0.704$），但实际行为偏移比单线索之和低 28%–38%，明确区分了表征空间的可加性与输出空间的非线性压缩。
3. **推理时方向投影消融干预**：在中介层集 $L^\star$ 上对用户平均对比方向进行正交化与单位化，通过 forward pre-hook 执行 $h'=(I-\alpha \hat{v}\hat{v}^\top)h$ 投影剥离；该方法在 SED-desc 上最大超越“ignore demographics”提示 9.2×，且 MMLU 精度最多仅下降 2.8 个百分点，实现了因果可控的去偏。
4. **模型/属性特异性的实证刻画**：系统性验证了选择性抑制（保留共存维度）仅在部分模型（Mistral、Llama）和维度（race/gender）成立，Qwen3-8B 等模型出现双维同步下降，为后续更精细的擦除算法（如 LEACE）提供了明确的基线对照。

## 方法详解
- **配对对话构建**：以 GPT-4o 生成 $n=50$ 个场景，每个场景构建 $K=5$ 轮多轮对话；仅改变是否包含隐式人口统计线索（种族：Black/Asian/White；性别：male/female；年龄：child/adolescent/adult/older adult），中性版本使用身份中立填充词，话题骨架完全一致。多线索实验采用剂量匹配因子设计（$K=4$，两维度各两个线索）。最终查询统一要求推荐 5 部影片并仅输出标题与年份，greedy decoding（$T=0$）。
- **行为评估双指标**：
  - **SED-desc**：将推荐影片的 TMDB 剧情描述拼接后输入 all-mpnet-base-v2 编码器，以 $1-\text{cosine}$ 计算提示与中性推荐集的语义距离，度量偏移幅度。
  - **CAR（Content Alignment Ratio）**：使用 GPT-4o 按固定分类法对每部影片标注身份关联标签，计算 $\mathrm{CAR}(C_{\mathrm{cue}}; t) - \mathrm{CAR}(C_{\mathrm{neutral}}; t)$，度量偏移方向；人工重标验证 GPT-4o 与人类整体一致性达 0.8。
- **内部表征度量**：定义第 $L$ 层最后一 query token 的残差流激活对比 $\Delta \mathbf{A}_L^{(i)} = \mathbf{A}_L(C_{\mathrm{cue}}^{(i)}, x) - \mathbf{A}_L(C_{\mathrm{neutral}}^{(i)}, x)$，归一化幅度 $s_{\Delta A}^{(i,L)} = \|\Delta \mathbf{A}_L^{(i)}\|_2 / \|\mathbf{A}_L(x)\|_2$，在中介层集 $L^\star$ 上平均得到单样本信号。
- **中介层定位（Activation Patching）**：对每层执行三步前向（ID 路径、NEUTRAL 路径、将 ID 层激活 patch 至 NEUTRAL 路径），计算下一 token 分布的 JSD 作为基线 $D_i$，归一化间接效应 $\mathrm{NIE}(i,\ell)=\mathrm{clip}(1-\mathrm{JSD}(p_{\mathrm{patched}}^{(\ell)}, p_{\mathrm{id}})/(D_i+\epsilon), 0, 1)$；排除基线 JSD 第 40 百分位以下的样本后，按平均 NIE 降序选取 top-10 层作为 $L^\star$。
- **多线索分解**：对混合样本拟合 $\mathbf{v}_{AB} \approx \alpha \mathbf{v}_A + \beta \mathbf{v}_B$，以残差拟合度 $\rho=1-\|\mathbf{v}_{AB}-\hat{\mathbf{v}}_{AB}\|_2/\|\mathbf{v}_{AB}\|_2$ 评估线性组合解释力；同时构造跨样本基底替换与单方向拟合对照。
- **方向消融干预**：对目标维度取跨样本平均单线索对比方向，Gram-Schmidt 正交化去除伴侣维度分量后单位化得 $\hat{v}_{\mathrm{dim}}$；在 $L^\star$ 的 union top-2 层安装 pre-hook，执行 $h'=(I-\alpha \hat{v}_{\mathrm{dim}}\hat{v}_{\mathrm{dim}}^\top)h$，扫描 $\alpha\in[0.5, 5]$，并与随机方向 sham 及显式“ignore demographics”提示对比。

## 实验与结果
- **评测模型**：Llama-3-8B、Mistral-7B-v0.3、Qwen3-8B、Qwen3-14B、Phi-4（7B–14B 指令微调）。
- **单线索追踪结果**：Black-cue 与 child-cue 为跨模型通用正锚点，所有模型均出现显著偏移；$s_{\Delta A}$ 与 SED/CAR 的 Pearson $r$ 最高达 **0.87**（Llama-3-8B Asian-cue SED）。行为无偏移条件（如 White-cue）的内部相关性峰值仅 0.34，且 Spearman $\rho=0.54\sim0.57$（$p<0.001$）证实信号强度随行为偏移量单调上升。
- **多线索内部几何**：两维联合拟合平均 $\bar{\rho}=0.646$，显著优于单维最佳拟合 0.443；跨样本基底替换使拟合骤降至 0.095，证明分解具有强样本特异性。
- **多线索行为压缩**：观测混合 SED 偏移比加性预测低 **28%–38%**（$p<10^{-37}$），且 co-tag 率普遍低于独立性乘积，无 super-additive 放大。
- **因果控制效果**：在 SED-desc 上，方向消融最大超越提示基线 **9.2×**（Qwen3-8B race×age）；提示在 Mistral 上甚至向刻板方向移动（backfire）。MMLU 宏观准确率在 $\alpha=5$ 时最多下降 **2.8 个百分点**（Qwen3-8B）。选择性方面，Mistral race×gender 剥离 gender 方向使 gender CAR 降 0.169 而 race CAR 仅 +0.004；Qwen3-8B race×age 则两维同步下降（−0.130 / −0.104）。
- **跨领域泛化**

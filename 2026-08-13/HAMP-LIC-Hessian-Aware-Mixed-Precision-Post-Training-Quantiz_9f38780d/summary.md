---
title: "HAMP-LIC-Hessian-Aware-Mixed-Precision-Post-Training-Quantiz"
source: https://arxiv.org/pdf/2608.12239v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 10:48:26"
field: "学习图像压缩量化"
keywords: ["学习图像压缩", "后训练量化", "混合精度量化", "Hessian敏感度", "率失真优化", "模型压缩"]
innovations: ["Hessian迹与率失真任务损失的复合敏感度度量", "帕累托前沿搜索将指数比特分配降至多项式复杂度", "块级缩放与圆整联合任务约束重建优化"]
benchmarks: ["Kodak", "Tecnick", "CLIC 2020 Validation"]
---

# 论文速读：HAMP-LIC: Hessian-Aware Mixed-Precision Post-Training Quantization for Learned Image Compression

## 一句话总结
本文提出 HAMP-LIC，一种面向学习图像压缩（LIC）的 Hessian 感知混合精度后训练量化（PTQ）框架，通过二阶敏感度估计、任务感知优化与帕累托前沿搜索实现非均匀比特分配，在仅 0.59% BD-rate 损失下实现 4.85× 模型压缩，并彻底消除跨平台编码/解码误差。

## 研究问题与动机
- **LIC 模型部署成本高**：现代 LIC 架构推理计算与内存开销大，难以在资源受限设备上部署；统一固定精度量化在低比特下会造成严重质量下降，且忽视了不同层对量化的敏感度差异。
- **跨平台浮点不一致性**：不同硬件（CPU/GPU）在熵建模和概率估计上的浮点行为不一致，导致编码/解码不匹配——全文精度 Cheng2020 在 CPU 编码/GPU 解码场景下 24 张 Kodak 图像全部解码失败，98/100 张 Tecnick 图像解码失败。
- **现有 PTQ 方法不足**：现有 LIC PTQ 方法（如 RDO-PTQ、RAQ）普遍采用统一比特分配，忽视网络组件的非均匀冗余；通用混合精度方法针对分类网络设计，难以直接迁移至 LIC 的率失真优化与熵约束潜表示场景。
- **Hessian 二阶信息的引入动机**：二阶信息（Hessian 矩阵）可为层粒度敏感度提供 principled 度量，从而支持更科学的精度分配，但其在 LIC 量化中的系统应用尚未被充分探索。

## 核心贡献（创新点）
1. **提出 HAMP-LIC 四阶段 PTQ 框架**：将 Hessian 二阶敏感度与率失真感知优化结合，为 LIC 模型设计专用混合精度量化流程。与 MPP-LIC（作者在 DCC 2026 的先前工作）相比，本文新增任务感知灵敏度细化与全局尺寸约束优化。
2. **任务感知灵敏度细化机制**：通过 Hessian trace 与率失真损失退化比的乘积定义敏感度度量，使比特分配决策直接反映压缩目标而非通用重建误差。与 HAWQ 等纯几何敏感度方法本质不同，HAMP-LIC 将敏感度与 LIC 特有的 R-D 损失耦合。
3. **帕累托前沿搜索实现高效比特分配**：将比特分配建模为带全局模型尺寸约束的整数优化问题，通过 Pareto-frontier 搜索将搜索空间从指数级（3^50 ≈ 7.12×10^23）降至多项式级（1326 种组合），计算可行性提升约 21 个数量级。
4. **块级重建优化（Scaling + Rounding）**：在比特分配完成后，采用顺序缩放优化与自适应圆整优化两步细化，显著抑制量化误差；与 RDO-PTQ 的 BRECQ 式重建相比，本文额外引入任务约束联合优化目标。

## 方法详解
**Step 1 — Hessian Trace 敏感度估计**：对每个量化块 $B_l$，基于校准集用 Hutchinson 方法（5 个 Rademacher 随机向量/块，共 12 个样本）高效估计 Hessian 迹 $\mathrm{Tr}(H_l) = \mathrm{Tr}\left(\frac{\partial^2 \mathcal{L}}{\partial W_l^2}\right)$，作为块级敏感度初始指标；值越大表示该块对量化扰动越敏感。

**Step 2 — 任务感知灵敏度构建**：以 8-bit 量化模型的任务损失 $\mathcal{L}_{\mathrm{int8}}$ 为参考，对每个块 $B_l$ 在不同候选比特 $b_l$ 下量化后评估任务损失 $\mathcal{L}_{\mathrm{qp},l}(b_l)$，定义任务感知敏感度：
$$\Omega_l(b_l) = \mathrm{Tr}(H_l) \cdot \frac{|\mathcal{L}_{\mathrm{qp},l}(b_l) - \mathcal{L}_{\mathrm{int8}}|}{|\mathcal{L}_{\mathrm{int8}}|}$$
该指标同时捕捉块的几何敏感度与对率失真目标的实际影响。LIC 的率失真损失为 $J_i = \lambda \cdot D(x_i, \hat{x}_i) + R_{\hat{Y},i} + R_{\hat{Z},i}$。

**Step 3 — 帕累托前沿比特分配**：在约束 $\frac{1}{8L}\sum_{l=1}^{L} b_l \leq \epsilon$（实验中 $\epsilon = 0.75$，允许 10% 容差）下求解 $\arg\min\sum_{l=1}^{L} \Omega_l(b_l)$。按敏感度降序排列块后，搜索空间被限定为单调非增分区的连续划分，候选组合数为 $|B| = \sum_{j=1}^{k}\binom{k}{j}\binom{L-1}{j-1}$（$L=50, k=3$ 时从 $3^{50}$ 降至 1326）。

**Step 4 — 块级重建优化**：
- **缩放优化**：顺序优化每个块的权重/偏置缩放因子 $(s_w^l, s_b^l)$，最小化量化后与全精度 R-D 损失之差 $(\widehat{J} - J_0)^2$，从前向后依次进行，已量化块参数固定。
- **圆整优化**：引入可学习变量 $V$ 参数化权重圆整（沿用 AdaRound/BRECQ 思路），最小化 $\|\Lambda(w^l x^l) - \Lambda(\hat{w}^l x^l)\|^2 + \beta f_{\mathrm{reg}}(V)$，并结合任务损失：$\min_{s,V} \lambda_t \mathcal{L}_{\mathrm{task}} + \mathcal{L}_{lq}$（$\lambda_t=1$）。

## 实验与结果
- **基线模型**：Minnen2018、Cheng2020（CompressAI 库 FP32 预训练，$\lambda \in \{0.0067, 0.013, 0.025, 0.0483\}$）。
- **对比方法**：RAQ、RDO-PTQ、FPQ、FMPQ（均为 LIC 专用 PTQ 方法）；传统基准：H.265/HEVC (BPG)、H.266/VVC (VTM 23.11)。
- **数据集**：Kodak（24 张，768×512）、Tecnick（100 张，1200×1200）、CLIC 验证集（41 张高质量图像）。
- **校准数据**：从 CLIC 随机采样 12 张，Adam 优化器，lr=0.001，batch_size=4，每块 20000 次迭代，输入裁剪至 256×256。
- **核心结果（Cheng2020, w=6.6, a=32）**：BD-rate 损失 Kodak 0.59%、Tecnick 1.79%，压缩比 4.85×，显著优于 FMPQ（0.89%/2.68%，压缩比 3.99×）。
- **跨平台鲁棒性**：FP32 Cheng2020 在 CPU/GPU 混用时 Kodak 解码错误率 24/24、Tecnick 98/100；HAMP-LIC 两个数据集、两种跨平台组合（CPU→GPU / GPU→CPU）错误率均为 **0**。
- **BOPS 节省**：相较 FP32 减少约 94%；相较 W8A8 均匀量化再减 5.3%~8.6%（Q3 最佳）。
- **消融发现**：Hessian 优于 Fisher 矩阵；帕累托优化优于阈值法（$\epsilon=0.75$ 下平均比特 5.719 vs 6，RD 性能相当）；主路径层（$g_a, g_s$）比特分配更高，超路径层（$h_a, h_s$）主要分配 4-bit。

## 相关工作脉络
- **HAWQ/HAWQ-V2**（Dong et al., 2019/2020）：开创 Hessian 迹感知的混合精度量化思路，但面向通用视觉分类任务；本文将其适配到 LIC 场景并耦合率失真损失，且引入帕累托搜索解决组合爆炸。
- **RAQ**（Hong et al., 2021）：仅对解码器做定点推理的范围自适应量化；本文扩展到全网络混合精度且支持更低比特。
- **BRECQ**（Li et al., 2021）：通用任务块级重建 PTQ；RDO-PTQ（Shi et al., 2024）将其引入 LIC；本文在块级重建基础上进一步引入任务约束联合优化（scaling + rounding + 任务损失）。
- **FMPQ**（Hossain et al., 2024）：针对 LIC 的灵活混合精度 PTQ，但平均比特目标限制在 8-bit 附近；本文突破至超低比特（平均 ~5.7-bit）同时保持极低 BD-rate 损失。
- **Q-LIC**（Sun et al., 2025）：通过通道分裂与剪枝减小激活动态范围以提升量化鲁棒性；本文从比特分配层面解决问题，两者思路互补。

## 局限性与未来方向
- 校准数据集极小（仅 12 张），可能影响敏感度估计的泛化性。
- 压缩比超参数 $\epsilon$ 仍需人工设定，自动化选择未实现。
- 仅验证了 VAE-based LIC 架构（Minnen2018、Cheng2020），未覆盖新兴的 Transformer-based LIC 模型。
- 每块 20000 次迭代重建优化计算开销较大，实际部署前的量化耗时需进一步优化。
- 作者自述未来方向：扩展至 Transformer-based LIC、自动化 $\epsilon$ 选择、降低校准开销。

## 研究启发与可借鉴点
- **Hessian trace + 任务损失的复合敏感度度量**：将几何敏感度（二阶曲率）与下游任务损失退化相乘的思路，可迁移至视频压缩、点云压缩等同样存在率失真优化的领域。
- **帕累托前沿搜索将指数搜索降至多项式**：按敏感度排序后限定为连续分区划分的搜索策略，是解决混合精度比特分配组合爆炸的有效范式，适用于其他需要逐层/逐模块分配资源比特的场景。
- **跨平台鲁棒性的量化价值**：文章揭示 FP32 LIC 模型在 CPU/GPU 混用时解码错误率极高，量化后可彻底消除——这一发现对边缘部署的鲁棒性验证具有方法论启示，可作为后续工作的安全基线评估指标。
- **主路径 vs 超路径敏感度差异**：HAMP-LIC 发现主路径层（$g_a, g_s$）比特分配显著高于超路径层（$h_a, h_s$），这一结构性洞察可用于指导其他 LIC 架构（如 HiFICo、VART）的量化策略设计。

## 关键术语表
- **Learned Image Compression (LIC)**：利用端到端神经网络联合优化非线性变换与熵模型，实现超越传统编解码器（如 H.265/VVC）的率失真性能。
- **Post-Training Quantization (PTQ)**：在不重新训练的情况下，仅用少量校准数据将预训练全精度模型转换为低精度表示的方法。
- **Hessian Trace**：Hessian 矩阵的迹，反映损失函数在参数空间的局部曲率，曲率越大表示该参数对扰动越敏感。
- **BD-rate**：Bjøntegaard 变化率，用于量化两条率失真曲线之间平均压缩效率差异的指标，负值表示性能增益。
- **Rate-Distortion (R-D) Loss**：率失真损失，由失真项（如 MSE/MS-SSIM）与码率项加权和构成，控制压缩质量与码率的权衡。
- **Pareto Frontier Search**：帕累托前沿搜索，在多维优化中寻找非支配解集合的方法，本文用于在压缩预算约束下高效搜索最优比特分配。
- **Hutchinson's Method**：基于随机 Hessian-vector product 估计的 Hessian 迹近似方法，用少量随机向量即可无偏估计矩阵迹。
- **Hyperprior**：LIC 中的辅助潜变量结构，用于建模主潜变量 $\hat{Y}$ 的条件概率分布，增强熵模型的表达能力。

## 可复现要素
- **数据集**：Kodak（公开）、Tecnick（公开）、CLIC 2020 验证集（公开）；校准集为 CLIC 中随机 12 张。
- **预训练模型**：CompressAI 库 [26] 提供的 Minnen2018 与 Cheng2020 FP32 权重（公开）。
- **代码**：论文未声明开源。
- **关键超参**：$\epsilon=0.75$（允许 10% 容差）、学习率 0.001、batch_size=4、每块迭代 20000 次、Hutchinson 随机向量数 5/块、$\lambda_t=1$、输入裁剪 256×256、$\lambda \in \{0.0067, 0.013, 0.025, 0.0483\}$、候选比特集 $B=\{2,4,8\}$。

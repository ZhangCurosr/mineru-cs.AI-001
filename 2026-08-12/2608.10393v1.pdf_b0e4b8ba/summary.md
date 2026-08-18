---
title: "Hidden in Plain Sight: Difusion-Based Unrestricted Robotic Attacks on Vision-Language-Action Models"
source: https://arxiv.org/pdf/2608.10393v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:30:00"
field: "机器人对抗机器学习"
keywords: ["VLA模型", "对抗攻击", "扩散模型", "黑盒攻击", "对抗补丁", "机器人安全"]
innovations: ["提出DURA，首次将扩散潜在轨迹用于VLA自然对抗patch生成，统一支持白盒与黑盒攻击", "设计score-function估计器在扩散潜在空间估计黑盒攻击方向，提升查询效率", "通过clean anchor软约束联合跨frame优化单个patch，实现高迁移性与视觉自然性"]
benchmarks: ["LIBERO", "BridgeData V2"]
---

# 论文速读：Hidden in Plain Sight: Difusion-Based Unrestricted Robotic Attacks on Vision-Language-Action Models

## 一句话总结
本文提出了 **DURA**，一种基于扩散模型的无约束机器人对抗攻击方法，通过在预训练扩散模型的潜在轨迹上优化，生成视觉上自然的对抗性 patch，在黑盒与白盒设置下均能高效地将 VLA 模型的机器人行为引导至攻击者指定的目标动作。

## 研究问题与动机
- **视觉自然性不足**：现有 VLA 攻击生成的 patch 多表现为无意义的高频噪声或明显伪影，在实际工作空间中容易被识别和规避。
- **黑盒部署性受限**：现有方法普遍依赖模型参数、梯度或 logits 等白盒信息，而闭源机器人系统通常不提供此类访问权限。
- **时间效率低下**：现有方法需要针对每个实例进行冗长的逐 patch 优化，查询成本和计算开销较高，限制了实际部署可行性。

## 核心贡献（创新点）
- **扩散引导的无约束 patch 生成框架**：将 patch 内容视为无约束变量，在冻结的预训练扩散模型的潜在轨迹上进行搜索，既保留视觉自然性，又通过 VLA 攻击目标引导机器人行为。
- **统一支持白盒与黑盒两种威胁模型**：白盒模式下通过 VLA 策略和 VAE 解码器的反向传播计算潜在更新；黑盒模式下仅需 action-output 查询，无需访问模型参数或梯度。
- **跨模型与跨场景的高迁移性**：在 OpenVLA 和 π₀-FAST 两个主流 VLA 模型、LIBERO 仿真基准以及真实 Franka 机械臂上均验证了攻击有效性，ASR 达到 79.3%–100%。
- **视觉自然性与攻击效果的联合优化**：通过 clean anchor 轨迹的软约束（α_w 控制锚点强度），防止攻击轨迹偏离原始去噪路径过远，在 6 项视觉质量指标上显著优于像素空间基线方法。

## 方法详解
**威胁模型**：攻击者可在机器人视野中放置单个局部 patch，但无法修改模型参数、训练数据、机器人状态或语言指令；目标是使受害策略在 patch 可见时持续执行攻击者指定的目标动作。

**核心优化目标**：
$$
\mathcal{L}_{\mathrm{attack}}(p) = \frac{1}{B} \sum_{b=1}^{B} \sum_{i \in \mathbb{Z}} w_i \mathcal{L}_i(y_{b,i}(p), a_i^\star)
$$
其中 $y_{b,i}(p)$ 为带 patch 输入下的攻击可见输出，白盒使用目标交叉熵损失（需 action-token 真值），黑盒使用预测动作与目标动作的 MSE 损失。

**扩散引导的 Patch 优化流程**：
1. **初始化**：将干净种子 patch $p_{\mathrm{clean}}$ 编码为潜在空间 $z_0^{\mathrm{clean}} = \mathcal{E}(p_{\mathrm{clean}})$，部分加噪至中间时刻 $t_{\mathrm{start}}$ 得到 $z_{t_{\mathrm{start}}}^{\mathrm{adv}}$。
2. **Clean Anchor 轨迹预计算**：从同一种子 patch 运行 unperturbed DDIM 得到 $\{z_t^{\mathrm{anc}}\}$。
3. **交替去噪与对抗引导**（对 $t = t_{\mathrm{start}}, \ldots, 1$）：
   - 软锚定：$z_t^{\mathrm{mix}} = (1 - \alpha_w) z_t^{\mathrm{adv}} + \alpha_w z_t^{\mathrm{anc}}$
   - DDIM 去噪一步：$u_{t-1} = \mathrm{DDIM}(z_t^{\mathrm{mix}}, t; \hat{\epsilon}_\phi)$
   - 注入攻击更新：$z_{t-1}^{\mathrm{adv}} = u_{t-1} - s \cdot g_t$
4. **最终解码**：$p_{\mathrm{adv}} = \mathcal{D}(z_0^{\mathrm{adv}})$

**攻击方向估计**：
- **白盒**：$g_t^{\mathrm{WB}} = \nabla_{u_{t-1}} \mathcal{L}_{\mathrm{attack}}(\mathcal{D}(u_{t-1}))$，通过 VLA 策略和 VAE 解码器反向传播。
- **黑盒**：基于 score-function 估计器，在当前去噪潜在 $u$ 附近采样 $K$ 个候选潜在 $z_{t,k} = \sqrt{\alpha_t} u + \sqrt{1-\alpha_t} \epsilon_k$，解码后渲染 patch 并查询受害者策略获取 loss，估计方向为：
$$
g_t^{\mathrm{BB}} = \frac{1}{K} \sum_{k=1}^{K} (\mathcal{L}_{\mathrm{attack}}^{(k)} - b) \frac{\sqrt{\alpha_t}}{\sqrt{1-\alpha_t}} \epsilon_k
$$
其中 $b$ 为方差约简基线。

## 实验与结果
- **数据集与模型**：LIBERO 仿真基准（4 个任务套件：Spatial、Object、Goal、Long）、BridgeData V2；OpenVLA-7B 和 π₀-FAST 两个 SOTA VLA 模型；真实 Franka 7-DoF 机械臂。
- **评估指标**：ASR（Attack Success Rate，rollout 级别任务失败率）、AP（Attack Precision，action-step 级别目标行为一致性）。
- **OpenVLA 白盒结果（Table 1）**：DURA 在全部 4 个任务套件达到 100% ASR，超越 target-only 基线 TMA（平均 99.8%）和 FreezeVLA（95.4%），匹敌 label-supervised 最强基线 UADA（100%）。
- **OpenVLA 黑盒结果（Table 1）**：仿真 patch 86.0% ASR，物理 patch 79.3% ASR，分别较 TMA-NES 提升 43.7 和 40.0 个百分点。
- **π₀-FAST 结果（Table 2）**：白盒与黑盒均在全部 4 个任务套件达到 100% ASR，远超最强基线（白盒 58.0%，黑盒 31.3%）。
- **物理实验（Figure 3）**：打印的对抗 patch 在真实 Franka 机械臂上可控制性地触发冻结（freeze）和错误前移等行为，移除 patch 后行为恢复正常，验证了 attack 的可控性与可重复性。
- **视觉自然性（Figure 5）**：DURA 在 NIQE、CLIP-Natural、DISTS、SSIM、BSE、Patch TV 六项指标上最接近无 patch 场景基线。
- **查询预算（Figure 6）**：黑盒模式下 K=512 即达到较高 ASR，K=2048 时达 100%；TMA-NES 在同预算下仅维持 40%–48%。
- **鲁棒性（Appendix E）**：面对 JPEG 压缩（Q=10）、位深削减、高斯噪声等常见输入变换防御，ASR 仍保持在 90%–100%。

## 相关工作脉络
- **Wang et al. 2025a（VLA patch attacks）**：研究白盒 patch 攻击以干扰或引导机器人动作，但需 victim model 梯度访问，且 patch 视觉上不自然。
- **Wang et al. 2025c（FreezeVLA）**：使用双层优化诱导 action freezing，同样依赖白盒访问，且未关注 patch 视觉自然性。
- **Lu et al. 2026（UPA-RFAS）、Xu et al. 2025（EDPA）、Fu et al. 2026（VLA-Hijack）、Huang et al. 2026（TRAP）**：可迁移攻击通过替代模型或共享表征优化，但效果依赖 surrogate-to-victim 迁移，跨架构/动作表示易退化。
- **Chen et al. 2023a (AdvDiff)、Guo et al. 2024 (AdvDifVLM)**：将对抗引导嵌入扩散反向过程以生成自然对抗样本，但攻击目标为图像类别或文本概念，而 VLA 的目标是动作序列，二者无直接视觉对应关系。
- **DURA 的定位差异**：首次将扩散 prior 用于 VLA 动作导向的自然 patch 生成，统一支持白盒/黑盒，且通过跨 frame 联合优化单个 patch 提升迁移性。

## 局限性与未来方向
- **攻击成功依赖 patch 可见性**：当前威胁模型假设 patch 始终在相机视野内，未考虑遮挡、视角变化或远距离探测等现实复杂条件。
- **黑盒查询成本仍较高**：虽然 DURA 在 K=512 时已表现良好，但大规模部署场景下每 patch 仍需 100×K 次查询，可能影响实时性。
- **仅限视觉通道攻击**：论文聚焦视觉 patch 攻击，未探索语言指令通道的联合攻击或 cross-modal 协同攻击。
- **防御机制研究不足**：虽验证了对常见输入变换防御的鲁棒性，但缺乏对专门面向自然对抗 patch 的新型防御方法的评估。

## 研究启发与可借鉴点
- **扩散潜在空间作为对抗优化先验**：将对抗样本生成约束在预训练扩散模型的潜在轨迹上，可有效平衡攻击效果与视觉自然性，该方法可迁移至其他视觉主导的决策模型（如 VLM、VLA 变体）的对抗攻击/防御研究。
- **Score-function 估计器适配扩散轨迹**：黑盒场景下利用 score-function 在扩散潜在空间估计攻击方向，避免了直接对 action-output 查询做有限差分，提高了查询效率，可为其他黑盒对抗攻击提供技术参考。
- **Clean anchor 软约束机制**：通过 $(1-\alpha_w)z^{\mathrm{adv}} + \alpha_w z^{\mathrm{anc}}$ 混合策略保持攻击轨迹不偏离自然分布，该正则化思想可推广至其他生成式对抗样本生成任务。
- **跨 frame 单 patch 联合优化提升迁移性**：在同一 patch 上联合优化多个不同观察帧和指令，使攻击泛化于场景、状态和指令变化，该设计值得在跨域对抗攻击中借鉴。

## 关键术语表
- **Vision-Language-Action (VLA) 模型**：统一视觉感知、语言理解和动作预测的机器人通用策略模型，如 OpenVLA、π₀。
- **对抗性 Patch（Adversarial Patch）**：放置在场景中的局部视觉扰动，用于误导模型决策，本文要求其视觉上自然且语义合理。
- **白盒攻击（White-box Attack）**：攻击者可以访问受害者模型的内部信息（如梯度、logits）的攻击设置。
- **黑盒攻击（Black-box Attack）**：攻击者只能提交观测输入并观察动作输出的攻击设置，无需模型内部信息。
- **DDIM（Denoising Diffusion Implicit Models）**：一种确定性扩散模型采样器，可在较少步数内高质量生成图像。
- **Attack Success Rate (ASR)**：rollout 级别指标，衡量攻击导致任务失败的比例。
- **Attack Precision (AP)**：action-step 级别指标，衡量执行行为与目标动作的一致性程度。
- **Score-Function Estimator**：基于 Williams (1992) 的梯度估计方法，通过采样和 loss 加权估计目标函数的梯度。

## 可复现要素
- **数据集**：LIBERO（仿真，论文使用 fine-tuned checkpoints）、BridgeData V2（真实数据）；数据集公开可用。
- **代码/权重**：论文未明确声明代码开源状态，但使用的 OpenVLA 和 π₀-FAST 模型均为开源模型。
- **关键超参**：$t_{\mathrm{start}} = 0.5$（部分加噪起始步），DDIM 优化步数 200，anchor 权重 $\alpha_w = 0.2$，目标动作 loss 权重（translation: 1.0, rotation: 0.5, gripper: 0.2），黑盒查询数 $K = 2048$，优化迭代次数 100（TMA-NES 与 DURA 公平比较时统一）。
- **实验平台**：单卡 H200 GPU，batch size 16。

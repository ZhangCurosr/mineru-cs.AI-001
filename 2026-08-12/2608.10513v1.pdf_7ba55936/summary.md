---
title: "SafeCap: Improving LVLM Safety with Image Captioning Reinforcement Learning"
source: https://arxiv.org/pdf/2608.10513v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:45:05"
field: "多模态大模型安全对齐"
keywords: ["LVLM安全", "强化学习对齐", "图像描述", "多模态安全", "GRPO", "安全奖励设计"]
innovations: ["将LVLM安全对齐形式化为自描述任务，通过caption-mediated奖励实现安全感知推理", "提出指数风险折扣函数S=u·γ^h替代线性奖励，有效平衡安全性与帮助性", "引入组件级组归一化防止多方奖励尺度耦合，结合多协议诊断评估框架"]
benchmarks: ["MM-SafetyBench", "MSSBench", "VLSBench", "FigStep", "MIS-Test", "MM-Vet", "BLINK", "MMVP", "ERQA", "VPCT", "MMStar"]
---

# 论文速读：SafeCap: Improving LVLM Safety with Image Captioning Reinforcement Learning

## 一句话总结
SafeCap 提出一种基于强化学习的框架，将 LVLM 安全对齐形式化为**自描述（self-captioning）任务**：策略模型先生成安全相关的图像 caption，再据此给出最终回答；caption 的质量通过其与冻结纯文本 LLM 回答的**安全一致性**来评价。实验显示，该方法在五个多模态安全基准和六个视觉效用基准上，显著提升了安全性（最高 +19.0 分）同时几乎不损失视觉能力。

## 研究问题与动机
1. **视觉输入可绕过文本侧安全对齐**：有害指令可嵌入排版图像、对象暗示或通过视觉推理隐藏，使仅依赖语言 backbone 安全策略的 LVLM 产生失效。
2. **推理时 caption 中介防御的信息损失**：ECSO 等方法将图像转为文本描述后交由文本模型响应，但图像→文本转换可能丢失细粒度视觉线索（可见文字、对象、上下文），导致下游安全判断不可靠。
3. **训练时对齐方法难以兼顾安全性与视觉效用**：直接优化安全偏好数据易导致过度拒绝或损害视觉感知能力；缺乏对"安全证据如何在推理前显式暴露"的机制设计。
4. **奖励塑形难题**：纯安全目标可能偏向拒绝，纯 caption 目标无法保证最终回答安全；需要结合两种信号的复合奖励设计。

## 核心贡献（创新点）
1. **安全感知自描述形式化**：将 LVLM 安全对齐重新表述为先生成 tagged caption 再回答的结构化生成任务，使中间表征成为安全推理的证据通道而非事后解释。
2. **Caption-mediated 双信号奖励机制**：引入结合直接答案评估（$R_{\mathrm{ans}}$）与冻结 LLM 一致性评估（$R_{\mathrm{cap}}$）的复合奖励，并采用**指数风险折扣** $S(u,h)=u\gamma^h$ 替代线性奖励，避免"什么都不做"的退化行为。
3. **组件级组归一化（component-wise group normalization）**：借鉴 GDPO 的解耦归一化思想，在 GRPO 训练路径中独立归一化各奖励分量，防止不同尺度分量在信用分配前耦合。
4. **多协议评估与消融体系**：设计 Direct、DirectCap、Prism 三种推理协议分别诊断模型原生响应、自描述路径和 caption 可迁移性，并提供系统的奖励组件与超参消融。

## 方法详解
**问题形式化**：对每个图像-问题对 $(x,q)$，策略模型 $\pi_\theta$ 生成结构化响应：
$$y = \langle \mathtt{caption} \rangle c \langle / \mathtt{caption} \rangle a$$
其中 $c$ 为图像描述，$a$ 为最终答案。训练目标为使 $c$ 编码对安全对齐推理有用的视觉证据。

**训练流程**：使用 GRPO 在 SPA-VL 数据集上训练，每个 prompt 采样 $K=8$ 个 rollout，优势函数为：
$$A_i = w_{\mathrm{tmp}}\tilde{r}_{i,\mathrm{tmp}} + w_{\mathrm{cap}}\tilde{r}_{i,\mathrm{cap}} + w_{\mathrm{ans}}\tilde{r}_{i,\mathrm{ans}}$$
其中 $\tilde{r}$ 为组内归一化后的奖励分量。

**模板奖励 $R_{\mathrm{tmp}}$**：基于规则的格式门控——响应必须包含恰好一个完整非空 caption 块且生成在长度限制内终止，否则跳过下游奖励。

**答案奖励 $R_{\mathrm{ans}}$**：使用 judge 模型给出保留分 $u(a)\in[0,5]$ 和风险分 $h(a)\in[0,5]$，采用指数风险折扣：
$$R_{\mathrm{ans}} = S(u,h) = u \cdot \gamma^h, \quad \gamma \in (0,1)$$
$\gamma=0.35$ 为默认值；每增加 1 单位风险，效用贡献乘以 $\gamma$，使高风险回答快速失去奖励。

**Caption 奖励 $R_{\mathrm{cap}}$**：
1. 将 caption $c$ 传给冻结纯文本 LLM（Qwen3-4B）与原始问题 $q$，生成 $a_f$；
2. Judge 评估 caption 描述覆盖率 $u(c)\in[0,5]$（关注主体、属性、动作、空间布局、环境、可见文字）；
3. 二值安全一致性判断 $g(q,a,a_f)\in\{0,1\}$：策略回答与冻结 LLM 回答在"是否识别风险"上一致则得 1；
$$R_{\mathrm{cap}} = g(q,a,a_f) \cdot u(c)$$

**组归一化**：
$$\tilde{r}_{i,k} = \frac{r_{i,k} - \mu_{G,k}}{\sigma_{G,k} + \epsilon}, \quad \epsilon=10^{-6}$$
对各分量独立归一化后再加权组合。

## 实验与结果
**数据集**：公开 SPA-VL 多模态安全对齐数据集（无私有数据补充）；与 SafeGRPO 对比时使用 SafeTag-VL-3K。

**评估基准**：
- 安全：MM-SafetyBench、MSSBench、VLSBench、FigStep、MIS-Test（报告 1−ASR，越高越好）
- 视觉效用：MM-Vet、BLINK、MMVP、ERQA、VPCT、MMStar

**主要结果（DirectCap 协议）**：
| 模型 | 安全提升（vs 零训练） | 视觉效用变化 |
|------|---------------------|-------------|
| Qwen3.5-2B | +5.29 | +0.10 |
| Qwen3.5-2B-Base | +5.57 | -0.01 |
| Qwen3.5-4B | +5.48 | -0.38 |
| Qwen3.5-4B-Base | **+19.0** | **+0.10** |

4B-Base 在 DirectCap 下安全平均从 40.43 提升至 **59.39**，视觉平均从 53.13 微降至 53.05，安全性提升最为显著。

**与 SFT/DPO/SafeGRPO 对比（Qwen3.5-4B-Base，同数据同步骤）**：
- DirectCap 下 SafeCap：S-Avg = **59.39**，V-Avg = 53.05
- SFT：S-Avg = 43.19，V-Avg = 46.87
- DPO：S-Avg = 41.36，V-Avg = 52.67
- SafeGRPO（SafeTag-VL-3K）：S-Avg = 41.27，V-Avg = 50.56

SafeCap 在 DirectCap 下显著优于所有基线，且 utility 保持接近零训练水平。

**消融结论**：
- 移除 $R_{\mathrm{ans}}$：DirectCap 安全下降 6.49 分（最大影响分量）
- 移除 $R_{\mathrm{cap}}$：DirectCap 安全下降 2.53 分
- 移除归一化：DirectCap 安全下降 2.02 分，但效用反而 +0.42 分
- $\gamma=0.20$（更强风险惩罚）比 $\gamma=0.35$ 在早期 checkpoint 收敛更快

**Prism 诊断**：caption 可迁移至未见过的 Qwen3-14B，安全平均 +4.72 分，说明 learned caption 不针对特定 frozen LLM。

## 相关工作脉络
1. **ECSO（Gou et al., 2024）**：推理时图像转文本防御，将不安全图像转为 query-aware 描述后由文本模型响应；SafeCap 与之本质不同——ECSO 是外挂式干预，SafeCap 训练模型自身生成安全 aware caption。
2. **SPA-VL（Zhang et al., 2024）**：多模态安全偏好数据集；本文在其上训练但引入 RL 而非 SFT/DPO，证明 caption-mediated 奖励比直接偏好优化更有效。
3. **CapRL（Xing et al., 2025）**：将 open-ended caption 质量转化为解耦的 VQA 信号以减少 reward hacking；SafeCap 在此基础上引入**二值安全一致性**判断而非仅感知质量。
4. **SafeGRPO（Rong et al., 2025）**：需要额外安全标注的 RL 方法；SafeCap 在相同公开数据上无需额外标注即超越 SafeGRPO。
5. **GDPO（Liu et al., 2026）**：提出组奖励解耦归一化；SafeCap 借鉴其 component-wise 归一化思想但在 GRPO 路径中实现，未引入 batch-level whitening。
6. **FigStep（Gong et al., 2023）/ HADES（Li et al., 2024）/ MIS（Ding et al., 2025）**：揭示视觉输入绕过安全对齐的攻击范式，构成本文动机来源。

## 局限性与未来方向
1. **Caption 事实错误无法被奖励信号捕捉**：冻结 LLM 不看到图像，只能判断安全一致性，无法检测 caption 中的幻觉或事实错误；作者通过人工监控 rollout 未发现系统性幻觉，但缺乏自动保障机制。
2. **安全奖励的可验证性有限**：judge 基于固定 rubric 评估风险与效用，属于弱监督信号；更强的可验证安全奖励仍是开放问题。
3. **模型规模受限**：实验仅在 Qwen3.5-2B/4B 上进行，未验证更大模型或其他模型族（如 LLaVA、Qwen2-VL 等）。
4. **缺乏 caption-mediated SFT 阶段**：作者指出先做 caption 中介的 SFT 再 RL 是 promising 方向，但当前缺少合适的监督数据。
5. **Prism 协议的下游 LLM 依赖**：Prism 性能强烈依赖 frozen LLM 能力（Qwen3-14B 比 Qwen3-4B 提升 +4.72 分），说明 caption 质量的上限受限于下游 reasoner。

## 研究启发与可借鉴点
1. **指数风险折扣函数** $S=u\gamma^h$：相比线性 $u-\beta h$，能更优雅地压制高风险回答同时保留安全有用回答的正奖励，对任何需平衡安全性与帮助性的 RLHF/RLAIF 任务均有借鉴价值。
2. **多协议诊断框架**（Direct / DirectCap / Prism）：分别检验原生响应、自描述路径和 caption 可迁移性，为安全对齐方法提供系统化评估范式，可推广至其他 multimodal alignment 工作。
3. **Caption 作为安全证据通道的思想**：将中间表征显式化为安全推理的 bridge，而非仅用于解释，可与 RAG、chain-of-thought 等结合探索更复杂的多步安全推理。
4. **冻结 LLM 替代 VLM judge**：用文本-only LLM 评估 caption 安全性可减少对多模态 judge 的依赖和视觉 bias，为低成本安全评估提供可行路径。
5. **组件级归一化在多方奖励 RL 中的适用性**：当任务存在尺度差异显著的奖励分量时，独立归一化可防止某一分量主导梯度，可迁移到多目标 RL 场景。

## 关键术语表
- **LVLM（Large Vision-Language Model）**：将大型语言模型的指令遵循能力扩展到视觉输入的模型架构。
- **DirectCap**：SafeCap 的目标推理协议，模型先生成 `<caption>` 标签包裹的图像描述，再给出最终回答。
- **Prism 协议**：将模型生成的 caption 提交给冻结的纯文本 LLM 进行回答，用于诊断 caption 的安全信息可迁移性。
- **Caption-mediated reward**：通过评估 caption 是否支持冻结 LLM 生成与策略回答安全一致的答案来计算的奖励信号。
- **Exponential risk-discount**：安全奖励函数 $S(u,h)=u\cdot\gamma^h$，以指数方式衰减风险惩罚，避免线性奖励的退化问题。
- **SPA-VL**：Multimodal Safety Preference alignment dataset，公开的安全偏好对齐数据集，本文主要训练数据源。
- **GRPO（Group Relative Policy Optimization）**：组相对策略优化算法，SafeCap 采用的强化学习优化器。
- **Component-wise group normalization**：对每个奖励分量在 rollout 组内独立做均值-方差归一化，防止不同尺度分量耦合。

## 可复现要素
- **数据集**：SPA-VL（公开）；SafeTag-VL-3K（用于与 SafeGRPO 对比，论文未明确公开状态）
- **代码**：https://github.com/Safe-VLM/SafeCap
- **项目页面**：https://safe-vlm.github.io/SafeCap
- **关键超参**：$K=8$ rollouts，batch size=64，minibatch size=64，学习率 $5\times10^{-7}$，weight decay=0.01，gradient clipping=1.0，entropy reg=0，KL loss=0，训练 200 steps，$\gamma=0.35$，权重 $(w_{\mathrm{tmp}}, w_{\mathrm{cap}}, w_{\mathrm{ans}})=(0.5, 0.5, 1.0)$，max tokens=4096
- **冻结 LLM**：Qwen3-4B
- **Judge 模型**：gpt-oss-20b（deterministic decoding）
- **硬件**：8× NVIDIA H200 GPU
- **模型 backbone**：Qwen3.5-2B / Qwen3.5-2B-Base / Qwen3.5-4B / Qwen3.5-4B-Base

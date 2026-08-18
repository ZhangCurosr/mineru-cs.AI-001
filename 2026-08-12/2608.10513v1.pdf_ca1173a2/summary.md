---
title: "SafeCap: Improving LVLM Safety with Image Captioning Reinforcement Learning"
source: https://arxiv.org/pdf/2608.10513v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:44:38"
field: "多模态大模型安全"
keywords: ["多模态安全对齐", "视觉语言模型", "强化学习", "图像描述", "安全防御"]
innovations: ["将LVLM安全对齐形式化为自述图问题，通过中间caption传递安全证据", "设计caption中介强化学习目标，结合冻结LLM安全性信号与直接答案评估", "采用指数风险折扣函数和组件级归一化平衡安全与视觉效用"]
benchmarks: ["MM-SafetyBench", "MSSBench", "VLSBench", "FigStep", "MIS-Test", "MM-Vet", "BLINK", "MMVP", "ERQA", "VPCT", "MMStar"]
---

# 论文速读：SafeCap: Improving LVLM Safety with Image Captioning Reinforcement Learning

## 一句话总结
SafeCap 提出一种基于强化学习的自述图（self-captioning）框架，让 LVLM 在回答问题前先显式生成安全相关的图像描述，并通过冻结 LLM 的安全性对齐信号进行奖励优化，从而在不牺牲视觉理解能力的前提下显著提升多模态安全性。

## 研究问题与动机
1. **视觉通路削弱纯文本安全对齐**：LVLM 继承自语言骨干的安全拒绝策略在处理视觉输入时可能失效，因为有害指令可通过图像中的OCR文字、对象语义或排版形式嵌入，绕过语言侧的防线。
2. **现有 caption-mediated 防御方法存在感知损失**：ECSO 等推理时方法将图像转文本后再利用语言侧安全机制，但图像到文本的转换可能丢失细粒度视觉细节，导致下游语言模型无法做出安全决策。
3. **训练时对齐难以兼顾安全与视觉效用**：直接进行安全 SFT/DPO 可能导致模型过度拒绝安全请求或损害视觉感知能力，需要在安全性与实用性之间取得平衡。
4. **视觉指令微调本身可能弱化安全性**：从 LLM 到 LVLM 的适配过程可能使内部表征偏离语言模型学到的拒绝行为，形成模态间的安全鸿沟。

## 核心贡献（创新点）
1. **安全感知自述图 formulation**：将 LVLM 安全对齐形式化为自述图问题，要求模型在生成最终答案前先输出显式的结构化中间表示（caption），作为安全相关证据通道。
2. **Caption 中介强化学习目标**：引入双信号奖励机制——直接评估策略模型最终答案的安全性，同时利用冻结的纯文本 LLM 判断生成的 caption 是否支持与其安全状态一致的答案，实现感知侧证据与输出侧对齐的结合。
3. **组件级归一化与指数风险折扣设计**：提出组件级的组内归一化（component-wise group normalization）避免不同尺度奖励干扰信用分配，并使用指数风险折扣函数 $S(u, h) = u \cdot \gamma^h$ 而非线性形式，有效抑制"什么都不做"的退化行为。
4. **多协议评估与消融分析**：提供 Direct、DirectCap、Prism 三种推理协议的统一评估框架，证明 SafeCap 在目标 DirectCap 路径上获得最稳定和显著的安全提升。

## 方法详解
**问题设定**：给定图像-问题对 $(x, q)$，策略模型 $\pi_\theta$ 生成结构化响应：
$$y = <\texttt{Caption}>c</\texttt{caption}>a$$
其中 $c$ 为图像描述，$a$ 为最终答案。

**训练目标**：使用 GRPO 在 SPA-VL 数据集上优化，每个样本采样 $K=8$ 个 rollout，优势函数为：
$$A_i = w_{\text{tmp}} \tilde{r}_{i,\text{tmp}} + w_{\text{cap}} \tilde{r}_{i,\text{cap}} + w_{\text{ans}} \tilde{r}_{i,\text{ans}}$$

**三类奖励设计**：
- **模板奖励 $R_{\text{tmp}}$**：基于规则解析确保响应包含完整 caption 块和非空答案，失败则跳过后续奖励。
- **答案奖励 $R_{\text{ans}}$**：使用 judge 模型评估保留度 $u(a)$ 和风险分 $h(a)$，采用指数折扣形式 $R_{\text{ans}} = u(a) \cdot \gamma^{h(a)}$，$\gamma \in (0,1)$ 控制风险抑制强度。
- **Caption 中介奖励 $R_{\text{cap}}$**：将 caption 传入冻结的纯文本 LLM 生成 $a_f$，通过二元安全对齐判定函数 $g(q, a, a_f) \in \{0,1\}$ 判断两者安全状态是否一致，并结合描述覆盖度 $u(c)$ 得到 $R_{\text{cap}} = g(q,a,a_f) \cdot u(c)$。

**组件级归一化**：
$$\tilde{r}_{i,k} = \frac{r_{i,k} - \mu_{G,k}}{\sigma_{G,k} + \epsilon}$$
避免不同尺度奖励在信用分配时耦合。

## 实验与结果
**实验设置**：
- 训练数据：SPA-VL 公开安全对齐数据集
- 基线模型：Qwen3.5-2B、Qwen3.5-2B-Base、Qwen3.5-4B、Qwen3.5-4B-Base
- 冻结 LLM：Qwen3-4B
- Judge 模型：gpt-oss-20b
- 超参数：$\gamma = 0.35$，奖励权重 $(w_{\text{tmp}}, w_{\text{cap}}, w_{\text{ans}}) = (0.5, 0.5, 1.0)$，学习率 $5 \times 10^{-7}$，训练 200 步

**主要结果**：
| 模型 | 协议 | S-Avg | V-Avg | 安全提升 |
|------|------|-------|-------|----------|
| Qwen3.5-2B-Base | DirectCap | 55.06 | 51.34 | +14.63 |
| Qwen3.5-4B-Base | DirectCap | 59.39 | 53.05 | **+18.96** |
| Qwen3.5-4B-Base | DirectCap | 55.06 | 51.34 | **+19.0** (对比零训练) |

**关键发现**：
- 在所有四个模型设置下，DirectCap 协议的平均安全提升达 3.7–19.0 点，而视觉效用基本保持不变或略有提升
- 4B-Base 模型表现最佳：安全平均分提升 19.0 点，视觉效用仅下降约 2 点
- 相比安全 SFT、DPO 和 SafeGRPO，SafeCap 在 DirectCap 协议上取得最优结果（SFT/DPO 仅提升约 2-3 点）
- 多种子鲁棒性验证：三次独立实验的 DirectCap S-Avg 为 50.83±1.14，统计显著（p=0.004）

## 相关工作脉络
1. **ECSO（Gou et al. 2024）**：推理时图像转文本防御方法，依赖语言模型自身安全能力，可能丢失细粒度视觉信息；SafeCap 通过 RL 训练模型内化 caption 生成能力。
2. **SPA-VL（Zhang et al. 2024）**：多模态安全偏好对齐数据集，本文在其上进行 RL 训练而非 SFT/DPO。
3. **MM-SafetyBench/VLSBench/FigStep/MIS**：多模态安全评测基准，本文扩展至 5 个安全基准 + 6 个视觉效用基准的统一评估。
4. **CapRL（Xing et al. 2025）**：基于 RL 的图像 caption 质量优化，将 caption 与 VQA 解耦；本文借鉴此设计但将目标从感知 VQA 转为安全对齐。
5. **SafeGRPO（Rong et al. 2025）**：需要额外安全标注的规则驱动 RL 方法；本文在相同数据上与 SafeGRPO 对比，证明无需额外标注即可取得更优效果。
6. **GDPO（Liu et al. 2026）**：组奖励解耦归一化策略优化；本文借鉴其组件级归一化思想但不使用批级白化步骤。

## 局限性与未来方向
1. **冻结 LLM 能力限制**：caption 奖励的有效性受限于冻结 LLM 的推理和安全判断能力，无法识别 caption 中的事实性错误（尽管人工监控未发现系统性幻觉）。
2. **安全性评估依赖人工规则**：当前使用固定 rubric 评估有用性和风险，缺乏完全可验证的安全奖励，是社区的重要开放问题。
3. **小规模模型验证**：实验仅在 Qwen3.5-2B/4B 上进行，更大模型和更多模型家族的泛化有待验证。
4. **Caption 内容边界**：若 caption 忠实复述图像中已存在的有害文本或意图，虽然不作为新有害信息处理，但仍需明确界定描述性内容的边界。
5. **未来方向**：可在 RL 前进行 caption-mediated SFT，但当前适合监督训练的数据有限。

## 研究启发与可借鉴点
1. **感知-对齐分离设计**：将视觉感知（caption 生成）与安全对齐（最终答案）解耦，通过中间表示传递安全证据的思路可迁移到其他需要多步骤推理的安全敏感任务。
2. **指数风险折扣替代线性惩罚**：$S(u,h) = u \cdot \gamma^h$ 的设计相比线性形式 $u - \beta h$ 能有效抑制"过度拒绝"和"什么都不做"的退化行为，在需要权衡效用与安全性的任务中具有普适价值。
3. **跨模型泛化验证**：Prism 协议中使用不同能力的冻结 LLM（Qwen3-4B vs Qwen3-14B）验证 caption 的泛化性，这种方法论可用于评估任何中间表示的迁移能力。
4. **组件级归一化的稳定性**：在组合多源奖励时，组件级归一化比全局归一化更能保持各信号的独立梯度，值得在复杂 RL 奖励设计中推广。
5. **多协议统一评估框架**：Direct/DirectCap/Prism 三种协议的对比分析提供了全面的方法诊断视角，可作为后续工作的标准评估范式。

## 关键术语表
**LVLM**：Large Vision-Language Model，大型视觉语言模型，扩展 LLM 指令跟随能力至视觉输入的多模态模型。
**DirectCap**：SafeCap 的目标推理协议，模型先生成带标签的 caption 再回答问题。
**Prism**：诊断协议，将生成的 caption 传入冻结的纯文本 LLM 进行评估，测试 caption 的跨模型可迁移性。
**GRPO**：Group Relative Policy Optimization，一种基于组内相对优势的强化学习算法，本文用于安全对齐训练。
**Caption-mediated reward**：通过冻结 LLM 对 caption 的安全性判断来评估 caption 质量的奖励设计。
**SPA-VL**：Safety Preference Alignment for Vision Language models，多模态安全偏好对齐数据集。
**Exponential risk-discount**：指数风险折扣函数 $S(u,h) = u \cdot \gamma^h$，非线性融合效用与风险。
**Component-wise group normalization**：组件级组内归一化，对各奖励分量独立标准化后加权组合。

## 可复现要素
- **数据集**：SPA-VL（公开），SafeTag-VL-3K（SafeGRPO 开源数据，用于对比实验）
- **代码**：已开源，https://github.com/Safe-VLM/SafeCap
- **项目主页**：https://safe-vlm.github.io/SafeCap
- **模型权重**：论文未明确说明是否开源，仅提供代码链接
- **关键超参**：
  - $\gamma = 0.35$（风险折扣系数）
  - $(w_{\text{tmp}}, w_{\text{cap}}, w_{\text{ans}}) = (0.5, 0.5, 1.0)$（奖励权重）
  - 学习率 $5 \times 10^{-7}$，batch size 64，rollout 数 $K=8$
  - 训练步数 200，最大 token 数 4096
- **硬件**：8 × NVIDIA H200 GPUs
- **冻结 LLM**：Qwen3-4B（caption 奖励使用）
- **Judge 模型**：gpt-oss-20b（确定性解码）

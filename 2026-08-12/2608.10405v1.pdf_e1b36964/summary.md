---
title: "Never Stop Speaking: a Denial-of-Service Attack on End-to-End Speech Language Models"
source: https://arxiv.org/pdf/2608.10405v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:34:50"
field: "多模态大模型安全"
keywords: ["DoS attack", "E2E ALLM", "adversarial audio", "speech large language model", "voice activity detection", "autoregressive generation", "multi-loss optimization"]
innovations: ["首个面向 E2E ALLM 的端到端 DoS 攻击框架，突破文本方法无法迁移的局限", "设计加权 EOS loss + top-k logit loss + 可微长度 loss + 语义对齐 loss 的复合优化目标，协同实现终止抑制与语义保持", "引入 VAD 区域约束策略，仅在有声段注入扰动，在攻击有效性与隐蔽性间取得更好权衡"]
benchmarks: ["OpenSLR", "QCRI", "LFM2.5-Audio-1.5B", "Fun-Audio-Chat-8B", "Qwen2-Audio-7B-Instruct"]
---

# 论文速读：Never Stop Speaking: a Denial-of-Service Attack on End-to-End Speech Language Models

## 一句话总结
本文首次系统研究了端到端语音大语言模型（E2E ALLM）的拒绝服务（DoS）攻击问题，提出了一种基于声学扰动的白盒攻击方法，通过多目标复合损失函数抑制 EOS 生成并延长解码过程，在 LFM2.5-Audio、Fun-Audio-Chat 和 Qwen2-Audio 上实现了 83%–87% 的攻击成功率，将平均输出长度延长至原始的 4 倍以上，同时几乎不损害语义一致性。

## 研究问题与动机
- 现有 DoS 攻击主要针对文本 LLM，依赖离散 token 操控（如 adversarial suffix、复杂语义构造），无法直接迁移到连续语音波形空间。
- 已有语音模型安全研究主要聚焦 ASR 或 TTS 系统，E2E ALLM 的 DoS 漏洞尚未被系统探索。
- 直接优化 EOS logits 仅能部分控制生成，可能破坏语义一致性或导致解码不稳定（提前终止而非延长生成）。
- 需要在连续音频空间中构造不可感知扰动，在保持输入语义不变的前提下，迫使模型持续自回归解码并产生大量无意义静音段，从而耗尽 GPU 计算资源。

## 核心贡献（创新点）
- **首个面向 E2E ALLM 的 DoS 攻击框架**：填补了端到端语音大模型安全领域的空白，突破了文本 DoS 方法无法迁移的局限。
- **多目标复合损失函数设计**：联合优化加权 EOS loss、top-k logit loss、可微长度 loss 和语义对齐 loss，实现"抑制终止 + 引导延长 + 保持语义"的协同控制。
- **VAD 驱动的隐蔽扰动注入策略**：仅在语音活动区域添加扰动，避免静音区引入可感知伪影，在攻击有效性与隐蔽性之间取得更好权衡。
- **系统性的实验验证与消融分析**：在三个开源 E2E ALLM 上验证了方法的有效性与跨解码策略的鲁棒性，揭示了当前 E2E ALLM 在生成控制层面的内在脆弱性。

## 方法详解
- **威胁模型**：白盒设置，攻击者拥有目标模型的完整架构与参数访问权限，可计算输入波形的梯度；仅能对输入音频添加不可感知的微小扰动（$\ell_\infty$ 约束）。
- **VAD 分割**：使用 Voice Activity Detection 将原始音频 $\mathbf{x}$ 分割为有声段 $V = \{x_1, x_2, \ldots, x_n\}$ 与静音段 $S = \{y_1, y_2, \ldots, y_n\}$，扰动仅注入有声段，最终重构为 $\hat{x} = \text{Concat}(\{x_i + \sigma_i\}, S)$。
- **复合损失函数**：$\mathcal{L}_{DoS} = \mathcal{L}_{eos} + \mathcal{L}_{topk} + \mathcal{L}_{len} + \mathcal{L}_{sem}$。
  - **加权 EOS loss**：$\mathcal{L}_{eos} = \frac{1}{N}\sum_{t=1}^N w_t \ell_{EOS,t}$，其中权重 $w_t$ 包含符号门控（仅对 $\ell_{EOS,t}>0$ 的步骤施加惩罚，排除无信息的非正 logit 步骤）与指数增长分量（$w_t = w_{high}\exp(\frac{t}{N}\kappa)$，逐步加大后期步骤的重视程度）。
  - **top-k logit loss**：$\mathcal{L}_{topk} = -\frac{1}{N}\sum_{t=1}^N \frac{1}{K}\sum_{k=1}^K \ell_{topk,t}^{(k)}$，通过增大 top-k token 的 logit 间接压低 EOS 概率，同时维持音频连贯性。
  - **长度 loss**：基于每步 EOS 概率 $p_t = \sigma(z_t)$ 建模期望生成长度 $\mathbb{E}[L] = \sum_{t=1}^N t(p_t \prod_{i=1}^{t-1}(1-p_i))$，令 $\mathcal{L}_{len} = (N_{max} - \mathbb{E}[L])^2 / N_{max}$，引导输出逼近最大 token 上限。
  - **语义对齐 loss**：$\mathcal{L}_{sem} = 1 - \frac{\langle s(x), s(x+\delta)\rangle}{\|s(x)\|_2 \|s(x+\delta)\|_2}$，通过 speech encoder 提取原始与扰动音频的特征并最大化余弦相似度，确保扰动不改变语义。
- **优化流程**：采用 PGD 迭代 200 步，每步将扰动初始化为高斯噪声后与静音段拼接，经 speech encoder + autoregressive decoder 后记录 EOS logit、top-k logit 与输出长度，反向传播更新扰动，约束 $\|\sigma_i\|_\infty \leq \epsilon = 10^{-4}$。

## 实验与结果
- **模型**：LFM2.5-Audio-1.5B、Fun-Audio-Chat-8B、Qwen2-Audio-7B-Instruct。
- **数据集**：OpenSLR 与 QCRI，各采 100 个干净音频样本进行评估。
- **超参**：$N_{max}=1024$，$w_{high}=4$，$\kappa=4$，$k=3$，PGD 迭代 200 次，$\epsilon=10^{-4}$。
- **主要结果**（Table 1 & Table 5）：
  - LFM2.5-Audio：ASR = 0.87，平均输出长度 = 941.88 tokens，GPU 显存 = 10.78 GB（clean 为 8.89 GB）。
  - Fun-Audio-Chat：ASR = 0.84，平均输出长度 = 920.24 tokens，GPU 显存 = 21.93 GB（clean 为 20.26 GB）。
  - Qwen2-Audio：ASR = 0.83，平均输出长度 = 913.07 tokens，GPU 显存 = 19.94 GB。
  - 响应质量（LLM-as-a-Judge，1–5 分）：4.75 分，与 clean 输入几乎无差异。
- **对比基线**：Simple Loss（仅 EOS 抑制）、Crabs（文本 LLM DoS）、ExtendAttack（LRM DoS）、Random Noise，本文方法在所有指标上均显著领先，ASR 提升约 10–53%（相对 Simple Loss），输出长度约为 clean 输入的 4.7 倍。
- **交叉解码策略验证**：在 greedy decoding 下仍保持 ASR 84%–86%，输出长度超过 900 tokens，说明攻击不依赖采样随机性。
- **迁移性**：跨模型黑盒迁移 ASR 为 7%–13%，同规模模型间迁移效果更好。
- **消融结论**：移除 $\mathcal{L}_{eos}$ 导致 ASR 骤降至 0.12、长度降至 289 tokens；移除 $\mathcal{L}_{sem}$ 使响应质量从 4.75 降至 3.92；加入 VAD 使响应质量提升 0.23 分；$\kappa=4$ 在各模型上表现最佳。

## 相关工作脉络
- **文本 LLM DoS 攻击**（Auto-DoS、P-DoS、NaturalSloth 等）：依赖离散 token 的 adversarial suffix 或复杂语义构造，无法直接迁移至连续音频域；本文填补了这一空白。
- **ASR 模型 DoS 攻击**（SlothSpeech, Haque et al. 2023）：仅抑制 EOS logits 作用于语音识别模块，未考虑 E2E ALLM 的复杂自回归生成过程；本文扩展至端到端生成场景并引入多损失联合优化。
- **文本 LLM DoS 攻击**（Crabs, Zhang et al. 2024；ExtendAttack, Zhu et al. 2025）：分别在自然 LLM 和 LRM 上有效，但在音频域 ASR 仅 0.34–0.43，验证了离散 token 策略对连续模态的失效。
- **语音对抗攻击**：既往工作多聚焦 ASR 误识别或 TTS 失真，对 E2E ALLM 的生成行为操控缺乏研究；本文首次系统性揭示该类模型的 DoS 脆弱性。
- **VAD 在对抗攻击中的应用**（Ko et al. 2026）：此前仅在语音识别对抗样本中用于限制扰动区域；本文将其与多损失优化结合，服务于生成长度的精确控制。

## 局限性与未来方向
- **压缩鲁棒性不足**：MP3 等有损压缩及重采样、滤波等信号变换会破坏优化得到的扰动模式，降低攻击效果（Figure 4(b)）。
- **防御成本与延迟权衡**：现有防御（压缩、降噪）会引入额外处理开销并可能损害音质，难以在实际语音交互系统中部署。
- **自适应攻击风险**：若攻击者感知预处理流水线，可针对该流程重新优化扰动，削弱防御有效性。
- **未来方向**：开发低延迟、保持音质的实用防御机制；研究跨模型通用防御策略；探索 E2E ALLM 生成控制层面的内在安全改进。

## 研究启发与可借鉴点
- **多损失联合优化范式**：将"直接控制目标（EOS/长度）"与"间接约束（top-k 分布/语义对齐）"分离设计，可迁移至其他连续模态大模型（如视频、多模态生成模型）的安全评估。
- **可微长度近似方法**：基于 EOS 概率分布建模期望生成长度（公式 6–7），为非可微离散长度优化提供了可借鉴的替代方案，适用于任何自回归生成任务的长度控制研究。
- **动态权重门控机制**：基于 logit 符号的门控 + 指数增长权重设计（公式 5–6），可有效筛选"有优化价值"的生成步骤，避免无效梯度噪声，可推广至其他序列生成对抗攻击。
- **VAD 区域约束策略**：将扰动限定在信息密集的有声段而非全波形，在提升隐蔽性的同时维持攻击效果，为语音/音频领域的对抗样本构造提供了实用的先验约束思路。
- **与团队方向的结合机会**：若团队关注多模态大模型安全或语音交互系统可靠性，可将此攻击框架扩展至图文/视频生成模型，或用于评估 E2E ALLM 在真实服务场景下的资源消耗边界。

## 关键术语表
- **E2E ALLM（End-to-End Audio Large Language Model）**：无需显式 ASR/TTS 中间环节、直接在连续语音波形与文本 token 之间进行双向理解与生成的 Transformer 基础架构大模型。
- **DoS（Denial-of-Service）攻击**：通过构造特殊输入迫使模型生成过量输出，耗尽计算资源并导致合法请求无法得到服务的攻击类型。
- **VAD（Voice Activity Detection）**：利用能量阈值、过零率或深度学习模型区分音频信号中有声段与静音/噪声段的基础语音处理技术。
- **EOS（End-of-Sentence）token**：自回归生成过程中用于标记输出终止的专用 token，其 logit 概率直接影响生成长度。
- **PGD（Projected Gradient Descent）**：在扰动约束空间内迭代执行梯度下降并投影回可行域的对抗样本优化算法。
- **ASR（Attack Success Rate）**：本文中指成功迫使模型输出达到最大 token 限制的 adversarial 样本比例。
- **LLM-as-a-Judge**：使用 ChatGPT 等大语言模型作为裁判，对生成文本的语义一致性与质量进行自动化评分的方法。

## 可复现要素
- **模型**：LFM2.5-Audio-1.5B（开源）、Fun-Audio-Chat-8B（开源）、Qwen2-Audio-7B-Instruct（开源）。
- **数据集**：OpenSLR（公开）、QCRI（公开）。
- **代码/权重**：论文未明确声明代码开源状态，模型权重为开源模型。
- **关键超参**：$N_{max}=1024$，PGD 迭代 200 次，$\epsilon=10^{-4}$（$\ell_\infty$），$w_{high}=4$，$\kappa=4$，$k=3$（top-k）。
- **评估工具**：ChatGPT 5.5（LLM-as-a-Judge，prompt 见 Appendix Figure 5）。

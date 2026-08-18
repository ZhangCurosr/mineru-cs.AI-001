---
title: "Never Stop Speaking: a Denial-of-Service Attack on End-to-End Speech Language Models"
source: https://arxiv.org/pdf/2608.10405v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:13:51"
field: "多模态大模型安全与对抗鲁棒性"
keywords: ["E2E ALLM", "Denial-of-Service", "Adversarial Attack", "Speech Language Model", "Audio Security", "Autoregressive Generation", "VAD"]
innovations: ["首次系统研究端到端语音大模型的DoS攻击漏洞，填补E2E ALLM安全空白", "提出多目标复合损失函数联合优化EOS抑制、长度引导与语义保持，显著提升攻击成功率与隐蔽性", "引入VAD引导的局部扰动策略，仅在人声区域注入对抗噪声以平衡攻击效能与听觉不可感知性"]
benchmarks: ["OpenSLR", "QCRI"]
---

# 论文速读：Never Stop Speaking: a Denial-of-Service Attack on End-to-End Speech Language Models

## 一句话总结
本文首次系统研究端到端语音大模型（E2E ALLMs）的拒绝服务（DoS）攻击漏洞，提出一种基于多目标优化损失函数的白盒对抗攻击方法，通过优化不可感知声学扰动抑制 EOS 预测并诱导模型持续自回归解码，在多个开源语音大模型上实现高攻击成功率与显著的计算资源消耗。

## 研究问题与动机
- **现有文本 DoS 攻击无法直接迁移**：现有 LLM DoS 攻击依赖离散 token 级别的提示词工程（如对抗后缀、语义诱导），而 E2E ALLMs 处理连续波形输入，token 级操控不可行。
- **语音模型安全研究存在空白**：既有语音模型安全工作主要聚焦 ASR 或 TTS 系统，端到端语音大模型的 DoS 脆弱性几乎未被探索。
- **直接优化 EOS logits 存在缺陷**：仅抑制 EOS 输出缺乏对生成其他维度的约束，可能导致语义一致性下降或解码不稳定。
- **连续波形扰动需同时满足多重约束**：扰动需在连续波形空间优化、保持不可感知性、不改变原始输入长度，且需保证语义不变。

## 核心贡献（创新点）
- **首次系统研究 E2E ALLM 的 DoS 攻击**：填补端到端语音大模型安全领域空白，揭示当前 ALLM 系统在自回归生成控制机制上的潜在脆弱性。
- **提出多目标复合损失函数**：联合加权 EOS loss、top-k logit loss、长度损失与语义对齐损失，实现终止抑制、长度延长与语义保持的协同优化，相比单一 EOS 抑制基线显著提升攻击成功率与稳定性。
- **引入 VAD 引导的局部扰动策略**：仅在人声活跃区域注入对抗扰动，避免沉默区域产生可听噪声伪影，在保持攻击效能的同时提升隐蔽性。
- **提供全面的攻击效能与鲁棒性评估**：在三个开源 E2E ALLM 上验证方法有效性，并系统分析解码策略、跨模型迁移性、压缩鲁棒性等维度。

## 方法详解
**威胁模型**：白盒攻击场景，攻击者拥有目标模型完整架构与参数访问权限，可计算输入波形的梯度；仅可引入低于人耳感知阈值的微小对抗扰动，不修改模型或部署环境。

**整体框架**（见图 2）：
1. 利用 VAD 将输入音频 $x$ 分割为有声段 $V = \{x_1, x_2, \ldots, x_n\}$ 与静音段 $S = \{y_1, y_2, \ldots, y_n\}$；
2. 对每个有声段优化加噪 $\sigma_i$，约束 $\|\sigma_i\|_\infty \leq \epsilon$；
3. 重构对抗样本 $\hat{x} = Concat(\{x_i + \sigma_i\}_{i=1}^n, S)$；
4. 通过 PGD 迭代优化，最小化复合损失 $\mathcal{L}_{DoS}$。

**复合损失函数**：$\mathcal{L}_{DoS} = \mathcal{L}_{eos} + \mathcal{L}_{topk} + \mathcal{L}_{len} + \mathcal{L}_{sem}$

**直接损失**：
- **加权 EOS logits loss**：对每个自回归步的 EOS logit 赋予动态权重 $w_t$，结合符号门控（仅正 logit 参与优化）与指数增长分量（关注后期步）：
$$w_t = \begin{cases} w_{high} \exp\left(\frac{t}{N}\kappa\right), & \ell_{EOS,t} > 0 \\ 0, & \ell_{EOS,t} \leq 0 \end{cases}$$
$$\mathcal{L}_{eos} = \frac{1}{N}\sum_{t=1}^N w_t \ell_{EOS,t}$$
- **长度损失**：基于 EOS 概率分布估计期望生成长度 $\mathbb{E}[L]$，通过 $(N_{max} - \mathbb{E}[L])^2$ 引导模型向最大长度边界延伸：
$$\mathbb{E}[L] = \sum_{t=1}^N t\left(p_t \prod_{i=1}^{t-1}(1-p_i)\right), \quad \mathcal{L}_{len} = \frac{(N_{max} - \mathbb{E}[L])^2}{N_{max}}$$

**间接损失**：
- **top-k logit loss**：增大模型 top-k 输出 token 的 logit，间接抑制 EOS 概率并保持生成连贯性：
$$\mathcal{L}_{topk} = -\frac{1}{N}\sum_{t=1}^N \frac{1}{K}\sum_{k=1}^K \ell_{topk,t}^{(k)}$$
- **语义对齐损失**：通过语音编码器提取原始音频与加噪音频特征，最大化余弦相似度：
$$\mathcal{L}_{sem} = 1 - \frac{\langle s(x), s(x+\delta)\rangle}{\|s(x)\|_2 \|s(x+\delta)\|_2}$$

**优化流程**：每步随机初始化高斯噪声，经 VAD mask 处理后送入目标模型语音编码器提取声学特征，再由自回归解码器生成；记录每步 EOS logit、top-k logits 与最终长度，计算 $\mathcal{L}_{DoS}$ 并通过 PGD 更新扰动，直至达到 $N_{max}=1024$ 或迭代上限。

## 实验与结果
**数据集与模型**：使用 OpenSLR 与 QCRI 两个公开语音数据集，每个采样 100 条干净音频；评估模型包括 LFM2.5-Audio-1.5B、Fun-Audio-Chat-8B、Qwen2-Audio-7B-Instruct。

**评估指标**：攻击成功率（ASR）、平均输出 token 长度、峰值 GPU 显存占用、响应质量（1-5 分，LLM-as-Judge 评估）。

**主要结果**（Table 1）：
- **LFM2.5-Audio**：ASR 达 87%，平均输出长度 941.88 tokens（相比干净输入 198.34 提升约 4.7 倍），显存 10.78 GB vs 8.89 GB；
- **Fun-Audio-Chat**：ASR 达 84%，平均输出长度 920.24 tokens（相比干净输入 213.57 提升约 4.3 倍），显存 21.93 GB vs 20.26 GB；
- **Qwen2-Audio**：ASR 达 83%，平均输出长度 913.07 tokens，显存 19.94 GB。

**基线对比**：显著优于 Simple Loss（ASR 约 0.77-0.80）、Crabs（ASR 约 0.34-0.43）、ExtendAttack（ASR 约 0.29-0.48），证明文本 DoS 策略无法直接迁移至连续语音域。

**消融实验**（Table 2）：移除 $\mathcal{L}_{eos}$ 导致 ASR 从 0.84 骤降至 0.12，验证其为核心驱动项；移除 $\mathcal{L}_{sem}$ 使响应质量从 4.75 降至 3.92，表明语义对齐对隐蔽性至关重要；VAD 策略对攻击成功率影响不大（0.84 vs 0.84），但能提升响应质量（4.75 vs 4.52）。

**参数分析**（Table 3）：$\kappa=4$ 时取得最佳或相当性能，过大或过小均导致 ASR 下降。

**鲁棒性**：在额外添加噪声时攻击仍保持有效；MP3 压缩可显著削弱攻击效果，提示实际部署中音频压缩可作为防御手段。

## 相关工作脉络
- **Text-based DoS 攻击**（如 Crabs、ExtendAttack、P-DoS）：依赖离散 token 操纵或 poisoned fine-tuning，通过对抗后缀或复杂语义诱导长输出；本文指出这些方法因模态差异无法直接迁移至连续语音域。
- **ASR DoS 攻击**（如 SlothSpeech）：仅针对语音识别模型的 EOS logit 进行抑制，未考虑 E2E ALLM 的端到端自回归生成过程与多模态对齐机制。
- **语音对抗样本生成**：传统方法多关注 ASR/TTS 系统的误分类或语义破坏，本文首次聚焦于"无限生成"导致的计算资源耗尽型攻击。
- **VAD 在对抗攻击中的应用**：已有工作（如 Ko et al. 2026）将 VAD 用于避免静音区域噪声，本文进一步将其与多目标优化联合，平衡攻击效能与隐蔽性。
- **E2E ALLM 架构**（如 Ichigo、LFM2.5）：采用 text-speech 对齐策略实现双向理解与生成，本文揭示此类模型在自回归终止控制机制上存在系统性漏洞。

## 局限性与未来方向
- **压缩鲁棒性不足**：MP3 等有损压缩会破坏精细声学扰动模式，降低攻击效果，限制在实际通信信道中的适用性。
- **仅验证白盒场景**：虽然分析了跨模型迁移性（7%-13% ASR），但未充分探索黑盒优化策略或自适应攻击者场景。
- **防御机制尚未建立**：论文指出当前缺乏能高效消除对抗扰动、同时保持低延迟与高语音质量的防御方法，这是重要未来方向。
- **目标模型范围有限**：仅评估三个开源模型，未涵盖更广泛的商用 E2E ALLM 系统（如 Real-time voice assistant 产品）。
- **扰动幅度假设理想化**：$\epsilon=10^{-4}$ 的 $\ell_\infty$ 约束在真实用户交互中可能被预处理管线抵消。

## 研究启发与可借鉴点
- **多目标损失设计的可迁移性**：加权 EOS suppression + length guidance + semantic preservation 的组合策略可用于其他模态（如视觉语言模型、多模态推理模型）的 DoS 攻击分析，揭示自回归生成控制的共性脆弱点。
- **VAD 局部扰动的通用范式**：仅在"信息密集"区域（人声、视觉关注区域）注入扰动、避免冗余区域的策略，可推广至视频、图像等多模态对抗攻击，提升隐蔽性与鲁棒性。
- **期望长度估计的可微分化技巧**：基于概率分布的 $\mathbb{E}[L]$ 公式将离散生成长度转化为可微目标，该思路可用于任何自回归模型的长度可控生成或攻击任务。
- **LLM-as-Judge 评估响应的范式**：使用 ChatGPT 对生成文本进行语义一致性评分，为多模态输出质量评估提供了轻量且可复用的基准方法。
- **与团队方向结合机会**：若团队关注语音大模型部署安全，可借鉴本文思路构建"主动探测"流程，系统扫描不同 ALLM 架构的 DoS 脆弱性；亦可反向思考，设计基于 early-stopping 或 EOS 正则化的防御机制。

## 关键术语表
- **E2E ALLM**（End-to-End Audio Large Language Model）：无需显式中间转录，直接处理连续波形输入并输出语音/文本的自回归大模型架构。
- **DoS Attack**（Denial-of-Service Attack）：通过构造对抗输入迫使模型生成异常长输出，耗尽计算资源从而导致服务不可用的攻击类型。
- **VAD**（Voice Activity Detection）：检测音频信号中人声活跃区域与静音/噪声区域的信号处理技术，本文用于引导局部扰动注入。
- **EOS Token**（End-of-Sentence Token）：自回归生成中的终止标记，抑制其概率可实现输出延长的核心攻击目标。
- **PGD**（Projected Gradient Descent）：受约束的梯度下降优化算法，用于迭代更新对抗扰动并投影回合法扰动范围。
- **Top-k Sampling**：解码策略，每次仅从概率最高的 k 个 token 中采样输出，本文用于间接增强非 EOS token 概率。
- **Semantic Alignment Loss**：通过余弦相似度约束原始音频与对抗音频的特征一致性，确保扰动不破坏语义内容。
- **LLM-as-Judge**：使用大规模语言模型作为裁判对生成结果进行自动质量评估的方法。

## 可复现要素
- **数据集**：OpenSLR 与 QCRI 公开数据集，论文未明确提及具体版本与许可信息；
- **代码**：论文未声明开源代码仓库；
- **模型权重**：使用的 LFM2.5-Audio-1.5B、Fun-Audio-Chat-8B、Qwen2-Audio-7B-Instruct 均为开源模型；
- **关键超参**：$N_{max}=1024$，$\epsilon=10^{-4}$（$\ell_\infty$ 约束），PGD 迭代 200 次，$w_{high}=4$，$\kappa=4$，$k=3$（top-k）；
- **评估工具**：ChatGPT 5.5 作为 LLM-as-Judge 评估响应质量，具体 prompt 见 Appendix Figure 5。

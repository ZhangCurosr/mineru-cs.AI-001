---
title: "Never Stop Speaking: a Denial-of-Service Attack on End-to-End Speech Language Models"
source: https://arxiv.org/pdf/2608.10405v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:31:48"
field: "多模态大模型安全与对抗鲁棒性"
keywords: ["端到端语音大模型", "拒绝服务攻击", "对抗扰动", "自回归解码", "语音安全", "VAD", "多目标优化", "音频大语言模型"]
innovations: ["首个面向E2E ALLMs的声学扰动DoS攻击框架，突破文本token级操作限制", "设计加权EOS+top-k+长度期望+语义对齐的四元复合损失函数实现稳定长序列生成诱导", "结合VAD局部注入策略在保持人类不可感知性的同时最大化攻击成功率"]
benchmarks: ["OpenSLR", "QCRI", "LFM2.5-Audio-1.5B", "FunAudioChat-8B", "Qwen2-Audio-7B-Instruct"]
---

# 论文速读：Never Stop Speaking: a Denial-of-Service Attack on End-to-End Speech Language Models

## 一句话总结
论文针对端到端语音大语言模型（E2E ALLMs）提出了一种基于声学扰动的拒绝服务（DoS）攻击方法，通过优化不可感知的语音扰动直接抑制自回归解码过程中的EOS（终止）token预测，诱导模型生成大量包含无语义沉默片段的冗长音频输出，从而耗尽GPU计算资源并造成服务降级。

## 研究问题与动机
1. **现有DoS攻击无法直接迁移至语音模态**：文本类LLM的DoS攻击主要依赖对抗性后缀、语义诱导或指令分解等离散token操作策略，而E2E ALLMs处理的是连续声学波形，token级操纵不可行。
2. **语音模型安全研究存在盲区**：既有语音模型安全研究主要聚焦ASR（自动语音识别）或TTS（语音合成）系统，仅通过简单压制EOS logit实现攻击，未能覆盖E2E ALLMs复杂的多轮自回归生成过程。
3. **直接优化离散生成长度不可微**：生成长度由EOS预测决定，属于离散操作，无法直接通过梯度下降优化；同时过度压制终止信号可能破坏语义一致性或引发解码崩溃。
4. **E2E ALLMs快速部署带来安全风险**：LFM2.5-Audio、FunAudioChat、Qwen2-Audio等端到端语音对话模型正加速落地语音助手场景，但其白盒/黑盒攻击下的资源消耗漏洞尚未被系统探索。

## 核心贡献（创新点）
1. **首个面向E2E ALLMs的端到端DoS攻击框架**：不同于文本LLM的prompt工程策略，本文首次在连续声学空间中构造对抗扰动，填补了端到端语音大模型DoS安全研究的空白。
2. **多目标复合损失函数设计**：提出包含加权EOS loss、top-k logit loss、长度loss和语义对齐loss的联合优化目标，相比单一EOS压制基线实现了更高的攻击成功率和更稳定的长序列生成。
3. **VAD驱动的局部扰动注入策略**：利用语音活动检测（VAD）将对抗扰动仅注入有声段，避免无声区域引入可感知伪影，在保证攻击效果的同时提升隐蔽性。
4. **系统性的实验验证与迁移性分析**：在三个开源E2E ALLM（LFM2.5-Audio-1.5B、FunAudioChat-8B、Qwen2-Audio-7B）及两种解码策略（top-k采样、greedy解码）下均验证了攻击有效性，并报告了跨模型迁移率（7%–13%）。

## 方法详解
- **威胁模型**：白盒攻击设定，攻击者拥有完整模型架构与参数访问权限，可在输入音频上添加幅度低于人耳感知阈值的微小扰动（$\ell_\infty$约束 $\epsilon=10^{-4}$），不修改模型或部署环境。
- **VAD分割与重构**：通过VAD将原始音频$\mathbf{X}$划分为有声段集合$V=\{x_1,\ldots,x_n\}$与无声段集合$S=\{y_1,\ldots,y_m\}$，对抗扰动仅添加至有声段，最终重构为$\hat{x}=\text{Concat}(\{x_i+\sigma_i\},S)$。
- **复合损失函数**：$\mathcal{L}_{DoS}=\mathcal{L}_{eos}+\mathcal{L}_{topk}+\mathcal{L}_{len}+\mathcal{L}_{sem}$
  - **加权EOS loss**：$\mathcal{L}_{eos}=\frac{1}{N}\sum_{t=1}^{N}w_t\ell_{EOS,t}$，其中权重$w_t$由符号门控（仅$\ell_{EOS,t}>0$时有效）与指数增长项$w_{high}\exp(\frac{t}{N}\kappa)$构成，$w_{high}=4,\kappa=4$。
  - **长度loss**：基于每步EOS概率$p_t=\sigma(z_t)$估计期望生成长度$\mathbb{E}[L]=\sum_{t=1}^{N}t\cdot p_t\prod_{i=1}^{t-1}(1-p_i)$，再令$\mathcal{L}_{len}=(N_{max}-\mathbb{E}[L])^2/N_{max}$引导模型逼近最大生成长度$N_{max}=1024$。
  - **top-k loss**：$\mathcal{L}_{topk}=-\frac{1}{N}\sum_{t=1}^{N}\frac{1}{K}\sum_{k=1}^{K}\ell_{topk,t}^{(k)}$，增大top-K（$K=3$）token的logit以间接抑制EOS并维持音频连贯性。
  - **语义对齐loss**：$\mathcal{L}_{sem}=1-\frac{\langle s(x),s(x+\delta)\rangle}{\|s(x)\|_2\|s(x+\delta)\|_2}$，通过语音编码器特征余弦相似度保持原始音频语义不变。
- **优化流程**：采用PGD进行200轮迭代优化，每轮随机初始化高斯噪声作为扰动起点，经语音编码器提取声学特征后送入自回归解码器，记录各步EOS logit、top-K logit及总输出长度，反向传播更新扰动。

## 实验与结果
- **数据集**：OpenSLR与QCRI两个公开语音数据集，各随机采样100条干净音频进行测试。
- **评估模型**：LFM2.5-Audio-1.5B（轻量化实时对话）、FunAudioChat-8B、Qwen2-Audio-7B-Instruct。
- **基线方法**：Clean audio、Random Noise、Simple Loss（纯EOS logit求和）、Crabs（文本LLM DoS）、ExtendAttack（文本LLM DoS）、全波形无VAD扰动。
- **主要结果（top-k解码）**：
  - LFM2.5-Audio：ASR=87%，平均输出长度=941.88 tokens，GPU显存10.78 GB（vs 干净输入8.89 GB）
  - FunAudioChat：ASR=84%，平均输出长度=920.24 tokens，GPU显存21.93 GB（vs 干净输入20.26 GB）
  - Qwen2-Audio：ASR=83%，平均输出长度=913.07 tokens，GPU显存19.94 GB
- **Greedy解码鲁棒性**：LFM2.5-Audio ASR=86%、长度927.69；FunAudioChat ASR=84%、长度905.36；Qwen2-Audio ASR=85%、长度934.02，证明攻击不依赖采样随机性。
- **跨模型迁移率**：白盒对角线ASR为87%/81%/83%，跨模型迁移ASR为7%–13%，同规模模型间迁移性更高。
- **响应质量**：纯净输入与对抗输入生成的文本转录内容基本一致，LLM-as-a-Judge评分4.75/5，证明语义一致性得到良好保持。
- **最强结果**：在LFM2.5-Audio上达到87% ASR与941.88平均输出长度，相较Simple Loss基线（ASR 79%，长度857.38）ASR提升8个百分点、长度增加约10%。

## 相关工作脉络
1. **文本LLM DoS攻击**：P-DoS（投毒降低EOS logit）、Crabs（白盒/黑盒节点扩展构造DoS树）、Reasoning-Bomb（诱导超长推理路径）；本文区别于其核心在于将攻击域从离散token空间迁移至连续波形空间。
2. **ASR DoS攻击**：SlothSpeech通过压制ASR模型EOS logit实现拒绝服务；本文指出该思路仅作用于语音识别模块，无法覆盖E2E ALLMs包含文本-语音双向生成与对齐的完整自回归解码流程。
3. **语音对抗样本生成**：Ko et al. (2026)提出无声区域不加噪的VAD约束策略用于ASR攻击；本文继承该隐蔽性设计并扩展至ALLM的复合多目标优化场景。
4. **端到端音频大模型架构**：Ichigo（混合模态早期融合实时语音助手）、Sparks of Large Audio Models综述；本文揭示了此类新兴架构在自回归长度控制机制上的系统性漏洞。
5. **多模态对抗攻击通用范式**：传统图像/文本对抗攻击侧重分类错误率，本文聚焦资源消耗型DoS而非功能破坏，体现了大模型安全评估维度的拓展。

## 局限性与未来方向
- **压缩鲁棒性有限**：MP3等有损压缩会量化或丢弃细粒度声学细节，显著削弱对抗扰动效果（图4b）；采样率变换、滤波等信号级变换同样会降低攻击有效性。
- **预处理管道可被绕过风险**：实际部署中语音系统往往包含噪声抑制、回声消除、编码压缩等预处理模块，当前攻击未显式建模此类变换，黑盒场景下效果可能进一步衰减。
- **白盒假设限制**：方法依赖完整模型参数访问与梯度计算，在纯黑盒API调用场景下迁移率仅7%–13%，实用攻击门槛仍需降低。
- **防御机制缺失**：作者指出开发低延迟、保真度的抗DoS防御机制（如动态长度截断、EOS概率监控、预处理去噪）是重要未来方向。

## 研究启发与可借鉴点
1. **复合损失函数设计范式**：将"直接抑制终止信号+间接引导生成分布+显式约束长度期望+语义保持不变"的四元联合优化思路可迁移至其他多模态大模型的安全评估（如视觉-语言模型的超长输出诱导）。
2. **VAD局部扰动策略的可移植性**：针对含静音/噪声区域的连续信号输入（如音频理解、语音驱动 avatar 生成），基于语音/音素活动检测的局部对抗优化可作为通用隐蔽攻击模板。
3. **期望长度可微化技巧**：通过伯努利终止概率分布推导$\mathbb{E}[L]$并构造二次惩罚项，为离散序列长度优化提供了可直接用于梯度训练的闭式近似，该方法可推广至文本LLM的长度控制攻击或防御设计。
4. **资源消耗型DoS评估指标体系**：除ASR与输出长度外，引入Peak GPU Memory Usage作为直接经济指标，为云服务级大模型安全评测提供了可量化的资源消耗基准。
5. **跨解码策略鲁棒性验证**：同时在top-k采样与greedy解码下验证攻击稳定性，证明了扰动对自回归机制本身的影响而非采样噪声的依赖，这一验证框架值得纳入后续大模型安全标准。

## 关键术语表
- **E2E ALLM（End-to-End Audio Large Language Model）**：端到端音频大语言模型，无需显式中间文本表示即可直接理解与生成语音的大规模语言模型。
- **DoS（Denial-of-Service）攻击**：拒绝服务攻击，通过构造特殊输入迫使模型生成过长输出或进入死循环，从而耗尽计算资源使正常服务不可用。
- **EOS（End-of-Sequence）token**：终止序列的特殊token，模型生成该token时停止自回归解码；攻击核心目标即持续压制其logit概率。
- **VAD（Voice Activity Detection）**：语音活动检测，用于区分音频中有声段与无声/静音段的基础信号处理技术。
- **PGD（Projected Gradient Descent）**：投影梯度下降，带约束的迭代梯度优化方法，每步更新后将扰动投影回可行域（如$\ell_\infty$范数球内）。
- **top-k采样**：自回归解码策略，每步仅从logit最大的k个候选token中随机采样输出，相比greedy解码更具多样性。
- **LLM-as-a-Judge**：利用大语言模型作为评估器对生成结果进行质量打分的方法论范式。
- **Adversarial Perturbation**：对抗扰动，为隐藏恶意意图而在输入上叠加的微小人工噪声，旨在改变模型输出但不被人感知。

## 可复现要素
- **数据集**：OpenSLR与QCRI（公开数据集，论文未提及具体版本/切割比例）
- **代码/权重**：论文未声明开源仓库；测试模型（LFM2.5-Audio-1.5B、FunAudioChat-8B、Qwen2-Audio-7B-Instruct）均为开源模型，权重可从官方或HuggingFace获取
- **关键超参**：$\epsilon=10^{-4}$（$\ell_\infty$扰动上限）、$N_{max}=1024$（最大生成长度）、$w_{high}=4$、$\kappa=4$（EOS权重指数增长系数）、$K=3$（top-k loss候选数）、PGD迭代次数200
- **运行环境**：需支持gradient computation的GPU环境；单条样本攻击优化时间未报告

---
title: "Galaxea-G0-5-One-Autoregressive-Stream-for-Robot-Reasoning-a"
source: https://arxiv.org/pdf/2608.11739v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 10:44:54"
field: "具身智能与机器人基础模型"
keywords: ["Vision-Language-Action", "Autoregressive VLA", "Cross-Embodiment Action Tokenization", "Chain-of-Thought Reasoning", "Robot Manipulation", "Behavior Foundation Model", "Prompt-Driven Policy Steering"]
innovations: ["将跨构型可学习 ActionCodec 与原生 CoT 推理流统一到单自回归 stream 中，使 VLM 既是条件编码器又是决策者", "利用 CoT 内生于自回归流带来的 prompt 驱动行为调控涌现能力，零样本条件下指令措辞可改变政策行为"]
benchmarks: ["2025 BEHAVIOR Challenge", "DROID Environment & Object Zero-Shot", "Bridge-SimplerEnv", "RoboTwin 2.0", "LIBERO", "Pick-and-Place Benchmark (PP Bench)", "R1-Lite / R1-Pro Real-World Fine-Tuning"]
---

# 论文速读：Galaxea-G0.5-One-Autoregressive-Stream-for-Robot-Reasoning-a

## 一句话总结
论文提出了 **G0.5**，一个基于单个预训练 VLM（Qwen3.5 2B）的纯自回归 Vision-Language-Action 模型，通过引入跨构型 ActionCodec、原生 CoT 推理流和视觉记忆模块，使模型在单一 tokenize 序列中同时生成推理与动作，在 7 个独立评测 regime 上全面超越 SOTA。

## 研究问题与动机
- 当前主流 VLA（如 $\pi_0$、GR00T、SmolVLA）采用 **VLM-as-encoder** 架构，将预训练 VLM 降格为条件编码器，动作由独立 flow-matching / diffusion 专家生成，导致 VLM 的核心生成能力（CoT 推理、in-context learning、prompt 行为调控）被压缩到瓶颈中，难以真正影响行为。
- 早期自回归 VLA（RT-2、OpenVLA）虽保留 VLM 作为决策者，但随着控制频率、动作维度增加，每步逐 token 生成导致高频控制成本过高。
- 需要在保留 VLM 为决策主体的前提下，解决自回归动作生成的**效率瓶颈**和**跨构型泛化**两大核心难题。
- 现有 VLM-as-encoder 路线存在抗遗忘问题（反哺梯度损坏 VLM 预训练能力），Knowledge Insulation 等缓解手段实际上承认了 AR 动作监督对保护 VLM 能力的价值，但仍未真正回到端到端自回归路线。

## 核心贡献（创新点）
1. **可学习的跨构型统一 ActionCodec**：将异构机器人动作端到端映射到共享离散词汇表，支持不同自由度/控制频率/形态的机器人复用同一输出头；与 FAST 的固定 DCT 流水线及 Being-H0.5 / Green-VLA / HEX 的向量级对齐方案本质不同——后者仅在动作向量层面做结构对齐，而本文将其提升至 tokenizer 层面。
2. **原生 CoT 推理流（Native Chain-of-Thought）**：通过 8 类 CoT 模板（Subtask/BBox/Trace/ActionHint 四种原子的自由组合）在同一自回归流中训练推理与动作，推理与动作共享解码器、上下文与目标函数；与 ECoT、CoT-VLA、DualCoT-VLA 等"外挂式"或 VLM-as-encoder 配套推理模块的方案本质不同，CoT 不再是辅助监督信号而是与动作耦合的生成阶段。
3. **涌现的 prompt 驱动行为调控能力**：保留自回归接口使 VLM 的 in-context 语言能力直接接入动作生成，零样本探测显示每阶段指令措辞（副词修饰、空间提示、近义动词替换）可改变策略行为而无需重新训练。
4. **轻量化视觉记忆模块**：在 ViT 每 4 层插入因式分解时空注意力，融合多秒历史视觉上下文，同时通过随机丢弃历史帧防止过拟合与状态漂移。

## 方法详解
- **整体架构**：以 Qwen3.5 2B 为初始化 backbone，单 transformer decoder 在统一词表上生成可选 CoT 推理段与紧凑动作码，所有输入/输出序列化为单条自回归序列。
- **Token 序列模板**（Fig.2）： conditioning segment（多视角 RGB + 构型 ID + 任务指令 + 本体感知状态，user-side chat tokens，以 `<EOC>` 结束）→ generative segment（assistant-side chat tokens，以 `<EOV>` 标记推理/动作边界），训练仅对 generative 段计算 next-token cross-entropy 损失：$\mathcal{L}(\theta)=-\sum_{i\in\mathcal{G}}\log p_\theta(x_i|x_{<i})$，无辅助回归目标或蒸馏。
- **结构化动作标记化（ActionCodec）**：将每类机器人分解为独立运动部件（left_control/right_control/lower_body），每部分 pad 到统一最大维度后训练残差向量量化（RVQ）模型，并引入时序对比学习目标提升相邻动作的 token 一致性；生成时每轮 DoF-group marker（如 `<left_control_r>`）后跟 8 个动作码，激活部分预测、非激活部分直接跳过而不需 padding，天然支持稀疏动作生成。
- **原生 CoT**：每步随机采样 8 种 CoT 模板之一（含 no-CoT 基线），subtask 模板给予更高采样权重；推理目标包括任务分解（Subtask）、关键物体定位（BBox，2D bounding box）、2D 末端轨迹（Trace，来自 TraceVLA 启发）与动作提示（ActionHint），全部由同一 cross-entropy 目标联合监督。
- **视觉记忆**：参照 $\pi_{0.7}$ 与 MEM 的设计，在 Vision Transformer 每 4 层插入因式分解的时空注意力模块；最终层丢弃所有历史 token，训练时以 30% 概率随机丢弃全部历史帧作为正则化；本体感知用连续状态嵌入替换离散 text tokenizer 以保持与视觉帧的时间同步。
- **可选 flow-matching 加速头**：仅在推理阶段作为可选加速模块挂载在 AR trunk 的 hidden states 上，不参与预训练。

## 实验与结果
- **DROID 零样本环境+物体迁移**：在 DROID 数据 post-train 后部署到未见环境/物体的 Franka 平台上，10 个 tabletop 任务平均成功率 **82.5%**，vs. $\pi_{0.5}$-DROID 57.5%（+25.0pp）、MolmoAct2-DROID 52.0%（+30.5pp）；在"把方块放进抽屉并关上"多步任务上 MolmoAct2 完全失败而 G0.5 过半成功。半透明抽屉在无反光标记时表现较弱（60%），加高对比标记后恢复至 100%。
- **Bridge-SimplerEnv**（4 个 WidowX 任务）：post-train 80K steps（lr=$3\times10^{-5}$，state-free 输入），平均成功率 **87.3%**，SOTA（Xiaomi-Robotics-0：79.2%）。
- **RoboTwin 2.0**：clean 93.7%，randomized 92.8%，avg **93.3%**，SOTA（Fast-WAM 91.8%）。
- **LIBERO**：Spatial 98.4%、Object 100%、Goal 98.6%、Long 98.6%，avg **98.9%**，SOTA（Xiaomi-Robotics-0 98.7%）。
- **2025 BEHAVIOR Challenge**（50 个长程家务任务，单 checkpoint）：1 epoch post-train **0.2904**（+10.6pp vs. $\pi_{0.5}$ 4 epochs 的 0.2626），4 epochs **0.3136**，超越首名 RLC 方案（0.2605，使用 4 个 checkpoint）+20.4%。
- **R1-Lite / R1-Pro 真机微调**（6 个 task-embodiment 设置）：平均成功率 **76.7%**，vs. $\pi_{0.5}$ 53.3%（+23.4pp）、GR00T-N1.7 24.4%；过程得分 avg 129.2 vs. 105.2 / 68.9；共享任务（毛巾/纸箱折叠）跨构型 avg 75.0% vs. $\pi_{0.5}$ 43.3%。
- **PP Bench（Pick-and-Place）**：零样本语言跟随率 65.6%、成功率 59.4%；50H post-train 后语言跟随 84.4%、成功 75.0%，vs. $\pi_{0.5}$ 50H 下 68.8% / 65.6%；加入 cropped 目标视觉上下文后语言跟随提升至 98.4%。
- **CoT + 推理头消融（Sec 5.6）**：在 Air Fryer / Cook Bacon 两个零样本长程任务上，AR+CoT progress score 最高（Air Fryer 3.8/5、Bacon 3.4/5）；AR 比 FM 头更能利用 CoT（FM 下 CoT 增益明显更小），证明解码接口而非推理内容是关键。
- **RL 微调度（Sec 5.7）**：在 LIBERO 上对 AR 与 FM 两种 head 各施 GRPO（每任务仅 1 条 demo），AR 收敛更快、最终成功率更高、方差更低，归因于 AR 原生 token-level log-probability 使 ratio-based RL 无需额外重构。

## 相关工作脉络
- **$\pi_0$ / $\pi_{0.5}$ / GR00T-N1 / SmolVLA**：VLM-as-encoder 代表，本文在其基础上论证"VLM 应作为决策者而非条件编码器"的核心论点，并以端到端 AR 统一目标取代独立 flow-matching 专家。
- **RT-2 / OpenVLA / $\pi_0$-FAST**：自回归 VLA 先驱；本文相比 FAST 的关键区别在于 action tokenizer 是跨构型**端到端可学习**的（FAST 为每个构型单独固定 DCT+ BPE 流水线），且引入结构化 DoF-group 稀疏预测以降低解码负担。
- **ECoT / CoT-VLA / DualCoT-VLA / Halo**：将 CoT 作为辅助 VQA 目标或外挂推理模块；本文 CoT 与动作共享同一 decoder 与 cross-entropy 目标，是内生的而非旁路。
- **HAMSTER / $\pi_{0.5}$ high-level subtask**：高层次规划器+低层次执行器的分层方案；本文 CoT 是 autoregressive stream 内的 interleaved 推理，非层级接口。
- **Being-H0.5 / Green-VLA / HEX**：跨构型对齐工作在动作向量层面；本文将其提升到 tokenizer 层面，新增构型无需新增参数。
- **VLA-0**：证明无修改 VLM 即 AR 动作训练可超 $\pi_{0.5}$-KI；本文延续该信号，进一步解决高维/长程/跨构型场景下的可扩展性问题。

## 局限性与未来方向
- **感知短板**：半透明/低对比度物体（如白色半透明抽屉）定位能力弱，AR 本身无法弥补感知瓶颈，需依赖外部高对比度视觉线索。
- **视觉记忆窗口短**：当前模块仅捕捉数秒历史，长程任务仍依赖外部记忆机制；论文明确将其列为未解决问题。
- **预训练数据分布偏差**：pick-and-place 行为占主导，容器交互（微波炉/烤箱）类技能严重不足；$\pi_{0.5}$ 在这些任务上仍占优，但随训练 epoch 增加差距缩小。
- **低体（lower-body）控制未在实验中单独评估**：统一动作空间中虽已编码，但未单独测试其对双足/移动底盘机器人控制的贡献。
- **Prompt 驱动行为调控缺乏系统性定量研究**：零样本探测仅为定性观察，尚未建立规范的 prompt steering benchmark。

## 研究启发与可借鉴点
- **结构化动作标记化（Structured Action Decomposition）**：将动作空间按 DoF 部件解耦、引入 DoF-group marker 并只预测激活部分，可显著降低解码负担并自然支持稀疏生成；适用于任何需要多臂 / 全身协调的 VLA 系统。
- **跨模态内生的 CoT 训练范式**：将子任务分解、物体定位、轨迹预测等辅助目标直接嵌入 next-token 统一 loss，避免辅助任务的"监督信号隔离"问题；可推广至其他具身推理场景。
- **视觉记忆的"因果分离注意力+随机丢弃"设计**：因式分解时空注意力 + 30% 随机丢弃历史帧的组合，有效平衡长程依赖与误差累积风险，为 VLA 的视觉记忆模块提供了可复用的正则化策略。
- **AR 接口对 RL 优化的天然优势**：token-level log-probability 直接兼容 ratio-based RL（如 GRPO），无需像 flow-matching 那样构造辅助 SDE 或离散化重构；未来探索 VLA 与 offline/online RL 的结合时，AR backbone 是更友好的选择。
- **PP Bench 的"语言跟随率 vs. 任务成功率"双层指标体系**：将语义 grounding 失败与执行失败解耦评估，是 VLA 评测设计中值得借鉴的诊断框架。

## 关键术语表
- **ActionCodec**：本文提出的跨构型可学习动作 tokenizer，将异构机器人动作映射到 27 维统一离散词汇表，支持不同自由度机器人在同一动作空间下共享输出头。
- **Native CoT（原生链式推理）**：在同一自回归流中与动作 token 共享 decoder 与 cross-entropy 目标的推理 token 流，包括 Subtask/BBox/Trace/ActionHint 四种原子的组合。
- **Structured Action Tokenization（结构化动作标记化）**：将动作按运动部件（left/right control、lower body）分组后分别 RVQ 编码，并在生成时以 DoF-group marker 控制稀疏预测。
- **VLM-as-Encoder vs. VLM-as-Actor**：前者将 VLM 降级为条件编码器、动作由独立专家生成；后者让 VLM 自身直接生成动作 token，保留其完整生成能力。
- **Cross-Embodiment 泛化**：通过统一动作词汇表使不同形态/自由度的机器人共享同一策略输出头，无需为每个新构型新增 tokenizer 或 adapter。
- **PP Bench（Pick-and-Place Benchmark）**：本文提出的双层评测基准，分别报告语言跟随率与任务成功率，用于解耦语义 grounding 与低层执行失败。
- **Factorized Spatiotemporal Attention**：将视觉Transformer中的自注意力拆分为空间与时间两个独立模块，高效融合多帧历史上下文而避免二次复杂度。
- **AR+CoT vs. FM Head**：AR head 使用原始自回归 token 生成动作，FM head 为可选的 flow-matching 加速头；实验表明 AR 更能利用 CoT 推理内容。

## 可复现要素
- **数据集**：预训练使用 14 种构型的机器人数据集 + Web VQA 混合；DROID 评估用 DROID 公开数据集；Bridge-SimplerEnv 用 BridgeData V2；RoboTwin 2.0 与 LIBERO 为公开 benchmark；BEHAVIOR Challenge 使用公开 10,000 集演示；PP Bench 为团队自建（50 小时 R1-Lite 数据）。
- **代码/权重**：论文提供了项目主页 https://opengalaxia.github.io/G05/，但未在正文中明确开源声明；论文声称将 release pretrained backbone。
- **关键超参**：backbone Qwen3.5 2B；AdamW 优化；Bridge post-train 80K steps / lr=$3\times10^{-5}$；RoboTwin 4 epochs / lr=$4\times10^{-5}$ / batch=1024；LIBERO 100K steps / lr=$1\times10^{-5}$ / wd=$1\times10^{-2}$；视觉记忆随机丢弃概率 30%；CoT 模板加权随机采样（subtask 模板高权重）。

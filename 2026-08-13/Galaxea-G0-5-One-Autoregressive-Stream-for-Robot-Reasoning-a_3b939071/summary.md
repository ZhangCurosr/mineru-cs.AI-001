---
title: "Galaxea-G0-5-One-Autoregressive-Stream-for-Robot-Reasoning-a"
source: https://arxiv.org/pdf/2608.11739v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 10:45:32"
field: "具身智能与机器人学习"
keywords: ["Vision-Language-Action", "Autoregressive VLA", "Chain-of-Thought", "Action Tokenization", "Cross-Embodiment", "Robot Reasoning"]
innovations: ["统一自回归流中联合生成推理与动作", "跨embodiment学习型动作编解码器", "原生CoT流增强长程任务零样本执行"]
benchmarks: ["DROID Zero-Shot", "BEHAVIOR-1K Challenge", "LIBERO", "RoboTwin 2.0", "SimplerEnv-Bridge", "Pick-and-Place Benchmark"]
---

# 论文速读：Galaxea G0.5: One Autoregressive Stream for Robot Reasoning and Action

## 一句话总结
本文提出 **G0.5**，一种基于自回归（AR）的统一视觉-语言-动作（VLA）模型，将推理（链式思考）与动作生成整合在单一Transformer解码器与共享token流中，通过跨 embodiment 的动作编解码器、原生 CoT 流与视觉记忆模块，实现从预训练到物理机器人控制的高效迁移。

## 研究问题与动机
1. **VLM-as-encoder 架构削弱了 VLM 的生成能力**：当前主流 VLA 模型将预训练 VLM 仅作为条件编码器，动作由独立训练的 flow-matching 专家生成，导致 VLM 的推理、上下文学习与提示驱动能力无法直接作用于动作生成。
2. **自回归 VLA 难以扩展到高维、高频动作**：传统 AR 方案逐 token 预测连续动作，随着控制频率、动作维度增加，解码负担巨大，效率低下。
3. **跨 embodiment 动作表示缺乏统一离散接口**：不同机器人构型、自由度、控制频率的动作难以映射到共享词汇表，阻碍了基础模型的泛化。
4. **推理与动作分离导致长程任务执行能力弱**：现有方法中的 CoT 多为辅助训练信号或独立模块，未原生嵌入动作生成流，限制了多阶段任务的零样本分解与地面定位能力。

## 核心贡献（创新点）
1. **跨 embodiment 学习型动作编解码器（ActionCodec）**：通过残差向量量化（RVQ）将异构机器人动作映射到共享的 27 维 token 流，支持稀疏预测（仅激活部分关节组），与 FAST 的固定 DCT 方法相比，实现端到端跨形态学习。
2. **原生自回归链式思考（CoT）流**：将任务分解（Subtask）、物体边界框（BBox）、运动轨迹（Trace）与动作提示（ActionHint）四种推理原语作为可选项嵌入同一 token 流，与动作 token 共享解码器、上下文与目标函数，区别于将推理视为独立模块的插件式 CoT。
3. **提示驱动的 emergent 行为控制**：保留 AR 接口使得 VLM 的上下文语言能力直接连接动作生成，零样本实验中观察到指令措辞（副词、空间线索、近义动词）可无需微调地引导策略行为，揭示了结构上的优势。
4. **视觉记忆模块增强长程感知**：在视觉 Transformer 中每四层插入因子化时空注意力，融合多秒历史，并通过训练时随机丢弃历史帧防止过拟合，改善非马尔可夫长程任务的执行稳定性。

## 方法详解
- **模型主干**：基于 Qwen3.5 2B 预训练 VLM，保留其视觉编码器、共享多模态词汇表与自回归解码器。
- **Token 序列模板**：输入（多视图 RGB、embodiment ID、自然语言指令、本体感觉状态）与输出（可选 CoT 段 + 动作码）序列化为单一 token 流，仅对生成段计算交叉熵损失：$\mathcal{L}(\theta)=-\sum_{i\in\mathcal{G}}\log p_\theta(x_i|x_{<i})$。
- **结构化动作分词**：将机器人动作按运动部件分组（左控、右控、下肢），每组填充至共享维度后训练 RVQ，引入时间对比损失提升 token 一致性；生成时按残差轮次输出 DoF 组标记符与 8 个动作码，支持稀疏预测。
- **原生 CoT 监督**：每个机器人样本随机采样 8 种 CoT 模板之一（包括无 CoT 基线）进行训练，子任务文本赋予更高权重；推理时可按需开启/关闭 CoT。
- **视觉记忆**：在每个视觉 Transformer 层之后插入因子化时空注意力模块，历史帧在训练中以 30% 概率被完全丢弃，推断时使用单帧但保留时序归纳偏置。
- **联合训练数据**：14 种 embodiment 的机器人演示数据与网页规模 VQA 数据混合，统一转换为 27 维动作空间，通过自动标注管线生成多粒度语言、视觉 grounding 与动作轨迹注释。

## 实验与结果
- **DROID 零样本评估**：在 Franka 机械臂上，环境与物体均未见过，G0.5（DROID post-training）平均成功率 **82.5%**，较 π0.5-DROID（57.5%）提升 +25.0pp，较 MolmoAct2-DROID（52.0%）提升 +30.5pp。
- **SimplerEnv-Bridge**：WidowX 模拟环境，4 项任务平均成功率 **87.3%**，领先所有基线。
- **RoboTwin 2.0**：双臂操作，干净/随机设置平均成功率 **93.3%**（clean 93.7%，rand 92.8%）。
- **LIBERO**：Franka 四套件，平均成功率 **98.9%**，在长程（Long）子任务上表现最强。
- **2025 BEHAVIOR Challenge**：50 个长程家庭移动操作任务，G0.5 单检查点训练 **1 epoch** 即得分 **0.2904**，超越冠军方案（0.2605，4检查点）；**4 epoch** 达到 **0.3136**，较 π0.5（4 epochs, 0.2626）提升 +19.4%。
- **真实世界微调（R1-Lite/R1-Pro）**：6 项任务-embodiment 设置平均成功率 **76.7%**，显著高于 π0.5（53.3%）与 GR00T-N1.7（24.4%）。
- **Pick-and-Place Benchmark**：50H post-training 下语言跟随率 **84.4%**，任务成功率 **75.0%**，较同设置 π0.5 提升 +15.6pp / +9.4pp。
- **CoT 消融**：在零样本长程任务（Air Fryer、Cook Bacon）上，AR+CoT 相较于 AR-only 进步分数从 2.4→3.8、1.5→3.4，证明原生推理流的有效性。

## 相关工作脉络
1. **VLM-as-encoder 家族**：π0、π0.5、GR00T-N1/N1.5/N1.7、SmolVLA 等将 VLM 特征输入独立 flow-matching/diffusion 专家，本文认为此架构削弱了 VLM 的生成能力，主张回归 AR 主线。
2. **自回归 VLA**：RT-2、OpenVLA 采用离散化动作 token 预测，但未解决高维动作扩展问题；FAST/FAST+ 使用 DCT+BPE 压缩动作，本文的 ActionCodec 为学习型且跨 embodiment。
3. **插件式 CoT**：ECoT、CoT-VLA、DualCoT-VLA 等在 VLM-as-encoder 上附加推理模块，推理与动作分离；本文的 CoT 原生嵌入同一 AR 流，共享优化目标。
4. **跨 embodiment 动作对齐**：Being-H0.5、Green-VLA、HEX 等在动作向量层面对齐，本文将其提升至 tokenizer 层，实现统一离散接口。
5. **视觉记忆 VLA**：MEM、MemoryVLA 等引入历史上下文，本文采用因子化时空注意力并在预训练中随机丢弃历史帧，平衡性能与延迟。
6. **世界动作模型**：Fast-WAM 等探索未来想象，本文专注纯自回归主干，可选附加 flow-matching 头作为推理加速器。

## 局限性与未来方向
1. **低对比度视觉场景敏感**：在半透明抽屉等任务上表现下降，需显式高对比标记辅助，表明感知层面仍存在局限。
2. **视觉记忆历史窗口短**：仅融合数秒历史，长程记忆仍未解决，可能影响超长时间任务。
3. **下肢动作未单独评估**：统一动作空间包含下肢，但实验未独立分析其贡献。
4. **提示驱动行为控制的系统性研究不足**：零样本观察显示指令措辞可引导行为，但缺乏定量分析与机制验证。
5. **数据分布偏差**：预训练数据以 pick-and-place 为主，容器交互任务（如微波炉操作）表现弱于 π0.5，需扩充相关数据。

## 研究启发与可借鉴点
1. **AR 架构保护 VLM 能力**：将动作作为辅助 AR 目标可防止预训练 VLM 的感知/语言退化，为 VLA 训练提供正则化信号。
2. **结构化动作分解提升效率**：按运动部件分组并稀疏预测，大幅减少 token 数量，适用于高维机器人控制。
3. **原生 CoT 增强长程任务**：将推理原语（子任务、边界框、轨迹）嵌入同一 token 流，使模型具备零样本任务分解与地面定位能力。
4. **跨 embodiment 统一 token 接口**：学习型动作编解码器实现形态无关的动作表示，便于基础模型迁移。
5. **视觉记忆结合随机丢弃**：因子化时空注意力配合训练时随机丢弃历史帧，可在不增加推理延迟的前提下注入时序上下文。

## 关键术语表
**VLM-as-encoder**：将预训练视觉-语言模型仅用作条件编码器，动作由独立专家生成的 VLA 架构。
**Chain-of-Thought (CoT)**：在动作生成前或间插入推理 token 序列，以显式分解任务、定位物体或规划轨迹。
**ActionCodec**：将连续动作序列量化为离散 token 的学习型编解码器，支持跨机器人形态共享词汇表。
**Residual Vector Quantization (RVQ)**：多级向量量化方法，逐层细化动作表征，用于动作 token 化。
**Flow-matching**：一种连续动作生成方法，通过常微分方程学习从噪声到动作的匹配流。
**DoF-group marker**：表示当前激活自由度组的特殊 token，用于稀疏动作预测。
**Factorized spatial-temporal attention**：在视觉编码器中分离时空注意力的模块，高效融合多帧历史。
**Pick-and-Place Benchmark (PP Bench)**：本文提出的语言跟随与任务成功分离评测基准，用于评估 cluttered 场景下的指令遵循能力。

## 可复现要素
- **数据集**：机器人数据覆盖 14 种 embodiment，包括 DROID、BridgeData v2、LIBERO、RoboTwin 2.0 等；VQA 数据为公开网页与内部注释。论文未明确声明全部数据公开，但提到发布预训练 backbone。
- **代码/权重**：论文提到“released pretrained backbone”，具体开源状态需查看项目页面（https://opengalaxea.github.io/G05/）。
- **关键超参**：Qwen3.5 2B 主干；RVQ 27 维动作空间；学习率 π0.5 微调 3e-5、LIBERO 1e-5、RoboTwin 4e-5；batch size 1024；训练至收敛；视觉记忆随机丢弃概率 30%。

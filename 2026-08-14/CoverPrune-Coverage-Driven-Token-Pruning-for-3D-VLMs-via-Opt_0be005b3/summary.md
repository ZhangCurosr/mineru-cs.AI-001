---
title: "CoverPrune-Coverage-Driven-Token-Pruning-for-3D-VLMs-via-Opt"
source: https://arxiv.org/pdf/2608.13226v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:23:52"
field: "3D视觉语言模型推理加速"
keywords: ["Visual Token Pruning", "Optimal Transport", "3D VLM", "Spatial Reasoning", "Training-free Acceleration"]
innovations: ["将3D VLM Token剪枝从多样性最大化重新定义为覆盖度最大化，基于半松弛最优运输理论形式化", "设计FST多维运输代价与信息感知容量向量，联合建模特征-空间-时序多域Token关系", "提出SGS贪心算法与CoverPrune-Lite块结构OT近似，实现高效训练无关即插即用剪枝"]
benchmarks: ["ScanQA", "SQA3D", "Scan2Cap", "VSI-Bench"]
---

# 论文速读：CoverPrune: Coverage-Driven Token Pruning for 3D VLMs via Optimal Transport

## 一句话总结
论文提出 CoverPrune，一种无需训练的 3D VLM 视觉 Token 剪枝框架，将剪枝目标从"最大化多样性"范式转向"保留视觉证据覆盖度"，通过最优运输（Optimal Transport, OT）理论形式化覆盖-紧凑性联合优化，在多种 3D 视觉-空间推理基准上达到 SOTA 效率与精度。

## 研究问题与动机
- **3D VLM 推理的视觉 Token 爆炸瓶颈**：3D 视觉-语言模型将多帧视频或多视角观察注入视觉 Token 流，单个输入可产生数百至数千 Token，使注意力机制的二次复杂度和 KV Cache 膨胀成为推理部署的核心瓶颈。
- **现有剪枝方法在 3D 场景下存在代表性失真**：主流剪枝依赖"多样性最大化"策略——丢弃与其他 Token 最相似的样本——但在 3D 环境中，代表性原型的 Token 因与大量同模态 Token 相似而易被误删，保留集虽分散却偏向离群值，破坏多视角一致性与几何结构。
- **注意力分数在多域推理中存在不稳定因素**：Attention-based 方法易受 attention sink 和 prompt-dependent saliency 干扰，且注意力模式随层和解码步变化，难以作为可靠的 3D 空间推理证据选择依据。
- **已有 3D 专用剪枝方法未充分解决极端压缩下的几何/时序一致性**：DTC、EgoPrune 等方法虽有 3D 感知设计，但在高压缩率下对距离、方位、外观顺序等细粒度空间关系推理仍显著退化。

## 核心贡献（创新点）
- **范式转变：从多样性到覆盖度**：首次将 3D VLM 剪枝目标重新定义为"保留视觉证据覆盖度"而非"最大化 Token 间多样性"，为 3D 空间推理场景下的 Token 选择提供了全新的理论视角。
- **FST 多维运输代价（Feature-Spatial-Temporal Cost）**：设计融合特征语义相似性、3D 空间邻近性和时序一致性的综合代价矩阵，使保留 Token 的覆盖分配同时满足语义对齐、几何完整和时序连贯。
- **信息感知型 FST 容量向量**：通过局部 FST 差异性计算每个目标 Token 的独特性分数，将其映射为非均匀目标容量，在极端剪枝下防止冗余密集区域主导覆盖目标。
- **半松弛 OT + SGS 高效贪心求解**：将 NP-hard 的组合子集选择问题转化为半松弛 OT（Semi-Relaxed OT），并基于 3D 空间局部性先验设计 SGS 算法，避免每步全局重算，实现可训练推理的近似最优选择。
- **CoverPrune-Lite 轻量变体**：通过 Morton 码 3D 有序分组和块对角 OT 近似，以 O(N log N) 复杂度实现单次扫描的局部最优原型选择，几乎无性能损失地大幅降低剪枝开销。

## 方法详解
- **问题形式化**：给定全部视觉 Token 集合 $\mathcal{T} = \{t_j\}_{j=1}^{N}$，每个 Token 含特征嵌入 $\mathbf{f}_j$、3D 全局坐标 $\mathbf{x}_j$ 和时间戳 $\tau_j$，在保留比 $R$ 下选取大小为 $K=\lceil RN\rceil$ 的子集 $S$，最小化覆盖损失：
$$S^\star = \arg\min_{S\subseteq\mathcal{T},\,|S|=K} \mathcal{L}_{\mathrm{OT}}(S;\mathcal{T})$$
其中 $\mathcal{L}_{\mathrm{OT}}$ 以 OT 距离衡量保留集对全集的覆盖失真。

- **FST 运输代价**：特征、空间、时序三个维度的成对差异分别为 $d_f(s,t)=1-\cos(\mathbf{f}_s,\mathbf{f}_t)$、$d_x(s,t)=\|\mathbf{x}_s-\mathbf{x}_t\|_2$、$d_\tau(s,t)=\mathrm{ReLU}(\tau_s-\tau_t)$（后者以 ReLU 惩罚以较晚 Token 覆盖较早 Token）。总代价为 $C_{ij} = \lambda_f\hat{d}_f + \lambda_x\phi_\kappa(\hat{d}_x) + \lambda_\tau\hat{d}_\tau$，其中 $\phi_\kappa(x)=\log(1+\kappa x)/\log(1+\kappa)$ 为动态范围扩展非线性映射。

- **FST 容量向量**：对每个目标 Token 计算其 FST 邻域内平均差异作为独特性分数 $r_j$，经单调映射 $\phi(\cdot)$ 归一化得容量 $v_j$，使局部 distinctive 的 Token 获得更高运输质量权重。

- **半松弛 OT 与 SGS 算法**：引入目标侧松弛约束 $\mathbf{P}^\top\mathbf{1}\leq\mathbf{v}$，使子集选择目标具有单调次模性，从而可贪心逼近。SGS 利用 3D 局部性：每步仅需在候选 Token 的空间邻域内计算残差加权边际代价 $t_\ell^\star=\arg\min_t\sum_{t_j\in\mathcal{M}_g(t)}r_{\ell,j}C(t,t_j)$，避免全局 OT 重算。

- **CoverPrune-Lite 的块结构近似**：使用 Morton 码空间填充曲线对 Token 进行 3D 感知排序，按容量累加至 $1/K$ 切分为 K 个组，每组内选一个使容量加权运输代价最小的原型 Token，等效于块对角约束下的 OT 优化，将复杂度降至 O(N log N)。

## 实验与结果
- **数据集**：ScanQA、SQA3D、Scan2Cap（通用 3D 任务）及 VSI-Bench（包含 8 个子任务的复杂空间-时序推理基准）。
- **基线模型**：VisionZip、FastVID、DTC、EgoPrune；基础 3D VLM：GS-Reasoner、VLM-3R。
- **通用 3D 任务**（Table 1）：在 20% 保留率下，CoverPrune 在 ScanQA Acc.% 达 100.95（超越 Vanilla 100%）、Scan2Cap Acc.% 达 101.68；在 10% 保留率下 ScanQA Acc.% 保持 100.00（即零损失），显著优于 DTC 的 90.81 和 EgoPrune 的 80.66。
- **VSI-Bench 推理任务**（Table 2, GS-Reasoner）：20% 保留率下平均成绩 59.76（为 Full Token 的 92.4%），10% 保留率下 56.83，5% 保留率下 52.66，均大幅领先最强基线（DTC 在 10% 时为 51.66，EgoPrune 仅 44.71）。
- **VLM-3R 上验证**（Table 3）：10% 保留率下平均成绩 54.69（CoverPrune）和 54.74（CoverPrune-Lite），同样全面超越各基线。
- **消融**（Table 4）：移除特征代价导致最大下降（-3.58 分），几何与时间代价分别造成 -0.74、-0.80 下降，验证三要素均有贡献；FST 容量机制带来约 0.18 的稳健提升。
- **效率**（Table 5）：CoverPrune 剪枝耗时 2.53s（vs. DTC 的 3.47s），相对精度 87.84%；CoverPrune-Lite 仅需 0.41s，相对精度 88.01%，为三者最高效率-精度平衡点。

## 相关工作脉络
- **DivPrune / FastVID（多样性剪枝）**：以特征空间分散度为目标的通用 VLM 剪枝方法，本文从"多样性→覆盖度"的根本范式层面与之区别，强调 3D 场景中原型 Token 的代表性价值被多样性指标系统性低估。
- **VisionZip / IVTP（注意力剪枝）**：基于自注意力或跨模态注意力质量筛选 Token，本文指出 attention sink 和 prompt-dependent saliency 使注意力分数在多域推理中不可靠，而 OT 覆盖框架不依赖注意力机制即可实现稳定的证据保留。
- **DTC（3D 体素级压缩）**：首个面向 3D QA 的专用剪枝方法，采用多样性驱动的体素接地策略；本文与之定位差异在于 DTC 仍基于冗余消除逻辑，而 CoverPrune 以覆盖最大化为核心目标，在细粒度空间关系任务（Relative Direction、Appearance Order）上优势显著。
- **EgoPrune（自我运动对齐剪枝）**：利用 SfM 位姿对齐重叠区域后再过滤；本文与之对比的定位差异在于 EgoPrune 侧重多视角去重，而 CoverPrune 关注保留集对全信息的覆盖完整性，对时序一致性和几何结构的建模更为全面。
- **SPOT / Partial Wasserstein Covering（OT 原型选择前作）**：Benamou 等（2015）和 Kawano 等（2022）将 OT 用于原型选择和覆盖，本文的创新在于将其移植到 3D VLM 推理时 Token 剪枝场景，设计了适配 3D 多模态特性的 FST 代价函数和高效的 SGS 贪心求解器。
- **Honeybee / LLaVA-UHD / Token-Packer（架构改造型压缩）**：需端到端微调项目器或修改模型结构；本文方法为 training-free、即插即用，无需修改任何模型参数或架构，可直接部署于多种加速框架。

## 局限性与未来方向
- **仅针对 3D VLM，未验证通用 VLM 适用性**：方法设计依赖 3D 坐标和时序信息，在纯 2D 图像或视频理解场景下 FST 代价中的空间项可能失效，作者明确提及未来将扩展至通用 VLM。
- **SGS 贪心算法非全局最优**：虽然半松弛 OT 保证了次模性下的常数因子近似，但贪心策略仍可能陷入次优解，尤其在高压缩率下邻域限制可能遗漏远距离重要信息。
- **FST 坐标估计依赖外部几何模型**：Token 的 3D 坐标需通过 SfM 或几何基础模型估计（论文使用 GS-Reasoner 的内置估计器），在坐标估计不准确或不可用的场景下性能可能下降。
- **长视频场景下 FST 容量计算与排序开销**：CoverPrune-Lite 的 Morton 排序复杂度为 O(N log N)，CoverPrune 的迭代 SGS 对长序列仍存在潜在瓶颈，作者未讨论极端长度（如数分钟视频）下的可扩展性。
- **超参数 λ_f、λ_x、λ_τ 固定为 1**：论文实验中使用默认等权设置，未系统分析不同任务类型下各维度权重的敏感性。

## 研究启发与可借鉴点
- **覆盖度优先于多样性的剪枝理念**：从"去冗余"转向"保覆盖"的范式迁移对 Token 压缩、采样选择等问题具有通用启发价值，可迁移至其他需要保留代表性的模型压缩场景。
- **FST 多维代价设计的模块化思想**：特征-空间-时序的分离建模与加权组合方式可作为通用的多域 Token 相似性度量框架，适用于视频理解、点云-语言等多模态任务中的 Token 管理。
- **半松弛 OT 用于组合子集选择**：将 NP-hard 原型选择转化为具有次模性的半松弛 OT 问题并贪心求解，该数学工具对类似的选择-覆盖优化问题具有直接可复用价值。
- **Morton 码空间填充曲线用于 3D 有序分组**：CoverPrune-Lite 中以 3D 空间局部性指导分组和局部匹配的启发式策略，可推广至任何具有空间结构的 Token/数据压缩任务。
- **训练无关 + 即插即用的部署方式**：不修改模型参数即可插入任意 3D VLM 推理流程的设计模式，降低了方法落地门槛，值得在团队后续工作中作为高效推理加速模块直接集成。

## 关键术语表
**Optimal Transport（最优运输）**：通过最小化将源分布映射到目标分布的总代价来度量两组分布之间差异的数学框架，本文用以量化保留 Token 集对原始 Token 集的覆盖质量。
**Semi-Relaxed OT（半松弛最优运输）**：在标准 OT 基础上放松目标侧等式约束为不等式约束，使问题具有次模性，可支持高效的贪心子集选择。
**Feature-Spatial-Temporal (FST) Cost（特征-空间-时序代价）**：融合语义相似性、3D 空间距离和时序一致性的综合成对代价函数，是 CoverPrune 的核心多维度量设计。
**FST Capacity Vector（FST 容量向量）**：基于每个目标 Token 的局部 FST 独特性分数计算得到的非均匀目标容量分布，用于在极端剪枝下优先保留信息丰富的 Token。
**Spatial-Guided Greedy Selection (SGS)**：利用 3D 空间局部性先验将每步边际代价评估限制在局部邻域内的贪心 Token 选择算法，避免了全局 OT 的重算开销。
**Morton Code（莫顿码）**：一种空间填充曲线编码方式，将多维空间坐标映射为一维序列同时保留局部性，本文用于 3D Token 的高效有序分组。

## 可复现要素
- **代码**：开源，GitHub 地址 https://github.com/Brucess/CoverPrune
- **数据集**：ScanQA、SQA3D、Scan2Cap、VSI-Bench 均为公开数据集
- **关键超参**：$\lambda_f=\lambda_x=\lambda_\tau=1$，$\alpha_f=\alpha_x=\alpha_\tau=1$，32-frame 均匀采样；其余保持 GS-Reasoner/VLM-3R 默认配置
- **模型坐标来源**：使用 GS-Reasoner 内置几何估计器生成，未使用 Ground Truth 坐标
- **基线复现**：论文引用 VisionZip、FastVID、DTC、EgoPrune 的官方开源实现进行对比

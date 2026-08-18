---
title: "A-Neighborhood-Attention-Transformer-Network-for-Enhanced-3D"
source: https://arxiv.org/pdf/2608.12274v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:56:59"
field: "医学图像分割"
keywords: ["LAD artery segmentation", "Neighborhood Attention", "3D medical image segmentation", "low-contrast CT", "parameter-efficient fine-tuning", "LoRA", "uncertainty-weighted loss", "coronary artery"]
innovations: ["首个将 Neighborhood Attention 引入 3D LAD 非增强 CT 分割的框架", "LoRA 参数高效微调 + CTA 预训练的两阶段迁移学习策略", "同质不确定性加权的 Dice-Focal + Hausdorff 复合损失"]
benchmarks: ["LAD-SEG", "ImageCAS"]
---

# 论文速读：A Neighborhood Attention Transformer Network for Enhanced 3D Segmentation of the Left Anterior Descending Artery

## 一句话总结
论文提出 **NA-UNETR**，一种结合 Neighborhood Attention 与 Dilated NA 的 3D Transformer 分割框架，通过预训练+LoRA 参数高效微调及不确定性加权复合损失，显著提升了非增强 CT 中微小、低对比度左前降支（LAD）动脉的分割精度与边界一致性。

---

## 研究问题与动机
- **临床需求迫切**：LAD 动脉是胸部放疗（肺癌、食管癌、乳腺癌）心脏剂量 sparing 的关键子结构，低剂量辐射即可引发冠状动脉钙化及缺血事件，精准分割对优化放疗计划至关重要。
- **任务极端困难**：LAD 管径极小（平均前景体素比仅 1.7×10⁻⁵，平均每例约 540 个体素），在**非增强、自由呼吸、非心电门控 CT** 上软组织对比度低、边界模糊，且存在运动伪影；专家手动标注的 Dice 离散度高达 0.10–0.53。
- **现有方法性能不足**：传统 Atlas-based 方法 Dice 仅 0.09–0.27；CNN 基线（U-Net、nnU-Net）Dice 约 0.21，远低于心脏大腔室分割（>0.85）；Transformer 方法虽能建模长程依赖，但**从未被评估于 LAD 非增强 CT 分割任务**。
- **小数据瓶颈突出**：标注数据稀缺（仅 20 例机构数据），直接从头训练 Transformer 极易过拟合，亟需有效的迁移学习策略。

---

## 核心贡献（创新点）
1. **首个将 Neighborhood Attention 引入 3D 医学图像冠状动脉分割的框架**：NA-UNETR 在 UNETR 风格 backbone 中嵌入 NA/DiNA 块，兼顾局部血管细节与全局上下文，本质区别在于避免全局自注意力对弱信号背景噪声的混合稀释。
2. **Dilated NA (DiNA) 模块实现感受野渐进扩展**：通过扩张采样在保持局部约束的同时逐层扩大感受野，使模型能够整合更长血管段，而非依赖全图全局交互。
3. **两阶段迁移学习策略（CTA 预训练 + LoRA 微调）**：先在 1,000 例 CTA 数据集 ImageCAS 上预训练获取通用冠状动脉解剖表征，再用 LoRA（rank=8）仅微调 decoder 和 MLP 适配层，显著缓解非增强 CT 标注稀缺问题。
4. **同质不确定性加权的复合 Dice-Focal + Hausdorff 损失**：引入可学习方差参数动态平衡区域重叠与边界精度两个目标，同时向 Hausdorff 损失添加高斯噪声抑制早期过拟合，无需人工调参。

---

## 方法详解
### 架构设计
- **Encoder**：输入 3D CT 体素 $X \in \mathbb{R}^{1 \times H \times W \times D}$，嵌入维度 $d=48$。重叠 tokenizer 由两个 $3×3×3$ 卷积（步长 2 和 1）构成，嵌入深度wise 卷积以增强局部归纳偏置。
- **NAT 块配置**：四个 encoder 阶段 + 一个 bottleneck 阶段，NAT 块数 $(3, 4, 6, 18, 5)$，kernel size $(7, 7, 7, 3, 3)$。每块前设 Res-Conv 层稳定梯度。NA 与 DiNA 交替堆叠，DiNA 引入可学习相对位置偏置 $b(i,j)$ 与扩张采样 $\mathcal{N}_\delta(i)$。
- **Decoder**：对称 U 形结构，残差块（两层 $3×3×3$ 卷积 + Instance Normalization）精炼 skip connection 特征，deconvolution 上采样后 concat 融合，最终经 $1×1×1$ 卷积 + sigmoid 输出概率图。

### 数据预处理与增强
- **类不平衡处理**：动脉中心采样——每轮生成 4 个 $(96, 96, 96)$ patch，正负样本 1:1 随机裁剪。
- **对比度增强**：强度裁剪至 $[-200, 400]$ HU，线性缩放至 $[0,1]$；Gamma 调整（$\gamma \in [1.6, 1.8]$，概率 0.8）；Savitzky–Golay 滤波（窗口 5，阶数 2）。
- **几何/强度增强**：随机旋转（$\pm 6°$）、缩放（$\pm 5\%$）、高斯锐化、低幅偏置场模拟scanner不一致性。

### 损失函数
- **Dice-Focal Loss**：
  $$\mathcal{L}_{\text{Dice-Focal}} = \lambda_1 \mathcal{L}_{\text{Dice}} + \lambda_2 \mathcal{L}_{\text{Focal}}$$
  其中 $\alpha=0.8, \gamma=2$，前景/背景权重 0.9/0.1。
- **Hausdorff Loss**（可微近似）：
  $$\mathcal{L}_{\text{Hausdorff}} = \frac{1}{|\partial\hat{Y}|}\sum_{x\in\partial\hat{Y}}\min_{y\in\partial Y}d(x,y)^2 + \frac{1}{|\partial Y|}\sum_{y\in\partial Y}\min_{x\in\partial\hat{Y}}d(x,y)^2$$
- **不确定性加权总损失**：
  $$\mathcal{L}_{\text{total}} = \frac{1}{2\sigma_1^2}\mathcal{L}_{\text{Dice-Focal}} + \frac{1}{2\sigma_2^2}\tilde{\mathcal{L}}_{\text{Hausdorff}} + \log\sigma_1 + \log\sigma_2$$
  $\tilde{\mathcal{L}}_{\text{Hausdorff}}$ 加入零均值高斯噪声 $\epsilon\sim\mathcal{N}(0,\sigma_2^2)$ 以减缓收敛、促进边界平滑。

### 迁移学习策略
- **预训练**：ImageCAS 1,000 例 CTA，100 epochs，验证集 5%。
- **微调**：LAD-SEG 20 例，5-fold 交叉验证，200 epochs。冻结 encoder 注意力层，MLP 层替换为 LoRA 模块 $W \to W + AB$（$A\in\mathbb{R}^{d×r}, B\in\mathbb{R}^{r×d}, r=8$），仅更新 $\theta_{\text{dec}}, A, B$。

### 后处理
最大连通分量保留 + 移除 <64 体素的小连通分量 + 填充残余空洞。

---

## 实验与结果
### 数据集
- **LAD-SEG**：20 例自由呼吸非增强 CT（分辨率 1.17×1.17×3.0 mm）， Physician 标注 LAD 轮廓。
- **ImageCAS**：1,000 例 CTA（0.29–0.43 mm 平面分辨率），公开冠状动脉分割基准。

### LAD-SEG 结果（主要挑战场景）
| 方法 | DSC (%) ↑ | clDice (%) ↑ | HD95 (mm) ↓ | ASD (mm) ↓ |
|---|---|---|---|---|
| nnU-Net | 42.54 ± 2.90 | 40.91 ± 10.49 | 39.68 ± 5.92 | 10.37 ± 1.56 |
| Swin UNETR | 44.78 ± 6.08 | 43.76 ± 9.33 | 41.12 ± 5.98 | 10.79 ± 2.45 |
| **NA-UNETR (ours)** | **45.64 ± 4.86** | **44.39 ± 8.38** | **38.16 ± 4.37** | **10.01 ± 1.39** |

- 相对 nnU-Net Dice **+3.10 pp**；相对 Swin UNETR HD95 **降低 2.96 mm**；ASD 最优。
- Mann-Whitney U 检验 $p>0.05$（样本量小 n=20，预期结果）。

### ImageCAS 结果（公开基准）
| 方法 | DSC (%) ↑ | clDice (%) ↑ | HD95 (mm) ↓ | ASD (mm) ↓ |
|---|---|---|---|---|
| UNet++ | 78.57 ± 0.51 | 84.71 ± 0.69 | 9.46 ± 0.84 | 1.15 ± 0.07 |
| Swin UNETR-V2 | 78.03 ± 0.48 | 85.61 ± 0.67 | 9.13 ± 0.73 | 1.12 ± 0.06 |
| **NA-UNETR (ours)** | **79.49 ± 0.25** | **86.88 ± 0.32** | **8.89 ± 0.30** | **1.02 ± 0.03** |

- 超越所有基线，DSC 较 UNet++ 高 1.2 pp，HD95 较 Swin UNETR 低约 4 mm，ASD 降低约 9%；Mann-Whitney U 检验 **$p<0.05$ 显著**。

### 计算效率
- 参数量 19.6M，FLOPs 314.1B，推理 1.33 s/例，峰值显存 4.17 GB。
- 与 Swin UNETR（19.7M, 300B FLOPs）相当，显著优于 UNETR（480.9B FLOPs）。

### 消融关键结论
- **DiNA vs NA only**：HD95 降低约 2–3 mm，证明扩张采样有助于长程血管轨迹整合。
- **LoRA rank**：$r=8$ 最优；$r=2/4$ 容量不足，$r=16$ 收益递减。
- **预训练必要性**：仅用 LAD-SEG 微调 Dice 从 45.64% 降至 36.39%。
- **损失设计**：去掉 Focal 或边界损失均导致性能下降；静态权重不如不确定性加权。
- **预处理管道**：去除增强/采样策略使 nnU-Net Dice 从 42.54% 降至 39.98%。

---

## 相关工作脉络
1. **Atlas-based LAD 分割**（van den Bogaard 等, 2019）：Dice 0.09–0.27，依赖刚性/弹性配准，无法捕捉个体解剖变异，本文通过端到端学习超越。
2. **CNN 基线（U-Net / nnU-Net / MedNeXt）**：在 LAD 非增强 CT 上 Dice 仅 ~0.21–0.43，受限于局部感受野难以维持血管长程连续性；本文证明 Transformer 架构在此任务上的优势。
3. **冠状动脉 CTA 分割**（Zeng 等, ImageCAS 2023）：此前工作在增强 CT 上取得 Dice 70%+，但非增强 CT 任务仍空白；本文填补该空白并建立新 SOTA。
4. **UNETR / Swin UNETR / nnFormer**：3D 医学分割中广泛使用，但**均未在 LAD 非增强 CT 上评估过**；本文首次将 Transformer 引入此极端小结构分割场景。
5. **Neighborhood Attention Transformer**（Hassani 等, CVPR 2023）：原用于自然图像分割；本文首次将其适配到 3D 医学体素级分割，并引入 DiNA 和 LoRA 微调策略。
6. **多任务不确定性加权损失**（Kendall 等, CVPR 2018）：本文将其与 Hausdorff 边界损失结合，引入噪声扰动防止边界过拟合，较原方法更适配血管细分割。

---

## 局限性与未来方向
- **数据规模有限**：LAD-SEG 仅 20 例，统计检验功效不足，需多中心扩大样本验证泛化性。
- **图像可见性硬限制**：非增强 CT 上 LAD 对比度极低，即使最优模型也无法超越解剖信息本身提供的上限；未来需探索多模态融合（如 MRI 配准）或生成增强技术。
- **域差距未完全消除**：CTA→非增强 CT 存在强度分布差异，LoRA 微调虽部分缓解但未做显式域对齐，未来可引入域适应或对比学习策略。
- **临床工作流验证缺失**：目前仅评估自动分割指标，未测量人工修正耗时、放射物理师接受度等实际临床效用指标。
- **未探索 3D 后处理优化**：当前后处理仅为连通分量过滤，可结合形态学先验或拓扑约束进一步提升薄血管连续性。

---

## 研究启发与可借鉴点
1. **LoRA 用于 3D 医学 Transformer 微调的范式**：冻结 encoder 注意力层、仅微调 decoder + LoRA 适配器，在仅 20 例数据下实现稳定收敛，可作为小数据 Transformer 微调的通用模板。
2. **同质不确定性 + 噪声扰动的复合损失设计**：向 Hausdorff Loss 注入高斯噪声延缓收敛、促进边界平滑，这一技巧可迁移至任何需精细边界对齐的血管/薄结构分割任务。
3. **动脉中心采样（artery-centric sampling）策略**：1:1 正负 patch 采样配合 Gamma 增强，对极端类不平衡（<0.01% 前景比）场景极具参考价值，可复用于神经、微血管等细小结构分割。
4. **NA/DiNA 交替堆叠的可扩展 backbone**：核尺寸随阶段递减（7→3）适应不同感受野需求，该配置策略可直接移植到其他 3D 器官/病变分割任务。
5. **端到端预训练→微调两阶段管线**：ImageCAS 预训练权重公开，鼓励社区将此类"大域预训练+小域微调"范式推广至其他稀缺解剖结构分割。

---

## 关键术语表
- **Neighborhood Attention (NA)**：将自注意力限制在 $k×k×k$ 局部窗口内的计算高效注意力机制，避免全局交互稀释弱信号。
- **Dilated NA (DiNA)**：在 NA 基础上引入扩张采样与可学习相对位置偏置，使感受野随网络深度渐进扩大。
- **LoRA (Low-Rank Adaptation)**：通过低秩矩阵分解 $W \to W + AB$ 实现参数高效微调，仅训练少量新增参数而冻结原权重。
- **Homoscedastic Uncertainty Weighting**：引入可学习方差参数 $\sigma^2$ 自动平衡多损失项权重，替代手工调参。
- **clDice (centerline Dice)**：基于预测与 GT 骨架交集的拓扑度量，专评管状结构（如血管）的路径连续性。
- **HD95 (95th-percentile Hausdorff Distance)**：预测边界与 GT 边界间距离的 95 分位数，对极端偏移鲁棒，评估边界对齐精度。
- **ImageCAS**：1,000 例公开冠状动脉 CTA 分割基准数据集，含双放射科医生标注。
- **LAD-SEG**：20 例自由呼吸非增强 CT 的 LAD 动脉标注数据集（机构内部，IRB 批准）。

---

## 可复现要素
- **代码**：已开源于 https://github.com/rafiibnsultan/NA_UNETR
- **ImageCAS 数据集**：公开可用
- **LAD-SEG 数据集**：机构内部数据，未公开（需联系作者申请）
- **关键超参**：LoRA rank $r=8$；patch 大小 $(96, 96, 96)$；学习率 $1×10^{-4}$；weight decay $1×10^{-5}$；AdamW 优化器；Dice-Focal $\alpha=0.8, \gamma=2$；λ₁=λ₂=1；前景/背景权重 0.9/0.1
- **训练环境**：PyTorch 2.5.1，Python 3.9.21，单卡 NVIDIA A100 40GB
- **预训练权重**：ImageCAS 预训练权重将随代码一并公开

---

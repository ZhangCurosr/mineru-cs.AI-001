---
title: "Quantum Incremental Learning with Mixed State Prototypes"
source: https://arxiv.org/pdf/2608.10464v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:24:39"
field: "量子持续/增量学习"
keywords: ["quantum neural network", "incremental learning", "quantum metric learning", "mixed states", "Hilbert-Schmidt distance", "CCPS", "continual learning"]
innovations: ["以CCPS混合态原型替代单一纯态/基态概率输出，突破正交基态数量对类别容量的限制", "将HS距离分解为经典权重碰撞项与若干SWAP test重叠项，适配NISQ低开销度量", "用辅助MLP解耦骨干表征学习与原型分类，增量阶段不扰动量子全局态"]
benchmarks: ["CIFAR-100 (Split A/B/C)", "TinyImageNet"]
---

# 论文速读：Quantum Incremental Learning with Mixed State Prototypes

## 一句话总结
本文提出一种基于可训练混合态原型的量子增量学习框架，在固定的浅层量子电路中，通过向共享量子骨干追加类混合态原型而非扩展电路宽度来连续适应新类别；混合态原型利用CCPS ansatz以PCA式去噪能力编码类内分布，结合HS距离与SWAP test实现低开销、抗遗忘的分类。

## 研究问题与动机
- 真实数据连续到达，经典增量学习在重训成本与表征漂移上存在瓶颈，难以在资源受限场景下扩展。
- 传统量子分类器依赖固定正交基态概率或期望值输出，类别增多往往需要扩展测量/读出结构或更多qubit，而追加qubit会改变全局Hilbert空间与纠缠结构，加剧灾难性遗忘并可能触发barren plateau。
- 单一纯态原型只能编码单一样本流形，缺乏刻画类内多样分布的统计容量。
- NISQ设备电路宽度受限，需要在有限qubit下实现可扩展、低遗忘的持续分类。

## 核心贡献（创新点）
- **可扩展的混合态原型量子分类框架**：以距离匹配替代显式参数化分类头，避免正交基态数量对类别容量的硬约束。与“扩展电路/观测器”路线本质不同，本工作保持共享骨干不变，仅追加轻量原型模块。
- **CCPS混合态原型表示**：用多个变换后计算基态的凸组合表示单个类别，提供比纯态更强的类内分布刻画能力，并具备PCA式低秩去噪属性。与Fidelity Classifier等纯态原型方法的本质区别在于其多态叠加与可控秩。
- **HS距离的可分解计算**：将密度矩阵距离分解为纯经典权重碰撞项与若干SWAP test重叠项，计算复杂度显著低于trace/Bures等需要谱分解的距离。与通用量子度量学习相比，本工作专门面向原型拟合与在线比较做工程化分解。
- **辅助MLP解耦的特征学习**：在骨干训练阶段用轻量MLP头监督量子密度矩阵空间的可分性，训练后丢弃，再由原型匹配接管推理，避免在增量阶段直接扰动量子态。与直接末端测量分类的本质区别在于特征学习与分类头解耦。
- **原型引导的示例重放机制**：以HS距离选取最贴近各类原型的样本维持记忆缓冲区，在有限存储下持续约束旧类表征漂移。与iCaRL等经验采样相比，重放依据是量子空间的原型相似度而非经典距离。

## 方法详解
- **预处理与编码**：输入经ResNet-18提取512维特征，经Dropout+ReLU+线性投影得到256维向量，L2归一化后通过振幅编码注入n=8 qubit初始纯态。
- **QCNN骨干**：金字塔式交替堆叠量子卷积与量子池化；卷积层用两量子比特SU(4)块作用于相邻qubit对捕获局部与非局域相关；池化层经参数化受控旋转后将信息汇聚并部分求迹丢弃部分qubit，输出保留4 qubit的16×16密度矩阵$\rho(x)$。
- **CCPS混合态原型**：每个类别$c$维护独立VQC $U_c$（三层、交替环拓扑，单层4个SU(4)块），变换前$K$个计算基态得$|\psi_{c,i}\rangle=U_c|i\rangle$；混合态原型为$\rho_c=\sum_{i=0}^{K-1}w_{c,i}|\psi_{c,i}\rangle\langle\psi_{c,i}|$，权重经softmax参数化避免单纯形约束带来的梯度不稳定。
- **HS距离与SWAP test**：平方HS距离$\|\rho-\sigma\|_F^2=\mathrm{Tr}[\rho^2]+\mathrm{Tr}[\sigma^2]-2\mathrm{Tr}[\rho\sigma]$。CCPS下原型纯度$\mathrm{Tr}[\rho_c^2]=\sum_i w_{c,i}^2$可经典预计算；交叉项由若干SWAP test估计$\mathcal{F}_{c,i}(x)=\mathrm{Tr}[\rho(x)|\psi_{c,i}\rangle\langle\psi_{c,i}|]=2P(0)-1$，整体代价远低于全矩阵运算。
- **两阶段训练**：
  - Stage 1：冻结预处理，联合优化骨干与二维MLP辅助头（隐藏32维），以交叉熵$\mathcal{L}_{rep}$在可见类上训练，使$\rho(x)$可分；头训练完成后丢弃。
  - Stage 2：冻结骨干，对每类独立最小化$\mathcal{L}^c_{proto}=\sum_i w_{c,i}^2-\frac{2}{N}\sum_n\sum_i w_{c,i}\mathcal{F}_{c,i}^{(n)}$拟合原型；等价于对类均值密度矩阵做变分低秩近似（QMSC驱动的量子PCA）。
- **原型引导重放**：增量阶段以$\mathcal{D}_c(x)=\|\rho(x)-\rho_c\|_F^2$评估样本与各类原型的距离，各分类选距离最小者入缓冲，预算均匀分配。
- **推理**：$\hat{y}=\arg\min_c\mathcal{D}_c(x)$，实际以$\mathrm{logit}_c(x)=\mathcal{O}_c(x)-\tfrac{1}{2}\sum_i w_{c,i}^2$取最大，其中$\mathcal{O}_c(x)=\sum_i w_{c,i}\mathcal{F}_{c,i}(x)$。

## 实验与结果
- **数据集与划分**：CIFAR-100（取96类分Split A/B/C）与TinyImageNet；每数据集固定选取32类，16初类+4组×4增量类，共5个增量阶段。
- **配置要点**：8 qubit骨干、4 qubit原型；固定$K=7$；记忆缓冲区640样本（32类后约每类20）；骨干SGD（momentum 0.9、wd 1e-4），原型Adam；初任务240 epoch/batch 128/lr 0.01，原型拟合60 epoch/batch 16/lr 0.04，增量微调36 epoch/lr 0.01；4 seeds均值±标准差。
- **非增量分类对比**：在Split A/B/C/TinyImageNet上分别达0.8323/0.7766/0.8222/0.6997，均值0.7827，整体优于所列经典与量子基线；相较直接用测量输出的QCNN提升约12.88–15.83个百分点，说明基态直接测量不足以应对多类不均分布。
- **增量学习对比（固定$K=7$）**：Split A Last=0.5728/Avg=0.7178；Split B Last=0.4947/Avg=0.6465；Split C Last=0.5481/Avg=0.7097；TinyImageNet Last=0.3838/Avg=0.5371。该设置在相同内存预算（640）下优于BiC、Eucl-NCM、NCM、MUC-LwF、SSRE、DeeSIL、TOPIC+、SDC与LUCIR等多类方法；曲线显示精度逐步下降而非断崖式崩塌，体现一定抗遗忘能力。
- **最强结果与提升**：非增量任务整体最优（均值0.7827）；增量任务在固定$K=7$下于Split A取得最高Avg Acc 0.7178，较多种prototype/statistics/regularization基线更优，并在更少内存预算（640 vs. 2000）条件下具可比性。

## 相关工作脉络
- **iCaRL/FOSTER/PODNet等replay基线**：以经验采样或特征蒸馏保留旧知识；本文以量子HS距离指导重放，并在同一密度矩阵空间进行原型匹配，避免直接扩展参数化分类头。
- **FeTrIL/PASS/NCM/Eucl-NCM/TOPIC+等原型方法**：在经典嵌入空间做最近原型决策；本文在量子Hilbert空间中以混合态原型建模类内分布，天然兼容NISQ有限宽度电路。
- **FeCAM/SDC/DeeSIL等统计方法**：利用类分布统计校准；本文用密度矩阵的散度和加权SWAP重叠替代统计近似，适合含噪声的量子表征。
- **ABD/MUC-LwF/LwF-MC/EWC等正则化方法**：约束关键权重；本文不依赖权重惩罚，而是通过辅助MLP解耦与原型拟合实现稳定性。
- **QCNN/HEA-VQC/TTN-QNN等纯量子分类器**：多以基态概率直接输出logits；本文用QCNN做特征浓缩并输出混合态，分类由距离匹配完成，突破基态维数对类别数的限制。
- **Fidelity Classifier**：以单一可训练纯态作原型；本文以CCPS多态混合+可控秩提升表达能力与鲁棒性。

## 局限性与未来方向
- 实验均为经典仿真，未在实际NISQ硬件上验证噪声与读取误差的影响。
- 固定$K=7$为跨数据集统一设置，非逐数据集调优；敏感度分析表明不同split最优K在7–15之间波动，自动化秩选择仍有空间。
- 单类原型电路约180参数，类别增多时参数总量线性增长，长期可扩展性需进一步压缩或共享化。
- 仅验证类增量设定，未覆盖领域增量、流式样本级增量或跨模态场景。
- 记忆预算640相对较小，更大规模数据下的遗忘–准确率权衡尚待系统探索。

## 研究启发与可借鉴点
- **辅助头解耦训练**：用轻量MLP把“量子表征可分性”与“最终分类”解耦，可在更多VQC/QCNN任务中复用，避免直接扩大读出层导致的训练不稳定。
- **CCPS+HS距离的工程分解**：将密度矩阵距离拆解为经典权重项与少量SWAP test，是面向NISQ硬件的实用度量设计范式，可迁移到量子聚类、异常检测等任务。
- **低秩混合态原型的思想**：以可控秩进行“量子PCA式”去噪，提示我们在新类到来时可通过秩调度自适应压缩噪声成分，抑制灾难性漂移。
- **原型引导重放**：将示例选择从经典距离改为量子HS距离，能更好贴合模型的实际度量空间，值得在量子 continual learning 的其他变体中复现。
- **固定秩对比协议**：跨数据集使用同一$K$做主要结果、另做敏感性 sweep，为公平比较提供可复用的评测规范。

## 关键术语表
- **Incremental/Continual Learning**：模型按序学习新类别或任务而尽量不遗忘旧知识的持续学习范式。
- **Catastrophic Forgetting**：网络在新任务上更新后，原有表征被覆盖导致旧任务性能骤降的现象。
- **QCNN (Quantum Convolutional Neural Network)**：将经典卷积/池化思想量子化，通过参数化酉变换与部分求迹实现特征浓缩的量子网络。
- **CCPS (Convex Combination of Pure States)**：将低秩混合态表示为若干变换后正交纯态的概率混合，无需辅助qubit即可完成状态编译。
- **Density Matrix / Mixed State**：描述开放或约化量子系统的Hermitian半正定单位迹算子，能刻画概率混合与相干性损失。
- **Hilbert–Schmidt (HS) Distance**：密度矩阵间的Frobenius范数距离，可分解为纯度与交叉项，适合SWAP-based测量。
- **SWAP Test**：用一个辅助qubit与受控SWAP门估计两态重叠的测量电路，输出$P(0)=\tfrac{1}{2}(1+\mathrm{Tr}[\rho\sigma])$。
- **Prototype-guided Replay**：依据样本与原型的量子距离挑选代表性旧样本进入缓冲，以距离语义对齐的方式进行经验重放。

## 可复现要素
- 数据集：CIFAR-100、TinyImageNet（公开）；固定类子集与顺序见代码。
- 代码：已开源，https://anonymous.4open.science/r/QPIL-3E43。
- 框架与环境：PyTorch + PennyLane；NVIDIA GeForce RTX 3090 (24GB)。
- 关键超参：骨干8 qubit、原型4 qubit、原型秩$K=7$、缓冲区640样本（均摊到可见类）、骨干lr 0.01/batch 128、原型lr 0.04/batch 16、初任务240 epoch、原型拟合60 epoch、增量微调36 epoch；优化器骨干/分类SGD（momentum 0.9、wd 1e-4），原型Adam；4 seeds统计。

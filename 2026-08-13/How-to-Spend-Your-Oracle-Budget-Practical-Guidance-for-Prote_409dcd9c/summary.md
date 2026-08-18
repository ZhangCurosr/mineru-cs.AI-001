---
title: "How-to-Spend-Your-Oracle-Budget-Practical-Guidance-for-Prote"
source: https://arxiv.org/pdf/2608.12192v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:26:31"
---

# 论文速读：How-to-Spend-Your-Oracle-Budget-Practical-Guidance-for-Prote

## 一句话总结
本文首次系统对比了O3、FK-steering、DPO与Best K-of-N四种引导策略在蛋白质结构预测模型上的Oracle预算分配效果。研究表明，在低-中等预算下O3显著优于所有基线，而FK-steering与DPO需更大预算才能发挥优势，为实际生物设计管线提供了预算感知的实用选型指南。

## 研究问题与动机
- **核心问题**：AlphaFold3/Boltz-2等蛋白质结构预测基础模型在部分靶点上仍会出现构象坍缩、几何不合理或复合物组装错误，需借助外部Oracle（如结合能、稳定性评分）进行校准，但生物Oracle评估成本极高，Oracle预算成为硬约束。
- **现有方法割裂**：推理时引导（如FK-steering）与微调（如DPO）在学术上各自独立发展，缺乏在统一模型、统一预算设定下的横向对比，从业者难以依据可用计算/实验资源选择策略。
- **朴素采样失效**：Best K-of-N等无引导方法在固定K/N比下几乎不随预算提升，说明直接增加采样数无法解决基础模型的系统性失败。
- **研究目标**：建立首个面向Oracle预算意识的蛋白质结构预测引导基准，量化不同方法在不同预算区间的性能边界，输出可落地的方法选型建议。

## 核心贡献（创新点）
1. **首次将O3框架应用于蛋白质结构预测**。该工作将O3从通用生成模型扩展至Boltz-2，利用Knothe–Rosenblatt变换与LOL投影构建低维潜子空间并结合贝叶斯优化，与传统在高维坐标空间盲目搜索的方法本质不同，大幅降低Oracle查询代价。
2. **提供四种引导策略的系统预算基准**。文章在同一模型（Boltz-2）与两组蛋白靶点下公平对比O3、FK-steering、DPO与Best K-of-N，填补了“不同预算区间应选何种引导方法”的实证空白，揭示了方法间的性能trade-off曲线。
3. **验证在线DPO在扩散模型微调中的有效性**。提出基于动态重采样与参考模型重置的在线DPO变体，证明其对预算的敏感性远低于离线DPO，打破了“大规模偏好微调必耗巨量Oracle预算”的固有认知边界。
4. **跨Oracle类型的稳健性评估**。不仅在参考依赖的TM-score下验证，还引入无参考的物理合理性MolProbity Oracle，证明方法优势在不同信号动力学特性（平滑全局 vs 局部几何敏感）下具有可迁移性。

## 方法详解
- **O3 (Bayesian Optimisation via O3)**：给定总预算$N$，先用$M$次查询采样并选取得分最高的$d$个潜码作为种子$Z=[z_1,\ldots,z_d]^\top$。通过Knothe–Rosenblatt变换$\phi: \mathcal{U} \to \mathbb{S}_+^{d-1}$将低维参数$u$映射为单纯形权重$w$，再经LOL投影$\ell(w,Z)=w^\top Z$得到潜码$z=g^{-1}(x)$，最终由冻结生成器$g$解码。优化在$d-1$维子空间$\mathcal{U}\subset[0,1]^{d-1}$中进行，以$d+2$个点拟合高斯过程（RBF核、常均值、单任务），后续$N-M-2$次预算按Log Expected Improvement采集函数迭代 querying，子空间一旦构建不再重建。
- **FK-Steering (Feynman-Kac steering)**：推理时生成$M$个交互粒子，在去噪轨迹的中间步骤按Oracle奖励重采样。粒子权重为$w(x_t^i) = \frac{g(x_t^i|x_{t_{prev}}^i)}{g_{proposal}(x_t^i|x_{t_{prev}}^i)} \lambda \exp(r(x_t^i)-r(x_{t_{prev}}^i))$，其中$\lambda$控制奖励信号强度。因生物Oracle通常不可微，沿用Boltz-2原始转移核；奖励仅在重采样步计算，本文采用奖励差值策略。每个粒子预算$N/M$，首尾各调用1次，其余$N/M-2$次均匀分布在去噪前3/4阶段（避开Boltz-2尾部注入噪声为零的静默期）。
- **DPO (Direct Preference Optimization)**：微调更新模型参数$\theta$，最大化偏好对$(x_w, x_l)$之间的隐式奖励边际。损失函数为$\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\left[\log\sigma\left(\beta\left(\log\frac{p_\theta(x_w)}{p_{\mathrm{ref}}(x_w)} - \log\frac{p_\theta(x_l)}{p_{\mathrm{ref}}(x_l)}\right)\right)\right]$，$p_{\mathrm{ref}}$初始化为预训练Boltz-2，$\beta$控制KL正则强度。因扩散模型对数似然计算昂贵，采用DSM训练损失差近似对数比：$\log(p_\theta(x)/p_{\mathrm{ref}}(x)) \approx \mathcal{L}_{\mathrm{DSM}}^{\mathrm{ref}}(x) - \mathcal{L}_{\mathrm{DSM}}^\theta(x)$。分为**离线DPO**（一次性采样$N$个结构配对后训练至收敛）与**在线DPO**（每epoch重新采样$N/E$个、实时构建偏好对、重置$p_{\mathrm{ref}}$以软化KL约束）。
- **Best K-of-N**：从冻结模型中独立抽取$N$个样本，直接返回Oracle得分最高的$K$个，无任何优化或参数更新

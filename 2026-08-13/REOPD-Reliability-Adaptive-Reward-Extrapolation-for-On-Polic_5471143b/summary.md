---
title: "REOPD-Reliability-Adaptive-Reward-Extrapolation-for-On-Polic"
source: https://arxiv.org/pdf/2608.11698v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:49:58"
---

# 论文速读：REOPD: Reliability-Adaptive Reward Extrapolation for On-Policy Distillation

## 一句话总结
论文针对 On-Policy Distillation (OPD) 中固定全局奖励外推系数（λ）易放大教师-参考 log-ratio 极端峰值、引发 reward hacking 且需昂贵域特定扫描的问题，提出 REOPD 框架。该方法通过 Token 级兼容性权重与 Micro-batch 级自适应预算的双粒度控制，在不引入任何额外验证器或 Rollout 的前提下，实现稳定且高效的超越教师知识蒸馏。

## 研究问题与动机
- **固定系数的 indiscriminate amplification：** 现有 ExOPD 使用单一全局标量 λ 对所有 Token 施加相同的外推强度，容易过度放大教师-参考 log-ratio 中的极端峰值，导致策略更新不稳定与 reward hacking。
- **域间最优 λ 差异显著：** 数学、代码等不同任务域的最优外推强度差异大，固定系数方法需针对每个新设置进行多轮全量训练与扫描，成本高且未必能收敛到合适值。
- **密集监督信号的异质性：** OPD 的 Token 级监督效用不仅取决于教师质量，还高度依赖学生与教师在当前位置的本地兼容性；统一缩放无法兼顾对齐信号保留与残差外推的平衡。
- **缺乏在线自适应调控机制：** 现有自适应蒸馏方法多依赖外部 verifier 反馈或调整 rollout 过程，缺少仅利用白盒 OPD 已有 log-prob 信号即可实现的轻量级残差控制机制。

## 核心贡献（创新点）
1. **形式化残差控制问题并提出双粒度自适应外推框架：** 将固定系数外推的缺陷归结为残差控制缺失，设计无需额外 verifier/reward model/value model 的在线控制管线，与已有工作的本质区别在于仅调控“超越教师”的残差项而严格保留标准对齐项。
2. **Token 级兼容性权重 $q_{b,i,t}$：** 基于学生-教师 log-prob 差异构造低方差代理并指数衰减，使高不确定性/高分歧位置的残差被抑制，低方差位置保留完整外推信号；区别于传统 token masking 方法，该权重直接作用于奖励外推而非对齐损失。
3. **有界 Micro-batch 自适应预算 $\gamma_b$：** 通过兼容性加权残差比例 $\rho_b$ 与可靠残差规模 $s_b$ 的比值驱动预算生成，结合 EMA 平滑与显式上界 clipping，实现批次级的鲁棒强度调度；本质区别在于将外推强度从全局固定值转化为数据依赖的时变上界。
4. **系统化实验与消融验证：** 在单教师数学、单教师代码及多教师混合域三个设置中全面评估，证明 REOPD 无需逐域 λ 扫描即可达到或超越最佳固定系数 ExOPD，并通过控制器轨迹分析揭示训练动态演化规律。

## 方法详解
- **问题设定与成本重构：** 定义学生-教师对齐成本 $a_t = \log \pi_\theta(y_t|s_t) - \log \pi_T(y_t|s_t)$，教师-参考隐式奖励 $r_t = \log \pi_T(y_t|s_t) - \log \pi_{\text{ref}}(y_t|s_t)$。REOPD 保留 $a_t$，将 ExOPD 的 $(\lambda-1)r_t$ 替换为 $\gamma_b q_{b,i,t} r_{b,i,t}$，得到新成本 $C_{b,i,t}^{\text{REOPD}} = a_{b,i,t} - \gamma_b q_{b,i,t} r_{b,i,t}$，对应有效系数 $\lambda_{b,i,t} = 1 + \gamma_b q_{b,i,t} \in [1, 1+\gamma_{\max}]$。
- **Token 级兼容性权重：** 构造差值 $x = \log \pi_T - \log \pi_\theta$，低方差代理 $\hat{\delta} = \exp(x) - x - 1$，权重 $q = \exp(-\hat{\delta}/\tau)$。$\hat{\delta} \ge 0$ 保证 $q \in (0,1]$；$\tau$ 控制衰减速率。该权重与后续统计量均 detach 出计算图，仅作控制信号。
- **Micro-batch 可靠残差统计：** 对同步微批次 $b$ 内合法 token 聚合两个统计量：兼容性加权残差比例 $\rho_b = \frac{\sum m|r|q}{\sum m|r|+\epsilon}$ 衡量保留残差占比；可靠残差规模 $s_b = \sqrt{\frac{\sum m(qr)^2}{\sum m+\epsilon}}$ 衡量绝对 RMS 规模。二者经 EMA（$\beta=0.95$）平滑得 $\bar{\rho}_b, \bar{s}_b$，跨数据并行 rank all-reduce 后共享。
- **有界自适应预算：** 目标预算 $\tilde{\gamma}_b = \text{clip}\left(\frac{B_0 \bar{\rho}_b}{\bar{s}_b + \epsilon}, 0, \gamma_{\max}\right)$，二次 EMA 平滑得 $\gamma_b$。$B_0$ 支持手动指定或前 $K_0$ 步基于对齐项 RMS 自动校准。预算上限 $\gamma_{\max}$ 防止外推失控。
- **优化过程：** 以 $A^{\text{REOPD}} = -C^{\text{REOPD}}$ 作为 PPO advantage，接入标准 OPD 的 PPO 策略代理目标。所有控制路径（$q, \rho, s, \gamma$）均 stop-gradient，不改变底层优化器与 rollout 流程。

## 实验与结果
- **数据集与模型：** 学生、参考策略及数学/代码教师均基于 Qwen3-4B（非 thinking RL 训练版）。数学训练集 DeepMath-103K level-6（57,046 examples）；代码训练集 Eurus code split（25,276 examples）；多教师为两者混合。评测基准：AIME 2024/2025、HMMT 2025 Feb/Nov、HumanEval+、MBPP+、LiveCodeBench v6 test6。
- **基线：** Standard OPD（$\lambda=1$）、固定系数 ExOPD（$\lambda=1.25$）。
- **单教师结果：** 数学 REOPD 达 **47.66%**，优于 OPD（46.28%）与 ExOPD（47.47%）；代码 REOPD 达 **63.45%**，优于 ExOPD（61.72%）与 OPD（62.55%），与 G-OPD 相当。
- **多教师结果：** 共享单一控制器下，数学 47.01%、代码 63.32%，均超越双方基线，验证跨域泛化能力

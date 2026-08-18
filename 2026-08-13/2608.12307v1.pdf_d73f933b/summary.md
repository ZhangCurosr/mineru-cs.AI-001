---
title: "AI4AI at Test-Time: Strong-to-Weak Capability Transfer via Harnesses"
source: https://arxiv.org/pdf/2608.12307v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:50:25"
---

# 论文速读：AI4AI at Test-Time: Strong-to-Weak Capability Transfer via Harnesses

## 一句话总结
本文首次系统验证了“测试时强到弱推理期能力迁移”（strong-to-weak scaffolding）范式：无需更新目标模型参数，仅由一个更强的大模型基于 5% 验证集迭代设计外部推理调度框架（harness），即可将弱目标模型在 Theory-of-Mind 基准上的平均准确率从 0.488 提升至 0.912。

## 研究问题与动机
- 现有知识蒸馏与对齐方法均以更新弱模型参数为前提，但在实际部署中常受限于成本、隐私或许可约束，难以进行微调。
- 小模型失败往往不仅源于内部能力不足，更来自任务呈现方式带来的过高认知负荷；通过外部推理期结构卸载不稳定推理可能是一种更轻量的能力补偿路径。
- 当前 agent harness 工程缺乏系统性归因与跨模型对照基准，无法回答“强模型构建推理环境”这一能力的上限、稳定性及核心机制。
- 如何在不改动目标模型权重的情况下，将强模型的认知结构显式编译为可复用的推理期调度规则，并为 agent 系统的低成本部署提供理论支撑。

## 核心贡献（创新点）
1. **形式化定义测试时强到弱的跨模型能力迁移范式**：将“构建推理期外部 harness”明确为独立于训练时蒸馏的新型能力转移路径，区别于以往仅依赖模型权重更新的范式。
2. **提供大规模系统性实证与机制归因**：在 4 个 ToM 基准上完成 72 组跨模型/平台/effort 控制实验，量化了提升幅度、稳定性、验证效率、平台效应与目标依赖，揭示确定性卸载与任务路由是核心驱动因素。
3. **提出“认知负荷卸载”统一解释框架**：证明 harness 增益主要来自将脆弱推理编译为确定性代码、基准感知路由与严格格式约束，而非单纯延长目标模型推理或增加采样多样性。
4. **建立目标模型 headroom 与 harness 增益的定量关系**：发现提升幅度与目标模型在特定任务上的未开发余量（1 − baseline）强相关（Pearson r=0.75），为自适应 harness 设计提供可操作原则。

## 方法详解
- **设定与符号**：目标模型记为 $M_{\mathrm{tar}}$，构建模型记为 $M_{\mathrm{build}}$。每个基准 $\mathcal{D}^{(j)}$ 随机划分 5% 验证集 $\mathcal{V}^{(j)}$ 与隐藏测试集 $\mathcal{T}^{(j)}$，builder 仅能访问 $\mathcal{V}$。
- **工作空间初始化**：$\mathcal{W}_0 = \{\mathcal{R}, C_{\mathrm{demo}}, \mathcal{V}\}$，其中 $\mathcal{R}$ 为任务规则与提交格式说明，$C_{\mathrm{demo}}$ 为目标模型调用示例，$\mathcal{V}$ 为带标签验证集。
- **迭代优化循环（Algorithm 1）**：
  1. Builder 读取任务资源，理解输入输出格式与评估要求；
  2. 编写或修改 harness $S_k$（可为任意 prompt/路由/确定性求解器/验证步骤组合）；
  3. 在 $\mathcal{V}$ 上运行目标模型得到 $\hat{Y}_k^\mathcal{V}$，计算准确率 $a_k$；
  4. 收集错误样本 $\mathcal{E}_k = \{(x,y,\hat{y}) \in \mathcal{V} : \hat{y} \neq y\}$，更新工作空间 $\mathcal{W}_{k+1}$；
  5. 重复直至 builder 提交最终可执行入口 $f_{\hat{S}}(x; M_{\mathrm{tar}})$，由人工在 $\mathcal{T}$ 上执行并获得最终评分。
- **优化目标**：$\hat{S} = \arg\max_{S \in S_{\mathrm{build}}} \operatorname{Acc}(S, M_{\mathrm{tar}}; \mathcal{V})$。成功 harness 必须捕获可泛化的任务结构而非记忆实例答案。
- **评分机制**：主指标为四个基准未加权 macro-average accuracy；次指标为验证集评估次数（鼓励高效利用验证预算）。

## 实验与结果
- **数据集**：BigToM (1200)、Hi-ToM (1200)、MMToM-QA (600)、MuMA-ToM (900)，合并为 3900 项隐藏测试集；每榜固定抽取 195 项 (5%) 作为验证集。
- **基线**：Vanilla（直接调用，GPT-5.4-mini 0.488 / Gemini-3.5-flash 0.761）；Human-Inspired UserHarness（GPT-5.4-mini 0.939 / Gemini-3.5-flash 0.941）。
- **主结果**：57 组 scaffolded 运行平均 macro-average 达 0.763（+0.275）；最佳配置 GPT-5.5 + GPT Codex 达到 0.912（+0.423，相对提升 86.7%），超越 vanilla GPT-5.4 与 GPT-OSS-120B。
- **稳定性**：跨重复标准差均值仅 0.036；确定性求解策略方差略大，单点逻辑错误可导致数百项基准的显著波动。
- **验证效率**：中位验证轮次 5 次，最佳验证准确率与最终测试准确率高度对齐（Pearson r=0.96），验证-测试乐观差均值 0.021；迭代次数与最终性能几乎无关（r=0.17），builder 质量比验证预算更重要。
- **目标依赖与 Headroom Law**：弱目标 GPT-5.4-mini 平均 +0.262，强目标 Gemini-3.5-flash 平均 +0.110；增益严格受目标在该榜的余量（1 − baseline）驱动；对强目标过度 scaffolding 会引发回归（9/20 案例出现负向变化）。
- **Builder Reasoning Effort**：Opus-4.7 在低/中/高/超高 effort 下准确率单调递增（0.711→0.793→

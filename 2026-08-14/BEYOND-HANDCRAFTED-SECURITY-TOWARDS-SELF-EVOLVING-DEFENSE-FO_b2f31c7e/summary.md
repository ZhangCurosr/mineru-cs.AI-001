---
title: "BEYOND-HANDCRAFTED-SECURITY-TOWARDS-SELF-EVOLVING-DEFENSE-FO"
source: https://arxiv.org/pdf/2608.12977v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:20:26"
---

# 论文速读：BEYOND-HANDCRAFTED-SECURITY-TOWARDS-SELF-EVOLVING-DEFENSE-FO

## 一句话总结
本文提出 HARD，一种基于 Agent Harness 的自主运行时防御演化框架，通过将执行失败轨迹自动归因并路由至可编辑的防御工件（上下文策略层与动作门控层），实现运行时安全机制的自演进；在 AgentCanary 等多类攻击设定下，HARD 显著降低攻击成功率并优于现有手搓静态防御，同时保持良好的良性任务效用。

## 研究问题与动机
- LLM Agent 已从被动文本生成演进为具备工具调用、状态维护与环境交互能力的复杂系统，安全风险随之从静态文本生成蔓延至运行时执行链路，单次不安全工具调用即可引发严重后果。
- 现有运行时防御（Runtime Defense）依赖人工预设规则与静态策略，面对 Agent 内在的开放式失败空间（Open-ended Failure Space）无法穷举覆盖，且部署后即冻结，难以应对持续演化的威胁。
- 自适应对抗攻击（Adaptive Attacks）与系统性红队测试会基于已部署防御的反饋不断调整攻击策略，导致防御所面对的失败分布呈现非平稳性，人工迭代诊断与修复的成本高、扩展性差。
- 核心研究问题：**运行时防御如何能够利用观测到的失败作为反馈，自主演化以持续适配新兴攻击？**

## 核心贡献（创新点）
- **Harness 中心化统一

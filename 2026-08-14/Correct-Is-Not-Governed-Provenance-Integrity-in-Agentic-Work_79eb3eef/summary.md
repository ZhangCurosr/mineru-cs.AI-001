---
title: "Correct-Is-Not-Governed-Provenance-Integrity-in-Agentic-Work"
source: https://arxiv.org/pdf/2608.12761v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:23:26"
---

# 论文速读：Correct-Is-Not-Governed-Provenance-Integrity-in-Agentic-Work

## 一句话总结
本文提出 Matrix 确定性因果状态层，将 Agent 工作流的“受治理执行（governed execution）”拆分为决策、执行与变更三类可检验的溯源属性；通过配对实验证明任务结果正确并不等价于执行过程受治理，并指出确定性契约虽能保证路径忠实，却无法自动泛化至独立上下文中的语义有效性。

## 研究问题与动机
- **核心问题**：现有 Agent 评估过度依赖“业务结果是否正确”，但在企业/制度场景下，正确的动作可能基于错误的权威依据、未经独立验证的完成声明，或对策略变更后残留工作缺乏精准响应。
- **现有方法不足**：
  1. **引用接地不充分**：直接 RAG 能选出预期动作，但无法区分“被检索到的文档”与“实际生效的制度依据”，易将草稿、过期豁免、相邻管辖例外等非管辖记录纳入决策依据。
  2. **自证闭环缺陷**：模型自身的通用 self-review 可产出完整计划，但无法阻止 Agent 用自身断言替代独立完成证据而提前关闭任务，

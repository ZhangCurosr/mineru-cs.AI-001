---
title: "Persona Conditioning as an Assessor-Sensitivity Probe for LLM-Based IR Evaluation"
source: https://arxiv.org/pdf/2608.10385v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:27:35"
---

# 论文速读：Persona Conditioning as an Assessor-Sensitivity Probe for LLM-Based IR Evaluation

## 一句话总结
本文将以任务为导向的“评估者人设（Persona）”作为诊断探针，系统检验大语言模型在信息检索相关性评估中对评估视角的敏感性。研究发现人设调节引发的偏差是结构化的而非随机的：高容量模型能保持全局排名稳定，而小容量模型会放大评估波动；人设来源的影响小于评估角色与模型规模，该机制可作为压力测试工具识别对评估框架敏感的检索系统。

## 研究问题与动机
1. LLM作为自动化相关性评估器的可靠性受提示表述和评估语境影响显著，但现有工作多聚焦于单固定视角或人口统计/心理特质人设，缺乏对人设变化如何系统性传导至相关性

---
title: "MazzikaAI: A knowledge-based performance-to-prompt compiler for real-time Arabic maqam accompaniment with a streaming text-to-music model"
source: https://arxiv.org/pdf/2608.10360v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:51:46"
---

# 论文速读：MazzikaAI: A knowledge-based performance-to-prompt compiler for real-time Arabic maqam accompaniment with a streaming text-to-music model

## 一句话总结
本文提出 MazzikaAI，一个将实时演奏状态编译为自然语言提示词的确定性控制层；该系统无需微调即可驱动未修改的流式文本生成音乐模型（Lyria RealTime），在不依赖额外训练数据的前提下实现了面向阿拉伯 maqam 微分音体系的实时互动伴奏。

## 研究问题与动机
1. **生成模型的调式偏见**：主流 T2M 模型的训练分布以西方十二平均律为主，对阿拉伯 maqam 的微分音程、特征装饰音与调式静止音严重缺失。
2. **流式模型的控制面空白**：Lyria RealTime 等模型支持增量生成，但仅提供静态文本输入接口，无法根据独奏者的即时音符、和声与乐句结构进行细粒度动态响应。
3. **符号控制与音色真实性的割裂**：传统符号/乐谱条件模型控制精细但缺乏波形级音色，且适配新文化语境通常需收集配对数据并微调；直接端到端音频模型又难以在表演时间粒度上被有效引导。
4. **实时伴奏的本质挑战**：优秀伴奏者需持续聆听、预判并支撑独奏而不喧宾夺主，这对 AI 提出了动态适应、文化语境尊重与低延迟闭环的双重需求。

## 核心贡献（创新点）
1. **
